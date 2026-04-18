"""Evaluate the CLIPSeg checkpoint on the held-out test split.

Runs two evaluations:

  1. Standard segmentation metrics — per-image (macro) mIoU and Dice per class
     and overall, plus average inference time.

  2. Prompt-sensitivity diagnostic — does the model condition on the text?
     Per test image we predict with three prompts (correct / wrong / paraphrase)
     and report mIoU(p+), false-activation rate, suppression rate, cross-prompt
     IoU and paraphrase IoU per class.

Usage:
    uv run python evaluate.py                # uses checkpoints/best_model.pt
    uv run python evaluate.py --ckpt PATH    # custom checkpoint
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_f1_score, binary_jaccard_index
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

import config
from dataset import PromptedSegDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESH_PROB = 0.5
SUPPRESSION_TOL = 0.01

# Default checkpoint — final recommended model (see report.md).
DEFAULT_CKPT = config.CHECKPOINT_DIR / "best_model.pt"


# ── Metric helpers ─────────────────────────────────────────────────────────

def _as_int_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Coerce mask to an int tensor on CPU. Accepts torch tensors or numpy arrays."""
    t = torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x
    return (t > 0).int().cpu()


def iou_score(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> float:
    """Per-image binary IoU. Empty pred AND empty target ⇒ 1.0 (perfect agreement)."""
    return float(binary_jaccard_index(_as_int_tensor(pred), _as_int_tensor(target),
                                      zero_division=1.0))


def dice_score(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> float:
    """Per-image binary Dice / F1. Empty pred AND empty target ⇒ 1.0."""
    return float(binary_f1_score(_as_int_tensor(pred), _as_int_tensor(target),
                                 zero_division=1.0))


def positive_ratio(mask: np.ndarray) -> float:
    return float(mask.sum()) / float(mask.size)


# ── Model loader ───────────────────────────────────────────────────────────

def load_inference_model(ckpt_path: Path | None = None) -> tuple[CLIPSegForImageSegmentation, CLIPSegProcessor]:
    """Load CLIPSeg + processor. If `ckpt_path` is given, load fine-tuned weights."""
    processor = CLIPSegProcessor.from_pretrained(config.MODEL_NAME)
    model = CLIPSegForImageSegmentation.from_pretrained(config.MODEL_NAME)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"  loaded checkpoint: {ckpt_path}")
    model.to(DEVICE).eval()
    return model, processor


# ── 1) Standard test evaluation ────────────────────────────────────────────

@torch.no_grad()
def evaluate_test(ckpt_path: Path | None = None) -> dict:
    """Per-image (macro) mIoU & Dice on the test split, per class and overall."""
    model, processor = load_inference_model(ckpt_path)
    ds = PromptedSegDataset("test", processor=processor, augment=False)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    per: dict[str, dict[str, list[float]]] = {n: {"iou": [], "dice": []} for n in config.DATASETS}
    times: list[float] = []

    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        pv = batch["pixel_values"].to(DEVICE)
        am = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        prompts = batch["prompt"]

        t0 = time.time()
        out = model(input_ids=ids, pixel_values=pv, attention_mask=am)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) / len(prompts))

        logits = out.logits
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = nn.functional.interpolate(
                logits.unsqueeze(1), size=labels.shape[-2:],
                mode="bilinear", align_corners=False,
            ).squeeze(1)
        preds = (torch.sigmoid(logits) > THRESH_PROB).float()

        for i, prompt in enumerate(prompts):
            for name, info in config.DATASETS.items():
                if prompt in info["prompts"]:
                    per[name]["iou"].append(iou_score(preds[i], labels[i]))
                    per[name]["dice"].append(dice_score(preds[i], labels[i]))
                    break

    out: dict = {}
    all_iou, all_dice = [], []
    for name, d in per.items():
        if not d["iou"]:
            continue
        out[name] = {
            "mIoU": float(np.mean(d["iou"])),
            "Dice": float(np.mean(d["dice"])),
            "n": len(d["iou"]),
        }
        all_iou += d["iou"]
        all_dice += d["dice"]
    out["overall"] = {
        "mIoU": float(np.mean(all_iou)),
        "Dice": float(np.mean(all_dice)),
        "n": len(all_iou),
    }
    out["avg_inference_ms"] = float(np.mean(times)) * 1000.0
    return out


# ── 2) Prompt-sensitivity diagnostic ───────────────────────────────────────

def _list_test_samples() -> list[tuple[str, Path, Path]]:
    samples: list[tuple[str, Path, Path]] = []
    for name in config.DATASETS:
        split_dir = config.DATA_DIR / name / "test"
        mask_dir = split_dir / "masks"
        if not split_dir.exists():
            continue
        for img_path in sorted(split_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            if img_path.parent.name == "masks":
                continue
            mask_path = mask_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                samples.append((name, img_path, mask_path))
    return samples


def _prompts_for(dataset_name: str) -> tuple[str, str, str]:
    """(correct, wrong / cross-class, paraphrase) prompts for this dataset."""
    own = config.DATASETS[dataset_name]["prompts"]
    other_names = [n for n in config.DATASETS if n != dataset_name]
    other = config.DATASETS[other_names[0]]["prompts"]
    return own[0], other[0], (own[1] if len(own) > 1 else own[0])


@torch.no_grad()
def _predict_native(model, processor, img_rgb: np.ndarray, prompt: str) -> np.ndarray:
    """Return a {0,1} uint8 mask at the model's native output resolution."""
    inputs = processor(text=[prompt], images=[img_rgb],
                       return_tensors="pt", padding="max_length", truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = model(**inputs).logits
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    return ((torch.sigmoid(logits) > THRESH_PROB).long().squeeze(0).cpu().numpy().astype(np.uint8))


def evaluate_prompt_sensitivity(ckpt_path: Path) -> dict:
    """Returns {dataset_name: {miou_correct, false_act_rate, suppression, cross_prompt_iou, paraphrase_iou, n}}."""
    model, processor = load_inference_model(ckpt_path)
    samples = _list_test_samples()

    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for i, (ds, img_path, mask_path) in enumerate(samples):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            gt = np.zeros(img.shape[:2], dtype=np.uint8)
        gt = (gt > 127).astype(np.uint8)

        correct, wrong, para = _prompts_for(ds)
        m_correct = _predict_native(model, processor, img, correct)
        m_wrong   = _predict_native(model, processor, img, wrong)
        m_para    = _predict_native(model, processor, img, para)

        h, w = m_correct.shape
        gt_r = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

        acc[ds]["miou_correct"].append(iou_score(m_correct, gt_r))
        acc[ds]["false_act_rate"].append(positive_ratio(m_wrong))
        acc[ds]["suppression"].append(1.0 if positive_ratio(m_wrong) < SUPPRESSION_TOL else 0.0)
        acc[ds]["cross_prompt_iou"].append(iou_score(m_correct, m_wrong))
        acc[ds]["paraphrase_iou"].append(iou_score(m_correct, m_para))

        if (i + 1) % 50 == 0:
            print(f"    processed {i + 1}/{len(samples)}")

    out: dict = {}
    for ds, metrics in acc.items():
        out[ds] = {k: float(np.mean(v)) for k, v in metrics.items()}
        out[ds]["n"] = len(metrics["miou_correct"])
    return out


# ── Pretty printing ────────────────────────────────────────────────────────

def _print_standard(label: str, r: dict) -> None:
    print(f"\n--- Standard test metrics: {label} ---")
    print(f"{'class':<10}  {'mIoU':>8}  {'Dice':>8}  {'n':>5}")
    print("-" * 40)
    for name in config.DATASETS:
        if name in r:
            v = r[name]
            print(f"{name:<10}  {v['mIoU']:>8.4f}  {v['Dice']:>8.4f}  {v['n']:>5}")
    v = r["overall"]
    print(f"{'overall':<10}  {v['mIoU']:>8.4f}  {v['Dice']:>8.4f}  {v['n']:>5}")
    print(f"avg inference: {r['avg_inference_ms']:.2f} ms / image")


def _print_sensitivity(result: dict) -> None:
    print("\n" + "=" * 60)
    print("PROMPT-SENSITIVITY DIAGNOSTIC")
    print("=" * 60)
    metrics = [
        ("mIoU(p+)",         "miou_correct",     "↑"),
        ("False-act rate",   "false_act_rate",   "↓"),
        ("Suppression rate", "suppression",      "↑"),
        ("Cross-prompt IoU", "cross_prompt_iou", "↓"),
        ("Paraphrase IoU",   "paraphrase_iou",   "↑"),
    ]
    for ds in config.DATASETS:
        if ds not in result:
            continue
        print(f"\n--- Class: {ds} (n={int(result[ds]['n'])}) ---")
        for nice, key, arrow in metrics:
            v = result[ds].get(key)
            print(f"  {nice + ' ' + arrow:<22}  {v:.4f}" if v is not None
                  else f"  {nice + ' ' + arrow:<22}  —")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT),
                        help="Path to best_model.pt to evaluate (default: checkpoints/best_model.pt).")
    parser.add_argument("--out", type=str, default=str(config.OUTPUT_DIR / "eval_results.json"),
                        help="Where to write the combined results JSON.")
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"[!] checkpoint not found: {ckpt}")
        return

    print(f"\n=== Standard eval ===")
    standard = evaluate_test(ckpt)
    _print_standard(ckpt.name, standard)

    print(f"\n=== Prompt-sensitivity ===")
    sensitivity = evaluate_prompt_sensitivity(ckpt)
    _print_sensitivity(sensitivity)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"ckpt": str(ckpt), "standard": standard, "sensitivity": sensitivity},
        indent=2,
    ))
    print(f"\nResults JSON -> {out_path}")


if __name__ == "__main__":
    main()
