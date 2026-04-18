"""Report figures for the final model.

Generates two figures by default:

  1. figures/test_examples.png
       Qualitative results on the held-out test split — one **worst**, one
       **median** and one **best** image per class (ranked by per-image IoU
       on non-empty-GT samples). Layout per row: original | GT | prediction.

  2. figures/training_curves.png
       Training/validation loss + validation mIoU/Dice from the most recent
       Lightning CSV log. Skipped silently if no log directory exists.

Uses the checkpoint at `checkpoints/best_model.pt` (override with --ckpt).

Usage:
    uv run python visualize.py                # both figures
    uv run python visualize.py --ckpt PATH    # custom checkpoint
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPSegProcessor

import config
from evaluate import DEVICE, dice_score, iou_score, load_inference_model

THRESH = 0.5
DEFAULT_CKPT = config.CHECKPOINT_DIR / "best_model.pt"


# ── Shared helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def _predict(model, processor, img_rgb: np.ndarray, prompt: str,
             target_hw: tuple[int, int]) -> np.ndarray:
    inputs = processor(text=[prompt], images=[img_rgb], return_tensors="pt",
                       padding="max_length", truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = model(**inputs).logits
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    logits = nn.functional.interpolate(logits.unsqueeze(1), size=target_hw,
                                        mode="bilinear", align_corners=False).squeeze(1)
    return (torch.sigmoid(logits)[0].cpu().numpy() > THRESH).astype(np.uint8)


def _list_test_samples() -> list[tuple[str, Path, Path]]:
    out: list[tuple[str, Path, Path]] = []
    for ds in config.DATASETS:
        split = config.DATA_DIR / ds / "test"
        masks = split / "masks"
        if not split.exists():
            continue
        for img_path in sorted(split.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            if img_path.parent.name == "masks":
                continue
            mp = masks / f"{img_path.stem}.png"
            if mp.exists():
                out.append((ds, img_path, mp))
    return out


# ── Figure 1: test examples (worst / median / best per class) ──────────────

def figure_test_examples(ckpt: Path = DEFAULT_CKPT, out_path: Path | None = None):
    out_path = out_path or (config.FIGURE_DIR / "test_examples.png")
    if not ckpt.exists():
        print(f"[!] checkpoint not found: {ckpt} — skipping")
        return

    print(f"\n[1] Test examples — loading {ckpt}")
    processor = CLIPSegProcessor.from_pretrained(config.MODEL_NAME)
    model, _ = load_inference_model(ckpt)

    samples = _list_test_samples()
    print(f"    scoring {len(samples)} test samples...")

    # Drop empty-GT images so the "best" tier shows an actual successful
    # prediction on a real structure, not a trivial empty/empty match.
    per_class: dict[str, list[tuple[str, Path, Path, float]]] = {ds: [] for ds in config.DATASETS}
    for ds_name, img_path, mask_path in samples:
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        gt = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if gt.sum() == 0:
            continue
        prompt = config.DATASETS[ds_name]["prompts"][0]
        pred = _predict(model, processor, img, prompt, gt.shape[:2])
        per_class[ds_name].append((ds_name, img_path, mask_path, iou_score(pred, gt)))

    chosen: list[tuple[str, Path, Path, float, str]] = []
    for ds in config.DATASETS:
        scored = sorted(per_class[ds], key=lambda t: t[3])
        if not scored:
            continue
        n = len(scored)
        chosen.append((*scored[0], "worst"))
        chosen.append((*scored[n // 2], "median"))
        chosen.append((*scored[-1], "best"))
        print(f"    {ds}: IoU range [{scored[0][3]:.3f}, {scored[-1][3]:.3f}]")

    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, 3.4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for r, (ds_name, img_path, mask_path, _, tier) in enumerate(chosen):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        gt = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        prompt = config.DATASETS[ds_name]["prompts"][0]
        pred = _predict(model, processor, img, prompt, gt.shape[:2])

        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{ds_name.upper()} · {tier}", fontsize=9)
        axes[r, 0].axis("off")
        axes[r, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[r, 1].set_title(f"GT (prompt: '{prompt}')", fontsize=9)
        axes[r, 1].axis("off")
        axes[r, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[r, 2].set_title(
            f"Prediction\nIoU={iou_score(pred, gt):.3f}  Dice={dice_score(pred, gt):.3f}",
            fontsize=9,
        )
        axes[r, 2].axis("off")

    plt.suptitle("Test predictions — worst / median / best per class", fontsize=12, y=1.0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"    saved -> {out_path}")


# ── Figure 2: training curves (from Lightning CSV log) ─────────────────────

def plot_training_curves(out_path: Path | None = None):
    out_path = out_path or (config.FIGURE_DIR / "training_curves.png")
    log_dir = config.OUTPUT_DIR / "lightning_logs"
    if not log_dir.exists():
        print("\n[2] Training curves — no Lightning logs found, skipping")
        return

    versions = sorted([p for p in log_dir.iterdir() if p.is_dir()],
                      key=lambda p: p.stat().st_mtime)
    if not versions:
        print("\n[2] Training curves — no log versions found, skipping")
        return
    metrics_file = versions[-1] / "metrics.csv"
    if not metrics_file.exists():
        print(f"\n[2] Training curves — no metrics.csv in {versions[-1]}, skipping")
        return

    print(f"\n[2] Training curves — reading {metrics_file}")
    rows = list(csv.DictReader(metrics_file.open()))

    epochs, val_loss, val_iou, val_dice, train_loss = [], [], [], [], []
    for row in rows:
        if row.get("val_mIoU"):
            epochs.append(int(row.get("epoch", 0)) + 1)
            val_loss.append(float(row.get("val_loss", 0)))
            val_iou.append(float(row["val_mIoU"]))
            val_dice.append(float(row.get("val_Dice", 0)))
        if row.get("train_loss_epoch"):
            train_loss.append(float(row["train_loss_epoch"]))

    if not epochs:
        print("    no epoch-level metrics found, skipping")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = list(range(1, len(epochs) + 1))
    if train_loss and len(train_loss) >= len(epochs):
        ax1.plot(x, train_loss[: len(epochs)], label="Train Loss", linewidth=2)
    if val_loss:
        ax1.plot(x, val_loss, label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(x, val_iou, label="Val mIoU", linewidth=2, color="green")
    ax2.plot(x, val_dice, label="Val Dice", linewidth=2, color="orange")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    saved -> {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT),
                        help="Checkpoint to use for the test-examples figure.")
    args = parser.parse_args()

    figure_test_examples(Path(args.ckpt))
    plot_training_curves()


if __name__ == "__main__":
    main()
