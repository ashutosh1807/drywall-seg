"""Generate prediction masks for the held-out test split.

For every image in the held-out test split (one image per dataset/prompt pair),
write a single-channel PNG mask at the source image's spatial size, with
values {0, 255}, named:

    predictions/{image_id}__{prompt_slug}.png

where `prompt_slug` is the primary prompt for that dataset with spaces
replaced by underscores (e.g. `crack_001__segment_crack.png`).

By default we use `checkpoints/best_model.pt`; pass `--ckpt PATH` to predict
with any other checkpoint, and `--splits test valid` to widen the set of
source images.

Usage:
    uv run python predict.py
    uv run python predict.py --ckpt /path/to/best_model.pt
    uv run python predict.py --splits test valid
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import config
from evaluate import DEVICE, load_inference_model

THRESH = 0.5
DEFAULT_CKPT = config.CHECKPOINT_DIR / "best_model.pt"


def _list_images(split_dir: Path) -> list[Path]:
    return sorted(
        p for p in split_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        and p.parent.name != "masks"
    )


@torch.no_grad()
def predict_mask(model, processor, img_rgb: np.ndarray, prompt: str) -> tuple[np.ndarray, float]:
    """Returns (uint8 mask in {0,255} at source resolution, inference seconds)."""
    h, w = img_rgb.shape[:2]
    inputs = processor(text=[prompt], images=[img_rgb], return_tensors="pt",
                       padding="max_length", truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    t0 = time.time()
    logits = model(**inputs).logits
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
    small = (probs > THRESH).astype(np.uint8) * 255
    mask = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask, dt


def generate_predictions(ckpt: Path, splits: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_inference_model(ckpt if ckpt.exists() else None)
    if not ckpt.exists():
        print(f"[!] checkpoint not found ({ckpt}) — using stock CLIPSeg weights")

    total_time, count = 0.0, 0
    for cat_name, cat_info in config.DATASETS.items():
        prompt = cat_info["prompts"][0]
        slug = prompt.replace(" ", "_")

        for split in splits:
            split_dir = config.DATA_DIR / cat_name / split
            if not split_dir.exists():
                continue
            imgs = _list_images(split_dir)
            for img_path in imgs:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mask, dt = predict_mask(model, processor, img_rgb, prompt)
                cv2.imwrite(str(out_dir / f"{img_path.stem}__{slug}.png"), mask)
                total_time += dt
                count += 1
            print(f"  [{cat_name}/{split}] {len(imgs)} images  prompt='{prompt}'")

    avg_ms = (total_time / max(count, 1)) * 1000
    print(f"\nWrote {count} masks to {out_dir}  (avg inference: {avg_ms:.1f} ms/image)")
    return {"count": count, "avg_inference_ms": avg_ms, "out_dir": str(out_dir)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT),
                        help="Checkpoint to use (default: checkpoints/best_model.pt).")
    parser.add_argument("--splits", nargs="+", default=["test"],
                        help="Which dataset splits to predict on (default: test).")
    parser.add_argument("--out", type=str, default=str(config.PREDICTION_DIR),
                        help="Output directory for prediction masks.")
    args = parser.parse_args()

    generate_predictions(Path(args.ckpt), args.splits, Path(args.out))


if __name__ == "__main__":
    main()
