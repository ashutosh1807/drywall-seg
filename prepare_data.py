"""Download datasets from Roboflow, convert annotations to binary masks, create splits.

For datasets that only ship bounding boxes (e.g. Taping), we refine each box
into a pixel mask with SAM 3 (text + positive box prompt) and clip the result
back to the box. The text prompt is configured per-dataset via
`config.DATASETS[<name>]["sam3_prompt"]`. If no `sam3_prompt` is set, we fall
back to a plain rectangle mask.

Usage:
    uv run python prepare_data.py
"""

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from roboflow import Roboflow

import config

random.seed(config.SEED)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Lazy-loaded SAM 3 handles (loaded on first box-only dataset).
_sam3_model = None
_sam3_processor = None
_sam3_device = None


def _list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTS and p.parent.name != "masks")


def _poly_to_mask(segmentation, h: int, w: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        return (mask_utils.decode(segmentation) > 0).astype(np.uint8)
    if isinstance(segmentation, list):
        rle = mask_utils.frPyObjects(segmentation, h, w)
        return (mask_utils.decode(mask_utils.merge(rle)) > 0).astype(np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


def _init_sam3():
    """Lazy-load SAM 3 the first time it is needed."""
    global _sam3_model, _sam3_processor, _sam3_device
    if _sam3_model is not None:
        return
    import torch
    from transformers import Sam3Model, Sam3Processor

    _sam3_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [SAM 3] loading on {_sam3_device}...")
    _sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
    _sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(_sam3_device).eval()
    print(f"  [SAM 3] loaded.")


def _sam3_boxes_to_mask(image_path: Path, boxes_xyxy: list, prompt: str, h: int, w: int) -> np.ndarray:
    """Run SAM 3 with text + positive box prompts, union masks, clip to boxes."""
    import torch
    from PIL import Image

    _init_sam3()

    image = Image.open(str(image_path)).convert("RGB")
    input_boxes = [[list(map(float, b)) for b in boxes_xyxy]]
    input_boxes_labels = [[1 for _ in boxes_xyxy]]

    inputs = _sam3_processor(
        images=image,
        text=prompt,
        input_boxes=input_boxes,
        input_boxes_labels=input_boxes_labels,
        return_tensors="pt",
    ).to(_sam3_device)

    with torch.no_grad():
        outputs = _sam3_model(**inputs)

    result = _sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = result["masks"]
    if len(masks) == 0:
        raw = np.zeros((h, w), dtype=np.uint8)
    else:
        raw = masks.any(dim=0).cpu().numpy().astype(np.uint8)

    # Clip to union of all boxes so SAM 3 cannot leak outside the GT region.
    keep = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        keep[y1:y2, x1:x2] = 1
    return (raw & keep).astype(np.uint8)


def download(name: str, info: dict) -> Path:
    rf = Roboflow(api_key=config.ROBOFLOW_API_KEY)
    project = rf.workspace(info["workspace"]).project(info["project"])
    dest = config.DATA_DIR / name
    fmt = info.get("download_format", "coco")
    project.version(info["version"]).download(fmt, location=str(dest), overwrite=True)
    return dest


def build_masks(dataset_dir: Path, info: dict) -> None:
    """Convert COCO annotations to per-image binary PNG masks.

    - Polygon annotations are rasterised directly.
    - Box-only annotations are refined with SAM 3 if `info["sam3_prompt"]` is
      set, otherwise filled as a plain rectangle.
    """
    sam3_prompt = info.get("sam3_prompt")

    for split in ("train", "valid", "test"):
        split_dir = dataset_dir / split
        ann_file = split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            continue

        coco = json.loads(ann_file.read_text())
        id_to_img = {img["id"]: img for img in coco["images"]}
        img_to_anns: dict[int, list] = {}
        for ann in coco["annotations"]:
            img_to_anns.setdefault(ann["image_id"], []).append(ann)

        mask_dir = split_dir / "masks"
        mask_dir.mkdir(exist_ok=True)

        for img_id, img_info in id_to_img.items():
            h, w = img_info["height"], img_info["width"]
            anns = img_to_anns.get(img_id, [])

            poly_mask = np.zeros((h, w), dtype=np.uint8)
            box_only = []  # [[x1,y1,x2,y2], ...] for boxes with no polygon
            for ann in anns:
                seg = ann.get("segmentation")
                if seg:
                    poly_mask = np.maximum(poly_mask, _poly_to_mask(seg, h, w))
                elif ann.get("bbox"):
                    x, y, bw, bh = [int(v) for v in ann["bbox"]]
                    box_only.append([x, y, x + bw, y + bh])

            if box_only:
                if sam3_prompt:
                    sam3_mask = _sam3_boxes_to_mask(
                        split_dir / img_info["file_name"], box_only, sam3_prompt, h, w
                    )
                    combined = np.maximum(poly_mask, sam3_mask)
                else:
                    rect = np.zeros((h, w), dtype=np.uint8)
                    for x1, y1, x2, y2 in box_only:
                        rect[y1:y2, x1:x2] = 1
                    combined = np.maximum(poly_mask, rect)
            else:
                combined = poly_mask

            mask_name = img_info["file_name"].rsplit(".", 1)[0] + ".png"
            cv2.imwrite(str(mask_dir / mask_name), combined * 255)

        print(f"  [{split}] {len(id_to_img)} masks -> {mask_dir}"
              + (f"  (SAM 3 prompt='{sam3_prompt}')" if sam3_prompt and split else ""))


def create_test_split(dataset_dir: Path, ratio: float = 0.5) -> None:
    valid_dir, test_dir = dataset_dir / "valid", dataset_dir / "test"
    imgs = _list_images(valid_dir)
    random.shuffle(imgs)
    n = int(len(imgs) * ratio)

    test_dir.mkdir(exist_ok=True)
    (test_dir / "masks").mkdir(exist_ok=True)

    for img in imgs[:n]:
        shutil.move(str(img), str(test_dir / img.name))
        mask = valid_dir / "masks" / f"{img.stem}.png"
        if mask.exists():
            shutil.move(str(mask), str(test_dir / "masks" / mask.name))

    print(f"  [split] {n} -> test, {len(imgs) - n} -> valid")


def main():
    if config.DATA_DIR.exists():
        shutil.rmtree(config.DATA_DIR)
    config.DATA_DIR.mkdir()

    for name, info in config.DATASETS.items():
        print(f"\n--- {name} ---")
        dest = download(name, info)
        build_masks(dest, info)
        if info.get("needs_test_split"):
            create_test_split(dest)

    print("\nDone. Summary:")
    for name in config.DATASETS:
        for split in ("train", "valid", "test"):
            d = config.DATA_DIR / name / split
            if d.exists():
                print(f"  {name}/{split}: {len(_list_images(d))} images")


if __name__ == "__main__":
    main()
