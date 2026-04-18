"""PyTorch Dataset for text-prompted binary segmentation."""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import CLIPSegProcessor

import config


class PromptedSegDataset(Dataset):
    """Yields (image, text_prompt, binary_mask) triples for CLIPSeg fine-tuning.

    Each train sample is augmented with a randomly chosen prompt synonym.

    Negative-prompt augmentation (train split only):
        With probability `config.NEG_PROMPT_PROB` (set to 0.0 to disable) we
        replace the sample's prompt with a prompt from a DIFFERENT dataset and
        set the target mask to all zeros. This forces the model to actually
        condition on the text — if the prompt asks for a concept that is not
        in the image, the model must output an empty mask. Validation and test
        never use negative prompts.
    """

    def __init__(
        self,
        split: str = "train",
        processor: CLIPSegProcessor | None = None,
        img_size: int = config.IMG_SIZE,
        augment: bool = False,
    ):
        super().__init__()
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == "train"
        self.processor = processor
        # Negative-prompt augmentation only applies on the train split.
        self.neg_prompt_prob = config.NEG_PROMPT_PROB if self.augment else 0.0

        # (img_path, mask_path, own_prompts, negative_prompts)
        self.samples: list[tuple[Path, Path, list[str], list[str]]] = []

        # Build per-sample lists of prompts from *other* datasets for negatives.
        all_prompts_by_dataset = {
            name: list(info["prompts"]) for name, info in config.DATASETS.items()
        }

        for name, info in config.DATASETS.items():
            split_dir = config.DATA_DIR / name / split
            mask_dir = split_dir / "masks"
            if not split_dir.exists():
                continue

            neg_pool = [
                p for other, prompts in all_prompts_by_dataset.items()
                if other != name for p in prompts
            ]

            img_files = sorted(
                p for p in split_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
                and p.parent.name != "masks"
            )

            for img_path in img_files:
                stem = img_path.stem
                mask_candidates = [
                    mask_dir / f"{stem}.png",
                    mask_dir / f"{stem}.jpg",
                ]
                mask_path = next((m for m in mask_candidates if m.exists()), None)
                if mask_path is None:
                    continue
                self.samples.append((img_path, mask_path, info["prompts"], neg_pool))

        neg_str = f", neg_prompt_prob={self.neg_prompt_prob}" if self.neg_prompt_prob > 0 else ""
        print(f"[{split}] loaded {len(self.samples)} samples{neg_str}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path, prompts, neg_pool = self.samples[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        if self.augment:
            img, mask = self._augment(img, mask)

        is_negative = self.augment and neg_pool and random.random() < self.neg_prompt_prob
        if is_negative:
            prompt = random.choice(neg_pool)
            mask = np.zeros_like(mask)
        else:
            prompt = random.choice(prompts) if self.augment else prompts[0]

        mask_resized = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy((mask_resized > 127).astype(np.float32))

        if self.processor is not None:
            inputs = self.processor(
                text=[prompt],
                images=[img],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": mask_tensor,
                "img_path": str(img_path),
                "prompt": prompt,
                "is_negative": is_negative,
            }

        return {
            "image": img,
            "mask": mask_tensor,
            "prompt": prompt,
            "img_path": str(img_path),
            "is_negative": is_negative,
        }

    @staticmethod
    def _augment(img: np.ndarray, mask: np.ndarray):
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if random.random() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        if random.random() > 0.5:
            # slight brightness / contrast jitter
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-20, 20)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
        return img, mask
