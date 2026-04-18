"""Train CLIPSeg for prompted binary segmentation of cracks and drywall taping.

Recipe:

  - Model     : CIDAS/clipseg-rd64-refined
  - Unfrozen  : decoder only (CLIP backbone fully frozen)
  - Loss      : DiceBCELoss (50% Dice + 50% BCE)
  - Aug       : prompt synonyms + image flips/jitter + cross-class
                negative-prompt augmentation (probability `config.NEG_PROMPT_PROB`,
                default 0.10; set to 0.0 in config.py to disable)
  - EMA weight averaging (decay = 0.999)
  - Early stopping on val_mIoU (patience = 7)

Usage:
    uv run python train.py
"""

from __future__ import annotations

import time

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    EarlyStopping,
    EMAWeightAveraging,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

import config
from dataset import PromptedSegDataset

torch.set_float32_matmul_precision("high")
L.seed_everything(config.SEED, workers=True)

CKPT_DIR = config.CHECKPOINT_DIR
LOG_DIR = config.OUTPUT_DIR
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Loss ────────────────────────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss (50/50). Dice handles class imbalance
    (thin cracks, noisy SAM 3 seams); BCE keeps gradients well-conditioned."""

    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(-2, -1))
        union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()

        return (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss


# ── Model ───────────────────────────────────────────────────────────────────

class CLIPSegDecoderDiceBCE(L.LightningModule):
    """CLIPSeg with decoder-only fine-tuning and DiceBCE loss."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        lr: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        warmup_ratio: float = config.WARMUP_RATIO,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name)
        self._freeze_backbone()
        self.loss_fn = DiceBCELoss(dice_weight=dice_weight)

        self.train_iou, self.train_dice = BinaryJaccardIndex(), BinaryF1Score()
        self.val_iou, self.val_dice = BinaryJaccardIndex(), BinaryF1Score()
        self.test_iou, self.test_dice = BinaryJaccardIndex(), BinaryF1Score()

    def _freeze_backbone(self) -> None:
        """Freeze everything; unfreeze only the CLIPSeg decoder."""
        self.freeze().unfreeze()
        self.model.requires_grad_(False)
        self.model.decoder.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[train] Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    def on_train_epoch_start(self):
        self.model.eval()
        self.model.decoder.train()

    def forward(self, input_ids, pixel_values, attention_mask):
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )

    def _align_logits(self, logits: torch.Tensor, target_shape) -> torch.Tensor:
        if logits.shape[-2:] != target_shape:
            logits = nn.functional.interpolate(
                logits.unsqueeze(1), size=target_shape,
                mode="bilinear", align_corners=False,
            ).squeeze(1)
        return logits

    def _shared_step(self, batch, stage: str):
        outputs = self(batch["input_ids"], batch["pixel_values"], batch["attention_mask"])
        logits = self._align_logits(outputs.logits, batch["labels"].shape[-2:])

        loss = self.loss_fn(logits, batch["labels"])
        preds = (torch.sigmoid(logits) > 0.5).long()
        targets = batch["labels"].long()

        getattr(self, f"{stage}_iou").update(preds, targets)
        getattr(self, f"{stage}_dice").update(preds, targets)

        bs = len(batch["labels"])
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=bs)
        self.log(f"{stage}_mIoU", getattr(self, f"{stage}_iou"),
                 prog_bar=True, on_epoch=True, on_step=False, batch_size=bs)
        self.log(f"{stage}_Dice", getattr(self, f"{stage}_dice"),
                 prog_bar=True, on_epoch=True, on_step=False, batch_size=bs)
        return loss

    def training_step(self, batch, _):   return self._shared_step(batch, "train")
    def validation_step(self, batch, _): return self._shared_step(batch, "val")
    def test_step(self, batch, _):       return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr, total_steps=total_steps,
            pct_start=max(warmup_steps / total_steps, 0.01),
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# ── Data ────────────────────────────────────────────────────────────────────

class SegDataModule(L.LightningDataModule):
    def __init__(self, processor: CLIPSegProcessor,
                 batch_size: int = config.BATCH_SIZE, num_workers: int = 4):
        super().__init__()
        self.save_hyperparameters(ignore=["processor"])
        self.processor = processor

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_ds = PromptedSegDataset("train", processor=self.processor, augment=True)
            self.val_ds = PromptedSegDataset("valid", processor=self.processor, augment=False)
        if stage in ("test", None):
            self.test_ds = PromptedSegDataset("test", processor=self.processor, augment=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True)


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Train CLIPSeg — decoder-only + DiceBCE + neg-prompt aug")
    print("=" * 60)
    print(f"  model            : {config.MODEL_NAME}")
    print(f"  loss             : DiceBCELoss(dice_weight=0.5)")
    print(f"  unfrozen         : decoder only (CLIP backbone fully frozen)")
    print(f"  epochs           : {config.NUM_EPOCHS}")
    print(f"  batch size       : {config.BATCH_SIZE}")
    print(f"  lr               : {config.LEARNING_RATE}")
    print(f"  neg-prompt prob. : {config.NEG_PROMPT_PROB}  ({'OFF' if config.NEG_PROMPT_PROB == 0 else 'ON'})")
    print(f"  ckpt dir         : {CKPT_DIR}")
    print(f"  log dir          : {LOG_DIR}")
    print()

    processor = CLIPSegProcessor.from_pretrained(config.MODEL_NAME)
    model = CLIPSegDecoderDiceBCE()
    datamodule = SegDataModule(processor=processor)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(CKPT_DIR),
        filename="best-{epoch:02d}-{val_mIoU:.4f}",
        monitor="val_mIoU", mode="max",
        save_top_k=1, save_last=True, verbose=True,
    )
    early_stop_cb = EarlyStopping(monitor="val_mIoU", mode="max", patience=7, verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    ema_cb = EMAWeightAveraging(decay=0.999)
    logger = CSVLogger(save_dir=str(LOG_DIR), name="lightning_logs")

    trainer = L.Trainer(
        max_epochs=config.NUM_EPOCHS,
        accelerator="auto", devices=1, precision="16-mixed",
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, ema_cb],
        logger=logger,
        deterministic=True, gradient_clip_val=1.0, log_every_n_steps=10,
    )

    t0 = time.time()
    trainer.fit(model, datamodule=datamodule)
    train_time = time.time() - t0

    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        state = torch.load(best_ckpt, map_location="cpu", weights_only=True)
        model_state = {k.replace("model.", "", 1): v
                       for k, v in state["state_dict"].items()
                       if k.startswith("model.")}
        torch.save(model_state, CKPT_DIR / "best_model.pt")
        print(f"Saved inference weights -> {CKPT_DIR / 'best_model.pt'}")

    meta = {
        "train_time_sec": train_time,
        "best_val_iou": float(checkpoint_cb.best_model_score or 0),
        "best_ckpt": best_ckpt,
        "num_epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "lr": config.LEARNING_RATE,
        "loss": "DiceBCELoss(dice_weight=0.5)",
        "unfrozen_layers": "decoder only",
        "neg_prompt_prob": config.NEG_PROMPT_PROB,
        "seed": config.SEED,
        "model": config.MODEL_NAME,
    }
    torch.save(meta, LOG_DIR / "train_meta.pt")
    print(f"\n[train] Training done in {train_time / 60:.1f} min")
    print(f"[train] Best val mIoU: {float(checkpoint_cb.best_model_score or 0):.4f}")

    return model, processor


if __name__ == "__main__":
    main()
