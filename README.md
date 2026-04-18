# Prompted Segmentation for Drywall QA

Text-conditioned binary segmentation for drywall quality assurance — detecting **cracks** and **taping areas** from natural-language prompts.

## Goal

Given an image and a natural-language prompt, produce a binary mask for:
- `"segment crack"` / `"segment wall crack"`
- `"segment taping area"` / `"segment joint tape"` / `"segment drywall seam"`

## Approach

**Model**: [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) (`CIDAS/clipseg-rd64-refined`) — a CLIP-based model with a lightweight decoder purpose-built for text-prompted image segmentation.

**Training framework**: [PyTorch Lightning](https://lightning.ai/) v2.6.1 with:
- **16-mixed precision (AMP)** — halves GPU memory, speeds up training
- **`torch.set_float32_matmul_precision("high")`** — enables Tensor Core acceleration on NVIDIA L4/A100
- **[TorchMetrics](https://torchmetrics.readthedocs.io/)** — GPU-accelerated `BinaryJaccardIndex` (IoU) and `BinaryF1Score` (Dice) with auto-reset per epoch, DDP-safe
- **EarlyStopping** on `val_mIoU` (patience=7)
- **ModelCheckpoint** saving top-1 by `val_mIoU`
- **CSVLogger** for training curves
- **Gradient clipping** (max_norm=1.0)

**Fine-tuning strategy**: Freeze the CLIP backbone entirely, train only the CLIPSeg decoder (~1.1M / 150.7M params trainable, 0.75%). Unfreezing CLIP vision layers was tried and overfit on this data scale (~2.4 k training images) — see `Final report.md` for the ablation. The released model uses **DiceBCE loss** + **negative-prompt augmentation** for genuine text conditioning.

**Why CLIPSeg?** It natively supports text-conditioned segmentation, is available in HuggingFace Transformers, and is efficient enough to fine-tune on a single GPU. For absolute SOTA, Grounded SAM 2 or OpenWorldSAM would be alternatives but are significantly more complex to fine-tune end-to-end.

## Datasets

| Dataset | Source | Type | Train | Valid | Test |
|---------|--------|------|------:|------:|-----:|
| Drywall-Join-Detect v2 | [Roboflow](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) | Object Detection | 820 | 101 | 101 |
| Crack Segmentation 2.0 v4 | [Roboflow](https://universe.roboflow.com/cracksegm20/crack-segmentation-2.0) | Instance Segmentation | 1564 | 274 | 152 |
| **Combined** | | | **2384** | **375** | **253** |

**Prompt mapping**: Each dataset maps to multiple text prompt synonyms. During training, a random synonym is selected per sample for robustness.

## Setup

```bash
# Install dependencies (requires uv)
uv sync

# Set your Roboflow API key
export ROBOFLOW_API_KEY="your_key_here"
```

### Requirements
- Python >= 3.11
- CUDA-capable GPU (tested on NVIDIA L4, 23 GB VRAM)
- `uv` package manager

## Usage

Run each stage in order:

```bash
uv run python prepare_data.py    # Download datasets & build binary masks (SAM 3 for Taping)
uv run python train.py            # Fine-tune CLIPSeg (PyTorch Lightning)
uv run python evaluate.py         # Test mIoU/Dice + prompt-sensitivity diagnostic
uv run python predict.py          # Generate {image_id}__{prompt}.png masks for the test split
uv run python visualize.py        # Generate report figures (test examples + training curves)
```

## Output Structure

```
checkpoints/          # Lightning checkpoints + best_model.pt (inference weights)
predictions/          # PNG masks: {image_id}__{prompt_slug}.png  (single-channel, {0,255})
figures/              # test_examples.png, training_curves.png (+ a few report-only figures)
outputs/              # eval_results.json, train_meta.pt, lightning_logs/
```

## Prediction Mask Format

- **Format**: PNG, single-channel, same spatial size as source image
- **Values**: `{0, 255}` (background / foreground)
- **Naming**: `{image_id}__{prompt_slug}.png` (e.g., `crack_001__segment_crack.png`)

## Project Files

| File | Description |
|------|-------------|
| `config.py` | Central configuration (paths, hyperparams, dataset info) |
| `prepare_data.py` | Download Roboflow datasets, convert COCO annotations to binary masks (SAM 3 refines box → mask for Taping) |
| `dataset.py` | PyTorch Dataset with text prompt randomization and augmentation |
| `train.py` | PyTorch Lightning training module and data module |
| `evaluate.py` | Test mIoU/Dice + prompt-sensitivity diagnostic |
| `predict.py` | Generate `{image_id}__{prompt}.png` masks at source resolution |
| `visualize.py` | Generate test-examples figure (worst/median/best per class) + training curves |
