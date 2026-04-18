"""Central configuration for the Prompted Segmentation pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
PREDICTION_DIR = ROOT / "predictions"
FIGURE_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "outputs"

for d in (DATA_DIR, CHECKPOINT_DIR, PREDICTION_DIR, FIGURE_DIR, OUTPUT_DIR):
    d.mkdir(exist_ok=True)

# ── Roboflow ───────────────────────────────────────────────────────────
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

DATASETS = {
    "taping": {
        "workspace": "objectdetect-pu6rn",
        "project": "drywall-join-detect",
        "version": 2,
        "download_format": "coco",
        "needs_test_split": True,
        # Boxes-only dataset → refine each box into a pixel mask using SAM 3
        # with the text prompt below as a positive box + text prompt.
        "sam3_prompt": "seam",
        "prompts": [
            "segment taping area",
            "segment joint tape",
            "segment drywall seam",
        ],
    },
    "cracks": {
        "workspace": "cracksegm20",
        "project": "crack-segmentation-2.0",
        "version": 4,
        "download_format": "coco-segmentation",
        "needs_test_split": False,
        "prompts": [
            "segment crack",
            "segment wall crack",
        ],
    },
}

# ── Model ──────────────────────────────────────────────────────────────
MODEL_NAME = "CIDAS/clipseg-rd64-refined"
IMG_SIZE = 352

# ── Training ───────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 40
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
UNFREEZE_CLIP_LAYERS = 4       # last N CLIP encoder layers to unfreeze
WARMUP_RATIO = 0.05
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Per-step probability of replacing a training sample's prompt with a prompt
# from a DIFFERENT dataset and forcing the target mask to zeros. Forces the
# model to actually condition on the text. Set to 0.0 to disable.
NEG_PROMPT_PROB = 0.1

# ── Evaluation ─────────────────────────────────────────────────────────
IOU_THRESHOLD = 0.5
