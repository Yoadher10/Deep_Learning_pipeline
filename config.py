#!/usr/bin/env python3
r"""
Central configuration for the fish direction / pose pipeline.

Every path and shared constant lives here so that no script needs
hard-coded absolute paths any more.

Data location
-------------
Code lives in this repo; the image / label / prediction folders are large
and live outside it.  Point the pipeline at your data with an env var:

    export FISH_PIPELINE_DATA=/path/to/deep_learning_data     # macOS / Linux
    set    FISH_PIPELINE_DATA=D:\deep_learning_data           # Windows

If the variable is unset, everything is assumed to sit under ``<repo>/data``.

Expected data layout under FISH_PIPELINE_DATA
--------------------------------------------
    frames/                     raw video frames (input to phase 1)
    rois/                       fish crops produced by phase 1
    predictions/                per-crop prediction JSON produced by phase 2
    labels/                     manual direction/pose labels (one JSON per crop)
    trainset/
        images/                 crops used to train the ViT classifier
        labels/                 matching manual label JSONs
        manifest.csv            built by model_training/.../build_manifest.py
        splits/{train,val,test}.csv
    vit_fish_output/            training checkpoints / logs
"""

import os
from pathlib import Path

# Allow unsupported Apple-GPU (MPS) ops to fall back to CPU rather than crash.
# Set here because config is imported before torch in every script.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# --------------------------------------------------------------------------
# Repo layout (checked-in files)
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

YOLO_WEIGHTS = (
    REPO_ROOT
    / "phase1_fish_detection"
    / "weights"
    / "yolo_fish_detector_v2.pt"
)

VIT_WEIGHTS = (
    REPO_ROOT
    / "phase2_direction_pose_inference"
    / "weights"
    / "best_model_v7.pt"
)

# --------------------------------------------------------------------------
# Data root (external, override with FISH_PIPELINE_DATA)
# --------------------------------------------------------------------------

DATA_ROOT = Path(
    os.environ.get("FISH_PIPELINE_DATA", REPO_ROOT / "data")
).expanduser().resolve()

# Phase 0 -> 1 -> 2 -> 3 pipeline folders
FRAMES_DIR = DATA_ROOT / "frames"          # phase 0 output (PNG frames)
ROIS_DIR = DATA_ROOT / "rois"              # phase 1 output (fish crops)
PREDICTIONS_DIR = DATA_ROOT / "predictions"  # phase 2 output (per-crop JSON)
SUMMARY_JSON = DATA_ROOT / "summary.json"  # phase 3 output (aggregate stats)
SUMMARY_TXT = DATA_ROOT / "summary.txt"    # phase 3 output (human-readable)
CHARTS_DIR = DATA_ROOT / "charts"          # phase 3 output (pie chart PNGs)

# Parent folder the GUI creates timestamped per-run folders in
RUNS_DIR = DATA_ROOT / "runs"


def new_run_dir(source_name: str) -> Path:
    """A fresh timestamped run folder, e.g. runs/clip_20260827-143002 ."""
    import re
    from datetime import datetime
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", source_name).strip("_") or "run"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return RUNS_DIR / f"{safe}_{stamp}"

# Manual labels (used by the labeling GUI / analysis tools)
LABELS_DIR = DATA_ROOT / "labels"

# --------------------------------------------------------------------------
# Direction-classifier training dataset
# --------------------------------------------------------------------------

TRAINSET_DIR = DATA_ROOT / "trainset"
TRAINSET_IMAGES = TRAINSET_DIR / "images"
TRAINSET_LABELS = TRAINSET_DIR / "labels"

MANIFEST_CSV = TRAINSET_DIR / "manifest.csv"

SPLITS_DIR = TRAINSET_DIR / "splits"
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"
TEST_CSV = SPLITS_DIR / "test.csv"

# Training checkpoints / analysis output
TRAIN_OUTPUT_DIR = DATA_ROOT / "vit_fish_output"

# --------------------------------------------------------------------------
# Shared model constants
# --------------------------------------------------------------------------

VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
NUM_DIRECTION_CLASSES = 9
NUM_POSE_CLASSES = 2

# Softer-than-inverse-frequency weighting for the rarer "upside down" pose.
# v7 training pool was ~3862 regular : ~796 upside-down (~4.85:1); sqrt-style ~2.2.
POSE_CLASS_WEIGHTS = [1.0, 2.2]

DIRECTION_NAMES = {
    0: "No Fish",
    1: "N",
    2: "NE",
    3: "E",
    4: "SE",
    5: "S",
    6: "SW",
    7: "W",
    8: "NW",
}

# Raw label direction values that carry no pose (used by GUIs / analysis)
DIRECTION_NAMES_WITH_FACE_TAIL = {**DIRECTION_NAMES, 9: "Face", 10: "Tail"}

POSE_NAMES = {
    0: "N/A",
    1: "Regular",
    2: "Upside Down",
}

# Directions for which pose is not applicable
POSE_NA_DIRECTIONS = {0, 1, 5}

INFERENCE_BATCH_SIZE = 32

# YOLO detector confidence threshold for phase 1
DETECTOR_CONF_THRESHOLD = 0.25

# Phase 1 static-background rejection.
# A per-video median background image is built from the frames. Every YOLO box
# is compared with the same box cut from that background; if the crop is too
# similar to the background it is almost certainly a coral / plant / equipment,
# not a fish, and it is diverted to rois_ignored_background/ instead of rois/.
# match_pct = 100 * (1 - mean_abs_RGB_diff / 255); 100 = identical to background.
BACKGROUND_FILTER_ENABLED = True
# On a test video the fish crops that were wrongly rejected topped out at ~93.5
# and the genuine background crops started at ~98, with an almost-empty valley
# between. 96 sits in that valley with margin on both sides. Check the valley
# per video with:  background_match_filter.py --data RUN --hist
BACKGROUND_MATCH_THRESHOLD = 96.0
# Number of frames sampled (evenly) to build the median background.
BACKGROUND_SAMPLE_FRAMES = 240

# --------------------------------------------------------------------------
# Phase 0 - video frame extraction
# --------------------------------------------------------------------------

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv", ".webm",
    ".mts", ".m2ts", ".ts",          # AVCHD / camcorder
}

# The recommended sampling fps targets ~this many extracted frames
# (~6 fish/frame -> ~18000 ROI crops). Clamped to <= the video's own fps.
TARGET_FRAMES = 3000

# Fallback fps when the video duration is unknown.
DEFAULT_SAMPLE_FPS = 5

# Candidate sampling rates offered to the user (filtered to <= video fps)
FPS_SUGGESTIONS = [1, 2, 3, 5, 10, 15, 25, 30]

# --------------------------------------------------------------------------
# Phase 3 - reporting
# --------------------------------------------------------------------------

# An "Upside Down" pose prediction is only counted as a real upside-down fish
# when the pose head is at least this confident; lower-confidence upside-down
# calls are folded back into "Regular" for the headline stats. Used by phase 3
# (summarize_predictions.py) and dataset_ops/sort_crops_by_class.py. On an
# unseen test video the confident calls were nearly all truly upside down and
# the low-confidence ones were nearly all regular.
UPSIDE_DOWN_CONF_THRESHOLD = 0.85


# --------------------------------------------------------------------------
# Progress reporting
# --------------------------------------------------------------------------
#
# When PIPELINE_STRUCTURED_OUTPUT=1 (the GUI sets it), phase scripts emit
# machine-readable progress lines the GUI parses:
#
#     @DEVICE <short> <human label>
#     @STEP <description>
#     @PROGRESS <done> <total> [label]
#     @CHART <absolute path to a .png>
#
# Run from a plain terminal, the same calls print friendly text instead.

import time as _time

STRUCTURED_OUTPUT = os.environ.get("PIPELINE_STRUCTURED_OUTPUT") == "1"

_last_progress_emit = 0.0


def emit_device(short, label):
    if STRUCTURED_OUTPUT:
        print(f"@DEVICE {short} {label}", flush=True)
    else:
        print(f"Device: {label}", flush=True)


def emit_step(description):
    if STRUCTURED_OUTPUT:
        print(f"@STEP {description}", flush=True)
    else:
        print(f">>> {description}", flush=True)


def emit_progress(done, total, label=""):
    """Report sub-step progress. Throttled to ~7/s except for the final tick."""
    global _last_progress_emit
    done, total = int(done), int(total)
    final = total > 0 and done >= total
    now = _time.monotonic()
    if not final and now - _last_progress_emit < 0.15:
        return
    _last_progress_emit = now

    if STRUCTURED_OUTPUT:
        print(f"@PROGRESS {done} {total} {label}".rstrip(), flush=True)
    else:
        end = "\n" if final else "\r"
        pct = f" ({100.0 * done / total:.0f}%)" if total else ""
        print(f"  {label} {done}/{total}{pct}   ".rstrip(), end=end, flush=True)


def emit_chart(path):
    if STRUCTURED_OUTPUT:
        print(f"@CHART {path}", flush=True)
    else:
        print(f"Chart: {path}", flush=True)
