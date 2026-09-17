#!/usr/bin/env python3
"""
Score a trained ViT checkpoint against the held-out validation split and write
out everything needed to see *how* it fails.

Unlike evaluation/review_predictions_gui.py, this needs ground-truth labels, so
it only works on a labelled split built by split_dataset.py -- not on a fresh
pipeline run. It writes, under <output-dir>:

    validation_predictions.csv      per-sample truth vs. prediction
    direction_confusion_matrix.csv  9x9 direction confusion matrix
    pose_confusion_matrix.csv       2x2 pose confusion matrix
    direction_errors/               copies of every misclassified crop
    pose_errors/                    likewise for pose

The output folder is wiped on each run so stale errors never linger.

Usage:
    python evaluation/validation_error_analysis.py
    python evaluation/validation_error_analysis.py --weights /path/to/best_model_v7.pt
    python evaluation/validation_error_analysis.py \
        --images /path/to/trainset/images \
        --splits-dir /path/to/trainset/splits \
        --output-dir /path/to/analysis

All path defaults come from config.py (i.e. from $FISH_PIPELINE_DATA).
"""

import sys
import csv
import shutil
import argparse
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config

# The dataset / model definitions live with the training code
sys.path.insert(0, str(config.REPO_ROOT / "model_training" / "direction_classifier"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import create_datasets, IMAGES_DIR
from train import MultiTaskFishViT


# ============================================================
# Configuration (defaults from config.py)
# ============================================================
#
# main() overwrites these from the command-line flags before anything reads
# them, so a flag and the default below are interchangeable throughout.

BEST_MODEL_PATH = config.VIT_WEIGHTS

OUTPUT_DIR = config.TRAIN_OUTPUT_DIR / "validation_error_analysis"

DIRECTION_ERRORS_DIR = OUTPUT_DIR / "direction_errors"
POSE_ERRORS_DIR = OUTPUT_DIR / "pose_errors"

PREDICTIONS_CSV = OUTPUT_DIR / "validation_predictions.csv"

DIRECTION_CM_CSV = (
    OUTPUT_DIR / "direction_confusion_matrix.csv"
)

POSE_CM_CSV = (
    OUTPUT_DIR / "pose_confusion_matrix.csv"
)

# Dataset location; overridable with --images / --splits-dir.
# IMAGES_DIR is imported from dataset.py (config.TRAINSET_IMAGES) and is what
# the error-copying code reads, so --images rebinds it here.
SPLITS_DIR = config.SPLITS_DIR

BATCH_SIZE = 16


DIRECTION_NAMES = {
    0: "NoFish",
    1: "N",
    2: "NE",
    3: "E",
    4: "SE",
    5: "S",
    6: "SW",
    7: "W",
    8: "NW",
}

POSE_NAMES = {
    0: "Regular",
    1: "UpsideDown",
}


# ============================================================
# Device
# ============================================================

def get_device():
    from torch_device import select_device
    return select_device()


# ============================================================
# Metrics
# ============================================================

def calculate_class_metrics(confusion_matrix):
    metrics = []

    num_classes = confusion_matrix.shape[0]

    for cls in range(num_classes):

        tp = confusion_matrix[cls, cls]

        fp = confusion_matrix[:, cls].sum() - tp
        fn = confusion_matrix[cls, :].sum() - tp

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        if precision + recall > 0:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        support = confusion_matrix[cls, :].sum()

        metrics.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
            }
        )

    return metrics


def save_confusion_matrix(
    path,
    matrix,
    class_names,
):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["true/pred"]
            + [class_names[i] for i in range(len(class_names))]
        )

        for i in range(len(class_names)):
            writer.writerow(
                [class_names[i]]
                + matrix[i].tolist()
            )


# ============================================================
# Main
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Confusion matrices and misclassified-crop dumps for a trained "
            "ViT checkpoint, against the held-out validation split."
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=BEST_MODEL_PATH,
        help="ViT checkpoint to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=IMAGES_DIR,
        help=(
            "Folder the split CSVs' image names resolve against "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_DIR,
        help=(
            "Folder holding train.csv / val.csv / test.csv "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=(
            "Where the CSVs and error crops are written. WIPED on each run "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Inference batch size (default: %(default)s)",
    )
    return parser.parse_args(argv)


def apply_args(args):
    """Push the parsed flags onto the module-level configuration constants."""
    global BEST_MODEL_PATH, IMAGES_DIR, SPLITS_DIR, BATCH_SIZE
    global OUTPUT_DIR, DIRECTION_ERRORS_DIR, POSE_ERRORS_DIR
    global PREDICTIONS_CSV, DIRECTION_CM_CSV, POSE_CM_CSV

    BEST_MODEL_PATH = args.weights
    IMAGES_DIR = args.images
    SPLITS_DIR = args.splits_dir
    BATCH_SIZE = args.batch_size

    OUTPUT_DIR = args.output_dir
    DIRECTION_ERRORS_DIR = OUTPUT_DIR / "direction_errors"
    POSE_ERRORS_DIR = OUTPUT_DIR / "pose_errors"
    PREDICTIONS_CSV = OUTPUT_DIR / "validation_predictions.csv"
    DIRECTION_CM_CSV = OUTPUT_DIR / "direction_confusion_matrix.csv"
    POSE_CM_CSV = OUTPUT_DIR / "pose_confusion_matrix.csv"


@torch.no_grad()
def main(argv=None):
    apply_args(parse_args(argv))

    device = get_device()

    print("=" * 70)
    print("VALIDATION ERROR ANALYSIS")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"Checkpoint: {BEST_MODEL_PATH}")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {BEST_MODEL_PATH}"
        )

    # Clean previous analysis so stale errors do not remain
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    DIRECTION_ERRORS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    POSE_ERRORS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        _,
        val_dataset,
        _,
        _,
    ) = create_datasets(
        images_dir=IMAGES_DIR,
        splits_dir=SPLITS_DIR,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Validation samples: {len(val_dataset)}")

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model_name = checkpoint["model_name"]

    model = MultiTaskFishViT(
        model_name=model_name,
        num_direction_classes=checkpoint[
            "num_direction_classes"
        ],
        num_pose_classes=checkpoint[
            "num_pose_classes"
        ],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    direction_cm = np.zeros(
        (9, 9),
        dtype=int,
    )

    pose_cm = np.zeros(
        (2, 2),
        dtype=int,
    )

    prediction_rows = []

    direction_error_count = 0
    pose_error_count = 0

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    for batch in val_loader:

        pixel_values = batch["pixel_values"].to(
            device,
            non_blocking=True,
        )

        direction_targets = batch["direction"].to(device)
        pose_targets = batch["pose"].to(device)

        image_names = batch["image_name"]

        direction_logits, pose_logits = model(
            pixel_values
        )

        direction_probs = torch.softmax(
            direction_logits,
            dim=1,
        )

        pose_probs = torch.softmax(
            pose_logits,
            dim=1,
        )

        direction_predictions = direction_logits.argmax(
            dim=1
        )

        pose_predictions = pose_logits.argmax(
            dim=1
        )

        for i, image_name in enumerate(image_names):

            true_direction = direction_targets[i].item()
            pred_direction = direction_predictions[i].item()

            direction_confidence = (
                direction_probs[i, pred_direction].item()
            )

            direction_cm[
                true_direction,
                pred_direction,
            ] += 1

            # ------------------------------------------------
            # Direction error copy
            # ------------------------------------------------

            if true_direction != pred_direction:

                direction_error_count += 1

                folder_name = (
                    f"true_{DIRECTION_NAMES[true_direction]}"
                    f"__pred_{DIRECTION_NAMES[pred_direction]}"
                )

                destination_dir = (
                    DIRECTION_ERRORS_DIR
                    / folder_name
                )

                destination_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                source = IMAGES_DIR / image_name

                shutil.copy2(
                    source,
                    destination_dir / image_name,
                )

            # ------------------------------------------------
            # Pose
            # ------------------------------------------------

            true_pose = pose_targets[i].item()

            pose_valid = true_pose != -100

            if pose_valid:

                pred_pose = pose_predictions[i].item()

                pose_confidence = (
                    pose_probs[i, pred_pose].item()
                )

                pose_cm[
                    true_pose,
                    pred_pose,
                ] += 1

                if true_pose != pred_pose:

                    pose_error_count += 1

                    folder_name = (
                        f"true_{POSE_NAMES[true_pose]}"
                        f"__pred_{POSE_NAMES[pred_pose]}"
                    )

                    destination_dir = (
                        POSE_ERRORS_DIR
                        / folder_name
                    )

                    destination_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    source = IMAGES_DIR / image_name

                    shutil.copy2(
                        source,
                        destination_dir / image_name,
                    )

            else:

                pred_pose = ""
                pose_confidence = ""

            # ------------------------------------------------
            # CSV row
            # ------------------------------------------------

            prediction_rows.append(
                {
                    "image": image_name,

                    "true_direction":
                        true_direction,

                    "true_direction_name":
                        DIRECTION_NAMES[true_direction],

                    "pred_direction":
                        pred_direction,

                    "pred_direction_name":
                        DIRECTION_NAMES[pred_direction],

                    "direction_confidence":
                        direction_confidence,

                    "direction_correct":
                        int(
                            true_direction
                            == pred_direction
                        ),

                    "true_pose":
                        true_pose,

                    "pred_pose":
                        pred_pose,

                    "pose_confidence":
                        pose_confidence,

                    "pose_valid":
                        int(pose_valid),

                    "pose_correct":
                        (
                            int(true_pose == pred_pose)
                            if pose_valid
                            else ""
                        ),
                }
            )

    # ========================================================
    # Save predictions
    # ========================================================

    with open(
        PREDICTIONS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=prediction_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(prediction_rows)

    save_confusion_matrix(
        DIRECTION_CM_CSV,
        direction_cm,
        DIRECTION_NAMES,
    )

    save_confusion_matrix(
        POSE_CM_CSV,
        pose_cm,
        POSE_NAMES,
    )

    # ========================================================
    # Print direction confusion matrix
    # ========================================================

    print()
    print("=" * 70)
    print("DIRECTION CONFUSION MATRIX")
    print("=" * 70)

    print(
        "Rows = TRUE class, "
        "columns = PREDICTED class"
    )

    print()

    header = "        "

    for cls in range(9):
        header += f"{DIRECTION_NAMES[cls]:>8}"

    print(header)

    for true_cls in range(9):

        row = f"{DIRECTION_NAMES[true_cls]:>8}"

        for pred_cls in range(9):
            row += f"{direction_cm[true_cls, pred_cls]:>8}"

        print(row)

    # ========================================================
    # Per-class direction metrics
    # ========================================================

    direction_metrics = calculate_class_metrics(
        direction_cm
    )

    print()
    print("=" * 70)
    print("DIRECTION PER-CLASS METRICS")
    print("=" * 70)

    print(
        f"{'Class':<10}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>10}"
    )

    print("-" * 56)

    for cls in range(9):

        metric = direction_metrics[cls]

        print(
            f"{DIRECTION_NAMES[cls]:<10}"
            f"{metric['precision']:>12.3f}"
            f"{metric['recall']:>12.3f}"
            f"{metric['f1']:>12.3f}"
            f"{metric['support']:>10}"
        )

    # ========================================================
    # Pose confusion matrix
    # ========================================================

    print()
    print("=" * 70)
    print("POSE CONFUSION MATRIX")
    print("=" * 70)

    print(
        "Rows = TRUE class, "
        "columns = PREDICTED class"
    )

    print()

    print(
        f"{'':15s}"
        f"{'Regular':>15}"
        f"{'UpsideDown':>15}"
    )

    for true_cls in range(2):

        print(
            f"{POSE_NAMES[true_cls]:15s}"
            f"{pose_cm[true_cls, 0]:>15}"
            f"{pose_cm[true_cls, 1]:>15}"
        )

    # ========================================================
    # Most common direction mistakes
    # ========================================================

    mistakes = []

    for true_cls in range(9):
        for pred_cls in range(9):

            if true_cls == pred_cls:
                continue

            count = direction_cm[
                true_cls,
                pred_cls,
            ]

            if count > 0:
                mistakes.append(
                    (
                        count,
                        true_cls,
                        pred_cls,
                    )
                )

    mistakes.sort(reverse=True)

    print()
    print("=" * 70)
    print("MOST COMMON DIRECTION ERRORS")
    print("=" * 70)

    for count, true_cls, pred_cls in mistakes:

        print(
            f"{DIRECTION_NAMES[true_cls]:8s}"
            f" -> "
            f"{DIRECTION_NAMES[pred_cls]:8s}"
            f": {count}"
        )

    # ========================================================
    # Done
    # ========================================================

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Direction errors copied: "
        f"{direction_error_count}"
    )

    print(
        f"Pose errors copied:      "
        f"{pose_error_count}"
    )

    print()
    print("Analysis directory:")
    print(OUTPUT_DIR)

    print()
    print("Predictions CSV:")
    print(PREDICTIONS_CSV)

    print()
    print("Direction confusion matrix:")
    print(DIRECTION_CM_CSV)

    print()
    print("Pose confusion matrix:")
    print(POSE_CM_CSV)


if __name__ == "__main__":
    main()