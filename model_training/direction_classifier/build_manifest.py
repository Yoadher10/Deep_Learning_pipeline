#!/usr/bin/env python3
"""
Step 1 of ViT training: fold the per-crop label JSONs into one manifest.csv.

Walks the training-set label folder, pairs each JSON with its image, converts
the raw label values into training targets, and writes one CSV row per usable
sample with the columns train.py and dataset.py expect:

    image, frame_id, direction_target, pose_target

The raw pose values (0 = N/A, 1 = Regular, 2 = Upside Down) become
-100 / 0 / 1, where -100 is torch's ignore_index -- that is how samples whose
direction makes pose meaningless (No Fish / N / S) are excluded from the pose
loss without being dropped from direction training. ``frame_id`` is parsed out
of the filename and is what split_dataset.py groups on to keep crops from the
same frame in the same split.

Usage:
    python model_training/direction_classifier/build_manifest.py
    python model_training/direction_classifier/build_manifest.py \
        --images /path/to/trainset/images \
        --labels /path/to/trainset/labels \
        --output /path/to/trainset/manifest.csv

Defaults come from config.py (i.e. from $FISH_PIPELINE_DATA).
"""

import sys
import csv
import json
import re
import argparse
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config

# Defaults; overridable with --images / --labels / --output (see main()).
IMAGES_DIR = config.TRAINSET_IMAGES
LABELS_DIR = config.TRAINSET_LABELS
OUTPUT_CSV = config.MANIFEST_CSV

# Example:
# frame_001347_fish_00.json
FRAME_PATTERN = re.compile(r"frame_(\d+)_fish_\d+")


def extract_frame_id(filename_stem):
    match = FRAME_PATTERN.fullmatch(filename_stem)

    if not match:
        raise ValueError(
            f"Could not extract frame number from: {filename_stem}"
        )

    return int(match.group(1))


def convert_pose(raw_pose):
    """
    Raw labels:
        0 = N/A
        1 = Regular
        2 = Upside Down

    Training targets:
       -100 = ignore pose loss
        0   = Regular
        1   = Upside Down
    """
    if raw_pose == 0:
        return -100, 0

    if raw_pose == 1:
        return 0, 1

    if raw_pose == 2:
        return 1, 1

    raise ValueError(f"Invalid raw pose: {raw_pose}")


def main():
    parser = argparse.ArgumentParser(
        description="Build manifest.csv from trainset labels + images.",
    )
    parser.add_argument("--images", type=Path, default=IMAGES_DIR,
                         help=f"Trainset images folder (default: {IMAGES_DIR})")
    parser.add_argument("--labels", type=Path, default=LABELS_DIR,
                         help=f"Trainset label JSON folder (default: {LABELS_DIR})")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV,
                         help=f"Output manifest CSV path (default: {OUTPUT_CSV})")
    args = parser.parse_args()

    images_dir = args.images
    labels_dir = args.labels
    output_csv = args.output

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    json_files = sorted(labels_dir.glob("*.json"))

    rows = []
    missing_images = []

    for json_path in json_files:

        with open(json_path, "r", encoding="utf-8") as f:
            label = json.load(f)

        direction = label["direction"]
        raw_pose = label["pose"]

        # After removing Face/Tail,
        # direction must be 0..8
        if direction not in range(9):
            raise ValueError(
                f"{json_path.name}: unexpected direction={direction}"
            )

        if raw_pose not in {0, 1, 2}:
            raise ValueError(
                f"{json_path.name}: unexpected pose={raw_pose}"
            )

        image_name = json_path.stem + ".png"
        image_path = images_dir / image_name

        if not image_path.exists():
            missing_images.append(image_name)
            continue

        frame_id = extract_frame_id(json_path.stem)

        pose_target, pose_valid = convert_pose(raw_pose)

        rows.append(
            {
                "image": image_name,
                "frame_id": frame_id,

                # Original annotation
                "direction_raw": direction,
                "pose_raw": raw_pose,

                # Model targets
                "direction_target": direction,
                "pose_target": pose_target,

                # 1 = participate in pose loss
                # 0 = ignored for pose
                "pose_valid": pose_valid,
            }
        )

    rows.sort(
        key=lambda row: (
            row["frame_id"],
            row["image"]
        )
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:

        fieldnames = [
            "image",
            "frame_id",
            "direction_raw",
            "pose_raw",
            "direction_target",
            "pose_target",
            "pose_valid",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 70)
    print("MANIFEST CREATED")
    print("=" * 70)

    print(f"Samples written: {len(rows)}")
    print(f"Missing images:  {len(missing_images)}")
    print(f"Output:          {output_csv}")

    if missing_images:
        print()
        print("Missing images:")

        for name in missing_images:
            print(f"  {name}")


if __name__ == "__main__":
    main()