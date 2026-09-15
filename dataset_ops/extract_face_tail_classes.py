#!/usr/bin/env python3

"""
Copy an images+labels dataset while dropping every sample whose direction is
Face (9) or Tail (10), producing the 9-class set the ViT is trained on.

    python extract_face_tail_classes.py [--images DIR] [--labels DIR] [--output DIR]
"""

import sys
import json
import shutil
import argparse
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config

IMAGES_DIR = config.TRAINSET_IMAGES
LABELS_DIR = config.TRAINSET_LABELS
OUTPUT_DIR = config.DATA_ROOT / "trainset_no_face_tail"

OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_LABELS_DIR = OUTPUT_DIR / "labels"

# Classes that we want to exclude
EXCLUDED_DIRECTIONS = {9, 10}  # Face, Tail


def main():
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")

    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"Labels folder not found: {LABELS_DIR}")

    json_files = sorted(LABELS_DIR.glob("*.json"))

    total_labels = 0
    copied = 0
    face_removed = 0
    tail_removed = 0
    missing_images = []

    for json_path in json_files:
        total_labels += 1

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                label = json.load(f)
        except Exception as e:
            print(f"ERROR reading {json_path.name}: {e}")
            continue

        direction = label.get("direction")

        # Remove Face and Tail samples
        if direction == 9:
            face_removed += 1
            continue

        if direction == 10:
            tail_removed += 1
            continue

        # Matching image:
        # frame_001347_fish_00.json
        # ->
        # frame_001347_fish_00.png
        image_path = IMAGES_DIR / f"{json_path.stem}.png"

        if not image_path.exists():
            missing_images.append(image_path.name)
            print(f"MISSING IMAGE: {image_path.name}")
            continue

        # Copy both image and label
        shutil.copy2(
            image_path,
            OUTPUT_IMAGES_DIR / image_path.name
        )

        shutil.copy2(
            json_path,
            OUTPUT_LABELS_DIR / json_path.name
        )

        copied += 1

    print()
    print("=" * 60)
    print("UNIFIED TRAIN SET CREATED")
    print("=" * 60)

    print(f"Original labels:      {total_labels}")
    print(f"Face removed:         {face_removed}")
    print(f"Tail removed:         {tail_removed}")
    print(f"Total removed:        {face_removed + tail_removed}")
    print(f"Samples copied:       {copied}")
    print(f"Missing images:       {len(missing_images)}")

    print()
    print(f"Images saved to:")
    print(f"  {OUTPUT_IMAGES_DIR}")

    print()
    print(f"Labels saved to:")
    print(f"  {OUTPUT_LABELS_DIR}")

    if missing_images:
        print()
        print("Missing images:")
        for filename in missing_images:
            print(f"  {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=IMAGES_DIR)
    parser.add_argument("--labels", type=Path, default=LABELS_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    IMAGES_DIR = args.images
    LABELS_DIR = args.labels
    OUTPUT_DIR = args.output
    OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
    OUTPUT_LABELS_DIR = OUTPUT_DIR / "labels"

    main()