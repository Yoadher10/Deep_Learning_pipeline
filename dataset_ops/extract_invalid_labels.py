#!/usr/bin/env python3

"""
Find label JSONs whose (direction, pose) combination breaks the labeling
rules (see is_valid_combination) and copy their images out for re-labelling.

    python extract_invalid_labels.py [--images DIR] [--labels DIR] [--output DIR]
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

LABELS_DIR = config.LABELS_DIR
IMAGES_DIR = config.ROIS_DIR
OUTPUT_DIR = config.DATA_ROOT / "fixed_invalid"


def is_valid_combination(direction, pose):
    """
    Valid labeling rules:

    No Fish -> pose must be 0
    N       -> pose must be 0
    S       -> pose must be 0

    All other directions -> pose must be 1 or 2
    """
    if direction in {0, 1, 5}:
        return pose == 0

    return pose in {1, 2}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(LABELS_DIR.glob("*.json"))

    invalid_count = 0
    copied_count = 0
    missing_images = []

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Could not read {json_path.name}: {e}")
            continue

        direction = data.get("direction")
        pose = data.get("pose")

        # Skip malformed label values
        if direction not in range(0, 11) or pose not in range(0, 3):
            continue

        if is_valid_combination(direction, pose):
            continue

        invalid_count += 1

        # frame_000017_fish_03.json
        # ->
        # frame_000017_fish_03.png
        image_name = json_path.stem + ".png"
        source_image = IMAGES_DIR / image_name
        destination_image = OUTPUT_DIR / image_name

        if not source_image.exists():
            missing_images.append(image_name)
            print(f"MISSING IMAGE: {image_name}")
            continue

        shutil.copy2(source_image, destination_image)
        copied_count += 1

        print(
            f"Copied: {image_name} "
            f"[direction={direction}, pose={pose}]"
        )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Invalid labels found: {invalid_count}")
    print(f"Images copied:        {copied_count}")
    print(f"Missing images:       {len(missing_images)}")

    if missing_images:
        print()
        print("Missing:")
        for image_name in missing_images:
            print(f"  {image_name}")


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

    main()