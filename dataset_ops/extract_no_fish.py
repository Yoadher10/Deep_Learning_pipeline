#!/usr/bin/env python3

"""
Pull every "No Fish" (direction 0) sample out of a labelled set into a
separate folder so the crops can be re-reviewed / re-labelled.

    python extract_no_fish.py [--images DIR] [--labels DIR] [--output DIR]
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

SOURCE_IMAGES_DIR = config.TRAINSET_IMAGES
SOURCE_LABELS_DIR = config.TRAINSET_LABELS

OUTPUT_DIR = config.DATA_ROOT / "no_fish_relabel"

OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_OLD_LABELS_DIR = OUTPUT_DIR / "old_labels"
OUTPUT_NEW_LABELS_DIR = OUTPUT_DIR / "new_labels"


# ============================================================
# Configuration
# ============================================================

NO_FISH_DIRECTION = 0


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Check source folders
    # --------------------------------------------------------

    if not SOURCE_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Images folder not found:\n{SOURCE_IMAGES_DIR}"
        )

    if not SOURCE_LABELS_DIR.exists():
        raise FileNotFoundError(
            f"Labels folder not found:\n{SOURCE_LABELS_DIR}"
        )

    # --------------------------------------------------------
    # Create output structure
    # --------------------------------------------------------

    OUTPUT_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_OLD_LABELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_NEW_LABELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find labels
    # --------------------------------------------------------

    json_files = sorted(
        SOURCE_LABELS_DIR.glob("*.json")
    )

    print("=" * 70)
    print("EXTRACTING NO-FISH SAMPLES")
    print("=" * 70)

    print(f"Labels found: {len(json_files)}")
    print()

    no_fish_count = 0
    copied_count = 0
    missing_images = []
    malformed_labels = []

    for json_path in json_files:

        # ----------------------------------------------------
        # Read label
        # ----------------------------------------------------

        try:
            with open(
                json_path,
                "r",
                encoding="utf-8",
            ) as f:
                label = json.load(f)

        except Exception as e:

            malformed_labels.append(
                (json_path.name, str(e))
            )

            continue

        direction = label.get("direction")

        # ----------------------------------------------------
        # We only want No Fish
        # ----------------------------------------------------

        if direction != NO_FISH_DIRECTION:
            continue

        no_fish_count += 1

        # ----------------------------------------------------
        # Find corresponding image
        #
        # frame_001347_fish_00.json
        # ->
        # frame_001347_fish_00.png
        # ----------------------------------------------------

        image_name = json_path.stem + ".png"

        source_image = (
            SOURCE_IMAGES_DIR / image_name
        )

        if not source_image.exists():

            missing_images.append(image_name)

            print(
                f"MISSING IMAGE: {image_name}"
            )

            continue

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        shutil.copy2(
            source_image,
            OUTPUT_IMAGES_DIR / image_name,
        )

        # ----------------------------------------------------
        # Copy original JSON as backup
        # ----------------------------------------------------

        shutil.copy2(
            json_path,
            OUTPUT_OLD_LABELS_DIR / json_path.name,
        )

        copied_count += 1

        print(
            f"Copied: {image_name}"
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"No-Fish labels found: {no_fish_count}"
    )

    print(
        f"Images copied:        {copied_count}"
    )

    print(
        f"Missing images:       {len(missing_images)}"
    )

    print(
        f"Malformed labels:     {len(malformed_labels)}"
    )

    print()
    print("Images to relabel:")
    print(OUTPUT_IMAGES_DIR)

    print()
    print("Original labels backup:")
    print(OUTPUT_OLD_LABELS_DIR)

    print()
    print("Put corrected labels here:")
    print(OUTPUT_NEW_LABELS_DIR)

    # --------------------------------------------------------
    # Problems
    # --------------------------------------------------------

    if missing_images:

        print()
        print("Missing images:")

        for image_name in missing_images:
            print(f"  {image_name}")

    if malformed_labels:

        print()
        print("Malformed labels:")

        for filename, error in malformed_labels:
            print(
                f"  {filename}: {error}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=SOURCE_IMAGES_DIR)
    parser.add_argument("--labels", type=Path, default=SOURCE_LABELS_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    SOURCE_IMAGES_DIR = args.images
    SOURCE_LABELS_DIR = args.labels
    OUTPUT_DIR = args.output
    OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
    OUTPUT_OLD_LABELS_DIR = OUTPUT_DIR / "old_labels"
    OUTPUT_NEW_LABELS_DIR = OUTPUT_DIR / "new_labels"

    main()