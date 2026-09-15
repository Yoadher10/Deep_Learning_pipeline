#!/usr/bin/env python3

import sys
import json
from pathlib import Path
from collections import Counter

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config

# Folder of manual label JSONs to analyse. Override with:
#     python analyze_labels.py /path/to/labels
LABELS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else config.LABELS_DIR

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
    9: "Face",
    10: "Tail",
}

POSE_NAMES = {
    0: "N/A",
    1: "Regular",
    2: "Upside Down",
}


def is_valid_combination(direction, pose):
    """
    Labeling rules:

    No Fish -> pose must be 0
    N       -> pose must be 0
    S       -> pose must be 0

    All other directions -> pose should be 1 or 2
    """
    if direction in {0, 1, 5}:
        return pose == 0

    return pose in {1, 2}


def main():
    json_files = sorted(LABELS_DIR.glob("*.json"))

    direction_counts = Counter()
    pose_counts = Counter()
    combination_counts = Counter()

    malformed_files = []
    invalid_values = []
    invalid_combinations = []

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            malformed_files.append((json_path.name, str(e)))
            continue

        direction = data.get("direction")
        pose = data.get("pose")

        if direction not in range(0, 11) or pose not in range(0, 3):
            invalid_values.append(
                (json_path.name, direction, pose)
            )
            continue

        direction_counts[direction] += 1
        pose_counts[pose] += 1
        combination_counts[(direction, pose)] += 1

        if not is_valid_combination(direction, pose):
            invalid_combinations.append(
                (json_path.name, direction, pose)
            )

    valid_samples = sum(direction_counts.values())

    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Folder:             {LABELS_DIR}")
    print(f"JSON files found:   {len(json_files)}")
    print(f"Valid labels:       {valid_samples}")
    print(f"Malformed files:    {len(malformed_files)}")
    print(f"Invalid values:     {len(invalid_values)}")

    print()

    # --------------------------------------------------------
    # Direction statistics
    # --------------------------------------------------------

    print("=" * 70)
    print("DIRECTION DISTRIBUTION")
    print("=" * 70)

    for direction in range(11):
        count = direction_counts[direction]
        percentage = (
            100 * count / valid_samples
            if valid_samples
            else 0
        )

        print(
            f"{direction:2d}  "
            f"{DIRECTION_NAMES[direction]:12s}  "
            f"{count:5d}  "
            f"{percentage:6.2f}%"
        )

    print()

    # --------------------------------------------------------
    # Raw pose statistics
    # --------------------------------------------------------

    print("=" * 70)
    print("RAW POSE DISTRIBUTION")
    print("=" * 70)

    for pose in range(3):
        count = pose_counts[pose]
        percentage = (
            100 * count / valid_samples
            if valid_samples
            else 0
        )

        print(
            f"{pose}  "
            f"{POSE_NAMES[pose]:15s}  "
            f"{count:5d}  "
            f"{percentage:6.2f}%"
        )

    print()

    # --------------------------------------------------------
    # Valid pose statistics
    # --------------------------------------------------------

    regular = pose_counts[1]
    upside_down = pose_counts[2]
    valid_pose_total = regular + upside_down

    print("=" * 70)
    print("POSE DISTRIBUTION, VALID POSE SAMPLES ONLY")
    print("=" * 70)

    if valid_pose_total > 0:
        print(
            f"Regular:       {regular:5d}  "
            f"{100 * regular / valid_pose_total:6.2f}%"
        )

        print(
            f"Upside Down:   {upside_down:5d}  "
            f"{100 * upside_down / valid_pose_total:6.2f}%"
        )

        print(f"Total valid pose samples: {valid_pose_total}")

    print()

    # --------------------------------------------------------
    # Direction x Pose
    # --------------------------------------------------------

    print("=" * 70)
    print("DIRECTION x POSE")
    print("=" * 70)

    header = (
        f"{'Dir':<5}"
        f"{'Name':<12}"
        f"{'N/A':>8}"
        f"{'Regular':>10}"
        f"{'Upside':>10}"
        f"{'Total':>8}"
    )

    print(header)
    print("-" * len(header))

    for direction in range(11):
        na = combination_counts[(direction, 0)]
        regular = combination_counts[(direction, 1)]
        upside = combination_counts[(direction, 2)]

        total = na + regular + upside

        print(
            f"{direction:<5}"
            f"{DIRECTION_NAMES[direction]:<12}"
            f"{na:>8}"
            f"{regular:>10}"
            f"{upside:>10}"
            f"{total:>8}"
        )

    print()

    # --------------------------------------------------------
    # Composite class distribution
    # --------------------------------------------------------

    print("=" * 70)
    print("COMPOSITE LABEL COUNTS")
    print("=" * 70)

    for (direction, pose), count in sorted(
        combination_counts.items()
    ):
        percentage = (
            100 * count / valid_samples
            if valid_samples
            else 0
        )

        print(
            f"direction={direction:2d} "
            f"({DIRECTION_NAMES[direction]:7s}), "
            f"pose={pose} "
            f"({POSE_NAMES[pose]:11s}) "
            f"-> {count:5d} "
            f"({percentage:6.2f}%)"
        )

    print()

    # --------------------------------------------------------
    # Validation problems
    # --------------------------------------------------------

    print("=" * 70)
    print("LABEL VALIDATION")
    print("=" * 70)

    print(f"Malformed JSON files:       {len(malformed_files)}")
    print(f"Invalid direction/pose:     {len(invalid_values)}")
    print(f"Invalid label combinations: {len(invalid_combinations)}")

    if malformed_files:
        print()
        print("Malformed files:")

        for filename, error in malformed_files:
            print(f"  {filename}: {error}")

    if invalid_values:
        print()
        print("Files with invalid values:")

        for filename, direction, pose in invalid_values:
            print(
                f"  {filename}: "
                f"direction={direction}, pose={pose}"
            )

    if invalid_combinations:
        print()
        print("Invalid direction/pose combinations:")

        for filename, direction, pose in invalid_combinations:
            print(
                f"  {filename}: "
                f"direction={direction} "
                f"({DIRECTION_NAMES[direction]}), "
                f"pose={pose} "
                f"({POSE_NAMES[pose]})"
            )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()