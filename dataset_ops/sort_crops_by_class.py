#!/usr/bin/env python3
"""
Sort phase-1 ROI crops into per-class folders using the phase-2 predictions.

For every image in the ROI folder, look up its prediction JSON (same file
stem) and copy the image into one of:

    no_fish/                direction predicted as "No Fish"
    upside_down/            fish, pose = Upside Down
    regular/               fish, pose = Regular
    pose_not_applicable/    fish facing N or S (the model does not score pose)
    unclassified/           no matching prediction JSON was found

By default every fish bucket (regular/, upside_down/, pose_not_applicable/) is
further split into per-direction sub-folders, e.g.

    regular/NE/            regular fish facing NE
    upside_down/E/         upside-down fish facing E
    pose_not_applicable/S/ fish facing S (pose not scored)

The upside_down/ bucket is additionally split by pose confidence, so a
human can review the confident calls first:

    upside_down/high_conf/NE/   pose_confidence >= threshold (default 0.95)
    upside_down/low_conf/NE/    pose_confidence <  threshold

    python sort_crops_by_class.py
    python sort_crops_by_class.py --rois DIR --predictions DIR --output DIR
    python sort_crops_by_class.py --move                 # move instead of copy
    python sort_crops_by_class.py --flat                 # no per-direction sub-folders
    python sort_crops_by_class.py --use-raw-pose         # score pose even for N/S
    python sort_crops_by_class.py --upside-conf-threshold 0.9
    python sort_crops_by_class.py --no-upside-conf-split  # single upside_down/ bucket
"""

import sys
import json
import shutil
import argparse
from pathlib import Path
from collections import Counter

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

NO_FISH_DIRECTION = 0
POSE_UPSIDE_DOWN = 1
POSE_REGULAR = 0

DIRECTION_ORDER = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

UPSIDE_CONF_LEVELS = ("high_conf", "low_conf")


def upside_conf_level(prediction: dict, threshold: float) -> str:
    """high_conf / low_conf sub-folder for an upside-down crop."""
    conf = prediction.get("pose_confidence")
    if conf is None or conf < threshold:
        return "low_conf"
    return "high_conf"


def classify(prediction: dict, use_raw_pose: bool) -> str:
    direction = prediction.get("direction")

    if direction == NO_FISH_DIRECTION:
        return "no_fish"

    pose = prediction.get("raw_pose_prediction") if use_raw_pose else prediction.get("pose")

    if pose == POSE_UPSIDE_DOWN:
        return "upside_down"
    if pose == POSE_REGULAR:
        return "regular"
    return "pose_not_applicable"


def sort_crops(rois_dir: Path, predictions_dir: Path, output_dir: Path,
               move: bool, by_direction: bool, use_raw_pose: bool,
               split_upside_conf: bool = True,
               upside_conf_threshold: float = 0.95) -> None:
    if not rois_dir.is_dir():
        raise FileNotFoundError(f"ROI folder not found: {rois_dir}")
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions folder not found: {predictions_dir}")

    images = sorted(p for p in rois_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No crop images in: {rois_dir}")

    total = len(images)
    verb = "Moving" if move else "Copying"
    print(f"{verb} {total} crops from {rois_dir}")
    config.emit_step(f"sorting {total} crops by class")

    bucket_counts = Counter()
    sub_counts = Counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    SUBDIVIDED_BUCKETS = ("regular", "upside_down", "pose_not_applicable")

    for index, image_path in enumerate(images, start=1):
        prediction_path = predictions_dir / f"{image_path.stem}.json"

        if prediction_path.is_file():
            try:
                with prediction_path.open("r", encoding="utf-8") as fh:
                    prediction = json.load(fh)
                bucket = classify(prediction, use_raw_pose)
                direction_name = prediction.get("direction_name")
            except (OSError, ValueError, json.JSONDecodeError):
                bucket, direction_name = "unclassified", None
        else:
            bucket, direction_name = "unclassified", None

        destination_dir = output_dir / bucket
        conf_level = None
        if bucket == "upside_down" and split_upside_conf and prediction_path.is_file():
            try:
                conf_level = upside_conf_level(prediction, upside_conf_threshold)
            except (TypeError, ValueError):
                conf_level = "low_conf"
            destination_dir = destination_dir / conf_level
        if by_direction and bucket in SUBDIVIDED_BUCKETS and direction_name:
            destination_dir = destination_dir / direction_name
            sub_counts[(bucket, direction_name)] += 1
        if conf_level is not None:
            bucket_counts[("upside_down", conf_level)] += 1
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination = destination_dir / image_path.name
        if move:
            shutil.move(str(image_path), destination)
        else:
            shutil.copy2(image_path, destination)

        bucket_counts[bucket] += 1
        config.emit_progress(index, total, "sorting")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    for bucket in ("no_fish", "regular", "upside_down",
                   "pose_not_applicable", "unclassified"):
        count = bucket_counts.get(bucket, 0)
        pct = 100.0 * count / total if total else 0.0
        print(f"  {bucket:<22} {count:>7}  ({pct:.1f}%)")
        if bucket == "upside_down" and split_upside_conf:
            for level in UPSIDE_CONF_LEVELS:
                lvl = bucket_counts.get(("upside_down", level), 0)
                label = f"{level} (>= {upside_conf_threshold:g})" if level == "high_conf" \
                    else f"{level} (< {upside_conf_threshold:g})"
                print(f"      {label:<18} {lvl:>7}")
        if by_direction and bucket in SUBDIVIDED_BUCKETS:
            for direction_name in DIRECTION_ORDER:
                sub = sub_counts.get((bucket, direction_name), 0)
                if sub:
                    print(f"      {direction_name:<18} {sub:>7}")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rois", type=Path, default=config.ROIS_DIR,
                        help="Folder of phase-1 crop images")
    parser.add_argument("--predictions", type=Path, default=config.PREDICTIONS_DIR,
                        help="Folder of phase-2 prediction JSON files")
    parser.add_argument("--output", type=Path,
                        default=config.DATA_ROOT / "sorted_by_class",
                        help="Folder to create the per-class subfolders in")
    parser.add_argument("--move", action="store_true",
                        help="Move the crops instead of copying them")
    parser.add_argument("--flat", action="store_true",
                        help="Do not create per-direction sub-folders")
    parser.add_argument("--use-raw-pose", action="store_true",
                        help="Use the raw pose head output even for N/S fish")
    parser.add_argument("--upside-conf-threshold", type=float,
                        default=config.UPSIDE_DOWN_CONF_THRESHOLD,
                        help="pose_confidence cutoff for upside_down/high_conf vs "
                             f"low_conf (default {config.UPSIDE_DOWN_CONF_THRESHOLD})")
    parser.add_argument("--no-upside-conf-split", action="store_true",
                        help="Keep a single upside_down/ bucket (no high/low conf split)")
    args = parser.parse_args()

    sort_crops(args.rois, args.predictions, args.output,
               args.move, not args.flat, args.use_raw_pose,
               split_upside_conf=not args.no_upside_conf_split,
               upside_conf_threshold=args.upside_conf_threshold)
