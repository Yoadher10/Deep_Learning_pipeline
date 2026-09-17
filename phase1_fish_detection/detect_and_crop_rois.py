#!/usr/bin/env python3
"""
Phase 1: run the fine-tuned YOLOv8 fish detector over every frame and write one
PNG crop (ROI) per detection, named ``frame_000123_fish_00.png``.

Detections are additionally screened against a per-video median background
image: anything that matches the static background closely enough is coral,
plant or equipment rather than a fish, and is diverted to
``rois_ignored_background/`` instead of ``rois/`` (see config.BACKGROUND_*).
Every box, kept or rejected, is logged to ``boxes.csv`` with its match
percentage, so the threshold can be re-tuned afterwards with
dataset_ops/background_match_filter.py without re-running YOLO.

Usage:
    python phase1_fish_detection/detect_and_crop_rois.py
    python phase1_fish_detection/detect_and_crop_rois.py \
        --input /path/to/frames --output /path/to/rois
    python phase1_fish_detection/detect_and_crop_rois.py --no-bg-filter

Defaults come from config.py: --input $FISH_PIPELINE_DATA/frames,
--output $FISH_PIPELINE_DATA/rois, --weights the bundled detector.
"""

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config
from torch_device import select_device, describe, ultralytics_arg

import cv2
import csv
import os
import argparse
import numpy as np
from ultralytics import YOLO


# --------------------------------------------------------------------------
# Static-background rejection
# --------------------------------------------------------------------------

def build_median_background(frame_paths, sample_count):
    """
    Median image over an even sample of frames. Anything that stays put for
    most of the video (coral, plants, equipment) survives the median; fish,
    which move, average out.
    """
    if len(frame_paths) > sample_count:
        idx = np.linspace(0, len(frame_paths) - 1, sample_count).astype(int)
        idx = sorted(set(idx.tolist()))
        sample = [frame_paths[i] for i in idx]
    else:
        sample = list(frame_paths)

    first = cv2.imread(str(sample[0]))
    if first is None:
        raise RuntimeError(f"Could not read frame: {sample[0]}")
    h, w = first.shape[:2]

    stack = np.empty((len(sample), h, w, 3), dtype=np.uint8)
    for k, p in enumerate(sample):
        img = cv2.imread(str(p))
        if img is None:
            stack[k] = 0
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        stack[k] = img

    return np.median(stack, axis=0).astype(np.uint8)


def background_match_pct(roi_patch, bg_patch):
    """100 = identical to the background, low = looks nothing like it."""
    if roi_patch.shape != bg_patch.shape or roi_patch.size == 0:
        return 0.0
    mad = np.abs(roi_patch.astype(np.float32) - bg_patch.astype(np.float32)).mean()
    return 100.0 * (1.0 - mad / 255.0)


# --------------------------------------------------------------------------
# Phase 1
# --------------------------------------------------------------------------

def extract_rois(input_dir, output_dir, weights_path,
                 conf_thresh=config.DETECTOR_CONF_THRESHOLD,
                 bg_filter=config.BACKGROUND_FILTER_ENABLED,
                 bg_threshold=config.BACKGROUND_MATCH_THRESHOLD):
    """
    Phase 1: Run the fine-tuned YOLOv8 fish detector over frames and crop ROIs.

    input_dir    folder of video frames        (default: config.FRAMES_DIR)
    output_dir   folder to write fish crops to  (default: config.ROIS_DIR)
    weights_path YOLO detector weights          (default: config.YOLO_WEIGHTS)

    Side outputs (next to output_dir's parent, i.e. the run/data folder):
      boxes.csv                  one row per YOLO box: crop, x1, y1, x2, y2,
                                 conf, match_pct, ignored
      median_background.png      the per-video median background
      rois_ignored_background/   crops rejected as static background
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = output_dir.parent
    boxes_csv_path = data_root / "boxes.csv"
    background_path = data_root / "median_background.png"
    ignored_dir = data_root / "rois_ignored_background"

    device = select_device()
    short, pretty = describe(device)
    config.emit_device(short, f"phase 1 YOLO on {pretty}")
    yolo_device = ultralytics_arg(device)

    config.emit_step("scanning frames")
    valid_extensions = ('.jpg', '.jpeg', '.png')
    frames = sorted(f for f in os.listdir(input_dir)
                    if f.lower().endswith(valid_extensions))
    total = len(frames)
    print(f"Found {total} frames in {input_dir}.")

    background = None
    if bg_filter:
        if total < 10:
            print("Too few frames for a background model; background filter off.")
            bg_filter = False
        else:
            config.emit_step("building median background")
            frame_paths = [Path(input_dir) / f for f in frames]
            background = build_median_background(
                frame_paths, config.BACKGROUND_SAMPLE_FRAMES)
            cv2.imwrite(str(background_path), background)
            ignored_dir.mkdir(parents=True, exist_ok=True)
            print(f"Median background saved to {background_path}")

    config.emit_step("loading YOLO fish detector")
    model = YOLO(weights_path)
    model.to(device)

    config.emit_step(f"detecting fish in {total} frames")

    total_rois_extracted = 0
    total_ignored = 0

    with open(boxes_csv_path, "w", newline="") as csv_fh:
        writer = csv.writer(csv_fh)
        writer.writerow(["crop", "x1", "y1", "x2", "y2",
                         "conf", "match_pct", "ignored"])

        for frame_index, frame_name in enumerate(frames, start=1):
            frame_path = os.path.join(input_dir, frame_name)
            frame = cv2.imread(frame_path)

            if frame is None:
                config.emit_progress(frame_index, total, "detecting")
                continue

            results = model(frame, conf=conf_thresh, verbose=False,
                            device=yolo_device)

            fish_count_in_frame = 0
            base_name = os.path.splitext(frame_name)[0]

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0]) if box.conf is not None else 0.0

                    roi_patch = frame[y1:y2, x1:x2]
                    if not (roi_patch.size > 0
                            and roi_patch.shape[0] > 10
                            and roi_patch.shape[1] > 10):
                        continue

                    match_pct = 0.0
                    if bg_filter:
                        bg_patch = background[y1:y2, x1:x2]
                        match_pct = background_match_pct(roi_patch, bg_patch)

                    ignored = bg_filter and match_pct >= bg_threshold

                    crop_name = f"{base_name}_fish_{fish_count_in_frame:02d}.png"
                    dest_dir = ignored_dir if ignored else output_dir
                    cv2.imwrite(str(Path(dest_dir) / crop_name), roi_patch)

                    writer.writerow([crop_name, x1, y1, x2, y2,
                                     f"{conf:.4f}", f"{match_pct:.2f}",
                                     int(ignored)])

                    fish_count_in_frame += 1
                    if ignored:
                        total_ignored += 1
                    else:
                        total_rois_extracted += 1

            config.emit_progress(frame_index, total, "detecting")

    print(f"Phase 1 complete. {total_rois_extracted} fish ROI patches -> "
          f"'{output_dir}'.")
    if bg_filter:
        print(f"{total_ignored} boxes rejected as static background -> "
              f"'{ignored_dir}'.")
    print(f"Box coordinates written to {boxes_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1: Fish YOLO detection and ROI extraction")
    parser.add_argument("--input", type=str, default=str(config.FRAMES_DIR),
                        help="Directory containing video frames")
    parser.add_argument("--output", type=str, default=str(config.ROIS_DIR),
                        help="Directory to save cropped ROI patches")
    parser.add_argument("--weights", type=str, default=str(config.YOLO_WEIGHTS),
                        help="Path to the YOLO detector weights")
    parser.add_argument("--conf", type=float, default=config.DETECTOR_CONF_THRESHOLD,
                        help="YOLO confidence threshold")
    parser.add_argument("--no-bg-filter", action="store_true",
                        help="Disable the static-background rejection")
    parser.add_argument("--bg-threshold", type=float,
                        default=config.BACKGROUND_MATCH_THRESHOLD,
                        help="Reject boxes with match_pct >= this value")

    args = parser.parse_args()

    extract_rois(args.input, args.output, args.weights, args.conf,
                 bg_filter=not args.no_bg_filter,
                 bg_threshold=args.bg_threshold)
