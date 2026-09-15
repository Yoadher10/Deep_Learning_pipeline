#!/usr/bin/env python3
"""
Re-apply the phase-1 static-background rejection at a different threshold,
without re-running YOLO.

Phase 1 writes  boxes.csv  (crop, x1, y1, x2, y2, conf, match_pct, ignored)
next to the rois/ folder. This script just re-sorts the existing crops between

    rois/                       kept  (match_pct <  threshold)
    rois_ignored_background/    background (match_pct >= threshold)

    python background_match_filter.py --data RUN_DIR --threshold 92
    python background_match_filter.py --data RUN_DIR --threshold 88 --hist

`match_pct` = 100 * (1 - mean_abs_RGB_diff / 255); 100 = identical to the
per-video median background.
"""

import csv
import shutil
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True,
                        help="folder containing boxes.csv and rois/")
    parser.add_argument("--threshold", type=float, default=90.0)
    parser.add_argument("--hist", action="store_true",
                        help="print the match_pct histogram and exit")
    args = parser.parse_args()

    boxes_csv = args.data / "boxes.csv"
    rois_dir = args.data / "rois"
    ignored_dir = args.data / "rois_ignored_background"

    rows = []
    with boxes_csv.open() as fh:
        for r in csv.DictReader(fh):
            r["match_pct"] = float(r["match_pct"])
            rows.append(r)

    pcts = sorted(r["match_pct"] for r in rows)
    print(f"{len(rows)} boxes")
    print("match_pct histogram:")
    for lo in range(0, 100, 5):
        n = sum(1 for p in pcts if lo <= p < lo + 5)
        print(f"  {lo:3d}-{lo+5:3d}%  {n:6d}  {'#' * (n * 60 // max(1, len(pcts)))}")
    for t in (80, 85, 88, 90, 92, 95):
        print(f"  >= {t}%: {sum(1 for p in pcts if p >= t)}")
    if args.hist:
        return

    ignored_dir.mkdir(parents=True, exist_ok=True)
    moved_out = moved_in = 0
    for r in rows:
        name = r["crop"]
        should_ignore = r["match_pct"] >= args.threshold
        in_rois = rois_dir / name
        in_ignored = ignored_dir / name

        if should_ignore and in_rois.is_file():
            shutil.move(str(in_rois), str(in_ignored))
            moved_out += 1
        elif not should_ignore and in_ignored.is_file():
            shutil.move(str(in_ignored), str(in_rois))
            moved_in += 1

    kept = sum(1 for r in rows if r["match_pct"] < args.threshold)
    print(f"\nthreshold {args.threshold}%: "
          f"{kept} kept, {len(rows) - kept} ignored "
          f"({moved_out} moved to ignored, {moved_in} moved back)")


if __name__ == "__main__":
    main()
