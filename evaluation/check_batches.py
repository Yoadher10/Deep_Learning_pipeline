#!/usr/bin/env python3
"""
Flag training/validation batches that contain zero usable pose labels.

The pose head is trained with ignore_index=-100, which is how samples whose
direction makes pose meaningless (No Fish / N / S) are excluded from the pose
loss. If an entire batch happens to consist of such samples, the pose loss for
that batch is NaN and can poison the epoch. This script walks a split CSV in
batch-sized windows and prints every batch where that happens, so a NaN loss
can be confirmed or ruled out as the cause.

Usage:
    python evaluation/check_batches.py
    python evaluation/check_batches.py --csv /path/to/val.csv --batch-size 32

Defaults come from config.VAL_CSV (i.e. $FISH_PIPELINE_DATA/trainset/splits/val.csv).
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Flag split batches that have zero valid pose labels.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=config.VAL_CSV,
        help="Split CSV to scan (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Batch size to simulate. Use the same value the training run "
            "used, or the report is meaningless (default: %(default)s)"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not args.csv.is_file():
        raise SystemExit(f"Split CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    print(f"Split CSV:           {args.csv}")
    print(f"Samples:             {len(df)}")
    print(f"Batch size:          {args.batch_size}")
    print()

    bad_batches = []

    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start:start + args.batch_size]

        batch_number = start // args.batch_size + 1

        valid_pose_count = (batch["pose_target"] != -100).sum()

        if valid_pose_count == 0:
            bad_batches.append(batch_number)

            print("=" * 70)
            print(f"BATCH {batch_number} HAS ZERO VALID POSE LABELS")
            print("=" * 70)

            print(
                batch[
                    [
                        "image",
                        "direction_target",
                        "pose_target"
                    ]
                ].to_string(index=False)
            )

            print()

    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    if bad_batches:
        print(
            f"Found {len(bad_batches)} batches with "
            f"zero valid pose samples:"
        )
        print(bad_batches)
    else:
        print("No all-ignored pose batches found.")
        print("Then we need to investigate another cause.")


if __name__ == "__main__":
    main()
