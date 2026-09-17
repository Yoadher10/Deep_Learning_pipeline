#!/usr/bin/env python3
"""
Step 2 of ViT training: split manifest.csv into train / val / test CSVs.

The split is by *temporal chunk*, not by row. Consecutive video frames are
nearly identical, so a random per-crop split would put near-copies of the same
fish in both train and test and report an accuracy the model does not have.
Frames are therefore grouped into blocks of CHUNK_SIZE_FRAMES and whole chunks
are assigned to a split, which removes that leakage.

Because upside-down fish are rare and clustered in time, a purely random chunk
assignment easily starves the validation or test split of them. The script
searches many random chunk assignments and keeps the one that lands closest to
the target 70/15/15 ratio while still meeting the MIN_UPSIDE_* floors.

Usage:
    python model_training/direction_classifier/split_dataset.py
    python model_training/direction_classifier/split_dataset.py \
        --manifest /path/to/trainset/manifest.csv \
        --splits-dir /path/to/trainset/splits

Defaults come from config.py (i.e. from $FISH_PIPELINE_DATA). The split is
seeded with RANDOM_SEED, so re-running it reproduces the same three CSVs.
"""

import sys
import csv
import random
import argparse
from collections import Counter, defaultdict
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config

# Defaults; overridable with --manifest / --splits-dir (see main()).
MANIFEST_PATH = config.MANIFEST_CSV
SPLITS_DIR = config.SPLITS_DIR


# ============================================================
# Configuration
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CHUNK_SIZE_FRAMES = 150

# Try many possible chunk assignments
SEARCH_ITERATIONS = 250_000

RANDOM_SEED = 42


# Desired minimum number of upside-down examples.
#
# With ~198 total, proportional allocation is approximately:
# train = 139
# val   = 30
# test  = 30
#
# These minimums prevent bad splits like test=21.
MIN_UPSIDE_TRAIN = 125
MIN_UPSIDE_VAL = 27
MIN_UPSIDE_TEST = 27


# ============================================================
# Loading
# ============================================================

def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["frame_id"] = int(row["frame_id"])
        row["direction_target"] = int(row["direction_target"])
        row["pose_target"] = int(row["pose_target"])
        row["pose_valid"] = int(row["pose_valid"])

    return rows


# ============================================================
# Temporal grouping
# ============================================================

def get_chunk_id(frame_id):
    return frame_id // CHUNK_SIZE_FRAMES


def build_chunks(rows):
    chunks = defaultdict(list)

    for row in rows:
        chunk_id = get_chunk_id(row["frame_id"])
        chunks[chunk_id].append(row)

    return dict(chunks)


def rows_from_chunks(chunks, chunk_ids):
    rows = []

    for chunk_id in chunk_ids:
        rows.extend(chunks[chunk_id])

    return rows


# ============================================================
# Statistics
# ============================================================

def calculate_stats(rows):
    direction_counts = Counter()
    pose_counts = Counter()

    for row in rows:
        direction_counts[row["direction_target"]] += 1

        if row["pose_valid"] == 1:
            pose_counts[row["pose_target"]] += 1

    return direction_counts, pose_counts


def normalized_distribution(counter, classes):
    total = sum(counter.values())

    if total == 0:
        return {c: 0.0 for c in classes}

    return {
        c: counter[c] / total
        for c in classes
    }


# ============================================================
# Scoring
# ============================================================

def size_error(actual, target):
    return abs(actual - target) / target


def direction_distribution_error(rows, global_dist):
    direction_counts, _ = calculate_stats(rows)

    split_dist = normalized_distribution(
        direction_counts,
        range(9)
    )

    error = 0.0

    for cls in range(9):
        error += abs(
            split_dist[cls] - global_dist[cls]
        )

    return error


def pose_distribution_error(rows, global_dist):
    _, pose_counts = calculate_stats(rows)

    split_dist = normalized_distribution(
        pose_counts,
        range(2)
    )

    error = 0.0

    for cls in range(2):
        error += abs(
            split_dist[cls] - global_dist[cls]
        )

    return error


def upside_count(rows):
    """
    pose_target:
        0 = Regular
        1 = Upside Down
    """
    return sum(
        1
        for row in rows
        if row["pose_valid"] == 1
        and row["pose_target"] == 1
    )


def upside_target_error(actual, target):
    """
    Relative error from desired number of upside-down examples.
    """
    return abs(actual - target) / target


def evaluate_split(
    train_rows,
    val_rows,
    test_rows,
    total_samples,
    total_upside,
    global_direction_dist,
    global_pose_dist,
):

    train_target_size = total_samples * TRAIN_RATIO
    val_target_size = total_samples * VAL_RATIO
    test_target_size = total_samples * TEST_RATIO

    train_target_upside = total_upside * TRAIN_RATIO
    val_target_upside = total_upside * VAL_RATIO
    test_target_upside = total_upside * TEST_RATIO

    train_up = upside_count(train_rows)
    val_up = upside_count(val_rows)
    test_up = upside_count(test_rows)

    # --------------------------------------------------------
    # Hard rejection of particularly bad splits
    # --------------------------------------------------------

    if train_up < MIN_UPSIDE_TRAIN:
        return float("inf")

    if val_up < MIN_UPSIDE_VAL:
        return float("inf")

    if test_up < MIN_UPSIDE_TEST:
        return float("inf")

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score = 0.0

    # Sample-count balance
    score += 5.0 * size_error(
        len(train_rows),
        train_target_size
    )

    score += 5.0 * size_error(
        len(val_rows),
        val_target_size
    )

    score += 5.0 * size_error(
        len(test_rows),
        test_target_size
    )

    # Direction distributions
    score += 2.0 * direction_distribution_error(
        train_rows,
        global_direction_dist
    )

    score += 2.0 * direction_distribution_error(
        val_rows,
        global_direction_dist
    )

    score += 2.0 * direction_distribution_error(
        test_rows,
        global_direction_dist
    )

    # Pose ratio
    score += 3.0 * pose_distribution_error(
        train_rows,
        global_pose_dist
    )

    score += 3.0 * pose_distribution_error(
        val_rows,
        global_pose_dist
    )

    score += 3.0 * pose_distribution_error(
        test_rows,
        global_pose_dist
    )

    # Explicitly encourage proportional upside-down counts
    score += 5.0 * upside_target_error(
        train_up,
        train_target_upside
    )

    score += 5.0 * upside_target_error(
        val_up,
        val_target_upside
    )

    score += 5.0 * upside_target_error(
        test_up,
        test_target_upside
    )

    return score


# ============================================================
# Random candidate generation
# ============================================================

def random_chunk_assignment(chunk_ids):
    shuffled = chunk_ids[:]
    random.shuffle(shuffled)

    num_chunks = len(shuffled)

    train_count = round(num_chunks * TRAIN_RATIO)
    val_count = round(num_chunks * VAL_RATIO)

    train_chunks = shuffled[:train_count]

    val_chunks = shuffled[
        train_count:
        train_count + val_count
    ]

    test_chunks = shuffled[
        train_count + val_count:
    ]

    return (
        train_chunks,
        val_chunks,
        test_chunks,
    )


# ============================================================
# Output
# ============================================================

def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")

    rows = sorted(
        rows,
        key=lambda r: (
            r["frame_id"],
            r["image"]
        )
    )

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def print_stats(name, rows):
    direction_counts, pose_counts = calculate_stats(rows)

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Samples: {len(rows)}")

    print()
    print("Direction:")

    for direction in range(9):
        count = direction_counts[direction]

        percentage = (
            100 * count / len(rows)
            if rows else 0
        )

        print(
            f"  {direction}: "
            f"{count:4d} "
            f"({percentage:6.2f}%)"
        )

    valid_pose = sum(pose_counts.values())

    print()
    print("Pose, valid samples only:")

    if valid_pose > 0:
        regular = pose_counts[0]
        upside = pose_counts[1]

        print(
            f"  Regular:     "
            f"{regular:4d} "
            f"({100 * regular / valid_pose:6.2f}%)"
        )

        print(
            f"  Upside Down: "
            f"{upside:4d} "
            f"({100 * upside / valid_pose:6.2f}%)"
        )

        print(
            f"  Valid pose samples: {valid_pose}"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Split manifest.csv into train/val/test by temporal chunk.",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                         help=f"Input manifest CSV (default: {MANIFEST_PATH})")
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR,
                         help=f"Output folder for train/val/test.csv "
                              f"(default: {SPLITS_DIR})")
    args = parser.parse_args()

    manifest_path = args.manifest
    splits_dir = args.splits_dir
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"

    random.seed(RANDOM_SEED)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    splits_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = load_manifest(manifest_path)

    total_samples = len(rows)

    chunks = build_chunks(rows)
    chunk_ids = sorted(chunks.keys())

    print(f"Loaded samples: {total_samples}")
    print(f"Temporal chunks: {len(chunk_ids)}")
    print(
        f"Chunk size: {CHUNK_SIZE_FRAMES} "
        f"source frames"
    )

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    global_direction_counts, global_pose_counts = (
        calculate_stats(rows)
    )

    global_direction_dist = normalized_distribution(
        global_direction_counts,
        range(9)
    )

    global_pose_dist = normalized_distribution(
        global_pose_counts,
        range(2)
    )

    total_upside = global_pose_counts[1]

    print(f"Total upside-down samples: {total_upside}")

    print()
    print("Target upside-down counts:")
    print(
        f"  Train: ~{total_upside * TRAIN_RATIO:.1f}"
    )
    print(
        f"  Val:   ~{total_upside * VAL_RATIO:.1f}"
    )
    print(
        f"  Test:  ~{total_upside * TEST_RATIO:.1f}"
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    best_score = float("inf")
    best_split = None

    valid_candidates = 0

    for iteration in range(SEARCH_ITERATIONS):

        (
            train_chunk_ids,
            val_chunk_ids,
            test_chunk_ids,
        ) = random_chunk_assignment(chunk_ids)

        train_rows = rows_from_chunks(
            chunks,
            train_chunk_ids
        )

        val_rows = rows_from_chunks(
            chunks,
            val_chunk_ids
        )

        test_rows = rows_from_chunks(
            chunks,
            test_chunk_ids
        )

        score = evaluate_split(
            train_rows,
            val_rows,
            test_rows,
            total_samples,
            total_upside,
            global_direction_dist,
            global_pose_dist,
        )

        if score == float("inf"):
            continue

        valid_candidates += 1

        if score < best_score:
            best_score = score

            best_split = (
                train_chunk_ids,
                val_chunk_ids,
                test_chunk_ids,
                train_rows,
                val_rows,
                test_rows,
            )

    if best_split is None:
        raise RuntimeError(
            "Could not find a valid split. "
            "Try lowering the minimum upside-down requirements."
        )

    (
        train_chunk_ids,
        val_chunk_ids,
        test_chunk_ids,
        train_rows,
        val_rows,
        test_rows,
    ) = best_split

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_csv(train_path, train_rows)
    save_csv(val_path, val_rows)
    save_csv(test_path, test_rows)

    print()
    print("=" * 70)
    print("BEST SPLIT FOUND")
    print("=" * 70)

    print(f"Score: {best_score:.6f}")
    print(f"Valid candidates tested: {valid_candidates}")

    print_stats("TRAIN", train_rows)
    print_stats("VALIDATION", val_rows)
    print_stats("TEST", test_rows)

    print()
    print("=" * 70)
    print("TEMPORAL CHUNKS")
    print("=" * 70)

    print("Train chunks:")
    print(sorted(train_chunk_ids))

    print()
    print("Validation chunks:")
    print(sorted(val_chunk_ids))

    print()
    print("Test chunks:")
    print(sorted(test_chunk_ids))

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(train_path)
    print(val_path)
    print(test_path)


if __name__ == "__main__":
    main()