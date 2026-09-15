#!/usr/bin/env python3
"""
Turn a review session's saved review JSONs (evaluation/review_predictions_
gui.py output, in <run>/reviews/) into the human-verified accuracy numbers
for a full pipeline run: folder correctness, background-filter mistakes, and
what the pose-confidence threshold (config.UPSIDE_DOWN_CONF_THRESHOLD) is
actually costing and buying you -- including a sweep showing what would
happen at other thresholds, using nothing but the ground truth you already
established while reviewing.

Re-run this any time after adding more reviews; it always reads the full
current set of review files, so results grow as you keep classifying.

    python evaluation/review_analysis.py
    python evaluation/review_analysis.py --reviews-dir /path/to/run/reviews
    python evaluation/review_analysis.py --thresholds 0.5,0.6,0.7,0.8,0.9
    python evaluation/review_analysis.py --out-json review_analysis.json

============================================================================
GROUND-TRUTH RULES THIS SCRIPT ASSUMES (confirmed with the user; do not
change these without re-confirming, since every number below depends on
them):

  - "Correct folder" = the review button was V. Anything else means the
    folder (which is a pure function of the raw prediction) does not match
    reality, because dataset_ops/sort_crops_by_class.py sorts crops purely
    from the model's prediction with no human input.
  - Pressing X means BOTH direction and pose were wrong.
  - P (WRONG_POSE) / D (WRONG_DIR) are judged against the GATED/reported
    pose (analysis_pose_name, after the 0.85 confidence fold-down to
    Regular) -- not the raw model pose call. So for a downgraded crop,
    "WRONG_POSE" means the reviewer determined the fish really WAS upside
    down, i.e. the gate wrongly buried a correct raw call.
  - V / A / WRONG_DIR all confirm the gated pose is correct (true_pose ==
    gated_pose_name); WRONG_POSE / X mean the opposite of the gated pose is
    true (true_pose == opposite(gated_pose_name)).
============================================================================
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config

REVIEWS_DIR = config.PREDICTIONS_DIR.parent / "reviews"

# Buttons that confirm the gated/reported pose is correct.
POSE_CONFIRMS_GATED = {"V", "A", "WRONG_DIR"}
# Buttons that say the gated/reported pose is wrong (truth is the opposite).
POSE_CONTRADICTS_GATED = {"WRONG_POSE", "X"}
# Buttons with no direction/pose verdict at all (presence-only).
PRESENCE_ONLY = {"NO_FISH", "FISH"}

DEFAULT_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # .50..0.95


def opposite_pose(pose_name):
    return "Regular" if pose_name == "Upside Down" else "Upside Down"


def load_reviews(reviews_dir):
    reviews = []
    for path in sorted(reviews_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                reviews.append(json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipped unreadable {path.name}: {exc}")
    return reviews


def true_pose_of(review):
    """
    The ground truth pose, derived from the review button and the gated
    pose at review time. Returns None when pose does not apply (no fish,
    background-rejected, N/S direction, or a presence-only button).
    """
    if review.get("is_background_rejected"):
        return None
    if review.get("pose_na"):
        return None

    result = review.get("review")
    gated = review.get("analysis_pose_name")

    if result in POSE_CONFIRMS_GATED:
        return gated
    if result in POSE_CONTRADICTS_GATED:
        return opposite_pose(gated)
    return None  # NO_FISH / FISH / anything unrecognized


def analyze(reviews):
    non_bg = [r for r in reviews if not r.get("is_background_rejected")]
    bg = [r for r in reviews if r.get("is_background_rejected")]

    # ---- 1) folder correctness -------------------------------------
    button_counts = Counter(r.get("review") for r in non_bg)
    correct_folder = button_counts.get("V", 0)
    total_foldered = len(non_bg)
    wrong_folder = total_foldered - correct_folder

    wrong_breakdown = {
        btn: button_counts.get(btn, 0)
        for btn in ("A", "WRONG_POSE", "WRONG_DIR", "X", "NO_FISH", "FISH")
        if button_counts.get(btn, 0)
    }

    # ---- 2) background-rejected crops -------------------------------
    bg_counts = Counter(r.get("review") for r in bg)
    bg_false_negatives = bg_counts.get("FISH", 0)  # filter wrongly discarded a real fish
    bg_true_negatives = bg_counts.get("NO_FISH", 0)

    # ---- 3) threshold effect at the threshold actually used ---------
    downgraded = [r for r in reviews if r.get("pose_gate_downgraded")]
    downgraded_with_truth = [
        r for r in downgraded if true_pose_of(r) is not None
    ]
    ruined = [r for r in downgraded_with_truth if true_pose_of(r) == "Upside Down"]
    saved = [r for r in downgraded_with_truth if true_pose_of(r) == "Regular"]

    # ---- 4) full raw-"Upside Down" population for the threshold sweep
    # Only crops where the RAW model call was "Upside Down" are affected by
    # moving the threshold at all -- a raw "Regular" call is never touched
    # by the gate, at any threshold.
    raw_upside = [
        r for r in reviews
        if r.get("predicted_pose_name") == "Upside Down"
        and true_pose_of(r) is not None
        and r.get("pose_confidence") is not None
    ]

    return {
        "total_reviews": len(reviews),
        "classified_reviewed": total_foldered,
        "background_rejected_reviewed": len(bg),
        "folder": {
            "correct": correct_folder,
            "wrong": wrong_folder,
            "wrong_breakdown": wrong_breakdown,
        },
        "background_filter": {
            "reviewed": len(bg),
            "correctly_rejected_TN": bg_true_negatives,
            "wrongly_rejected_FN": bg_false_negatives,
        },
        "threshold_as_reviewed": {
            "threshold": config.UPSIDE_DOWN_CONF_THRESHOLD,
            "downgraded_reviewed": len(downgraded),
            "downgraded_with_pose_truth": len(downgraded_with_truth),
            "correctly_downgraded": len(saved),
            "wrongly_downgraded_ruined": len(ruined),
            "ruined_images": [r["image"] for r in ruined],
        },
        "raw_upside_down_population": raw_upside,
    }


def sweep_thresholds(raw_upside, thresholds):
    """
    For every candidate threshold T, replay the SAME gate logic
    (derive_pose_gate in review_predictions_gui.py) against the ground
    truth already established by review, restricted to crops where the raw
    model call was "Upside Down" (the only ones any threshold can affect).
    """
    rows = []
    for t in thresholds:
        ruined = 0        # true Upside Down, but gated-at-T says Regular
        shown_wrong = 0    # true Regular, but gated-at-T still says Upside Down
        correct = 0
        for r in raw_upside:
            conf = float(r["pose_confidence"])
            truth = true_pose_of(r)
            reported_at_t = "Upside Down" if conf >= t else "Regular"
            if reported_at_t == truth:
                correct += 1
            elif truth == "Upside Down":
                ruined += 1
            else:
                shown_wrong += 1
        rows.append({
            "threshold": t,
            "correct": correct,
            "ruined_buried_real_upside_down": ruined,
            "shown_wrong_as_upside_down": shown_wrong,
            "total": len(raw_upside),
        })
    return rows


def print_report(result, sweep):
    print("=" * 78)
    print("REVIEW ANALYSIS")
    print("=" * 78)
    print(f"Reviews loaded: {result['total_reviews']}")
    print(f"  classified crops reviewed         : {result['classified_reviewed']}")
    print(f"  background-rejected crops reviewed: {result['background_rejected_reviewed']}")

    print()
    print("-" * 78)
    print("1) FOLDER CORRECTNESS  (V pressed = correct folder; anything else = wrong)")
    print("-" * 78)
    f = result["folder"]
    total = f["correct"] + f["wrong"]
    pct = lambda n: (100.0 * n / total) if total else 0.0
    print(f"  Correct folder (V) : {f['correct']:5d} / {total}  ({pct(f['correct']):.1f}%)")
    print(f"  Wrong folder       : {f['wrong']:5d} / {total}  ({pct(f['wrong']):.1f}%)")
    if f["wrong_breakdown"]:
        print("  Wrong-folder breakdown by button pressed:")
        meaning = {
            "A": "direction ~45deg off",
            "WRONG_POSE": "direction right, pose wrong",
            "WRONG_DIR": "pose right, direction wrong",
            "X": "both direction AND pose wrong",
            "NO_FISH": "false positive: no fish, model said fish",
            "FISH": "presence-only fish flag",
        }
        for btn, n in sorted(f["wrong_breakdown"].items(), key=lambda kv: -kv[1]):
            print(f"    {btn:12s} {n:4d}   ({meaning.get(btn, '')})")

    print()
    print("-" * 78)
    print("2) BACKGROUND FILTER  (crops discarded before classification, no folder)")
    print("-" * 78)
    bgf = result["background_filter"]
    if bgf["reviewed"]:
        print(f"  Reviewed                : {bgf['reviewed']}")
        print(f"  Correctly rejected (TN)  : {bgf['correctly_rejected_TN']}")
        print(f"  Wrongly rejected (FN)    : {bgf['wrongly_rejected_FN']}  "
              f"(real fish the filter threw away)")
    else:
        print("  No background-rejected crops reviewed yet.")

    print()
    print("-" * 78)
    print(f"3) POSE-CONFIDENCE THRESHOLD AS ACTUALLY USED "
          f"({result['threshold_as_reviewed']['threshold']})")
    print("-" * 78)
    th = result["threshold_as_reviewed"]
    print(f"  Downgraded crops reviewed (raw Upside Down, low conf) : "
          f"{th['downgraded_reviewed']}")
    print(f"  ...with a pose verdict from review                    : "
          f"{th['downgraded_with_pose_truth']}")
    print(f"  Correctly downgraded (truly Regular)                  : "
          f"{th['correctly_downgraded']}")
    print(f"  WRONGLY downgraded / ruined (truly Upside Down)        : "
          f"{th['wrongly_downgraded_ruined']}")
    if th["ruined_images"]:
        print("  Ruined images:")
        for name in th["ruined_images"]:
            print(f"    - {name}")

    print()
    print("-" * 78)
    print("4) THRESHOLD SWEEP")
    print("   Only crops where the RAW model call was \"Upside Down\" are shown --")
    print("   raw \"Regular\" calls are never touched by any threshold choice.")
    print("-" * 78)
    n = sweep[0]["total"] if sweep else 0
    print(f"  Population: {n} reviewed crops with a raw Upside Down call\n")
    print(f"  {'threshold':>9s}  {'correct':>7s}  {'ruined':>7s}  {'shown_wrong':>11s}  "
          f"{'accuracy':>8s}")
    for row in sweep:
        acc = (100.0 * row["correct"] / row["total"]) if row["total"] else 0.0
        marker = "  <- current" if abs(
            row["threshold"] - config.UPSIDE_DOWN_CONF_THRESHOLD
        ) < 1e-9 else ""
        print(
            f"  {row['threshold']:>9.2f}  {row['correct']:>7d}  "
            f"{row['ruined_buried_real_upside_down']:>7d}  "
            f"{row['shown_wrong_as_upside_down']:>11d}  {acc:>7.1f}%{marker}"
        )
    print()
    print("  ruined       = a genuinely upside-down fish reported as Regular")
    print("  shown_wrong  = a genuinely regular fish reported as Upside Down")
    print("  Lowering the threshold trades ruined for shown_wrong (and vice")
    print("  versa raising it) -- the sweep shows exactly that trade at each point.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reviews-dir", type=Path, default=REVIEWS_DIR,
        help=f"Folder of review JSONs (default: {REVIEWS_DIR})",
    )
    parser.add_argument(
        "--thresholds", type=str, default=None,
        help="Comma-separated thresholds to sweep, e.g. 0.5,0.6,0.7,0.8,0.9 "
             "(default: 0.50..0.95 in steps of 0.05, plus the current "
             f"threshold {config.UPSIDE_DOWN_CONF_THRESHOLD} if not already included)",
    )
    parser.add_argument(
        "--out-json", type=Path, default=None,
        help="Optional path to also write the full result as JSON.",
    )
    args = parser.parse_args()

    if not args.reviews_dir.is_dir():
        raise FileNotFoundError(f"Reviews folder not found: {args.reviews_dir}")

    reviews = load_reviews(args.reviews_dir)
    if not reviews:
        raise RuntimeError(f"No review JSON files in {args.reviews_dir}")

    result = analyze(reviews)

    if args.thresholds:
        thresholds = sorted(
            {round(float(t), 4) for t in args.thresholds.split(",")}
        )
    else:
        thresholds = sorted(
            set(DEFAULT_THRESHOLDS) | {config.UPSIDE_DOWN_CONF_THRESHOLD}
        )

    sweep = sweep_thresholds(result["raw_upside_down_population"], thresholds)

    print_report(result, sweep)

    if args.out_json:
        out = dict(result)
        out.pop("raw_upside_down_population")  # internal working data only
        out["threshold_sweep"] = sweep
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
