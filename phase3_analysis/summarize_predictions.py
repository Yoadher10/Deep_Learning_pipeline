#!/usr/bin/env python3
"""
Phase 3: read every per-crop prediction JSON from phase 2 and produce an
aggregate report:

    - how many crops, how many "no fish", how many fish
    - direction distribution over the fish crops (count + %)
    - pose distribution and, in particular, upside-down statistics
      (overall rate, breakdown by direction, mean confidence, worst offenders)

An "Upside Down" pose call is only counted as upside-down when the pose head is
at least config.UPSIDE_DOWN_CONF_THRESHOLD confident; weaker calls are folded
into "Regular". The report also carries the raw pose-head counts, a confidence
histogram, and the list of frames that contain a confirmed upside-down fish.

Writes summary.json (machine-readable) and summary.txt (human-readable),
and prints the text report to stdout.

    python summarize_predictions.py
    python summarize_predictions.py --predictions DIR --out-json X --out-txt Y
"""

import re
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config

DIRECTION_NAMES = config.DIRECTION_NAMES          # 0..8
POSE_NA_DIRECTIONS = config.POSE_NA_DIRECTIONS     # {0, 1, 5}
NO_FISH = 0
UPSIDE_DOWN = 1                                    # pose head index
REGULAR = 0

# A pose-head "Upside Down" call is only trusted at/above this confidence;
# weaker calls are counted as Regular for the headline numbers.
UPSIDE_CONF_THRESHOLD = config.UPSIDE_DOWN_CONF_THRESHOLD

_FRAME_RE = re.compile(r"frame_(\d+)_fish")


def _pct(part, whole):
    return round(100.0 * part / whole, 2) if whole else 0.0


def _frame_id(image_name):
    m = _FRAME_RE.match(image_name or "")
    return int(m.group(1)) if m else None


def load_predictions(predictions_dir: Path) -> list[dict]:
    config.emit_step("reading prediction files")
    files = sorted(predictions_dir.glob("*.json"))
    total = len(files)
    preds = []
    for index, path in enumerate(files, start=1):
        try:
            with path.open("r", encoding="utf-8") as fh:
                preds.append(json.load(fh))
        except (OSError, ValueError, json.JSONDecodeError) as err:
            print(f"  skipped unreadable {path.name}: {err}")
        if index % 200 == 0 or index == total:
            config.emit_progress(index, total, "reading")
    return preds


def summarize(preds: list[dict]) -> dict:
    total = len(preds)

    no_fish = [p for p in preds if p.get("direction") == NO_FISH]
    fish = [p for p in preds if p.get("direction") not in (None, NO_FISH)]

    # ---- direction distribution over fish crops --------------------------
    direction_counts = Counter(p["direction"] for p in fish)
    direction_conf_sum = defaultdict(float)
    for p in fish:
        direction_conf_sum[p["direction"]] += float(p.get("direction_confidence", 0.0))

    direction_distribution = []
    for d in range(1, config.NUM_DIRECTION_CLASSES):        # 1..8
        c = direction_counts.get(d, 0)
        direction_distribution.append({
            "direction": d,
            "name": DIRECTION_NAMES[d],
            "count": c,
            "percent_of_fish": _pct(c, len(fish)),
            "mean_direction_confidence": round(direction_conf_sum[d] / c, 4) if c else None,
        })

    # ---- pose / upside-down --------------------------------------------
    # Pose is only meaningful where the model produced a pose (direction
    # not in No Fish / N / S).
    posed = [p for p in fish
             if p.get("direction") not in POSE_NA_DIRECTIONS and p.get("pose") is not None]

    def _pose_conf(p):
        return float(p.get("pose_confidence", 0.0))

    upside_raw = [p for p in posed if p.get("pose") == UPSIDE_DOWN]
    regular_raw = [p for p in posed if p.get("pose") == REGULAR]

    # Confidence split: trust only the confident upside-down calls; the rest
    # are folded into "regular" for the headline stats.
    upside = [p for p in upside_raw if _pose_conf(p) >= UPSIDE_CONF_THRESHOLD]
    upside_below = [p for p in upside_raw if _pose_conf(p) < UPSIDE_CONF_THRESHOLD]
    regular = regular_raw + upside_below

    # Histogram of every raw upside-down call's confidence, to show where the
    # mass sits relative to the threshold.
    hist_edges = [0.5, 0.6, 0.7, 0.8, 0.9, UPSIDE_CONF_THRESHOLD, 1.0001]
    hist_counts = [0] * (len(hist_edges) - 1)
    for p in upside_raw:
        c = _pose_conf(p)
        for i in range(len(hist_counts)):
            if hist_edges[i] <= c < hist_edges[i + 1]:
                hist_counts[i] += 1
                break
    upside_confidence_histogram = [
        {
            "range": f"{hist_edges[i]:.2f}-{min(hist_edges[i + 1], 1.0):.2f}",
            "count": hist_counts[i],
            "above_threshold": hist_edges[i] >= UPSIDE_CONF_THRESHOLD,
        }
        for i in range(len(hist_counts))
    ]

    # Which frames contain a confirmed upside-down fish (the review shortlist).
    frames_with_upside = Counter()
    for p in upside:
        fid = _frame_id(p.get("image"))
        if fid is not None:
            frames_with_upside[fid] += 1
    frames_with_upside_rows = [
        {"frame_id": fid, "upside_down_fish": n}
        for fid, n in sorted(frames_with_upside.items())
    ]

    upside_by_direction = Counter(p["direction"] for p in upside)
    upside_by_direction_rows = [
        {
            "direction": d,
            "name": DIRECTION_NAMES[d],
            "upside_down": upside_by_direction.get(d, 0),
            "posed_total": sum(1 for p in posed if p["direction"] == d),
            "percent_upside_down": _pct(
                upside_by_direction.get(d, 0),
                sum(1 for p in posed if p["direction"] == d),
            ),
        }
        for d in range(1, config.NUM_DIRECTION_CLASSES)
        if any(p["direction"] == d for p in posed)
    ]

    upside_sorted = sorted(
        upside, key=lambda p: float(p.get("pose_confidence", 0.0)), reverse=True
    )
    mean_upside_conf = (
        round(sum(float(p.get("pose_confidence", 0.0)) for p in upside) / len(upside), 4)
        if upside else None
    )

    mean_dir_conf = (
        round(sum(float(p.get("direction_confidence", 0.0)) for p in fish) / len(fish), 4)
        if fish else None
    )

    return {
        "totals": {
            "crops": total,
            "no_fish": len(no_fish),
            "fish": len(fish),
            "percent_no_fish": _pct(len(no_fish), total),
            "mean_direction_confidence_fish": mean_dir_conf,
        },
        "direction_distribution": direction_distribution,
        "pose": {
            "posed_crops": len(posed),
            "confidence_threshold": UPSIDE_CONF_THRESHOLD,
            "regular": len(regular),
            "upside_down": len(upside),
            "percent_upside_down": _pct(len(upside), len(posed)),
            "mean_upside_down_confidence": mean_upside_conf,
            # Raw pose-head output, before the confidence split.
            "pose_head_regular": len(regular_raw),
            "pose_head_upside_down": len(upside_raw),
            "upside_down_below_threshold": len(upside_below),
            "upside_confidence_histogram": upside_confidence_histogram,
            "frames_with_upside_down": len(frames_with_upside_rows),
            "frames_with_upside_down_detail": frames_with_upside_rows,
            "upside_down_by_direction": upside_by_direction_rows,
            "top_upside_down_examples": [
                {
                    "image": p.get("image"),
                    "direction_name": DIRECTION_NAMES.get(p.get("direction"), "?"),
                    "pose_confidence": round(float(p.get("pose_confidence", 0.0)), 4),
                }
                for p in upside_sorted[:15]
            ],
        },
    }


def render_text(report: dict) -> str:
    t = report["totals"]
    p = report["pose"]
    L = []
    bar = "=" * 60

    L += [bar, "FISH DIRECTION / POSE SUMMARY", bar,
          f"Total crops classified : {t['crops']}",
          f"  No fish              : {t['no_fish']}  ({t['percent_no_fish']}%)",
          f"  Fish                 : {t['fish']}",
          f"Mean direction conf.   : {t['mean_direction_confidence_fish']}",
          ""]

    L += [bar, "DIRECTION DISTRIBUTION (fish crops)", bar,
          f"{'dir':<5}{'count':>8}{'% fish':>10}{'mean conf':>12}"]
    for row in report["direction_distribution"]:
        L.append(f"{row['name']:<5}{row['count']:>8}{row['percent_of_fish']:>10}"
                 f"{str(row['mean_direction_confidence']):>12}")
    L.append("")

    L += [bar, "POSE / UPSIDE-DOWN", bar,
          f"Crops with a pose prediction : {p['posed_crops']}",
          f"  Regular (belly down)       : {p['regular']}",
          f"  Upside down (belly up)     : {p['upside_down']}  ({p['percent_upside_down']}%)",
          f"  Mean upside-down conf.     : {p['mean_upside_down_confidence']}",
          "",
          f"Upside-down is confirmed at pose confidence >= {p['confidence_threshold']}.",
          f"  Pose head flagged {p['pose_head_upside_down']} crops upside-down; "
          f"{p['upside_down_below_threshold']} were below the threshold and are "
          f"counted as regular.",
          ""]

    if p["upside_confidence_histogram"]:
        L += ["Upside-down confidence (all pose-head upside-down calls):"]
        for row in p["upside_confidence_histogram"]:
            mark = "  <- confirmed" if row["above_threshold"] else ""
            L.append(f"  {row['range']:<12}{row['count']:>6}{mark}")
        L.append("")

    if p["frames_with_upside_down"]:
        L += [f"Confirmed upside-down fish appear in {p['frames_with_upside_down']} "
              f"distinct frame(s):"]
        detail = p["frames_with_upside_down_detail"]
        for row in detail[:20]:
            extra = f"  ({row['upside_down_fish']} fish)" if row["upside_down_fish"] > 1 else ""
            L.append(f"  frame {row['frame_id']:06d}{extra}")
        if len(detail) > 20:
            L.append(f"  ... and {len(detail) - 20} more")
        L.append("")

    if p["upside_down_by_direction"]:
        L += ["Upside-down rate by direction:",
              f"  {'dir':<5}{'up':>6}{'posed':>8}{'% up':>9}"]
        for row in p["upside_down_by_direction"]:
            L.append(f"  {row['name']:<5}{row['upside_down']:>6}{row['posed_total']:>8}"
                     f"{row['percent_upside_down']:>9}")
        L.append("")

    if p["top_upside_down_examples"]:
        L += ["Highest-confidence upside-down crops:"]
        for ex in p["top_upside_down_examples"]:
            L.append(f"  {ex['image']:<32} {ex['direction_name']:<4} conf={ex['pose_confidence']}")
        L.append("")

    return "\n".join(L)


def main(predictions_dir: Path, out_json: Path, out_txt: Path, charts_dir: Path):
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions folder not found: {predictions_dir}")

    preds = load_predictions(predictions_dir)
    if not preds:
        raise RuntimeError(f"No prediction JSON files in {predictions_dir}")

    config.emit_step("computing distribution + upside-down stats")
    report = summarize(preds)
    text = render_text(report)

    config.emit_step("writing summary.json / summary.txt")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with out_txt.open("w", encoding="utf-8") as fh:
        fh.write(text + "\n")

    print(text)
    print(f"\nWrote {out_json}\nWrote {out_txt}")

    config.emit_step("rendering pie charts")
    from charts import build_charts
    for chart_path in build_charts(report, charts_dir):
        print(f"Wrote {chart_path}")
        config.emit_chart(chart_path)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", type=Path, default=config.PREDICTIONS_DIR)
    parser.add_argument("--out-json", type=Path, default=config.SUMMARY_JSON)
    parser.add_argument("--out-txt", type=Path, default=config.SUMMARY_TXT)
    parser.add_argument("--charts-dir", type=Path, default=config.CHARTS_DIR)
    args = parser.parse_args()

    main(args.predictions, args.out_json, args.out_txt, args.charts_dir)
