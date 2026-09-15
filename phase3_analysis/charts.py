#!/usr/bin/env python3
"""
Render the phase-3 summary as pie charts (PNG, matplotlib Agg backend).

`build_charts(report, out_dir)` takes the dict produced by
summarize_predictions.summarize() and returns the list of PNG paths it wrote.
If matplotlib is not installed it prints a note and returns [].
"""

from pathlib import Path

# Fixed colours so a direction keeps the same colour across every chart.
DIRECTION_COLOR = {
    "N":  "#4e79a7",
    "NE": "#59a14f",
    "E":  "#f28e2b",
    "SE": "#e15759",
    "S":  "#b07aa1",
    "SW": "#76b7b2",
    "W":  "#edc948",
    "NW": "#ff9da7",
}


def _fig_pie(plt, labels, values, colors, title, path, subtitle=None):
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=140)
    wedges, _texts, autotexts = ax.pie(
        values, labels=labels, colors=colors, autopct="%1.0f%%",
        pctdistance=0.78, startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5),
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_color("#222")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    if subtitle:
        ax.text(0, -1.32, subtitle, ha="center", fontsize=8, color="#666")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_charts(report: dict, out_dir) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as err:  # noqa: BLE001
        print(f"charts skipped (matplotlib unavailable: {err})")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    totals = report["totals"]
    pose = report["pose"]

    # 1. Fish vs No Fish -----------------------------------------------------
    if totals["crops"]:
        paths.append(_fig_pie(
            plt,
            ["Fish", "No fish"],
            [totals["fish"], totals["no_fish"]],
            ["#4e79a7", "#bab0ac"],
            "Fish vs No Fish",
            out_dir / "01_fish_vs_nofish.png",
            subtitle=f'{totals["crops"]} crops classified',
        ))

    # 2. Direction distribution -------------------------------------------
    dd = [(r["name"], r["count"]) for r in report["direction_distribution"] if r["count"]]
    if dd:
        labels = [n for n, _ in dd]
        paths.append(_fig_pie(
            plt,
            labels,
            [c for _, c in dd],
            [DIRECTION_COLOR.get(n, "#999999") for n in labels],
            "Swimming direction",
            out_dir / "02_direction_distribution.png",
            subtitle=f'{totals["fish"]} fish crops',
        ))

    # 3. Pose: regular vs upside down -----------------------------------
    if pose["posed_crops"]:
        thr = pose.get("confidence_threshold")
        sub = (f'{pose["posed_crops"]} crops with a pose · '
               f'{pose["percent_upside_down"]}% upside down')
        if thr is not None:
            sub += f' (pose conf >= {thr})'
        paths.append(_fig_pie(
            plt,
            ["Regular", "Upside down"],
            [pose["regular"], pose["upside_down"]],
            ["#59a14f", "#e15759"],
            "Pose",
            out_dir / "03_pose.png",
            subtitle=sub,
        ))

    # 4. Where the upside-down fish are ---------------------------------
    ud = [(r["name"], r["upside_down"]) for r in pose["upside_down_by_direction"]
          if r["upside_down"]]
    if ud:
        labels = [n for n, _ in ud]
        paths.append(_fig_pie(
            plt,
            labels,
            [c for _, c in ud],
            [DIRECTION_COLOR.get(n, "#999999") for n in labels],
            "Upside-down fish by direction",
            out_dir / "04_upside_down_by_direction.png",
            subtitle=f'{pose["upside_down"]} upside-down crops',
        ))

    return paths
