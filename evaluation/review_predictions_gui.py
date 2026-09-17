#!/usr/bin/env python3
"""
Manual review GUI for scoring a full pipeline run (phases 0-2) on an unseen
video, where there is no ground truth other than a human's eyes.

Two independent stages can each throw away a real fish, and this GUI reviews
both of them together so the reported accuracy is for the *whole* pipeline,
not just the classifier:

    1. The phase-1 median-background filter (dataset_ops/background_match_
       filter.py) discards a box before it ever reaches the classifier if it
       looks too much like the static scene. Those crops sit in
       rois_ignored_background/ with no prediction JSON at all.
    2. The classifier's own No Fish class (phase2_direction_pose_inference).

The fish/no-fish decision reproduces the actual pipeline logic in
phase3_analysis/summarize_predictions.py: a crop is "fish" whenever the
direction head did not call No Fish -- there is no direction-confidence
threshold in the real pipeline. (The pose head does have a real confidence
gate, config.UPSIDE_DOWN_CONF_THRESHOLD, applied separately below.)

Review controls:
    If the raw model prediction is a fish direction:
        V = Fish is present, predicted direction is correct
        A = Fish is present, predicted direction is almost correct,
            exactly one 45-degree neighboring direction away
        X = Fish is present, predicted direction is wrong
        P = Direction is correct, but the pose call (Regular/Upside Down)
            is WRONG -- flags a pose mistake without touching direction
        D = Pose is correct, but the direction call is WRONG -- the
            reverse mismatch
            (P/D are only enabled where pose is actually defined: a real
            fish, not N/S/No Fish; V/A/X still record direction as usual
            in the mismatch case, so accuracy stays consistent)

    If the raw model prediction is No Fish, or the crop was discarded by the
    background filter before ever reaching the classifier:
        V = Correct, the crop really contains no fish (classifier only)
        X = Wrong, there really is a fish in the crop (classifier only)
        A, P, D = Disabled: no direction/pose calls exist to be almost-
            right or mismatched against
        F/N = the only options for a background-filter-rejected crop, since
              no model prediction exists to score V/X against

    F = Fish is present, presence-only review, do not score direction
    N = No Fish is actually present in the crop
    U = Undo last review made in this session

Image order:
    By default, each new image is randomly selected from the remaining
    unreviewed images, drawn from both rois/ (classified) and
    rois_ignored_background/ (discarded pre-classification), pooled
    together. Images are sampled without replacement, so an image is not
    shown again after it has been reviewed unless you undo that review.

    --split-dir DIR reviews from a dataset_ops/sort_crops_by_class.py output
    folder instead (e.g. sorted_by_class/, or sorted_by_class_0.85/). Each
    per-class sub-folder becomes its own sampling class, and every pick
    chooses a random CLASS first, then a random image within it -- not a
    random image from the pooled whole. Without this, a small review
    session is almost entirely "regular" fish crops (the majority class);
    with it, No Fish / Upside Down / background-rejected crops get sampled
    just as often, so a handful of reviewed photos actually covers every
    class instead of mostly one.

        python review_predictions_gui.py --split-dir             # this run's own sorted_by_class/
        python review_predictions_gui.py --split-dir sorted_by_class_0.85

Choosing the run:
    --run RUN_DIR points at the folder holding rois/, predictions/ and
    boxes.csv; reviews/ and review_summary.json are written inside it. It
    defaults to the config.py data root, so setting FISH_PIPELINE_DATA works
    just as well.

        python review_predictions_gui.py --run data/runs/clip_20260917-160002

Why both direction and presence reviews exist:
    V/A/X measure direction quality only when the model produced a fish
    direction and a real fish is present. For a raw No Fish prediction (or a
    background-rejected crop), V/X are interpreted as presence correctness
    instead. F/N always provide an explicit presence-only label. These
    labels measure whether the pipeline, end to end, correctly keeps fish
    and rejects non-fish crops.

The neural network is NOT loaded here.
This GUI only reads previously saved JSON predictions (and, for background-
rejected crops, boxes.csv).
"""

import os
import json
import math
import sys
import random
import argparse
from pathlib import Path

import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config


# ============================================================
# Paths (defaults from config.py)
# ============================================================
#
# Everything here hangs off one run folder. main() rebinds the whole block
# from --run before any of it is read, so the flag and the config defaults are
# interchangeable throughout.

IMAGES_DIR = config.ROIS_DIR
PREDICTIONS_DIR = config.PREDICTIONS_DIR
RUN_DIR = config.PREDICTIONS_DIR.parent
REVIEWS_DIR = RUN_DIR / "reviews"
SUMMARY_PATH = RUN_DIR / "review_summary.json"

# Crops the phase-1 median-background filter discarded *before* they ever
# reached the classifier (see dataset_ops/background_match_filter.py). They
# have no prediction JSON at all. Scoring a full pipeline run on an unseen
# video means reviewing these too, otherwise a fish the background filter
# wrongly threw away is invisible to every metric below.
BACKGROUND_REJECTED_DIR = RUN_DIR / "rois_ignored_background"

# crop name -> match_pct, read from boxes.csv (phase 1 output) when present.
BOXES_CSV = RUN_DIR / "boxes.csv"

# Default --split-dir target: dataset_ops/sort_crops_by_class.py's output.
DEFAULT_SPLIT_DIR = RUN_DIR / "sorted_by_class"


def set_run_dir(run_dir):
    """
    Point every path above at ``run_dir`` (a data root, or one timestamped
    folder under runs/). This is what --run does.
    """
    global RUN_DIR, IMAGES_DIR, PREDICTIONS_DIR, REVIEWS_DIR
    global SUMMARY_PATH, BACKGROUND_REJECTED_DIR, BOXES_CSV, DEFAULT_SPLIT_DIR

    RUN_DIR = Path(run_dir)
    IMAGES_DIR = RUN_DIR / "rois"
    PREDICTIONS_DIR = RUN_DIR / "predictions"
    REVIEWS_DIR = RUN_DIR / "reviews"
    SUMMARY_PATH = RUN_DIR / "review_summary.json"
    BACKGROUND_REJECTED_DIR = RUN_DIR / "rois_ignored_background"
    BOXES_CSV = RUN_DIR / "boxes.csv"
    DEFAULT_SPLIT_DIR = RUN_DIR / "sorted_by_class"


# ============================================================
# Configuration
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

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
}

COMPASS_DIRS = [
    (1, "N", -90),
    (2, "NE", -45),
    (3, "E", 0),
    (4, "SE", 45),
    (5, "S", 90),
    (6, "SW", 135),
    (7, "W", 180),
    (8, "NW", -135),
]

# Display sizes
IMG_W = 720
IMG_H = 620
RIGHT_W = 430
COMPASS_W = 280
COMPASS_H = 250

BG = "#1e1e2e"
PANEL_BG = "#0d0d1a"
TEXT = "#dddddd"
MUTED = "#aaaaaa"
GOLD = "#FFD700"
GREEN = "#7ee787"
RED = "#ff7b72"
ORANGE = "#f2cc60"
BLUE = "#88aaff"


# ============================================================
# Helpers
# ============================================================

def safe_percent(value):
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def derive_analysis_fields(prediction):
    """
    Reproduce the actual pipeline analyze logic (phase3_analysis/
    summarize_predictions.py: ``fish = direction not in (None, NO_FISH)``).

    There is NO direction-confidence threshold in the real pipeline -- a
    previous version of this GUI reconstructed a fictional 60% gate for
    JSONs that lacked an explicit ``use_for_analysis`` field, but no
    prediction JSON the pipeline actually writes has ever carried that
    field, so that fallback was silently applying a threshold the pipeline
    itself does not use. This function now matches phase 3 exactly.
    """
    if "use_for_analysis" in prediction:
        return {
            "use_for_analysis": bool(prediction["use_for_analysis"]),
            "analysis_status": prediction.get("analysis_status", "UNKNOWN"),
            "rejection_reason": prediction.get("rejection_reason"),
        }

    direction = prediction.get("direction")

    if direction is None:
        return {
            "use_for_analysis": False,
            "analysis_status": "NO_PREDICTION",
            "rejection_reason": "missing_direction",
        }

    if int(direction) == 0:
        return {
            "use_for_analysis": False,
            "analysis_status": "NO_FISH",
            "rejection_reason": "predicted_no_fish",
        }

    return {
        "use_for_analysis": True,
        "analysis_status": "ACCEPTED",
        "rejection_reason": None,
    }


def load_boxes_match_pct(boxes_csv):
    """crop filename -> background match_pct, from phase 1's boxes.csv."""
    if not boxes_csv.exists():
        return {}

    import csv

    match_pct = {}
    try:
        with open(boxes_csv, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("crop")
                pct = row.get("match_pct")
                if name and pct not in (None, ""):
                    try:
                        match_pct[name] = float(pct)
                    except ValueError:
                        pass
    except OSError:
        return {}

    return match_pct


def discover_leaf_classes(split_dir):
    """
    Walk a dataset_ops/sort_crops_by_class.py output tree (e.g. sorted_by_
    class/) and return {class_name: [image paths]} for every directory that
    directly holds images -- no_fish/, regular/NE/, upside_down/high_conf/E/,
    pose_not_applicable/S/, unclassified/, etc.

    sort_crops_by_class.py only ever writes images into the deepest
    applicable folder, so any directory containing an image is a leaf class;
    an intermediate folder such as regular/ holds only sub-folders, never
    images directly, and is skipped automatically.
    """
    classes = {}
    for dirpath, _dirnames, filenames in os.walk(split_dir):
        images = sorted(
            Path(dirpath) / name
            for name in filenames
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            continue
        rel = Path(dirpath).relative_to(split_dir)
        class_name = "." if str(rel) == "." else rel.as_posix()
        classes[class_name] = images
    return classes


def derive_pose_gate(prediction):
    """
    Mirror the 0.85 upside-down confidence gate from
    phase3_analysis/summarize_predictions.py so the reviewer sees the same
    "final" pose call that ends up in summary.json, not just the raw model
    output.

    A raw "UpsideDown" call below config.UPSIDE_DOWN_CONF_THRESHOLD is folded
    into "Regular" for the gated result. Pose is undefined for the direction
    classes in config.POSE_NA_DIRECTIONS (N, S, No Fish), matching phase 3.
    """
    threshold = config.UPSIDE_DOWN_CONF_THRESHOLD

    direction = int(prediction.get("direction", 0))
    pose = prediction.get("pose")
    pose_confidence = prediction.get("pose_confidence")

    pose_na = (
        direction in config.POSE_NA_DIRECTIONS
        or pose is None
    )

    if pose_na:
        return {
            "threshold": threshold,
            "pose_na": True,
            "gated_pose": None,
            "gated_pose_name": "N/A",
            "gate_downgraded": False,
        }

    is_upside_down = int(pose) == 1
    confidence = float(pose_confidence) if pose_confidence is not None else 0.0

    if is_upside_down and confidence < threshold:
        # Raw call was UpsideDown but too weak to trust; folded to Regular.
        return {
            "threshold": threshold,
            "pose_na": False,
            "gated_pose": 0,
            "gated_pose_name": "Regular",
            "gate_downgraded": True,
        }

    return {
        "threshold": threshold,
        "pose_na": False,
        "gated_pose": pose,
        "gated_pose_name": prediction.get("pose_name"),
        "gate_downgraded": False,
    }


def presence_outcome(human_presence, use_for_analysis):
    """
    Treat use_for_analysis=True as 'pipeline says usable fish'.

    Returns the standard confusion-matrix label:
        TP = real fish kept
        TN = no-fish rejected
        FP = no-fish kept
        FN = real fish rejected
    """
    if human_presence == "fish":
        return "TP" if use_for_analysis else "FN"
    if human_presence == "no_fish":
        return "FP" if use_for_analysis else "TN"
    return None


# ============================================================
# GUI
# ============================================================

class InferenceReviewer:

    def __init__(self, root, split_dir=None):
        self.root = root
        self.root.title("Fish ViT Prediction Review")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

        self.match_pct = load_boxes_match_pct(BOXES_CSV)

        if BACKGROUND_REJECTED_DIR.exists():
            background_rejected = sorted(
                p
                for p in BACKGROUND_REJECTED_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            background_rejected = []

        self.background_rejected_stems = {
            p.stem for p in background_rejected
        }

        # self.classes: {class_name: [image paths]}. In the default (flat)
        # mode there is one pseudo-class holding every crop, so sampling is
        # uniform over the whole pool, same as before. With --split-dir, each
        # per-class folder from dataset_ops/sort_crops_by_class.py becomes
        # its own class, so _load_next samples a random CLASS first and then
        # a random image within it -- rare classes (No Fish, Upside Down)
        # get the same attention as the majority classes instead of being
        # drowned out, which is the whole point of reviewing a small number
        # of crops and still seeing real variation.
        self.stratified = split_dir is not None

        if self.stratified:
            if not split_dir.is_dir():
                messagebox.showerror(
                    "Split folder not found",
                    f"--split-dir does not exist:\n{split_dir}\n\n"
                    "Run dataset_ops/sort_crops_by_class.py on this run "
                    "first.",
                )
                self.root.destroy()
                return

            self.classes = discover_leaf_classes(split_dir)
        else:
            classified = sorted(
                p
                for p in IMAGES_DIR.iterdir()
                if (
                    p.is_file()
                    and p.suffix.lower() in IMAGE_EXTENSIONS
                    and (PREDICTIONS_DIR / f"{p.stem}.json").exists()
                )
            )
            # Stems are unique across the run: a crop lives in exactly one
            # of rois/ or rois_ignored_background/, never both.
            self.classes = {"all": classified + background_rejected}

        if background_rejected:
            # Reviewing the background filter's own mistakes belongs in
            # every mode -- fold it in as one more class so it gets sampled
            # too instead of being invisible whenever a split folder (which
            # sort_crops_by_class.py never sees these crops through) is used.
            if self.stratified:
                self.classes["background_rejected"] = background_rejected
            # (non-stratified mode already included it in "all" above)

        self.image_class = {
            p.stem: class_name
            for class_name, paths in self.classes.items()
            for p in paths
        }

        self.all_images = [
            p
            for paths in self.classes.values()
            for p in paths
        ]

        self.reviewed_set = {
            p.stem
            for p in REVIEWS_DIR.glob("*.json")
        }

        self.remaining_by_class = {
            class_name: [p for p in paths if p.stem not in self.reviewed_set]
            for class_name, paths in self.classes.items()
        }

        self.remaining_images = [
            p
            for p in self.all_images
            if p.stem not in self.reviewed_set
        ]

        self.current_image = None
        self.current_prediction = None
        self.current_gate = None
        self.current_pose_gate = None
        self.is_background_rejected = False

        # Undo is intentionally one step, same behavior as the original GUI.
        self.last_review = None
        self._photo = None

        self._build_ui()

        if not self.all_images:
            messagebox.showerror(
                "No crops",
                "No classified crops (with prediction JSONs) and no "
                "background-rejected crops found for this run.",
            )
            self.root.destroy()
            return

        if not self.remaining_images:
            self._finished()
            return

        self._load_next()

    # ========================================================
    # UI
    # ========================================================

    def _build_ui(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(padx=10, pady=10)

        self.image_canvas = tk.Canvas(
            main,
            width=IMG_W,
            height=IMG_H,
            bg=PANEL_BG,
            highlightthickness=1,
            highlightbackground="#3a3a5c",
        )
        self.image_canvas.pack(side=tk.LEFT, padx=(0, 12))

        right = tk.Frame(main, width=RIGHT_W, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.Y)
        right.pack_propagate(False)

        tk.Label(
            right,
            text="MODEL PREDICTION",
            font=("Arial", 15, "bold"),
            fg="white",
            bg=BG,
        ).pack(pady=(2, 5))

        # ------------------------- Compass -------------------------
        self.compass = tk.Canvas(
            right,
            width=COMPASS_W,
            height=COMPASS_H,
            bg=PANEL_BG,
            highlightthickness=1,
            highlightbackground="#3a3a5c",
        )
        self.compass.pack()

        # ---------------------- Prediction info --------------------
        info = tk.Frame(right, bg=BG)
        info.pack(fill=tk.X, pady=(5, 2))

        self.direction_label = tk.Label(
            info,
            text="Raw direction: --",
            font=("Arial", 14, "bold"),
            fg=GOLD,
            bg=BG,
        )
        self.direction_label.pack(fill=tk.X)

        self.direction_conf_label = tk.Label(
            info,
            text="Direction confidence: --",
            font=("Arial", 10),
            fg=MUTED,
            bg=BG,
        )
        self.direction_conf_label.pack(fill=tk.X)

        self.no_fish_conf_label = tk.Label(
            info,
            text="No Fish confidence: --",
            font=("Arial", 10),
            fg=MUTED,
            bg=BG,
        )
        self.no_fish_conf_label.pack(fill=tk.X)

        self.gate_label = tk.Label(
            info,
            text="Pipeline gate: --",
            font=("Arial", 12, "bold"),
            fg=MUTED,
            bg=BG,
        )
        self.gate_label.pack(fill=tk.X, pady=(4, 0))

        self.gate_detail_label = tk.Label(
            info,
            text="",
            font=("Arial", 9),
            fg=MUTED,
            bg=BG,
            wraplength=RIGHT_W - 20,
        )
        self.gate_detail_label.pack(fill=tk.X)

        self.pose_label = tk.Label(
            info,
            text="Raw pose: --",
            font=("Arial", 11, "bold"),
            fg=BLUE,
            bg=BG,
        )
        self.pose_label.pack(fill=tk.X, pady=(4, 0))

        self.pose_conf_label = tk.Label(
            info,
            text="Pose confidence: --",
            font=("Arial", 9),
            fg=MUTED,
            bg=BG,
        )
        self.pose_conf_label.pack(fill=tk.X)

        # ------------------------- Buttons -------------------------
        tk.Label(
            right,
            text="HUMAN REVIEW",
            font=("Arial", 11, "bold"),
            fg="white",
            bg=BG,
        ).pack(pady=(7, 3))

        direction_buttons = tk.Frame(right, bg=BG)
        direction_buttons.pack(fill=tk.X)

        self.v_button = self._make_button(
            direction_buttons,
            "V  Correct",
            lambda: self._review("V"),
            GREEN,
        )
        self.v_button.pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3)
        )

        self.a_button = self._make_button(
            direction_buttons,
            "A  Almost",
            lambda: self._review("A"),
            ORANGE,
        )
        self.a_button.pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=3
        )

        self.x_button = self._make_button(
            direction_buttons,
            "X  Wrong",
            lambda: self._review("X"),
            RED,
        )
        self.x_button.pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0)
        )

        mixed_buttons = tk.Frame(right, bg=BG)
        mixed_buttons.pack(fill=tk.X, pady=(3, 0))

        self.wrong_pose_button = self._make_button(
            mixed_buttons,
            "P  Wrong Pose",
            lambda: self._review("WRONG_POSE"),
            ORANGE,
        )
        self.wrong_pose_button.pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3)
        )

        self.wrong_dir_button = self._make_button(
            mixed_buttons,
            "D  Wrong Dir",
            lambda: self._review("WRONG_DIR"),
            ORANGE,
        )
        self.wrong_dir_button.pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0)
        )

        presence_buttons = tk.Frame(right, bg=BG)
        presence_buttons.pack(fill=tk.X, pady=(5, 0))

        self._make_button(
            presence_buttons,
            "F  Fish",
            lambda: self._review("FISH"),
            BLUE,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3))

        self._make_button(
            presence_buttons,
            "N  No Fish",
            lambda: self._review("NO_FISH"),
            GOLD,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0))

        self._make_button(
            right,
            "U  Undo last review",
            self._undo,
            "#bbbbbb",
        ).pack(fill=tk.X, pady=(5, 0))

        self.review_help_label = tk.Label(
            right,
            text=(
                "V/A/X = fish present and direction scored\n"
                "F = fish present, presence only\n"
                "N = no fish in the crop"
            ),
            font=("Arial", 8),
            fg=MUTED,
            bg=BG,
            justify=tk.LEFT,
            wraplength=RIGHT_W - 10,
        )
        self.review_help_label.pack(anchor="w", pady=(4, 3))

        # -------------------------- Stats --------------------------
        self.stats_label = tk.Label(
            right,
            text="",
            font=("Courier", 9),
            fg=TEXT,
            bg=BG,
            justify=tk.LEFT,
        )
        self.stats_label.pack(anchor="w", pady=(2, 2))

        self.class_label = tk.Label(
            right,
            text="",
            font=("Arial", 9, "bold"),
            fg=BLUE,
            bg=BG,
            wraplength=RIGHT_W - 10,
        )
        self.class_label.pack(anchor="w", pady=(4, 0))

        self.file_label = tk.Label(
            right,
            text="",
            font=("Arial", 8),
            fg="#777777",
            bg=BG,
            wraplength=RIGHT_W - 10,
        )
        self.file_label.pack(anchor="w", pady=(2, 1))

        self.progress_label = tk.Label(
            right,
            text="",
            font=("Arial", 9),
            fg=MUTED,
            bg=BG,
        )
        self.progress_label.pack(anchor="w")

        self.root.bind("<KeyPress>", self._on_key)
        self.root.focus_set()

    def _make_button(self, parent, text, command, fg):
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Arial", 10, "bold"),
            fg=fg,
            bg="#2b2b40",
            activebackground="#3b3b55",
            activeforeground=fg,
            relief=tk.RAISED,
            bd=1,
            padx=5,
            pady=5,
            cursor="hand2",
        )

    def _update_review_controls(self, predicted_direction):
        """Update V/A/X semantics for the current raw prediction."""
        # P (Wrong Pose) and D (Wrong Dir) only make sense when pose is
        # actually defined for this crop: a real fish, not N/S/No Fish, and
        # not a background-rejected crop with no prediction at all.
        pose_applicable = (
            predicted_direction is not None
            and predicted_direction != 0
            and not self.current_pose_gate["pose_na"]
        )
        pose_state = tk.NORMAL if pose_applicable else tk.DISABLED
        self.wrong_pose_button.config(state=pose_state)
        self.wrong_dir_button.config(state=pose_state)

        if predicted_direction is None:
            # Background-rejected: no model prediction exists at all, so
            # there is nothing for V/A/X to score against. Only F/N apply.
            self.v_button.config(text="V  (n/a)", state=tk.DISABLED)
            self.x_button.config(text="X  (n/a)", state=tk.DISABLED)
            self.a_button.config(state=tk.DISABLED)
            self.review_help_label.config(
                text=(
                    "Discarded by the background filter BEFORE "
                    "classification:\n"
                    "F = actually a fish (background filter mistake)\n"
                    "N = correctly not a fish"
                )
            )
            return

        self.v_button.config(state=tk.NORMAL)
        self.x_button.config(state=tk.NORMAL)

        if predicted_direction == 0:
            self.v_button.config(text="V  Correct No Fish")
            self.x_button.config(text="X  Wrong, Fish")
            self.a_button.config(state=tk.DISABLED)
            self.review_help_label.config(
                text=(
                    "Raw prediction is NO FISH:\n"
                    "V = correct, actually no fish\n"
                    "X = wrong, actually a fish\n"
                    "A = not applicable\n"
                    "P/D = not applicable, no pose exists\n"
                    "F/N = explicit presence-only labels"
                )
            )
        else:
            self.v_button.config(text="V  Correct")
            self.x_button.config(text="X  Wrong")
            self.a_button.config(state=tk.NORMAL)
            self.review_help_label.config(
                text=(
                    "Raw prediction is a fish direction:\n"
                    "V/A/X = fish present and direction scored\n"
                    "P = direction correct, pose is WRONG\n"
                    "D = pose correct, direction is WRONG\n"
                    "F = fish present, presence only\n"
                    "N = no fish in the crop"
                )
            )

    # ========================================================
    # Image
    # ========================================================

    def _folder_label(self, path):
        """
        The actual on-disk folder this image file is sitting in, relative
        to the run folder -- e.g. "rois/", "rois_ignored_background/", or
        (in --split-dir mode) "sorted_by_class_0.85/upside_down/high_conf/E/".
        This is the literal source folder, independent of any bookkeeping.
        """
        try:
            rel = path.parent.relative_to(RUN_DIR)
            rel_str = "" if str(rel) == "." else f"{rel.as_posix()}/"
        except ValueError:
            rel_str = f"{path.parent.name}/"
        return rel_str or f"{path.parent.name}/"

    def _display_image(self, path, folder_label=None):
        self.image_canvas.delete("all")

        with Image.open(path) as source:
            img = source.convert("RGB")

        iw, ih = img.size
        scale = min(IMG_W / iw, IMG_H / ih)

        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))

        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)

        self.image_canvas.create_image(
            IMG_W // 2,
            IMG_H // 2,
            image=self._photo,
            anchor=tk.CENTER,
        )

        if folder_label:
            # A readable banner drawn directly on the picture, so the
            # source folder travels with the image itself rather than
            # sitting only in a side-panel label.
            pad_x, pad_y = 8, 5
            text_id = self.image_canvas.create_text(
                14,
                10,
                text=folder_label,
                anchor="nw",
                fill="white",
                font=("Arial", 11, "bold"),
            )
            x0, y0, x1, y1 = self.image_canvas.bbox(text_id)
            rect_id = self.image_canvas.create_rectangle(
                x0 - pad_x,
                y0 - pad_y,
                x1 + pad_x,
                y1 + pad_y,
                fill="#000000",
                outline="",
            )
            self.image_canvas.tag_lower(rect_id, text_id)

    # ========================================================
    # Compass
    # ========================================================

    def _draw_compass(self, predicted_direction):
        c = self.compass
        c.delete("all")

        cx = COMPASS_W // 2
        cy = 115
        r = 78

        c.create_oval(
            cx - r,
            cy - r,
            cx + r,
            cy + r,
            outline="#2a3a5a",
            width=2,
        )

        for direction, name, angle_deg in COMPASS_DIRS:
            angle = math.radians(angle_deg)

            x1 = cx + 0.2 * r * math.cos(angle)
            y1 = cy + 0.2 * r * math.sin(angle)
            x2 = cx + 0.7 * r * math.cos(angle)
            y2 = cy + 0.7 * r * math.sin(angle)

            selected = direction == predicted_direction
            color = GOLD if selected else "#29415c"
            width = 4 if selected else 2

            c.create_line(
                x1,
                y1,
                x2,
                y2,
                arrow=tk.LAST,
                width=width,
                fill=color,
            )

            lx = cx + (r + 19) * math.cos(angle)
            ly = cy + (r + 19) * math.sin(angle)

            c.create_text(
                lx,
                ly,
                text=name,
                fill=color,
                font=(
                    "Arial",
                    8,
                    "bold" if selected else "normal",
                ),
            )

        if predicted_direction is None:
            c.create_text(
                cx,
                cy,
                text="BACKGROUND\nREJECTED",
                fill=RED,
                font=("Arial", 13, "bold"),
                justify=tk.CENTER,
            )
        elif predicted_direction == 0:
            c.create_text(
                cx,
                cy,
                text="NO FISH",
                fill=GOLD,
                font=("Arial", 15, "bold"),
            )

    # ========================================================
    # Load next
    # ========================================================

    def _load_next(self, preferred_image=None):
        if not self.remaining_images:
            self._finished()
            return

        # Undo can pass a preferred image so the image being undone is
        # immediately shown again instead of jumping to another random one.
        if (
            preferred_image is not None
            and preferred_image in self.remaining_images
        ):
            self.current_image = preferred_image
        elif self.stratified:
            # Pick a random CLASS first (uniformly among classes that still
            # have unreviewed crops), then a random image within it -- not
            # a random image from the whole pool. This is what makes rare
            # classes show up just as often as common ones.
            available_classes = [
                name for name, imgs in self.remaining_by_class.items()
                if imgs
            ]
            class_name = random.choice(available_classes)
            self.current_image = random.choice(
                self.remaining_by_class[class_name]
            )
        else:
            self.current_image = random.choice(self.remaining_images)

        stem = self.current_image.stem
        self.is_background_rejected = stem in self.background_rejected_stems

        prediction_path = PREDICTIONS_DIR / f"{stem}.json"

        if self.is_background_rejected:
            # This crop never reached the classifier: the median-background
            # filter (phase 1) discarded it first. There is no model
            # prediction to show or to score V/A/X against.
            self.current_prediction = {"image": self.current_image.name}
        else:
            try:
                with open(prediction_path, "r", encoding="utf-8") as f:
                    self.current_prediction = json.load(f)
            except Exception as exc:
                messagebox.showerror(
                    "Prediction read error",
                    f"Could not read:\n{prediction_path}\n\n{exc}",
                )
                return

        if self.is_background_rejected:
            self.current_gate = {
                "use_for_analysis": False,
                "analysis_status": "BACKGROUND_REJECTED",
                "rejection_reason": "background_match_filter",
            }
            self.current_pose_gate = {
                "threshold": config.UPSIDE_DOWN_CONF_THRESHOLD,
                "pose_na": True,
                "gated_pose": None,
                "gated_pose_name": "N/A",
                "gate_downgraded": False,
            }
        else:
            self.current_gate = derive_analysis_fields(
                self.current_prediction
            )
            self.current_pose_gate = derive_pose_gate(
                self.current_prediction
            )

        self._display_image(
            self.current_image,
            folder_label=self._folder_label(self.current_image),
        )

        direction = (
            None
            if self.is_background_rejected
            else int(self.current_prediction.get("direction", 0))
        )

        self._draw_compass(direction)
        self._update_review_controls(direction)

        if self.is_background_rejected:
            match = self.match_pct.get(self.current_image.name)
            match_text = (
                f", background match {match:.1f}%" if match is not None
                else ""
            )
            self.direction_label.config(
                text="Raw direction: -- (rejected pre-classification)"
            )
            self.direction_conf_label.config(
                text=f"No model prediction exists for this crop{match_text}",
                fg=ORANGE,
            )
            self.no_fish_conf_label.config(text="No Fish confidence: N/A")
        else:
            direction_name = self.current_prediction.get(
                "direction_name",
                DIRECTION_NAMES.get(direction, str(direction)),
            )
            direction_confidence = float(
                self.current_prediction.get("direction_confidence", 0.0)
            )
            no_fish_confidence = self.current_prediction.get(
                "no_fish_confidence"
            )

            self.direction_label.config(
                text=f"Raw direction: {direction_name}"
            )
            self.direction_conf_label.config(
                text=(
                    f"Direction confidence: "
                    f"{direction_confidence * 100:.2f}%"
                ),
                fg=GREEN if direction != 0 else ORANGE,
            )
            self.no_fish_conf_label.config(
                text=(
                    "No Fish confidence: "
                    f"{safe_percent(no_fish_confidence)}"
                )
            )

        if self.current_gate["use_for_analysis"]:
            gate_text = "Pipeline gate: ACCEPT FISH"
            gate_color = GREEN
        else:
            gate_text = "Pipeline gate: REJECT"
            gate_color = RED

        self.gate_label.config(
            text=gate_text,
            fg=gate_color,
        )

        status = self.current_gate["analysis_status"]
        reason = self.current_gate["rejection_reason"]
        if reason:
            detail = f"Status: {status} | reason: {reason}"
        else:
            detail = f"Status: {status}"

        analysis_direction_name = self.current_prediction.get(
            "analysis_direction_name"
        )
        if analysis_direction_name is not None:
            detail += f" | final direction: {analysis_direction_name}"

        self.gate_detail_label.config(text=detail)

        pose_name = self.current_prediction.get("pose_name", "N/A")
        pose_confidence = self.current_prediction.get("pose_confidence")

        gated_pose_name = self.current_pose_gate["gated_pose_name"]
        threshold_pct = self.current_pose_gate["threshold"] * 100

        if self.current_pose_gate["gate_downgraded"]:
            gate_note = f" (< {threshold_pct:.0f}% gate -> downgraded to Regular)"
            gate_color = ORANGE
        elif self.current_pose_gate["pose_na"]:
            gate_note = ""
            gate_color = MUTED
        else:
            gate_note = f" (>= {threshold_pct:.0f}% gate)"
            gate_color = GREEN if gated_pose_name == "UpsideDown" else TEXT

        self.pose_label.config(
            text=f"Raw pose: {pose_name}  |  Gated pose: {gated_pose_name}{gate_note}",
            fg=gate_color,
        )
        self.pose_conf_label.config(
            text=f"Pose confidence: {safe_percent(pose_confidence)}"
        )

        if self.stratified:
            class_name = self.image_class.get(stem, "?")
            remaining_in_class = len(
                self.remaining_by_class.get(class_name, [])
            )
            total_in_class = len(self.classes.get(class_name, []))
            self.class_label.config(
                text=(
                    f"Class: {class_name}  "
                    f"({remaining_in_class}/{total_in_class} left)"
                )
            )
        else:
            self.class_label.config(text="")

        roi_tag = (
            "  [BACKGROUND-REJECTED]" if self.is_background_rejected else ""
        )
        self.file_label.config(
            text=f"ROI: {self.current_image.name}{roi_tag}"
        )

        self._refresh_stats()

    # ========================================================
    # Review
    # ========================================================

    def _review(self, result):
        if self.current_image is None:
            return

        if result not in {
            "V", "A", "X", "WRONG_POSE", "WRONG_DIR", "FISH", "NO_FISH",
        }:
            return

        # Background-rejected crops have no model prediction at all, so
        # only the presence-only buttons (F/N) mean anything.
        if self.is_background_rejected:
            if result not in {"FISH", "NO_FISH"}:
                return

            if result == "FISH":
                human_presence = "fish"
                review_name = (
                    "Fish present (background filter mistake)"
                )
            else:
                human_presence = "no_fish"
                review_name = "No Fish present (background filter correct)"

            use_for_analysis = False  # never reached the classifier
            outcome = presence_outcome(human_presence, use_for_analysis)

            review = {
                "review_semantics_version": 2,
                "image": self.current_image.name,
                "roi_name": self.current_image.name,
                "is_background_rejected": True,
                "background_match_pct": self.match_pct.get(
                    self.current_image.name
                ),

                "review": result,
                "review_name": review_name,
                "human_presence": human_presence,
                "direction_review": None,
                "raw_prediction_is_no_fish": None,

                "predicted_direction": None,
                "predicted_direction_name": None,
                "direction_confidence": None,
                "no_fish_confidence": None,
                "predicted_pose": None,
                "predicted_pose_name": None,
                "pose_confidence": None,

                "direction_confidence_threshold": None,
                "use_for_analysis": use_for_analysis,
                "analysis_status": self.current_gate["analysis_status"],
                "rejection_reason": self.current_gate["rejection_reason"],
                "analysis_direction": None,
                "analysis_direction_name": None,
                "pose_confidence_threshold": self.current_pose_gate[
                    "threshold"
                ],
                "pose_na": True,
                "pose_gate_downgraded": False,
                "analysis_pose": None,
                "analysis_pose_name": "N/A",
                "pose_review": "not_applicable",

                "pipeline_presence_prediction": "no_fish",
                "presence_outcome": outcome,
                "presence_correct": outcome in {"TP", "TN"},
            }

            review_path = REVIEWS_DIR / f"{self.current_image.stem}.json"
            with open(review_path, "w", encoding="utf-8") as f:
                json.dump(review, f, indent=2)

            self.last_review = (self.current_image, review_path)
            self.reviewed_set.add(self.current_image.stem)
            self.remaining_images.remove(self.current_image)
            self._remove_from_class(self.current_image)

            self._save_summary()
            self._load_next()
            return

        predicted_direction = int(
            self.current_prediction.get("direction", 0)
        )
        predicted_no_fish = predicted_direction == 0

        # P (Wrong Pose) / D (Wrong Dir) only make sense where pose is
        # actually defined -- a real fish, not N/S/No Fish.
        pose_applicable = (
            not predicted_no_fish
            and not self.current_pose_gate["pose_na"]
        )

        if result in {"WRONG_POSE", "WRONG_DIR"} and not pose_applicable:
            return

        if result in {"WRONG_POSE", "WRONG_DIR"}:
            pose_review = "wrong" if result == "WRONG_POSE" else "correct"
        elif pose_applicable:
            pose_review = "not_reviewed"
        else:
            pose_review = "not_applicable"

        # V/X change meaning when the raw prediction itself is No Fish.
        # A has no meaning in that case and is ignored.
        if predicted_no_fish:
            if result == "A":
                return
            if result == "V":
                human_presence = "no_fish"
                direction_review = None
                review_name = "Correct No Fish prediction"
            elif result == "X":
                human_presence = "fish"
                direction_review = None
                review_name = "Wrong No Fish prediction, fish is present"
            elif result == "FISH":
                human_presence = "fish"
                direction_review = None
                review_name = "Fish present, presence only"
            else:  # NO_FISH
                human_presence = "no_fish"
                direction_review = None
                review_name = "No Fish present"
        else:
            names = {
                "V": "Correct direction",
                "A": "Almost, 45-degree neighbor",
                "X": "Wrong direction",
                "WRONG_POSE": "Correct direction, wrong pose",
                "WRONG_DIR": "Wrong direction, correct pose",
                "FISH": "Fish present, presence only",
                "NO_FISH": "No Fish present",
            }
            review_name = names[result]

            if result in {"V", "A", "X", "WRONG_POSE", "WRONG_DIR", "FISH"}:
                human_presence = "fish"
            else:
                human_presence = "no_fish"

            direction_review = {
                "V": "V",
                "A": "A",
                "X": "X",
                "WRONG_POSE": "V",  # direction itself was correct
                "WRONG_DIR": "X",   # direction itself was wrong
            }.get(result)

        use_for_analysis = self.current_gate["use_for_analysis"]
        outcome = presence_outcome(
            human_presence,
            use_for_analysis,
        )

        review = {
            "review_semantics_version": 2,
            "image": self.current_image.name,
            "roi_name": self.current_image.name,
            "is_background_rejected": False,
            "background_match_pct": self.match_pct.get(
                self.current_image.name
            ),

            # Human review
            "review": result,
            "review_name": review_name,
            "human_presence": human_presence,
            "direction_review": direction_review,

            # Context needed to interpret V/X correctly later
            "raw_prediction_is_no_fish": predicted_no_fish,

            # Raw model output
            "predicted_direction": self.current_prediction.get("direction"),
            "predicted_direction_name": self.current_prediction.get(
                "direction_name"
            ),
            "direction_confidence": self.current_prediction.get(
                "direction_confidence"
            ),
            "no_fish_confidence": self.current_prediction.get(
                "no_fish_confidence"
            ),
            "predicted_pose": self.current_prediction.get("pose"),
            "predicted_pose_name": self.current_prediction.get("pose_name"),
            "pose_confidence": self.current_prediction.get("pose_confidence"),

            # Final analysis gate (no direction-confidence threshold in the
            # real pipeline; see derive_analysis_fields)
            "direction_confidence_threshold": None,
            "use_for_analysis": use_for_analysis,
            "analysis_status": self.current_gate["analysis_status"],
            "rejection_reason": self.current_gate["rejection_reason"],
            "analysis_direction": self.current_prediction.get(
                "analysis_direction"
            ),
            "analysis_direction_name": self.current_prediction.get(
                "analysis_direction_name"
            ),
            "pose_confidence_threshold": self.current_pose_gate["threshold"],
            "pose_na": self.current_pose_gate["pose_na"],
            "pose_gate_downgraded": self.current_pose_gate["gate_downgraded"],
            "analysis_pose": self.current_pose_gate["gated_pose"],
            "analysis_pose_name": self.current_pose_gate["gated_pose_name"],
            "pose_review": pose_review,

            # Presence-filter evaluation
            "pipeline_presence_prediction": (
                "fish" if use_for_analysis else "no_fish"
            ),
            "presence_outcome": outcome,
            "presence_correct": outcome in {"TP", "TN"},
        }

        review_path = (
            REVIEWS_DIR
            / f"{self.current_image.stem}.json"
        )

        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(review, f, indent=2)

        self.last_review = (
            self.current_image,
            review_path,
        )

        self.reviewed_set.add(self.current_image.stem)
        self.remaining_images.remove(self.current_image)
        self._remove_from_class(self.current_image)

        self._save_summary()
        self._load_next()

    def _remove_from_class(self, image_path):
        class_name = self.image_class.get(image_path.stem)
        bucket = self.remaining_by_class.get(class_name)
        if bucket and image_path in bucket:
            bucket.remove(image_path)

    def _restore_to_class(self, image_path):
        class_name = self.image_class.get(image_path.stem)
        bucket = self.remaining_by_class.get(class_name)
        if bucket is not None and image_path not in bucket:
            bucket.append(image_path)

    # ========================================================
    # Undo
    # ========================================================

    def _undo(self):
        if self.last_review is None:
            return

        image_path, review_path = self.last_review

        if review_path.exists():
            review_path.unlink()

        self.reviewed_set.discard(image_path.stem)

        if image_path not in self.remaining_images:
            self.remaining_images.append(image_path)
        self._restore_to_class(image_path)

        self.last_review = None

        self._save_summary()
        self._load_next(preferred_image=image_path)

    # ========================================================
    # Keyboard
    # ========================================================

    def _on_key(self, event):
        # Avoid accidental duplicate actions from special keys.
        key = event.char.lower() if event.char else ""

        if key == "v":
            self._review("V")
        elif key == "a":
            if int(self.current_prediction.get("direction", 0)) != 0:
                self._review("A")
        elif key == "x":
            self._review("X")
        elif key == "p":
            self._review("WRONG_POSE")
        elif key == "d":
            self._review("WRONG_DIR")
        elif key == "f":
            self._review("FISH")
        elif key == "n":
            self._review("NO_FISH")
        elif key == "u":
            self._undo()

    # ========================================================
    # Stats
    # ========================================================

    def _get_stats(self):
        stats = {
            "total_review_files": 0,

            # Direction review, only when a real fish was shown and the
            # model produced a fish direction.
            "V": 0,
            "A": 0,
            "X": 0,
            "direction_reviewed": 0,

            # Actual button presses
            "button_V": 0,
            "button_A": 0,
            "button_X": 0,
            "FISH": 0,
            "NO_FISH": 0,

            # Human presence
            "human_fish": 0,
            "human_no_fish": 0,
            "presence_reviewed": 0,

            # Full-pipeline fish/no-fish confusion matrix (background filter
            # rejections + classifier No Fish class, combined).
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,

            # Same confusion matrix, split by which stage made the call, so
            # you can see whether errors come from the background filter or
            # from the classifier's No Fish class.
            "bg_reviewed": 0,
            "bg_TN": 0,   # background filter correctly discarded no-fish
            "bg_FN": 0,   # background filter wrongly discarded a real fish
            "clf_TP": 0,
            "clf_TN": 0,
            "clf_FP": 0,
            "clf_FN": 0,

            # P (Wrong Pose) / D (Wrong Dir): the reviewer flagging a
            # mismatch between the direction and pose calls -- direction
            # right but pose wrong, or the reverse. These are exception
            # flags on top of V/A/X, not a full independent pose review, so
            # they are reported as raw counts rather than an "accuracy"
            # that would only reflect the mismatches anyone bothered to
            # flag.
            "button_wrong_pose": 0,
            "button_wrong_dir": 0,

            # Legacy review bookkeeping
            "legacy_presence_unknown": 0,
            "invalid_almost_on_no_fish": 0,
        }

        for review_path in REVIEWS_DIR.glob("*.json"):
            try:
                with open(review_path, "r", encoding="utf-8") as f:
                    review = json.load(f)
            except Exception:
                continue

            stats["total_review_files"] += 1

            result = review.get("review")
            predicted_direction = review.get("predicted_direction")
            try:
                predicted_direction = int(predicted_direction)
            except (TypeError, ValueError):
                predicted_direction = None

            # Count the physical button press separately from direction stats.
            if result == "V":
                stats["button_V"] += 1
            elif result == "A":
                stats["button_A"] += 1
            elif result == "X":
                stats["button_X"] += 1
            elif result == "FISH":
                stats["FISH"] += 1
            elif result == "NO_FISH":
                stats["NO_FISH"] += 1
            elif result == "WRONG_POSE":
                stats["button_wrong_pose"] += 1
            elif result == "WRONG_DIR":
                stats["button_wrong_dir"] += 1

            # Direction scoring is valid only when the raw prediction was a
            # fish direction. This also retroactively fixes reviews created
            # by the previous GUI version where V on No Fish was misread as
            # a real fish direction review.
            direction_result = review.get("direction_review")
            if "direction_review" not in review:
                direction_result = (
                    result if result in {"V", "A", "X"} else None
                )

            if predicted_direction == 0:
                if result == "A":
                    stats["invalid_almost_on_no_fish"] += 1
                direction_result = None

            if direction_result in {"V", "A", "X"}:
                stats[direction_result] += 1
                stats["direction_reviewed"] += 1

            # Interpret human presence. Explicit F/N always win. For V/X on
            # a raw No Fish prediction, V means actual No Fish and X means
            # actual Fish. This intentionally overrides the buggy semantics
            # saved by the immediately previous GUI version.
            if result == "FISH":
                human_presence = "fish"
            elif result == "NO_FISH":
                human_presence = "no_fish"
            elif predicted_direction == 0:
                if result == "V":
                    human_presence = "no_fish"
                elif result == "X":
                    human_presence = "fish"
                else:
                    human_presence = None
            elif result in {"V", "A", "X"}:
                human_presence = "fish"
            else:
                human_presence = review.get("human_presence")

            # Legacy X with no prediction context is still ambiguous.
            if human_presence not in {"fish", "no_fish"}:
                if result == "X" and predicted_direction is None:
                    stats["legacy_presence_unknown"] += 1
                continue

            stats["presence_reviewed"] += 1
            if human_presence == "fish":
                stats["human_fish"] += 1
            else:
                stats["human_no_fish"] += 1

            # Prefer the value saved in the review. If this is a legacy
            # review, reconstruct the gate from the prediction JSON using
            # the real pipeline logic (no confidence threshold).
            is_bg = review.get("is_background_rejected", False)

            if "use_for_analysis" in review:
                use_for_analysis = bool(review["use_for_analysis"])
            else:
                prediction = None
                prediction_path = (
                    PREDICTIONS_DIR
                    / f"{review_path.stem}.json"
                )

                if prediction_path.exists():
                    try:
                        with open(
                            prediction_path,
                            "r",
                            encoding="utf-8",
                        ) as f:
                            prediction = json.load(f)
                    except Exception:
                        prediction = None

                if prediction is not None:
                    use_for_analysis = derive_analysis_fields(
                        prediction
                    )["use_for_analysis"]
                elif predicted_direction is not None:
                    use_for_analysis = predicted_direction != 0
                else:
                    # No prediction JSON and no direction on the review:
                    # this is a background-rejected crop, which the
                    # pipeline never accepts.
                    use_for_analysis = False

            outcome = presence_outcome(
                human_presence,
                use_for_analysis,
            )

            if outcome in {"TP", "TN", "FP", "FN"}:
                stats[outcome] += 1

            if is_bg:
                stats["bg_reviewed"] += 1
                if outcome == "TN":
                    stats["bg_TN"] += 1
                elif outcome == "FN":
                    stats["bg_FN"] += 1
            elif outcome in {"TP", "TN", "FP", "FN"}:
                stats[f"clf_{outcome}"] += 1

        return stats

    def _refresh_stats(self):
        stats = self._get_stats()

        total = len(self.all_images)
        reviewed = stats["total_review_files"]

        self.progress_label.config(
            text=f"Progress: {reviewed} / {total}"
        )

        direction_n = stats["direction_reviewed"]
        exact = safe_div(stats["V"], direction_n)
        within_45 = safe_div(
            stats["V"] + stats["A"],
            direction_n,
        )

        presence_n = stats["presence_reviewed"]
        presence_accuracy = safe_div(
            stats["TP"] + stats["TN"],
            presence_n,
        )
        fish_recall = safe_div(
            stats["TP"],
            stats["TP"] + stats["FN"],
        )
        accepted_precision = safe_div(
            stats["TP"],
            stats["TP"] + stats["FP"],
        )
        no_fish_rejection = safe_div(
            stats["TN"],
            stats["TN"] + stats["FP"],
        )

        text = (
            "DIRECTION, REAL FISH ONLY\n"
            f"V Correct : {stats['V']:5d}\n"
            f"A Almost  : {stats['A']:5d}\n"
            f"X Wrong   : {stats['X']:5d}\n"
            f"Exact     : {exact * 100:6.2f}%\n"
            f"V + A     : {within_45 * 100:6.2f}%\n"
            f"P Wrong Pose (dir OK)  : {stats['button_wrong_pose']:5d}\n"
            f"D Wrong Dir (pose OK)  : {stats['button_wrong_dir']:5d}\n"
            "\n"
            "FULL-PIPELINE FISH / NO-FISH GATE\n"
            f"TP Fish kept      : {stats['TP']:5d}\n"
            f"TN NoFish rejected: {stats['TN']:5d}\n"
            f"FP NoFish kept    : {stats['FP']:5d}\n"
            f"FN Fish rejected  : {stats['FN']:5d}\n"
            f"Gate accuracy     : {presence_accuracy * 100:6.2f}%\n"
            f"Fish recall       : {fish_recall * 100:6.2f}%\n"
            f"Accepted precision: {accepted_precision * 100:6.2f}%\n"
            f"NoFish rejection  : {no_fish_rejection * 100:6.2f}%\n"
            "\n"
            "BY STAGE (of the FN/TN above)\n"
            f"Bg filter reviewed: {stats['bg_reviewed']:5d}\n"
            f"  bg TN (correct) : {stats['bg_TN']:5d}\n"
            f"  bg FN (mistake) : {stats['bg_FN']:5d}\n"
            f"Classifier TP/TN/FP/FN: "
            f"{stats['clf_TP']}/{stats['clf_TN']}/"
            f"{stats['clf_FP']}/{stats['clf_FN']}"
        )

        if stats["legacy_presence_unknown"]:
            text += (
                "\n"
                f"Legacy X presence unknown: "
                f"{stats['legacy_presence_unknown']}"
            )

        self.stats_label.config(text=text)

    # ========================================================
    # Save summary
    # ========================================================

    def _save_summary(self):
        stats = self._get_stats()

        direction_n = stats["direction_reviewed"]
        presence_n = stats["presence_reviewed"]

        exact_accuracy = safe_div(stats["V"], direction_n)
        within_45_accuracy = safe_div(
            stats["V"] + stats["A"],
            direction_n,
        )
        wrong_rate = safe_div(stats["X"], direction_n)

        presence_accuracy = safe_div(
            stats["TP"] + stats["TN"],
            presence_n,
        )
        fish_recall = safe_div(
            stats["TP"],
            stats["TP"] + stats["FN"],
        )
        accepted_fish_precision = safe_div(
            stats["TP"],
            stats["TP"] + stats["FP"],
        )
        no_fish_rejection_rate = safe_div(
            stats["TN"],
            stats["TN"] + stats["FP"],
        )
        false_positive_rate = safe_div(
            stats["FP"],
            stats["FP"] + stats["TN"],
        )
        false_negative_rate = safe_div(
            stats["FN"],
            stats["FN"] + stats["TP"],
        )

        summary = {
            "total_images": len(self.all_images),
            "reviewed": stats["total_review_files"],
            "remaining": (
                len(self.all_images)
                - stats["total_review_files"]
            ),

            "direction_metrics": {
                "direction_reviewed": direction_n,
                "correct_V": stats["V"],
                "almost_A": stats["A"],
                "wrong_X": stats["X"],
                "exact_accuracy": exact_accuracy,
                "within_45_accuracy": within_45_accuracy,
                "wrong_rate": wrong_rate,
            },

            # Direction/pose mismatch flags (P/D buttons): raw counts, not
            # an "accuracy" -- these are only the exceptions someone chose
            # to flag on top of V/A/X, not an independent full pose review.
            "direction_pose_mismatch_flags": {
                "wrong_pose_correct_direction": stats["button_wrong_pose"],
                "wrong_direction_correct_pose": stats["button_wrong_dir"],
            },

            "presence_gate_metrics": {
                "presence_reviewed": presence_n,
                "human_fish": stats["human_fish"],
                "human_no_fish": stats["human_no_fish"],
                "TP_fish_kept": stats["TP"],
                "TN_no_fish_rejected": stats["TN"],
                "FP_no_fish_kept": stats["FP"],
                "FN_fish_rejected": stats["FN"],
                "accuracy": presence_accuracy,
                "fish_recall": fish_recall,
                "accepted_fish_precision": accepted_fish_precision,
                "no_fish_rejection_rate": no_fish_rejection_rate,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
            },

            "by_stage": {
                "background_filter": {
                    "reviewed": stats["bg_reviewed"],
                    "TN_correct_reject": stats["bg_TN"],
                    "FN_wrongly_rejected_fish": stats["bg_FN"],
                },
                "classifier_no_fish_class": {
                    "TP": stats["clf_TP"],
                    "TN": stats["clf_TN"],
                    "FP": stats["clf_FP"],
                    "FN": stats["clf_FN"],
                },
            },

            "review_button_counts": {
                "V": stats["button_V"],
                "A": stats["button_A"],
                "X": stats["button_X"],
                "Fish_presence_only": stats["FISH"],
                "No_Fish": stats["NO_FISH"],
            },

            "legacy_presence_unknown": (
                stats["legacy_presence_unknown"]
            ),
            "invalid_almost_on_no_fish": (
                stats["invalid_almost_on_no_fish"]
            ),
        }

        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._refresh_stats()

    # ========================================================
    # Finished
    # ========================================================

    def _finished(self):
        self._save_summary()
        stats = self._get_stats()

        direction_n = stats["direction_reviewed"]
        presence_n = stats["presence_reviewed"]

        exact = safe_div(stats["V"], direction_n) * 100
        within_45 = safe_div(
            stats["V"] + stats["A"],
            direction_n,
        ) * 100
        gate_accuracy = safe_div(
            stats["TP"] + stats["TN"],
            presence_n,
        ) * 100
        accepted_precision = safe_div(
            stats["TP"],
            stats["TP"] + stats["FP"],
        ) * 100

        messagebox.showinfo(
            "Review complete",
            (
                "All predictions reviewed!\n\n"
                "Direction, real fish only:\n"
                f"Correct: {stats['V']}\n"
                f"Almost:  {stats['A']}\n"
                f"Wrong:   {stats['X']}\n"
                f"Exact accuracy: {exact:.2f}%\n"
                f"Within 45 degrees: {within_45:.2f}%\n"
                f"Wrong Pose (dir OK): {stats['button_wrong_pose']}\n"
                f"Wrong Dir (pose OK): {stats['button_wrong_dir']}\n\n"
                "Full-pipeline fish / no-fish gate:\n"
                f"TP fish kept: {stats['TP']}\n"
                f"TN no-fish rejected: {stats['TN']}\n"
                f"FP no-fish kept: {stats['FP']}\n"
                f"FN fish rejected: {stats['FN']}\n"
                f"Gate accuracy: {gate_accuracy:.2f}%\n"
                f"Accepted fish precision: "
                f"{accepted_precision:.2f}%"
            ),
        )


# ============================================================
# Main
# ============================================================

# Marker for "--split-dir was passed with no path", resolved after --run is
# known (the run folder decides where the default sorted_by_class/ lives).
_USE_RUN_SPLIT_DIR = Path("<run>/sorted_by_class")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=RUN_DIR,
        help=(
            "The run folder to review: the one holding rois/, predictions/ "
            "and boxes.csv. reviews/ and review_summary.json are written "
            "inside it (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        nargs="?",
        const=_USE_RUN_SPLIT_DIR,
        default=None,
        metavar="DIR",
        help=(
            "Review from a sorted_by_class folder (the output of "
            "dataset_ops/sort_crops_by_class.py) instead of the flat rois/ "
            "pool. Each per-class sub-folder (no_fish/, regular/NE/, "
            "upside_down/high_conf/E/, ...) becomes its own sampling class: "
            "every pick chooses a random CLASS first, then a random image "
            "within it, instead of a random image from the whole pool. Rare "
            "classes get the same attention as common ones, so a small "
            "review session sees real variation. Pass with no path to use "
            "the run's own sorted_by_class/, or give a specific folder "
            "(e.g. sorted_by_class_0.85/)."
        ),
    )
    args = parser.parse_args()

    set_run_dir(args.run)

    split_dir = args.split_dir
    if split_dir == _USE_RUN_SPLIT_DIR:
        split_dir = DEFAULT_SPLIT_DIR

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Images folder not found:\n{IMAGES_DIR}"
        )

    if not PREDICTIONS_DIR.exists():
        raise FileNotFoundError(
            "Inference labels do not exist.\n\n"
            "Run the final inference script first."
        )

    root = tk.Tk()
    InferenceReviewer(root, split_dir=split_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
