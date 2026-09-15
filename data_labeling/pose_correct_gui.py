#!/usr/bin/env python3
"""
Tiny GUI to label the pose of already-sorted crops as correct or false.

Point it at a folder of crop images (for example the upside_down/ folder
produced by sort_crops_by_class.py) and step through every image:

    C / <Right>   pose label is CORRECT for this crop
    F / <Left>    pose label is FALSE for this crop
    S / <Down>    skip, decide later
    U            undo the last decision
    <Esc>        quit (progress is always saved)

Decisions are written incrementally to  pose_labels.csv  next to the images,
so you can stop and resume at any time.

    python pose_correct_gui.py
    python pose_correct_gui.py --images DIR
    python pose_correct_gui.py --images DIR --output labels.csv
"""

import csv
import sys
import argparse
from pathlib import Path

import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

IMG_W = 820
IMG_H = 720

BG = "#1e1e2e"
PANEL_BG = "#0d0d1a"
TEXT = "#dddddd"
MUTED = "#aaaaaa"
GREEN = "#7ee787"
RED = "#ff7b72"
GOLD = "#FFD700"


def load_labels(path: Path) -> dict:
    labels = {}
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                labels[row["image"]] = row["label"]
    return labels


def write_labels(path: Path, labels: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "label"])
        for image in sorted(labels):
            writer.writerow([image, labels[image]])


class PoseLabeler:
    def __init__(self, root, images_dir: Path, output_path: Path):
        self.root = root
        self.images_dir = images_dir
        self.output_path = output_path
        self.root.title(f"Pose review - {images_dir.name}")
        self.root.configure(bg=BG)

        self.images = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.images:
            messagebox.showerror("No images", f"No crop images in:\n{images_dir}")
            self.root.destroy()
            return

        self.labels = load_labels(output_path)
        self.history = []
        self._photo = None

        self.index = self._first_unlabeled()

        self._build_ui()
        self._show()

    def _first_unlabeled(self) -> int:
        for i, p in enumerate(self.images):
            if p.name not in self.labels:
                return i
        return len(self.images)

    def _build_ui(self):
        self.canvas = tk.Canvas(
            self.root, width=IMG_W, height=IMG_H, bg=PANEL_BG,
            highlightthickness=1, highlightbackground="#3a3a5c",
        )
        self.canvas.pack(padx=10, pady=(10, 6))

        self.file_label = tk.Label(
            self.root, text="", font=("Arial", 10), fg=MUTED, bg=BG,
        )
        self.file_label.pack()

        self.progress_label = tk.Label(
            self.root, text="", font=("Arial", 11, "bold"), fg=TEXT, bg=BG,
        )
        self.progress_label.pack(pady=(2, 6))

        buttons = tk.Frame(self.root, bg=BG)
        buttons.pack(pady=(0, 10))

        self._button(buttons, "F  False", lambda: self._label("false"), RED).pack(
            side=tk.LEFT, padx=4
        )
        self._button(buttons, "C  Correct", lambda: self._label("correct"), GREEN).pack(
            side=tk.LEFT, padx=4
        )
        self._button(buttons, "S  Skip", lambda: self._label("skip"), GOLD).pack(
            side=tk.LEFT, padx=4
        )
        self._button(buttons, "U  Undo", self._undo, "#bbbbbb").pack(
            side=tk.LEFT, padx=4
        )

        tk.Label(
            self.root,
            text="C / Right = correct     F / Left = false     "
                 "S / Down = skip     U = undo     Esc = quit",
            font=("Arial", 8), fg=MUTED, bg=BG,
        ).pack(pady=(0, 8))

        self.root.bind("<KeyPress>", self._on_key)
        self.root.focus_set()

    def _button(self, parent, text, command, fg):
        return tk.Button(
            parent, text=text, command=command, font=("Arial", 12, "bold"),
            fg=fg, bg="#2b2b40", activebackground="#3b3b55", activeforeground=fg,
            relief=tk.RAISED, bd=1, padx=12, pady=8, cursor="hand2",
        )

    def _show(self):
        if self.index >= len(self.images):
            self._finished()
            return

        path = self.images[self.index]
        self.canvas.delete("all")

        with Image.open(path) as source:
            img = source.convert("RGB")
        iw, ih = img.size
        scale = min(IMG_W / iw, IMG_H / ih)
        img = img.resize((max(1, int(iw * scale)), max(1, int(ih * scale))),
                         Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(IMG_W // 2, IMG_H // 2, image=self._photo,
                                 anchor=tk.CENTER)

        existing = self.labels.get(path.name)
        suffix = f"   (current label: {existing})" if existing else ""
        self.file_label.config(text=f"{path.name}{suffix}")

        done = len(self.labels)
        counts = {"correct": 0, "false": 0, "skip": 0}
        for v in self.labels.values():
            counts[v] = counts.get(v, 0) + 1
        self.progress_label.config(
            text=(f"{self.index + 1} / {len(self.images)}      "
                  f"labeled {done}   "
                  f"correct {counts['correct']}   "
                  f"false {counts['false']}   "
                  f"skip {counts['skip']}")
        )

    def _label(self, value):
        if self.index >= len(self.images):
            return
        name = self.images[self.index].name
        self.history.append((self.index, self.labels.get(name)))
        self.labels[name] = value
        write_labels(self.output_path, self.labels)
        self.index += 1
        self._show()

    def _undo(self):
        if not self.history:
            return
        prev_index, prev_label = self.history.pop()
        name = self.images[prev_index].name
        if prev_label is None:
            self.labels.pop(name, None)
        else:
            self.labels[name] = prev_label
        write_labels(self.output_path, self.labels)
        self.index = prev_index
        self._show()

    def _on_key(self, event):
        key = event.keysym.lower()
        if key in ("c", "right"):
            self._label("correct")
        elif key in ("f", "left"):
            self._label("false")
        elif key in ("s", "down"):
            self._label("skip")
        elif key == "u":
            self._undo()
        elif key == "escape":
            self.root.destroy()

    def _finished(self):
        self.canvas.delete("all")
        counts = {"correct": 0, "false": 0, "skip": 0}
        for v in self.labels.values():
            counts[v] = counts.get(v, 0) + 1
        self.progress_label.config(text="All images labeled.")
        self.file_label.config(text=f"Saved to {self.output_path}")
        messagebox.showinfo(
            "Done",
            (f"All {len(self.images)} crops labeled.\n\n"
             f"correct: {counts['correct']}\n"
             f"false:   {counts['false']}\n"
             f"skip:    {counts['skip']}\n\n"
             f"Saved to:\n{self.output_path}"),
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True,
                        help="Folder of crop images to review "
                             "(e.g. a run's sorted_by_class/upside_down/ folder)")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV file for labels (default: pose_labels.csv in the images folder)")
    args = parser.parse_args()

    images_dir = args.images
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    output_path = args.output or (images_dir / "pose_labels.csv")

    root = tk.Tk()
    PoseLabeler(root, images_dir, output_path)
    root.mainloop()


if __name__ == "__main__":
    main()
