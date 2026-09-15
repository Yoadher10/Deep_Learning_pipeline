#!/usr/bin/env python3
"""
Fish Photo Labeling GUI
Labels each image with direction (0-10) and pose (0-2), saves as JSON.

Direction keys:
  q=NW(8)  w=N(1)  e=NE(2)
  a=W(7)           d=E(3)
  z=SW(6)  x=S(5)  c=SE(4)
  t = Tail/In (9)
  f = Face/Out (10)
  0 = No Fish (0)

  North and South auto-save with pose=0 (N/A).
  No Fish also auto-saves with pose=0.

Pose keys (after selecting direction):
  1 = Regular (belly)
  2 = Upside Down (back)

Other:
  Space / Right = Skip
  u = Undo last label
"""

import sys
import json
import math
import random
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
try:
    import config
    _INITIAL_DIR = str(config.ROIS_DIR) if config.ROIS_DIR.exists() else None
except Exception:
    _INITIAL_DIR = None

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Key -> (direction_value, display_name)
DIRECTION_KEYS = {
    '0': (0,  'No Fish'),
    'w': (1,  'North'),
    'e': (2,  'North-East'),
    'd': (3,  'East'),
    'c': (4,  'South-East'),
    'x': (5,  'South'),
    'z': (6,  'South-West'),
    'a': (7,  'West'),
    'q': (8,  'North-West'),
    't': (9,  'Tail (In)'),
    'f': (10, 'Face (Out)'),
}

# These directions auto-save with pose=0 (N/A) — no pose input needed
AUTO_SAVE_DIRECTIONS = {0, 1, 5}   # No Fish, North, South

POSE_KEYS = {'1': 1, '2': 2}
POSE_NAMES = {0: 'N/A', 1: 'Regular (belly)', 2: 'Upside Down (back)'}

# 8 compass arrows: (dir_value, key, short_label, screen_angle_degrees)
# Screen angles: 0=right(E), 90=down(S), -90=up(N)
COMPASS_DIRS = [
    (1, 'w', 'N',   -90),
    (2, 'e', 'NE',  -45),
    (3, 'd', 'E',     0),
    (4, 'c', 'SE',   45),
    (5, 'x', 'S',    90),
    (6, 'z', 'SW',  135),
    (7, 'a', 'W',   180),
    (8, 'q', 'NW', -135),
]

IMG_W, IMG_H = 600, 480
COMPASS_W, COMPASS_H = 290, 350


class FishLabeler:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Fish Photo Labeler")
        self.root.configure(bg='#1e1e2e')
        self.root.resizable(False, False)

        self.source_folder: Path = None
        self.labels_folder: Path = None
        self.all_images: list[Path] = []
        self.labeled_set: set[str] = set()
        self.unlabeled_pool: list[Path] = []

        self.current_image_path: Path = None
        self.current_direction: int = None
        self.last_saved = None   # (Path, label_dict, json_path) for undo
        self._photo = None       # keep reference to prevent GC

        self._build_ui()
        self.root.after(50, self._select_folder)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        top = tk.Frame(self.root, bg='#1e1e2e')
        top.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: image
        self.image_canvas = tk.Canvas(
            top, width=IMG_W, height=IMG_H,
            bg='#0d0d1a', highlightthickness=1, highlightbackground='#3a3a5c')
        self.image_canvas.pack(side=tk.LEFT, padx=(0, 10))
        self._show_placeholder("No image loaded")

        # Right panel
        right = tk.Frame(top, bg='#1e1e2e', width=COMPASS_W)
        right.pack(side=tk.LEFT, fill=tk.Y)
        right.pack_propagate(False)

        # Compass
        self.compass_canvas = tk.Canvas(
            right, width=COMPASS_W, height=COMPASS_H,
            bg='#0d0d1a', highlightthickness=1, highlightbackground='#3a3a5c')
        self.compass_canvas.pack()
        self._draw_compass(None)

        # Status labels
        self.dir_label = self._status_label(right, "Direction: press a key")
        self.pose_label = self._status_label(right, "Pose: —")
        self.file_label = self._status_label(right, "File: —", small=True)
        self.progress_label = self._status_label(right, "Progress: — / —", small=True)

        # Buttons
        btn_row = tk.Frame(right, bg='#1e1e2e')
        btn_row.pack(fill=tk.X, pady=(8, 0))

        self.skip_btn = tk.Button(
            btn_row, text="Skip [Space]", command=self._skip,
            bg='#2e2e4e', fg='#aaa', relief=tk.FLAT, padx=8, pady=4,
            activebackground='#3a3a6a', activeforeground='white', cursor='hand2')
        self.skip_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.undo_btn = tk.Button(
            btn_row, text="Undo [u]", command=self._undo,
            bg='#4e2e2e', fg='#aaa', relief=tk.FLAT, padx=8, pady=4,
            activebackground='#6a3a3a', activeforeground='white', cursor='hand2',
            state=tk.DISABLED)
        self.undo_btn.pack(side=tk.LEFT)

        # Key reference legend
        legend_text = (
            "────────────────\n"
            "q=NW   w=N   e=NE\n"
            "a=W          d=E\n"
            "z=SW   x=S   c=SE\n"
            "t = Tail (in)\n"
            "f = Face (out)\n"
            "0 = No Fish\n"
            "────────────────\n"
            "1 = Regular (belly)\n"
            "2 = Upside Down\n"
            "(N/A auto for N/S/NoFish)\n"
            "────────────────\n"
            "Space / → = Skip\n"
            "u = Undo"
        )
        tk.Label(right, text=legend_text, font=('Courier', 8),
                 fg='#444', bg='#1e1e2e', justify=tk.LEFT
                 ).pack(pady=(6, 0), anchor='w')

        self.root.bind('<KeyPress>', self._on_key)
        self.root.focus_set()

    def _status_label(self, parent, text, small=False):
        lbl = tk.Label(parent, text=text,
                       font=('Arial', 9 if small else 10),
                       fg='#666' if small else '#888',
                       bg='#1e1e2e', anchor='w', wraplength=COMPASS_W - 4)
        lbl.pack(fill=tk.X, pady=1)
        return lbl

    def _show_placeholder(self, msg):
        self.image_canvas.delete('all')
        self.image_canvas.create_text(
            IMG_W // 2, IMG_H // 2, text=msg,
            fill='#333', font=('Arial', 16))

    # ------------------------------------------------------------------ Folder / pool

    def _select_folder(self):
        folder = filedialog.askdirectory(
            title="Select source image folder", initialdir=_INITIAL_DIR)
        if not folder:
            self.root.destroy()
            return

        self.source_folder = Path(folder)
        self.labels_folder = self.source_folder.parent / 'labels'
        self.labels_folder.mkdir(exist_ok=True)

        self._load_pool()

        if not self.unlabeled_pool:
            messagebox.showinfo("All Done",
                                "All images in this folder are already labeled!")
            self.root.destroy()
            return

        self.root.title(f"Fish Photo Labeler — {self.source_folder.name}")
        self._load_random_image()

    def _load_pool(self):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.all_images = sorted(
            p for p in self.source_folder.iterdir()
            if p.suffix.lower() in exts)
        self.labeled_set = {p.stem for p in self.labels_folder.glob('*.json')}
        self.unlabeled_pool = [p for p in self.all_images
                                if p.stem not in self.labeled_set]
        self._refresh_progress()

    # ------------------------------------------------------------------ Image display

    def _load_random_image(self):
        if not self.unlabeled_pool:
            messagebox.showinfo("All Done!",
                                f"All {len(self.all_images)} images labeled!")
            return

        self.current_image_path = random.choice(self.unlabeled_pool)
        self.current_direction = None
        self._display_image(self.current_image_path)
        self._draw_compass(None)
        self._refresh_status()

    def _display_image(self, path: Path):
        self.image_canvas.delete('all')

        if not HAS_PIL:
            self.image_canvas.create_text(
                IMG_W // 2, IMG_H // 2,
                text="Pillow not installed.\nRun: pip install Pillow",
                fill='#f55', font=('Arial', 13), justify=tk.CENTER)
            return

        img = Image.open(path).convert('RGB')

        # Scale to fit canvas (upscale allowed for small fish crops)
        iw, ih = img.size
        scale = min(IMG_W / iw, IMG_H / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        img = img.resize((nw, nh), Image.LANCZOS)

        self._photo = ImageTk.PhotoImage(img)
        self.image_canvas.create_image(
            IMG_W // 2, IMG_H // 2, anchor=tk.CENTER, image=self._photo)

    # ------------------------------------------------------------------ Compass

    def _draw_compass(self, selected_direction):
        c = self.compass_canvas
        c.delete('all')

        cx = COMPASS_W // 2
        cy = 140   # center of the compass circle
        r = 100    # radius

        # Outer ring
        c.create_oval(cx - r, cy - r, cx + r, cy + r,
                      fill='#0a0a18', outline='#2a3a5a', width=2)

        # Draw 8 compass arrows
        for dir_val, key, short, angle_deg in COMPASS_DIRS:
            angle = math.radians(angle_deg)
            x1 = cx + 0.22 * r * math.cos(angle)
            y1 = cy + 0.22 * r * math.sin(angle)
            x2 = cx + 0.72 * r * math.cos(angle)
            y2 = cy + 0.72 * r * math.sin(angle)

            selected = (selected_direction == dir_val)
            auto = dir_val in AUTO_SAVE_DIRECTIONS
            arrow_col = '#88ee88' if (selected and auto) else '#FFD700' if selected else '#1e3a5a'
            text_col  = '#88ee88' if (selected and auto) else '#FFD700' if selected else '#3a5a7a'

            c.create_line(x1, y1, x2, y2, fill=arrow_col,
                          width=3 if selected else 2,
                          arrow=tk.LAST, arrowshape=(10, 13, 4))

            # Label outside circle
            lx = cx + (r + 22) * math.cos(angle)
            ly = cy + (r + 22) * math.sin(angle)
            c.create_text(lx, ly, text=f"[{key}]\n{short}", fill=text_col,
                          font=('Arial', 7, 'bold' if selected else 'normal'),
                          justify=tk.CENTER)

        # Center dot
        c.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                      fill='#334', outline='#556')

        # Extra entries below the circle: Tail, Face, No Fish
        rows = [
            (9,  't', 'Tail  (In  / מתרחק)'),
            (10, 'f', 'Face (Out / מתקרב)'),
            (0,  '0', 'No Fish'),
        ]
        y = cy + r + 28
        for dir_val, key, name in rows:
            selected = (selected_direction == dir_val)
            auto = dir_val in AUTO_SAVE_DIRECTIONS
            if selected:
                fg = '#88ee88' if auto else '#FFD700'
                font = ('Arial', 9, 'bold')
            else:
                fg = '#5a2a2a' if dir_val == 0 else '#2a4a6a'
                font = ('Arial', 9)
            c.create_text(cx, y, text=f"[{key}]  {name}",
                          fill=fg, font=font)
            y += 24

    # ------------------------------------------------------------------ Key handling

    def _on_key(self, event):
        key = event.char.lower() if event.char else ''
        sym = event.keysym

        # Navigation
        if sym in ('Right', 'space') or key == ' ':
            self._skip()
            return
        if key == 'u':
            self._undo()
            return

        # Direction key
        if key in DIRECTION_KEYS:
            dir_val, _ = DIRECTION_KEYS[key]
            self.current_direction = dir_val
            self._draw_compass(dir_val)
            self._refresh_status()
            if dir_val in AUTO_SAVE_DIRECTIONS:
                self._save_label(0)
            return

        # Pose key (only after direction selected)
        if key in POSE_KEYS:
            if self.current_direction is None:
                self.dir_label.config(
                    text="Select direction first!", fg='#ff7777')
                return
            self._save_label(POSE_KEYS[key])

    # ------------------------------------------------------------------ Save / undo

    def _save_label(self, pose: int):
        if self.current_image_path is None:
            return

        data = {"direction": self.current_direction, "pose": pose}
        json_path = self.labels_folder / (self.current_image_path.stem + '.json')

        with open(json_path, 'w') as fh:
            json.dump(data, fh)

        self.labeled_set.add(self.current_image_path.stem)
        if self.current_image_path in self.unlabeled_pool:
            self.unlabeled_pool.remove(self.current_image_path)

        self.last_saved = (self.current_image_path, data, json_path)
        self.undo_btn.config(state=tk.NORMAL)

        self._refresh_progress()
        self._load_random_image()

    def _skip(self):
        candidates = [p for p in self.unlabeled_pool
                      if p != self.current_image_path]
        if not candidates:
            return
        self.current_image_path = random.choice(candidates)
        self.current_direction = None
        self._display_image(self.current_image_path)
        self._draw_compass(None)
        self._refresh_status()

    def _undo(self):
        if self.last_saved is None:
            return
        img_path, _, json_path = self.last_saved
        if json_path.exists():
            json_path.unlink()
        self.labeled_set.discard(img_path.stem)
        if img_path not in self.unlabeled_pool:
            self.unlabeled_pool.append(img_path)

        self.last_saved = None
        self.undo_btn.config(state=tk.DISABLED)

        self.current_image_path = img_path
        self.current_direction = None
        self._display_image(img_path)
        self._draw_compass(None)
        self._refresh_progress()
        self._refresh_status()

    # ------------------------------------------------------------------ Status

    def _refresh_status(self):
        if self.current_direction is not None:
            dv = self.current_direction
            name = next(v[1] for k, v in DIRECTION_KEYS.items() if v[0] == dv)
            auto = dv in AUTO_SAVE_DIRECTIONS
            self.dir_label.config(
                text=f"Direction: {name}  ({dv})",
                fg='#88ee88' if auto else '#FFD700')
            self.pose_label.config(
                text="Pose: N/A  (auto-saved)" if auto
                     else "Pose: press  1  or  2",
                fg='#88ee88' if auto else '#88aaff')
        else:
            self.dir_label.config(
                text="Direction: press a direction key", fg='#888')
            self.pose_label.config(text="Pose: —", fg='#888')

        if self.current_image_path:
            self.file_label.config(text=f"File: {self.current_image_path.name}")

    def _refresh_progress(self):
        total = len(self.all_images)
        done = len(self.labeled_set)
        self.progress_label.config(
            text=f"Progress: {done} / {total}  ({total - done} remaining)")


# ------------------------------------------------------------------ Entry point

def main():
    root = tk.Tk()

    if not HAS_PIL:
        root.withdraw()
        messagebox.showerror(
            "Missing dependency",
            "Pillow is required.\n\nInstall it with:\n\n    pip install Pillow\n\nthen re-run the script.")
        root.destroy()
        return

    FishLabeler(root)
    root.mainloop()


if __name__ == '__main__':
    main()
