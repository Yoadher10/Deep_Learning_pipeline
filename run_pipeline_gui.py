#!/usr/bin/env python3
"""
End-to-end pipeline GUI.

Choose where to start, browse to the matching input, and run the rest:

    Phase 0  video file        -> ffmpeg extracts PNG frames
    Phase 1  folder of frames  -> YOLO detects fish, crops ROIs
    Phase 2  folder of ROIs    -> ViT classifies direction / pose
    Phase 3  folder of pred.   -> distribution + upside-down summary + pie charts

Each run writes to its own timestamped folder:  <data root>/runs/<name>_<stamp>/
(frames/, rois/, predictions/, summary.txt, summary.json, charts/)

The "Sort crops" tab runs dataset_ops/sort_crops_by_class.py on a finished run:
pick the run folder (it fills in rois/, predictions/ and an output folder) and
sort the crops into per-class / per-direction folders, with the upside_down
bucket split into high_conf / low_conf for review.

"Stop", or closing the window mid-run, terminates the running phase process and
its children (each phase runs in its own process group / session).

Needs ffmpeg + ffprobe on PATH for phase 0 and the ML deps for phases 1-3.
"""

import os
import sys
import time
import queue
import signal
import threading
import subprocess
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from phase0_frame_extraction.extract_frames import (
    probe_video,
    suggest_sample_fps,
    estimate_extracted_frames,
    format_probe_report,
    FFmpegNotFound,
)

REPO_ROOT = Path(__file__).resolve().parent

FRAME_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# (menu label, script, short tag for the progress bar, relative wall-clock weight)
PHASES = [
    ("Phase 0 — video → frames",  REPO_ROOT / "phase0_frame_extraction" / "extract_frames.py",       "Frames",   0.10),
    ("Phase 1 — detect + crop fish", REPO_ROOT / "phase1_fish_detection" / "detect_and_crop_rois.py", "Detect",   0.45),
    ("Phase 2 — classify direction/pose", REPO_ROOT / "phase2_direction_pose_inference" / "run_classifier.py", "Classify", 0.42),
    ("Phase 3 — summarise + charts", REPO_ROOT / "phase3_analysis" / "summarize_predictions.py",      "Report",   0.03),
]

# What the user browses for, per start phase
START_BROWSE = [
    ("video",       "Select a video file"),
    ("frames",      "Select a folder of frame images"),
    ("rois",        "Select a folder of ROI crop images"),
    ("predictions", "Select a folder of prediction JSON files"),
]


def pipeline_python() -> str:
    """
    Interpreter used to run the phase scripts. The GUI may be launched by a
    Python that has Tkinter but not the ML deps (common on macOS). Prefer:
    $PIPELINE_PYTHON, a .venv beside the repo, an active VIRTUAL_ENV, then self.
    """
    candidates = []
    if os.environ.get("PIPELINE_PYTHON"):
        candidates.append(Path(os.environ["PIPELINE_PYTHON"]))
    for base in (REPO_ROOT / ".venv", REPO_ROOT.parent / ".venv"):
        candidates += [base / "bin" / "python", base / "Scripts" / "python.exe"]
    if os.environ.get("VIRTUAL_ENV"):
        venv = Path(os.environ["VIRTUAL_ENV"])
        candidates += [venv / "bin" / "python", venv / "Scripts" / "python.exe"]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def open_in_file_manager(path: Path):
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except OSError:
        pass


# ------------------------------------------------------------------ widgets

class SegmentedBar(tk.Canvas):
    """A determinate progress bar with tick marks at phase boundaries."""

    def __init__(self, master, height=30, **kw):
        super().__init__(master, height=height, highlightthickness=0,
                         bg="#e6e6ec", **kw)
        self._frac = 0.0
        self._segments = []          # (start_frac, end_frac, label)
        self.bind("<Configure>", lambda _e: self._redraw())

    def set_segments(self, segments):
        self._segments = segments
        self._redraw()

    def set_fraction(self, frac: float):
        self._frac = max(0.0, min(1.0, frac))
        self._redraw()

    def _redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1:
            return

        # filled portion
        self.create_rectangle(0, 0, w * self._frac, h, fill="#3b82f6", width=0)

        for start, end, label in self._segments:
            # boundary tick (skip the far right edge)
            if end < 0.999:
                x = w * end
                self.create_line(x, 0, x, h, fill="#3a3a3a", width=2)
            mid = (start + end) / 2
            done = self._frac >= end - 1e-6
            active = start - 1e-6 <= self._frac < end
            colour = "white" if self._frac > mid else "#555"
            weight = "bold" if (done or active) else "normal"
            self.create_text(w * mid, h / 2, text=label, fill=colour,
                             font=("TkDefaultFont", 9, weight))

        self.create_rectangle(1, 1, w - 1, h - 1, outline="#9a9aa5", width=1)


class ChartStrip(ttk.Frame):
    """Shows chart PNGs (scrollable); falls back to a message + open-folder button."""

    def __init__(self, master):
        super().__init__(master)
        self._imgs = []            # keep refs
        top = ttk.Frame(self)
        top.pack(fill=tk.X)
        self._msg = ttk.Label(top, text="Results appear here after a run.", padding=10)
        self._msg.pack(side=tk.LEFT)
        self._open_btn = ttk.Button(top, text="Open results folder", state=tk.DISABLED)
        self._open_btn.pack(side=tk.RIGHT, padx=8, pady=6)

        canvas = tk.Canvas(self, highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._grid = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self._grid, anchor="nw")
        self._grid.bind("<Configure>",
                        lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))

    def show(self, chart_paths, results_dir: Path):
        self._open_btn.config(state=tk.NORMAL,
                              command=lambda: open_in_file_manager(results_dir))
        for child in self._grid.winfo_children():
            child.destroy()
        self._imgs.clear()

        try:
            from PIL import Image, ImageTk
            has_pil = True
        except Exception:
            has_pil = False

        if chart_paths and has_pil:
            self._msg.config(text=f"Charts saved in {results_dir}")
            cols = 2
            for i, p in enumerate(chart_paths):
                try:
                    img = Image.open(p)
                    img.thumbnail((440, 440))
                    photo = ImageTk.PhotoImage(img)
                except Exception:
                    continue
                self._imgs.append(photo)
                lbl = ttk.Label(self._grid, image=photo)
                lbl.grid(row=i // cols, column=i % cols, padx=8, pady=8)
        elif chart_paths:
            self._msg.config(
                text=f"{len(chart_paths)} charts saved in {results_dir} "
                     f"(install 'pillow' in the GUI's Python to preview them here)")
        else:
            self._msg.config(text=f"Run finished — see {results_dir}")


SORT_SCRIPT = REPO_ROOT / "dataset_ops" / "sort_crops_by_class.py"


class SortPanel(ttk.Frame):
    """
    Tab for dataset_ops/sort_crops_by_class.py — sort a run's ROI crops into
    per-class / per-direction folders using the phase-2 predictions.
    """

    def __init__(self, master, app: "PipelineGUI"):
        super().__init__(master)
        self.app = app

        # Scrollable body: the tab is short, the form is not.
        canvas = tk.Canvas(self, highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.body = ttk.Frame(canvas, padding=10)
        body_id = canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(body_id, width=e.width))

        def _wheel(event):
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        def _bind_wheel(_e):
            canvas.bind_all("<MouseWheel>", _wheel)
            canvas.bind_all("<Button-4>", _wheel)
            canvas.bind_all("<Button-5>", _wheel)

        def _unbind_wheel(_e):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

        self.run_dir = tk.StringVar()
        self.rois = tk.StringVar()
        self.predictions = tk.StringVar()
        self.output = tk.StringVar()

        self.move = tk.BooleanVar(value=False)
        self.flat = tk.BooleanVar(value=False)
        self.raw_pose = tk.BooleanVar(value=False)
        self.no_conf_split = tk.BooleanVar(value=False)
        # Same gate phase 3 and the review tools use, so the folders you
        # review match the numbers in summary.json.
        self.conf_threshold = tk.StringVar(
            value=f"{config.UPSIDE_DOWN_CONF_THRESHOLD:g}")

        intro = ("Point this at a finished run folder (or fill the three folders "
                 "by hand). Crops are copied into <output>/{no_fish, regular, "
                 "upside_down, pose_not_applicable}/<DIRECTION>/. The upside_down "
                 "bucket is also split into high_conf / low_conf so you can review "
                 "the confident calls first.")
        ttk.Label(self.body, text=intro, wraplength=880, justify="left",
                  foreground="#555").grid(row=0, column=0, columnspan=3,
                                          sticky="w", pady=(0, 8))

        self._folder_row(1, "Run folder (optional):", self.run_dir,
                         self._pick_run_dir)
        self._folder_row(2, "ROI crops:", self.rois,
                         lambda: self._pick_dir(self.rois, "Select the ROI crops folder"))
        self._folder_row(3, "Predictions:", self.predictions,
                         lambda: self._pick_dir(self.predictions,
                                                "Select the predictions folder"))
        self._folder_row(4, "Output folder:", self.output,
                         lambda: self._pick_dir(self.output,
                                                "Select the output folder", must_exist=False))

        opts = ttk.LabelFrame(self.body, text="Options", padding=8)
        opts.grid(row=5, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Checkbutton(opts, text="Move crops instead of copying",
                        variable=self.move).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(opts, text="Flat — no per-direction sub-folders",
                        variable=self.flat).grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(opts, text="Score pose even for N/S fish (raw pose head)",
                        variable=self.raw_pose).grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(opts, text="Single upside_down bucket (no confidence split)",
                        variable=self.no_conf_split,
                        command=self._sync_conf_entry).grid(row=3, column=0, sticky="w",
                                                            padx=4, pady=2)
        conf_row = ttk.Frame(opts)
        conf_row.grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(conf_row, text="Upside-down confidence threshold:").pack(side=tk.LEFT)
        self.conf_entry = ttk.Entry(conf_row, width=6, textvariable=self.conf_threshold)
        self.conf_entry.pack(side=tk.LEFT, padx=6)

        run_row = ttk.Frame(self.body)
        run_row.grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.run_btn = ttk.Button(run_row, text="Sort crops", command=self._run)
        self.run_btn.pack(side=tk.LEFT)
        self.open_btn = ttk.Button(run_row, text="Open output folder",
                                   state=tk.DISABLED,
                                   command=lambda: open_in_file_manager(Path(self.output.get())))
        self.open_btn.pack(side=tk.LEFT, padx=6)
        self.status = ttk.Label(run_row, text="", foreground="#555")
        self.status.pack(side=tk.LEFT, padx=10)

        self.body.columnconfigure(1, weight=1)

    # -- widgets -------------------------------------------------------

    def _folder_row(self, row, label, var, command):
        ttk.Label(self.body, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(self.body, textvariable=var).grid(row=row, column=1, sticky="ew",
                                                    padx=6, pady=2)
        ttk.Button(self.body, text="Browse…", command=command).grid(row=row, column=2,
                                                                    pady=2)

    def _sync_conf_entry(self):
        self.conf_entry.config(state=tk.DISABLED if self.no_conf_split.get() else tk.NORMAL)

    # -- browse -------------------------------------------------------

    def _pick_dir(self, var, title, must_exist=True):
        chosen = filedialog.askdirectory(title=title)
        if not chosen:
            return
        if must_exist and not Path(chosen).is_dir():
            messagebox.showerror("Not found", f"Folder does not exist:\n{chosen}")
            return
        var.set(chosen)

    def _pick_run_dir(self):
        chosen = filedialog.askdirectory(title="Select a finished run folder")
        if not chosen:
            return
        run = Path(chosen)
        self.run_dir.set(chosen)
        for name, var in (("rois", self.rois), ("predictions", self.predictions)):
            sub = run / name
            if sub.is_dir():
                var.set(str(sub))
        self.output.set(str(run / "sorted_by_class"))
        missing = [n for n in ("rois", "predictions") if not (run / n).is_dir()]
        if missing:
            messagebox.showwarning(
                "Incomplete run",
                f"This folder has no {', '.join(missing)} sub-folder.\n"
                "Fill the paths in by hand if they live elsewhere.")

    # -- run --------------------------------------------------------

    def _run(self):
        if self.app.worker and self.app.worker.is_alive():
            messagebox.showinfo("Busy", "Another job is already running.")
            return
        rois, preds, out = self.rois.get().strip(), self.predictions.get().strip(), \
            self.output.get().strip()
        if not (rois and preds and out):
            messagebox.showerror("Missing folders",
                                 "Set the ROI crops, Predictions and Output folders.")
            return
        if not Path(rois).is_dir():
            messagebox.showerror("Not found", f"ROI folder does not exist:\n{rois}")
            return
        if not Path(preds).is_dir():
            messagebox.showerror("Not found", f"Predictions folder does not exist:\n{preds}")
            return

        threshold = self.conf_threshold.get().strip()
        if not self.no_conf_split.get():
            try:
                t = float(threshold)
                if not 0.0 <= t <= 1.0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Bad threshold",
                                     "Confidence threshold must be a number between 0 and 1.")
                return

        cmd = [pipeline_python(), "-u", str(SORT_SCRIPT),
               "--rois", rois, "--predictions", preds, "--output", out]
        if self.move.get():
            cmd.append("--move")
        if self.flat.get():
            cmd.append("--flat")
        if self.raw_pose.get():
            cmd.append("--use-raw-pose")
        if self.no_conf_split.get():
            cmd.append("--no-upside-conf-split")
        else:
            cmd += ["--upside-conf-threshold", threshold]

        verb = "Move" if self.move.get() else "Copy"
        if not messagebox.askokcancel(
                "Sort crops",
                f"{verb} crops from\n  {rois}\ninto\n  {out}\n\nProceed?"):
            return

        self.run_btn.config(state=tk.DISABLED)
        self.open_btn.config(state=tk.DISABLED)
        self.status.config(text="Sorting…")
        self.app.start_sort_job(cmd, Path(out), self)

    def on_done(self, ok: bool):
        self.run_btn.config(state=tk.NORMAL)
        self.status.config(text="Done" if ok else "Failed")
        if ok:
            self.open_btn.config(state=tk.NORMAL)


# ------------------------------------------------------------------ main GUI

class PipelineGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Fish Pipeline")
        root.geometry("1000x860")
        root.minsize(900, 600)

        self.start_var = tk.IntVar(value=0)         # phase index to start from
        self.fps_var = tk.StringVar()
        self.source_path: Path | None = None        # browsed video file or folder
        self.video_info: dict | None = None
        self.input_count: int = 0

        self.log_queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.current_proc: subprocess.Popen | None = None
        self.aborting = False
        self.chart_paths: list[str] = []
        self.results_dir: Path | None = None
        self._rate_samples: list[tuple[float, int]] = []
        self._job = "pipeline"                       # "pipeline" | "sort"
        self._sort_panel: SortPanel | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_log()

    # ------------------------------------------------------------- UI

    def _build_ui(self):
        pad = dict(padx=10, pady=5)

        start_frame = tk.LabelFrame(self.root, text="Start from")
        start_frame.pack(fill=tk.X, **pad)
        for idx, (label, *_rest) in enumerate(PHASES):
            tk.Radiobutton(start_frame, text=label, variable=self.start_var,
                           value=idx, command=self._on_start_change
                           ).pack(side=tk.LEFT, padx=6, pady=4)

        browse_row = tk.Frame(self.root)
        browse_row.pack(fill=tk.X, **pad)
        self.browse_btn = tk.Button(browse_row, text="Browse…", command=self._browse)
        self.browse_btn.pack(side=tk.LEFT)
        self.source_label = tk.Label(browse_row, text="No input selected", anchor="w")
        self.source_label.pack(side=tk.LEFT, padx=10)

        self.info_box = tk.Text(self.root, height=9, wrap="word", state=tk.DISABLED,
                                bg="#f4f4f4", relief=tk.FLAT)
        self.info_box.pack(fill=tk.X, **pad)

        self.fps_frame = tk.LabelFrame(self.root, text="Sampling frame rate")
        self.fps_frame.pack(fill=tk.X, **pad)
        self.fps_choices = tk.Frame(self.fps_frame)
        self.fps_choices.pack(fill=tk.X, padx=8, pady=4)
        custom_row = tk.Frame(self.fps_frame)
        custom_row.pack(fill=tk.X, padx=8, pady=4)
        tk.Radiobutton(custom_row, text="Custom fps:", variable=self.fps_var,
                       value="__custom__").pack(side=tk.LEFT)
        self.custom_entry = tk.Entry(custom_row, width=8)
        self.custom_entry.pack(side=tk.LEFT, padx=4)
        self.custom_entry.bind("<KeyRelease>", lambda _e: self._update_estimate())
        self.estimate_label = tk.Label(custom_row, text="", anchor="w")
        self.estimate_label.pack(side=tk.LEFT, padx=12)

        run_row = tk.Frame(self.root)
        run_row.pack(fill=tk.X, **pad)
        self.run_button = tk.Button(run_row, text="Run pipeline", state=tk.DISABLED,
                                    command=self._run)
        self.run_button.pack(side=tk.LEFT)
        self.stop_button = tk.Button(run_row, text="Stop", state=tk.DISABLED,
                                     command=self._stop)
        self.stop_button.pack(side=tk.LEFT, padx=4)
        self.device_label = tk.Label(run_row, text="Device: —",
                                     font=("TkDefaultFont", 10, "bold"))
        self.device_label.pack(side=tk.LEFT, padx=14)

        self.bar = SegmentedBar(self.root)
        self.bar.pack(fill=tk.X, padx=10, pady=(4, 0))
        sub = tk.Frame(self.root)
        sub.pack(fill=tk.X, padx=10)
        self.status_label = tk.Label(sub, text="", anchor="w", fg="#555")
        self.status_label.pack(side=tk.LEFT)
        self.step_label = tk.Label(sub, text="", anchor="e", fg="#333")
        self.step_label.pack(side=tk.RIGHT)

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        log_tab = ttk.Frame(self.tabs)
        self.tabs.add(log_tab, text="Log")
        self.log = tk.Text(log_tab, wrap="none", bg="#101418", fg="#d6dee6",
                           insertbackground="#d6dee6")
        self.log.pack(fill=tk.BOTH, expand=True)

        results_tab = ttk.Frame(self.tabs)
        self.tabs.add(results_tab, text="Results")
        self.charts = ChartStrip(results_tab)
        self.charts.pack(fill=tk.BOTH, expand=True)

        self._sort_panel = SortPanel(self.tabs, self)
        self.tabs.add(self._sort_panel, text="Sort crops")

        self._on_start_change()

    # ------------------------------------------------------------- helpers

    def _set_info(self, text: str):
        self.info_box.config(state=tk.NORMAL)
        self.info_box.delete("1.0", tk.END)
        self.info_box.insert(tk.END, text)
        self.info_box.config(state=tk.DISABLED)

    def _append_log(self, text: str):
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    def _active_phases(self):
        return PHASES[self.start_var.get():]

    def _on_start_change(self):
        self.source_path = None
        self.video_info = None
        self.source_label.config(text="No input selected")
        self.run_button.config(state=tk.DISABLED)
        kind, prompt = START_BROWSE[self.start_var.get()]
        self.browse_btn.config(text=f"Browse… ({kind})")
        is_video = kind == "video"
        state = tk.NORMAL if is_video else tk.DISABLED
        for child in self.fps_choices.winfo_children():
            try:
                child.config(state=state)
            except tk.TclError:
                pass
        self.custom_entry.config(state=state)
        self.fps_frame.config(
            text="Sampling frame rate" + ("" if is_video else "  (phase 0 only)"))
        self._set_info(f"{prompt}.")
        # show phase segments on the bar right away
        self.bar.set_segments(self._segments())
        self.bar.set_fraction(0.0)

    def _segments(self):
        phases = self._active_phases()
        total = sum(w for *_r, w in phases) or 1.0
        segs, acc = [], 0.0
        for _label, _script, tag, weight in phases:
            segs.append((acc / total, (acc + weight) / total, tag))
            acc += weight
        return segs

    # ------------------------------------------------------------- browse

    def _browse(self):
        kind, prompt = START_BROWSE[self.start_var.get()]
        if kind == "video":
            exts = " ".join(f"*{e}" for e in sorted(config.VIDEO_EXTENSIONS))
            chosen = filedialog.askopenfilename(
                title=prompt,
                filetypes=[("Video files", exts), ("All files", "*.*")])
        else:
            chosen = filedialog.askdirectory(title=prompt)
        if not chosen:
            return
        path = Path(chosen)

        if kind == "video":
            if not path.is_file():
                messagebox.showerror("Not found", f"File does not exist:\n{path}")
                return
            try:
                info = probe_video(path)
            except FFmpegNotFound as err:
                messagebox.showerror("ffmpeg missing", str(err))
                return
            except Exception as err:  # noqa: BLE001
                messagebox.showerror("Could not read video", str(err))
                return
            self.source_path = path
            self.video_info = info
            suggestions = suggest_sample_fps(info["fps"], info["duration"])
            self._set_info(format_probe_report(path, info, suggestions))
            self._populate_fps(suggestions)
        else:
            if not path.is_dir():
                messagebox.showerror("Not found", f"Folder does not exist:\n{path}")
                return
            if kind == "predictions":
                items = list(path.glob("*.json"))
            else:
                items = [p for p in path.iterdir()
                         if p.is_file() and p.suffix.lower() in FRAME_EXTENSIONS]
            if not items:
                what = "prediction JSON files" if kind == "predictions" else "image files"
                messagebox.showerror("Empty folder", f"No {what} in:\n{path}")
                return
            self.source_path = path
            self.input_count = len(items)
            noun = {"frames": "frames", "rois": "ROI crops",
                    "predictions": "prediction files"}[kind]
            self._set_info(
                f"{path}\n{self.input_count} {noun}\n\n"
                f"Running {self._active_phases()[0][0]} onward on this folder.")

        self.source_label.config(text=str(path))
        self.run_button.config(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)

    def _populate_fps(self, suggestions):
        for child in self.fps_choices.winfo_children():
            child.destroy()
        default = None
        for row in suggestions:
            label = f"{row['fps']:g} fps  (~{row['estimated_frames']} frames)"
            if row["recommended"]:
                label += "   ★ recommended"
                default = f"{row['fps']:g}"
            tk.Radiobutton(self.fps_choices, text=label, variable=self.fps_var,
                           value=f"{row['fps']:g}", command=self._update_estimate
                           ).pack(anchor="w")
        self.fps_var.set(default or f"{suggestions[0]['fps']:g}")
        self._update_estimate()

    def _selected_fps(self):
        raw = self.fps_var.get()
        if raw == "__custom__":
            raw = self.custom_entry.get().strip()
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if value > 0 else None

    def _update_estimate(self):
        if not self.video_info:
            return
        fps = self._selected_fps()
        if fps is None:
            self.estimate_label.config(text="")
            return
        vfps = self.video_info["fps"]
        note = f"  (> video {vfps:.2f} fps — will be clamped)" if fps > vfps + 1e-6 else ""
        est = estimate_extracted_frames(self.video_info["duration"], min(fps, vfps))
        self.estimate_label.config(text=f"≈ {est} frames through the pipeline{note}")

    # ------------------------------------------------------------- run

    def _run(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.source_path:
            messagebox.showerror("No input", "Browse to an input first.")
            return

        start = self.start_var.get()
        kind = START_BROWSE[start][0]
        phases = self._active_phases()

        extra = {}
        if kind == "video":
            fps = self._selected_fps()
            if fps is None:
                messagebox.showerror("Bad fps", "Choose or type a positive sampling fps.")
                return
            fps = min(fps, self.video_info["fps"])
            extra["video_path"] = str(self.source_path)
            extra["fps"] = fps
            est = estimate_extracted_frames(self.video_info["duration"], fps)
            head = f"Sample {self.source_path.name} at {fps:g} fps (~{est} frames)"
        else:
            extra[f"{kind}_input"] = str(self.source_path)
            head = f"{phases[0][0]} on {self.input_count} items from {self.source_path.name}"

        workdir = config.new_run_dir(self.source_path.stem)
        if not messagebox.askokcancel(
            "Run pipeline",
            f"{head}\n\nOutputs → {workdir}\n\nProceed?"):
            return

        self._job = "pipeline"
        self.aborting = False
        self.chart_paths = []
        self.results_dir = workdir
        self._rate_samples = []
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.device_label.config(text="Device: detecting…")
        self.status_label.config(text="")
        self.step_label.config(text="")
        self.bar.set_segments(self._segments())
        self.bar.set_fraction(0.0)
        self.log.delete("1.0", tk.END)
        self.tabs.select(0)

        self.worker = threading.Thread(
            target=self._run_steps, args=(phases, workdir, extra), daemon=True)
        self.worker.start()

    def start_sort_job(self, cmd: list[str], output_dir: Path, panel: "SortPanel"):
        if self.worker and self.worker.is_alive():
            return
        self._job = "sort"
        self.aborting = False
        self.chart_paths = []
        self.results_dir = output_dir
        self._rate_samples = []
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Sorting crops")
        self.step_label.config(text="")
        self.bar.set_segments([(0.0, 1.0, "Sort")])
        self.bar.set_fraction(0.0)
        self.log.delete("1.0", tk.END)
        self.tabs.select(0)
        self.worker = threading.Thread(
            target=self._sort_worker, args=(cmd, output_dir), daemon=True)
        self.worker.start()

    def _sort_worker(self, cmd: list[str], output_dir: Path):
        env = os.environ.copy()
        env["PIPELINE_STRUCTURED_OUTPUT"] = "1"
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.log_queue.put(("log", f"Interpreter: {cmd[0]}\n"))
        self.log_queue.put(("log", "Command: " + " ".join(cmd) + "\n\n"))
        self.log_queue.put(("overall", 0.0))
        code = self._stream(cmd, env, 0.0, 1.0, 1.0)
        if self.aborting:
            self.log_queue.put(("log", "\nSort stopped by user.\n"))
            self.log_queue.put(("done", "stopped"))
        elif code != 0:
            self.log_queue.put(("log", f"\nSort failed (exit {code}).\n"))
            self.log_queue.put(("done", False))
        else:
            self.log_queue.put(("done", True))

    def _stop(self):
        if not (self.worker and self.worker.is_alive()):
            return
        self.aborting = True
        self.status_label.config(text="Stopping…")
        self._terminate_current()

    def _terminate_current(self):
        proc = self.current_proc
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True)
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno(
                    "Quit", "A pipeline run is in progress.\nStop it and quit?"):
                return
            self.aborting = True
            self._terminate_current()
        self.root.destroy()

    # ------------------------------------------------------------- worker

    def _run_steps(self, phases, workdir: Path, extra: dict):
        env = os.environ.copy()
        env["FISH_PIPELINE_DATA"] = str(workdir)
        env["PIPELINE_STRUCTURED_OUTPUT"] = "1"
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        workdir.mkdir(parents=True, exist_ok=True)

        python_bin = pipeline_python()
        self.log_queue.put(("log", f"Interpreter: {python_bin}\n"))
        if python_bin == sys.executable and "VIRTUAL_ENV" not in os.environ:
            self.log_queue.put(("log",
                "  (no .venv found — if a phase reports ModuleNotFoundError, create "
                "one and `pip install -r requirements.txt`, or set PIPELINE_PYTHON)\n"))
        self.log_queue.put(("log", f"Working directory: {workdir}\n\n"))

        total_weight = sum(w for *_r, w in phases) or 1.0
        base_weight = 0.0

        for label, script, _tag, weight in phases:
            self.log_queue.put(("status", label))
            self.log_queue.put(("newphase", None))
            self.log_queue.put(("log", f"\n{'=' * 60}\n{label}\n{'=' * 60}\n"))
            self.log_queue.put(("overall", base_weight / total_weight))

            cmd = [python_bin, "-u", str(script)]
            if script.name == "extract_frames.py":
                cmd += ["--video", extra["video_path"], "--fps", str(extra["fps"])]
            elif script.name == "detect_and_crop_rois.py" and extra.get("frames_input"):
                cmd += ["--input", extra["frames_input"]]
            elif script.name == "run_classifier.py" and extra.get("rois_input"):
                cmd += ["--images", extra["rois_input"]]
            elif script.name == "summarize_predictions.py" and extra.get("predictions_input"):
                cmd += ["--predictions", extra["predictions_input"]]

            code = self._stream(cmd, env, base_weight, weight, total_weight)
            if self.aborting:
                self.log_queue.put(("log", f"\n[{label}] stopped by user.\n"))
                self.log_queue.put(("done", "stopped"))
                return
            if code != 0:
                self.log_queue.put(("log", f"\n[{label}] failed (exit {code}). Stopping.\n"))
                self.log_queue.put(("done", False))
                return
            base_weight += weight
            self.log_queue.put(("overall", base_weight / total_weight))

        summary_txt = workdir / "summary.txt"
        if summary_txt.is_file():
            self.log_queue.put(("log", "\n\n" + summary_txt.read_text(encoding="utf-8")))
        self.log_queue.put(("done", True))

    def _stream(self, cmd, env, base_weight, phase_weight, total_weight) -> int:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(REPO_ROOT), env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, **kwargs)
        except OSError as err:
            self.log_queue.put(("log", f"Could not launch: {err}\n"))
            return 1

        self.current_proc = proc
        try:
            for line in proc.stdout:
                self._handle_line(line, base_weight, phase_weight, total_weight)
            return proc.wait()
        finally:
            self.current_proc = None

    def _handle_line(self, line: str, base_weight, phase_weight, total_weight):
        text = line.rstrip("\n")
        if text.startswith("@DEVICE "):
            rest = text.split(" ", 2)[2] if len(text.split(" ", 2)) > 2 else text
            self.log_queue.put(("device", rest))
            self.log_queue.put(("log", f"   • device: {rest}\n"))
        elif text.startswith("@STEP "):
            desc = text[6:]
            self.log_queue.put(("newphase", None))
            self.log_queue.put(("substep", (desc, None, None)))
            self.log_queue.put(("log", f"   → {desc}\n"))
        elif text.startswith("@PROGRESS "):
            parts = text.split(" ", 3)
            try:
                done, total = int(parts[1]), int(parts[2])
            except (ValueError, IndexError):
                return
            label = parts[3] if len(parts) > 3 else ""
            frac = (done / total) if total else 0.0
            overall = (base_weight + phase_weight * frac) / total_weight
            self.log_queue.put(("overall", overall))
            self.log_queue.put(("substep", (label, done, total)))
        elif text.startswith("@CHART "):
            self.chart_paths.append(text[7:])
        else:
            self.log_queue.put(("log", line))

    # ------------------------------------------------------------- pump

    def _fmt_eta(self, seconds: float) -> str:
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m{seconds % 60:02d}s"
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"

    def _render_substep(self, label, done, total):
        if done is None:
            self.step_label.config(text=label)
            return
        now = time.monotonic()
        self._rate_samples.append((now, done))
        while self._rate_samples and now - self._rate_samples[0][0] > 5.0:
            self._rate_samples.pop(0)

        txt = f"{label}  {done}/{total}"
        if len(self._rate_samples) >= 2:
            dt = self._rate_samples[-1][0] - self._rate_samples[0][0]
            dd = self._rate_samples[-1][1] - self._rate_samples[0][1]
            if dt > 0 and dd > 0:
                rate = dd / dt
                remaining = max(0, total - done)
                txt += f"   ·   {rate:.0f}/s   ·   ETA {self._fmt_eta(remaining / rate)}"
        self.step_label.config(text=txt)

    def _poll_log(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "status":
                    self.status_label.config(text=payload)
                elif kind == "newphase":
                    self._rate_samples = []
                elif kind == "substep":
                    self._render_substep(*payload)
                elif kind == "device":
                    self.device_label.config(text=f"Device: {payload}")
                elif kind == "overall":
                    self.bar.set_fraction(payload)
                elif kind == "done":
                    self.run_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    self.step_label.config(text="")
                    self.status_label.config(text={
                        True: "Finished", False: "Failed", "stopped": "Stopped",
                    }.get(payload, "Failed"))
                    if self._job == "sort":
                        if payload is True:
                            self.bar.set_fraction(1.0)
                        if self._sort_panel:
                            self._sort_panel.on_done(payload is True)
                        if payload is True:
                            messagebox.showinfo(
                                "Done", f"Crops sorted into\n{self.results_dir}")
                    elif payload is True:
                        self.bar.set_fraction(1.0)
                        if self.results_dir:
                            self.charts.show(self.chart_paths, self.results_dir)
                        self.tabs.select(1)
                        messagebox.showinfo("Done", "Pipeline finished — see the Results tab.")
        except queue.Empty:
            pass
        self.root.after(80, self._poll_log)


def main():
    root = tk.Tk()
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
