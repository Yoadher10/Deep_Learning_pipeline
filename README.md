# Fish Direction & Pose Pipeline

Estimate a fish's **swimming direction** (8 compass classes + "no fish") and
**pose** (regular / upside-down) from underwater video frames.

The pipeline is four stages backed by two trained models:

```
video        ──▶ [phase 0] ffmpeg frame extraction   ──▶ PNG frames
PNG frames   ──▶ [phase 1] YOLOv8 fish detector      ──▶ fish crops (ROIs)
fish crops   ──▶ [phase 2] multi-task ViT classifier ──▶ per-crop prediction JSON
predictions  ──▶ [phase 3] aggregate report          ──▶ summary.json / summary.txt
```

`run_pipeline_gui.py` runs all four from a single window: browse to a video,
pick a sampling frame rate, watch it run.

Everything else in this repo (model training, labeling, dataset surgery,
evaluation, review GUIs) supports building those two models but is **not** part
of the runtime inference path.

---

## Layout

| Folder | What it is |
|---|---|
| `run_pipeline_gui.py` | **Start here.** GUI: pick a video, choose fps, run phases 0-3. |
| `config.py` | **Every path and shared constant.** Edit here, not in the scripts. |
| `phase0_frame_extraction/` | Runtime stage 0 — video → PNG frames (ffmpeg). |
| `phase1_fish_detection/` | Runtime stage 1 — detect fish, crop ROIs. Weights in `weights/`. |
| `phase2_direction_pose_inference/` | Runtime stage 2 — classify direction + pose. Weights in `weights/`. |
| `phase3_analysis/` | Runtime stage 3 — distribution + upside-down stats (`summarize_predictions.py`) and pie charts (`charts.py`). |
| `model_training/fish_detector/` | Train the YOLOv8 detector. |
| `model_training/direction_classifier/` | Build manifest → split → train the ViT. |
| `data_labeling/` | Manual direction/pose labeling GUI + label-set analysis. |
| `dataset_ops/` | Dataset-surgery tools (sorting predictions, filtering, quality selection). |
| `evaluation/` | Error analysis, batch sanity checks, the prediction-review GUI. |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Also needed outside pip:

- **ffmpeg / ffprobe** on `PATH` (phase 0) — `brew install ffmpeg` /
  `sudo apt install ffmpeg` / <https://ffmpeg.org>. Or set `FFMPEG_BIN` /
  `FFPROBE_BIN`.
- **Tkinter** for the GUIs — bundled with the python.org installers; on Ubuntu
  `sudo apt install python3-tk`.

For a specific CUDA build of PyTorch, install `torch`/`torchvision` first per
<https://pytorch.org/get-started/locally/>, then `pip install -r requirements.txt`.

### Device (GPU) selection

`torch_device.py` picks the best backend automatically: **CUDA** (NVIDIA) →
**MPS** (Apple-Silicon GPU) → **CPU**. The original training ran on a
Windows/CUDA box; on this Mac the GPU is Metal/`mps`, which
`torch.cuda.is_available()` never reports — that's why phases 1-2 previously ran
on CPU. Both the YOLO detector and the ViT now use `select_device()`.

The GUI shows the active device in bold and prints it per phase. Unsupported
MPS ops fall back to CPU automatically (`PYTORCH_ENABLE_MPS_FALLBACK=1`, set in
`config.py`). To force CPU, call `select_device(prefer="cpu")`.

Quick check that MPS is visible to your interpreter:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

---

## Configuring paths

Code lives in this repo; the image/label/prediction folders are large and live
outside it. Point the pipeline at your data with one environment variable:

```bash
export FISH_PIPELINE_DATA=/path/to/deep_learning_data      # macOS / Linux
set    FISH_PIPELINE_DATA=D:\deep_learning_data            # Windows
```

If unset, everything defaults to `full_pipeline/data/`. Expected layout under
`FISH_PIPELINE_DATA` (see `config.py` for the full list):

```
frames/                     PNG frames              (phase 0 output → phase 1 input)
rois/                       fish crops              (phase 1 output → phase 2 input)
predictions/                prediction JSON         (phase 2 output)
summary.json / summary.txt  aggregate report        (phase 3 output)
charts/*.png                result pie charts       (phase 3 output)
labels/                     manual label JSONs
trainset/
    images/  labels/        crops + labels for ViT training
    manifest.csv  splits/   built by model_training/direction_classifier/
vit_fish_output/            training checkpoints / logs
runs/<name>_<timestamp>/    one self-contained folder per GUI run (all of the above)
```

`ffmpeg` and `ffprobe` must be on `PATH` for phase 0 (or set `FFMPEG_BIN` /
`FFPROBE_BIN`).

The trained weights are bundled in the repo:
`phase1_fish_detection/weights/yolo_fish_detector_v2.pt` and
`phase2_direction_pose_inference/weights/best_model_v7.pt` (tracked with
**Git LFS** — see `.gitattributes` — because the second file is ~330MB, over
GitHub's 100MB per-file limit without it; run `git lfs install` once before
cloning/pushing if you haven't already).

**No script has a hardcoded personal path.** Every script resolves its paths
from `config.py`, which in turn resolves from `FISH_PIPELINE_DATA` — so
relocating your entire data tree is always a single environment variable,
never a source edit. On top of that, most standalone scripts *also* accept
direct `--flag` overrides for their own inputs/outputs (run any of them with
`--help` to see which) for one-off overrides without touching the env var at
all; a few (`train.py`, `validation_error_analysis.py`, and `dataset.py`,
which they share) intentionally rely on the env var alone, since their paths
come from one shared dataset-loading module used by several scripts at once.

---

## Running the inference pipeline

Easiest: `python run_pipeline_gui.py`.

1. **Start from** — pick the phase to begin at (0 video · 1 frames folder ·
   2 ROI-crops folder · 3 predictions folder).
2. **Browse…** — one button; the dialog matches the chosen start phase.
3. For a video, pick a sampling fps (the GUI shows the estimated frame count and
   a recommended value).
4. **Run pipeline** — the remaining phases run in order.

Every run writes to its own timestamped folder,
`<data root>/runs/<name>_<YYYYmmdd-HHMMSS>/`, so repeated runs never overwrite.

- **One overall progress bar** with **tick marks at each phase boundary** and the
  phase name on each segment. It's weighted by phase and driven by sub-step
  events the scripts emit: ffmpeg frame count, model loading, per-frame
  detection, per-batch classification, chart rendering.
- The line under the bar shows the current sub-step with a **live rate and ETA**
  (e.g. `classifying 2000/4000 · 46/s · ETA 0m43s`).
- The active **device** (CUDA / MPS / CPU) is shown in bold and per phase.
- **Results tab** — the phase-3 pie charts (fish vs no-fish, direction
  distribution, pose, upside-down by direction) plus the text summary, with an
  "Open results folder" button.
- **Sort crops tab** — runs `dataset_ops/sort_crops_by_class.py` on a finished
  run: pick the run folder and it sorts the ROI crops into per-class /
  per-direction folders, splitting `upside_down/` into `high_conf/` and
  `low_conf/` for review.

Or run the stages by hand:

```bash
# Phase 0 — video → PNG frames at a chosen sampling rate
python phase0_frame_extraction/extract_frames.py --video clip.mp4 --fps 5
python phase0_frame_extraction/extract_frames.py --video clip.mp4 --probe   # info only

# Phase 1 — detect fish and crop ROIs
python phase1_fish_detection/detect_and_crop_rois.py
#   defaults: --input $DATA/frames  --output $DATA/rois  --weights <bundled>

# Phase 2 — classify direction + pose, write one JSON per crop
python phase2_direction_pose_inference/run_classifier.py
#   defaults: --images $DATA/rois  --output $DATA/predictions  --weights <bundled>

# Phase 3 — aggregate report + pie charts
python phase3_analysis/summarize_predictions.py
#   defaults: --predictions $DATA/predictions  --charts-dir $DATA/charts
```

Each phase-2 JSON is one crop's prediction — `direction` / `direction_name` /
`direction_confidence`, `no_fish_confidence`, and `pose` / `pose_name` /
`pose_confidence` (pose is `null` for No Fish / N / S). No accept/reject gate.

Phase 3 aggregates those into `summary.txt` / `summary.json` plus `charts/*.png`
(pie charts): crop and no-fish counts, the direction distribution over fish
crops, and the pose breakdown.

For pose, an "Upside Down" call is only counted as upside-down when the pose
head's confidence is at least `config.UPSIDE_DOWN_CONF_THRESHOLD`; weaker calls
are folded into "Regular". On an unseen test video the confident calls were
almost all truly upside-down and the low-confidence ones almost all regular. The
report keeps the raw pose-head counts too, a confidence histogram of every raw
upside-down call, the per-direction upside-down rate, and the list of frames
that contain a confirmed upside-down fish (the review shortlist).

---

## Reproducing the models

### Fish detector (YOLOv8)

```bash
cp model_training/fish_detector/data.yaml.example model_training/fish_detector/data.yaml
#   edit data.yaml to point at your YOLO dataset
python model_training/fish_detector/train_yolo.py
```

### Direction / pose classifier (ViT)

Run in order:

```bash
python model_training/direction_classifier/build_manifest.py   # trainset/labels → manifest.csv
python model_training/direction_classifier/split_dataset.py     # manifest.csv → splits/{train,val,test}.csv (temporal-chunk split, no frame leakage)
python model_training/direction_classifier/train.py             # → vit_fish_output/best_model_v7.pt
```

`dataset.py` is the shared dataset module (imported by `train.py` and the
evaluation scripts); it is not run directly. It applies a **horizontal-flip**
augmentation on the training split only (mirroring the direction label,
E↔W / NE↔NW / SE↔SW; pose unchanged). Vertical flip is deliberately not used —
see *What we tried* below.

Training results for the bundled checkpoint (`best_model_v7.pt`) are in
`model_training/direction_classifier/training_log.txt`
(best epoch 18, val loss 0.42, test direction acc 88% / macro F1 0.83,
pose macro F1 0.97).

---

## Supporting tools

| Script | Purpose |
|---|---|
| `data_labeling/direction_pose_gui.py` | Label crops with direction + pose (writes one JSON per crop). |
| `data_labeling/analyze_labels.py` | Class distribution + validity report for a label folder. |
| `dataset_ops/sort_crops_by_class.py` | Sort ROI crops into `no_fish/ regular/ upside_down/ …` (and per-direction) folders using the phase-2 predictions. `upside_down/` is split into `high_conf/` vs `low_conf/` at `--upside-conf-threshold`. Flags: `--move`, `--flat`, `--use-raw-pose`, `--no-upside-conf-split`. Also available as the **"Sort crops" tab** in the GUI. |
| `dataset_ops/select_best_crops.py` | Score crops by visual quality, copy the top-N. |
| `dataset_ops/folder_complement.py` | Copy `SOURCE` minus filenames present in `SUBTRACT`. |
| `dataset_ops/extract_face_tail_classes.py` | Drop Face(9)/Tail(10) samples → the 9-class set. |
| `dataset_ops/extract_no_fish.py` | Pull "no fish" samples out for re-review. |
| `dataset_ops/extract_invalid_labels.py` | Pull samples whose (direction, pose) breaks the rules. |
| `evaluation/validation_error_analysis.py` | Confusion matrices + copies of misclassified crops, against the held-out **validation split** (needs ground-truth labels). |
| `evaluation/check_batches.py` | Flag training batches with zero valid pose labels. |
| `evaluation/review_predictions_gui.py` | Manually score a full pipeline **run** on unseen video (no ground truth) — see below. |
| `evaluation/review_analysis.py` | Turn that GUI's saved reviews into accuracy numbers — see below. |

### `evaluation/review_predictions_gui.py` — manual review of a full run

Unlike `validation_error_analysis.py` (which needs an already-labeled split),
this reviews a **fresh, unlabeled run's** actual output crop by crop, so you can
score the pipeline end-to-end on unseen video. It covers both classified crops
(`rois/`) *and* crops the phase-1 background filter discarded before they ever
reached the classifier (`rois_ignored_background/`), since a fish the filter
wrongly threw away is otherwise invisible to every other metric.

```bash
# default: pooled random sampling across the whole run
FISH_PIPELINE_DATA=/path/to/run python evaluation/review_predictions_gui.py

# class-stratified sampling: pick a random per-class folder first (from
# dataset_ops/sort_crops_by_class.py output), then a random image inside it,
# so rare classes get reviewed as often as the majority class
FISH_PIPELINE_DATA=/path/to/run python evaluation/review_predictions_gui.py --split-dir
FISH_PIPELINE_DATA=/path/to/run python evaluation/review_predictions_gui.py --split-dir /path/to/run/sorted_by_class_0.85
```

Review controls: `V`/`A`/`X` score direction (correct / 45°-off / wrong); `P`
flags a correct direction with a wrong pose call, `D` the reverse; `F`/`N` are
presence-only (fish / no fish); `U` undoes the last review. Reviews are written
to `<run>/reviews/*.json`, one file per crop, and a running
`<run>/review_summary.json` is kept up to date after every review — safe to
stop and resume across sessions, and safe to mix `--split-dir` and default runs
on the same `reviews/` folder (a crop reviewed once is never shown again in
either mode).

### `evaluation/review_analysis.py` — turn those reviews into numbers

Reads every file in `<run>/reviews/` and reports: how many crops landed in the
correct class (a `V` review) vs. not, broken down by what actually went wrong;
the background filter's own false-negative rate; and — specifically for the
pose head's `UPSIDE_DOWN_CONF_THRESHOLD` gate — how many low-confidence
"Upside Down" calls it correctly downgraded to "Regular" vs. wrongly buried,
plus a sweep showing what would happen at other threshold values on the same
reviewed ground truth. Re-run any time after reviewing more crops; it always
reads the full current set.

```bash
FISH_PIPELINE_DATA=/path/to/run python evaluation/review_analysis.py
FISH_PIPELINE_DATA=/path/to/run python evaluation/review_analysis.py --thresholds 0.5,0.6,0.7,0.8,0.9
FISH_PIPELINE_DATA=/path/to/run python evaluation/review_analysis.py --out-json results.json
```

All of these read their defaults from `config.py`; most also take
`--images` / `--labels` / `--output` overrides.

---

## What we tried

- **Direction-confidence gating (dropped).** An earlier build reported, and at
  one point down-weighted, predictions with `direction_confidence` below 0.60.
  It conflated three unrelated things — "not a fish", "genuinely hard pose", and
  head-on N/S fish (which are inherently low-confidence) — so it was removed.
  Nothing is gated on direction confidence now; the raw confidence is still in
  every prediction JSON if you want to filter downstream.
- **Vertical-flip augmentation for the pose head (failed).** To synthesise
  upside-down training crops we flipped whole crops top-to-bottom and inverted
  the pose label. That also inverts the scene's top-to-bottom lighting gradient,
  so the model learned "inverted lighting ⇒ upside-down" instead of body
  orientation: test metrics looked excellent but real inference got worse. It
  was replaced with real mined upside-down crops plus **horizontal-flip only**
  (lighting-safe).
- **Pose head, current state.** Even the v7 model over-predicts upside-down for
  fish seen at oblique angles at this video resolution, where belly-up vs
  belly-down is barely visible. The practical mitigation is
  `UPSIDE_DOWN_CONF_THRESHOLD`: confident calls are reliable, low-confidence
  calls are counted as regular and can be eyeballed in the sort tool's
  `upside_down/low_conf/` folder.
- **Static-background rejection (kept).** Tank decorations that pass YOLO are
  rejected by comparing each detection box to a per-video median-background
  image (`BACKGROUND_MATCH_THRESHOLD`, `BACKGROUND_SAMPLE_FRAMES`).

## Notes

- `phase6`–`phase8` in the old layout were empty and were removed.
- Training data and past training bundles live in `../pipeline_archive/`, not in
  this tree.
- **This folder ships code + trained weights only.** No videos, no frame/ROI/
  prediction data, no `.venv/`. Create a venv from `requirements.txt` and point
  `FISH_PIPELINE_DATA` at your own data folder (or just run the GUI on a video —
  it makes `data/runs/<name>_<timestamp>/` for you).
- `phase2_direction_pose_inference/weights/best_model_v7.pt` is the
  inference-only checkpoint (~350 MB: model weights + metadata, no optimizer
  state). The full training checkpoint is archived separately.
