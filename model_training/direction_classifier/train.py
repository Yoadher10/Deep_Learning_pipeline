#!/usr/bin/env python3
"""
Train the multi-task Fish ViT: one ViT backbone with a 9-class direction head
and a 2-class pose head.

Reads the splits built by split_dataset.py, trains with early stopping on
validation loss, saves every improving epoch to ``good_epochs/`` and the best
one to the checkpoint path, then evaluates that best checkpoint on the test
split.

Usage:
    python model_training/direction_classifier/train.py
    python model_training/direction_classifier/train.py --batch-size 16 --epochs 40
    python model_training/direction_classifier/train.py \
        --images /path/to/trainset/images \
        --splits-dir /path/to/trainset/splits \
        --output-dir /path/to/vit_fish_output

All path defaults come from config.py (i.e. from $FISH_PIPELINE_DATA); the
flags exist for one-off runs that do not match that layout.
"""

import sys
import argparse
from pathlib import Path
import time

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTModel

from dataset import create_datasets, MODEL_NAME


# ============================================================
# Configuration
# ============================================================
#
# These are the defaults. main() overwrites the ones that have a matching
# command-line flag (see parse_args) before any of them is read, so a flag
# and the default are interchangeable everywhere below.

# 64 fits an 8 GB GPU (~6 GB peak); drop to 16-32 on smaller cards.
BATCH_SIZE = 64

# High enough ceiling, early stopping will normally stop sooner
NUM_EPOCHS = 60
PATIENCE = 10

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

NUM_DIRECTION_CLASSES = config.NUM_DIRECTION_CLASSES
NUM_POSE_CLASSES = config.NUM_POSE_CLASSES

# Pose classes are imbalanced (~3862 regular : ~796 upside-down for v7).
# A softer-than-inverse-frequency weight is used; see config.POSE_CLASS_WEIGHTS.
POSE_CLASS_WEIGHTS = config.POSE_CLASS_WEIGHTS

OUTPUT_DIR = config.TRAIN_OUTPUT_DIR

BEST_MODEL_PATH = OUTPUT_DIR / "best_model_v7.pt"

# Every epoch that improves validation loss is also saved here.
GOOD_EPOCHS_DIR = OUTPUT_DIR / "good_epochs"

# Dataset location; overridable with --images / --splits-dir.
IMAGES_DIR = config.TRAINSET_IMAGES
SPLITS_DIR = config.SPLITS_DIR


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

POSE_NAMES = {
    0: "Regular",
    1: "Upside Down",
}


# ============================================================
# Timing helpers
# ============================================================

def synchronize_device(device):
    """
    CUDA operations are asynchronous.

    Synchronizing before taking a timestamp makes GPU timing
    measurements represent the actual completed computation.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


def format_time(seconds):
    """
    Convert seconds into a human-readable string.
    """

    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, seconds = divmod(seconds, 60)

    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"

    hours, minutes = divmod(minutes, 60)

    return (
        f"{int(hours)}h "
        f"{int(minutes)}m "
        f"{seconds:.1f}s"
    )


# ============================================================
# Device / GPU
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")

    return torch.device("cpu")


def print_device_info(device):
    print("=" * 70)
    print("DEVICE CHECK")
    print("=" * 70)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if device.type == "cuda":
        gpu_index = torch.cuda.current_device()

        props = torch.cuda.get_device_properties(gpu_index)

        print()
        print("GPU TRAINING ENABLED")
        print(f"GPU:             {torch.cuda.get_device_name(gpu_index)}")
        print(f"CUDA version:    {torch.version.cuda}")
        print(f"GPU VRAM:        {props.total_memory / 1024**3:.2f} GB")
        print(f"Device selected: {device}")

        print()
        print(">>> CUDA IS WORKING, ViT WILL TRAIN ON THE GPU <<<")

    elif device.type == "mps":
        print()
        print("Apple MPS acceleration enabled.")
        print(">>> MODEL WILL TRAIN ON THE APPLE GPU <<<")

    else:
        print()
        print("WARNING: CUDA/MPS NOT AVAILABLE")
        print(">>> MODEL WILL TRAIN ON CPU <<<")

    print()


def print_gpu_memory(prefix="GPU"):
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"{prefix} VRAM: "
        f"allocated={allocated:.2f} GB | "
        f"reserved={reserved:.2f} GB | "
        f"peak={peak:.2f} GB"
    )


# ============================================================
# Model
# ============================================================

class MultiTaskFishViT(nn.Module):

    def __init__(
        self,
        model_name,
        num_direction_classes=9,
        num_pose_classes=2,
    ):
        super().__init__()

        # Pretrained ViT backbone
        self.vit = ViTModel.from_pretrained(model_name)

        hidden_size = self.vit.config.hidden_size

        dropout_probability = self.vit.config.hidden_dropout_prob

        self.dropout = nn.Dropout(dropout_probability)

        # Direction:
        # No Fish + eight compass directions
        self.direction_head = nn.Linear(
            hidden_size,
            num_direction_classes,
        )

        # Pose:
        # Regular / Upside Down
        self.pose_head = nn.Linear(
            hidden_size,
            num_pose_classes,
        )

    def forward(self, pixel_values):

        outputs = self.vit(
            pixel_values=pixel_values
        )

        # ViT output:
        # [batch, sequence, hidden]
        #
        # position 0 is the CLS token
        cls_embedding = outputs.last_hidden_state[:, 0]

        cls_embedding = self.dropout(cls_embedding)

        direction_logits = self.direction_head(
            cls_embedding
        )

        pose_logits = self.pose_head(
            cls_embedding
        )

        return direction_logits, pose_logits


# ============================================================
# Loss
# ============================================================

class MultiTaskLoss(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.direction_loss_fn = nn.CrossEntropyLoss()

        pose_weights = torch.tensor(
            POSE_CLASS_WEIGHTS,
            dtype=torch.float32,
            device=device,
        )

        self.pose_loss_fn = nn.CrossEntropyLoss(
            weight=pose_weights
        )

    def forward(
        self,
        direction_logits,
        pose_logits,
        direction_targets,
        pose_targets,
    ):

        # Direction is valid for every image
        direction_loss = self.direction_loss_fn(
            direction_logits,
            direction_targets,
        )

        # Pose is only valid when pose_target != -100
        valid_pose_mask = pose_targets != -100

        if valid_pose_mask.any():
            pose_loss = self.pose_loss_fn(
                pose_logits[valid_pose_mask],
                pose_targets[valid_pose_mask],
            )
        else:
            # Zero loss, while keeping connection
            # to the computation graph.
            pose_loss = pose_logits.sum() * 0.0

        total_loss = direction_loss + pose_loss

        return (
            total_loss,
            direction_loss,
            pose_loss,
        )


# ============================================================
# Metrics helpers
# ============================================================

def calculate_macro_f1(
    predictions,
    targets,
    num_classes,
):
    f1_scores = []

    for cls in range(num_classes):

        tp = sum(
            p == cls and t == cls
            for p, t in zip(predictions, targets)
        )

        fp = sum(
            p == cls and t != cls
            for p, t in zip(predictions, targets)
        )

        fn = sum(
            p != cls and t == cls
            for p, t in zip(predictions, targets)
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )

        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    use_amp,
):
    model.train()

    total_loss = 0.0
    total_direction_loss = 0.0
    total_pose_loss = 0.0

    correct_direction = 0
    total_direction = 0

    correct_pose = 0
    total_pose = 0

    first_batch = True

    synchronize_device(device)
    epoch_start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):

        pixel_values = batch["pixel_values"].to(
            device,
            non_blocking=True,
        )

        direction_targets = batch["direction"].to(
            device,
            non_blocking=True,
        )

        pose_targets = batch["pose"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------------------
        # Mixed precision
        # ----------------------------------------------------

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp,
        ):

            direction_logits, pose_logits = model(
                pixel_values
            )

            (
                loss,
                direction_loss,
                pose_loss,
            ) = criterion(
                direction_logits,
                pose_logits,
                direction_targets,
                pose_targets,
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        # ----------------------------------------------------
        # GPU sanity check after actual computation
        # ----------------------------------------------------

        if first_batch:
            first_batch = False

            print()
            print(
                f"First batch tensor device: "
                f"{pixel_values.device}"
            )

            if pixel_values.is_cuda:
                print(
                    ">>> CONFIRMED: TRAINING BATCH IS ON NVIDIA GPU <<<"
                )

                print_gpu_memory("After first batch")

            print()

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        total_loss += loss.item()
        total_direction_loss += direction_loss.item()
        total_pose_loss += pose_loss.item()

        # ----------------------------------------------------
        # Direction accuracy
        # ----------------------------------------------------

        direction_predictions = direction_logits.argmax(dim=1)

        correct_direction += (
            direction_predictions == direction_targets
        ).sum().item()

        total_direction += direction_targets.size(0)

        # ----------------------------------------------------
        # Pose accuracy
        # ----------------------------------------------------

        valid_pose_mask = pose_targets != -100

        if valid_pose_mask.any():

            pose_predictions = pose_logits.argmax(dim=1)

            correct_pose += (
                pose_predictions[valid_pose_mask]
                ==
                pose_targets[valid_pose_mask]
            ).sum().item()

            total_pose += valid_pose_mask.sum().item()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_idx == 1
            or batch_idx % 25 == 0
            or batch_idx == len(loader)
        ):

            elapsed = time.perf_counter() - epoch_start_time

            avg_batch_time = elapsed / batch_idx

            remaining_batches = len(loader) - batch_idx

            eta = avg_batch_time * remaining_batches

            print(
                f"\rBatch "
                f"{batch_idx:4d}/{len(loader):4d} | "
                f"Loss {loss.item():.4f} | "
                f"Elapsed {format_time(elapsed)} | "
                f"ETA {format_time(eta)}",
                end="",
                flush=True,
            )

    synchronize_device(device)

    epoch_time = (
        time.perf_counter()
        - epoch_start_time
    )

    print()

    num_batches = len(loader)

    return {
        "loss":
            total_loss / num_batches,

        "direction_loss":
            total_direction_loss / num_batches,

        "pose_loss":
            total_pose_loss / num_batches,

        "direction_accuracy":
            correct_direction / total_direction,

        "pose_accuracy":
            (
                correct_pose / total_pose
                if total_pose > 0
                else 0
            ),

        "time":
            epoch_time,
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp,
):
    model.eval()

    total_loss = 0.0
    total_direction_loss = 0.0
    total_pose_loss = 0.0

    all_direction_predictions = []
    all_direction_targets = []

    all_pose_predictions = []
    all_pose_targets = []

    synchronize_device(device)
    evaluation_start_time = time.perf_counter()

    for batch in loader:

        pixel_values = batch["pixel_values"].to(
            device,
            non_blocking=True,
        )

        direction_targets = batch["direction"].to(
            device,
            non_blocking=True,
        )

        pose_targets = batch["pose"].to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp,
        ):

            direction_logits, pose_logits = model(
                pixel_values
            )

            (
                loss,
                direction_loss,
                pose_loss,
            ) = criterion(
                direction_logits,
                pose_logits,
                direction_targets,
                pose_targets,
            )

        total_loss += loss.item()
        total_direction_loss += direction_loss.item()
        total_pose_loss += pose_loss.item()

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        direction_predictions = direction_logits.argmax(dim=1)

        all_direction_predictions.extend(
            direction_predictions.cpu().tolist()
        )

        all_direction_targets.extend(
            direction_targets.cpu().tolist()
        )

        # ----------------------------------------------------
        # Pose
        # ----------------------------------------------------

        valid_pose_mask = pose_targets != -100

        if valid_pose_mask.any():

            pose_predictions = pose_logits.argmax(dim=1)

            valid_predictions = pose_predictions[
                valid_pose_mask
            ]

            valid_targets = pose_targets[
                valid_pose_mask
            ]

            all_pose_predictions.extend(
                valid_predictions.cpu().tolist()
            )

            all_pose_targets.extend(
                valid_targets.cpu().tolist()
            )

    synchronize_device(device)

    evaluation_time = (
        time.perf_counter()
        - evaluation_start_time
    )

    # --------------------------------------------------------
    # Direction metrics
    # --------------------------------------------------------

    direction_correct = sum(
        p == t
        for p, t in zip(
            all_direction_predictions,
            all_direction_targets,
        )
    )

    direction_accuracy = (
        direction_correct
        / len(all_direction_targets)
    )

    direction_macro_f1 = calculate_macro_f1(
        all_direction_predictions,
        all_direction_targets,
        NUM_DIRECTION_CLASSES,
    )

    # --------------------------------------------------------
    # Pose metrics
    # --------------------------------------------------------

    pose_correct = sum(
        p == t
        for p, t in zip(
            all_pose_predictions,
            all_pose_targets,
        )
    )

    pose_accuracy = (
        pose_correct / len(all_pose_targets)
        if all_pose_targets
        else 0
    )

    pose_macro_f1 = calculate_macro_f1(
        all_pose_predictions,
        all_pose_targets,
        NUM_POSE_CLASSES,
    )

    # --------------------------------------------------------
    # Regular and upside-down recall
    # --------------------------------------------------------

    regular_total = sum(
        t == 0
        for t in all_pose_targets
    )

    regular_correct = sum(
        p == 0 and t == 0
        for p, t in zip(
            all_pose_predictions,
            all_pose_targets,
        )
    )

    upside_total = sum(
        t == 1
        for t in all_pose_targets
    )

    upside_correct = sum(
        p == 1 and t == 1
        for p, t in zip(
            all_pose_predictions,
            all_pose_targets,
        )
    )

    regular_recall = (
        regular_correct / regular_total
        if regular_total
        else 0
    )

    upside_recall = (
        upside_correct / upside_total
        if upside_total
        else 0
    )

    num_batches = len(loader)

    return {
        "loss":
            total_loss / num_batches,

        "direction_loss":
            total_direction_loss / num_batches,

        "pose_loss":
            total_pose_loss / num_batches,

        "direction_accuracy":
            direction_accuracy,

        "direction_macro_f1":
            direction_macro_f1,

        "pose_accuracy":
            pose_accuracy,

        "pose_macro_f1":
            pose_macro_f1,

        "regular_recall":
            regular_recall,

        "upside_recall":
            upside_recall,

        "upside_total":
            upside_total,

        "direction_predictions":
            all_direction_predictions,

        "direction_targets":
            all_direction_targets,

        "pose_predictions":
            all_pose_predictions,

        "pose_targets":
            all_pose_targets,

        "time":
            evaluation_time,
    }


# ============================================================
# Pretty printing
# ============================================================

def print_metrics(title, metrics):

    print()
    print(title)
    print("-" * 70)

    print(
        f"Loss:                 "
        f"{metrics['loss']:.4f}"
    )

    print(
        f"Direction loss:       "
        f"{metrics['direction_loss']:.4f}"
    )

    print(
        f"Pose loss:            "
        f"{metrics['pose_loss']:.4f}"
    )

    print(
        f"Direction accuracy:   "
        f"{metrics['direction_accuracy'] * 100:.2f}%"
    )

    if "direction_macro_f1" in metrics:
        print(
            f"Direction macro F1:   "
            f"{metrics['direction_macro_f1']:.4f}"
        )

    print(
        f"Pose accuracy:        "
        f"{metrics['pose_accuracy'] * 100:.2f}%"
    )

    if "pose_macro_f1" in metrics:

        print(
            f"Pose macro F1:        "
            f"{metrics['pose_macro_f1']:.4f}"
        )

        print(
            f"Regular recall:       "
            f"{metrics['regular_recall'] * 100:.2f}%"
        )

        print(
            f"Upside-down recall:   "
            f"{metrics['upside_recall'] * 100:.2f}% "
            f"(n={metrics['upside_total']})"
        )

    if "time" in metrics:
        print(
            f"Time:                 "
            f"{format_time(metrics['time'])}"
        )


# ============================================================
# Per-class direction results
# ============================================================

def print_direction_per_class(metrics):

    predictions = metrics["direction_predictions"]
    targets = metrics["direction_targets"]

    print()
    print("DIRECTION PER-CLASS RECALL")
    print("-" * 70)

    for cls in range(NUM_DIRECTION_CLASSES):

        total = sum(
            target == cls
            for target in targets
        )

        correct = sum(
            prediction == cls and target == cls
            for prediction, target in zip(
                predictions,
                targets,
            )
        )

        recall = (
            correct / total
            if total
            else 0
        )

        print(
            f"{cls} {DIRECTION_NAMES[cls]:8s}: "
            f"{recall * 100:6.2f}% "
            f"({correct}/{total})"
        )


# ============================================================
# Checkpoint helper
# ============================================================

def create_checkpoint(
    epoch,
    model,
    optimizer,
    val_metrics,
    train_metrics,
):
    return {
        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "val_loss":
            val_metrics["loss"],

        "val_direction_accuracy":
            val_metrics["direction_accuracy"],

        "val_direction_macro_f1":
            val_metrics["direction_macro_f1"],

        "val_pose_accuracy":
            val_metrics["pose_accuracy"],

        "val_pose_macro_f1":
            val_metrics["pose_macro_f1"],

        "train_loss":
            train_metrics["loss"],

        "model_name":
            MODEL_NAME,

        "num_direction_classes":
            NUM_DIRECTION_CLASSES,

        "num_pose_classes":
            NUM_POSE_CLASSES,

        "pose_class_weights":
            POSE_CLASS_WEIGHTS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "batch_size":
            BATCH_SIZE,
    }


# ============================================================
# Main
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train the multi-task (direction + pose) Fish ViT classifier."
        ),
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=IMAGES_DIR,
        help="Folder of training crops (default: %(default)s)",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_DIR,
        help=(
            "Folder holding train.csv / val.csv / test.csv, as written by "
            "split_dataset.py (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=(
            "Where checkpoints and good_epochs/ are written "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--checkpoint-name",
        default=BEST_MODEL_PATH.name,
        help=(
            "Filename of the best-epoch checkpoint inside --output-dir "
            "(default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            "64 needs ~6 GB of VRAM; drop to 16-32 on smaller cards "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Maximum epochs; early stopping usually ends sooner "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help="Epochs without validation improvement before stopping "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="AdamW learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help="AdamW weight decay (default: %(default)s)",
    )
    parser.add_argument(
        "--pose-weights",
        default=",".join(str(w) for w in POSE_CLASS_WEIGHTS),
        help=(
            "Comma-separated class weights for the pose head, "
            "regular,upside_down (default: %(default)s)"
        ),
    )

    return parser.parse_args(argv)


def apply_args(args):
    """
    Push the parsed flags onto the module-level configuration constants.

    The training functions below read those constants directly, so this is
    what makes a flag and its default interchangeable.
    """
    global IMAGES_DIR, SPLITS_DIR, OUTPUT_DIR
    global BEST_MODEL_PATH, GOOD_EPOCHS_DIR
    global BATCH_SIZE, NUM_EPOCHS, PATIENCE
    global LEARNING_RATE, WEIGHT_DECAY, POSE_CLASS_WEIGHTS

    IMAGES_DIR = args.images
    SPLITS_DIR = args.splits_dir

    OUTPUT_DIR = args.output_dir
    BEST_MODEL_PATH = OUTPUT_DIR / args.checkpoint_name
    GOOD_EPOCHS_DIR = OUTPUT_DIR / "good_epochs"

    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    PATIENCE = args.patience
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay

    POSE_CLASS_WEIGHTS = [
        float(w) for w in str(args.pose_weights).split(",") if w.strip()
    ]

    if len(POSE_CLASS_WEIGHTS) != NUM_POSE_CLASSES:
        raise SystemExit(
            f"--pose-weights needs {NUM_POSE_CLASSES} comma-separated "
            f"values, got: {args.pose_weights}"
        )


def main(argv=None):
    apply_args(parse_args(argv))

    script_start_time = time.perf_counter()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    GOOD_EPOCHS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print_device_info(device)

    use_amp = device.type == "cuda"

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("=" * 70)
    print("CREATING DATASETS")
    print("=" * 70)

    dataset_start_time = time.perf_counter()

    (
        train_dataset,
        val_dataset,
        test_dataset,
        _,
    ) = create_datasets(
        images_dir=IMAGES_DIR,
        splits_dir=SPLITS_DIR,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    dataset_time = (
        time.perf_counter()
        - dataset_start_time
    )

    print()
    print(f"Dataset setup time: {format_time(dataset_time)}")

    print()
    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")
    print(f"Test:  {len(test_dataset)}")

    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("Loading pretrained ViT...")

    model_start_time = time.perf_counter()

    model = MultiTaskFishViT(
        model_name=MODEL_NAME,
        num_direction_classes=NUM_DIRECTION_CLASSES,
        num_pose_classes=NUM_POSE_CLASSES,
    )

    model = model.to(device)

    synchronize_device(device)

    model_load_time = (
        time.perf_counter()
        - model_start_time
    )

    # Verify actual model location
    model_device = next(model.parameters()).device

    print(f"Model device: {model_device}")

    if model_device.type == "cuda":
        print(
            ">>> CONFIRMED: MODEL PARAMETERS ARE ON THE NVIDIA GPU <<<"
        )

    print_gpu_memory("After model load")

    print(
        f"Model load time: "
        f"{format_time(model_load_time)}"
    )

    print()

    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------

    criterion = MultiTaskLoss(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float("inf")

    epochs_without_improvement = 0
    best_epoch = 0

    good_epoch_paths = []

    total_train_compute_time = 0.0
    total_validation_time = 0.0

    training_start_time = time.perf_counter()

    print("=" * 70)
    print("TRAINING START")
    print("=" * 70)

    print(f"Batch size:          {BATCH_SIZE}")
    print(f"Max epochs:          {NUM_EPOCHS}")
    print(f"Early-stop patience: {PATIENCE}")
    print(f"Learning rate:       {LEARNING_RATE}")
    print(f"Pose weights:        {POSE_CLASS_WEIGHTS}")
    print(f"Mixed precision:     {use_amp}")
    print(f"Good epochs dir:     {GOOD_EPOCHS_DIR}")

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        epoch_total_start_time = time.perf_counter()

        print()
        print("=" * 70)
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print("=" * 70)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        total_train_compute_time += train_metrics["time"]

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        total_validation_time += val_metrics["time"]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        print_metrics(
            "TRAIN",
            train_metrics,
        )

        print_metrics(
            "VALIDATION",
            val_metrics,
        )

        print_gpu_memory(
            f"Epoch {epoch}"
        )

        # ----------------------------------------------------
        # Save every good epoch
        # ----------------------------------------------------

        if val_metrics["loss"] < best_val_loss:

            previous_best = best_val_loss

            best_val_loss = val_metrics["loss"]
            best_epoch = epoch

            epochs_without_improvement = 0

            checkpoint = create_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_metrics=val_metrics,
                train_metrics=train_metrics,
            )

            # Unique checkpoint for this good epoch
            good_epoch_path = (
                GOOD_EPOCHS_DIR
                / (
                    f"epoch_{epoch:03d}_"
                    f"val_loss_{best_val_loss:.4f}.pt"
                )
            )

            save_start_time = time.perf_counter()

            torch.save(
                checkpoint,
                good_epoch_path,
            )

            # Also overwrite the simple "best model" checkpoint.
            torch.save(
                checkpoint,
                BEST_MODEL_PATH,
            )

            save_time = (
                time.perf_counter()
                - save_start_time
            )

            good_epoch_paths.append(
                good_epoch_path
            )

            print()
            print("*" * 70)
            print("*** NEW BEST VALIDATION MODEL ***")
            print("*" * 70)

            if previous_best == float("inf"):
                print(
                    f"Validation loss: "
                    f"{best_val_loss:.4f}"
                )
            else:
                improvement = (
                    previous_best
                    - best_val_loss
                )

                print(
                    f"Validation loss: "
                    f"{previous_best:.4f} "
                    f"-> "
                    f"{best_val_loss:.4f}"
                )

                print(
                    f"Improvement:     "
                    f"{improvement:.4f}"
                )

            print()
            print(
                f"Good epoch saved: "
                f"{good_epoch_path}"
            )

            print(
                f"Best model saved: "
                f"{BEST_MODEL_PATH}"
            )

            print(
                f"Checkpoint save time: "
                f"{format_time(save_time)}"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                f"No validation-loss improvement "
                f"for {epochs_without_improvement}/"
                f"{PATIENCE} epochs."
            )

        # ----------------------------------------------------
        # Epoch timing summary
        # ----------------------------------------------------

        synchronize_device(device)

        epoch_total_time = (
            time.perf_counter()
            - epoch_total_start_time
        )

        print()
        print("EPOCH TIMING")
        print("-" * 70)

        print(
            f"Training:   "
            f"{format_time(train_metrics['time'])}"
        )

        print(
            f"Validation: "
            f"{format_time(val_metrics['time'])}"
        )

        print(
            f"Total:      "
            f"{format_time(epoch_total_time)}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print()
            print("=" * 70)
            print("EARLY STOPPING TRIGGERED")
            print("=" * 70)

            break

    synchronize_device(device)

    total_training_wall_time = (
        time.perf_counter()
        - training_start_time
    )

    # ========================================================
    # Training summary
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print(f"Best epoch:           {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    print()
    print("TIMING SUMMARY")
    print("-" * 70)

    print(
        f"Pure training time:   "
        f"{format_time(total_train_compute_time)}"
    )

    print(
        f"Validation time:      "
        f"{format_time(total_validation_time)}"
    )

    print(
        f"Training wall time:   "
        f"{format_time(total_training_wall_time)}"
    )

    print()
    print(
        f"Good checkpoints saved: "
        f"{len(good_epoch_paths)}"
    )

    for path in good_epoch_paths:
        print(f"  {path}")

    # ========================================================
    # Final test
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL TEST")
    print("=" * 70)

    print("Loading best checkpoint...")

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Loaded epoch "
        f"{checkpoint['epoch']} "
        f"with validation loss "
        f"{checkpoint['val_loss']:.4f}"
    )

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
    )

    print_metrics(
        "FINAL TEST SET",
        test_metrics,
    )

    print_direction_per_class(
        test_metrics
    )

    print()

    if device.type == "cuda":
        print_gpu_memory("Final")

        print()
        print(
            ">>> TRAINING COMPLETED USING NVIDIA CUDA GPU <<<"
        )

    total_script_time = (
        time.perf_counter()
        - script_start_time
    )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Best model:          "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"Best epoch:          "
        f"{best_epoch}"
    )

    print(
        f"Best val loss:       "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Good models saved:   "
        f"{len(good_epoch_paths)}"
    )

    print(
        f"Final test time:     "
        f"{format_time(test_metrics['time'])}"
    )

    print(
        f"Total script time:   "
        f"{format_time(total_script_time)}"
    )


if __name__ == "__main__":
    main()