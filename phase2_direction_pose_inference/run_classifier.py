#!/usr/bin/env python3

"""
Phase 2: run the fine-tuned multi-task Fish ViT over every crop in the ROIs
folder and write one prediction JSON per crop:

    {
      "image": "frame_000123_fish_00.png",
      "direction": 4,
      "direction_name": "SE",
      "direction_confidence": 0.91,
      "no_fish_confidence": 0.01,
      "pose": 1,                 # null when direction is No Fish / N / S
      "pose_name": "Upside Down",
      "pose_confidence": 0.97
    }

Aggregate distribution / upside-down statistics are produced separately by
phase3_analysis/summarize_predictions.py.

Crops whose JSON already exists are skipped, so the script can safely resume.
"""

import sys
import json
import argparse
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config
from torch_device import select_device, describe

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ViTModel


# ============================================================
# Paths / configuration (see config.py; override via CLI flags)
# ============================================================

IMAGES_DIR = config.ROIS_DIR
MODEL_PATH = config.VIT_WEIGHTS
LABELS_DIR = config.PREDICTIONS_DIR

BATCH_SIZE = config.INFERENCE_BATCH_SIZE

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
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


POSE_NAMES = {
    0: "Regular",
    1: "Upside Down",
}


# Pose is N/A for these directions according to your labeling scheme
POSE_NA_DIRECTIONS = {
    0,  # No Fish
    1,  # N
    5,  # S
}


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

        self.vit = ViTModel.from_pretrained(
            model_name
        )

        hidden_size = self.vit.config.hidden_size

        self.dropout = nn.Dropout(
            self.vit.config.hidden_dropout_prob
        )

        self.direction_head = nn.Linear(
            hidden_size,
            num_direction_classes,
        )

        self.pose_head = nn.Linear(
            hidden_size,
            num_pose_classes,
        )

    def forward(self, pixel_values):

        outputs = self.vit(
            pixel_values=pixel_values
        )

        cls_embedding = (
            outputs.last_hidden_state[:, 0]
        )

        cls_embedding = self.dropout(
            cls_embedding
        )

        direction_logits = self.direction_head(
            cls_embedding
        )

        pose_logits = self.pose_head(
            cls_embedding
        )

        return direction_logits, pose_logits


# ============================================================
# Dataset
# ============================================================

class FishInferenceDataset(Dataset):

    def __init__(
        self,
        image_paths,
        processor,
    ):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        processed = self.processor(
            images=image,
            return_tensors="pt",
        )

        return {
            "pixel_values":
                processed["pixel_values"].squeeze(0),

            "image_path":
                str(image_path),
        }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("FISH ViT INFERENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not IMAGES_DIR.exists():

        raise FileNotFoundError(
            f"Image folder not found:\n"
            f"{IMAGES_DIR}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    LABELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    config.emit_step("scanning crops")
    all_images = sorted(
        p
        for p in IMAGES_DIR.iterdir()
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    print(
        f"Images found: {len(all_images)}"
    )

    # --------------------------------------------------------
    # Skip images already inferred
    # --------------------------------------------------------

    remaining_images = []

    already_done = 0

    for image_path in all_images:

        output_json = (
            LABELS_DIR
            / f"{image_path.stem}.json"
        )

        if output_json.exists():
            already_done += 1
        else:
            remaining_images.append(image_path)

    print(
        f"Already inferred: {already_done}"
    )

    print(
        f"Remaining:        "
        f"{len(remaining_images)}"
    )

    if not remaining_images:

        print()
        print("Everything is already inferred.")
        return

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = select_device()
    short, pretty = describe(device)
    config.emit_device(short, f"phase 2 ViT on {pretty}")
    print()
    print(f"Device: {pretty}")

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    config.emit_step("loading ViT checkpoint (~350 MB)")
    print("Loading checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model_name = checkpoint.get(
        "model_name",
        "google/vit-base-patch16-224-in21k",
    )

    num_direction_classes = (
        checkpoint.get(
            "num_direction_classes",
            9,
        )
    )

    num_pose_classes = (
        checkpoint.get(
            "num_pose_classes",
            2,
        )
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', '?')}"
    )

    print(
        f"Model: {model_name}"
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    config.emit_step("loading image processor")
    processor = (
        AutoImageProcessor.from_pretrained(
            model_name
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    config.emit_step("building ViT and loading weights")
    model = MultiTaskFishViT(
        model_name=model_name,
        num_direction_classes=
            num_direction_classes,
        num_pose_classes=
            num_pose_classes,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    config.emit_step(f"moving model to {short}")
    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Dataset / loader
    # --------------------------------------------------------

    dataset = FishInferenceDataset(
        remaining_images,
        processor,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    total = len(dataset)
    completed = 0

    print()
    print("=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70)
    config.emit_step(f"classifying {total} crops ({short})")
    config.emit_progress(0, total, "classifying")

    with torch.inference_mode():

        for batch in loader:

            pixel_values = (
                batch["pixel_values"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            image_paths = (
                batch["image_path"]
            )

            # -----------------------------------------------
            # Forward
            # -----------------------------------------------

            if device.type == "cuda":

                with torch.amp.autocast(
                    "cuda"
                ):

                    (
                        direction_logits,
                        pose_logits,
                    ) = model(
                        pixel_values
                    )

            else:

                (
                    direction_logits,
                    pose_logits,
                ) = model(
                    pixel_values
                )

            # -----------------------------------------------
            # Probabilities
            # -----------------------------------------------

            direction_probs = (
                torch.softmax(
                    direction_logits,
                    dim=1,
                )
            )

            pose_probs = (
                torch.softmax(
                    pose_logits,
                    dim=1,
                )
            )

            (
                direction_confidences,
                direction_predictions,
            ) = direction_probs.max(
                dim=1
            )

            (
                pose_confidences,
                pose_predictions,
            ) = pose_probs.max(
                dim=1
            )

            # -----------------------------------------------
            # Save JSON per image
            # -----------------------------------------------

            for i, path_string in enumerate(
                image_paths
            ):

                image_path = Path(
                    path_string
                )

                direction = int(
                    direction_predictions[
                        i
                    ].item()
                )

                direction_confidence = float(
                    direction_confidences[
                        i
                    ].item()
                )

                # Probability assigned specifically to class 0,
                # regardless of which direction class won.
                no_fish_confidence = float(
                    direction_probs[
                        i, 0
                    ].item()
                )

                raw_pose = int(
                    pose_predictions[
                        i
                    ].item()
                )

                raw_pose_confidence = float(
                    pose_confidences[
                        i
                    ].item()
                )

                # -------------------------------------------
                # Pose is not applicable for No Fish / N / S
                # -------------------------------------------

                if direction in POSE_NA_DIRECTIONS:
                    pose = None
                    pose_name = "N/A"
                    pose_confidence = None
                else:
                    pose = raw_pose
                    pose_name = POSE_NAMES[raw_pose]
                    pose_confidence = raw_pose_confidence

                # -------------------------------------------
                # JSON
                # -------------------------------------------

                result = {
                    "image": image_path.name,
                    "direction": direction,
                    "direction_name": DIRECTION_NAMES[direction],
                    "direction_confidence": direction_confidence,
                    "no_fish_confidence": no_fish_confidence,
                    "pose": pose,
                    "pose_name": pose_name,
                    "pose_confidence": pose_confidence,
                    # raw pose head output, kept even when pose is N/A
                    "raw_pose_prediction": raw_pose,
                    "raw_pose_name": POSE_NAMES[raw_pose],
                    "raw_pose_confidence": raw_pose_confidence,
                }

                output_json = (
                    LABELS_DIR
                    / f"{image_path.stem}.json"
                )

                with open(
                    output_json,
                    "w",
                    encoding="utf-8",
                ) as f:

                    json.dump(
                        result,
                        f,
                        indent=2,
                    )

            completed += len(image_paths)
            config.emit_progress(completed, total, "classifying")

    print()
    print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Predictions saved to:\n"
        f"{LABELS_DIR}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: run the fish ViT over crops and write prediction JSON"
    )
    parser.add_argument("--images", type=Path, default=IMAGES_DIR,
                        help="Folder of fish crops (default: config.ROIS_DIR)")
    parser.add_argument("--weights", type=Path, default=MODEL_PATH,
                        help="ViT checkpoint (default: config.VIT_WEIGHTS)")
    parser.add_argument("--output", type=Path, default=LABELS_DIR,
                        help="Folder to write prediction JSON (default: config.PREDICTIONS_DIR)")
    args = parser.parse_args()

    IMAGES_DIR = args.images
    MODEL_PATH = args.weights
    LABELS_DIR = args.output

    main()