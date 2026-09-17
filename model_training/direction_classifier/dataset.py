#!/usr/bin/env python3
"""
Shared torch Dataset for the multi-task (direction + pose) Fish ViT.

Imported by model_training/direction_classifier/train.py and by
evaluation/validation_error_analysis.py; it is not a CLI entry point, so its
paths come from config.py rather than from flags. Callers that need different
folders pass them to ``create_datasets(images_dir=..., splits_dir=...)`` —
that is how those two scripts forward their own --images / --splits-dir flags.

Running this file directly performs a small sanity check: it builds the three
splits and prints the shape and labels of one training sample.
"""

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor


# ============================================================
# Configuration (see config.py)
# ============================================================

IMAGES_DIR = config.TRAINSET_IMAGES
SPLITS_DIR = config.SPLITS_DIR

TRAIN_CSV = config.TRAIN_CSV
VAL_CSV = config.VAL_CSV
TEST_CSV = config.TEST_CSV

MODEL_NAME = config.VIT_MODEL_NAME


# ============================================================
# Horizontal-flip direction remapping
# ============================================================
#
# Screen-relative directions (N = up, E = right):
#   0 No Fish, 1 N, 2 NE, 3 E, 4 SE, 5 S, 6 SW, 7 W, 8 NW
#
# A horizontal (left-right) flip swaps E<->W and leaves the
# N/S component untouched.
HFLIP_DIRECTION_MAP = {
    0: 0,  # No Fish
    1: 1,  # N
    2: 8,  # NE -> NW
    3: 7,  # E  -> W
    4: 6,  # SE -> SW
    5: 5,  # S
    6: 4,  # SW -> SE
    7: 3,  # W  -> E
    8: 2,  # NW -> NE
}


# ============================================================
# Dataset
# ============================================================

class FishViTDataset(Dataset):
    def __init__(
        self,
        csv_path,
        images_dir,
        image_processor,
        train=False,
    ):
        self.df = pd.read_csv(csv_path)

        self.images_dir = Path(images_dir)
        self.image_processor = image_processor
        self.train = train

        # ----------------------------------------------------
        # Training augmentation
        # ----------------------------------------------------
        #
        # Horizontal flip IS used, applied on-the-fly here
        # (not materialized to disk) and only for the train
        # split. When an image is flipped we mirror the
        # direction label with HFLIP_DIRECTION_MAP. Pose is
        # unchanged by a horizontal flip (a belly-down fish
        # stays belly-down).
        #
        # This is lighting-safe: underwater scenes are lit
        # top-to-bottom, and a horizontal flip preserves that
        # gradient. Vertical flip is still forbidden - it
        # inverts the lighting gradient and the model learns
        # "inverted lighting = upside-down" instead of body
        # orientation (this was the v6 failure).
        #
        # Bonus: flipping spreads the scarce upside-down
        # examples across mirrored directions (NE<->NW,
        # E<->W, SE<->SW).
        # ----------------------------------------------------

        self.hflip_prob = 0.5 if train else 0.0

        self.train_augmentation = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),

            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3,
                        sigma=(0.1, 1.0),
                    )
                ],
                p=0.10,
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.images_dir / row["image"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Always RGB for ViT
        image = Image.open(image_path).convert("RGB")

        direction_target = int(row["direction_target"])
        pose_target = int(row["pose_target"])

        # Training-only augmentation
        if self.train:
            if torch.rand(1).item() < self.hflip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                direction_target = HFLIP_DIRECTION_MAP[direction_target]

            image = self.train_augmentation(image)

        # Hugging Face ViT preprocessing:
        # resize
        # rescale
        # normalize
        processed = self.image_processor(
            images=image,
            return_tensors="pt",
        )

        # Processor adds a batch dimension:
        #
        # [1, 3, 224, 224]
        #
        # Dataset should return:
        #
        # [3, 224, 224]
        pixel_values = processed["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,

            # Direction:
            # 0..8
            "direction": torch.tensor(
                direction_target,
                dtype=torch.long,
            ),

            # Pose:
            # -100 = ignore
            # 0    = Regular
            # 1    = Upside Down
            "pose": torch.tensor(
                pose_target,
                dtype=torch.long,
            ),

            # Useful later for debugging/results
            "image_name": row["image"],
        }


# ============================================================
# Factory function
# ============================================================

def create_datasets(
    model_name=MODEL_NAME,
    images_dir=None,
    splits_dir=None,
):
    """
    Build the train / val / test datasets and the shared image processor.

    ``images_dir`` and ``splits_dir`` default to config.TRAINSET_IMAGES and
    config.SPLITS_DIR; scripts with their own --images / --splits-dir flags
    pass those values through here. ``splits_dir`` is expected to contain
    train.csv, val.csv and test.csv.
    """
    images_dir = Path(images_dir) if images_dir else IMAGES_DIR
    splits_dir = Path(splits_dir) if splits_dir else SPLITS_DIR

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    image_processor = AutoImageProcessor.from_pretrained(
        model_name
    )

    train_dataset = FishViTDataset(
        csv_path=train_csv,
        images_dir=images_dir,
        image_processor=image_processor,
        train=True,
    )

    val_dataset = FishViTDataset(
        csv_path=val_csv,
        images_dir=images_dir,
        image_processor=image_processor,
        train=False,
    )

    test_dataset = FishViTDataset(
        csv_path=test_csv,
        images_dir=images_dir,
        image_processor=image_processor,
        train=False,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        image_processor,
    )


# ============================================================
# Sanity test
# ============================================================

if __name__ == "__main__":
    (
        train_dataset,
        val_dataset,
        test_dataset,
        image_processor,
    ) = create_datasets()

    print("=" * 70)
    print("DATASET CHECK")
    print("=" * 70)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    print()

    sample = train_dataset[0]

    print("Example sample:")
    print(f"Image:     {sample['image_name']}")
    print(f"Direction: {sample['direction'].item()}")
    print(f"Pose:      {sample['pose'].item()}")
    print(
        f"Tensor:    "
        f"{tuple(sample['pixel_values'].shape)}"
    )
    print(
        f"dtype:     "
        f"{sample['pixel_values'].dtype}"
    )

    print()
    print("Pixel range:")
    print(
        f"min = "
        f"{sample['pixel_values'].min().item():.4f}"
    )
    print(
        f"max = "
        f"{sample['pixel_values'].max().item():.4f}"
    )

    print()
    print("Dataset successfully loaded.")