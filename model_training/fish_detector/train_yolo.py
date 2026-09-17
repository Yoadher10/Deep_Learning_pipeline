#!/usr/bin/env python3
"""
Train the YOLOv8 fish detector used by phase 1.

Ultralytics needs a data.yaml describing the train/val image folders and the
class names. Copy data.yaml.example to data.yaml next to this script and edit
it, or point --data somewhere else.

Usage:
    python model_training/fish_detector/train_yolo.py
    python model_training/fish_detector/train_yolo.py --data /path/to/data.yaml
    python model_training/fish_detector/train_yolo.py --imgsz 1024 --batch 8

Weights land in Ultralytics' own runs/ folder; the path is printed at the end.
Copy the best.pt over phase1_fish_detection/weights/ to use it in the pipeline.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# YOLO dataset definition. Ultralytics needs a data.yaml pointing at the
# train/val image folders and class names. Keep it next to this script,
# or pass --data. A template lives at data.yaml.example.
DEFAULT_DATA_YAML = Path(__file__).resolve().parent / "data.yaml"


def main(data_yaml=DEFAULT_DATA_YAML, model="yolov8s.pt", imgsz=1280,
         batch=16, epochs=300, patience=30, device=None, workers=8,
         cache=False):
    # yolov8s (small) over nano: the fish are small objects (~60 px in a
    # 1920x1080 frame) and nano lacks the capacity to resolve them.
    net = YOLO(model)

    results = net.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,      # frames are 1920x1080; 640 shrank a 60 px fish to
                          # ~20 px. 1280 keeps it ~40 px.
                          # NOTE: run inference at the same imgsz.
        batch=batch,      # 8 GB VRAM (e.g. RTX 2060S) can't fit s@1280 b16
                          # (~13.5 GB). Use --batch 4, or --imgsz 1024 --batch 8.
        device=device,    # e.g. 0 for first GPU, or "cpu"
        workers=workers,  # lower to 4 if the pin-memory thread OOMs
        cache=cache,
        name="custom_marine_model",
    )

    save_dir = str(results.save_dir)
    print(f"Training complete! Results saved to {save_dir}")
    print(f"Best weights: {save_dir}/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the YOLOv8 fish detector")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML,
                        help="Path to the Ultralytics data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Base weights")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16,
                        help="Lower for small GPUs. 8 GB VRAM: try 4 at imgsz 1280.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default=None, help='GPU index e.g. 0, or "cpu"')
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM/disk")
    args = parser.parse_args()
    main(args.data, args.model, args.imgsz, args.batch, args.epochs,
         args.patience, args.device, args.workers, args.cache)
