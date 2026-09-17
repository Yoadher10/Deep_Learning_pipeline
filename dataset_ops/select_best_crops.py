#!/usr/bin/env python3
"""
Score a folder of crops by visual quality and copy the best ones out.

Phase 1 produces far more crops than are worth labelling by hand, and many are
blurred, near-black or near-duplicates of the frame before. This ranks every
image on sharpness, exposure and contrast, drops exact-duplicate files (by
SHA-256) and optionally anything visually close to a set of reference images,
and copies the top N into the output folder along with a CSV of the scores.

Typical use is narrowing a raw ROI dump down to a labelling batch:

    python dataset_ops/select_best_crops.py \
        --input /path/to/rois --output /path/to/label_batch --top-n 5000

    # also drop anything that looks like these known-bad crops
    python dataset_ops/select_best_crops.py \
        --input /path/to/rois --output /path/to/label_batch \
        --exclude-similar-to blurry.png --exclude-similar-to empty_tank.png

--input and --output are required; this tool is deliberately not wired to
config.py, since it is run on arbitrary ad-hoc folders.
"""

import argparse
import csv
import hashlib
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SIGNATURE_SIZE = (32, 32)


def file_hash(image_path: Path) -> str:
    digest = hashlib.sha256()
    with image_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visual_signature(image_path: Path) -> tuple[float, ...] | None:
    try:
        with Image.open(image_path) as source_image:
            pixels = list(
                source_image.convert("L")
                .resize(SIGNATURE_SIZE, Image.Resampling.LANCZOS)
                .getdata()
            )
    except (OSError, ValueError):
        return None

    mean = sum(pixels) / len(pixels)
    centered = [pixel - mean for pixel in pixels]
    magnitude = math.sqrt(sum(value * value for value in centered))
    if magnitude == 0:
        return tuple(0.0 for _ in centered)
    return tuple(value / magnitude for value in centered)


def signature_distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


def _ac_balance(arr: np.ndarray) -> float:
    """Lag-1 autocorrelation balance between horizontal and vertical directions.

    Returns a value in (-inf, 1].  Values near 1 mean the image has
    isotropic pixel correlations (natural fish texture).  Values near 0 or
    negative indicate strong directional structure such as motion blur,
    scan-line compression artefacts, or vertical/horizontal banding — all
    of which make fish crops unsuitable for classification.

    Formula:
        r_H = mean over rows of  Pearson(row[:-1], row[1:])   (per-row centred)
        r_V = mean over cols of  Pearson(col[:-1], col[1:])   (per-col centred)
        ac_balance = min(r_H, r_V) / max(r_H, r_V)
    """
    # Horizontal: for each row, Pearson corr between left-shifted and right-shifted pixels
    a = arr[:, :-1]
    b = arr[:, 1:]
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    r_h = float(
        ((a_c * b_c).sum(axis=1)
         / np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1) + 1e-9)
         ).mean()
    )

    # Vertical: for each column, Pearson corr between top-shifted and bottom-shifted pixels
    a = arr[:-1, :]
    b = arr[1:, :]
    a_c = a - a.mean(axis=0, keepdims=True)
    b_c = b - b.mean(axis=0, keepdims=True)
    r_v = float(
        ((a_c * b_c).sum(axis=0)
         / np.sqrt((a_c ** 2).sum(axis=0) * (b_c ** 2).sum(axis=0) + 1e-9)
         ).mean()
    )

    return min(r_h, r_v) / (max(r_h, r_v) + 1e-9)


def score_image(image_path: Path) -> dict | None:
    """Return quality metrics for a fish-crop image, or None if unreadable.

    Scoring formula (calibrated on 10 good / 25 bad labelled crops):

        base  = 100 * (0.40 * resolution  +  0.35 * contrast
                       + 0.20 * exposure  +  0.05 * sharpness)

        score = base * max(0.70, ac_balance)

    Components
    ----------
    resolution  = min(1, pixels / 10 000)           — rewards large crops
    contrast    = min(1, pixel_std / 50)             — rewards rich texture
    exposure    = max(0, 1 − |mean − 110| / 110)    — rewards mid-range brightness
    sharpness   = min(1, edge_std / 50)              — edge variation via FIND_EDGES
    ac_balance  = min(rH,rV) / max(rH,rV)           — isotropy of lag-1 autocorrelation;
                  penalises motion blur and scan-line artefacts (multiplier clamped at 0.70)

    Achieved separation on calibration set: 2.67 σ  (GOOD mean 70.6, BAD mean 41.8).
    """
    try:
        with Image.open(image_path) as source_image:
            image = source_image.convert("L")
            width, height = image.size
            arr = np.array(image, dtype=np.float32)
            brightness = float(arr.mean())
            contrast = float(arr.std())
            edge_arr = np.array(image.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
            sharpness = float(edge_arr.std())
    except (OSError, ValueError):
        return None

    pixels = width * height
    resolution_score = min(1.0, pixels / 10_000.0)
    contrast_score = min(1.0, contrast / 50.0)
    exposure_score = max(0.0, 1.0 - abs(brightness - 110.0) / 110.0)
    sharpness_score = min(1.0, sharpness / 50.0)

    base_score = 100.0 * (
        0.40 * resolution_score
        + 0.35 * contrast_score
        + 0.20 * exposure_score
        + 0.05 * sharpness_score
    )

    # Anisotropy penalty: motion blur / banding / scan-lines reduce this below 0.70
    ac_bal = _ac_balance(arr)
    quality_score = base_score * max(0.70, ac_bal)

    return {
        "path": image_path,
        "width": width,
        "height": height,
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "ac_balance": ac_bal,
        "score": quality_score,
    }


def select_photos(
    input_dir: Path,
    output_dir: Path,
    top_n: int,
    reference_paths: list[Path],
    similarity_threshold: float,
) -> None:
    image_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under: {input_dir}")

    results = []
    unreadable_count = 0
    duplicate_count = 0
    similar_count = 0
    seen_hashes = set()
    reference_signatures = []
    for reference_path in reference_paths:
        reference_signature = visual_signature(reference_path)
        if reference_signature is None:
            raise ValueError(f"Reference image cannot be read: {reference_path}")
        reference_signatures.append(reference_signature)
    for image_path in image_paths:
        image_hash = file_hash(image_path)
        if image_hash in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(image_hash)

        metrics = score_image(image_path)
        if metrics is None:
            unreadable_count += 1
            continue

        distance = None
        if reference_signatures:
            image_signature = visual_signature(image_path)
            if image_signature is None:
                unreadable_count += 1
                continue
            distance = min(
                signature_distance(reference_signature, image_signature)
                for reference_signature in reference_signatures
            )
            if distance <= similarity_threshold:
                similar_count += 1
                continue

        metrics["relative_path"] = image_path.relative_to(input_dir)
        metrics["reference_distance"] = distance
        results.append(metrics)

    results.sort(key=lambda item: (-item["score"], str(item["relative_path"]).lower()))
    selected = results[:top_n]

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "quality_scores.csv"
    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(
            [
                "rank",
                "relative_path",
                "score",
                "width",
                "height",
                "sharpness",
                "brightness",
                "contrast",
                "ac_balance",
                "reference_distance",
                "selected",
            ]
        )
        selected_paths = {item["relative_path"] for item in selected}
        for rank, item in enumerate(results, start=1):
            writer.writerow(
                [
                    rank,
                    item["relative_path"],
                    f'{item["score"]:.4f}',
                    item["width"],
                    item["height"],
                    f'{item["sharpness"]:.4f}',
                    f'{item["brightness"]:.4f}',
                    f'{item["contrast"]:.4f}',
                    f'{item["ac_balance"]:.4f}',
                    "" if item["reference_distance"] is None else f'{item["reference_distance"]:.6f}',
                    item["relative_path"] in selected_paths,
                ]
            )

    for item in selected:
        destination = output_dir / item["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["path"], destination)

    print(f"Found {len(image_paths)} image files.")
    print(f"Found {len(results)} unique readable images.")
    print(f"Ignored {duplicate_count} exact duplicate files.")
    if reference_paths:
        print(
            f"Ignored {similar_count} images visually similar to "
            f"{len(reference_paths)} reference images "
            f"(distance <= {similarity_threshold})."
        )
    print(f"Selected {len(selected)} images into: {output_dir}")
    print(f"Wrote full score report to: {report_path}")
    if unreadable_count:
        print(f"Skipped {unreadable_count} unreadable image files.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score images by visual quality and copy the highest-scoring images."
    )
    parser.add_argument("--input", type=Path, required=True, help="Source image folder")
    parser.add_argument("--output", type=Path, required=True, help="Output folder")
    parser.add_argument("--top-n", type=int, default=3000, help="Number of images to keep")
    parser.add_argument(
        "--exclude-similar-to",
        type=Path,
        action="append",
        help="Reference image to exclude; repeat this option for multiple references",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.35,
        help="Maximum normalized pixel distance for exclusion (default: 0.35)",
    )
    args = parser.parse_args()

    if args.top_n <= 0:
        parser.error("--top-n must be greater than zero")
    if not args.input.is_dir():
        parser.error(f"Input folder does not exist: {args.input}")
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output folders must be different")
    reference_paths = args.exclude_similar_to or []
    for reference_path in reference_paths:
        if not reference_path.is_file():
            parser.error(f"Reference image does not exist: {reference_path}")
    if args.similarity_threshold < 0:
        parser.error("--similarity-threshold must not be negative")

    select_photos(
        args.input.resolve(),
        args.output.resolve(),
        args.top_n,
        [reference_path.resolve() for reference_path in reference_paths],
        args.similarity_threshold,
    )


if __name__ == "__main__":
    main()
