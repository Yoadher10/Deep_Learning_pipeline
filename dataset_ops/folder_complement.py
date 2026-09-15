#!/usr/bin/env python3
"""
Copy every file from SOURCE that is NOT present (by filename) in SUBTRACT
into OUTPUT, preserving the directory structure.

Used while building the training set, e.g. "all crops minus the 5000 already
labelled" or "training crops minus the 400 no-fish crops".

    python folder_complement.py --source SRC --subtract SUB --output OUT
"""

import argparse
import shutil
from pathlib import Path


def folder_complement(source_dir: Path, subtract_dir: Path, output_dir: Path) -> None:
    subtract_names = {
        file.name for file in subtract_dir.rglob("*") if file.is_file()
    }
    print(f"Files in subtract folder: {len(subtract_names)}")

    if output_dir.exists():
        print(f"Removing existing output folder: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    copied = 0
    removed = 0
    for source_file in source_dir.rglob("*"):
        if not source_file.is_file():
            continue
        if source_file.name in subtract_names:
            removed += 1
            continue

        destination = output_dir / source_file.relative_to(source_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)
        copied += 1

    print()
    print("Done.")
    print(f"Excluded (present in subtract folder): {removed}")
    print(f"Copied to new folder:                 {copied}")
    print(f"Output folder:                        {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Folder to copy FROM")
    parser.add_argument("--subtract", type=Path, required=True,
                        help="Folder whose filenames should be excluded")
    parser.add_argument("--output", type=Path, required=True, help="Folder to create")
    args = parser.parse_args()

    folder_complement(args.source, args.subtract, args.output)
