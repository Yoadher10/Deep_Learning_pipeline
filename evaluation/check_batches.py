import sys
import pandas as pd
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config

VAL_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else config.VAL_CSV

BATCH_SIZE = 16

df = pd.read_csv(VAL_CSV)

print(f"Validation samples: {len(df)}")
print(f"Batch size: {BATCH_SIZE}")
print()

bad_batches = []

for start in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[start:start + BATCH_SIZE]

    batch_number = start // BATCH_SIZE + 1

    valid_pose_count = (batch["pose_target"] != -100).sum()
    ignored_pose_count = (batch["pose_target"] == -100).sum()

    if valid_pose_count == 0:
        bad_batches.append(batch_number)

        print("=" * 70)
        print(f"BATCH {batch_number} HAS ZERO VALID POSE LABELS")
        print("=" * 70)

        print(
            batch[
                [
                    "image",
                    "direction_target",
                    "pose_target"
                ]
            ].to_string(index=False)
        )

        print()

print("=" * 70)
print("RESULT")
print("=" * 70)

if bad_batches:
    print(
        f"Found {len(bad_batches)} batches with "
        f"zero valid pose samples:"
    )
    print(bad_batches)
else:
    print("No all-ignored pose batches found.")
    print("Then we need to investigate another cause.")