#!/usr/bin/env python3
"""
repair_validation_folds.py — Back-fill missing validation_fold in meta.json files.

Run this once after download_tcd_all.py if some tiles were downloaded before
the validation_fold field was added to save_tile(). Streams HF metadata only
(no image bytes) so runs much faster than a full re-download.

Usage:
    source venv310/bin/activate
    python phase23/repair_validation_folds.py
"""

import json
from pathlib import Path
from datasets import load_dataset, Image as HFImage
from tqdm import tqdm

TRAIN_DIR = Path("data/tcd/images/data/tcd/raw")

ds = load_dataset("restor/tcd", split="train", streaming=True).cast_column(
    "image", HFImage(decode=False)
)

fixed = skipped = 0
for i, item in enumerate(tqdm(ds, total=4169, desc="Repairing folds")):
    meta_path = TRAIN_DIR / f"tcd_tile_{i}_meta.json"
    if not meta_path.exists() or meta_path.stat().st_size == 0:
        skipped += 1
        continue
    meta = json.loads(meta_path.read_text())
    if meta.get("validation_fold") is None:
        meta["validation_fold"] = item.get("validation_fold")
        meta_path.write_text(json.dumps(meta))
        fixed += 1

print(f"\nFixed {fixed} meta.json files  |  Skipped {skipped} (missing/empty)")
