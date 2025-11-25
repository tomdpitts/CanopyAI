#!/usr/bin/env python3
"""
prepare_data.py — Prepare TCD dataset for Detectree2 training.


Important: must be run without --already_downloaded at least once to create metadata files.

Steps:
1. Download or reuse TCD tiles (.tif)
2. Save metadata into <tile>_meta.json
3. Convert COCO annotation (from HF) → polygon GeoDataFrame
4. Tile all orthomosaics with crowns into training chips

This script produces:

data/tcd/raw/
    tcd_tile_0.tif
    tcd_tile_0_meta.json
    ...
data/tcd/chips/
    tcd_tile_0_chips/
        chip_000.tif
        chip_001.tif
        ...
"""

import argparse
from pathlib import Path
import random
import shutil
from utils import download_tcd_tiles_streaming, tile_all_tcd_tiles

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare TCD dataset for Detectree2 training."
    )
    p.add_argument(
        "--already_downloaded",
        action="store_true",
        help=("Skip HuggingFace download and use existing .tif files in data/tcd/raw/"),
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=20,
        help=("Number of TCD tiles to download (ignored if already_downloaded=True)"),
    )
    p.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help=(
            "Fraction of data to use for training "
            "(default: 0.8 = 80%% train, 20%% test)"
        ),
    )
    return p.parse_args()


# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------


def main():
    args = parse_args()

    # === 1. Define directories ===
    data_root = Path("data/tcd")
    raw_dir = data_root / "raw"
    raw_test_dir = data_root / "raw_test"
    chips_root = data_root / "chips"

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_test_dir.mkdir(parents=True, exist_ok=True)
    chips_root.mkdir(parents=True, exist_ok=True)

    # === 2. Download or reuse tiles ===
    if not args.already_downloaded:
        print("🌐 Downloading TCD tiles via HuggingFace...")
        download_tcd_tiles_streaming(save_dir=raw_dir, max_images=args.max_images)
    else:
        print("⏭️ Using existing .tif tiles in raw/")

    # === 3. Create metadata JSON if missing ===
    print("\n📝 Writing metadata JSON files (if missing)...")

    tif_files = sorted(raw_dir.glob("tcd_tile_*.tif"))
    for tif_path in tif_files:
        meta_path = tif_path.with_name(tif_path.stem + "_meta.json")
        if meta_path.exists():
            print(f"  ↪ {meta_path.name} already exists.")
            continue

        print(f"  🏷️ Creating metadata: {meta_path.name}")

        # Extract metadata directly from the downloaded HF example
        # If not downloaded in this run, metadata is missing, so we load from utils helper
        # → load_tcd_meta_for_tile is unnecessary because infer.py uses meta.json for inference
        raise RuntimeError(
            f"{meta_path} does not exist — but metadata creation "
            "requires image_info from HuggingFace streaming. "
            "Please re-run without --already_downloaded at least once."
        )

    # === 3b. Split into train/test sets ===
    train_pct = args.train_split
    test_pct = 1 - args.train_split
    print(f"\n📊 Splitting data: {train_pct:.0%} train, {test_pct:.0%} test...")

    # Get all tiles (both .tif and _meta.json)
    all_tif_files = sorted(raw_dir.glob("tcd_tile_*.tif"))

    if len(all_tif_files) == 0:
        print("⚠️ No tiles found to split.")
    else:
        # Set random seed for reproducibility
        random.seed(42)

        # Shuffle and split
        shuffled_files = all_tif_files.copy()
        random.shuffle(shuffled_files)

        n_train = int(len(shuffled_files) * args.train_split)
        train_files = shuffled_files[:n_train]
        test_files = shuffled_files[n_train:]

        # Move test files to raw_test/
        for tif_path in test_files:
            meta_path = tif_path.with_name(tif_path.stem + "_meta.json")

            # Move .tif
            dest_tif = raw_test_dir / tif_path.name
            shutil.move(str(tif_path), str(dest_tif))

            # Move _meta.json if it exists
            if meta_path.exists():
                dest_meta = raw_test_dir / meta_path.name
                shutil.move(str(meta_path), str(dest_meta))

        print("✅ Split complete:")
        print(f"   Training: {len(train_files)} tiles in {raw_dir}")
        print(f"   Test: {len(test_files)} tiles in {raw_test_dir}")

    # === 4. Tile all orthomosaics (training tiles with crowns) ===
    print("\n🧩 Tiling training tiles for training...")
    tile_all_tcd_tiles(raw_dir, chips_root)

    print("\n🎉 Dataset preparation complete!")
    print(f"Training chips are in: {chips_root}")
    print(f"Test tiles (raw) are in: {raw_test_dir}")


# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()
