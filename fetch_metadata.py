#!/usr/bin/env python3
"""
fetch_metadata.py — Retrieve TCD metadata for existing local tiles.

This script streams the TCD dataset from HuggingFace, applies the same
"is_rangeland" filtering used during the original data download, and
saves the corresponding *_meta.json files for any tcd_tile_*.tif files
found in the target directory.

Usage:
    python fetch_metadata.py --data_dir data/tcd/images/data/tcd/raw_test
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Inline the filtering logic from utils.py to avoid importing detectree2 transitively.
# These lists must match utils.py exactly.
arid_rangeland = [
    "Sechura desert",
    "Sonoran desert",
    "Gulf of Oman desert and semi-desert",
    "Chilean matorral",
    "Central Mexican matorral",
    "Low Monte",
    "Zambezian and Mopane woodlands",
    "East Sudanian savanna",
    "West Sudanian savanna",
]

def is_rangeland(x):
    return x.get("biome_name") in arid_rangeland

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch metadata for existing TCD tiles")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/tcd/images/data/tcd/raw_test",
        help="Directory containing tcd_tile_*.tif files",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    # 1. Identify which tiles we have locally
    # Local files are named tcd_tile_{INDEX}.tif
    existing_tiles = list(data_dir.glob("tcd_tile_*.tif"))
    if not existing_tiles:
        print(f"⚠️ No tcd_tile_*.tif files found in {data_dir}")
        return

    print(f"📂 Found {len(existing_tiles)} local tiles in {data_dir}")

    # Parse indices from filenames
    needed_indices = set()
    max_index = -1
    
    for p in existing_tiles:
        try:
            # stem is like "tcd_tile_45"
            parts = p.stem.split("_")
            idx = int(parts[-1])
            needed_indices.add(idx)
            max_index = max(max_index, idx)
        except (ValueError, IndexError):
            print(f"⚠️ Skipping non-standard filename: {p.name}")

    if not needed_indices:
        print("❌ No valid tile indices found.")
        return

    print(f"📊 Highest tile index needed: {max_index}")
    print("🌐 Streaming TCD dataset to recover metadata...")

    # 2. Stream dataset and match indices
    # We must replicate the EXACT filtering order:
    # 1. Load stream
    # 2. Filter by is_rangeland
    # 3. enumerate gives us the index 0, 1, 2... matching tcd_tile_0, tcd_tile_1...
    
    # Disable auto PIL decoding to avoid JPEG/TIFF crash (same fix as utils.download_tcd_tiles_streaming)
    from datasets import Image as HFImage
    ds = load_dataset("restor/tcd", split="train", streaming=True).cast_column("image", HFImage(decode=False))
    
    # We must replicate the 'count' variable logic from utils.download_tcd_tiles_streaming
    stream_idx = 0
    saved_count = 0 
    
    # Iterate through dataset stream
    for i, item in enumerate(tqdm(ds, desc="Scanning dataset")):
        
        # Apply strict filtering: only rangeland tiles are counted
        if not is_rangeland(item):
            continue

        # item is now the `stream_idx`-th rangeland tile
        if stream_idx in needed_indices:
            # Need metadata for this one
            meta_path = data_dir / f"tcd_tile_{stream_idx}_meta.json"
            
            if not meta_path.exists():
                # Construct metadata payload (same as utils.py)
                try:
                    meta = {
                        "image_id": item["image_id"],
                        "bounds": item["bounds"],
                        "crs": str(item["crs"]),
                        "coco_annotations": item.get("coco_annotations", []),
                        "biome": item.get("biome"),
                        "biome_name": item.get("biome_name"),
                        "country": item.get("country"),
                    }
                    
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f)
                        
                    saved_count += 1
                except Exception as e:
                    print(f"❌ Error saving metadata for tile {stream_idx}: {e}")

        # Check if we can stop early
        if stream_idx >= max_index:
            print(f"\n🏁 Reached max needed index ({max_index}). Stopping.")
            break
            
        stream_idx += 1

    print(f"\n🎉 Metadata recovery complete.")
    print(f"   Restored {saved_count} missing metadata files.")

if __name__ == "__main__":
    main()
