#!/usr/bin/env python3
"""
download_tcd_holdout.py — Download the TCD holdout test split (439 tiles).

This is the paper's 10% biome-stratified test set, used ONLY for final
benchmarking against the Restor TCD paper results. Do NOT use these tiles
for training or early stopping.

Run this AFTER training is complete.

Output:
    data/tcd/images/data/tcd/test/
        tcd_test_tile_{N}.tif
        tcd_test_tile_{N}_meta.json   (includes coco_annotations)

Usage:
    source venv310/bin/activate
    python phase23/download_tcd_holdout.py
    python phase23/download_tcd_holdout.py --skip-existing   # resume
    python phase23/download_tcd_holdout.py --max-tiles 10    # quick test
"""

import argparse
import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds

OUT_DIR = Path("data/tcd/images/data/tcd/test")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",       default=str(OUT_DIR))
    ap.add_argument("--max-tiles",     type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    return ap.parse_args()


def save_tile(item: dict, img_path: Path, meta_path: Path):
    img_bytes = item["image"]["bytes"]
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image as PILImage
        pil_img = PILImage.open(BytesIO(img_bytes))
        img = np.array(pil_img)
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    bounds    = item["bounds"]
    crs       = item["crs"]
    transform = from_bounds(*bounds, width=w, height=h)

    with rasterio.open(img_path, "w", driver="GTiff",
                       height=h, width=w, count=3, dtype=img.dtype,
                       crs=crs, transform=transform) as dst:
        for b in range(3):
            dst.write(img[:, :, b], b + 1)

    meta = {
        "image_id":         item.get("image_id"),
        "bounds":           bounds,
        "crs":              str(crs),
        "width":            w,
        "height":           h,
        "coco_annotations": item.get("coco_annotations") or [],
        "biome":            item.get("biome"),
        "biome_name":       item.get("biome_name"),
        "country":          item.get("country"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset, Image as HFImage
    print("📦 Loading TCD holdout test split (439 tiles) ...")
    ds = load_dataset("restor/tcd", split="test", streaming=True).cast_column(
        "image", HFImage(decode=False)
    )

    saved = 0
    for i, item in enumerate(ds):
        if args.max_tiles is not None and saved >= args.max_tiles:
            break

        img_path  = out_dir / f"tcd_test_tile_{i}.tif"
        meta_path = out_dir / f"tcd_test_tile_{i}_meta.json"

        if args.skip_existing and img_path.exists() and meta_path.stat().st_size > 0:
            print(f"  ⏭  tcd_test_tile_{i} exists, skipping")
            continue

        biome  = item.get("biome_name", "unknown")
        img_id = item.get("image_id", "?")
        print(f"  [{i}] image_id={img_id}  biome={biome} ...", end=" ", flush=True)
        try:
            save_tile(item, img_path, meta_path)
            print("✓")
            saved += 1
        except Exception as e:
            print(f"✗  ({e})")

    total = sum(1 for _ in out_dir.glob("tcd_test_tile_*.tif"))
    print(f"\n✅ Done. {saved} new tiles saved ({total} total) → {out_dir}")
    print("\nTo benchmark:")
    print(f"  python benchmark_tcd.py --tcd-dir {out_dir} \\")
    print( "      --output-root benchmark_results/phase23_holdout")


if __name__ == "__main__":
    main()
