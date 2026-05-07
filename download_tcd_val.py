#!/usr/bin/env python3
"""
download_tcd_val.py — Download the full TCD validation split from HuggingFace.

Mirrors the structure of the train set in data/tcd/images/data/tcd/raw/:
    data/tcd/images/data/tcd/val/
        tcd_val_tile_0.tif
        tcd_val_tile_0_meta.json
        ...

No biome filter is applied — downloads all validation tiles across all biomes
so results are directly comparable to the OAM-TCD paper benchmarks.

Usage:
    python download_tcd_val.py
    python download_tcd_val.py --skip-existing        # resume interrupted run
    python download_tcd_val.py --max-tiles 50         # quick test
"""

import argparse
import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds


OUT_DIR = Path("data/tcd/images/data/tcd/val")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help="Output directory (default: data/tcd/images/data/tcd/val)")
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="Stop after this many tiles (default: all)")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip tiles already on disk (default: True)")
    return ap.parse_args()


def save_tile(image_info, img_path, meta_path):
    img_bytes = image_info["image"]["bytes"]

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
    bounds = image_info["bounds"]
    crs    = image_info["crs"]
    transform = from_bounds(*bounds, width=w, height=h)

    with rasterio.open(
        img_path, "w", driver="GTiff",
        height=h, width=w, count=3, dtype=img.dtype,
        crs=crs, transform=transform,
    ) as dst:
        for b in range(3):
            dst.write(img[:, :, b], b + 1)

    meta = {
        "image_id":         image_info["image_id"],
        "bounds":           bounds,
        "crs":              str(crs),
        "width":            w,
        "height":           h,
        "coco_annotations": image_info.get("coco_annotations", []),
        "biome":            image_info.get("biome"),
        "biome_name":       image_info.get("biome_name"),
        "country":          image_info.get("country"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset, Image

    print(f"📦 Loading TCD holdout test split ...")
    ds = load_dataset("restor/tcd", split="test", streaming=True).cast_column(
        "image", Image(decode=False)
    )

    saved = 0
    for i, image_info in enumerate(ds):
        if args.max_tiles is not None and saved >= args.max_tiles:
            break

        img_path  = out_dir / f"tcd_val_tile_{i}.tif"
        meta_path = out_dir / f"tcd_val_tile_{i}_meta.json"

        if args.skip_existing and img_path.exists() and meta_path.exists():
            print(f"  ⏭  tcd_val_tile_{i} exists, skipping")
            continue

        biome   = image_info.get("biome_name", "unknown")
        img_id  = image_info.get("image_id", "?")
        print(f"  [{i}] image_id={img_id}  biome={biome} ...", end=" ", flush=True)

        try:
            save_tile(image_info, img_path, meta_path)
            print("✓")
            saved += 1
        except Exception as e:
            print(f"✗  ({e})")

    total = sum(1 for _ in out_dir.glob("tcd_val_tile_*.tif"))
    print(f"\n✅  Done. {saved} new tiles saved ({total} total) → {out_dir}")
    print(f"\nTo benchmark on the val set:")
    print(f"  python benchmark_tcd.py --models detectree2 --names detectree2 \\")
    print(f"      --tcd-dir {out_dir} --output-root benchmark_results_val")


if __name__ == "__main__":
    main()
