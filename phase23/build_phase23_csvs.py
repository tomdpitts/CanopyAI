#!/usr/bin/env python3
"""
build_phase23_csvs.py — Build phase23 TCD training and early-stopping val CSVs.

Streams restor/tcd split="train" (4,169 tiles) from HuggingFace.
As a side-effect, repairs the empty _meta.json files on disk (lost during
Mac transfer) by rewriting them from the streamed annotation data.

Outputs:
    phase23/phase23_tcd_train.csv   folds 0–3  (~3,335 tiles)
    phase23/phase23_tcd_val.csv     fold 4     (~834 tiles, early stopping)

The holdout test split (439 tiles) is intentionally NOT touched here.
Download it separately with phase23/download_tcd_holdout.py after training.

--- Tile matching assumption ---
Tiles were originally downloaded in HuggingFace streaming order, so
stream item i → tcd_tile_{i}.tif. This holds if the original download ran
to completion without interruption (all 4,169 tif files exist on disk).
A warning is printed for any item whose tif file is missing.

--- Canopy annotation strategy ---
category_id 2 (ITC): COCO bbox used directly → label="Tree"
category_id 1 (canopy): large polygon subdivided into a grid of pseudo-ITC
    bboxes. Only cells where Area(cell ∩ polygon) / Area(cell) ≥ 0.5 are kept.
    Cell size = median ITC bbox width in same tile, or CANOPY_DEFAULT_SIZE (100px).

--- Shadow vectors ---
Joined from data/tcd/tcd_shadow_vectors.json for manually_reviewed tiles only.

Usage:
    source venv310/bin/activate
    python phase23/build_phase23_csvs.py
    python phase23/build_phase23_csvs.py --dry-run        # counts only, no writes
    python phase23/build_phase23_csvs.py --skip-repair    # skip meta.json rewrite
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import box as shapely_box, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

ROOT      = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/tcd/images/data/tcd/raw"
SHADOW_JSON = ROOT / "data/tcd/tcd_shadow_vectors.json"
OUT_DIR   = ROOT / "phase23"

CANOPY_CAT            = 1
ITC_CAT               = 2
CANOPY_OVERLAP_THRESH = 0.5
CANOPY_DEFAULT_SIZE   = 100   # px fallback when tile has no ITC annotations
VAL_FOLD              = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ann(a) -> dict | None:
    """Parse a COCO annotation that may arrive as a JSON string, dict, or junk."""
    if isinstance(a, dict):
        return a
    if isinstance(a, (str, bytes)):
        s = a.strip() if isinstance(a, str) else a.decode("utf-8", errors="ignore").strip()
        if not s:
            return None
        try:
            result = json.loads(s)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def load_shadow_vectors(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {stem: v for stem, v in data.items() if v.get("manually_reviewed")}


def coco_bbox_to_xyxy(bbox: list) -> tuple:
    x, y, w, h = bbox
    return float(x), float(y), float(x + w), float(y + h)


def parse_polygon(seg, img_w: int, img_h: int):
    """Parse COCO segmentation field → Shapely geometry, or None on failure."""
    if isinstance(seg, list) and seg and isinstance(seg[0], list):
        coords = np.array(seg[0], dtype=float).reshape(-1, 2)
        if len(coords) < 3:
            return None
        poly = Polygon(coords)
        return poly if poly.is_valid else poly.buffer(0)

    if isinstance(seg, dict) and "counts" in seg:
        try:
            import pycocotools.mask as mask_utils
            import rasterio.features
            from shapely.geometry import shape
            mask = mask_utils.decode(seg)
            shapes = list(rasterio.features.shapes(mask.astype(np.uint8), mask > 0))
            polys = [shape(geom) for geom, val in shapes if val == 1]
            return unary_union(polys) if polys else None
        except Exception:
            return None

    return None


def subdivide_canopy(polygon: Polygon, cell_size: int, img_w: int, img_h: int) -> list:
    """Return (xmin, ymin, xmax, ymax) tuples for grid cells inside the polygon."""
    minx, miny, maxx, maxy = polygon.bounds
    minx = max(0, int(math.floor(minx)))
    miny = max(0, int(math.floor(miny)))
    maxx = min(img_w, int(math.ceil(maxx)))
    maxy = min(img_h, int(math.ceil(maxy)))

    boxes = []
    x = minx
    while x + cell_size <= maxx:
        y = miny
        while y + cell_size <= maxy:
            cell = shapely_box(x, y, x + cell_size, y + cell_size)
            try:
                overlap = cell.intersection(polygon).area / cell.area
            except Exception:
                overlap = 0.0
            if overlap >= CANOPY_OVERLAP_THRESH:
                boxes.append((float(x), float(y),
                              float(x + cell_size), float(y + cell_size)))
            y += cell_size
        x += cell_size
    return boxes


def process_tile(item: dict, tile_stem: str, img_path: Path,
                 shadow_vecs: dict) -> tuple[list, int, int]:
    """
    Extract training rows for one tile.
    Returns (rows, n_itc, n_pseudo_canopy).
    """
    raw = item.get("coco_annotations") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    anns = [x for x in (_parse_ann(a) for a in raw) if x is not None]
    img_w  = item.get("width", 2048)
    img_h  = item.get("height", 2048)
    fold   = item.get("validation_fold", -1)

    # ITC annotations (category 2)
    itc_rows    = []
    itc_widths  = []
    for ann in anns:
        if ann.get("category_id") != ITC_CAT:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        xmin, ymin, xmax, ymax = coco_bbox_to_xyxy(bbox)
        if xmax <= xmin or ymax <= ymin:
            continue
        itc_widths.append(xmax - xmin)
        itc_rows.append((xmin, ymin, xmax, ymax))

    cell_size = max(20, int(np.percentile(itc_widths, 60))) if itc_widths else CANOPY_DEFAULT_SIZE

    # Canopy annotations (category 1) → pseudo-ITC grid
    canopy_rows = []
    for ann in anns:
        if ann.get("category_id") != CANOPY_CAT:
            continue
        seg = ann.get("segmentation")
        if not seg:
            continue
        poly = parse_polygon(seg, img_w, img_h)
        if poly is None or poly.is_empty or poly.area < (cell_size ** 2) * 0.25:
            continue
        canopy_rows.extend(subdivide_canopy(poly, cell_size, img_w, img_h))

    all_boxes = itc_rows + canopy_rows
    if not all_boxes:
        return [], 0, 0

    sv = shadow_vecs.get(tile_stem)
    shadow_angle = sv["shadow_angle_deg"] if sv else float("nan")
    shadow_x     = sv["shadow_x"]         if sv else float("nan")
    shadow_y     = sv["shadow_y"]         if sv else float("nan")

    img_str = str(img_path)
    rows = [
        {
            "image_path":   img_str,
            "xmin":         b[0], "ymin": b[1],
            "xmax":         b[2], "ymax": b[3],
            "label":        "Tree",
            "shadow_angle": shadow_angle,
            "shadow_x":     shadow_x,
            "shadow_y":     shadow_y,
            "domain":       "TCD",
            "fold":         fold,
        }
        for b in all_boxes
    ]
    return rows, len(itc_rows), len(canopy_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",      action="store_true",
                    help="Print counts without writing any files")
    ap.add_argument("--skip-repair",  action="store_true",
                    help="Skip rewriting meta.json files")
    ap.add_argument("--from-disk",    action="store_true",
                    help="Read annotations from local meta.json files instead of "
                         "streaming HuggingFace (much faster; requires download_tcd_all.py "
                         "to have run first so meta.json files are populated)")
    ap.add_argument("--max",          type=int, default=None,
                    help="Stop after N tiles (for testing)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    shadow_vecs = load_shadow_vectors(SHADOW_JSON)
    print(f"Shadow vectors: {len(shadow_vecs)} manually reviewed tiles")

    all_rows     = []
    total_itc    = 0
    total_pseudo = 0
    missing_tif  = 0
    empty_tiles  = 0

    if args.from_disk:
        tifs = sorted(
            TRAIN_DIR.glob("tcd_tile_*.tif"),
            key=lambda p: int(p.stem.split("_")[-1])
        )
        if args.max:
            tifs = tifs[:args.max]
        print(f"Reading from disk: {len(tifs)} tiles in {TRAIN_DIR}")

        for tif_path in tqdm(tifs, desc="Building phase23 CSVs"):
            tile_stem = tif_path.stem
            meta_path = TRAIN_DIR / f"{tile_stem}_meta.json"
            i = int(tile_stem.split("_")[-1])

            if tif_path.stat().st_size == 0:
                tqdm.write(f"  ⚠  {tile_stem}.tif is 0 bytes — skipping")
                missing_tif += 1
                continue

            if not meta_path.exists() or meta_path.stat().st_size == 0:
                tqdm.write(f"  ⚠  {tile_stem}_meta.json missing/empty — skipping")
                missing_tif += 1
                continue

            with open(meta_path) as f:
                item = json.load(f)

            rows, n_itc, n_pseudo = process_tile(item, tile_stem, tif_path, shadow_vecs)

            if not rows:
                empty_tiles += 1
            else:
                all_rows.extend(rows)
                total_itc    += n_itc
                total_pseudo += n_pseudo

            if i < 5 or i % 500 == 0:
                tqdm.write(
                    f"  [{i:4d}] {tile_stem}  itc={n_itc}  pseudo={n_pseudo}"
                    f"  fold={item.get('validation_fold','?')}"
                    f"  biome={item.get('biome_name','?')}"
                )
    else:
        from datasets import load_dataset, Image as HFImage
        ds = load_dataset("restor/tcd", split="train", streaming=True).cast_column(
            "image", HFImage(decode=False)
        )

        for i, item in enumerate(tqdm(ds, total=args.max or 4169, desc="Building phase23 CSVs")):
            if args.max is not None and i >= args.max:
                break
            tile_stem = f"tcd_tile_{i}"
            img_path  = TRAIN_DIR / f"{tile_stem}.tif"
            meta_path = TRAIN_DIR / f"{tile_stem}_meta.json"

            if not img_path.exists():
                tqdm.write(f"  ⚠  {tile_stem}.tif missing on disk — skipping")
                missing_tif += 1
                continue

            if not args.skip_repair and not args.dry_run:
                meta = {
                    "image_id":         item.get("image_id"),
                    "bounds":           item.get("bounds"),
                    "crs":              str(item.get("crs", "")),
                    "width":            item.get("width", 2048),
                    "height":           item.get("height", 2048),
                    "coco_annotations": [
                        x for x in (_parse_ann(a)
                                     for a in (item.get("coco_annotations") or []))
                        if x is not None
                    ],
                    "biome":            item.get("biome"),
                    "biome_name":       item.get("biome_name"),
                    "country":          item.get("country"),
                    "validation_fold":  item.get("validation_fold"),
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f)

            rows, n_itc, n_pseudo = process_tile(item, tile_stem, img_path, shadow_vecs)

            if not rows:
                empty_tiles += 1
            else:
                all_rows.extend(rows)
                total_itc    += n_itc
                total_pseudo += n_pseudo

            if i < 5 or i % 500 == 0:
                tqdm.write(
                    f"  [{i:4d}] {tile_stem}  itc={n_itc}  pseudo={n_pseudo}"
                    f"  fold={item.get('validation_fold','?')}"
                    f"  biome={item.get('biome_name','?')}"
                )

    df = pd.DataFrame(all_rows)
    print(f"\n{'─'*50}")
    print(f"Tiles processed : {i + 1 - missing_tif}")
    print(f"Missing tif     : {missing_tif}")
    print(f"Empty tiles     : {empty_tiles}")
    print(f"Total rows      : {len(df)}")
    print(f"  ITC           : {total_itc}")
    print(f"  Pseudo-canopy : {total_pseudo}")
    if "fold" in df.columns:
        print(f"\nFold distribution (rows):")
        print(df["fold"].value_counts().sort_index().to_string())
    if "shadow_angle" in df.columns:
        print(f"\nShadow-annotated rows: {df['shadow_angle'].notna().sum()}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    train_df = df[df["fold"] != VAL_FOLD].drop(columns=["fold"])
    val_df   = df[df["fold"] == VAL_FOLD].drop(columns=["fold"])

    train_path = OUT_DIR / "phase23_tcd_train.csv"
    val_path   = OUT_DIR / "phase23_tcd_val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)

    n_train_tiles = train_df["image_path"].nunique()
    n_val_tiles   = val_df["image_path"].nunique()
    print(f"\nTrain CSV : {len(train_df):6d} rows  {n_train_tiles} tiles → {train_path.name}")
    print(f"Val CSV   : {len(val_df):6d} rows  {n_val_tiles} tiles → {val_path.name}")
    print("\nNext:")
    print("  modal volume put canopyai-deepforest-data "
          "phase23/phase23_tcd_train.csv phase23_tcd_train.csv")
    print("  modal volume put canopyai-deepforest-data "
          "phase23/phase23_tcd_val.csv phase23_tcd_val.csv")


if __name__ == "__main__":
    main()
