#!/usr/bin/env python3
"""
evaluate_batch.py — Run IoU/IoP evaluation on Foxtrot batch predictions.

Given a directory of prediction GeoJSONs (*_canopyai.geojson) and a directory
of TCD metadata files (*_meta.json), this script:

  1. Matches each prediction to its corresponding metadata by tile name.
  2. Calls clean_validate_predictions_vs_tcd_segments from utils.py.
  3. Saves per-tile visualisation overlays.
  4. Prints an aggregated summary table.

Usage:
    python evaluate_batch.py \\
        --predictions output/  \\
        --images      data/tcd/images/data/tcd/raw_test/ \\
        --metadata    data/tcd/images/data/tcd/raw_test/

    (metadata defaults to --images dir if not supplied)
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.affinity import affine_transform

# Add repo root to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import clean_validate_predictions_vs_tcd_segments, visualize_validation_results


def parse_args():
    p = argparse.ArgumentParser(description="Batch IoU/IoP evaluation of Foxtrot predictions")
    p.add_argument(
        "--predictions",
        type=str,
        default="output",
        help="Directory containing *_canopyai.geojson prediction files",
    )
    p.add_argument(
        "--images",
        type=str,
        default="data/tcd/images/data/tcd/raw_test",
        help="Directory containing tcd_tile_*.tif images",
    )
    p.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Directory containing *_meta.json files (defaults to --images dir)",
    )
    p.add_argument(
        "--iou_thresh_tree",
        type=float,
        default=0.5,
        help="IoU threshold for tree true-positives (default: 0.5)",
    )
    p.add_argument(
        "--iop_thresh_canopy",
        type=float,
        default=0.7,
        help="IoP threshold for canopy true-positives (default: 0.7)",
    )
    p.add_argument(
        "--no_viz",
        action="store_true",
        help="Skip per-tile visualisation output",
    )
    return p.parse_args()


def main():
    args = parse_args()

    pred_dir = Path(args.predictions)
    images_dir = Path(args.images)
    meta_dir = Path(args.metadata) if args.metadata else images_dir

    if not pred_dir.exists():
        print(f"❌ Predictions directory not found: {pred_dir}")
        sys.exit(1)

    pred_files = sorted(pred_dir.glob("*_canopyai.geojson"))
    if not pred_files:
        print(f"⚠️ No *_canopyai.geojson files found in {pred_dir}")
        sys.exit(1)

    print(f"📂 Found {len(pred_files)} prediction files in {pred_dir}")
    print(f"📂 Looking for metadata in {meta_dir}")
    print(f"{'─' * 60}")

    all_results = []

    for pred_path in pred_files:
        # Infer tile name: e.g. "tcd_tile_1_canopyai.geojson" → "tcd_tile_1"
        tile_stem = pred_path.stem.replace("_canopyai", "")

        meta_path = meta_dir / f"{tile_stem}_meta.json"
        tif_path = images_dir / f"{tile_stem}.tif"

        if not meta_path.exists():
            print(f"⚠️  [{tile_stem}] No metadata found at {meta_path} — skipping")
            continue

        print(f"\n🔍 Evaluating {tile_stem} ...")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Inject width/height from the TIF (not stored in metadata when using decode=False)
        if "width" not in meta or "height" not in meta:
            import rasterio as rio
            if tif_path.exists():
                with rio.open(tif_path) as src:
                    meta["width"] = src.width
                    meta["height"] = src.height
            else:
                print(f"   ⚠️  [{tile_stem}] TIF not found, cannot compute transform — skipping")
                continue

        # --- Pre-process: transform predictions from pixel to world coords ---
        # foxtrot.py saves polygon coordinates in pixel space but labels the CRS
        # as the TIF's projected CRS. We must apply the same affine used for GT
        # before calling the evaluation function.
        pred_gdf_px = gpd.read_file(str(pred_path))
        bounds = meta["bounds"]
        pix_transform = from_bounds(
            *bounds,
            width=meta["width"],
            height=meta["height"],
        )
        # Rasterio affine: (a, b, d, e, c, f)  — shapely takes [a, b, d, e, xoff, yoff]
        affine_coeffs = [
            pix_transform.a,
            pix_transform.b,
            pix_transform.d,
            pix_transform.e,
            pix_transform.c,
            pix_transform.f,
        ]
        pred_world_geoms = [affine_transform(g, affine_coeffs) for g in pred_gdf_px.geometry]
        pred_gdf_world = pred_gdf_px.copy()
        pred_gdf_world["geometry"] = pred_world_geoms
        pred_gdf_world = pred_gdf_world.set_crs(meta["crs"], allow_override=True)

        # Save to a temp GeoJSON for the eval function
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False, mode="w") as tmp:
            tmp_path = tmp.name
        pred_gdf_world.to_file(tmp_path, driver="GeoJSON")

        metrics_all, pred_gdf, gt_gdf, score_pair, _ = clean_validate_predictions_vs_tcd_segments(
            pred_geojson_path=tmp_path,
            image_tif=meta,
            iou_thresh_tree=args.iou_thresh_tree,
            iop_thresh_canopy=args.iop_thresh_canopy,
        )
        os.unlink(tmp_path)  # clean up temp file

        if metrics_all is None:
            print(f"   ❌ Evaluation returned None (likely no annotations)")
            continue

        # Print per-tile summary (only "trees" and "canopy" are metric dicts)
        for metric_key in ("trees", "canopy"):
            if metric_key not in metrics_all:
                continue
            m = metrics_all[metric_key]
            print(
                f"   [{metric_key}] "
                f"P={m['precision']:.3f} R={m['recall']:.3f} "
                f"F1={m['f1_score']:.3f} mean_overlap={m['mean_overlap']:.3f} "
                f"TP={m['n_tp']}"
            )

        all_results.append({"tile": tile_stem, "metrics": metrics_all})

        # Per-tile visualisation
        if not args.no_viz:
            rgb_path = str(tif_path) if tif_path.exists() else None
            # coco_annotations may be a JSON string — parse it if needed
            raw_anns = meta.get("coco_annotations", [])
            if isinstance(raw_anns, str):
                raw_anns = json.loads(raw_anns)
            visualize_validation_results(
                pred=pred_gdf,
                gt=gt_gdf,
                ious=score_pair,
                site_path=str(pred_dir),
                tile_name=tile_stem,
                rgb_path=rgb_path,
                iou_thresh_tree=args.iou_thresh_tree,
                iop_thresh_canopy=args.iop_thresh_canopy,
                coco_anns=raw_anns,
            )

    # === Aggregated summary ===
    if not all_results:
        print("\n❌ No results to aggregate.")
        return

    print(f"\n{'═' * 60}")
    print(f"  AGGREGATED RESULTS  ({len(all_results)} tiles)")
    print(f"{'═' * 60}")

    for metric_key in ("trees", "canopy"):
        vals = [r["metrics"][metric_key] for r in all_results if metric_key in r["metrics"]]
        if not vals:
            continue

        avg_p    = np.mean([v["precision"]    for v in vals])
        avg_r    = np.mean([v["recall"]       for v in vals])
        avg_f1   = np.mean([v["f1_score"]     for v in vals])
        avg_ov   = np.mean([v["mean_overlap"] for v in vals])
        total_tp = sum(v["n_tp"]              for v in vals)

        print(f"\n  [{metric_key}]")
        print(f"    Precision   : {avg_p:.3f}")
        print(f"    Recall      : {avg_r:.3f}")
        print(f"    F1 Score    : {avg_f1:.3f}")
        print(f"    Mean Overlap: {avg_ov:.3f}")
        print(f"    Total TPs   : {total_tp}")

    print(f"\n{'═' * 60}")
    print("✅ Evaluation complete.")


if __name__ == "__main__":
    main()
