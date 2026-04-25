#!/usr/bin/env python3
"""
Detectree2 end-to-end inference script (local version)
-----------------------------------------------------
Runs the full Detectree2 workflow:
  1. Tiles an orthomosaic -- Update: no tiling for now as TCD is 2048x2048px images
  2. Runs Detectron2 model inference
  3. Projects predictions to GeoJSON
  4. Stitches and cleans crowns if necessary
  5. Writes the output -- Work In Progress

Useage quickstart:
    python infer.py --use_test_data --weights finetuned --image_path data/tcd/bin_liang/tcd_tile_WON.tif

"""

from __future__ import annotations
import os
from pathlib import Path

from detectree2.preprocessing.tiling import tile_data
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns, post_clean
from detectree2.models.predict import predict_on_data
from detectree2.models.train import setup_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import wget
import cv2
import argparse
from detectron2.utils.visualizer import Visualizer
from datasets import load_dataset
import cv2

import rasterio
from rasterio.transform import from_bounds
import pandas as pd
import geopandas as gpd
from utils import download_tcd_tiles_streaming
from utils import clean_validate_predictions_vs_tcd_segments
from utils import visualize_validation_results
from utils import compute_final_metric
from utils import filter_raw_predictions
from utils import load_tcd_meta_for_tile
import torch
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

# Key Hyperparameters

filter_threshold = 0.18
nms_dedupe_threshold = 0.3


# --------------------------------------------------
# Utility: ensure directory exists
# --------------------------------------------------
def ensure_dir(p: str | Path) -> str:
    """Create directory (and parents) if missing."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return str(p if str(p).endswith(os.sep) else str(p) + os.sep)


def smoke_test(model_path: Path):
    test_img = Path("samples/test.png")
    if not test_img.exists():
        raise FileNotFoundError(
            f"❌ Smoke test image not found at {test_img}\n"
            "Place a small PNG or JPG at samples/test.png"
        )

    # 1) Ensure model is present (download if missing)
    if not model_path.exists():
        url = "https://zenodo.org/records/10522461/files/230103_randresize_full.pth"
        print(f"📦 Downloading model: {url}")
        wget.download(url, out=str(model_path))
        print("\n✅ Model download complete.")

    # 2) Set up predictor
    print("⚙️  Initializing predictor (smoke test mode) ...")
    cfg = setup_cfg(update_model=str(model_path))
    set_device(cfg)
    predictor = DefaultPredictor(cfg)

    # 3) Load RGB and run inference
    img_bgr = cv2.imread(str(test_img), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {test_img}")

    outputs = predictor(img_bgr[:, :, ::-1])  # BGR→RGB for Detectron2

    # 4) Visualize & save overlay
    vis = Visualizer(img_bgr[:, :, ::-1], metadata=None, scale=1.0)
    vis_out = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_path = test_img.with_name(test_img.stem + "_pred_overlay.png")
    cv2.imwrite(str(out_path), vis_out.get_image()[:, :, ::-1])
    print(f"🧪 Smoke test complete. Overlay saved to:\n  {out_path}")


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------
def make_predictor():
    """Initialise and return a Detectron2 predictor (called once per process)."""
    model_path = Path("model_echo29.pth") if args.weights == "finetuned" else Path("230103_randresize_full.pth")

    if not model_path.exists():
        url = f"https://zenodo.org/records/10522461/files/{model_path}"
        print(f"📦 Model not found locally — downloading from {url} ...")
        wget.download(url, out=str(model_path))
        print("\n✅ Model download complete.")

    print("\n⚙️  Initializing Detectron2 predictor ...")
    cfg = setup_cfg()
    if args.weights == "finetuned":
        config_path = Path("configs/full_train.yaml")
        if config_path.exists():
            cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(model_path)
    set_device(cfg)
    predictor = DefaultPredictor(cfg)
    print("✅ Predictor ready.")
    return predictor


def process_tile(img_path, predictor, working_root, output_dir=None):
    """
    Run full detectree2 pipeline on a single tile.
    If output_dir is given, writes {stem}_canopyai.geojson there and cleans up
    the working directory. Otherwise uses the legacy validation/overlay flow.
    """
    image_info = load_tcd_meta_for_tile(img_path)
    image_id   = image_info.get("image_id", "unknown") if image_info else "unknown"

    site_path      = Path(working_root)
    pred_tiles_path = ensure_dir(site_path / "tiles_pred")
    chip_dir        = Path(pred_tiles_path) / f"{img_path.stem}_chips"
    ensure_dir(chip_dir)

    buffer     = 30
    tile_width = args.tile_size
    tile_height = args.tile_size

    try:
        tile_data(
            str(img_path), chip_dir, buffer, tile_width, tile_height,
            dtype_bool=True, full_coverage=True,
        )
    except AttributeError as e:
        print(f"⚠️ Non-georeferenced image — skipping CRS: {e}")

    chips = list(Path(chip_dir).glob("*.tif"))
    chip_geo_dir = chip_dir / "predictions_geo"
    chip_geo_dir.mkdir(parents=True, exist_ok=True)
    merged_geojson = chip_geo_dir / f"{img_path.stem}_merged.geojson"

    if len(chips) == 0:
        print(f"  ⚠️  no usable chips (nodata tile) — writing empty GeoJSON")
        import json as _json
        with open(merged_geojson, "w") as _f:
            _json.dump({"type": "FeatureCollection", "features": []}, _f)
    else:
        chip_pred_dir = chip_dir / "predictions"
        predict_on_data(chip_dir, out_folder=chip_pred_dir, predictor=predictor, save=True)
        filter_raw_predictions(chip_pred_dir, score_thresh=filter_threshold, overwrite=True)
        project_to_geojson(tiles_path=chip_dir, pred_fold=chip_pred_dir, output_fold=chip_geo_dir)

        try:
            crowns_raw = stitch_crowns(str(chip_geo_dir), shift=1)
        except (FileNotFoundError, ValueError):
            crowns_raw = gpd.GeoDataFrame(columns=["Confidence_score", "geometry"])

        if crowns_raw.empty:
            import json as _json
            with open(merged_geojson, "w") as _f:
                _json.dump({"type": "FeatureCollection", "features": []}, _f)
        else:
            crowns_clean = clean_crowns(
                crowns_raw, iou_threshold=0.7, confidence=0.2,
                area_threshold=2, containment_threshold=0.85,
            )
            if crowns_clean.empty:
                crowns_raw.to_file(str(merged_geojson), driver="GeoJSON")
            else:
                crowns_final = post_clean(crowns_raw, crowns_clean, iou_threshold=0.3)
                crowns_final.to_file(str(merged_geojson), driver="GeoJSON")

    if output_dir is not None:
        # Batch mode: copy out and clean up intermediates
        import shutil
        shutil.copy(merged_geojson, Path(output_dir) / f"{img_path.stem}_canopyai.geojson")
        shutil.rmtree(site_path, ignore_errors=True)
        return True

    # Legacy single-tile mode: run validation/visualisation
    if image_id in ("unknown", "WON"):
        print(f"⚠️ No ground truth metadata for {img_path.name} — skipping validation metrics.")
        visualize_validation_results(pred=gpd.read_file(merged_geojson), gt=None, ious=None,
                                     site_path=site_path, rgb_path=img_path,
                                     tile_name=img_path.stem, image_id=image_id)
        return True

    metrics_all, pred, gt, scores, coco_anns = clean_validate_predictions_vs_tcd_segments(
        pred_geojson_path=merged_geojson, image_tif=image_info,
        iou_thresh_tree=0.5, iop_thresh_canopy=0.7,
    )
    if metrics_all is None:
        print(f"⚠️ No GT for tile {image_id} — skipping.")
        return True

    visualize_validation_results(pred, gt, scores, coco_anns, site_path=site_path,
                                 rgb_path=img_path, tile_name=img_path.stem, image_id=image_id)
    return metrics_all


def main():
    # ── Batch mode (benchmark_tcd.py) ─────────────────────────────────────────
    if args.tcd_dir:  # set via --tcd-dir (argparse converts - to _)
        tcd_dir    = Path(args.tcd_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        working_root = output_dir / f"_detectree2_working_{os.getpid()}"

        tile_set = set(args.tiles) if args.tiles else None
        tifs = sorted(tcd_dir.glob("*.tif"))
        if tile_set:
            tifs = [t for t in tifs if t.stem in tile_set]

        with open(os.devnull, "w") as devnull:
            old_fds = os.dup(1), os.dup(2)
            os.dup2(devnull.fileno(), 1); os.dup2(devnull.fileno(), 2)
            try:
                predictor = make_predictor()
            finally:
                os.dup2(old_fds[0], 1); os.close(old_fds[0])
                os.dup2(old_fds[1], 2); os.close(old_fds[1])
        print("⚙️  Predictor ready.")
        ok = skipped = 0
        for tif in tifs:
            out_file = output_dir / f"{tif.stem}_canopyai.geojson"
            if args.skip_existing and out_file.exists():
                skipped += 1
                ok += 1
                continue
            print(f"    {tif.name} ... ", end="", flush=True)
            try:
                with open(os.devnull, "w") as devnull:
                    old_fds = os.dup(1), os.dup(2)
                    os.dup2(devnull.fileno(), 1)
                    os.dup2(devnull.fileno(), 2)
                    try:
                        process_tile(tif, predictor, working_root, output_dir=output_dir)
                    finally:
                        os.dup2(old_fds[0], 1); os.close(old_fds[0])
                        os.dup2(old_fds[1], 2); os.close(old_fds[1])
                print("✓")
                ok += 1
            except Exception as e:
                print(f"✗  ({e})")
        if skipped:
            print(f"  {skipped} tiles skipped (already exist)")
        print(f"  {ok}/{len(tifs)} successful")
        return

    # ── Single-tile / legacy mode ──────────────────────────────────────────────
    if args.output_root:
        site_path = Path(args.output_root)
    else:
        site_path = Path.home() / "dphil" / "canopyAI" / "data" / "tcd"

    if args.image_path:
        files_to_process = [Path(args.image_path)]
        print(f"\n🎯 Processing single image: {args.image_path}")
    elif args.test_data_dir:
        files_to_process = sorted(Path(args.test_data_dir).glob("tcd_tile_*.tif"))
    elif args.use_test_data:
        files_to_process = sorted(Path("data/tcd/raw_test").glob("tcd_tile_*.tif"))
    else:
        files_to_process = sorted(Path("data/tcd/raw").glob("tcd_tile_*.tif"))

    if not files_to_process:
        raise FileNotFoundError("❌ No TCD tiles found.")

    predictor = make_predictor()

    all_tree_scores   = []
    all_canopy_scores = []
    total_pred = total_gt_trees = total_gt_canopy = 0

    for img_path in files_to_process:
        image_info = load_tcd_meta_for_tile(img_path)
        image_id   = image_info.get("image_id", "unknown") if image_info else "unknown"
        print(f"\n================ Processing {image_id} ================")
        print(f"Biome: {image_info.get('biome_name', 'N/A') if image_info else 'N/A'}")

        result = process_tile(img_path, predictor, site_path, output_dir=None)
        if isinstance(result, dict):
            metrics_all = result
            total_pred      += metrics_all["n_pred"]
            total_gt_trees  += metrics_all["n_gt_trees"]
            total_gt_canopy += metrics_all["n_gt_canopy"]
            scores_trees, scores_canopy = metrics_all.get("scores", ([], []))
            all_tree_scores.extend(scores_trees)
            all_canopy_scores.extend(scores_canopy)

    final_tree = compute_final_metric(
        all_tree_scores, thresh=0.5, n_pred=total_pred, n_gt=total_gt_trees
    )
    final_canopy = compute_final_metric(
        all_canopy_scores, thresh=0.7, n_pred=total_pred, n_gt=total_gt_canopy
    )
    print("============= Cohort Metrics ================")
    print_metrics("Trees (IoU)", final_tree)
    print_metrics("Canopy (IoP)", final_canopy)


def print_metrics(name, m):
    print(f"\n📊 {name} metrics")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:12s}: {v:.4f}")
        else:
            print(f"  {k:12s}: {v}")


def merge_tile_geojsons(geo_dir: Path, out_file: Path):
    import geopandas as gpd

    geo_dir = Path(geo_dir)
    files = sorted(geo_dir.glob("Prediction_*.geojson"))

    if not files:
        raise FileNotFoundError(f"No tile GeoJSONs found in {geo_dir}")

    gdfs = [gpd.read_file(f) for f in files]
    merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    merged.to_file(out_file, driver="GeoJSON")

    print(f"🧩 Merged {len(files)} tile GeoJSONs → {out_file}")


# CLI Args
def parse_args():
    ap = argparse.ArgumentParser(description="CanopyAI runner")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run non-geospatial smoke test on a single RGB image.",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default="detectree2",
        choices=["detectree2", "finetuned"],
        help="Which model weights to use: detectree2 (pretrained baseline) or finetuned",
    )
    ap.add_argument(
        "--use_test_data",
        action="store_true",
        help=(
            "Run inference on test data (data/tcd/raw_test/) instead of training data"
        ),
    )
    ap.add_argument(
        "--tile_size",
        type=int,
        default=40,
        help="Tile size in meters (default: 40). Reduce this for high-res imagery to avoid downscaling.",
        # still not 100% clear if this is useful or not tbh
        # might be worth experimenting with at some point
    )

    ap.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to a single image file to run inference on (overrides --use_test_data)",
    )
    ap.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
        help="Directory containing tcd_tile_*.tif files to benchmark (overrides --use_test_data)",
    )
    ap.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root directory for all outputs (tiles_pred/, overlays_validation/). "
             "Defaults to ~/dphil/canopyAI/data/tcd/",
    )
    # ── Batch mode (used by benchmark_tcd.py) ────────────────────────────────
    ap.add_argument(
        "--tcd-dir",
        type=str,
        default=None,
        help="Directory of .tif tiles to process in batch (model loaded once).",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for *_canopyai.geojson files (batch mode).",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tiles whose *_canopyai.geojson already exists in --output-dir.",
    )
    ap.add_argument(
        "--tiles",
        nargs="+",
        default=None,
        help="Restrict to these tile stems (e.g. tcd_tile_3 tcd_tile_7).",
    )

    return ap.parse_args()


def set_device(cfg):
    cfg.MODEL.DEVICE = "cpu"
    cfg.DATALOADER.NUM_WORKERS = 4


def visualize_saved_prediction_with_masks(
    img_path, pred_json_path, out_dir, image_id=None
):
    """
    Visualize Detectree2 predictions from JSON over the original RGB image.
    Focuses on segmentation masks rather than bounding boxes.
    Automatically names the output file using the image_id and tile index.
    """

    import re
    import torch
    import numpy as np
    import cv2
    import json
    from detectron2.structures import Boxes, Instances
    from detectron2.utils.visualizer import Visualizer
    from pycocotools import mask as mask_utils
    from pathlib import Path

    # --- Load image ---
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"❌ Could not read {img_path}")

    H, W = img.shape[:2]

    # --- Load predictions ---
    with open(pred_json_path) as f:
        data = json.load(f)

    if not data:
        print(f"⚠️ No predictions found in {pred_json_path}")
        return

    # --- Decode segmentation masks (supports compressed + uncompressed RLE) ---
    masks = []
    for d in data:
        seg = d.get("segmentation")
        if not seg:
            masks.append(np.zeros((H, W), dtype=np.uint8))
            continue

        try:
            # Handle compressed RLE (string) or uncompressed (list)
            if isinstance(seg, dict) and "counts" in seg:
                if isinstance(seg["counts"], list):
                    # Convert uncompressed → compressed RLE first
                    seg = mask_utils.frPyObjects(seg, *seg["size"])
                m = mask_utils.decode(seg)
            else:
                # Segmentation not RLE; fallback blank
                m = np.zeros((H, W), dtype=np.uint8)

            if m.ndim == 3:
                m = np.any(m, axis=2)
            masks.append(m)
        except Exception as e:
            print(f"⚠️ Failed to decode RLE segmentation: {e}")
            masks.append(np.zeros((H, W), dtype=np.uint8))

    if not masks:
        print(f"⚠️ No valid masks decoded for {pred_json_path}")
        return

    masks = torch.as_tensor(np.stack(masks))  # [N, H, W]

    # --- Dummy boxes (masks are the primary output) ---
    boxes = torch.tensor([[0, 0, W, H]], dtype=torch.float32).repeat(len(masks), 1)

    # --- Scores / Classes ---
    scores = torch.tensor([d.get("score", 0) for d in data])
    classes = torch.tensor([d.get("category_id", 0) for d in data])

    # --- Build Detectron2 Instances ---
    instances = Instances((H, W))
    instances.pred_boxes = Boxes(boxes)
    instances.scores = scores
    instances.pred_classes = classes
    instances.pred_masks = masks

    # --- Labels for overlay ---
    labels = [f"Tree {s * 100:.0f}%" for s in instances.scores]

    # --- Visualization ---
    vis = Visualizer(img[:, :, ::-1], scale=1.0)
    vis_out = vis.overlay_instances(masks=instances.pred_masks, labels=labels)

    # --- Construct output path ---
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_index = re.search(r"tile_(\d+)", pred_json_path.name)
    tile_str = f"tile_{tile_index.group(1)}" if tile_index else "tile"

    if image_id is None:
        image_id = "unknown"

    out_path = out_dir / f"{tile_str}_tcd{image_id}.png"

    # --- Write file ---
    cv2.imwrite(str(out_path), vis_out.get_image()[:, :, ::-1])
    print(f"✅ Saved overlay with masks → {out_path}")


# --------------------------------------------------
# Entrypoint
# --------------------------------------------------
if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import logging
    for _noisy in ("detectree2", "fvcore", "detectron2"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    args = parse_args()

    if args.smoke:
        model_path = Path("230103_randresize_full.pth")
        smoke_test(model_path)
    else:
        main()
