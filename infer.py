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
"""

from __future__ import annotations
import os
from pathlib import Path

from detectree2.preprocessing.tiling import tile_data
from detectree2.models.outputs import project_to_geojson
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
from utils import download_tcd_tiles_streaming
from utils import clean_validate_predictions_vs_tcd_segments
from utils import visualize_validation_results
from utils import compute_final_metric
from utils import filter_raw_predictions
from utils import load_tcd_meta_for_tile
from utils import apply_nms_to_geojson
import torch
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

# Key Hyperparameters

filter_threshold = 0.65
nms_dedupe_threshold = 0.18


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
def main():
    # Model path
    if args.weights == "finetuned":
        model_path = Path("data/tcd/train_outputs/model_final.pth")
    else:
        model_path = Path("230103_randresize_full.pth")

    # === 1. Define key paths ===
    home = Path.home()
    site_name = "tcd"
    site_path = home / "dphil" / "canopyAI" / "data" / site_name
    raw_dir = site_path / "raw"

    # # === 1b. Download tiles ===
    # if not args.already_downloaded:
    #     print("🌐 Downloading via HF...")
    #     tiles_info = download_tcd_tiles_streaming(raw_dir, max_images=max_images)
    #     print(f"✅ Downloaded {len(tiles_info)} TCD tiles for processing.")
    # else:
    #     print("⏭️ Using existing tiles in raw/")

    # === 2. Create output/working directories ===
    pred_tiles_path = ensure_dir(site_path / "tiles_pred")

    # === 3. Download pretrained model if missing ===
    if not model_path.exists():
        url = f"https://zenodo.org/records/10522461/files/{model_path}"
        print(f"📦 Model not found locally — downloading from {url} ...")
        wget.download(url, out=str(model_path))
        print("\n✅ Model download complete.")

    # === 4. Initialize Detectron2 predictor ===
    print("\n⚙️  Initializing Detectron2 predictor ...")
    cfg = setup_cfg(update_model=str(model_path))
    set_device(cfg)
    predictor = DefaultPredictor(cfg)
    print("✅ Predictor ready.")

    # Initialize accumulators
    all_tree_scores = []
    all_canopy_scores = []
    total_pred = 0
    total_gt_trees = 0
    total_gt_canopy = 0

    # === 5–12. Process each tile ===
    raw_dir = Path("data/tcd/raw")

    if len(list(raw_dir.glob("tcd_tile_*.tif"))) == 0:
        raise FileNotFoundError(
            f"❌ No TCD tiles found in {raw_dir}.\n"
            "Please run prepare_data.py first to download and tile the dataset."
        )

    for img_path in sorted(raw_dir.glob("tcd_tile_*.tif")):
        image_info = load_tcd_meta_for_tile(img_path)

        image_id = image_info["image_id"]

        print(f"\n================ Processing {image_id} ================")
        print(f"Biome: {image_info.get('biome_name', 'N/A')}")

        # ------------------------------------------------------------
        # 5. Tile orthomosaic into chips for inference
        # ------------------------------------------------------------
        print("\n🧩 Tiling image into smaller chips ...")

        chip_dir = Path(pred_tiles_path) / f"{img_path.stem}_chips"
        ensure_dir(chip_dir)

        buffer = 30
        tile_width = 40
        tile_height = 40

        try:
            # Your Detectree2 tiler (no CRS loss if the input GeoTIFF is georeferenced)
            tile_data(
                str(img_path),
                chip_dir,  # output directory
                buffer,
                tile_width,
                tile_height,
                dtype_bool=True,  # Detectree2 expects this for mask chips
            )
            print("✅ Tiling complete.")
        except AttributeError as e:
            print(f"⚠️ Non-georeferenced image — skipping CRS: {e}")

        # If no tiles were created, just skip this image and continue.
        chips = list(Path(chip_dir).glob("*.tif"))
        if len(chips) == 0:
            print(
                f"⚠️  Skipping {image_id} — no tiles produced (likely nodata or invalid raster)."
            )
            continue

        # ------------------------------------------------------------
        # 6. Run Detectron2 inference on chips
        # ------------------------------------------------------------
        print("\n🔮 Running model inference on tiled chips ...")
        predict_on_data(chip_dir, predictor=predictor, save=True)
        print("✅ Inference complete.")

        # ------------------------------------------------------------
        # 7. Filter raw Detectron2 predictions *inside chip folder*
        # ------------------------------------------------------------
        chip_pred_dir = chip_dir / "predictions"
        filter_raw_predictions(
            chip_pred_dir, score_thresh=filter_threshold, overwrite=True
        )

        # ------------------------------------------------------------
        # 8. Reproject tiled predictions → GeoJSON in global CRS
        # ------------------------------------------------------------
        print("\n🗺️  Projecting tile predictions to GeoJSON ...")
        chip_geo_dir = chip_dir / "predictions_geo"
        ensure_dir(chip_geo_dir)

        project_to_geojson(
            tiles_path=chip_dir, pred_fold=chip_pred_dir, output_fold=chip_geo_dir
        )
        print("✅ GeoJSON projection complete.")

        # ------------------------------------------------------------
        # 9. Visualize & Validate on the *merged* predictions
        # ------------------------------------------------------------
        # Detectree2 produces one GeoJSON per tile — merge them for evaluation
        merged_geojson = chip_geo_dir / f"{img_path.stem}_merged.geojson"
        merge_tile_geojsons(chip_geo_dir, merged_geojson)
        # Apply NMS to remove duplicate overlapping polygons
        apply_nms_to_geojson(merged_geojson, iou_threshold=nms_dedupe_threshold)

        # visualize_saved_prediction_with_masks(
        #     img_path,                     # original tile
        #     merged_geojson,               # merged prediction mask JSON
        #     overlays_path,
        #     image_id
        # )

        metrics_all, pred, gt, scores, coco_anns = (
            clean_validate_predictions_vs_tcd_segments(
                pred_geojson_path=merged_geojson,
                image_tif=image_info,
                iou_thresh_tree=0.5,
                iop_thresh_canopy=0.7,
            )
        )

        if metrics_all is None:
            print(f"⚠️ No GT for tile {image_id} — skipping.")
            continue
        scores_trees, scores_canopy = scores

        total_pred += metrics_all["n_pred"]
        total_gt_trees += metrics_all["n_gt_trees"]
        total_gt_canopy += metrics_all["n_gt_canopy"]

        # Extend raw overlap score lists
        all_tree_scores.extend(scores_trees.tolist())
        all_canopy_scores.extend(scores_canopy.tolist())

        visualize_validation_results(
            pred,
            gt,
            scores,
            coco_anns,
            site_path=site_path,
            rgb_path=img_path,
            tile_name=img_path.stem,
            image_id=image_id,
        )

    final_tree = compute_final_metric(
        all_tree_scores, thresh=0.5, n_pred=total_pred, n_gt=total_gt_trees
    )

    final_canopy = compute_final_metric(
        all_canopy_scores, thresh=0.7, n_pred=total_pred, n_gt=total_gt_canopy
    )

    print(f"============= Cohort Metrics ================")
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
        default="baseline",
        choices=["baseline", "finetuned"],
        help="Which model weights to use: baseline or finetuned",
    )

    return ap.parse_args()


def set_device(cfg):
    # Prefer Apple MPS, else CPU (no CUDA on Apple Silicon)
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    cfg.MODEL.DEVICE = device
    print(f"🖥️ Using device: {device}")
    cfg.DATALOADER.NUM_WORKERS = 0
    torch.set_num_threads(1)


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

    # --- Dummy boxes (since we mainly care about masks) ---
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
    args = parse_args()

    if args.smoke:
        model_path = Path("230103_randresize_full.pth")
        smoke_test(model_path)
    else:
        main()
