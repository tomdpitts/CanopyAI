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
import shutil

from detectree2.preprocessing.tiling import tile_data
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns
from detectree2.models.predict import predict_on_data
from detectree2.models.train import setup_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import Visualizer
import wget
import cv2
import argparse
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch
from datasets import load_dataset
from pycocotools import mask as mask_utils
import json, cv2, torch
import rasterio
import geopandas as gpd

import requests

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

import geopandas as gpd
import numpy as np
import json
from shapely.geometry import Polygon

import matplotlib.pyplot as plt

import rasterio
from rasterio.transform import from_bounds
from shapely.affinity import affine_transform

from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection


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
    # SET MODEL HERE
    model_name = "230103_randresize_full"

    # === 1. Define key paths ===
    # Adjust this base path if your dataset lives elsewhere
    home = Path.home()
    site_name = "tcd"
    site_path = home / "dphil" / "detectree2" / "data" / site_name
    # img_path = site_path / "rgb" / "2015.06.10_07cm_ORTHO.tif" # update with correct .tif
    # img_path = download_one_tcd_tile(site_path / "rgb")
    img_path, ann_path, example, image_id = download_one_tcd_tile(site_path / "rgb")
    model_path = Path(f"{model_name}.pth")

    # === 2. Create output/working directories ===
    pred_tiles_path = ensure_dir(site_path / "tiles_pred")
    preds_path = ensure_dir(Path(pred_tiles_path) / "predictions")
    preds_geo_path = ensure_dir(Path(pred_tiles_path) / "predictions_geo")
    overlays_path = ensure_dir(site_path / "overlays")

    # === 3. Download pretrained model if missing ===
    if not model_path.exists():
        url = f"https://zenodo.org/records/10522461/files/{model_name}.pth" # update with model
        print(f"📦 Model not found locally — downloading from {url} ...")
        wget.download(url, out=str(model_path))
        print("\n✅ Model download complete.")

    # === 4. Check that orthomosaic exists ===
    
    if not img_path.exists():
        print(f"⚠️ GeoTIFF not found at {img_path}.")
        return

    # # === 5. Tile orthomosaic === SKIPPING FOR NOW AS TCD IS 2048x2048px images
    # print("\n🧩 Tiling image into smaller chips ...")
    # buffer = 30
    # tile_width = 40
    # tile_height = 40
    # # tile_data(str(img_path), pred_tiles_path, buffer, tile_width, tile_height, dtype_bool=True)
    # try:
    #     tile_data(str(img_path), pred_tiles_path, buffer, tile_width, tile_height, dtype_bool=True)
    # except AttributeError as e:
    #     print(f"⚠️ Non-georeferenced image — skipping CRS: {e}")
    # print("✅ Tiling complete.")


    tile_dir = Path(pred_tiles_path)
    tif_tiles = list(tile_dir.glob("*.tif"))

    if len(tif_tiles) == 0:
        # Deprecated, len should be > 0
        # If tiler produced nothing (maybe because it's not a georeferenced image), copy original image in as a single tile
        single_tile_path = tile_dir / f"{img_path.stem}_tile.tif"
        shutil.copy(img_path, single_tile_path)
        print(f"📎 Source image copied to tiles folder as {single_tile_path}")
    else:
        print(f"✅ {len(tif_tiles)} tiles created.")

    # === 6. Set up Detectron2 predictor ===
    print("\n⚙️  Initializing Detectron2 predictor ...")
    cfg = setup_cfg(update_model=str(model_path))
    set_device(cfg) # is this needed?
    predictor = DefaultPredictor(cfg)
    print("✅ Predictor ready.")

    # === 7. Run inference on all tiles ===
    print("\n🔮 Running model inference on tiles ... (this may take a while)")
    predict_on_data(pred_tiles_path, predictor=predictor)
    print("✅ Inference complete.")

    tile_name = f"{img_path.stem}_tile"
    visualize_saved_prediction_with_masks(
        Path(pred_tiles_path) / f"{tile_name}.tif",
        Path(pred_tiles_path) / "predictions" / f"prediction_{tile_name}.json",
        Path(overlays_path) / f"{tile_name}_overlay.png",
        score_thresh=0.8
    )

    with rasterio.open(img_path) as ds:
        print(ds.crs)
    if not has_geodata(img_path):
        print("🚫 Non-georeferenced image — skipping GeoJSON projection and crown stitching.")
        return

    # === 8. Convert predictions to GeoJSON ===
    print("\n🗺️  Projecting predictions to GeoJSON ...")
    project_to_geojson(pred_tiles_path, preds_path, preds_geo_path)
    print("✅ GeoJSON projection complete.")

    # # === 9. Stitch, clean, and simplify crowns ===
    # print("\n🌿 Stitching and cleaning crown polygons ...")
    # crowns = stitch_crowns(preds_geo_path) # what is nproc?
    # clean = clean_crowns(crowns, 0.6, confidence=0.5)
    # clean = clean.set_geometry(clean.geometry.simplify(0.3))
    # print("✅ Crowns cleaned and simplified.")

    # # === 10. Write final output ===
    # out_gpkg = site_path / "crowns_out.gpkg"
    # clean.to_file(out_gpkg)
    # print(f"\nAll done! Results saved to:\n  {out_gpkg}\n")


    # === 11. Validate predictions vs TCD segments ===  
    _, pred, gt, ious, coco_anns = validate_predictions_vs_tcd_segments(
        pred_geojson_path="data/tcd/tiles_pred/predictions_geo/Prediction_tcd_tile_0_tile.geojson",
        tcd_example=example,
        iou_thresh=0.5
    )

    # === 12. Visualize validation results ===
    visualize_validation_results(
        pred, gt, ious,
        coco_anns,
        iou_thresh=0.5,
        site_path=site_path,
        rgb_path=img_path,
        tile_name=f"{img_path.stem}_tile",
        image_id=image_id
    )

    

# CLI Args
def parse_args():
    ap = argparse.ArgumentParser(description="Detectree2 runner")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run non-geospatial smoke test on a single RGB image."
    )
    return ap.parse_args()

def set_device(cfg):
    # Prefer Apple MPS, else CPU (no CUDA on Apple Silicon)
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    cfg.MODEL.DEVICE = device
    print(f"🖥️ Using device: {device}")





from rasterio.transform import from_bounds
def download_one_tcd_tile(save_dir: Path) -> tuple[Path, Path]:
    """
    Download one orthomosaic (.tif) and crowns annotation (.gpkg) from restor/tcd,
    and ensure the image is saved as a fully georeferenced GeoTIFF (with CRS & transform).
    """
    print("📦 Loading TCD dataset metadata...")
    ds = load_dataset("restor/tcd", split="train")
    example = ds[0]
    image_id = example["image_id"]
    print("Available keys:", list(example.keys()))

    save_dir.mkdir(parents=True, exist_ok=True)
    img_path = save_dir / "tcd_tile_0.tif"
    ann_path = save_dir / "tcd_tile_0_crowns.gpkg"

    # --- Image ---
    img = np.array(example["image"])  # PIL → NumPy
    height, width = img.shape[:2]

    # Extract CRS and bounds from metadata
    crs = example.get("crs")
    bounds = example.get("bounds")
    if crs is None or bounds is None:
        raise ValueError("❌ Dataset entry missing 'crs' or 'bounds' — cannot proceed without georef info.")

    transform = from_bounds(*bounds, width=width, height=height)

    with rasterio.open(
        img_path, "w", driver="GTiff",
        height=height, width=width, count=3,
        dtype=img.dtype, crs=crs, transform=transform
    ) as dst:
        for i in range(3):
            dst.write(img[:, :, i], i + 1)

    print(f"✅ Saved georeferenced tile → {img_path}")
    print(f"📐 CRS: {crs}")
    print(f"🔢 Bounds: {bounds}")

    # --- Annotations ---
    crown_url = example.get("annotation") or example.get("crowns_polygon")
    if isinstance(crown_url, str) and crown_url.startswith("http"):
        print(f"🌿 Downloading crown polygons from {crown_url}")
        with requests.get(crown_url, stream=True) as r:
            r.raise_for_status()
            with open(ann_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✅ Saved annotations → {ann_path}")
    else:
        print("⚠️ No valid annotation URL found; skipping crowns download.")

    return img_path, ann_path, example, image_id

def is_australia(example):
    return (example["lat"] >= -44 and example["lat"] <= -10
            and example["lon"] >= 112 and example["lon"] <= 154)

# ds_au = ds.filter(is_australia) # example usage


def visualize_saved_prediction_with_masks(img_path, pred_json_path, out_path, score_thresh=0.5):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")

    with open(pred_json_path) as f:
        data = json.load(f)

    detections = [d for d in data if d.get("score", 0) >= score_thresh]
    if len(detections) == 0:
        print(f"⚠️ No detections above threshold ({score_thresh}) in {pred_json_path}")
        return

    # --- Boxes ---
    boxes = torch.tensor([d["bbox"] for d in detections], dtype=torch.float32)
    boxes[:, 2:] += boxes[:, :2]

    # --- Masks (decode directly from compressed RLE) ---
    masks = []
    for d in detections:
        seg = d.get("segmentation")
        if seg and "counts" in seg:
            m = mask_utils.decode(seg)  # ✅ directly decode
            if m.ndim == 3:
                m = np.any(m, axis=2)  # collapse if multi-channel
            masks.append(m)
        else:
            masks.append(np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8))
    masks = torch.as_tensor(np.stack(masks))  # [N, H, W]

    # --- Instances ---
    scores = torch.tensor([d["score"] for d in detections])
    classes = torch.tensor([d.get("category_id", 0) for d in detections])

    instances = Instances((img.shape[0], img.shape[1]))
    instances.pred_boxes = Boxes(boxes)
    instances.scores = scores
    instances.pred_classes = classes
    instances.pred_masks = masks

    # Custom labels ("Tree XX%")
    labels = [f"Tree {s * 100:.0f}%" for s in instances.scores]

    vis = Visualizer(img[:, :, ::-1], scale=1.0)
    vis_out = vis.overlay_instances(
        boxes=instances.pred_boxes,
        masks=instances.pred_masks,
        labels=labels
    )

    cv2.imwrite(str(out_path), vis_out.get_image()[:, :, ::-1])
    print(f"✅ Saved overlay with masks → {out_path}")


def has_geodata(tif_path: str | Path) -> bool:
    """Return True if GeoTIFF has valid CRS and affine transform."""
    with rasterio.open(tif_path) as ds:
        # CRS and transform must both be defined and not identity
        has_crs = ds.crs is not None
        has_transform = ds.transform != rasterio.Affine.identity()
        return has_crs and has_transform



def validate_predictions_vs_tcd_segments(pred_geojson_path, tcd_example, iou_thresh=0.5, score_thresh=0.8):
    """Validate Detectree2 predictions against TCD 'segments' (bbox/segmentation polygons)."""
    print("📂 Loading Detectree2 predictions ...")
    pred = gpd.read_file(pred_geojson_path)
    print(f"  → {len(pred)} predicted polygons")

    # Filter low-confidence predictions
    if "score" in pred.columns:
        before = len(pred)
        pred = pred[pred["score"] >= score_thresh].copy()
        print(f"  → Filtered {before - len(pred)} low-confidence predictions (score < {score_thresh})")

    # Parse and normalize ground-truth segments
    segs = tcd_example.get("segments")
    if not segs:
        raise ValueError("❌ No 'segments' found in TCD example.")

    if isinstance(segs, str):  # JSON string → list
        segs = json.loads(segs)
    if not isinstance(segs, list):
        raise ValueError("❌ 'segments' field not in expected list format.")

    # gt_polys = []
    # for s in segs:
    #     if isinstance(s, str):
    #         try:
    #             s = json.loads(s)
    #         except json.JSONDecodeError:
    #             continue
    #     if not isinstance(s, dict):
    #         continue

    #     if "segmentation" in s and isinstance(s["segmentation"], list):
    #         coords = np.array(s["segmentation"][0]).reshape(-1, 2)
    #         poly = Polygon(coords)
    #     elif "bbox" in s:
    #         x, y, w, h = s["bbox"]
    #         poly = Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
    #     else:
    #         continue

    #     if poly.is_valid and poly.area > 0:
    #         gt_polys.append(poly)

    # # --- Convert pixel polygons to world CRS ---

    # bounds = tcd_example["bounds"]
    # width, height = tcd_example["width"], tcd_example["height"]
    # transform = from_bounds(*bounds, width=width, height=height)

    # gt_world = [pixel_to_world(p, transform) for p in gt_polys if p.is_valid]
    # gt = gpd.GeoDataFrame(geometry=gt_world, crs=tcd_example["crs"])

    # --- Convert COCO-style pixel polygons to world CRS ---
    coco_annotations = tcd_example.get("coco_annotations", [])

    if isinstance(coco_annotations, str):
        try:
            coco_annotations = json.loads(coco_annotations)
        except json.JSONDecodeError:
            raise ValueError("❌ 'coco_annotations' field is not valid JSON.")
    if not isinstance(coco_annotations, list) or not coco_annotations:
        raise ValueError("❌ No valid 'coco_annotations' found in TCD example — cannot derive crowns.")

    print(f"Annotations 1: {len(coco_annotations)} total")
    #  if this works, it feels inefficient to filter here after we've already pulled the coco_anns - update to do it sooner
    # Update: not sure this is needed or actually doing anything - 52 before filter, still 52 after
    image_id = tcd_example.get("image_id")
    if image_id is not None:
        coco_annotations = [a for a in coco_annotations if a.get("image_id") == image_id]
        print(f"Filtered to {len(coco_annotations)} annotations for image_id={image_id}")
    else:
        print("⚠️ No image_id found; using all annotations.")


    #     #just to check...
    # # =============================================================
    # print(f"Annotations: {len(coco_annotations)} total")
    # check_cats = [a.get("category_id") for a in coco_annotations]
    # print(f"  Canopy: {check_cats.count(1)}, Trees: {check_cats.count(2)}")

    # for i, a in enumerate(coco_annotations[:3]):
    #     print(f"Ann {i}: seg count {len(a.get('segmentation', []))}, first seg len={len(a['segmentation'][0]) if a['segmentation'] else 0}")

    #     # =============================================================

    gt_polys = []
    gt_cats = []
    for ann in coco_annotations:
        segs = ann.get("segmentation", [])
        if not segs or not isinstance(segs[0], list):
            continue
        coords = np.array(segs[0]).reshape(-1, 2)
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = make_valid(poly) if hasattr(make_valid, "__call__") else poly.buffer(0)
        if poly.is_valid and poly.area > 0:
            # print(f"Ann {ann.get('id')}: {ann.get('category_id')}")
            gt_polys.append(poly)
            gt_cats.append(ann.get("category_id", 1))
        else:
            print(f"Ann {ann.get('id')}: Invalid polygon")
            print(f"  Area: {poly.area}")
            print(f"  Is valid: {poly.is_valid}")
            print(f"  Is empty: {poly.is_empty}")
            print(f"  Coords: {coords}")
            print(f"  Segs: {segs}")
            print(f"  Segs[0]: {segs[0]}")
            print(f"  Segs[0][0]: {segs[0][0]}")
            print(f"  Segs[0][1]: {segs[0][1]}")

        # print(f"Segs length: {len(segs)}")
        # if len(segs) > 1:
        #     print("WOAH")

    # This print is "GT Cats [] - Canopy: 3, Trees: 48", but coco_annotations has 52 entries - Help!
    print(f" GT Cats [] - Canopy: {gt_cats.count(1)}, Trees: {gt_cats.count(2)}")

    # Use image georeferencing transform to map pixel → world
    width, height = tcd_example["width"], tcd_example["height"]
    bounds = tcd_example["bounds"]
    transform = from_bounds(*bounds, width=width, height=height)

    # gt_world = [pixel_to_world(p, transform) for p in gt_polys if p.is_valid]
    # gt = gpd.GeoDataFrame(geometry=gt_world, crs=tcd_example["crs"])

    gt_world = [pixel_to_world(p, transform) for p in gt_polys if p.is_valid]
    gt = gpd.GeoDataFrame({"geometry": gt_world, "category": gt_cats}, crs=tcd_example["crs"])

    print(f"  → {len(gt)} ground-truth polygons (from coco_annotations)")

    # gt = gpd.GeoDataFrame(geometry=gt_polys, crs=tcd_example.get("crs", "EPSG:3395"))
    print(f"  → {len(gt)} ground-truth polygons")

    if pred.crs != gt.crs:
        pred = pred.to_crs(gt.crs)

    # IoU computation
    def iou(a, b):
        inter, union = a.intersection(b).area, a.union(b).area
        return inter / union if union > 0 else 0.0

    def iop(a, b):
        inter = a.intersection(b).area
        denom = a.area
        return inter / denom if denom > 0 else 0.0

    scores = []
    for p in pred.geometry:
        if not p.is_valid or p.is_empty:
            continue
        best_score = 0.0
        for g, cat in zip(gt.geometry, gt["category"]):
            score = iou(p, g) if cat == 1 else iop(p, g)
            if score > best_score:
                best_score = score
        scores.append(best_score)

    n_pred, n_gt = len(pred), len(gt)
    n_tp = sum(s >= iou_thresh for s in scores)
    mean_score = np.mean(scores) if scores else 0

    metrics = {
        "precision": n_tp / n_pred if n_pred else 0,
        "recall": n_tp / n_gt if n_gt else 0,
        "mean_overlap": mean_score,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_tp": n_tp,
    }

    print("\n📊 Validation Results (IoU for trees, IoP for canopy):")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.3f}" if isinstance(v, float) else f"  {k:12s}: {v}")

    print("GT CRS:", tcd_example.get("crs"))
    print("Pred CRS:", pred.crs)
    print("GT number:", len(gt_polys))
    print("Pred number:", len(pred))

    print(gt.geometry.iloc[0].bounds)
    print(pred.geometry.iloc[0].bounds)

    return metrics, pred, gt, scores, coco_annotations


def visualize_validation_results(pred, gt, ious, coco_anns=None, iou_thresh=0.5,
                                 site_path=None, rgb_path=None,
                                 tile_name=None, image_id=None):
    """
    Visualize Detectree2 vs TCD polygons over RGB base image.

    Colors:
      🟩 Teal   → True Positives (IoU ≥ threshold)
      🟧 Orange → False Positives (IoU < threshold)
      🟪 Purple → Ground Truth crowns
    """

    # === 1. Output path ===
    out_dir = Path(site_path) / "overlays_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = ["validation_overlay"]
    if image_id:
        parts.append(str(image_id))
    if tile_name:
        parts.append(tile_name)
    out_path = out_dir / f"{'_'.join(parts)}.png"

    # === 2. Load background image ===
    img, extent = None, None
    if rgb_path and Path(rgb_path).exists():
        with rasterio.open(rgb_path) as src:
            img = src.read([1, 2, 3])
            img = np.moveaxis(img, 0, -1)
            img = (img - img.min()) / (img.max() - img.min() + 1e-9)
            extent = [src.bounds.left, src.bounds.right,
                      src.bounds.bottom, src.bounds.top]

    # === 3. Plot setup ===
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Detectree2 vs TCD — IoU ≥ {iou_thresh}")
    ax.set_aspect("equal")

    if img is not None:
        ax.imshow(img, extent=extent, origin="upper")

    # === 4. Draw predictions ===
    for i, p in enumerate(pred.geometry):
        if not p.is_valid or p.is_empty:
            continue
        color = "#00F0FF" if i < len(ious) and ious[i] >= iou_thresh else "#FF8800"  # teal / orange
        ax.plot(*p.exterior.xy, color=color, linewidth=1.2, alpha=0.9)

    # === 5. Draw ground-truth crowns & canopy ===
    for idx, g in enumerate(gt.geometry):
        if not g.is_valid or g.is_empty:
            continue
        # Match color by category (1=tree, 2=canopy)
        cat = None
        if coco_anns and idx < len(coco_anns):
            cat = coco_anns[idx].get("category_id", 1)
        color = "#C266FF" if cat == 1 else "#0a20ad"  # purple vs dark-blue
        if isinstance(g, Polygon):
            geoms = [g]
        elif isinstance(g, MultiPolygon):
            geoms = list(g.geoms)
        elif isinstance(g, GeometryCollection):
            geoms = [geom for geom in g.geoms if isinstance(geom, (Polygon, MultiPolygon))]
        else:
            geoms = []

        for geom in geoms:
            if isinstance(geom, Polygon):
                x, y = geom.exterior.xy
                ax.fill(x, y, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.25)
                ax.plot(x, y, color=color, linewidth=0.8, alpha=0.9)
            elif isinstance(geom, MultiPolygon):
                for sub in geom.geoms:
                    x, y = sub.exterior.xy
                    ax.fill(x, y, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.25)
                    ax.plot(x, y, color=color, linewidth=0.8, alpha=0.9)

    # === 6. Legend ===
    import matplotlib.patches as mpatches
    legend_elems = [
        mpatches.Patch(color="#00F0FF", label="True Positive (IoU ≥ threshold)"),
        mpatches.Patch(color="#FF8800", label="False Positive (IoU < threshold)"),
        mpatches.Patch(color="#C266FF", label="Ground Truth — Canopy"),
        mpatches.Patch(color="#0a20ad", label="Ground Truth — Tree")
    ]

    ax.legend(handles=legend_elems, loc="lower right", frameon=True, fontsize=8)

    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close(fig)

    print(f"🖼️  Saved validation overlay → {out_path}")

    # === 7. Also save pure Ground Truth visualization for comparison ===

    fig_gt, ax_gt = plt.subplots(figsize=(8, 8))
    ax_gt.set_title(f"TCD Ground Truth (image_id={image_id})")
    ax_gt.set_aspect("equal")

    if img is not None:
        ax_gt.imshow(img, extent=extent, origin="upper")

    for idx, g in enumerate(gt.geometry):
        if g.is_empty:
            continue

        # Category colouring
        cat = None
        if coco_anns and idx < len(coco_anns):
            cat = coco_anns[idx].get("category_id", 1)
        color = "#C266FF" if cat == 1 else "#0a20ad"  # purple=canopy, blue=tree

        # Handle all geometry types safely
        if isinstance(g, Polygon):
            geoms = [g]
        elif isinstance(g, MultiPolygon):
            geoms = list(g.geoms)
        elif isinstance(g, GeometryCollection):
            geoms = [geom for geom in g.geoms if isinstance(geom, (Polygon, MultiPolygon))]
        else:
            geoms = []

        for geom in geoms:
            if isinstance(geom, Polygon):
                x, y = geom.exterior.xy
                ax_gt.fill(x, y, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.25)
                ax_gt.plot(x, y, color=color, linewidth=0.8, alpha=0.9)
            elif isinstance(geom, MultiPolygon):
                for sub in geom.geoms:
                    x, y = sub.exterior.xy
                    ax_gt.fill(x, y, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.25)
                    ax_gt.plot(x, y, color=color, linewidth=0.8, alpha=0.9)

    ax_gt.set_xlabel("Easting")
    ax_gt.set_ylabel("Northing")

    out_gt_path = out_path.with_name(out_path.stem + "_groundtruth.png")
    plt.tight_layout()
    plt.savefig(out_gt_path, dpi=250)
    plt.close(fig_gt)

    print(f"🗺️  Saved pure ground-truth overlay → {out_gt_path}")
# ======================

    # fig_gt, ax_gt = plt.subplots(figsize=(8, 8))
    # ax_gt.set_title(f"TCD Ground Truth (image_id={image_id})")
    # ax_gt.set_aspect("equal")

    # if img is not None:
    #     ax_gt.imshow(img, extent=extent, origin="upper")

    # for idx, g in enumerate(gt.geometry):
    #     if not g.is_valid or g.is_empty:
    #         continue
    #     cat = None
    #     if coco_anns and idx < len(coco_anns):
    #         cat = coco_anns[idx].get("category_id", 1)
    #     color = "#C266FF" if cat == 1 else "#0a20ad"  # purple=tree, blue=canopy
    #     x, y = g.exterior.xy
    #     ax_gt.fill(x, y, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.25)
    #     ax_gt.plot(x, y, color=color, linewidth=0.8, alpha=0.9)

    # ax_gt.set_xlabel("Easting")
    # ax_gt.set_ylabel("Northing")

    # out_gt_path = out_path.with_name(out_path.stem + "_groundtruth.png")
    # plt.tight_layout()
    # plt.savefig(out_gt_path, dpi=250)
    # plt.close(fig_gt)

    # print(f"🗺️  Saved pure ground-truth overlay → {out_gt_path}")



    # for idx, g in enumerate(gt.geometry):
    #     if g.is_empty:
    #         continue

    #     # Match color by category (1 = tree, 2 = canopy)
    #     cat = None
    #     if coco_anns and idx < len(coco_anns):
    #         cat = coco_anns[idx].get("category_id", 1)
    #     color = "#C266FF" if cat == 1 else "#0a20ad"  # purple vs dark-blue

    #     # --- Handle all geometry types safely ---
    #     if isinstance(g, Polygon):
    #         geoms = [g]
    #     elif isinstance(g, MultiPolygon):
    #         geoms = list(g.geoms)
    #     elif isinstance(g, GeometryCollection):
    #         geoms = [geom for geom in g.geoms if isinstance(geom, (Polygon, MultiPolygon))]
    #     else:
    #         geoms = []

    #     for geom in geoms:
    #         if isinstance(geom, Polygon):
    #             xs, ys = geom.exterior.xy
    #             ax.fill(xs, ys, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.25)
    #             ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.8)
    #         elif isinstance(geom, MultiPolygon):
    #             for sub in geom.geoms:
    #                 xs, ys = sub.exterior.xy
    #                 ax.fill(xs, ys, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.25)
    #                 ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.8)



def pixel_to_world(poly, transform):
    """
    Convert a polygon from pixel to world (map) coordinates using a rasterio Affine transform.
    Works safely with both Affine objects and tuple/list transforms.
    """
    # Ensure we have 6 coefficients in order: [a, b, d, e, xoff, yoff]
    if isinstance(transform, rasterio.Affine):
        coeffs = [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]
    else:
        a, b, c, d, e, f = transform[:6]
        coeffs = [a, b, d, e, c, f]

    return affine_transform(poly, coeffs)

# --------------------------------------------------
# Entrypoint
# --------------------------------------------------
if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = parse_args()
    model_path = Path("230103_randresize_full.pth")

    if args.smoke:
        smoke_test(model_path)
    else:
        main()