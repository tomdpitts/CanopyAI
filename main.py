#!/usr/bin/env python3
"""
Detectree2 end-to-end inference script (local version)
-----------------------------------------------------
Runs the full Detectree2 workflow:
  1. Tiles an orthomosaic
  2. Runs Detectron2 model inference
  3. Projects predictions to GeoJSON
  4. Stitches and cleans crowns
  5. Writes the output GeoPackage
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

    model_name = "230103_randresize_full"

    # === 1. Define key paths ===
    # Adjust this base path if your dataset lives elsewhere
    home = Path.home()
    site_name = "TCD"
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

    # # === 5. Tile orthomosaic ===
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
        Path(pred_tiles_path) / "predictions" / f"Prediction_{tile_name}.json",
        Path(overlays_path) / f"{tile_name}_overlay.png",
        score_thresh=0.5
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


    validate_predictions_vs_tcd_segments(
        pred_geojson_path="data/tcd/tiles_pred/predictions_geo/Prediction_tcd_tile_5_tile.geojson",
        tcd_example=example,
        iou_thresh=0.5
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


from datasets import load_dataset
import requests
from pathlib import Path

# def download_one_tcd_tile(save_dir: Path) -> Path:
#     """Download a single tile (.tif) from the restor/tcd dataset."""
#     ds = load_dataset("restor/tcd", split="train")  # load metadata

#     # Examine first example
#     first = ds[0]
#     print("Available keys:", first.keys())

#     # The dataset stores imagery in a field like 'image' (a PIL.Image or link)
#     # If it's a dict with 'path' or 'url', handle accordingly:
#     img_field = first.get("image")

#     # Case 1: it's a PIL image — save it manually
#     if hasattr(img_field, "save"):
#         save_dir.mkdir(parents=True, exist_ok=True)
#         out_path = save_dir / "tcd_tile_0.tif"
#         img_field.save(out_path)
#         print(f"🪶 Saved tile to {out_path} (PIL)")
#         return out_path

#     # Case 2: it's a remote URL (common for large EO datasets)
#     if isinstance(img_field, str) and img_field.startswith("http"):
#         save_dir.mkdir(parents=True, exist_ok=True)
#         out_path = save_dir / Path(img_field).name
#         if not out_path.exists():
#             print(f"🌐 Downloading {img_field} ...")
#             with requests.get(img_field, stream=True) as r:
#                 r.raise_for_status()
#                 with open(out_path, "wb") as f:
#                     for chunk in r.iter_content(chunk_size=8192):
#                         f.write(chunk)
#         print(f"✅ Saved tile to {out_path} (remote URL)")
#         return out_path

#     raise ValueError(f"Unexpected format for 'image' field: {type(img_field)}")

from datasets import load_dataset
from pathlib import Path
import requests
import rasterio

# def download_one_tcd_tile(save_dir: Path) -> tuple[Path, Path]:
#     """
#     Download a single orthomosaic (.tif) and corresponding crown annotations (.gpkg)
#     from the Restor Foundation TCD dataset on HuggingFace.
    
#     Handles both PIL.Image and remote URL formats for imagery.
#     Returns (image_path, annotation_path).
#     """
#     print("📦 Loading TCD dataset metadata...")
#     ds = load_dataset("restor/tcd", split="train")

#     # Inspect first example
#     first = ds[0]
#     print("Available keys:", list(first.keys()))

#     img_field = first.get("image")
#     crown_field = first.get("crowns_polygon")

#     save_dir.mkdir(parents=True, exist_ok=True)
#     img_path = save_dir / "tcd_tile_0.tif"
#     ann_path = save_dir / "tcd_tile_0_crowns.gpkg"

#     # --- 🖼️ IMAGE HANDLING ---
#     if hasattr(img_field, "save"):
#         # Case: PIL.Image object (no CRS)
#         img_field.save(img_path)
#         print(f"🪶 Saved tile (PIL) → {img_path}")
#     elif isinstance(img_field, dict) and "path" in img_field:
#         # Case: dataset stores file path or URL inside dict
#         url = img_field["path"]
#         print(f"🌐 Downloading GeoTIFF from {url} ...")
#         with requests.get(url, stream=True) as r:
#             r.raise_for_status()
#             with open(img_path, "wb") as f:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     f.write(chunk)
#         print(f"✅ Saved tile (from dict URL) → {img_path}")
#     elif isinstance(img_field, str) and img_field.startswith("http"):
#         # Case: direct URL string
#         print(f"🌐 Downloading GeoTIFF from {img_field} ...")
#         with requests.get(img_field, stream=True) as r:
#             r.raise_for_status()
#             with open(img_path, "wb") as f:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     f.write(chunk)
#         print(f"✅ Saved tile (remote URL) → {img_path}")
#     else:
#         raise ValueError(f"Unexpected format for 'image' field: {type(img_field)}")

#     # --- 🌿 ANNOTATION HANDLING ---
#     if isinstance(crown_field, str) and crown_field.startswith("http"):
#         print(f"🌐 Downloading crowns annotation from {crown_field} ...")
#         with requests.get(crown_field, stream=True) as r:
#             r.raise_for_status()
#             with open(ann_path, "wb") as f:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     f.write(chunk)
#         print(f"✅ Saved annotations → {ann_path}")
#     elif crown_field is None:
#         print("⚠️ No crowns_polygon field found in this record.")
#     else:
#         print(f"⚠️ Unexpected crowns format: {type(crown_field)}")

#     # --- 🧭 Validate geospatial metadata ---
#     try:
#         with rasterio.open(img_path) as ds_img:
#             print(f"📐 CRS: {ds_img.crs}")
#             print(f"🔢 Transform: {ds_img.transform}")
#             if ds_img.crs is None:
#                 print("⚠️ Image has no CRS — non-georeferenced tile.")
#     except Exception as e:
#         print(f"⚠️ Could not read GeoTIFF metadata: {e}")

#     print("\n✅ Download complete.")
#     return img_path, ann_path
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

    print("Annotation: ", example["annotation"])
    print("Segments: ", example["segments"])
    print("COCO Annotations: ", example["coco_annotations"])

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
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

def validate_predictions_vs_tcd_segments(pred_geojson_path, tcd_example, iou_thresh=0.5):
    """
    Validate Detectree2 polygon predictions against TCD ground truth 'segments'.

    Parameters
    ----------
    pred_geojson_path : str or Path
        Path to Detectree2's GeoJSON predictions (one tile).
    tcd_example : dict
        One dataset example from restor/tcd (ds[i]).
        Must contain 'segments' and 'crs' keys.
    iou_thresh : float, optional
        Intersection-over-Union threshold for counting True Positives.

    Returns
    -------
    dict
        Summary metrics: precision, recall, mean_iou, n_pred, n_gt, n_tp
    """
    print("📂 Loading Detectree2 predictions ...")
    pred = gpd.read_file(pred_geojson_path)
    print(f"  → {len(pred)} predicted polygons")

    # --- Build ground-truth polygons from TCD 'segments' ---
    segs = tcd_example.get("segments", [])
    if not segs:
        raise ValueError("❌ No 'segments' found in TCD example — cannot validate.")

    gt_polys = []
    for seg in segs:
        # The dataset provides 'bbox' in [x, y, width, height]
        # Some may also have 'segmentation' arrays — prefer them if available
        if "segmentation" in seg and isinstance(seg["segmentation"], list):
            # segmentation is [[x1,y1,x2,y2,...]] — convert to Polygon
            coords = seg["segmentation"][0]
            poly = Polygon(np.array(coords).reshape(-1, 2))
        else:
            x, y, w, h = seg["bbox"]
            poly = Polygon([
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h)
            ])
        if poly.is_valid and poly.area > 0:
            gt_polys.append(poly)

    gt = gpd.GeoDataFrame(geometry=gt_polys, crs=tcd_example.get("crs", "EPSG:3395"))
    print(f"  → {len(gt)} ground-truth polygons from TCD segments")

    # --- CRS alignment ---
    if pred.crs != gt.crs:
        print(f"🔄 Aligning CRS from {pred.crs} → {gt.crs}")
        pred = pred.to_crs(gt.crs)

    # --- IoU helper ---
    def iou(a, b):
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0

    # --- Compute IoUs ---
    ious = []
    for p in pred.geometry:
        if not p.is_valid or p.is_empty:
            continue
        best = max((iou(p, g) for g in gt.geometry), default=0.0)
        ious.append(best)

    n_pred = len(pred)
    n_gt = len(gt)
    n_tp = sum(i >= iou_thresh for i in ious)
    precision = n_tp / n_pred if n_pred else 0
    recall = n_tp / n_gt if n_gt else 0
    mean_iou = np.mean(ious) if ious else 0

    metrics = {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_tp": n_tp,
    }

    print("\n📊 Validation Results:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.3f}" if isinstance(v, float) else f"  {k:10s}: {v}")

    return metrics

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