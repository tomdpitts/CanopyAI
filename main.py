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

import numpy as np
from shapely.geometry import Polygon, MultiPolygon


import numpy as np
import json
from shapely.geometry import Polygon

import matplotlib.pyplot as plt

import rasterio
from rasterio.transform import from_bounds
from shapely.affinity import affine_transform

from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.strtree import STRtree
import pandas as pd

# Key Hyperparameters
max_images = 30
filter_threshold = 0.8


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
    home = Path.home()
    site_name = "tcd"
    site_path = home / "dphil" / "detectree2" / "data" / site_name

    # === 1b. Download multiple tiles ===
    tiles_info = download_tcd_tiles_streaming(site_path / "raw", max_images=max_images)
    print(f"✅ Downloaded {len(tiles_info)} TCD tiles for processing.")
    
    
    # === 2. Create output/working directories ===
    pred_tiles_path = ensure_dir(site_path / "tiles_pred")
    preds_path = ensure_dir(Path(pred_tiles_path) / "predictions")
    preds_geo_path = ensure_dir(Path(pred_tiles_path) / "predictions_geo")
    # overlays_path = ensure_dir(site_path / "overlays")

    # === 3. Download pretrained model if missing ===
    model_path = Path(f"{model_name}.pth")
    if not model_path.exists():
        url = f"https://zenodo.org/records/10522461/files/{model_name}.pth"
        print(f"📦 Model not found locally — downloading from {url} ...")
        wget.download(url, out=str(model_path))
        print("\n✅ Model download complete.")

    # === 4. Initialize Detectron2 predictor (once) ===
    print("\n⚙️  Initializing Detectron2 predictor ...")
    cfg = setup_cfg(update_model=str(model_path))
    set_device(cfg)
    predictor = DefaultPredictor(cfg)
    print("✅ Predictor ready.")

    # === 5–12. Process each tile ===
    for img_path, ann_path, tile_info, image_id in tiles_info:
        print(f"\n================ Processing {image_id} ================")
        print(f"Biome: {tile_info.get('biome_name', 'N/A')}")

        # tile_dir = Path(pred_tiles_path)
        # ------------------------------------------------------------
        # 5. Tile orthomosaic into chips for inference
        # ------------------------------------------------------------
        print("\n🧩 Tiling image into smaller chips ...")

        chip_dir = Path(pred_tiles_path) / f"{img_path.stem}_chips"
        ensure_dir(chip_dir)

        buffer = 10
        tile_width = 40
        tile_height = 40

        try:
            # Your Detectree2 tiler (no CRS loss if the input GeoTIFF is georeferenced)
            tile_data(
                str(img_path),
                chip_dir,        # output directory
                buffer,
                tile_width,
                tile_height,
                dtype_bool=True  # Detectree2 expects this for mask chips
            )
            print("✅ Tiling complete.")
        except AttributeError as e:
            print(f"⚠️ Non-georeferenced image — skipping CRS: {e}")
            
        # If no tiles were created, just skip this image and continue.
        chips = list(Path(chip_dir).glob("*.tif"))
        if len(chips) == 0:
            print(f"⚠️  Skipping {image_id} — no tiles produced (likely nodata or invalid raster).")
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
        filter_raw_predictions(chip_pred_dir, score_thresh=filter_threshold, overwrite=True)

        # ------------------------------------------------------------
        # 8. Reproject tiled predictions → GeoJSON in global CRS
        # ------------------------------------------------------------
        print("\n🗺️  Projecting tile predictions to GeoJSON ...")
        chip_geo_dir = chip_dir / "predictions_geo"
        ensure_dir(chip_geo_dir)

        project_to_geojson(
            tiles_path=chip_dir,
            pred_fold=chip_pred_dir,
            output_fold=chip_geo_dir
        )
        print("✅ GeoJSON projection complete.")

        # ------------------------------------------------------------
        # 9. Visualize & Validate on the *merged* predictions
        # ------------------------------------------------------------
        # Detectree2 produces one GeoJSON per tile — merge them for evaluation
        merged_geojson = chip_geo_dir / f"{img_path.stem}_merged.geojson"
        merge_tile_geojsons(chip_geo_dir, merged_geojson)

        # visualize_saved_prediction_with_masks(
        #     img_path,                     # original tile
        #     merged_geojson,               # merged prediction mask JSON
        #     overlays_path,
        #     image_id
        # )

        metrics_all, pred, gt, ious, coco_anns = clean_validate_predictions_vs_tcd_segments(
            pred_geojson_path=merged_geojson,
            tcd_example=tile_info
        )

        visualize_validation_results(
            pred, gt, ious,
            coco_anns,
            site_path=site_path,
            rgb_path=img_path,
            tile_name=img_path.stem,
            image_id=image_id
        )

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
    tile_info = ds[0]
    image_id = tile_info["image_id"]
    print("Available keys:", list(tile_info.keys()))

    save_dir.mkdir(parents=True, exist_ok=True)
    img_path = save_dir / "tcd_tile_0.tif"
    ann_path = save_dir / "tcd_tile_0_crowns.gpkg"

    # --- Image ---
    img = np.array(tile_info["image"])  # PIL → NumPy
    height, width = img.shape[:2]

    # Extract CRS and bounds from metadata
    crs = tile_info.get("crs")
    bounds = tile_info.get("bounds")
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
    crown_url = tile_info.get("annotation") or tile_info.get("crowns_polygon")
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

    return img_path, ann_path, tile_info, image_id

def download_tcd_tiles(save_dir: Path, max_images: int = 3):
    """
    Download multiple orthomosaics (.tif) and crowns annotation (.gpkg) from restor/tcd.
    Returns a list of tuples: (img_path, ann_path, example, image_id)
    """
    print("📦 Loading TCD dataset metadata...")
    ds = load_dataset("restor/tcd", split="train")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ds_au = ds.filter(is_australia)


    all_entries = []
    for i, tile_info in enumerate(ds.select(range(max_images))):
        image_id = tile_info["image_id"]
        print(f"📸 Downloading image {i}: {image_id}")

        img_path = save_dir / f"tcd_tile_{i}.tif"
        ann_path = save_dir / f"tcd_tile_{i}_crowns.gpkg"

        img = np.array(tile_info["image"])
        height, width = img.shape[:2]
        crs = tile_info.get("crs")
        bounds = tile_info.get("bounds")
        if crs is None or bounds is None:
            print(f"⚠️ Skipping {image_id} — missing CRS/bounds.")
            continue

        transform = from_bounds(*bounds, width=width, height=height)
        with rasterio.open(
            img_path, "w", driver="GTiff",
            height=height, width=width, count=3,
            dtype=img.dtype, crs=crs, transform=transform
        ) as dst:
            for b in range(3):
                dst.write(img[:, :, b], b + 1)

        crown_url = tile_info.get("annotation") or tile_info.get("crowns_polygon")
        if isinstance(crown_url, str) and crown_url.startswith("http"):
            with requests.get(crown_url, stream=True) as r:
                r.raise_for_status()
                with open(ann_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        all_entries.append((img_path, ann_path, tile_info, image_id))
        print(f"✅ Saved georeferenced tile → {img_path}")

    return all_entries

def is_australia(x):
    return (
        -44 <= x["lat"] <= -10 and
        112 <= x["lon"] <= 154
    )
aus_tiles = [
1207, 4347, 4159, 4893, 4406, 2100, 1104, 3956, 2859, 3684, 5001, 3469,
5012, 4660, 536, 4315, 4506, 3624, 4127, 4963, 423, 1703, 3016, 4643,
922, 4221, 4955, 4909, 3219, 1671, 195, 4923, 4556, 4086, 1969, 3611,
4336, 2581, 1033, 314, 2491, 4720, 4421, 5005, 4435, 2654, 62, 4593,
5057, 1612, 1417, 1278, 2403, 2270, 367, 1339, 1117, 4507, 4040, 577,
439, 2888, 4326, 1875, 760, 678, 3456, 4108, 1029, 2515, 4996, 876,
4639, 4933, 2031, 750, 1248, 743, 2293, 3277, 3875, 1077
]

def is_really_australia(x):
    return x["image_id"] in aus_tiles

rangeland = [
    "West Sudanian savanna",
    "East Sudanian savanna",
    "Northern Congolian forest-savanna mosaic",
    "Northern mixed grasslands",
    "Central and Southern mixed grasslands",
    "Pontic steppe",
    "East European forest steppe",
    "Snake-Columbia shrub steppe",
    "Chilean matorral",
    "Central Mexican matorral",
    "Sechura desert",
    "Sonoran desert",
    "Gulf of Oman desert and semi-desert",
    "Low Monte",
    "Central Andean puna",
    "Central Andean dry puna",
    "Zambezian and Mopane woodlands"
]

def is_rangeland(x):
    return x["biome_name"] in rangeland

from pathlib import Path
from datasets import load_dataset
import requests

def download_tcd_tiles_streaming(save_dir: Path, max_images: int = 3):
    print("📦 Loading TCD dataset in streaming mode...")
    ds = load_dataset("restor/tcd", split="train", streaming=True)

    save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    count = 0

    for example in ds:
        if not is_rangeland(example):
            print("Nope!")
            continue

        # stop early
        if count >= max_images:
            break

        image_id = example["image_id"]
        print(f"📸 Downloading Rangeland tile {count}: {image_id}")

        img_path = save_dir / f"tcd_tile_{count}.tif"
        ann_path = save_dir / f"tcd_tile_{count}_crowns.gpkg"

        # ------------------
        # Save image (.tif)
        # ------------------
        img = np.array(example["image"])
        h, w = img.shape[:2]
        crs = example["crs"]
        bounds = example["bounds"]

        transform = from_bounds(*bounds, width=w, height=h)

        with rasterio.open(
            img_path, "w", driver="GTiff",
            height=h, width=w, count=3,
            dtype=img.dtype, crs=crs, transform=transform
        ) as dst:
            for b in range(3):
                dst.write(img[:, :, b], b + 1)

        # ------------------
        # Download annotation
        # ------------------
        crown_url = example.get("annotation") or example.get("crowns_polygon")

        # Case 1: Annotation is a URL
        if isinstance(crown_url, str) and crown_url.startswith("http"):
            with requests.get(crown_url, stream=True) as r:
                r.raise_for_status()
                with open(ann_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)

        # Case 2: Annotation is an image (PIL)
        elif hasattr(crown_url, "save"):
            png_path = ann_path.with_suffix(".png")
            crown_url.save(png_path)
            print(f"🟦 Saved PNG mask annotation → {png_path}")

        # Case 3: Something else — skip politely
        else:
            print(f"⚠️ Skipping annotation for tile {image_id}: unsupported type {type(crown_url)}")

        results.append((img_path, ann_path, example, image_id))
        print(f"✅ Saved Australian tile → {img_path}")

        count += 1

    return results


def filter_raw_predictions(pred_dir: Path, score_thresh: float = 0.8, overwrite=True) -> None:
    """
    Filter raw Detectron2 prediction JSONs by confidence score.

    If overwrite=True, original JSONs are replaced in place so
    Detectree2.project_to_geojson() automatically picks them up.
    """

    pred_dir = Path(pred_dir)
    json_files = sorted(pred_dir.glob("Prediction_*.json"))
    if not json_files:
        raise FileNotFoundError(f"❌ No Detectron2 predictions found in {pred_dir}")

    for fpath in json_files:
        with open(fpath, "r") as f:
            preds = json.load(f)

        before = len(preds)
        preds = [p for p in preds if p.get("score", 0) >= score_thresh]
        after = len(preds)

        if overwrite:
            out_path = fpath
        else:
            out_path = fpath.with_name(f"{fpath.stem}_filtered_{score_thresh}.json")

        with open(out_path, "w") as f:
            json.dump(preds, f)

        print(f"📊 {fpath.name}: kept {after}/{before} predictions (≥ {score_thresh})")

    print(f"✅ Filtering complete — overwrote {len(json_files)} files at ≥ {score_thresh}.")


def visualize_saved_prediction_with_masks(img_path, pred_json_path, out_dir, image_id=None):
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
    vis_out = vis.overlay_instances(
        masks=instances.pred_masks,
        labels=labels
    )

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


def has_geodata(tif_path: str | Path) -> bool:
    """Return True if GeoTIFF has valid CRS and affine transform."""
    with rasterio.open(tif_path) as ds:
        # CRS and transform must both be defined and not identity
        has_crs = ds.crs is not None
        has_transform = ds.transform != rasterio.Affine.identity()
        return has_crs and has_transform



def validate_predictions_vs_tcd_segments(pred_geojson_path, tcd_example, iou_thresh_tree=0.5, iop_thresh_canopy=0.7):
    """Validate Detectree2 predictions against TCD 'segments' (bbox/segmentation polygons)."""
    print("📂 Loading Detectree2 predictions ...")
    pred = gpd.read_file(pred_geojson_path)
    print(f"  → {len(pred)} predicted polygons")

    # # Filter low-confidence predictions
    # if "score" in pred.columns:
    #     before = len(pred)
    #     pred = pred[pred["score"] >= score_thresh].copy()
    #     print(f"  → Filtered {before - len(pred)} low-confidence predictions (score < {score_thresh})")
        
    #     # ✅ Overwrite the GeoJSON file with only filtered predictions
    #     pred.to_file(pred_geojson_path, driver="GeoJSON")
    #     print(f"  ✅ Saved filtered predictions ({len(pred)}) back to {pred_geojson_path}")

    # --- Convert COCO-style pixel polygons to world CRS ---
    coco_annotations = tcd_example.get("coco_annotations", [])

    if isinstance(coco_annotations, str):
        try:
            coco_annotations = json.loads(coco_annotations)
        except json.JSONDecodeError:
            raise ValueError("❌ 'coco_annotations' field is not valid JSON.")
    if not isinstance(coco_annotations, list) or not coco_annotations:
        raise ValueError("❌ No valid 'coco_annotations' found in TCD example — cannot derive crowns.")

    gt_polys = []
    gt_cats = []
    for ann in coco_annotations:
        segs = ann.get("segmentation", [])
        # Handle both polygon and RLE segmentations
        if not segs:
            continue
        if isinstance(segs, dict):
            # RLE mask — skip, as we only support polygon-style annotations here
            continue
        if not isinstance(segs, list) or not isinstance(segs[0], list):
            continue

        coords = np.array(segs[0], dtype=float).reshape(-1, 2)
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

    # This print is "GT Cats [] - Canopy: 3, Trees: 48", but coco_annotations has 52 entries - Help!
    print(f" GT Cats [] - Canopy: {gt_cats.count(1)}, Trees: {gt_cats.count(2)}")

    # Use image georeferencing transform to map pixel → world
    width, height = tcd_example["width"], tcd_example["height"]
    bounds = tcd_example["bounds"]
    transform = from_bounds(*bounds, width=width, height=height)

    gt_world = [pixel_to_world(p, transform) for p in gt_polys if p.is_valid]
    gt = gpd.GeoDataFrame({"geometry": gt_world, "category": gt_cats}, crs=tcd_example["crs"])

    print(f"  → {len(gt)} ground-truth polygons (from coco_annotations)")

    if pred.crs != gt.crs:
        print(f"Aligning CRS: {pred.crs} → {gt.crs}")
        pred = pred.to_crs(gt.crs)

    # IoU computation
    def iou(a, b):
        inter, union = a.intersection(b).area, a.union(b).area
        return inter / union if union > 0 else 0.0

    # Intersection over polygon
    def iop(a, b):
        inter = a.intersection(b).area
        denom = a.area
        return inter / denom if denom > 0 else 0.0

    # --- Category-aware per-prediction scores (aligned with pred.index) ---
    from shapely.strtree import STRtree

    # Build spatial index on GT and keep parallel arrays for lookup
    gt_geoms = list(gt.geometry)
    gt_cats_arr = np.asarray(gt["category"], dtype=int)
    gt_tree = STRtree(gt_geoms)

    # Pre-size per-pred arrays so indices align 1:1 with pred.geometry
    n_pred = len(pred)
    scores_trees = np.zeros(n_pred, dtype=float)   # best IoU vs any tree GT (cat==2)
    scores_canopy = np.zeros(n_pred, dtype=float)  # best IoP vs any canopy GT (cat==1)

    # Helper to plot-safe iterate prediction geometry parts
    def polygon_parts(g):
        from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
        if g.is_empty:
            return []
        if isinstance(g, Polygon):
            return [g]
        if isinstance(g, MultiPolygon):
            return list(g.geoms)
        if isinstance(g, GeometryCollection):
            return [h for h in g.geoms if h.geom_type in ("Polygon", "MultiPolygon")]
        return []

    # Compute best per-pred scores using only intersecting GT candidates
    for i, p in enumerate(pred.geometry):
        if p is None or p.is_empty or not p.is_valid:
            continue

        # Query intersecting GT indices (fast prune)
        # NOTE: STRtree.query with shapely>=2 returns numpy indices with predicate
        try:
            cand_idx = gt_tree.query(p, predicate="intersects")
        except TypeError:
            # Fallback if predicate= isn’t available (older shapely): filter manually
            cand_idx = [j for j, ggt in enumerate(gt_geoms) if p.intersects(ggt)]

        if len(cand_idx) == 0:
            continue

        best_tree = 0.0
        best_canopy = 0.0

        for j in cand_idx:
            ggt = gt_geoms[j]
            cat = gt_cats_arr[j]

            # Robust intersection to avoid topology errors
            if not p.intersects(ggt):
                continue

            inter_area = p.intersection(ggt).area
            if inter_area <= 0.0:
                continue

            if cat == 2:
                # Tree: IoU (symmetric)
                union_area = p.union(ggt).area
                if union_area > 0:
                    best_tree = max(best_tree, inter_area / union_area)
            else:
                # Canopy (cat==1): IoP = overlap proportion of PREDICTION inside GT
                denom = p.area
                if denom > 0:
                    best_canopy = max(best_canopy, inter_area / denom)

        scores_trees[i] = best_tree
        scores_canopy[i] = best_canopy

    # --- Compute category-wise metrics ---
    def compute_metrics(scores, thresh, n_pred, n_gt):
        scores = np.asarray(scores)
        n_tp = np.sum(scores >= thresh)
        mean_score = np.mean(scores) if scores.size > 0 else 0.0
        return {
            "precision": n_tp / n_pred if n_pred else 0,
            "recall": n_tp / n_gt if n_gt else 0,
            "mean_overlap": mean_score,
            "n_tp": int(n_tp),
        }

    n_pred, n_gt = len(pred), len(gt)
    metrics_trees = compute_metrics(scores_trees, iou_thresh_tree, n_pred, gt["category"].eq(2).sum())
    metrics_canopy = compute_metrics(scores_canopy, iop_thresh_canopy, n_pred, gt["category"].eq(1).sum())

    # --- Combine overall summary ---
    metrics_all = {
        "trees": metrics_trees,
        "canopy": metrics_canopy,
        "n_pred": n_pred,
        "n_gt_total": n_gt,
    }

    print("\n📊 Validation Results:")
    print("  🌳 Trees (IoU):")
    for k, v in metrics_trees.items():
        print(f"    {k:12s}: {v:.3f}" if isinstance(v, float) else f"    {k:12s}: {v}")

    print("  🌿 Canopy (IoP):")
    for k, v in metrics_canopy.items():
        print(f"    {k:12s}: {v:.3f}" if isinstance(v, float) else f"    {k:12s}: {v}")

    print(f"\n  Total predictions: {n_pred}")
    print(f"  Total GT polygons: {n_gt} (Trees: {gt['category'].eq(2).sum()}, Canopy: {gt['category'].eq(1).sum()})")

    return metrics_all, pred, gt, (scores_trees, scores_canopy), coco_annotations

def clean_validate_predictions_vs_tcd_segments(
    pred_geojson_path,
    tcd_example,
    iou_thresh_tree=0.5,
    iop_thresh_canopy=0.7,
):
    """
    Validate Detectree2 predictions against TCD 'segments' polygons.
    Robust to all COCO segmentation formats (Polygon, RLE string/list),
    and complex geometry types (MultiPolygon, GeometryCollection).
    Returns: (metrics_all, pred_gdf, gt_gdf, (scores_trees, scores_canopy), coco_annotations)
    """
    import json
    import numpy as np
    import geopandas as gpd
    import rasterio
    import rasterio.features
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, shape
    from shapely.validation import make_valid
    from shapely.affinity import affine_transform
    from shapely.strtree import STRtree
    from pycocotools import mask as mask_utils
    from rasterio.transform import from_bounds

    # ---------------------------
    # Helper: flatten geometry parts
    # ---------------------------
    def polygon_parts(geom):
        """Return list of Polygon parts from any shapely geometry."""
        if geom is None or geom.is_empty:
            return []
        if isinstance(geom, Polygon):
            return [geom]
        if isinstance(geom, MultiPolygon):
            return list(geom.geoms)
        if isinstance(geom, GeometryCollection):
            out = []
            for g in geom.geoms:
                out.extend(polygon_parts(g))
            return out
        return []

    # ---------------------------
    # Helper: pixel → world transform
    # ---------------------------
    def pixel_to_world_geom(geom, transform):
        """Apply rasterio Affine transform to shapely geometry."""
        coeffs = [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]
        return affine_transform(geom, coeffs)

    # ---------------------------
    # Metric helpers
    # ---------------------------
    def iou(a, b):
        inter = a.intersection(b).area
        if inter <= 0.0:
            return 0.0
        union = a.union(b).area
        return inter / union if union > 0 else 0.0

    def iop(a, b):
        inter = a.intersection(b).area
        return inter / a.area if a.area > 0 else 0.0

    def compute_metrics(scores, thresh, n_pred, n_gt):
        scores = np.asarray(scores, dtype=float)
        n_tp = int(np.sum(scores >= thresh))
        precision = n_tp / n_pred if n_pred else 0.0
        recall = n_tp / n_gt if n_gt else 0.0
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        mean_score = float(np.mean(scores)) if scores.size > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mean_overlap": mean_score,
            "n_tp": n_tp,
        }

    # ---------------------------
    # 1) Load predictions
    # ---------------------------
    print("📂 Loading Detectree2 predictions ...")
    pred = gpd.read_file(pred_geojson_path)
    print(f"  → {len(pred)} predicted polygons")

    # ---------------------------
    # 2) Parse COCO annotations
    # ---------------------------
    coco_annotations = tcd_example.get("coco_annotations", [])
    if isinstance(coco_annotations, str):
        coco_annotations = json.loads(coco_annotations)
    if not isinstance(coco_annotations, list) or not coco_annotations:
        raise ValueError("❌ No valid 'coco_annotations' found in TCD example.")

    gt_polys_px = []
    gt_cats = []

    for ann in coco_annotations:
        segs = ann.get("segmentation", None)
        cat = int(ann.get("category_id", 1))  # 1=canopy, 2=tree

        if not segs:
            continue

        polys = []

        # Case A: Polygon list (e.g. [[x1,y1,...]])
        if isinstance(segs, list) and isinstance(segs[0], list):
            try:
                coords = np.array(segs[0], dtype=float).reshape(-1, 2)
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = make_valid(poly)
                polys = polygon_parts(poly)
            except Exception as e:
                print(f"⚠️ Skipping invalid polygon segmentation: {e}")
                polys = []

        # Case B: RLE dict (either string or list counts)
        elif isinstance(segs, dict) and "counts" in segs and "size" in segs:
            try:
                mask = mask_utils.decode(segs)  # works for both string and list counts
                shapes = rasterio.features.shapes(mask.astype(np.uint8), mask > 0)
                polys = [shape(geom) for geom, val in shapes if val == 1]
            except Exception as e:
                print(f"⚠️ Failed to decode RLE segmentation: {e}")
                polys = []

        else:
            print(f"⚠️ Unknown segmentation format in annotation id={ann.get('id')} — skipping.")
            polys = []

        for p in polys:
            if p.is_valid and p.area > 0:
                gt_polys_px.append(p)
                gt_cats.append(cat)

    print(f" GT Categories — Canopy: {gt_cats.count(1)}, Trees: {gt_cats.count(2)}")

    # ---------------------------
    # 3) Pixel → world transform
    # ---------------------------
    width, height = tcd_example["width"], tcd_example["height"]
    bounds = tcd_example["bounds"]
    transform = from_bounds(*bounds, width=width, height=height)

    gt_world_parts = [pixel_to_world_geom(p, transform) for p in gt_polys_px]
    gt = gpd.GeoDataFrame({"geometry": gt_world_parts, "category": gt_cats}, crs=tcd_example["crs"])
    print(f"  → {len(gt)} ground-truth polygons (after normalization)")

    # ---------------------------
    # 4) CRS alignment
    # ---------------------------
    if pred.crs != gt.crs:
        print(f"Aligning CRS: {pred.crs} → {gt.crs}")
        pred = pred.to_crs(gt.crs)

    # ---------------------------
    # 5) Overlap evaluation via STRtree
    # ---------------------------
    gt_geoms = list(gt.geometry)
    gt_cats_arr = np.asarray(gt["category"], dtype=int)
    gt_tree = STRtree(gt_geoms)

    n_pred = len(pred)
    scores_trees = np.zeros(n_pred, dtype=float)
    scores_canopy = np.zeros(n_pred, dtype=float)

    for i, p in enumerate(pred.geometry):
        if p is None or p.is_empty or not p.is_valid:
            continue
        try:
            cand_idx = gt_tree.query(p, predicate="intersects")
        except TypeError:
            cand_idx = [j for j, ggt in enumerate(gt_geoms) if p.intersects(ggt)]
        if len(cand_idx) == 0:
            continue

        best_tree, best_canopy = 0.0, 0.0
        for j in cand_idx:
            ggt = gt_geoms[j]
            if not p.intersects(ggt):
                continue
            if gt_cats_arr[j] == 2:
                best_tree = max(best_tree, iou(p, ggt))
            else:
                best_canopy = max(best_canopy, iop(p, ggt))
        scores_trees[i] = best_tree
        scores_canopy[i] = best_canopy

    # ---------------------------
    # 6) Metrics (+F1)
    # ---------------------------
    n_gt = len(gt)
    n_gt_trees = int(np.sum(gt_cats_arr == 2))
    n_gt_canopy = int(np.sum(gt_cats_arr == 1))

    metrics_trees = compute_metrics(scores_trees, iou_thresh_tree, n_pred, n_gt_trees)
    metrics_canopy = compute_metrics(scores_canopy, iop_thresh_canopy, n_pred, n_gt_canopy)

    metrics_all = {
        "trees": metrics_trees,
        "canopy": metrics_canopy,
        "n_pred": n_pred,
        "n_gt_total": n_gt,
        "n_gt_trees": n_gt_trees,
        "n_gt_canopy": n_gt_canopy,
    }

    # ---------------------------
    # 7) Summary
    # ---------------------------
    print("\n📊 Validation Results:")
    print("  🌳 Trees (IoU):")
    for k, v in metrics_trees.items():
        print(f"    {k:12s}: {v:.3f}" if isinstance(v, float) else f"    {k:12s}: {v}")
    print("  🌿 Canopy (IoP):")
    for k, v in metrics_canopy.items():
        print(f"    {k:12s}: {v:.3f}" if isinstance(v, float) else f"    {k:12s}: {v}")
    print(f"\n  Total predictions: {n_pred}")
    print(f"  Total GT polygons: {n_gt} (Trees: {n_gt_trees}, Canopy: {n_gt_canopy})")

    return metrics_all, pred, gt, (scores_trees, scores_canopy), coco_annotations

def visualize_validation_results(pred, gt, ious, coco_anns=None, 
                                 iou_thresh_tree=0.5, iop_thresh_canopy=0.7,
                                 site_path=None, rgb_path=None,
                                 tile_name=None, image_id=None):
    """
    Visualize Detectree2 vs TCD polygons over RGB base image.
    """

    # === 1. Output path ===
    out_dir = Path(site_path) / "overlays_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path_constructor = ["validate"]
    if tile_name:
        path_constructor.append(tile_name)
    if image_id:
        path_constructor.append(f"tcd{image_id}")

    out_path = out_dir / f"{'_'.join(path_constructor)}.png"

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
    ax.set_title(f"Detectree2 vs TCD — IoU ≥ {iou_thresh_tree}, IoP ≥ {iop_thresh_canopy}")
    ax.set_aspect("equal")

    if img is not None:
        ax.imshow(img, extent=extent, origin="upper")

    # === 4. Draw predictions ===

    def draw_pred_outline(ax, geom, color):
        from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
        if geom.is_empty: 
            return
        if isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, color=color, linewidth=1.2, alpha=0.9)
        elif isinstance(geom, MultiPolygon):
            for sub in geom.geoms:
                x, y = sub.exterior.xy
                ax.plot(x, y, color=color, linewidth=1.2, alpha=0.9)
        elif isinstance(geom, GeometryCollection):
            for sub in geom.geoms:
                draw_pred_outline(ax, sub, color)


    for i, p in enumerate(pred.geometry):
        if not p.is_valid or p.is_empty:
            continue

        # --- Extract per-prediction score ---
        score_tree = score_canopy = 0.0
        if isinstance(ious, tuple) and len(ious) == 2:
            scores_trees, scores_canopy = ious
            if i < len(scores_trees):
                score_tree = scores_trees[i]
            if i < len(scores_canopy):
                score_canopy = scores_canopy[i]
        elif isinstance(ious, list):
            if i < len(ious):
                score_tree = ious[i]  # fallback

        # --- Pick color based on which test passes ---
        if score_tree >= iou_thresh_tree:
            color = "#00F0FF"      # bright teal  → good tree (IoU)
        elif score_canopy >= iop_thresh_canopy:
            color = "#00FF9D"      # neon green  → good canopy (IoP)
        else:
            color = "#FF8800"      # orange      → low-overlap / FP

        draw_pred_outline(ax, p, color)

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
        mpatches.Patch(color="#00F0FF", label="Tree TP (IoU ≥ 0.5)"),
        mpatches.Patch(color="#00FF9D", label="Canopy TP (IoP ≥ 0.7)"),
        mpatches.Patch(color="#FF8800", label="False Positive / Low Overlap"),
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

    # fig_gt, ax_gt = plt.subplots(figsize=(8, 8))
    # ax_gt.set_title(f"TCD Ground Truth (image_id={image_id})")
    # ax_gt.set_aspect("equal")

    # if img is not None:
    #     ax_gt.imshow(img, extent=extent, origin="upper")

    # for idx, g in enumerate(gt.geometry):
    #     if g.is_empty:
    #         continue

    #     # Category colouring
    #     cat = None
    #     if coco_anns and idx < len(coco_anns):
    #         cat = coco_anns[idx].get("category_id", 1)
    #     color = "#C266FF" if cat == 1 else "#0a20ad"  # purple=canopy, blue=tree

    #     # Handle all geometry types safely
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
    #             x, y = geom.exterior.xy
    #             ax_gt.fill(x, y, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.25)
    #             ax_gt.plot(x, y, color=color, linewidth=0.8, alpha=0.9)
    #         elif isinstance(geom, MultiPolygon):
    #             for sub in geom.geoms:
    #                 x, y = sub.exterior.xy
    #                 ax_gt.fill(x, y, facecolor=color, edgecolor=color, linewidth=0.8, alpha=0.25)
    #                 ax_gt.plot(x, y, color=color, linewidth=0.8, alpha=0.9)

    # ax_gt.set_xlabel("Easting")
    # ax_gt.set_ylabel("Northing")

    # out_gt_path = out_path.with_name(out_path.stem + "_groundtruth.png")
    # plt.tight_layout()
    # plt.savefig(out_gt_path, dpi=250)
    # plt.close(fig_gt)

    # print(f"🗺️  Saved pure ground-truth overlay → {out_gt_path}")



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