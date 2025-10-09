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

from detectree2.preprocessing.tiling import tile_data
from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns
from detectree2.models.predict import predict_on_data
from detectree2.models.train import setup_cfg
from detectron2.engine import DefaultPredictor
import wget
import cv2
import argparse
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch


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
    # === 1. Define key paths ===
    # Adjust this base path if your dataset lives elsewhere
    home = Path.home()
    site_path = home / "dphil" / "detectree2" / "data" / "BCI_50ha"
    img_path = site_path / "rgb" / "2015.06.10_07cm_ORTHO.tif"
    model_path = Path("230103_randresize_full.pth")

    # === 2. Create output/working directories ===
    tiles_path = ensure_dir(site_path / "tilespred")
    preds_path = ensure_dir(Path(tiles_path) / "predictions")
    preds_geo_path = ensure_dir(Path(tiles_path) / "predictions_geo")

    # === 3. Download pretrained model if missing ===
    if not model_path.exists():
        url = "https://zenodo.org/records/10522461/files/230103_randresize_full.pth"
        print(f"📦 Model not found locally — downloading from {url} ...")
        wget.download(url, out=str(model_path))
        print("\n✅ Model download complete.")

    # === 4. Check that orthomosaic exists ===
    
    if not img_path.exists():
        print(f"⚠️ GeoTIFF not found at {img_path}.")
        return

    # === 5. Tile orthomosaic ===
    print("\n🧩 Tiling image into smaller chips ...")
    buffer, tile_width, tile_height = 30, 40, 40
    tile_data(str(img_path), tiles_path, buffer, tile_width, tile_height, dtype_bool=True)
    print("✅ Tiling complete.")

    # === 6. Set up Detectron2 predictor ===
    print("\n⚙️  Initializing Detectron2 predictor ...")
    cfg = setup_cfg(update_model=str(model_path))
    set_device(cfg)
    predictor = DefaultPredictor(cfg)
    print("✅ Predictor ready.")

    # === 7. Run inference on all tiles ===
    print("\n🔮 Running model inference on tiles ... (this may take a while)")
    predict_on_data(tiles_path, predictor=predictor)
    print("✅ Inference complete.")

    # === 8. Convert predictions to GeoJSON ===
    print("\n🗺️  Projecting predictions to GeoJSON ...")
    project_to_geojson(tiles_path, preds_path, preds_geo_path)
    print("✅ GeoJSON projection complete.")

    # === 9. Stitch, clean, and simplify crowns ===
    print("\n🌿 Stitching and cleaning crown polygons ...")
    crowns = stitch_crowns(preds_geo_path, nproc=1)
    clean = clean_crowns(crowns, 0.6, confidence=0.5)
    clean = clean.set_geometry(clean.geometry.simplify(0.3))
    print("✅ Crowns cleaned and simplified.")

    # === 10. Write final output ===
    out_gpkg = site_path / "crowns_out.gpkg"
    clean.to_file(out_gpkg)
    print(f"\nAll done! Results saved to:\n  {out_gpkg}\n")


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