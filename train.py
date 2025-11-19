#!/usr/bin/env python3
"""
train.py — Train a Detectree2 model on the TCD dataset.

This script:
1. Loads an orthomosaic + crown polygons (TCD format)
2. Tiles them into 40×40 (or configurable) chips
3. Splits into train/val folders
4. Registers training data with Detectron2
5. Trains Mask R-CNN with Detectree2's MyTrainer wrapper
"""

import argparse
import os
from pathlib import Path

import geopandas as gpd
import rasterio

from detectree2.preprocessing.tiling import tile_data, to_traintest_folders
from detectree2.models.train import register_train_data, MyTrainer, setup_cfg


# ---------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------

def prepare_tiles(img_path: Path,
                  crowns_path: Path,
                  out_dir: Path,
                  tile_width: int = 40,
                  tile_height: int = 40,
                  buffer: int = 30,
                  threshold: float = 0.6,
                  test_frac: float = 0.15):
    """
    Tile orthomosaic + crowns → train/val split.
    """

    print("📦 Tiling dataset...")

    # Load crowns and match CRS
    with rasterio.open(img_path) as src:
        img_crs = src.crs

    crowns = gpd.read_file(crowns_path)
    crowns = crowns.to_crs(img_crs)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Tile into chips
    tile_data(
        img_path=str(img_path),
        out_dir=str(out_dir),
        buffer=buffer,
        tile_width=tile_width,
        tile_height=tile_height,
        crowns=crowns,
        threshold=threshold,
        mode="rgb"
    )

    print("📁 Creating train/val split...")
    to_traintest_folders(str(out_dir), str(out_dir), test_frac=test_frac)

    print(f"✅ Tiling and split complete → {out_dir}")


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def run_training(tiles_dir: Path,
                 site_name: str,
                 base_model: str,
                 output_dir: Path,
                 workers: int = 4,
                 eval_period: int = 100,
                 max_iter: int = 3000,
                 patience: int = 5,
                 resume: bool = False):
    """
    Register data, set up config, and start Detectree2 training.
    """

    train_dir = tiles_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"No training directory found at {train_dir}")

    print("📚 Registering training data...")
    register_train_data(str(train_dir), site_name, val_fold=5)

    trains = (f"{site_name}_train",)
    tests = (f"{site_name}_val",)

    cfg = setup_cfg(
        base_model=base_model,
        trains=trains,
        tests=tests,
        workers=workers,
        eval_period=eval_period,
        max_iter=max_iter,
        out_dir=str(output_dir)
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting training...")
    trainer = MyTrainer(cfg, patience=patience)
    trainer.resume_or_load(resume=resume)
    trainer.train()

    print(f"🎉 Training complete. Model saved to {output_dir}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Detectree2 on TCD dataset")

    # Required inputs
    p.add_argument("--img", required=True, type=str,
                   help="Path to orthomosaic .tif")
    p.add_argument("--crowns", required=True, type=str,
                   help="Path to crowns .gpkg/.shp")
    p.add_argument("--tiles", required=True, type=str,
                   help="Output directory for tiled dataset")

    # Training
    p.add_argument("--site", default="TCDsite", type=str,
                   help="Site name for dataset registration")
    p.add_argument("--output", default="./train_outputs", type=str,
                   help="Model output directory")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from checkpoint")

    # Model / config
    p.add_argument("--base", default="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                   help="Detectron2 model zoo base model")
    p.add_argument("--max_iter", default=3000, type=int)
    p.add_argument("--eval_period", default=100, type=int)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--patience", default=5, type=int)

    # Tiling (advanced)
    p.add_argument("--tile_width", default=40, type=int)
    p.add_argument("--tile_height", default=40, type=int)
    p.add_argument("--buffer", default=30, type=int)
    p.add_argument("--threshold", default=0.6, type=float)
    p.add_argument("--test_frac", default=0.15, type=float)

    return p.parse_args()


# ---------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------

def main():
    args = parse_args()

    img_path = Path(args.img)
    crowns_path = Path(args.crowns)
    tiles_path = Path(args.tiles)
    output_dir = Path(args.output)

    # Step 1 — tile dataset
    prepare_tiles(
        img_path=img_path,
        crowns_path=crowns_path,
        out_dir=tiles_path,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        buffer=args.buffer,
        threshold=args.threshold,
        test_frac=args.test_frac
    )

    # Step 2 — train
    run_training(
        tiles_dir=tiles_path,
        site_name=args.site,
        base_model=args.base,
        output_dir=output_dir,
        workers=args.workers,
        eval_period=args.eval_period,
        max_iter=args.max_iter,
        patience=args.patience,
        resume=args.resume
    )


if __name__ == "__main__":
    main()