#!/usr/bin/env python3
"""
train.py — Fine-tune Detectree2’s official ResNet50-FPN model on local TCD tiles.

Data layout (expected):

data/tcd/raw/
    tcd_tile_0.tif
    tcd_tile_0_meta.json
    ...

data/tcd/tiles_pred/
    tcd_tile_0_chips/
        ... tiles produced by tile_data

Usage:

  python train.py
  python train.py --already_downloaded
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, shape
from shapely.affinity import affine_transform
from shapely.validation import make_valid
from pycocotools import mask as mask_utils

from detectree2.preprocessing.tiling import tile_data
from detectree2.models.train import register_train_data, MyTrainer, setup_cfg

import torch

from utils import download_tcd_tiles_streaming
from utils import coco_meta_to_geodf
from utils import tile_all_tcd_tiles

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
torch.set_num_threads(1)

# ============================================================
#   TRAIN DETECTREE2
# ============================================================

def train_detectree2(tiles_root: Path, output_dir: Path, preset="tiny"):
    """
    Fine-tune Detectree2 official ResNet50-FPN model.
    """

    print("📚 Registering training dataset…")

    site_name = "TCD"
    val_fold = 1  # default: second folder is validation

    register_train_data(str(tiles_root), site_name, val_fold=val_fold)

    train_name = f"{site_name}_train"
    val_name = f"{site_name}_val"

    # --- Load correct architecture & weights ---
    cfg = setup_cfg(
        base_model="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        trains=(train_name,),
        tests=(val_name,),
        workers=4,
        eval_period=100,
        max_iter=3000,
        out_dir=str(output_dir),
    )

    cfg.MODEL.WEIGHTS = "230103_randresize_full.pth"

    cfg.MODEL.DEVICE = "cpu"
    print("🧠 Using CPU backend")
    cfg.DATALOADER.NUM_WORKERS = 0

    # --- Apply preset ---
    if preset == "tiny":
        cfg = apply_tiny_config(cfg, train_name, val_name)
    else:
        cfg = apply_fast_config(cfg, train_name, val_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Training starting…")
    trainer = MyTrainer(cfg, patience=5)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print("🎉 Training complete — model_final.pth ready")


# ============================================================
#   CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--already_downloaded", default=False, action="store_true")
    p.add_argument("--preset", choices=["tiny", "fast"], default="tiny")
    return p.parse_args()


def main():
    args = parse_args()

    data_root = Path("data/tcd")
    raw_dir = data_root / "raw"
    tiles_root = data_root / "tiles_pred"
    output_dir = data_root / "train_outputs"

    raw_dir.mkdir(parents=True, exist_ok=True)
    tiles_root.mkdir(parents=True, exist_ok=True)

    # # Step 1: download or reuse tiles
    # if not args.already_downloaded:
    #     print("🌐 Downloading via HF...")
    #     download_tcd_tiles_streaming(raw_dir, max_images=3)
    # else:
    #     print("⏭️ Using existing tiles in raw/")

    # Step 2: tile everything
    # tile_all_tcd_tiles(raw_dir, tiles_root)

    # Step 3: train
    train_detectree2(tiles_root, output_dir, preset=args.preset)

# ============================================================
#   TRAINING CONFIG PRESETS (ResNet50-FPN)
# ============================================================

def apply_tiny_config(cfg, train_name, val_name, num_classes=1):
    """
    Tiny preset — FAST — but preserves the Detectree2 architecture.
    Does NOT overwrite backbone.
    """
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)

    # Very small images → very fast CPU training
    cfg.INPUT.MIN_SIZE_TRAIN = (192,)
    cfg.INPUT.MIN_SIZE_TEST = 192
    cfg.INPUT.MAX_SIZE_TRAIN = 256
    cfg.INPUT.MAX_SIZE_TEST = 256
    cfg.INPUT.RANDOM_FLIP = "horizontal"

    # Small iterator count
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 3e-4
    cfg.SOLVER.MAX_ITER = 120
    cfg.SOLVER.WARMUP_ITERS = 20
    cfg.TEST.EVAL_PERIOD = 60
    cfg.SOLVER.CHECKPOINT_PERIOD = 120

    return cfg


def apply_fast_config(cfg, train_name, val_name, num_classes=1):
    """
    Larger than tiny preset but still optimised for CPU speed.
    """
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)

    cfg.INPUT.MIN_SIZE_TRAIN = (256,)
    cfg.INPUT.MIN_SIZE_TEST = 256
    cfg.INPUT.MAX_SIZE_TRAIN = 512
    cfg.INPUT.MAX_SIZE_TEST = 512

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.MAX_ITER = 800
    cfg.SOLVER.WARMUP_ITERS = 50
    cfg.TEST.EVAL_PERIOD = 200

    return cfg


if __name__ == "__main__":
    main()