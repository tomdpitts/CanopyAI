#!/usr/bin/env python3
"""
train.py — Fine-tune Detectree2 using tiny/fast/full YAML presets.

Usage:
    python train.py --preset tiny   --weights baseline
    python train.py --preset fast   --weights baseline
    python train.py --preset full   --weights baseline
    python train.py --weights finetuned
"""

import argparse
import os
from pathlib import Path

import torch
from detectron2.engine import launch
from detectron2.config import get_cfg
from detectree2.models.train import register_train_data, MyTrainer
from utils import download_tcd_tiles_streaming, tile_all_tcd_tiles


# ============================================================
#   Load preset YAML + override weights + device
# ============================================================

def load_preset_cfg(preset: str, weights: str, output_dir: Path):
    cfg = get_cfg()

    # Preset selection
    preset_map = {
        "tiny": "config/tiny_train.yaml",
        "fast": "config/fast_train.yaml",
        "full": "config/full_train.yaml",
    }
    cfg.merge_from_file(preset_map[preset])

    # ---- Set model weights ----
    if weights == "baseline":
        cfg.MODEL.WEIGHTS = "230103_randresize_full.pth"
    elif weights == "finetuned":
        cfg.MODEL.WEIGHTS = str(output_dir / "model_final.pth")
    else:
        cfg.MODEL.WEIGHTS = weights  # custom path

    # ---- Device selection ----
    if torch.backends.mps.is_available():
        cfg.MODEL.DEVICE = "mps"
        print("💻 Using Apple MPS backend")
    elif torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        print("🚀 Using CUDA GPU")
    else:
        cfg.MODEL.DEVICE = "cpu"
        print("🧠 Using CPU")

    cfg.OUTPUT_DIR = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return cfg


# ============================================================
#   TRAINING
# ============================================================

def train_detectree2(tiles_root: Path, output_dir: Path, preset: str, weights: str):
    print("📚 Registering training dataset…")

    site_name = "TCD"
    val_fold = 1
    register_train_data(str(tiles_root), site_name, val_fold)

    train_name = f"{site_name}_train"
    val_name = f"{site_name}_val"

    cfg = load_preset_cfg(preset, weights, output_dir)

    print("🚀 Training starting…")
    trainer = MyTrainer(cfg, patience=5)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("🎉 Training complete → model_final.pth saved")


# ============================================================
#   Worker run by launch()
# ============================================================

def main_worker(args):
    data_root = Path("data/tcd")
    raw_dir = data_root / "raw"
    tiles_root = data_root / "tiles_pred"
    output_dir = data_root / "train_outputs"

    raw_dir.mkdir(exist_ok=True)
    tiles_root.mkdir(exist_ok=True)

    # 1. Download
    if not args.already_downloaded:
        print("🌐 Downloading HF tiles…")
        download_tcd_tiles_streaming(raw_dir, max_images=args.max_images)
    else:
        print("⏭️ Using existing raw tiles")

    # 2. Tile (only needs to happen once)
    tile_all_tcd_tiles(raw_dir, tiles_root)

    # 3. Train
    train_detectree2(tiles_root, output_dir,
                     preset=args.preset,
                     weights=args.weights)


# ============================================================
#   CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--already_downloaded", action="store_true")
    p.add_argument("--max_images", type=int, default=3)

    p.add_argument("--preset",
                   default="tiny",
                   choices=["tiny", "fast", "full"])

    p.add_argument("--weights",
                   default="baseline",
                   help="'baseline' | 'finetuned' | /path/to/model.pth")

    return p.parse_args()


# ============================================================

if __name__ == "__main__":
    args = parse_args()

    num_gpus = int(os.environ.get("NUM_GPUS", "1"))

    launch(
        main_worker,
        num_gpus_per_machine=num_gpus,
        num_machines=int(os.environ.get("NUM_MACHINES", "1")),
        machine_rank=int(os.environ.get("MACHINE_RANK", "0")),
        dist_url=os.environ.get("DIST_URL", "tcp://127.0.0.1:29500"),
        args=(args,),
    )