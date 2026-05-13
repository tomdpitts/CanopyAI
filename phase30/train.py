#!/usr/bin/env python3
"""
Phase 30 training launcher.

Usage:
    python run_phase30.py \
        --train-csv /data/phase30_tcd_train.csv \
        --val-csv   /data/phase30_tcd_val.csv \
        --checkpoint /checkpoints/phase22_B_L4.pth

Add --fast-dev-run for a quick 1-batch sanity check before submitting to the cluster.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

from train_deepforest import train_deepforest

TCD_AUGMENTATIONS_COLOUR = [
    {"GaussianBlur":             {"blur_limit": [3, 7], "p": 0.3}},
    {"RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5}},
    {"HueSaturationValue":       {"hue_shift_limit": 10, "sat_shift_limit": 20,
                                  "val_shift_limit": 20, "p": 0.5}},
]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 30 TCD tree crown training")
    p.add_argument("--train-csv",    required=True,  help="Path to training CSV")
    p.add_argument("--val-csv",      required=True,  help="Path to validation CSV")
    p.add_argument("--checkpoint",   required=True,  help="Starting checkpoint (.pth or .ckpt)")
    p.add_argument("--output-dir",   default="checkpoints", help="Where to save checkpoints")
    p.add_argument("--run-name",     default="phase30_tcd_L2", help="Output subfolder name")
    p.add_argument("--batch-size",   type=int,   default=32,     help="Batch size (32 for A100-40GB)")
    p.add_argument("--lr",           type=float, default=0.0001, help="Learning rate")
    p.add_argument("--epochs",       type=int,   default=50,     help="Max epochs")
    p.add_argument("--patience",     type=int,   default=10,     help="Early-stop patience")
    p.add_argument("--fast-dev-run", action="store_true",        help="1 train + 1 val batch then exit")
    # Canopy positive policy (binary: presence of --canopy-polygons turns it on)
    p.add_argument("--canopy-polygons", default=None,
                   help="Path to phase30_tcd_canopy_polygons.json. When set, "
                        "anchors with IoP≥0.4 against a canopy polygon are treated "
                        "as positives (cls target=1, regression suppressed).")
    p.add_argument("--canopy-loss-scale", type=float, default=1.0,
                   help="Dampener for summed canopy cls contribution. "
                        "1.0 = full positive, 0.0 = iscrowd-like ignore.")
    args = p.parse_args()

    train_deepforest(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        shadow_loss_reweight=True,
        shadow_loss_weight=2.0,
        canopy_polygons_path=args.canopy_polygons,
        canopy_loss_scale=args.canopy_loss_scale,
        augmentations=TCD_AUGMENTATIONS_COLOUR,
        accelerator="gpu",
        fast_dev_run=args.fast_dev_run,
    )
