#!/usr/bin/env python3
"""Zero-shot shadow ablation — train ONE shadow weight, NATIVE phase22 trainer.

Reproduces the phase22 *zero-shot* regime where shadow loss reweighting won:
train on BRU/WON/NEON (non-TCD, dense per-box shadow vectors) from the
weecology/NEON base, using deepforest_custom/train_deepforest.py (the original
phase22 trainer — canopy-free, center-crop val that handles the small tiles).
Models are evaluated zero-shot on the TCD holdout afterwards.

  shadow_loss_weight: 0.0 = ignore shadow boxes, 1.0 = neutral (no reweight),
  >1 = upweight shadow-casting boxes.

Usage:
  ./venv310/bin/python phase30/zeroshot/train_one.py --shadow-weight 2 --run-name zs_sw_2
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deepforest_custom"))   # NATIVE phase22 trainer

# macOS DataLoader fix: the native trainer uses num_workers=8, and the default
# 'file_descriptor' sharing strategy exhausts macOS's low FD limit ->
# "Bad file descriptor" / Abort trap: 6 at worker startup.  'file_system' avoids it.
import torch.multiprocessing as _mp
_mp.set_sharing_strategy("file_system")

from train_deepforest import train_deepforest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shadow-weight", type=float, required=True)
    p.add_argument("--run-name",      required=True)
    p.add_argument("--epochs",        type=int,   default=40)
    p.add_argument("--batch-size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--output-dir",    default=str(REPO / "checkpoints" / "zeroshot_shadow"))
    p.add_argument("--fast-dev-run",  action="store_true")
    a = p.parse_args()

    train_deepforest(
        train_csv=str(REPO / "phase30" / "zeroshot" / "phase22_train_filt.csv"),
        val_csv=str(REPO / "phase30" / "zeroshot" / "phase22_val_filt.csv"),
        output_dir=a.output_dir,
        run_name=a.run_name,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        patience=99,                     # fixed budget, no early stop (fair sweep)
        pretrained=True,                 # weecology/NEON base
        checkpoint=None,                 # (no custom init -> pretrained base)
        shadow_loss_reweight=True,       # enable shadow weighting; strength below
        shadow_loss_weight=a.shadow_weight,
        accelerator=None,                # None -> auto-selects MPS on this machine
        augmentations=None,              # native default (crop + flip)
        fast_dev_run=a.fast_dev_run,
    )
