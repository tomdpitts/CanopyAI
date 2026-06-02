#!/usr/bin/env python3
"""Regenerate the local phase22 train/val CSVs for the zero-shot shadow ablation.

The committed phase22/phase22_{train,val}.csv reference tiles by an old absolute
path (/Users/tompitts/dphil/CanopyAI/...).  This remaps that prefix to the current
repo root and drops any tile smaller than the 400x400 training crop, writing:

    phase30/zeroshot/phase22_{train,val}_filt.csv   (gitignored — machine-specific)

Run once before training:  ./venv310/bin/python phase30/zeroshot/prepare_phase22.py
"""
import csv
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
OLD_PREFIX = "/Users/tompitts/dphil/CanopyAI/"
MIN_SIZE = 400  # native trainer random-crops to 400x400

for split in ("train", "val"):
    src = REPO / "phase22" / f"phase22_{split}.csv"
    dst = REPO / "phase30" / "zeroshot" / f"phase22_{split}_filt.csv"
    rows, kept, dropped = [], [], set()
    with open(src) as f:
        r = csv.reader(f)
        header = next(r)
        ip = header.index("image_path")
        rows = list(r)
    sizes = {}
    for row in rows:
        p = row[ip]
        if p.startswith(OLD_PREFIX):
            p = str(REPO) + "/" + p[len(OLD_PREFIX):]
        row[ip] = p
        if p not in sizes:
            try:
                with Image.open(p) as im:
                    sizes[p] = im.size
            except Exception:
                sizes[p] = (0, 0)
    with open(dst, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(header)
        for row in rows:
            w_, h_ = sizes[row[ip]]
            if w_ >= MIN_SIZE and h_ >= MIN_SIZE:
                kept.append(row); w.writerow(row)
            else:
                dropped.add(row[ip])
    imgs = len({row[ip] for row in kept})
    print(f"{split}: wrote {len(kept)} rows / {imgs} imgs to {dst.name} "
          f"({len(dropped)} imgs dropped <{MIN_SIZE}px)")
