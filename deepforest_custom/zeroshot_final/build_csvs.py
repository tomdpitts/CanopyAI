#!/usr/bin/env python3
"""
build_csvs.py — build the self-contained training manifests for the FINAL
zero-shot shadow ablation (run family `zsfinal`).

Why this exists
---------------
The canonical zero-shot models (phase21_baseline / phase22_B_L4) were trained by
`deepforest_custom/train_deepforest.py` from `phase22/phase22_{train,val}.csv`.
Those CSVs carry the legacy absolute prefix `/Users/tompitts/dphil/CanopyAI/...`
(pre-iCloud) which no longer resolves, and they live outside this folder. To make
the final comparison a clean, single-trainer, single-data experiment we regenerate
the manifests HERE, beside the driver, with:
  • paths remapped to the current repo root, and
  • sub-400px tiles dropped (the 400x400 RandomCrop in deepforest_custom has NO
    padding — albumentations raises CropSizeError on any tile whose shorter side
    is < 400, so such a tile cannot be sampled and could not have trained the
    originals either). Exactly which tiles are dropped is printed below.

Output: deepforest_custom/zeroshot_final/{train,val}.csv  (gitignored — the abs
paths are machine-specific; THIS script is the tracked, reproducible source).

Run: ./venv310/bin/python deepforest_custom/zeroshot_final/build_csvs.py
"""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OLD_PREFIX = "/Users/tompitts/dphil/CanopyAI/"
NEW_PREFIX = str(REPO) + "/"
MIN_SIZE = 400  # deepforest_custom RandomCrop(400,400), no padding

SRC = {
    "train": REPO / "phase22" / "phase22_train.csv",
    "val":   REPO / "phase22" / "phase22_val.csv",
}


def remap(p: str) -> str:
    return p.replace(OLD_PREFIX, NEW_PREFIX) if p.startswith(OLD_PREFIX) else p


def build(split: str, src: Path) -> None:
    df = pd.read_csv(src)
    df["image_path"] = df["image_path"].map(remap)

    sizes, dropped, missing = {}, set(), set()
    for p in df["image_path"].unique():
        if not Path(p).exists():
            missing.add(p)
            continue
        try:
            with Image.open(p) as im:
                sizes[p] = im.size  # (W, H)
        except Exception as e:
            print(f"   ⚠️  unreadable {p}: {e}")
            sizes[p] = (0, 0)
    if missing:
        print(f"❌ {split}: {len(missing)} image paths do NOT resolve locally, e.g.:")
        for p in list(missing)[:5]:
            print(f"     {p}")
        sys.exit(1)

    def keep(p):
        w, h = sizes[p]
        return min(w, h) >= MIN_SIZE

    for p in df["image_path"].unique():
        if not keep(p):
            dropped.add(p)
    out = df[df["image_path"].map(keep)].copy()

    dst = HERE / f"{split}.csv"
    out.to_csv(dst, index=False)

    n_img = out["image_path"].nunique()
    by_dom = out.groupby("domain")["image_path"].nunique().to_dict() if "domain" in out else {}
    print(f"✅ {split}: {n_img} images, {len(out)} boxes  domains={by_dom}")
    print(f"   → {dst.relative_to(REPO)}")
    if dropped:
        print(f"   dropped {len(dropped)} tile(s) <{MIN_SIZE}px (unusable by 400-crop):")
        for p in sorted(dropped):
            w, h = sizes[p]
            print(f"     {Path(p).name}  ({w}x{h})")


if __name__ == "__main__":
    for split, src in SRC.items():
        if not src.exists():
            print(f"❌ source CSV missing: {src}")
            sys.exit(1)
        build(split, src)
    print("\nDone. Train with deepforest_custom/zeroshot_final/train_all.sh")
