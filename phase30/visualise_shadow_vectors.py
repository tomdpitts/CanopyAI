#!/usr/bin/env python3
"""Render 20 random manually-reviewed shadow-vector examples from
tcd_shadow_vectors_by_id.json with an arrow indicating shadow direction.

Output: phase30/shadow_vectors_review.png

Convention check:
    JSON shadow_x, shadow_y is a unit vector in geographic coords
    (x = east component, y = north component).
    Image coords flip y, so the arrow on the image points in
    (shadow_x, -shadow_y).  shadow_angle_deg in the JSON is atan2(x, y) —
    degrees CW from north (compass bearing).
"""
import json
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from PIL import Image

ROOT        = Path(__file__).resolve().parent.parent
TILES_DIR   = ROOT / "data/tcd/images/data/tcd/raw"
SHADOW_JSON = ROOT / "data/tcd/tcd_shadow_vectors_by_id.json"
OUT_PNG     = ROOT / "phase30/shadow_vectors_review.png"

SEED        = 42
N_SAMPLES   = 20
ARROW_LEN   = 700          # pixels — arrow length on a 2048×2048 tile

C_BG    = "#0f172a"
C_PANEL = "#1e293b"
C_TEXT  = "#e2e8f0"
C_ARROW = "#fde047"        # bright yellow


def build_id_to_tile_map(tiles_dir: Path) -> dict:
    """Map COCO image_id → tcd_tile_{i}.tif path by scanning meta.json files."""
    mapping = {}
    for meta_file in sorted(tiles_dir.glob("tcd_tile_*_meta.json")):
        try:
            meta = json.loads(meta_file.read_text())
            image_id = meta.get("image_id")
            if image_id is None:
                continue
            tile_stem = meta_file.stem.replace("_meta", "")
            tile_path = tiles_dir / f"{tile_stem}.tif"
            if tile_path.exists() and tile_path.stat().st_size > 0:
                mapping[image_id] = tile_path
        except Exception:
            continue
    return mapping


def main():
    shadow_data = json.loads(SHADOW_JSON.read_text())

    mr_keys = [k for k, v in shadow_data.items()
               if v.get("manually_reviewed") and not v.get("excluded")]
    print(f"Manually-reviewed shadow records: {len(mr_keys)}")

    print("Scanning meta.json files for image_id → tile mapping ...")
    id_to_path = build_id_to_tile_map(TILES_DIR)
    print(f"  Found {len(id_to_path)} tile mappings on disk")

    # Filter to records whose tile is actually on disk
    available = []
    for k in mr_keys:
        try:
            image_id = int(k.replace("tcd_", ""))
        except ValueError:
            continue
        if image_id in id_to_path:
            available.append(k)
    print(f"  Manually-reviewed records with tile available: {len(available)}")

    random.seed(SEED)
    samples = random.sample(available, min(N_SAMPLES, len(available)))

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.patch.set_facecolor(C_BG)

    for ax, key in zip(axes.flat, samples):
        sv       = shadow_data[key]
        image_id = int(key.replace("tcd_", ""))
        tile_path = id_to_path[image_id]
        try:
            img = np.array(Image.open(tile_path))
        except Exception as e:
            ax.text(0.5, 0.5, f"load error\n{key}\n{e}",
                    color="white", ha="center", va="center")
            ax.set_axis_off()
            continue

        H, W = img.shape[:2]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(C_PANEL); s.set_linewidth(2)

        # Normalise (be defensive against un-normalised vectors)
        vx, vy = float(sv["shadow_x"]), float(sv["shadow_y"])
        norm = (vx * vx + vy * vy) ** 0.5
        if norm > 1e-8:
            vx, vy = vx / norm, vy / norm

        # Image coords: x right, y down → flip the y component
        cx, cy = W / 2, H / 2
        dx, dy = vx * ARROW_LEN, -vy * ARROW_LEN

        arr = ax.annotate(
            "",
            xy=(cx + dx, cy + dy), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->,head_length=0.6,head_width=0.4",
                            color=C_ARROW, lw=4, shrinkA=0, shrinkB=0),
        )
        arr.arrow_patch.set_path_effects(
            [withStroke(linewidth=7, foreground=C_BG, alpha=0.7)]
        )

        # Small dot at the arrow's origin
        ax.plot(cx, cy, "o", ms=8, mec=C_BG, mfc=C_ARROW, mew=1.5)

        angle = sv.get("shadow_angle_deg")
        title = f"{key}   bearing={angle:.0f}°" if angle is not None else key
        ax.text(20, 50, title, color=C_TEXT, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", fc=C_PANEL,
                          ec="none", alpha=0.85))

    # Hide any unused panels (in case fewer than 20 samples available)
    for ax in axes.flat[len(samples):]:
        ax.set_axis_off()

    fig.suptitle(
        f"Manually-reviewed shadow vectors (sample of {len(samples)} from "
        f"{len(available)} available)  •  arrow = direction shadows cast "
        f"(geographic ENU, image y flipped)",
        color="white", fontsize=13, fontweight="bold", y=0.995,
    )
    plt.subplots_adjust(top=0.965, bottom=0.005, left=0.01, right=0.99,
                        wspace=0.04, hspace=0.08)
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nWrote {OUT_PNG}")


if __name__ == "__main__":
    main()
