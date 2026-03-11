#!/usr/bin/env python3
"""
Generate Phase 5 Training Dataset: Combine WON + BRU with Single Random Rotation.

Strategy:
- 1 random rotation per unique image (both WON and BRU)
- 80/20 train/val split BY IMAGE COUNT (not annotation count)
- Each domain is split independently before combining to ensure both domains
  appear in both train and val sets.

Reads:
  deepforest_custom/won_train_pruned.csv
  deepforest_custom/bru_train.csv

Writes rotated tiles to:
  deepforest_custom/phase5_tiles/

Writes CSVs to:
  deepforest_custom/phase5_train_aug.csv
  deepforest_custom/phase5_val_aug.csv
"""

import os
import math
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
WON_CSV     = "deepforest_custom/won_train_pruned.csv"
BRU_CSV     = "deepforest_custom/bru_train.csv"
TRAIN_CSV   = "deepforest_custom/phase5_train_aug.csv"
VAL_CSV     = "deepforest_custom/phase5_val_aug.csv"
OUTPUT_DIR  = "deepforest_custom/phase5_tiles"

WON_BASE_ANGLE = 215.0   # Shadow azimuth for WON (0=North CW)
BRU_BASE_ANGLE = 118.0   # Shadow azimuth for BRU (0=North CW)
TRAIN_FRAC     = 0.80
RANDOM_SEED    = 42
MIN_BOX_DIM    = 10      # Minimum box width AND height in pixels after rotation
MIN_CONTENT_RATIO = 0.40 # Discard boxes where < 40% of pixels are non-black (padding)

# Bad BRU source tiles identified by manual inspection
BRU_EXCLUDE = {
    "bru_tile_1256_4400.png",
    "bru_tile_1656_4400.png",
    "bru_tile_2456_1200.png",
    "bru_tile_2856_1600.png",
    "bru_tile_2856_2400.png",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Rotation helpers ─────────────────────────────────────────────────────────
def rotate_image_and_boxes(image: np.ndarray, boxes: np.ndarray, angle_deg: float):
    """Rotate image and bounding boxes by angle_deg (CCW, cv2 convention)."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    if len(boxes) == 0:
        return rotated, boxes

    out_boxes = []
    for xmin, ymin, xmax, ymax in boxes:
        corners = np.array([[xmin, ymin], [xmax, ymin],
                            [xmax, ymax], [xmin, ymax]], dtype=np.float32)
        ones = np.ones((4, 1))
        rotated_corners = M.dot(np.hstack([corners, ones]).T).T
        nx0 = int(np.clip(rotated_corners[:, 0].min(), 0, w))
        ny0 = int(np.clip(rotated_corners[:, 1].min(), 0, h))
        nx1 = int(np.clip(rotated_corners[:, 0].max(), 0, w))
        ny1 = int(np.clip(rotated_corners[:, 1].max(), 0, h))

        # Discard boxes that are too small after clipping
        if (nx1 - nx0 < MIN_BOX_DIM) or (ny1 - ny0 < MIN_BOX_DIM):
            continue

        # Discard boxes mostly over black padding (warpAffine fill)
        crop = rotated[ny0:ny1, nx0:nx1]
        if crop.size > 0:
            non_black = np.count_nonzero(crop.sum(axis=2))
            total = crop.shape[0] * crop.shape[1]
            if non_black / total < MIN_CONTENT_RATIO:
                continue

        out_boxes.append([nx0, ny0, nx1, ny1])

    return rotated, np.array(out_boxes, dtype=np.float32)


def shadow_vector(base_angle_deg, image_rotation_ccw):
    """
    Compute shadow vector components after rotating the image CCW by image_rotation_ccw.
    Convention: shadow_angle is azimuth (0=North CW).
    cv2 positive angle = CCW rotation, so shadow rotates CW = subtract angle.
    """
    angle = (base_angle_deg - image_rotation_ccw) % 360.0
    rad = math.radians(angle)
    return math.sin(rad), math.cos(rad), angle


# ── Per-domain processing ────────────────────────────────────────────────────
def process_domain(csv_path, base_angle, domain_tag):
    """
    Returns list of row-dicts for all images in csv_path.
    Each source image → 1 random rotation → rotated file saved to OUTPUT_DIR.
    Split (train/val) is NOT applied here; caller does it by image.
    """
    df = pd.read_csv(csv_path)
    unique_images = list(df["image_path"].unique())
    print(f"  {domain_tag}: {len(unique_images)} source images")

    rows_by_image = {}   # original_path → list[row-dict]

    for img_path in tqdm(unique_images, desc=f"  {domain_tag}"):
        # Skip manually excluded source tiles
        if Path(img_path).name in BRU_EXCLUDE:
            continue

        if not os.path.exists(img_path):
            print(f"    ⚠️  Missing: {img_path}")
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"    ⚠️  Unreadable: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_rows = df[df["image_path"] == img_path]
        boxes  = img_rows[["xmin", "ymin", "xmax", "ymax"]].values
        labels = img_rows["label"].values

        # Single random rotation
        angle = random.uniform(0.0, 360.0)
        rot_img, rot_boxes = rotate_image_and_boxes(img_rgb, boxes, angle)

        stem       = Path(img_path).stem
        save_name  = f"{stem}_rot{int(angle)}.tif"
        save_path  = os.path.abspath(os.path.join(OUTPUT_DIR, save_name))
        cv2.imwrite(save_path, cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR))

        sx, sy, sh = shadow_vector(base_angle, angle)

        row_list = []
        for idx, box in enumerate(rot_boxes):
            row_list.append({
                "image_path":   save_path,
                "xmin":         int(box[0]),
                "ymin":         int(box[1]),
                "xmax":         int(box[2]),
                "ymax":         int(box[3]),
                "label":        labels[idx] if idx < len(labels) else "Tree",
                "shadow_angle": sh,
                "shadow_x":     sx,
                "shadow_y":     sy,
                "domain":       domain_tag,
            })
        rows_by_image[img_path] = row_list

    return rows_by_image  # dict: original_path → list[row-dict]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("📂 Processing WON dataset...")
    won_by_img = process_domain(WON_CSV, WON_BASE_ANGLE, "WON")

    print("\n📂 Processing BRU dataset...")
    bru_by_img = process_domain(BRU_CSV, BRU_BASE_ANGLE, "BRU")

    def split_images(img_dict, frac):
        keys = list(img_dict.keys())
        random.shuffle(keys)
        n_train = max(1, int(len(keys) * frac))
        return keys[:n_train], keys[n_train:]

    won_train_keys, won_val_keys = split_images(won_by_img, TRAIN_FRAC)
    bru_train_keys, bru_val_keys = split_images(bru_by_img, TRAIN_FRAC)

    def collect_rows(img_dict, keys):
        rows = []
        for k in keys:
            rows.extend(img_dict[k])
        return rows

    train_rows = collect_rows(won_by_img, won_train_keys) + collect_rows(bru_by_img, bru_train_keys)
    val_rows   = collect_rows(won_by_img, won_val_keys)   + collect_rows(bru_by_img, bru_val_keys)

    train_df = pd.DataFrame(train_rows)
    val_df   = pd.DataFrame(val_rows)

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV,   index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    def stats(df, label):
        if df.empty:
            print(f"\n  {label}: (empty)")
            return
        for domain in ("WON", "BRU"):
            sub = df[df["domain"] == domain]
            imgs = sub["image_path"].nunique()
            anns = len(sub)
            print(f"    {domain}: {imgs} images, {anns} annotations")

    print(f"\n{'='*55}")
    print(f"  TRAIN CSV → {TRAIN_CSV}")
    stats(train_df, "TRAIN")
    print(f"\n  VAL CSV   → {VAL_CSV}")
    stats(val_df, "VAL")
    print(f"{'='*55}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
