#!/usr/bin/env python3
"""
Prepare TCD dataset for DeepForest training.

Converts TCD COCO annotations to DeepForest CSV format:
image_path, xmin, ymin, xmax, ymax, label

Usage:
    python prepare_deepforest_data.py --data_dir data/tcd/raw --output train_annotations.csv

Note: modal_deepforest.py will update file paths in csv to support uploaded data paths.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from pycocotools import mask as mask_utils
import numpy as np


def bbox_from_mask(mask_rle, image_shape):
    """Extract bounding box from RLE mask."""
    try:
        mask = mask_utils.decode(mask_rle)
        # Find non-zero pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return None

        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        return [int(xmin), int(ymin), int(xmax), int(ymax)]
    except Exception as e:
        print(f"  Warning: Failed to decode mask: {e}")
        return None


def bbox_from_polygon(polygon_coords):
    """Extract bounding box from polygon coordinates."""
    try:
        coords = np.array(polygon_coords).reshape(-1, 2)
        xmin = int(coords[:, 0].min())
        xmax = int(coords[:, 0].max())
        ymin = int(coords[:, 1].min())
        ymax = int(coords[:, 1].max())
        return [xmin, ymin, xmax, ymax]
    except Exception as e:
        print(f"  Warning: Failed to extract bbox from polygon: {e}")
        return None


def convert_tcd_to_deepforest(data_dir, output_csv, category_filter=None, min_area=50):
    """
    Convert TCD metadata JSON files to DeepForest CSV format.

    Args:
        data_dir: Directory containing tcd_tile_*.tif and *_meta.json files
        output_csv: Output CSV path
        category_filter: List of category IDs to include (None = all)
        min_area: Minimum bounding box area in pixels
    """
    data_dir = Path(data_dir)

    # Find all metadata files (recursive search in subdirectories)
    all_meta_files = sorted(data_dir.glob("**/*_meta.json"))

    # Exclude archive folders
    meta_files = [f for f in all_meta_files if "archive" not in str(f)]

    if not meta_files:
        raise FileNotFoundError(f"No *_meta.json files found in {data_dir}")

    print(f"📂 Found {len(meta_files)} metadata files")

    all_annotations = []
    total_trees = 0
    total_canopy = 0
    skipped = 0

    for meta_path in meta_files:
        # Get corresponding image path (same directory as metadata file)
        image_path = meta_path.parent / meta_path.name.replace("_meta.json", ".tif")

        if not image_path.exists():
            print(f"  ⚠️  Image not found: {image_path}")
            continue

        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)

        image_id = meta.get("image_id", "unknown")
        coco_anns = meta.get("coco_annotations", [])

        if isinstance(coco_anns, str):
            coco_anns = json.loads(coco_anns)

        if not coco_anns:
            print(f"  ⚠️  No annotations in {meta_path.name}")
            continue

        print(
            f"📄 Processing {image_path.name} (ID: {image_id}) - {len(coco_anns)} annotations"
        )

        image_shape = [meta["height"], meta["width"]]

        for ann in coco_anns:
            category_id = ann.get("category_id")

            # Filter by category if specified
            if category_filter and category_id not in category_filter:
                continue

            # Track counts
            if category_id == 1:
                total_canopy += 1
            elif category_id == 2:
                total_trees += 1

            # Extract bounding box from segmentation
            seg = ann.get("segmentation")
            bbox = None

            # Try to get bbox from annotation first
            if "bbox" in ann and ann["bbox"]:
                x, y, w, h = ann["bbox"]
                bbox = [int(x), int(y), int(x + w), int(y + h)]

            # Otherwise extract from segmentation
            elif seg:
                # Polygon format: list of lists
                if isinstance(seg, list) and isinstance(seg[0], list):
                    bbox = bbox_from_polygon(seg[0])

                # RLE format: dict with 'counts' and 'size'
                elif isinstance(seg, dict) and "counts" in seg:
                    bbox = bbox_from_mask(seg, image_shape)

            if bbox is None:
                skipped += 1
                continue

            xmin, ymin, xmax, ymax = bbox

            # Filter by minimum area
            area = (xmax - xmin) * (ymax - ymin)
            if area < min_area:
                skipped += 1
                continue

            # Ensure bbox is within image bounds
            xmin = max(0, min(xmin, image_shape[1] - 1))
            xmax = max(0, min(xmax, image_shape[1] - 1))
            ymin = max(0, min(ymin, image_shape[0] - 1))
            ymax = max(0, min(ymax, image_shape[0] - 1))

            # Skip invalid boxes
            if xmax <= xmin or ymax <= ymin:
                skipped += 1
                continue

            # DeepForest format: image_path, xmin, ymin, xmax, ymax, label
            all_annotations.append(
                {
                    "image_path": str(image_path.resolve().absolute()),
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "label": "Tree",  # DeepForest expects string label
                }
            )

    # Create DataFrame and save
    df = pd.DataFrame(all_annotations)

    if len(df) == 0:
        raise ValueError("No valid annotations found!")

    df.to_csv(output_csv, index=False)

    print(f"\n✅ Conversion complete!")
    print(f"   Total annotations: {len(df)}")
    print(f"   Trees: {total_trees}")
    print(f"   Canopy: {total_canopy}")
    print(f"   Skipped (too small/invalid): {skipped}")
    print(f"   Output: {output_csv}")

    return df


def split_train_val(csv_path, val_fraction=0.2, random_seed=42):
    """Split annotations into train/val CSVs."""
    df = pd.read_csv(csv_path)

    # Group by image to avoid data leakage
    images = df["image_path"].unique()
    np.random.seed(random_seed)
    np.random.shuffle(images)

    split_idx = int(len(images) * (1 - val_fraction))
    train_images = set(images[:split_idx])
    val_images = set(images[split_idx:])

    train_df = df[df["image_path"].isin(train_images)]
    val_df = df[df["image_path"].isin(val_images)]

    train_path = csv_path.replace(".csv", "_train.csv")
    val_path = csv_path.replace(".csv", "_val.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\n📊 Split complete:")
    print(f"   Train: {len(train_df)} boxes from {len(train_images)} images")
    print(f"   Val:   {len(val_df)} boxes from {len(val_images)} images")
    print(f"   → {train_path}")
    print(f"   → {val_path}")

    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="Convert TCD to DeepForest format")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/tcd/raw",
        help="Directory with TCD tiles and metadata",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="deepforest/deepforest_annotations.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--category",
        type=int,
        choices=[1, 2],
        help="Filter by category (1=canopy, 2=tree, None=both)",
    )
    parser.add_argument(
        "--min_area", type=int, default=50, help="Minimum box area in pixels"
    )
    parser.add_argument(
        "--split", action="store_true", help="Also create train/val split"
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Validation fraction (default: 0.2)",
    )

    args = parser.parse_args()

    # Convert
    category_filter = [args.category] if args.category else None
    convert_tcd_to_deepforest(
        args.data_dir,
        args.output,
        category_filter=category_filter,
        min_area=args.min_area,
    )

    # Split if requested
    if args.split:
        split_train_val(args.output, args.val_fraction)


if __name__ == "__main__":
    main()
