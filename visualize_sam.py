#!/usr/bin/env python3

import json
import cv2
import numpy as np
from pathlib import Path
import argparse


def visualize_sam_annotations(data_dir):
    data_dir = Path(data_dir)
    vis_dir = data_dir / "visualise"
    vis_dir.mkdir(exist_ok=True)

    # Find all meta.json files
    meta_files = sorted(list(data_dir.glob("*_meta.json")))
    print(f"Found {len(meta_files)} tiles to visualize.")

    for meta_path in meta_files:
        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Load image
        img_filename = meta["file_name"]
        img_path = data_dir / img_filename

        if not img_path.exists():
            print(f"⚠️ Image not found: {img_path}")
            continue

        # Read image with cv2 (BGR)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Failed to read image: {img_path}")
            continue

        # Create overlay
        overlay = img.copy()

        annotations = meta.get("coco_annotations", [])
        for ann in annotations:
            segs = ann.get("segmentation", [])
            for seg in segs:
                # Polygon format: [x1, y1, x2, y2, ...]
                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)

                # Draw filled polygon on overlay
                cv2.fillPoly(overlay, [poly], color=(0, 255, 0))  # Green

                # Draw contour on original image for sharpness
                cv2.polylines(
                    img, [poly], isClosed=True, color=(0, 255, 0), thickness=1
                )

        # Blend overlay
        alpha = 0
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Save
        out_path = vis_dir / f"{img_path.stem}_vis.png"
        cv2.imwrite(str(out_path), img)
        print(f"Saved {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="won003_sam",
        help="Directory containing tiles and meta.json",
    )
    args = parser.parse_args()

    visualize_sam_annotations(args.data_dir)
