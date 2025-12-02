#!/usr/bin/env python3

import argparse
import os
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
import xml.etree.ElementTree as ET
from segment_anything import SamPredictor, sam_model_registry
import cv2  # Re-keeping cv2 as it is needed for findContours.
import torch


def get_polygon_from_mask(mask):
    """Convert binary mask to list of polygons (COCO format)."""
    # mask is boolean, convert to uint8
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        # Flatten and convert to list
        poly = contour.flatten().tolist()
        if len(poly) > 4:  # at least 3 points (6 coords)
            polygons.append(poly)
    return polygons


def process_won003(root_dir, output_dir, checkpoint_path):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    img_dir = root_dir / "images"
    ann_dir = root_dir / "annotations"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize SAM
    print(f"Loading SAM model from {checkpoint_path}...")
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    predictor = SamPredictor(sam)

    xml_files = sorted(list(ann_dir.glob("*.xml")))
    print(f"Found {len(xml_files)} XML files.")

    for idx, xml_file in enumerate(xml_files):
        print(f"Processing {xml_file.name} ({idx + 1}/{len(xml_files)})...")

        # 1. Load Image
        stem = xml_file.stem
        img_files = list(img_dir.glob(f"{stem}.*"))
        if not img_files:
            print(f"⚠️ Warning: No image found for {xml_file.name}, skipping.")
            continue
        img_path = img_files[0]

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"❌ Failed to read image {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Parse XML for bboxes
        tree = ET.parse(xml_file)
        root = tree.getroot()

        boxes = []
        for member in root.findall("object"):
            if member.find("name").text != "tree":
                continue
            bndbox = member.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        if not boxes:
            print("  No tree objects found.")
            continue

        # 3. Run SAM
        predictor.set_image(image)
        input_boxes = np.array(boxes)

        # SAM can take batch of boxes
        # transform boxes to SAM format if needed, but set_image handles original size
        transformed_boxes = predictor.transform.apply_boxes_torch(
            torch.as_tensor(input_boxes, device=predictor.device), image.shape[:2]
        )

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # masks is (B, 1, H, W)
        masks = masks.cpu().numpy().squeeze(1)

        # 4. Prepare COCO annotations
        coco_anns = []
        for i, mask in enumerate(masks):
            polys = get_polygon_from_mask(mask)
            if not polys:
                continue

            # Calculate area and bbox from mask for correctness
            # But we can reuse original bbox or recalculate
            # Let's use the polygon to be consistent

            ann = {
                "id": i,
                "image_id": idx,
                "category_id": 0,  # 0=tree
                "segmentation": polys,
                "bbox": boxes[
                    i
                ],  # [x,y,x,y] -> need [x,y,w,h] for COCO. Detectron2 accepts XYXY_ABS if specified, but standard COCO is XYWH.
                "bbox_mode": 1,  # BoxMode.XYWH_ABS = 1
                "iscrowd": 0,
            }
            # Convert XYXY to XYWH
            x1, y1, x2, y2 = boxes[i]
            ann["bbox"] = [x1, y1, x2 - x1, y2 - y1]

            coco_anns.append(ann)

        # 5. Save as GeoTIFF (with dummy georef)
        h, w = image.shape[:2]
        # Dummy transform: 0,0 top left, 1 pixel = 1 meter (or degree)
        transform = from_origin(0, h, 1, 1)
        # Actually from_origin(west, north, xsize, ysize)
        # origin at (0,0) implies top-left is 0,0?
        # rasterio uses: (west, south, east, north) bounds
        # simple identity-like transform
        # transform = rasterio.transform.from_bounds(0, 0, w, h, w, h) # 0,0 to w,h

        out_tif_path = output_dir / f"won_tile_{idx}.tif"
        out_meta_path = output_dir / f"won_tile_{idx}_meta.json"

        with rasterio.open(
            out_tif_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=3,
            dtype=image.dtype,
            crs="EPSG:4326",  # Dummy CRS
            transform=transform,
        ) as dst:
            # Write RGB
            dst.write(image[:, :, 0], 1)
            dst.write(image[:, :, 1], 2)
            dst.write(image[:, :, 2], 3)

        # 6. Save Metadata
        meta = {
            "image_id": idx,
            "file_name": out_tif_path.name,
            "width": w,
            "height": h,
            "coco_annotations": coco_anns,
            "crs": "EPSG:4326",
            "bounds": [0, 0, w, h],  # Dummy bounds
        }

        with open(out_meta_path, "w") as f:
            json.dump(meta, f)

    print("✅ Processing complete.")


def parse_args():
    ap = argparse.ArgumentParser(description="CanopyAI SAM runner")
    ap.add_argument(
        "--won003_root",
        default="won003",
        help="Path to WON003 root (containing images/ and annotations/)",
    )
    ap.add_argument(
        "--output_dir",
        default="won003_sam",
        help="Output directory for processed tiles",
    )
    ap.add_argument(
        "--checkpoint",
        default="sam_vit_b_01ec64.pth",
        help="Path to SAM checkpoint",
    )
    return ap.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = parse_args()

    process_won003(args.won003_root, args.output_dir, args.checkpoint)
