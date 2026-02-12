"""
Test script for mask extraction functions.

Imports the REAL functions from mask_utils.py and creates visualizations
for manual inspection.
"""

import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

# Import the actual functions we're using in training
from mask_utils import extract_tree_mask_for_box, extract_shadow_for_tree


def parse_voc(xml_path):
    """Parse VOC XML annotation file."""
    tree = ET.parse(xml_path)
    boxes = []
    for obj in tree.getroot().findall("object"):
        bbox = obj.find("bndbox")
        boxes.append(
            {
                "xmin": int(float(bbox.find("xmin").text)),
                "ymin": int(float(bbox.find("ymin").text)),
                "xmax": int(float(bbox.find("xmax").text)),
                "ymax": int(float(bbox.find("ymax").text)),
            }
        )
    return boxes


def create_visualization(img, box, tree_mask, tree_shadow, full_canopy):
    """Create side-by-side visualization."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]

    # Panel 1: Original image with bbox
    panel1 = img.copy()
    cv2.rectangle(panel1, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(
        panel1,
        "Image + BBox",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Panel 2: Full canopy mask (all trees)
    panel2 = np.zeros_like(img)
    panel2[:, :, 1] = (full_canopy * 255).astype(np.uint8)  # Green
    cv2.rectangle(panel2, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(
        panel2,
        "Full Canopy (all trees)",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Panel 3: Extracted tree mask (only intersecting tree)
    panel3 = np.zeros_like(img)
    panel3[:, :, 1] = (tree_mask * 255).astype(np.uint8)  # Green
    cv2.rectangle(panel3, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(
        panel3,
        "Extracted Tree (GT)",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Panel 4: Tree shadow
    panel4 = np.zeros_like(img)
    panel4[:, :, 1] = (tree_mask * 255).astype(np.uint8)  # Green for tree
    panel4[:, :, 2] = (tree_shadow * 255).astype(np.uint8)  # Red for shadow
    cv2.rectangle(panel4, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(
        panel4,
        "Tree (G) + Shadow (R)",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Combine panels (2x2 grid)
    top_row = np.hstack([panel1, panel2])
    bottom_row = np.hstack([panel3, panel4])
    combined = np.vstack([top_row, bottom_row])

    return combined


def main():
    # Paths
    data_dir = Path("won003_train")  # Update 12th Feb: won003 -> won003_train
    output_dir = Path("mask_extraction_test")
    output_dir.mkdir(exist_ok=True)

    image_dir = data_dir / "images"
    ann_dir = data_dir / "annotations"
    mask_dir = data_dir / "train_masks"

    # Get all annotation files
    ann_files = list(ann_dir.glob("*.xml"))
    print(f"Found {len(ann_files)} annotation files")
    print("Using functions from mask_utils.py")

    # Process first 3 images, up to 3 boxes each
    count = 0
    max_total = 9

    for ann_file in ann_files[:5]:
        name = ann_file.stem
        img_path = image_dir / f"{name}.png"
        canopy_path = mask_dir / f"canopy_mask_{name}.png"
        shadow_path = mask_dir / f"shadow_mask_{name}.png"

        if not all(p.exists() for p in [img_path, canopy_path, shadow_path]):
            continue

        # Load data
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        canopy = cv2.imread(str(canopy_path), 0)
        shadow = cv2.imread(str(shadow_path), 0)

        canopy = (canopy > 127).astype(np.float32)
        shadow = (shadow > 127).astype(np.float32)

        boxes = parse_voc(ann_file)
        print(f"\n{name}: {len(boxes)} boxes")

        for i, box in enumerate(boxes[:3]):
            # Extract per-tree masks using the REAL functions
            tree_mask = extract_tree_mask_for_box(canopy, box)
            tree_shadow = extract_shadow_for_tree(shadow, tree_mask)

            # Stats
            full_pixels = canopy.sum()
            tree_pixels = tree_mask.sum()
            shadow_pixels = tree_shadow.sum()
            print(
                f"  Box {i}: full={full_pixels:.0f}, tree={tree_pixels:.0f}, "
                f"shadow={shadow_pixels:.0f}"
            )

            # Create visualization
            vis = create_visualization(img, box, tree_mask, tree_shadow, canopy)

            # Save
            out_path = output_dir / f"{name}_box{i}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {out_path}")

            count += 1
            if count >= max_total:
                break

        if count >= max_total:
            break

    print(f"\n✅ Done! Check {output_dir}/ for visualizations")


if __name__ == "__main__":
    main()
