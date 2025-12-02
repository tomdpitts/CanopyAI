#!/usr/bin/env python3

import cv2
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def visualize_bboxes(root_dir):
    root_dir = Path(root_dir)
    img_dir = root_dir / "images"
    ann_dir = root_dir / "annotations"
    vis_dir = root_dir / "visualise_bboxes"
    vis_dir.mkdir(exist_ok=True)

    xml_files = sorted(list(ann_dir.glob("*.xml")))
    print(f"Found {len(xml_files)} XML files.")

    for idx, xml_file in enumerate(xml_files):
        # 1. Load Image
        stem = xml_file.stem
        img_files = list(img_dir.glob(f"{stem}.*"))
        if not img_files:
            print(f"⚠️ Warning: No image found for {xml_file.name}, skipping.")
            continue
        img_path = img_files[0]

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Failed to read image {img_path}")
            continue

        # 2. Parse XML for bboxes
        tree = ET.parse(xml_file)
        root = tree.getroot()

        boxes = []
        for member in root.findall("object"):
            if member.find("name").text != "tree":
                continue
            bndbox = member.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            boxes.append([xmin, ymin, xmax, ymax])

        # 3. Draw BBoxes
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            # Draw rectangle (Blue, thickness 2)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        # 4. Save
        out_path = vis_dir / f"{stem}_bbox_vis.png"
        cv2.imwrite(str(out_path), img)
        print(f"Saved {out_path.name}")

    print("✅ Visualization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--won003_root", default="won003", help="Path to WON003 root")
    args = parser.parse_args()

    visualize_bboxes(args.won003_root)
