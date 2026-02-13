#!/usr/bin/env python3
"""
Visualize DeepForest training annotations to verify data quality.

Usage:
    python visualize_annotations.py --csv tcd_train.csv --num-samples 3
"""

import argparse
import random
from pathlib import Path

import cv2
import pandas as pd


def visualize_annotations(csv_path, output_dir="visualization_output", num_samples=3):
    """
    Visualize bounding boxes from DeepForest CSV on sample images.

    Args:
        csv_path: Path to DeepForest CSV file
        output_dir: Directory to save visualizations
        num_samples: Number of random images to visualize
    """
    # Read annotations
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} annotations from {csv_path}")

    # Get unique images
    unique_images = df["image_path"].unique()
    print(f"📷 Found {len(unique_images)} unique images")

    # Sample random images
    sample_images = random.sample(
        list(unique_images), min(num_samples, len(unique_images))
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n🎨 Visualizing {len(sample_images)} images...")

    for img_path in sample_images:
        # Get all annotations for this image
        img_annotations = df[df["image_path"] == img_path]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️  Could not load {img_path}")
            continue

        print(f"\n📸 {Path(img_path).name}")
        print(f"   Boxes: {len(img_annotations)}")
        print(f"   Image size: {image.shape[1]}x{image.shape[0]}")

        # Draw bounding boxes
        for _, row in img_annotations.iterrows():
            xmin, ymin = int(row["xmin"]), int(row["ymin"])
            xmax, ymax = int(row["xmax"]), int(row["ymax"])
            label = row["label"]

            # Draw rectangle (green for trees)
            color = (0, 255, 0) if label == "Tree" else (0, 0, 255)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

            # Add label
            cv2.putText(
                image,
                label,
                (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # Save visualization
        output_file = output_path / f"{Path(img_path).stem}_annotated.png"
        cv2.imwrite(str(output_file), image)
        print(f"   ✅ Saved to {output_file}")

    print(f"\n✨ Done! Visualizations saved to {output_path}/")


def main():
    parser = argparse.ArgumentParser(description="Visualize DeepForest annotations")
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to DeepForest CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualization_output",
        help="Directory to save visualizations (default: visualization_output)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of random images to visualize (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    visualize_annotations(
        csv_path=args.csv,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
