#!/usr/bin/env python3
"""
Prepare local images for annotation.
Copies images from a directory to make them easy to load in the browser tool.
"""

import os
import shutil
import argparse
from pathlib import Path


def prepare_local_images(image_dir, output_dir="annotation_images", n_samples=None):
    """
    Copy images from a local directory for annotation.

    Args:
        image_dir: Source directory with images
        output_dir: Output directory for annotation
        n_samples: Number of random samples (None = all images)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all image files
    image_dir = Path(image_dir)
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        image_files.extend(list(image_dir.glob(ext)))

    print(f"Found {len(image_files)} images in {image_dir}")

    # Sample if requested
    if n_samples and n_samples < len(image_files):
        import random

        image_files = random.sample(image_files, n_samples)
        print(f"Randomly selected {n_samples} images")

    # Copy to output directory
    print(f"Copying {len(image_files)} images to {output_dir}/...")
    for i, img_path in enumerate(image_files):
        dest = Path(output_dir) / img_path.name
        shutil.copy2(img_path, dest)

        if (i + 1) % 10 == 0:
            print(f"  Copied {i + 1}/{len(image_files)}")

    print(f"\nDone! {len(image_files)} images ready for annotation.")
    print(f"\nNext steps:")
    print(f"1. Open annotate.html in your browser")
    print(
        f"2. Click 'Choose Files' and select all images from: {os.path.abspath(output_dir)}"
    )
    print(f"3. Annotate shadow directions")
    print(f"4. Export annotations as shadow_annotations.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare local images for annotation")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing images to annotate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="annotation_images",
        help="Output directory for annotation workflow",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of random samples (default: all)",
    )

    args = parser.parse_args()
    prepare_local_images(args.image_dir, args.output_dir, args.n_samples)
