#!/usr/bin/env python3
"""
Download 100 SJER images from NeonTreeEvaluation for manual annotation.
"""

import os
import random
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def prepare_sjer_for_annotation(output_dir="annotation_data/sjer_images", n_images=100):
    """Download SJER images from HuggingFace and prepare for annotation."""

    os.makedirs(output_dir, exist_ok=True)

    print("Loading NeonTreeEvaluation dataset from HuggingFace...")
    print("(This may take a few minutes on first run)\n")

    # Load all splits and combine
    all_images = []
    for split in ["train", "validation", "test"]:
        print(f"Loading {split} split...")
        dataset = load_dataset("CanopyRS/NeonTreeEvaluation", split=split)

        # Filter for SJER site
        sjer_samples = [
            sample for sample in dataset if "SJER" in sample.get("tile_name", "")
        ]
        all_images.extend(sjer_samples)
        print(f"  Found {len(sjer_samples)} SJER images in {split}")

    print(f"\nTotal SJER images available: {len(all_images)}")

    if len(all_images) < n_images:
        print(
            f"Warning: Only {len(all_images)} SJER images available, using all of them."
        )
        n_images = len(all_images)

    # Randomly sample
    sampled = random.sample(all_images, n_images)

    print(f"\nSaving {n_images} SJER images to {output_dir}...")

    for i, sample in enumerate(tqdm(sampled, desc="Saving images")):
        # Get image and metadata
        img = sample["image"]
        tile_name = sample["tile_name"]

        # Save as PNG
        output_path = os.path.join(output_dir, f"{tile_name}.png")
        img.save(output_path)

    print(f"\n✓ {n_images} SJER images saved to {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Open annotation_data/annotate.html in your browser")
    print(f"  2. Annotate shadow directions for each image")
    print(f"  3. Export CSV when done")
    print(f"  4. Save CSV as 'data/sjer_annotations.csv'")
    print(f"  5. Train model with combined WON003 + SJER data")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare SJER images for annotation")
    parser.add_argument("--output_dir", type=str, default="annotation_data/sjer_images")
    parser.add_argument("--n_images", type=int, default=100)

    args = parser.parse_args()

    prepare_sjer_for_annotation(args.output_dir, args.n_images)
