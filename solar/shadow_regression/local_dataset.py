#!/usr/bin/env python3
"""
Custom dataset class for loading local images with manual annotations.
"""

import os
import math
import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class LocalShadowDataset(Dataset):
    """
    Load images from a local directory with optional manual annotations.

    Args:
        image_dir: Directory containing images
        annotation_csv: Path to CSV with columns: filename, shadow_azimuth, skipped
        transform: Transform to apply to images
    """

    def __init__(self, image_dir, annotation_csv=None, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Load annotations if provided
        self.annotations = {}
        if annotation_csv and os.path.exists(annotation_csv):
            with open(annotation_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["skipped"].lower() == "true":
                        continue  # Skip images without shadows
                    filename = row["filename"]
                    azimuth = float(row["shadow_azimuth"])
                    self.annotations[filename] = azimuth
            print(f"Loaded {len(self.annotations)} annotations from {annotation_csv}")

        # Find all image files
        self.image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
            self.image_files.extend(list(self.image_dir.glob(ext)))

        # Filter to only annotated images if annotations provided
        if self.annotations:
            self.image_files = [
                f for f in self.image_files if f.name in self.annotations
            ]
            print(f"Filtered to {len(self.image_files)} annotated images")
        else:
            print(f"Found {len(self.image_files)} images (no annotations)")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        # Get shadow azimuth from annotations or return None
        shadow_azimuth = self.annotations.get(img_path.name, None)

        if shadow_azimuth is None:
            # No ground truth - return zero vector as placeholder
            target_vector = torch.tensor([0.0, 0.0], dtype=torch.float32)
        else:
            # Convert azimuth to unit vector
            theta_rad = math.radians(shadow_azimuth)
            target_vector = torch.tensor(
                [math.sin(theta_rad), math.cos(theta_rad)], dtype=torch.float32
            )

        if self.transform:
            image, target_vector = self.transform(image, target_vector)

        return image, target_vector


# Simple transform without rotation (for validation)
class SimpleTransform:
    def __init__(self, size=224):
        self.size = size
        self.resize = transforms.Resize((size, size))
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, target_vector):
        image = self.resize(image)
        image = self.to_tensor(image)
        return image, target_vector
