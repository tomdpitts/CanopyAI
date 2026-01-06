"""
Box-Prompted SAM Dataset for Canopy Segmentation.

Loads bounding box annotations, crops regions, and provides per-box training samples
with canopy (positive) and shadow (negative supervision) masks.

Usage:
    from sam_box_dataset import BoxPromptSAMDataset, get_train_val_split

    train_ds, val_ds = get_train_val_split(
        image_dir="won003_train/images",
        annotation_dir="won003_train/annotations",
        mask_dir="won003_train/train_masks",
    )
"""

import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset


def parse_voc_annotation(xml_path: Path) -> List[Dict[str, int]]:
    """Parse Pascal VOC XML and return list of bounding boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        box = {
            "xmin": int(float(bbox.find("xmin").text)),
            "ymin": int(float(bbox.find("ymin").text)),
            "xmax": int(float(bbox.find("xmax").text)),
            "ymax": int(float(bbox.find("ymax").text)),
        }
        boxes.append(box)

    return boxes


class BoxPromptSAMDataset(Dataset):
    """
    Dataset providing per-box training samples for SAM finetuning.

    Each sample contains:
    - image: Full image or cropped region
    - box: Bounding box coordinates [x1, y1, x2, y2]
    - canopy_mask: Binary mask (positive supervision)
    - shadow_mask: Binary mask (negative supervision)
    """

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        mask_dir: str,
        augment: bool = True,
        seed: int = 42,
        target_size: int = 1024,  # SAM expects 1024x1024
    ):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        self.seed = seed
        self.target_size = target_size

        # Load all box samples
        self.samples = []  # List of (image_name, box_dict)

        for ann_file in self.annotation_dir.glob("*.xml"):
            name = ann_file.stem

            # Check all files exist
            image_path = self.image_dir / f"{name}.png"
            canopy_path = self.mask_dir / f"canopy_mask_{name}.png"
            shadow_path = self.mask_dir / f"shadow_mask_{name}.png"

            if not all(p.exists() for p in [image_path, canopy_path, shadow_path]):
                continue

            # Parse boxes
            boxes = parse_voc_annotation(ann_file)
            for box in boxes:
                self.samples.append((name, box))

        print(
            f"BoxPromptSAMDataset: {len(self.samples)} box samples from "
            f"{len(list(self.annotation_dir.glob('*.xml')))} images"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        name, box = self.samples[idx]

        # Load image and masks
        image = cv2.imread(str(self.image_dir / f"{name}.png"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        canopy_mask = cv2.imread(
            str(self.mask_dir / f"canopy_mask_{name}.png"), cv2.IMREAD_GRAYSCALE
        )
        shadow_mask = cv2.imread(
            str(self.mask_dir / f"shadow_mask_{name}.png"), cv2.IMREAD_GRAYSCALE
        )

        # Normalize masks to [0, 1]
        canopy_mask = (canopy_mask > 127).astype(np.float32)
        shadow_mask = (shadow_mask > 127).astype(np.float32)

        # Get box coordinates
        x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]

        # Apply augmentation if enabled
        if self.augment:
            aug_seed = self.seed + idx
            image, canopy_mask, shadow_mask, (x1, y1, x2, y2) = self._augment(
                image, canopy_mask, shadow_mask, (x1, y1, x2, y2), aug_seed
            )

        # Resize to SAM's expected size
        orig_h, orig_w = image.shape[:2]
        image_resized = cv2.resize(image, (self.target_size, self.target_size))
        canopy_resized = cv2.resize(canopy_mask, (self.target_size, self.target_size))
        shadow_resized = cv2.resize(shadow_mask, (self.target_size, self.target_size))

        # Scale box coordinates
        scale_x = self.target_size / orig_w
        scale_y = self.target_size / orig_h
        box_scaled = [
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y,
        ]

        # Convert to tensors
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        canopy_tensor = torch.from_numpy(canopy_resized).float()
        shadow_tensor = torch.from_numpy(shadow_resized).float()
        box_tensor = torch.tensor(box_scaled, dtype=torch.float32)

        return {
            "image": image_tensor,
            "box": box_tensor,
            "canopy_mask": canopy_tensor,
            "shadow_mask": shadow_tensor,
            "name": name,
            "original_size": (orig_h, orig_w),
        }

    def _augment(
        self,
        image: np.ndarray,
        canopy: np.ndarray,
        shadow: np.ndarray,
        box: Tuple[int, int, int, int],
        seed: int,
    ):
        """Apply augmentations to image, masks, and box."""
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        h, w = image.shape[:2]
        x1, y1, x2, y2 = box

        # Horizontal flip
        if rng.random() < 0.5:
            image = np.fliplr(image).copy()
            canopy = np.fliplr(canopy).copy()
            shadow = np.fliplr(shadow).copy()
            x1, x2 = w - x2, w - x1

        # Vertical flip
        if rng.random() < 0.5:
            image = np.flipud(image).copy()
            canopy = np.flipud(canopy).copy()
            shadow = np.flipud(shadow).copy()
            y1, y2 = h - y2, h - y1

        # Gaussian noise (image only)
        noise_std = rng.uniform(0.01, 0.05) * 255
        noise = np_rng.normal(0, noise_std, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Brightness
        brightness = rng.uniform(0.8, 1.2)
        image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        return image, canopy, shadow, (x1, y1, x2, y2)


def get_train_val_split(
    image_dir: str,
    annotation_dir: str,
    mask_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    **kwargs,
) -> Tuple[Dataset, Dataset]:
    """Create train/val datasets with deterministic split by IMAGE (not by box)."""

    # Get unique image names
    ann_dir = Path(annotation_dir)
    image_names = [p.stem for p in ann_dir.glob("*.xml")]

    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(image_names)

    # Split
    n_val = max(1, int(len(image_names) * val_ratio))
    val_names = set(image_names[:n_val])
    train_names = set(image_names[n_val:])

    print(f"Train/Val split: {len(train_names)} / {len(val_names)} images")

    # Create datasets
    train_ds = BoxPromptSAMDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        mask_dir=mask_dir,
        augment=True,
        seed=seed,
        **kwargs,
    )

    val_ds = BoxPromptSAMDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        mask_dir=mask_dir,
        augment=False,
        seed=seed,
        **kwargs,
    )

    # Filter samples by image name
    train_ds.samples = [(n, b) for n, b in train_ds.samples if n in train_names]
    val_ds.samples = [(n, b) for n, b in val_ds.samples if n in val_names]

    print(f"Train boxes: {len(train_ds.samples)}, Val boxes: {len(val_ds.samples)}")

    return train_ds, val_ds


if __name__ == "__main__":
    # Test the dataset
    train_ds, val_ds = get_train_val_split(
        image_dir="won003_train/images",
        annotation_dir="won003_train/annotations",
        mask_dir="won003_train/train_masks",
        val_ratio=0.2,
    )

    # Test a sample
    sample = train_ds[0]
    print(f"\nSample from '{sample['name']}':")
    print(f"  Image: {sample['image'].shape}")
    print(f"  Box: {sample['box'].tolist()}")
    print(f"  Canopy sum: {sample['canopy_mask'].sum():.0f}")
    print(f"  Shadow sum: {sample['shadow_mask'].sum():.0f}")
