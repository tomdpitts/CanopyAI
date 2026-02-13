"""
Custom rotation augmentation for FiLM training.

This module provides rotation augmentation that simultaneously:
1. Rotates the image
2. Transforms bounding box coordinates
3. Rotates the shadow vector

When an image is rotated by θ degrees, the shadow direction also rotates by θ degrees.
This provides natural shadow diversity for FiLM training.
"""

import torch
import cv2
import numpy as np
from typing import Tuple


def rotate_image_and_boxes(
    image: np.ndarray, boxes: np.ndarray, angle_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image and transform bounding boxes.

    Args:
        image: Image array (H, W, C)
        boxes: Bounding boxes (N, 4) in [xmin, ymin, xmax, ymax] format
        angle_deg: Rotation angle in degrees (positive = counter-clockwise)

    Returns:
        rotated_image: Rotated image
        rotated_boxes: Transformed bounding boxes
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # Rotate image
    rotated_image = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Transform bounding boxes
    if len(boxes) == 0:
        return rotated_image, boxes

    rotated_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box

        # Get all 4 corners
        corners = np.array(
            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32
        )

        # Add ones for affine transform
        ones = np.ones((4, 1))
        corners_homogeneous = np.hstack([corners, ones])

        # Transform corners
        rotated_corners = M.dot(corners_homogeneous.T).T

        # Get new bbox from rotated corners
        new_xmin = rotated_corners[:, 0].min()
        new_ymin = rotated_corners[:, 1].min()
        new_xmax = rotated_corners[:, 0].max()
        new_ymax = rotated_corners[:, 1].max()

        # Clip to image bounds
        new_xmin = np.clip(new_xmin, 0, w)
        new_ymin = np.clip(new_ymin, 0, h)
        new_xmax = np.clip(new_xmax, 0, w)
        new_ymax = np.clip(new_ymax, 0, h)

        # Only keep valid boxes (width > 0 and height > 0)
        if (new_xmax > new_xmin) and (new_ymax > new_ymin):
            rotated_boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])

    # If all boxes were rotated out of frame, return empty array
    if not rotated_boxes:
        return rotated_image, np.empty((0, 4), dtype=np.float32)

    return rotated_image, np.array(rotated_boxes, dtype=np.float32)


def rotate_shadow_vector(shadow_vector: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate shadow vector by the same angle as the image.

    When image rotates by θ, shadow direction also rotates by θ.
    If shadow was at angle α, after rotation it's at angle (α + θ).

    Args:
        shadow_vector: Shadow vector (2,) as [sin(α), cos(α)]
        angle_deg: Rotation angle in degrees

    Returns:
        Rotated shadow vector (2,) as [sin(α + θ), cos(α + θ)]
    """
    # Get current shadow angle
    current_angle = np.arctan2(shadow_vector[0], shadow_vector[1])

    # Add rotation
    new_angle = current_angle + np.radians(angle_deg)

    # Convert back to (sin, cos)
    return np.array([np.sin(new_angle), np.cos(new_angle)], dtype=np.float32)


def create_rotation_collate_fn(base_shadow_angle_deg=215.0, rotation_range=180):
    """
    Create a custom collate function that applies rotation augmentation.

    Args:
        base_shadow_angle_deg: Base shadow direction for dataset
        rotation_range: Maximum rotation angle (±degrees)

    Returns:
        Collate function for DataLoader
    """
    # Convert base shadow to vector
    base_angle_rad = np.radians(base_shadow_angle_deg)
    base_shadow_vector = np.array([np.sin(base_angle_rad), np.cos(base_angle_rad)])

    def collate_fn(batch):
        """
        Collate batch with rotation augmentation.

        batch: List of (image, target) tuples from dataset
        Returns: (images, targets, shadow_vectors)
        """
        images = []
        targets = []
        shadow_vectors = []

        for image, target in batch:
            # Random rotation angle
            angle = np.random.uniform(-rotation_range, rotation_range)

            # Convert image to numpy if tensor
            if isinstance(image, torch.Tensor):
                img_np = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
            else:
                img_np = image

            # Extract boxes from target
            boxes_np = (
                target["boxes"].numpy()
                if isinstance(target["boxes"], torch.Tensor)
                else target["boxes"]
            )

            # Rotate image and boxes
            rotated_img, rotated_boxes = rotate_image_and_boxes(img_np, boxes_np, angle)

            # Rotate shadow vector
            rotated_shadow = rotate_shadow_vector(base_shadow_vector, angle)

            # Convert back to tensor
            if isinstance(image, torch.Tensor):
                rotated_img = torch.from_numpy(rotated_img).permute(
                    2, 0, 1
                )  # (H, W, C) -> (C, H, W)

            # Update target with rotated boxes
            target["boxes"] = torch.from_numpy(rotated_boxes).float()

            images.append(rotated_img)
            targets.append(target)
            shadow_vectors.append(rotated_shadow)

        # Stack shadow vectors
        shadow_vectors = torch.from_numpy(np.array(shadow_vectors)).float()

        return images, targets, shadow_vectors

    return collate_fn
