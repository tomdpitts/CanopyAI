"""
Mask extraction utilities for SAM training.

These functions extract per-tree masks from full canopy/shadow masks
based on bounding box prompts.
"""

import cv2
import numpy as np


def extract_tree_mask_for_box(canopy_mask, box):
    """Extract the single tree crown with maximum IoU with the bounding box.

    Uses connected component labeling to isolate individual tree crowns,
    then selects the ONE component with highest IoU with the box region.
    This ensures only one tree per box, even with partial overlaps.

    Args:
        canopy_mask: Binary mask (H, W) with all canopy pixels
        box: Dict with xmin, ymin, xmax, ymax

    Returns:
        Binary mask (H, W) containing only the single best-matching tree
    """
    x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
    box_area = (x2 - x1) * (y2 - y1)

    # Label connected components in the canopy mask
    num_labels, labels = cv2.connectedComponents((canopy_mask > 0.5).astype(np.uint8))

    # Create box region mask
    box_region = np.zeros_like(canopy_mask, dtype=np.uint8)
    box_region[y1:y2, x1:x2] = 1

    # Find which labels intersect with the box
    labels_in_box = np.unique(labels * box_region)
    labels_in_box = labels_in_box[labels_in_box > 0]  # Exclude background

    if len(labels_in_box) == 0:
        # No tree intersects - return empty mask
        return np.zeros_like(canopy_mask, dtype=np.float32)

    # Calculate IoU for each candidate tree
    best_label = None
    best_iou = -1

    for label in labels_in_box:
        component_mask = (labels == label).astype(np.float32)
        component_area = component_mask.sum()

        # Intersection with box
        intersection = (component_mask * box_region).sum()

        # Union = component + box - intersection
        union = component_area + box_area - intersection

        iou = intersection / (union + 1e-6)

        if iou > best_iou:
            best_iou = iou
            best_label = label

    # Create output mask with only the best tree
    tree_mask = np.zeros_like(canopy_mask, dtype=np.float32)
    if best_label is not None:
        tree_mask[labels == best_label] = 1.0

    return tree_mask


def extract_shadow_for_tree(shadow_mask, tree_mask, dilation_px=10):
    """Extract shadow region associated with this tree.

    Strategy: Dilate the tree mask and find shadow pixels nearby.
    This captures the cast shadow even if it's not directly connected.

    Args:
        shadow_mask: Binary mask (H, W) with all shadow pixels
        tree_mask: Binary mask (H, W) with this tree's canopy
        dilation_px: Pixels to dilate tree mask for shadow searching

    Returns:
        Binary mask (H, W) containing shadow near this tree
    """
    # Dilate tree mask to create search region
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
    )
    search_region = cv2.dilate(tree_mask.astype(np.uint8), kernel)

    # Find shadow within search region (but not overlapping canopy)
    tree_shadow = shadow_mask * search_region * (1 - tree_mask)

    return tree_shadow.astype(np.float32)
