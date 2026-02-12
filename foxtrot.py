#!/usr/bin/env python3

"""
Two-Stage Tree Detection Pipeline: DeepForest + SAM
---------------------------------------------------
This script implements a custom 2-stage model that:
1. Uses DeepForest to detect trees in sparse arid landscapes
2. Produces bounding boxes around detected trees
3. Uses SAM (Segment Anything Model) to segment the object within each bounding box

Usage:
    python foxtrot.py --image_path data/tcd/bin_liang/tcd_tile_WON.tif

"""

import argparse
import os
import json
import tempfile
import gc
import shutil
import time
import math
import random
import numpy as np
import rasterio
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from deepforest import main as deepforest_main
from deepforest import utilities
from shapely.geometry import Polygon

# Fix MPS float64 compatibility
torch.set_default_dtype(torch.float32)


def load_image_from_tif(tif_path):
    """
    Load image from TIF file and return as RGB numpy array.

    Args:
        tif_path: Path to input TIF file

    Returns:
        image: RGB image as (H, W, 3) numpy array
        crs: Coordinate reference system
        transform: Geospatial transform
        bounds: Geographic bounds
    """
    with rasterio.open(tif_path) as src:
        # Read RGB bands (assuming bands 1,2,3 are RGB)
        image = src.read([1, 2, 3])  # Shape: (3, H, W)
        image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, 3)

        # Store georeferencing info
        crs = src.crs
        transform = src.transform
        bounds = src.bounds

        print(f"✅ Loaded image: {image.shape}")
        print(f"   CRS: {crs}")
        print(f"   Bounds: {bounds}")

    # Normalize image to 0-255 if needed
    if image.max() > 255:
        image = (image / image.max() * 255).astype(np.uint8)

    return image, crs, transform, bounds


def detect_shadows(image, threshold_ratio=0.7, blur_size=51):
    """
    Detect shadow regions using adaptive luminance thresholding.

    Args:
        image: RGB image as numpy array (H, W, C), uint8
        threshold_ratio: Pixels below (local_mean * ratio) are shadows
        blur_size: Gaussian blur kernel size for local mean

    Returns:
        Binary shadow mask (H, W), uint8 with 1=shadow, 0=non-shadow
    """
    # Convert to LAB and extract luminance
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)

    # Compute local mean luminance
    local_mean = cv2.GaussianBlur(L, (blur_size, blur_size), 0)

    # Threshold: pixels darker than local mean are shadows
    shadow_mask = (L < local_mean * threshold_ratio).astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    return shadow_mask


def compute_iou(box1, box2):
    """Compute Intersection over Union between two boxes [xmin, ymin, xmax, ymax]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    iou = intersection / union if union > 0 else 0

    # Also compute coverage: what fraction of each box is covered by intersection
    coverage1 = intersection / area1 if area1 > 0 else 0
    coverage2 = intersection / area2 if area2 > 0 else 0

    return iou, coverage1, coverage2


def apply_nms(bboxes, scores, iou_threshold=0.5, coverage_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    Suppresses if IoU > iou_threshold OR if one box is >coverage_threshold
    covered by another (handles small boxes inside larger ones).

    Args:
        bboxes: List of bounding boxes [xmin, ymin, xmax, ymax]
        scores: List of confidence scores
        iou_threshold: IoU threshold for suppression (default 0.5)
        coverage_threshold: Coverage threshold - suppress if box is this
                           fraction covered by another (default 0.5)

    Returns:
        keep_indices: List of indices to keep
    """
    if len(bboxes) == 0:
        return []

    # Sort by score (descending)
    indices = np.argsort(scores)[::-1].tolist()
    keep = []

    while indices:
        # Keep the highest scoring box
        current = indices.pop(0)
        keep.append(current)

        # Remove boxes with high IoU overlap OR high coverage
        remaining = []
        for idx in indices:
            iou, coverage_current, coverage_idx = compute_iou(
                bboxes[current], bboxes[idx]
            )
            # Suppress if: high IoU OR the candidate box is mostly covered
            if iou < iou_threshold and coverage_idx < coverage_threshold:
                remaining.append(idx)
        indices = remaining

    return keep


# =============================================================================
# Stage 0: Shadow Vector Prediction (Global Context)
# =============================================================================


def extract_safe_crops(image, n_crops=30, crop_size=500, margin_fraction=0.2):
    """
    Extract random 500x500 crops from center region of image, avoiding edges.

    Args:
        image: RGB image as (H, W, 3) numpy array
        n_crops: Number of crops to extract
        crop_size: Size of each crop (default 500px)
        margin_fraction: Fraction of image to avoid at edges (default 0.2 = 20%)

    Returns:
        List of PIL Image crops
    """
    h, w = image.shape[:2]

    # Define safe extraction zone (center, avoiding margins)
    margin_w = int(w * margin_fraction)
    margin_h = int(h * margin_fraction)

    safe_min_col = margin_w
    safe_max_col = w - margin_w - crop_size
    safe_min_row = margin_h
    safe_max_row = h - margin_h - crop_size

    # Fallback if image is too small
    if safe_max_col <= safe_min_col or safe_max_row <= safe_min_row:
        safe_min_col = max(0, crop_size)
        safe_max_col = max(crop_size, w - crop_size)
        safe_min_row = max(0, crop_size)
        safe_max_row = max(0, h - crop_size)

    crops = []
    for _ in range(n_crops):
        if safe_max_col <= safe_min_col or safe_max_row <= safe_min_row:
            # Image too small for safe cropping
            break

        col_off = random.randint(safe_min_col, safe_max_col)
        row_off = random.randint(safe_min_row, safe_max_row)

        crop_np = image[row_off : row_off + crop_size, col_off : col_off + crop_size]
        crop_pil = Image.fromarray(crop_np)
        crops.append(crop_pil)

    return crops


def predict_shadow_vector_from_crops(crops, model, device):
    """
    Predict shadow vector from multiple crops using the trained model.

    Args:
        crops: List of PIL Image crops (500x500)
        model: Trained shadow regression model
        device: torch device

    Returns:
        List of predicted vectors (N, 2) as numpy arrays
    """
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    vectors = []
    model.eval()

    with torch.no_grad():
        for crop in crops:
            img_tensor = preprocess(crop).unsqueeze(0).to(device)
            pred = model(img_tensor).cpu().squeeze().numpy()
            vectors.append(pred)

    return np.array(vectors)  # (N, 2)


def compute_circular_mean_with_outlier_rejection(vectors, outlier_threshold_deg=45.0):
    """
    Compute circular mean of shadow vectors with democratic outlier rejection.
    Majority wins, minority is ignored.

    Args:
        vectors: Array of shadow vectors (N, 2)
        outlier_threshold_deg: Reject vectors > this many degrees from median

    Returns:
        mean_vector: (2,) array - circular mean of inliers
        n_inliers: Number of agreeing crops
        n_outliers: Number of rejected outliers
        circular_std: Circular standard deviation in degrees
    """
    # Convert vectors to angles
    angles = np.arctan2(vectors[:, 0], vectors[:, 1])  # radians
    angles_deg = np.degrees(angles) % 360

    # Compute circular median as robust center estimate
    # Use mean of sin and cos for circular median approximation
    sin_mean = np.mean(np.sin(np.radians(angles_deg)))
    cos_mean = np.mean(np.cos(np.radians(angles_deg)))
    median_angle = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360

    # Compute circular distance from median
    def circular_distance(a1, a2):
        """Compute minimum angular distance between two angles."""
        diff = np.abs(a1 - a2)
        return np.minimum(diff, 360 - diff)

    distances = circular_distance(angles_deg, median_angle)

    # Democratic outlier rejection: keep predictions within threshold
    inlier_mask = distances <= outlier_threshold_deg
    n_inliers = inlier_mask.sum()
    n_outliers = (~inlier_mask).sum()

    if n_inliers == 0:
        # All rejected (shouldn't happen with 45° threshold, but handle it)
        # Fall back to using all
        inlier_mask = np.ones(len(vectors), dtype=bool)
        n_inliers = len(vectors)
        n_outliers = 0

    # Compute circular mean of inliers
    inlier_angles_rad = angles[inlier_mask]
    mean_sin = np.mean(np.sin(inlier_angles_rad))
    mean_cos = np.mean(np.cos(inlier_angles_rad))
    mean_angle_rad = np.arctan2(mean_sin, mean_cos)

    # Convert to unit vector
    mean_vector = np.array([np.sin(mean_angle_rad), np.cos(mean_angle_rad)])

    # Compute circular standard deviation (measure of spread)
    # R = length of mean vector (1 = perfect agreement, 0 = uniform spread)
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    circular_std = np.degrees(np.sqrt(-2 * np.log(R))) if R > 0 else 180.0

    return mean_vector, n_inliers, n_outliers, circular_std


def predict_shadow_vector(
    image, shadow_model_path, device, n_crops=30, outlier_threshold_deg=45.0
):
    """
    Stage 0: Predict global shadow direction from orthomosaic.

    Uses ResNet-34 shadow regression model to predict shadow vectors from
    multiple random crops, then computes circular mean with outlier rejection.

    Args:
        image: RGB image as (H, W, 3) numpy array
        shadow_model_path: Path to trained shadow model (.pth)
        device: torch device
        n_crops: Number of crops to sample (default 30)
        outlier_threshold_deg: Outlier rejection threshold in degrees (default 45)

    Returns:
        shadow_vector: (2,) unit vector or None if prediction failed
        stats: Dict with prediction statistics
    """
    print(f"\n🌞 Stage 0: Shadow Vector Prediction...")
    print(f"   Sampling {n_crops} random 500×500 crops...")

    # Import model class (inline to avoid external dependency)
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "solar" / "shadow_regression"))
    from train_combined import ShadowResNet34

    # Load model
    model = ShadowResNet34()
    model.load_state_dict(torch.load(shadow_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Extract crops
    crops = extract_safe_crops(image, n_crops=n_crops)

    if len(crops) == 0:
        print("   ⚠️  Image too small for crop extraction")
        return None, {"error": "image_too_small"}

    if len(crops) < n_crops:
        print(f"   ⚠️  Only extracted {len(crops)}/{n_crops} crops (image small)")

    # Predict on each crop
    print(f"   Running inference on {len(crops)} crops...")
    vectors = predict_shadow_vector_from_crops(crops, model, device)

    # Compute circular mean with outlier rejection
    mean_vector, n_inliers, n_outliers, circular_std = (
        compute_circular_mean_with_outlier_rejection(
            vectors, outlier_threshold_deg=outlier_threshold_deg
        )
    )

    # Convert to angle for logging
    shadow_angle = math.degrees(math.atan2(mean_vector[0], mean_vector[1])) % 360

    # Log consensus statistics
    consensus_pct = (n_inliers / len(crops)) * 100
    print(f"   ✅ Shadow direction: {shadow_angle:.1f}°")
    print(
        f"   📊 Consensus: {n_inliers}/{len(crops)} crops ({consensus_pct:.0f}%) | "
        f"Rejected: {n_outliers} outliers"
    )
    print(f"   📏 Circular std: {circular_std:.1f}°")

    # Prepare stats
    stats = {
        "shadow_angle_deg": shadow_angle,
        "n_crops": len(crops),
        "n_inliers": n_inliers,
        "n_outliers": n_outliers,
        "consensus_pct": consensus_pct,
        "circular_std_deg": circular_std,
        "shadow_vector": mean_vector.tolist(),
    }

    # Quality check: warn if low consensus
    if consensus_pct < 70:
        print(
            f"   ⚠️  Low consensus ({consensus_pct:.0f}%) - predictions may be unreliable"
        )
        stats["reliable"] = False
    else:
        stats["reliable"] = True

    return mean_vector, stats


def detect_trees_deepforest(
    image,
    model_path=None,
    tile_size=400,
    tile_overlap=0.05,
    confidence_threshold=0.35,
):
    """
    Pass 1: Run DeepForest detection on 400px tiles.

    Args:
        image: RGB image as (H, W, 3) numpy array
        model_path: Optional path to custom DeepForest model (.pth file)
        tile_size: Size of each tile (default 400px - DeepForest's native)
        tile_overlap: Overlap between tiles (default 0.05 = 5%)
        confidence_threshold: Minimum confidence score for detections

    Returns:
        all_bboxes: List of bounding boxes in global coordinates [xmin, ymin, xmax, ymax]
        all_scores: List of confidence scores
    """
    import warnings

    print("\n🌲 Pass 1: DeepForest Detection...")

    # Initialize DeepForest model
    df_model = deepforest_main.deepforest()

    if model_path:
        print(f"   Loading custom model: {model_path}")
        df_model.model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        print("   Loading default model from Hugging Face...")
        df_model.load_model("weecology/deepforest-tree")

    # Set device for GPU acceleration
    # NOTE: DeepForest v2.0.0 uses self.device to move input tensors during predict_image.
    # Simply calling model.to(device) is NOT sufficient - must set df_model.device.
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        # Move model to MPS if available to speed up detection
        device = torch.device("mps")
        df_model.model.to(device)
        print(f"   🖥️  DeepForest on: {device}")
    else:
        print("   🖥️  DeepForest on: cpu")

    # Important: Set to evaluation mode for inference
    # This prevents "targets should not be none" error when running manual inference
    df_model.model.eval()

    # Convert image to float32 for DeepForest
    if image.dtype == np.uint8:
        image_float = image.astype("float32")
    else:
        image_float = image

    h, w = image.shape[:2]
    stride = int(tile_size * (1 - tile_overlap))

    # Calculate tile count
    num_tiles_y = max(1, (h - tile_size) // stride + 1) if h > tile_size else 1
    num_tiles_x = max(1, (w - tile_size) // stride + 1) if w > tile_size else 1
    total_tiles = num_tiles_y * num_tiles_x

    print(f"   Image size: {w}x{h}")
    print(f"   Tile size: {tile_size}px, overlap: {int(tile_overlap * 100)}%")
    print(f"   Processing {total_tiles} tiles\n")

    # Suppress DeepForest warnings
    warnings.filterwarnings("ignore", message=".*image_path.*")
    warnings.filterwarnings("ignore", message=".*root_dir.*")

    all_bboxes = []
    all_scores = []
    tile_count = 0

    # Process tiles
    y_starts = range(0, h, stride) if h > tile_size else [0]
    x_starts = range(0, w, stride) if w > tile_size else [0]

    for y_start in tqdm(list(y_starts), desc="   DF tiles"):
        for x_start in x_starts:
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)

            # Extract tile
            tile = image_float[y_start:y_end, x_start:x_end]

            # Skip very small edge tiles
            if tile.shape[0] < 200 or tile.shape[1] < 200:
                continue

            tile_count += 1

            # Run prediction with manual device handling to support MPS
            try:
                # Custom prediction logic to bypass DeepForest bug
                # Prepare input tensor on correct device
                image_tensor = torch.tensor(tile).permute(2, 0, 1).float() / 255.0
                image_tensor = image_tensor.to(device).unsqueeze(0)

                # Inference
                with torch.no_grad():
                    prediction = df_model.model(image_tensor)

                # Post-process
                if len(prediction[0]["boxes"]) > 0:
                    tile_preds = utilities.format_boxes(prediction[0])
                    # Filter by confidence strictly
                    tile_preds = tile_preds[tile_preds["score"] >= confidence_threshold]
                else:
                    tile_preds = None
            except Exception as e:
                # Log error but continue (likely edge case or corrupt tile)
                print(f"❌ Error on tile ({x_start},{y_start}): {e}")
                continue

            if tile_preds is None or len(tile_preds) == 0:
                continue

            # Filter by confidence
            tile_preds = tile_preds[tile_preds["score"] >= confidence_threshold]
            if len(tile_preds) == 0:
                continue

            # Convert local boxes to global coordinates
            for _, row in tile_preds.iterrows():
                global_box = [
                    row["xmin"] + x_start,
                    row["ymin"] + y_start,
                    row["xmax"] + x_start,
                    row["ymax"] + y_start,
                ]
                all_bboxes.append(global_box)
                all_scores.append(row["score"])

    warnings.filterwarnings("default")

    print(f"\n✅ DeepForest: {tile_count} tiles, {len(all_bboxes)} detections")

    return all_bboxes, all_scores


def segment_trees_sam(
    image,
    sam_predictor,
    bboxes,
    scores,
    tile_size=1024,
    tile_overlap=0.1,  # 10% overlap prevents edge clipping
    cache_dir=None,
    batch_size=500,
):
    """
    Pass 2: Run SAM segmentation on 1024px tiles with batched bboxes.

    Args:
        image: RGB image as (H, W, 3) numpy array (uint8)
        sam_predictor: Initialized SamPredictor
        bboxes: List of bounding boxes in global coordinates
        scores: List of confidence scores
        tile_size: Size of SAM tiles (default 1024px - SAM's native)
        tile_overlap: Overlap between tiles (default 0.1 = 10% prevents edge clipping)
        cache_dir: Directory for caching masks (temp dir if None)
        batch_size: Number of masks per cache file

    Returns:
        cache_files: List of paths to cached mask files
        final_bboxes: List of bounding boxes (post-NMS)
        final_scores: List of confidence scores (post-NMS)
    """
    print("\n🎯 Pass 2: SAM Segmentation...")

    h, w = image.shape[:2]
    stride = int(tile_size * (1 - tile_overlap))

    # Create cache directory
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="foxtrot_cache_")
        print(f"   Created temp cache: {cache_dir}")
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)
    cache_dir = Path(cache_dir)

    # Calculate tile count
    num_tiles_y = max(1, (h - tile_size) // stride + 1) if h > tile_size else 1
    num_tiles_x = max(1, (w - tile_size) // stride + 1) if w > tile_size else 1
    total_tiles = num_tiles_y * num_tiles_x

    print(f"   Tile size: {tile_size}px (SAM native)")
    print(f"   Processing {total_tiles} tiles with {len(bboxes)} boxes\n")

    # Pre-compute box centers for tile assignment
    box_centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in bboxes]

    # Track which boxes have been processed
    processed_boxes = set()
    all_masks = []
    all_processed_bboxes = []
    all_processed_scores = []
    all_tile_bounds = []  # Store tile bounds for sparse mask reconstruction
    cache_files = []

    total_boxes = len(bboxes)
    tiles_with_boxes = 0

    # Process tiles with overlap to prevent edge clipping
    # Overlap ensures trees near edges get full context
    y_starts = range(0, h, stride) if h > tile_size else [0]
    x_starts = range(0, w, stride) if w > tile_size else [0]

    overlap_pct = tile_overlap * 100
    print(f"   Tile overlap: {overlap_pct:.0f}% (prevents edge clipping)")

    for y_start in tqdm(list(y_starts), desc="   SAM tiles"):
        for x_start in x_starts:
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)

            # Find boxes whose CENTER falls within this tile
            tile_box_indices = []
            for i, (cx, cy) in enumerate(box_centers):
                if i not in processed_boxes:
                    if x_start <= cx < x_end and y_start <= cy < y_end:
                        tile_box_indices.append(i)
                        processed_boxes.add(i)

            if not tile_box_indices:
                continue

            tiles_with_boxes += 1

            # Extract tile at NATIVE 1024px (no padding/resizing)
            tile_rgb = image[y_start:y_end, x_start:x_end]

            # Progress logging
            pct = len(processed_boxes) / total_boxes * 100
            print(
                f"   Tile ({x_start},{y_start}): {len(tile_box_indices)} boxes "
                f"| {len(processed_boxes)}/{total_boxes} total ({pct:.0f}%)"
            )

            # Set SAM image (expensive - done once per tile)
            sam_predictor.set_image(tile_rgb)

            # Convert global boxes to tile-relative coordinates
            local_boxes = []
            for i in tile_box_indices:
                box = bboxes[i]
                local_box = [
                    max(0, box[0] - x_start),
                    max(0, box[1] - y_start),
                    min(tile_rgb.shape[1], box[2] - x_start),
                    min(tile_rgb.shape[0], box[3] - y_start),
                ]
                local_boxes.append(local_box)

            # Transform boxes for SAM
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                torch.as_tensor(
                    local_boxes, device=sam_predictor.device, dtype=torch.float32
                ),
                tile_rgb.shape[:2],
            )

            # Run SAM batch prediction with multiple mask options
            try:
                masks, iou_preds, _ = sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=True,  # Get 3 mask candidates per box
                )
                # masks: (N, 3, tile_H, tile_W)
                all_masks_np = masks.cpu().numpy()
            except Exception as e:
                print(f"⚠️  SAM failed on tile ({x_start}, {y_start}): {e}")
                continue

            # For each box, select the mask with best overlap with bbox
            tile_h, tile_w = tile_rgb.shape[:2]
            mask_h, mask_w = all_masks_np.shape[2], all_masks_np.shape[3]
            scale_x = mask_w / tile_w
            scale_y = mask_h / tile_h

            for mask_idx, (box_idx, local_box) in enumerate(
                zip(tile_box_indices, local_boxes)
            ):
                x1 = int(local_box[0] * scale_x)
                y1 = int(local_box[1] * scale_y)
                x2 = int(local_box[2] * scale_x)
                y2 = int(local_box[3] * scale_y)

                # Clamp to mask bounds
                y1_c, x1_c = max(0, y1), max(0, x1)
                y2_c, x2_c = min(mask_h, y2), min(mask_w, x2)

                # Select mask with highest overlap with bbox
                best_mask = None
                best_overlap = -1
                for m_idx in range(all_masks_np.shape[1]):
                    candidate = all_masks_np[mask_idx, m_idx]
                    overlap = candidate[y1_c:y2_c, x1_c:x2_c].sum()
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_mask = candidate

                if best_mask is None or best_overlap == 0:
                    continue

                # Isolate only the connected component that overlaps with bbox
                # (finetuned model predicts multiple trees, we want just the one)
                mask_uint8 = best_mask.astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(mask_uint8)

                # Find which component overlaps with bbox
                bbox_labels = labels[y1_c:y2_c, x1_c:x2_c]
                overlapping_labels = set(bbox_labels.flatten()) - {0}

                if not overlapping_labels:
                    continue

                # Keep only the largest overlapping component
                best_label = None
                best_size = 0
                for lbl in overlapping_labels:
                    size = (labels == lbl).sum()
                    if size > best_size:
                        best_size = size
                        best_label = lbl

                # Create filtered mask with only the selected component
                filtered_mask = labels == best_label

                all_masks.append(filtered_mask)
                all_processed_bboxes.append(bboxes[box_idx])
                all_processed_scores.append(scores[box_idx])
                all_tile_bounds.append((y_start, y_end, x_start, x_end, h, w))

            # Flush to disk periodically
            if len(all_masks) >= batch_size:
                batch_file = cache_dir / f"batch_{len(cache_files):04d}.npz"
                # Create object array for variable-shaped masks
                masks_arr = np.empty(len(all_masks), dtype=object)
                for i, m in enumerate(all_masks):
                    masks_arr[i] = m
                np.savez_compressed(
                    batch_file,
                    masks=masks_arr,
                    bboxes=np.array(all_processed_bboxes, dtype=np.float32),
                    bounds=np.array(all_tile_bounds, dtype=np.int32),
                )
                cache_files.append(batch_file)
                mb = batch_file.stat().st_size / (1024**2)
                print(f"   💾 Cached {len(all_masks)} masks ({mb:.1f}MB)")
                all_masks = []
                all_processed_bboxes = []
                all_tile_bounds = []

    # Flush remaining
    if len(all_masks) > 0:
        batch_file = cache_dir / f"batch_{len(cache_files):04d}.npz"
        masks_arr = np.empty(len(all_masks), dtype=object)
        for i, m in enumerate(all_masks):
            masks_arr[i] = m
        np.savez_compressed(
            batch_file,
            masks=masks_arr,
            bboxes=np.array(all_processed_bboxes, dtype=np.float32),
            bounds=np.array(all_tile_bounds, dtype=np.int32),
        )
        cache_files.append(batch_file)
        mb = batch_file.stat().st_size / (1024**2)
        print(f"   💾 Cached final {len(all_masks)} masks ({mb:.1f}MB)")

    # Results summary
    # Check for unprocessed boxes (edge cases)
    unprocessed = len(bboxes) - len(processed_boxes)
    if unprocessed > 0:
        print(f"   ⚠️  {unprocessed} boxes not processed (outside tile bounds)")

    print(f"\n✅ SAM: {len(processed_boxes)} trees segmented")
    print(f"   Results cached in {len(cache_files)} files")

    # Load bboxes for NMS
    all_bboxes_loaded = []
    for cf in cache_files:
        with np.load(cf) as data:
            all_bboxes_loaded.extend(data["bboxes"].tolist())

    # Apply NMS
    print("\n🔄 Applying NMS (IoU=0.5, coverage=0.5)...")
    keep_indices = apply_nms(
        all_bboxes_loaded,
        all_processed_scores,
        iou_threshold=0.5,
        coverage_threshold=0.5,
    )
    keep_set = set(keep_indices)

    removed = len(all_bboxes_loaded) - len(keep_indices)
    print(f"   Removed {removed} overlapping detections")
    print(f"   Keeping {len(keep_indices)} unique trees")

    # Filter results
    final_bboxes = [all_bboxes_loaded[i] for i in keep_indices]
    final_scores = [all_processed_scores[i] for i in keep_indices]

    # Stream sparse masks to new cache files (memory-efficient)
    # Build mapping from global index to position in final output
    keep_order = {idx: pos for pos, idx in enumerate(keep_indices)}

    print("   Filtering and re-caching sparse masks...")

    # Streaming approach: read and write in batches
    new_cache_files = []
    current_batch_masks = []
    current_batch_bboxes = []
    current_batch_bounds = []
    masks_written = 0

    global_idx = 0
    for cf in cache_files:
        with np.load(cf, allow_pickle=True) as data:
            masks = data["masks"]
            bounds = data["bounds"]
            for local_idx in range(len(masks)):
                if global_idx in keep_set:
                    pos = keep_order[global_idx]
                    current_batch_masks.append((pos, masks[local_idx]))
                    current_batch_bboxes.append((pos, final_bboxes[pos]))
                    current_batch_bounds.append((pos, bounds[local_idx]))

                    # Flush batch when full
                    if len(current_batch_masks) >= batch_size:
                        # Sort by position within batch
                        current_batch_masks.sort(key=lambda x: x[0])
                        current_batch_bboxes.sort(key=lambda x: x[0])
                        current_batch_bounds.sort(key=lambda x: x[0])

                        batch_file = cache_dir / f"nms_{len(new_cache_files):04d}.npz"
                        masks_arr = np.empty(len(current_batch_masks), dtype=object)
                        for i, (_, m) in enumerate(current_batch_masks):
                            masks_arr[i] = m
                        np.savez_compressed(
                            batch_file,
                            masks=masks_arr,
                            bboxes=np.array(
                                [b for _, b in current_batch_bboxes], dtype=np.float32
                            ),
                            bounds=np.array(
                                [b for _, b in current_batch_bounds], dtype=np.int32
                            ),
                        )
                        new_cache_files.append(batch_file)
                        masks_written += len(current_batch_masks)
                        print(
                            f"   💾 Saved batch {len(new_cache_files)}: "
                            f"{masks_written}/{len(keep_indices)}"
                        )
                        current_batch_masks = []
                        current_batch_bboxes = []
                        current_batch_bounds = []

                global_idx += 1

    # Flush remaining
    if current_batch_masks:
        current_batch_masks.sort(key=lambda x: x[0])
        current_batch_bboxes.sort(key=lambda x: x[0])
        current_batch_bounds.sort(key=lambda x: x[0])

        batch_file = cache_dir / f"nms_{len(new_cache_files):04d}.npz"
        masks_arr = np.empty(len(current_batch_masks), dtype=object)
        for i, (_, m) in enumerate(current_batch_masks):
            masks_arr[i] = m
        np.savez_compressed(
            batch_file,
            masks=masks_arr,
            bboxes=np.array([b for _, b in current_batch_bboxes], dtype=np.float32),
            bounds=np.array([b for _, b in current_batch_bounds], dtype=np.int32),
        )
        new_cache_files.append(batch_file)
        masks_written += len(current_batch_masks)
        print(f"   💾 Saved final batch: {masks_written}/{len(keep_indices)}")

    # Clear old cache files
    for old_file in cache_files:
        old_file.unlink(missing_ok=True)

    return new_cache_files, final_bboxes, final_scores


def load_masks_from_cache(cache_files):
    """
    Generator to load masks from cache files one at a time.
    Reconstructs full-image masks from sparse format on-demand.

    Args:
        cache_files: List of cache file paths

    Yields:
        mask: Full-size binary mask array
    """
    for cache_file in cache_files:
        # Load compressed numpy arrays
        with np.load(cache_file, allow_pickle=True) as data:
            masks = data["masks"]
            bounds = data["bounds"]
            for local_mask, bound in zip(masks, bounds):
                # Reconstruct full-size mask from sparse
                y_start, y_end, x_start, x_end, h, w = bound
                full_mask = np.zeros((h, w), dtype=bool)
                full_mask[y_start:y_end, x_start:x_end] = local_mask
                yield full_mask


def mask_to_polygon(mask, simplify_tolerance=1.0):
    """
    Convert binary mask to polygon coordinates.

    Args:
        mask: Binary mask (H, W)
        simplify_tolerance: Tolerance for polygon simplification

    Returns:
        Polygon coordinates or None if conversion fails
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Check minimum area
    area = cv2.contourArea(largest_contour)
    if area < 50:  # Minimum 50 pixels
        return None

    # Convert contour to polygon
    if len(largest_contour) < 3:
        return None

    # Reshape contour to polygon coordinates
    coords = largest_contour.reshape(-1, 2).tolist()

    # Close the polygon
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    # Create Shapely polygon and simplify
    poly = Polygon(coords)
    if simplify_tolerance > 0:
        poly = poly.simplify(simplify_tolerance, preserve_topology=True)

    return poly


def save_results(
    cache_files, bboxes, deepforest_scores, output_dir, tif_stem, crs=None
):
    """
    Save segmentation results as GeoJSON and visualization.
    Loads masks from cache to minimize memory usage.

    Args:
        cache_files: List of cache file paths containing masks
        bboxes: List of bounding boxes
        deepforest_scores: DeepForest confidence scores
        output_dir: Output directory path
        tif_stem: Stem name of input TIF file
        crs: Coordinate reference system
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert masks to GeoJSON features
    features = []

    # Load masks one at a time from cache
    mask_generator = load_masks_from_cache(cache_files)

    for i, (bbox, score) in enumerate(zip(bboxes, deepforest_scores)):
        mask = next(mask_generator)
        polygon = mask_to_polygon(mask)

        if polygon is None:
            continue

        # Create feature
        feature = {
            "type": "Feature",
            "id": i,
            "properties": {
                "tree_id": i,
                "deepforest_score": float(score),
                "area_pixels": float(polygon.area),
                "bbox": bbox,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(polygon.exterior.coords)],
            },
        }

        features.append(feature)

    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {
            "type": "name",
            "properties": {"name": str(crs) if crs else "EPSG:4326"},
        },
    }

    # Save GeoJSON
    output_geojson_path = output_dir / f"{tif_stem}_canopyai.geojson"
    print(f"\n💾 Saving GeoJSON to {output_geojson_path}...")
    with open(output_geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"✅ Saved {len(features)} tree segmentations")

    return output_geojson_path, features


def _process_mask_batch(args):
    """Worker function to extract contours from a batch of masks."""
    masks, bounds, smooth_masks = args
    results = []
    for local_mask, bound in zip(masks, bounds):
        y_start, y_end, x_start, x_end, h, w = bound
        mask_uint8 = local_mask.astype(np.uint8) * 255

        # Optional smoothing
        if smooth_masks:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Offset contours to global coordinates
        offset_contours = []
        for cnt in contours:
            offset_cnt = cnt.copy()
            offset_cnt[:, :, 0] += x_start
            offset_cnt[:, :, 1] += y_start
            offset_contours.append(offset_cnt)

        results.append(offset_contours)
    return results


def create_visualization(
    image, cache_files, bboxes, scores, output_dir, tif_stem, smooth_masks=False
):
    """
    Create visualization showing both bounding boxes and segmentation masks.
    Uses parallel processing for contour extraction.

    Args:
        image: RGB image
        cache_files: List of cache file paths containing masks
        bboxes: List of bounding boxes
        scores: List of confidence scores
        output_dir: Output directory
        tif_stem: Stem name of input TIF file
        smooth_masks: Apply morphological smoothing to noisy masks
    """
    import multiprocessing as mp
    import platform

    # Use 'spawn' on macOS to avoid fork issues (audio glitches, resource copying)
    if platform.system() == "Darwin":
        ctx = mp.get_context("spawn")
    else:
        ctx = mp.get_context("fork")

    print("\n🎨 Creating visualization...")

    vis_image = image.copy()

    # Colors
    bbox_color = (33, 240, 255)  # Electric blue for bboxes
    polygon_color = (11, 89, 214)  # Steel blue for polygons

    # Font settings for confidence labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1

    # First pass: Draw all bboxes and labels (fast)
    print("   Drawing bounding boxes...")
    for bbox, score in zip(bboxes, scores):
        xmin, ymin, xmax, ymax = [int(c) for c in bbox]
        cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), bbox_color, 1)

        # Draw confidence label above bbox
        conf_text = f"{score * 100:.0f}%"
        text_size = cv2.getTextSize(conf_text, font, font_scale, font_thickness)[0]
        text_x = xmin
        text_y = max(ymin - 4, text_size[1] + 2)

        # Background rectangle for text readability
        cv2.rectangle(
            vis_image,
            (text_x - 1, text_y - text_size[1] - 2),
            (text_x + text_size[0] + 1, text_y + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            vis_image,
            conf_text,
            (text_x, text_y),
            font,
            font_scale,
            bbox_color,
            font_thickness,
        )

    # Second pass: Extract contours in parallel from cache files.
    # Each cache file becomes a work unit (avoids full-image reconstruction)
    n_workers = mp.cpu_count()
    print(f"   Extracting contours in parallel ({n_workers} workers)...")

    work_units = []
    for cache_file in cache_files:
        with np.load(cache_file, allow_pickle=True) as data:
            masks = list(data["masks"])
            bounds = data["bounds"].tolist()
            work_units.append((masks, bounds, smooth_masks))

    # Process in parallel
    all_contours = []
    with ctx.Pool(processes=mp.cpu_count()) as pool:
        batch_results = pool.map(_process_mask_batch, work_units)
        for batch in batch_results:
            all_contours.extend(batch)

    # Third pass: Draw all contours (fast, single-threaded)
    print(f"   Drawing {len(all_contours)} mask contours...")
    for contours in all_contours:
        cv2.drawContours(vis_image, contours, -1, polygon_color, 1)

    # Save visualization
    vis_output_path = output_dir / f"{tif_stem}_canopyai_visualization.png"
    cv2.imwrite(str(vis_output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    print(f"✅ Saved visualization to {vis_output_path}")

    return vis_output_path


def main(args):
    """
    Main pipeline execution.
    """
    tif_path = Path(args.image_path)

    if not tif_path.exists():
        raise FileNotFoundError(f"❌ Image not found: {tif_path}")

    print(f"\n{'=' * 60}")
    print("🚀 Unified DeepForest + SAM Pipeline (Per-Tile)")
    print(f"{'=' * 60}")
    print(f"Input: {tif_path}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'=' * 60}\n")

    # Load image
    print("📁 Loading image...")
    image, crs, transform, bounds = load_image_from_tif(tif_path)

    # Initialize SAM first (needed by unified function)
    print(f"\n🔧 Loading SAM model from {args.sam_checkpoint}...")
    if not Path(args.sam_checkpoint).exists():
        raise FileNotFoundError(
            f"❌ SAM checkpoint not found: {args.sam_checkpoint}\n"
            "Download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"  # noqa: E501
        )

    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_checkpoint)

    # Optional: Load finetuned decoder weights
    if args.sam_finetuned_decoder:
        print(f"🎯 Loading finetuned decoder from {args.sam_finetuned_decoder}...")
        if not Path(args.sam_finetuned_decoder).exists():
            raise FileNotFoundError(
                f"❌ Finetuned decoder not found: {args.sam_finetuned_decoder}"
            )
        state = torch.load(args.sam_finetuned_decoder, map_location="cpu")
        sam.mask_decoder.load_state_dict(state["mask_decoder"])
        if "prompt_encoder" in state:
            sam.prompt_encoder.load_state_dict(state["prompt_encoder"])
        mode = state.get("mode", "unknown")
        iou = state.get("best_iou", state.get("final_iou", "?"))
        print(f"   Mode: {mode}, IoU: {iou}")

    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"🖥️  Using device: {device}")
    sam.to(device=device)

    # Compile SAM for faster inference (PyTorch 2.0+)
    try:
        sam = torch.compile(sam)
        print("⚡ SAM compiled for faster inference")
    except Exception as e:
        print(f"⚠️  torch.compile() not available: {e}")

    sam_predictor = SamPredictor(sam)

    # Stage 0: Shadow Vector Prediction (Optional)
    shadow_vector = None
    shadow_stats = None
    if args.shadow_model:
        try:
            shadow_vector, shadow_stats = predict_shadow_vector(
                image,
                args.shadow_model,
                device,
                n_crops=args.shadow_n_crops,
                outlier_threshold_deg=args.shadow_outlier_threshold,
            )
        except Exception as e:
            print(f"   ⚠️  Shadow prediction failed: {e}")
            print("   Continuing without shadow context...")
            shadow_vector = None

    # Track timing for each stage
    timings = {}
    pipeline_start = time.time()

    # Pass 1: DeepForest detection at native 400px tiles
    df_start = time.time()
    all_bboxes, all_scores = detect_trees_deepforest(
        image,
        model_path=args.deepforest_model,
        tile_size=args.df_tile_size,
        confidence_threshold=args.deepforest_confidence,
    )
    timings["DeepForest"] = time.time() - df_start

    if len(all_bboxes) == 0:
        print("\n❌ No trees detected by DeepForest. Exiting.")
        return

    # Apply NMS immediately to remove duplicates
    print(f"\n🔄 Applying NMS to {len(all_bboxes)} detections...")
    keep_indices = apply_nms(
        all_bboxes, all_scores, iou_threshold=0.5, coverage_threshold=0.5
    )
    valid_bboxes = [all_bboxes[i] for i in keep_indices]
    valid_scores = [all_scores[i] for i in keep_indices]

    removed = len(all_bboxes) - len(valid_bboxes)
    print(f"   Removed {removed} overlapping detections")
    print(f"   Keeping {len(valid_bboxes)} unique trees")

    # Debug: limit to single box
    if args.debug_single_box and len(valid_bboxes) > 0:
        print("\n⚠️  DEBUG: Limiting to single bbox")
        valid_bboxes = valid_bboxes[:1]
        valid_scores = valid_scores[:1]

    # Pass 2: SAM segmentation
    sam_start = time.time()
    cache_files, valid_bboxes, valid_scores = segment_trees_sam(
        image,
        sam_predictor,
        valid_bboxes,
        valid_scores,
        tile_size=args.sam_tile_size,
        tile_overlap=args.sam_tile_overlap,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
    timings["SAM"] = time.time() - sam_start

    if len(valid_bboxes) == 0:
        print("\n❌ No trees detected or segmented. Exiting.")
        return

    # Optional: VLM shadow filtering
    if args.vlm_shadow_filter:
        from vlm_shadow_filter import filter_shadows

        shadow_indices = filter_shadows(image, valid_bboxes, valid_scores)

        if shadow_indices:
            # Remove shadows from results
            print(f"   Removing {len(shadow_indices)} shadow detections...")
            keep_mask = np.ones(len(valid_bboxes), dtype=bool)
            keep_mask[shadow_indices] = False
            keep_indices = np.where(keep_mask)[0].tolist()

            # Filter bboxes and scores
            valid_bboxes = [valid_bboxes[i] for i in keep_indices]
            valid_scores = [valid_scores[i] for i in keep_indices]

            # Reload and filter masks from cache
            all_masks = []
            for cf in cache_files:
                with np.load(cf) as data:
                    all_masks.extend(list(data["masks"]))
            filtered_masks = [all_masks[i] for i in keep_indices]

            # Clear old cache and save filtered
            for old_file in cache_files:
                old_file.unlink(missing_ok=True)
            cache_files = []
            if filtered_masks:
                cache_dir = (
                    Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp())
                )
                batch_file = cache_dir / "vlm_filtered_batch.npz"
                np.savez_compressed(
                    batch_file,
                    masks=np.array(filtered_masks, dtype=bool),
                    bboxes=np.array(valid_bboxes, dtype=np.float32),
                )
                cache_files = [batch_file]

            print(f"   ✅ {len(valid_bboxes)} trees remaining after shadow removal")

    # Save results (loads masks from cache)
    save_start = time.time()
    output_geojson_path, features = save_results(
        cache_files, valid_bboxes, valid_scores, args.output_dir, tif_path.stem, crs
    )
    timings["Save GeoJSON"] = time.time() - save_start

    # Create visualization (loads masks from cache)
    vis_start = time.time()
    vis_path = create_visualization(
        image,
        cache_files,
        valid_bboxes,
        valid_scores,
        Path(args.output_dir),
        tif_path.stem,
        smooth_masks=args.smooth_masks,
    )
    timings["Visualization"] = time.time() - vis_start

    # Cleanup cache if using temp directory
    if args.cache_dir is None and cache_files:
        cache_dir = cache_files[0].parent
        print(f"\n🧹 Cleaning up temp cache: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Calculate total time
    total_time = time.time() - pipeline_start

    # Print summary with timings
    print(f"\n{'=' * 60}")
    print("✅ Pipeline Complete!")
    print(f"{'=' * 60}")
    print(f"Trees detected & segmented: {len(valid_bboxes)}")
    print(f"Valid features saved:       {len(features)}")
    print("\n⏱️  Timing:")
    for stage, duration in timings.items():
        if duration >= 60:
            print(f"  {stage:20s} {duration / 60:5.1f} min")
        else:
            print(f"  {stage:20s} {duration:5.1f} s")
    print(f"  {'─' * 28}")
    if total_time >= 60:
        print(f"  {'TOTAL':20s} {total_time / 60:5.1f} min")
    else:
        print(f"  {'TOTAL':20s} {total_time:5.1f} s")
    print("\nOutputs:")
    print(f"  📄 GeoJSON:      {output_geojson_path}")
    print(f"  🖼️  Visualization: {vis_path}")
    print(f"{'=' * 60}\n")


def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(
        description="Two-Stage Tree Detection: DeepForest + SAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input TIF file",
    )

    ap.add_argument(
        "--output_dir",
        type=str,
        default="foxtrot_output",
        help="Output directory for results (default: foxtrot_output)",
    )

    ap.add_argument(
        "--sam_checkpoint",
        type=str,
        default="sam_vit_b_01ec64.pth",
        help="Path to SAM checkpoint (default: sam_vit_b_01ec64.pth)",
    )

    ap.add_argument(
        "--sam_model",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model type (default: vit_b)",
    )

    ap.add_argument(
        "--sam_finetuned_decoder",
        type=str,
        default=None,
        help="Path to finetuned SAM decoder weights (.pth). "
        "If provided, loads finetuned mask decoder and prompt encoder.",
    )

    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to run inference on (default: auto)",
    )

    ap.add_argument(
        "--deepforest_confidence",
        type=float,
        default=0.35,
        help="Minimum confidence threshold for DeepForest detections (default: 0.35)",
    )

    ap.add_argument(
        "--deepforest_model",
        type=str,
        default=None,
        help="Path to custom DeepForest model (.pth file). "
        "If not provided, uses default pretrained model",
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of trees to process per batch (default: 50). "
        "Lower values use less memory but may be slower",
    )

    ap.add_argument(
        "--df_tile_size",
        type=int,
        default=400,
        help="DeepForest tile size in pixels (default: 400). "
        "DeepForest works best at 400px tiles.",
    )

    ap.add_argument(
        "--sam_tile_size",
        type=int,
        default=1024,
        help="SAM tile size in pixels (default: 1024). "
        "SAM internally uses 1024x1024, so this matches its native resolution.",
    )

    ap.add_argument(
        "--sam_tile_overlap",
        type=float,
        default=0.15,
        help="SAM tile overlap as fraction (default: 0.1 = 10%%). "
        "Overlap prevents mask clipping at tile edges.",
    )

    ap.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching masks. "
        "If not provided, uses a temporary directory that is cleaned up after",
    )

    # Stage 0: Shadow Vector Prediction
    ap.add_argument(
        "--shadow_model",
        type=str,
        default=None,
        help="Path to fine-tuned ResNet-34 shadow prediction model (.pth). "
        "If provided, predicts global shadow direction before detection. "
        "This enables future FiLM-conditioned training of oscar50.",
    )

    ap.add_argument(
        "--shadow_n_crops",
        type=int,
        default=30,
        help="Number of random crops for shadow prediction (default: 30). "
        "More crops = more robust but slower.",
    )

    ap.add_argument(
        "--shadow_outlier_threshold",
        type=float,
        default=45.0,
        help="Outlier rejection threshold in degrees (default: 45). "
        "Crops with predictions > this far from median are ignored.",
    )

    ap.add_argument(
        "--vlm_shadow_filter",
        action="store_true",
        help="Use VLM (Moondream 2) to filter shadow false positives. "
        "Requires: pip install moondream",
    )

    ap.add_argument(
        "--shadow_negative_prompts",
        action="store_true",
        help="Detect shadows and inject as negative points to SAM, "
        "excluding shadow regions from tree masks.",
    )

    ap.add_argument(
        "--smooth_masks",
        action="store_true",
        help="Apply morphological smoothing to mask contours. "
        "Reduces jagged edges from finetuned models.",
    )

    ap.add_argument(
        "--debug_single_box",
        action="store_true",
        help="Debug: limit to single bbox to inspect SAM output.",
    )

    return ap.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = parse_args()
    main(args)
