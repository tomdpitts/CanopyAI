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

Author: CanopyAI
"""

import argparse
import os
import json
import tempfile
import gc
import shutil
import numpy as np
import rasterio
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
import cv2
import torch
from tqdm import tqdm
from deepforest import main as deepforest_main
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


def apply_nms(bboxes, scores, iou_threshold=0.5, coverage_threshold=0.6):
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    Suppresses if IoU > iou_threshold OR if one box is >coverage_threshold
    covered by another (handles small boxes inside larger ones).

    Args:
        bboxes: List of bounding boxes [xmin, ymin, xmax, ymax]
        scores: List of confidence scores
        iou_threshold: IoU threshold for suppression (default 0.5)
        coverage_threshold: Coverage threshold - suppress if box is this
                           fraction covered by another (default 0.6)

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


def detect_and_segment_per_tile(
    image,
    sam_predictor,
    confidence_threshold=0.3,
    model_path=None,
    patch_size=400,
    patch_overlap=0.05,
    cache_dir=None,
    batch_size=500,
):
    """
    Unified DeepForest + SAM pipeline that processes both on the SAME tiles.
    This preserves image resolution for SAM (which downscales to 1024px internally).

    Args:
        image: RGB image as (H, W, 3) numpy array
        sam_predictor: Initialized SamPredictor
        confidence_threshold: Minimum confidence score for DeepForest detections
        model_path: Optional path to custom DeepForest model (.pth file)
        patch_size: Size of each tile (default 400px)
        patch_overlap: Overlap between tiles (default 0.05 = 5%)
        cache_dir: Directory for caching masks (temp dir if None)
        batch_size: Number of masks per cache file

    Returns:
        cache_files: List of paths to cached mask files
        all_bboxes: List of bounding boxes (global coordinates)
        all_scores: List of confidence scores
    """
    import pandas as pd
    import warnings

    print("\n🌲🎯 Running unified DeepForest + SAM per-tile pipeline...")

    # Initialize DeepForest model
    df_model = deepforest_main.deepforest()

    if model_path:
        print(f"   Loading custom DeepForest model: {model_path}")
        df_model.model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        print("   Loading default DeepForest model from Hugging Face...")
        df_model.load_model("weecology/deepforest-tree")

    # Convert image to float32 for DeepForest
    if image.dtype == np.uint8:
        image_float = image.astype("float32")
    else:
        image_float = image

    h, w = image.shape[:2]
    stride = int(patch_size * (1 - patch_overlap))

    # Create cache directory
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="foxtrot_cache_")
        print(f"   Created temp cache: {cache_dir}")
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)
    cache_dir = Path(cache_dir)

    # Calculate tile count
    num_tiles_y = max(1, (h - patch_size) // stride + 1) if h > patch_size else 1
    num_tiles_x = max(1, (w - patch_size) // stride + 1) if w > patch_size else 1
    total_tiles = num_tiles_y * num_tiles_x

    print(f"   Image size: {w}x{h}")
    print(f"   Tile size: {patch_size}px, overlap: {int(patch_overlap * 100)}%")
    print(f"   Processing {total_tiles} tiles (DeepForest → SAM per tile)\n")

    # Suppress DeepForest warnings
    warnings.filterwarnings("ignore", message=".*image_path.*")
    warnings.filterwarnings("ignore", message=".*root_dir.*")

    # Accumulate results across tiles
    all_masks = []
    all_bboxes = []
    all_scores = []
    cache_files = []
    tile_count = 0
    tree_count = 0

    # Process tiles
    y_starts = range(0, h, stride) if h > patch_size else [0]
    x_starts = range(0, w, stride) if w > patch_size else [0]

    for y_start in tqdm(list(y_starts), desc="   Tile rows"):
        for x_start in x_starts:
            y_end = min(y_start + patch_size, h)
            x_end = min(x_start + patch_size, w)

            # Extract tile
            tile = image_float[y_start:y_end, x_start:x_end]
            tile_rgb = image[y_start:y_end, x_start:x_end]  # uint8 for SAM

            # Skip very small edge tiles
            if tile.shape[0] < 200 or tile.shape[1] < 200:
                continue

            tile_count += 1

            # === STAGE 1: DeepForest on this tile ===
            try:
                tile_preds = df_model.predict_image(image=tile)
            except Exception:
                continue

            if tile_preds is None or len(tile_preds) == 0:
                continue

            # Filter by confidence
            tile_preds = tile_preds[tile_preds["score"] >= confidence_threshold]
            if len(tile_preds) == 0:
                continue

            # === STAGE 2: SAM on this tile with detected boxes ===
            # Set SAM image to this tile (preserve full resolution)
            sam_predictor.set_image(tile_rgb)

            # Get local bounding boxes (relative to tile)
            local_boxes = tile_preds[["xmin", "ymin", "xmax", "ymax"]].values
            tile_scores = tile_preds["score"].values.tolist()

            # Transform boxes for SAM
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                torch.as_tensor(
                    local_boxes, device=sam_predictor.device, dtype=torch.float32
                ),
                tile_rgb.shape[:2],
            )

            # Run SAM prediction on tile
            try:
                masks, _, _ = sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                # masks: (N, 1, tile_H, tile_W) -> (N, tile_H, tile_W)
                tile_masks = masks.cpu().numpy().squeeze(1)
            except Exception as e:
                print(f"⚠️  SAM failed on tile ({x_start}, {y_start}): {e}")
                continue

            # Convert local masks to global masks and store results
            for i, (local_mask, local_box, score) in enumerate(
                zip(tile_masks, local_boxes, tile_scores)
            ):
                # Create full-size mask
                full_mask = np.zeros((h, w), dtype=bool)
                full_mask[y_start:y_end, x_start:x_end] = local_mask

                # Convert local box to global coordinates
                global_box = [
                    local_box[0] + x_start,
                    local_box[1] + y_start,
                    local_box[2] + x_start,
                    local_box[3] + y_start,
                ]

                all_masks.append(full_mask)
                all_bboxes.append(global_box)
                all_scores.append(score)
                tree_count += 1

            # Flush to disk periodically to avoid OOM
            if len(all_masks) >= batch_size:
                batch_file = cache_dir / f"batch_{len(cache_files):04d}.npz"
                np.savez_compressed(
                    batch_file,
                    masks=np.array(all_masks, dtype=bool),
                    bboxes=np.array(all_bboxes, dtype=np.float32),
                )
                cache_files.append(batch_file)
                file_size_mb = batch_file.stat().st_size / (1024**2)
                print(f"   💾 Cached {len(all_masks)} masks ({file_size_mb:.1f}MB)")
                all_masks = []
                all_bboxes_for_cache = all_bboxes.copy()
                all_bboxes = []  # Clear for next batch (but keep scores)

    # Flush remaining masks
    if len(all_masks) > 0:
        batch_file = cache_dir / f"batch_{len(cache_files):04d}.npz"
        np.savez_compressed(
            batch_file,
            masks=np.array(all_masks, dtype=bool),
            bboxes=np.array(all_bboxes, dtype=np.float32),
        )
        cache_files.append(batch_file)
        file_size_mb = batch_file.stat().st_size / (1024**2)
        print(f"   💾 Cached final {len(all_masks)} masks ({file_size_mb:.1f}MB)")

    # Re-enable warnings
    warnings.filterwarnings("default")

    print(f"\n✅ Processed {tile_count} tiles, found {tree_count} trees (before NMS)")
    print(f"   Results cached in {len(cache_files)} files")

    # Load all results from cache for NMS
    all_masks_loaded = []
    all_bboxes_loaded = []
    for cf in cache_files:
        with np.load(cf) as data:
            all_masks_loaded.extend(list(data["masks"]))
            all_bboxes_loaded.extend(data["bboxes"].tolist())

    # Apply NMS to remove overlapping detections from tile boundaries
    print(f"\n🔄 Applying NMS (IoU threshold=0.5)...")
    keep_indices = apply_nms(all_bboxes_loaded, all_scores, iou_threshold=0.5)

    # Filter results
    final_masks = [all_masks_loaded[i] for i in keep_indices]
    final_bboxes = [all_bboxes_loaded[i] for i in keep_indices]
    final_scores = [all_scores[i] for i in keep_indices]

    removed_count = len(all_bboxes_loaded) - len(keep_indices)
    print(f"   Removed {removed_count} overlapping detections")
    print(f"   Keeping {len(keep_indices)} unique trees")

    # Clear old cache files and save filtered results
    for old_file in cache_files:
        old_file.unlink(missing_ok=True)

    cache_files = []
    if len(final_masks) > 0:
        # Save all filtered masks in batches
        for batch_start in range(0, len(final_masks), batch_size):
            batch_end = min(batch_start + batch_size, len(final_masks))
            batch_file = cache_dir / f"nms_batch_{len(cache_files):04d}.npz"
            np.savez_compressed(
                batch_file,
                masks=np.array(final_masks[batch_start:batch_end], dtype=bool),
                bboxes=np.array(final_bboxes[batch_start:batch_end], dtype=np.float32),
            )
            cache_files.append(batch_file)

    return cache_files, final_bboxes, final_scores


def load_masks_from_cache(cache_files):
    """
    Generator to load masks from cache files one at a time.
    Reduces memory footprint by loading masks on-demand.

    Args:
        cache_files: List of cache file paths

    Yields:
        mask: Binary mask array
    """
    for cache_file in cache_files:
        # Load compressed numpy arrays
        with np.load(cache_file) as data:
            masks = data["masks"]
            for mask in masks:
                yield mask


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
    output_geojson_path = output_dir / f"{tif_stem}_deepforest_sam.geojson"
    print(f"\n💾 Saving GeoJSON to {output_geojson_path}...")
    with open(output_geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"✅ Saved {len(features)} tree segmentations")

    return output_geojson_path, features


def create_visualization(image, cache_files, bboxes, output_dir, tif_stem):
    """
    Create visualization showing both bounding boxes and segmentation masks.
    Loads masks from cache to minimize memory usage.

    Args:
        image: RGB image
        cache_files: List of cache file paths containing masks
        bboxes: List of bounding boxes
        output_dir: Output directory
        tif_stem: Stem name of input TIF file
    """
    print("\n🎨 Creating visualization...")

    vis_image = image.copy()

    # Load masks one at a time from cache
    mask_generator = load_masks_from_cache(cache_files)

    # Soft magenta color for all trees (RGB)
    color = (200, 100, 180)

    # Draw each detection
    for i, bbox in enumerate(bboxes):
        mask = next(mask_generator)

        # Draw bounding box (from DeepForest)
        xmin, ymin, xmax, ymax = [int(c) for c in bbox]
        cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), color, 1)

        # Draw segmentation mask (from SAM)
        # Create colored mask overlay
        mask_colored = np.zeros_like(vis_image)
        mask_colored[mask] = color

        # Blend mask with image (alpha=0 means no fill, only contours)
        alpha = 0
        vis_image = cv2.addWeighted(vis_image, 1, mask_colored, alpha, 0)

        # Draw mask contour
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_image, contours, -1, color, 1)

    # Save visualization
    vis_output_path = output_dir / f"{tif_stem}_deepforest_sam_visualization.png"
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
    sam_predictor = SamPredictor(sam)

    # Run unified DeepForest + SAM per-tile pipeline
    cache_files, valid_bboxes, valid_scores = detect_and_segment_per_tile(
        image,
        sam_predictor,
        confidence_threshold=args.deepforest_confidence,
        model_path=args.deepforest_model,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )

    if len(valid_bboxes) == 0:
        print("\n❌ No trees detected or segmented. Exiting.")
        return

    # Save results (loads masks from cache)
    output_geojson_path, features = save_results(
        cache_files, valid_bboxes, valid_scores, args.output_dir, tif_path.stem, crs
    )

    # Create visualization (loads masks from cache)
    vis_path = create_visualization(
        image, cache_files, valid_bboxes, Path(args.output_dir), tif_path.stem
    )

    # Cleanup cache if using temp directory
    if args.cache_dir is None and cache_files:
        cache_dir = cache_files[0].parent
        print(f"\n🧹 Cleaning up temp cache: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Print summary
    print(f"\n{'=' * 60}")
    print("✅ Pipeline Complete!")
    print(f"{'=' * 60}")
    print(f"Trees detected & segmented: {len(valid_bboxes)}")
    print(f"Valid features saved:       {len(features)}")
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to run inference on (default: auto)",
    )

    ap.add_argument(
        "--deepforest_confidence",
        type=float,
        default=0.3,
        help="Minimum confidence threshold for DeepForest detections (default: 0.3)",
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
        default=500,
        help="Number of trees to process per batch (default: 500). "
        "Lower values use less memory but may be slower",
    )

    ap.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching masks. "
        "If not provided, uses a temporary directory that is cleaned up after",
    )

    return ap.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = parse_args()
    main(args)
