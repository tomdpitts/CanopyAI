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


def detect_trees_with_deepforest(image, confidence_threshold=0.3, model_path=None):
    """
    Stage 1: Use DeepForest to detect trees and generate bounding boxes.

    Args:
        image: RGB image as (H, W, 3) numpy array
        confidence_threshold: Minimum confidence score for detections
        model_path: Optional path to custom DeepForest model (.pth file)

    Returns:
        bboxes: List of bounding boxes as [xmin, ymin, xmax, ymax]
        scores: Confidence scores for each detection
    """
    print("\n🌲 Stage 1: Running DeepForest tree detection...")

    # Initialize DeepForest model
    model = deepforest_main.deepforest()

    # Load model
    if model_path:
        print(f"   Loading custom model: {model_path}")
        # Load custom fine-tuned model
        model.model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        print("   Loading default pretrained model from Hugging Face...")
        # Load the pretrained model from Hugging Face
        model.load_model("weecology/deepforest-tree")

    # Convert image to float32 to avoid warning and ensure proper format
    if image.dtype == np.uint8:
        image_float = image.astype("float32")
        print(f"   Converted image from uint8 to float32")
    else:
        image_float = image

    print(f"   Image shape: {image_float.shape}")
    print(f"   Image value range: [{image_float.min():.1f}, {image_float.max():.1f}]")

    # DeepForest expects RGB format for numpy array predictions
    # For very large images, we might need to process in patches
    h, w = image_float.shape[:2]

    # If image is very large (>2000px), use predict_tile which handles patches
    if h > 2000 or w > 2000:
        print(f"   Large image detected, using patch-based prediction...")

        # Use larger patches and less overlap for faster processing
        patch_size = 400  # Larger patches = fewer patches
        patch_overlap = 0.05  # Less overlap = faster
        stride = int(patch_size * (1 - patch_overlap))

        # Calculate number of patches
        num_patches_y = (h - patch_size) // stride + 1
        num_patches_x = (w - patch_size) // stride + 1
        total_patches = num_patches_y * num_patches_x

        print(
            f"   Processing {total_patches} patches ({patch_size}px, {int(patch_overlap * 100)}% overlap)..."
        )

        # Suppress DeepForest warnings about missing image_path
        import warnings

        warnings.filterwarnings("ignore", message=".*image_path.*")
        warnings.filterwarnings("ignore", message=".*root_dir.*")

        all_predictions = []
        patch_count = 0

        for y_start in tqdm(range(0, h, stride), desc="   Rows"):
            for x_start in range(0, w, stride):
                y_end = min(y_start + patch_size, h)
                x_end = min(x_start + patch_size, w)

                # Extract patch
                patch = image_float[y_start:y_end, x_start:x_end]

                # Skip very small edge patches
                if patch.shape[0] < 200 or patch.shape[1] < 200:
                    continue

                # Predict on patch
                try:
                    patch_preds = model.predict_image(image=patch)

                    if patch_preds is not None and len(patch_preds) > 0:
                        # Adjust bounding box coordinates to global image
                        patch_preds["xmin"] += x_start
                        patch_preds["xmax"] += x_start
                        patch_preds["ymin"] += y_start
                        patch_preds["ymax"] += y_start
                        all_predictions.append(patch_preds)
                        patch_count += len(patch_preds)
                except Exception:
                    # Silently skip failed patches
                    continue

        # Re-enable warnings
        warnings.filterwarnings("default")

        print(f"   Found {patch_count} raw detections across all patches")

        # Combine all predictions
        if all_predictions:
            import pandas as pd

            predictions = pd.concat(all_predictions, ignore_index=True)
        else:
            predictions = None
    else:
        predictions = model.predict_image(image=image_float)

    print(f"   Raw predictions: {len(predictions) if predictions is not None else 0}")

    if predictions is None or len(predictions) == 0:
        print("⚠️  No trees detected by DeepForest")
        print(
            f"   Try lowering --deepforest_confidence (current: {confidence_threshold})"
        )
        return [], []

    # Show score distribution before filtering
    print(
        f"   Score range: [{predictions['score'].min():.3f}, {predictions['score'].max():.3f}]"
    )

    # Filter by confidence threshold
    predictions = predictions[predictions["score"] >= confidence_threshold]

    print(
        f"✅ DeepForest detected {len(predictions)} trees (conf >= {confidence_threshold})"
    )

    # Extract bounding boxes and scores
    bboxes = predictions[["xmin", "ymin", "xmax", "ymax"]].values.tolist()
    scores = predictions["score"].values.tolist()

    return bboxes, scores


def segment_trees_with_sam(
    image, bboxes, sam_predictor, batch_size=500, cache_dir=None
):
    """
    Stage 2: Use SAM to segment trees within each bounding box.
    Uses batch processing with disk caching to avoid OOM errors.

    Args:
        image: RGB image as (H, W, 3) numpy array
        bboxes: List of bounding boxes as [xmin, ymin, xmax, ymax]
        sam_predictor: Initialized SAM predictor
        batch_size: Number of trees to process per batch before flushing to disk
        cache_dir: Directory for caching masks (temp dir if None)

    Returns:
        cache_files: List of paths to cached mask files
        valid_bboxes: Corresponding bounding boxes
    """
    print("\n🎯 Stage 2: Running SAM segmentation on detected trees...")
    print(f"   Using batch size: {batch_size}")

    # Create cache directory
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="foxtrot_cache_")
        print(f"   Created temp cache: {cache_dir}")
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)
        print(f"   Using cache: {cache_dir}")

    cache_dir = Path(cache_dir)

    # Set image for SAM predictor
    sam_predictor.set_image(image)

    cache_files = []
    valid_bboxes = []

    # Process in batches
    num_batches = (len(bboxes) + batch_size - 1) // batch_size
    print(f"   Processing {len(bboxes)} trees in {num_batches} batches\n")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(bboxes))
        batch_bboxes = bboxes[start_idx:end_idx]

        print(
            f"📦 Batch {batch_idx + 1}/{num_batches}: Processing trees {start_idx}-{end_idx - 1}"
        )

        batch_masks = []
        batch_valid_bboxes = []

        # Process this batch
        for bbox in tqdm(batch_bboxes, desc=f"  Segmenting", leave=False):
            try:
                # Convert bbox to SAM format [x, y, x2, y2]
                bbox_array = np.array(bbox)

                # Predict mask using bounding box prompt
                masks, scores, logits = sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=bbox_array[None, :],  # SAM expects (1, 4) shape
                    multimask_output=False,  # Single mask per box
                )

                # Take the first (and only) mask
                mask = masks[0]
                batch_masks.append(mask)
                batch_valid_bboxes.append(bbox)

            except Exception as e:
                print(f"⚠️  Failed to segment tree: {e}")
                continue

        # Save batch to disk using numpy compression (~95% size reduction)
        batch_file = cache_dir / f"batch_{batch_idx:04d}.npz"
        np.savez_compressed(
            batch_file,
            masks=np.array(batch_masks, dtype=bool),
            bboxes=np.array(batch_valid_bboxes, dtype=np.float32),
        )

        cache_files.append(batch_file)
        valid_bboxes.extend(batch_valid_bboxes)

        # Report file size
        file_size_mb = batch_file.stat().st_size / (1024**2)
        print(
            f"   ✓ Saved {len(batch_masks)} masks to {batch_file.name} ({file_size_mb:.1f}MB)"
        )

        # Clear batch from memory
        del batch_masks
        del batch_valid_bboxes
        gc.collect()

    print(f"\n✅ Successfully segmented {len(valid_bboxes)} trees with SAM")
    print(f"   Cached in {len(cache_files)} batch files")

    return cache_files, valid_bboxes


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

    # Draw each detection
    for i, bbox in enumerate(bboxes):
        mask = next(mask_generator)

        # Generate random color for this tree
        color = tuple(np.random.randint(50, 255, 3).tolist())

        # Draw bounding box (from DeepForest)
        xmin, ymin, xmax, ymax = [int(c) for c in bbox]
        cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), color, 1)

        # Draw segmentation mask (from SAM)
        # Create colored mask overlay
        mask_colored = np.zeros_like(vis_image)
        mask_colored[mask] = color

        # Blend mask with image
        alpha = 0.1
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
    print("🚀 Two-Stage Tree Detection: DeepForest + SAM")
    print(f"{'=' * 60}")
    print(f"Input: {tif_path}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'=' * 60}\n")

    # Load image
    print("📁 Loading image...")
    image, crs, transform, bounds = load_image_from_tif(tif_path)

    # Stage 1: DeepForest detection
    bboxes, deepforest_scores = detect_trees_with_deepforest(
        image,
        confidence_threshold=args.deepforest_confidence,
        model_path=args.deepforest_model,
    )

    if len(bboxes) == 0:
        print("\n❌ No trees detected. Exiting.")
        return

    # Initialize SAM
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

    # Stage 2: SAM segmentation with batch processing
    cache_files, valid_bboxes = segment_trees_with_sam(
        image,
        bboxes,
        sam_predictor,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

    if len(valid_bboxes) == 0:
        print("\n❌ No trees successfully segmented. Exiting.")
        return

    # Filter scores to match valid bboxes
    valid_scores = deepforest_scores[: len(valid_bboxes)]

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
    print(f"Trees detected (DeepForest): {len(bboxes)}")
    print(f"Trees segmented (SAM):       {len(valid_bboxes)}")
    print(f"Valid features saved:        {len(features)}")
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
