#!/usr/bin/env python3

"""
SAM Tree Segmentation Script
----------------------------
Runs SAM (Segment Anything Model) inference on a single TIF file to segment trees.
Similar to sam.py but designed for single large TIF inference.
"""

import argparse
import os
import json
import numpy as np
import rasterio
from pathlib import Path
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import torch
from tqdm import tqdm


# Fix MPS float64 compatibility
torch.set_default_dtype(torch.float32)


def get_polygon_from_mask(mask):
    """Convert binary mask to list of polygons (COCO format)."""
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        # Flatten and convert to list
        poly = contour.flatten().tolist()
        if len(poly) > 4:  # at least 3 points (6 coords)
            polygons.append(poly)
    return polygons


def mask_to_geojson_feature(mask, area_threshold=100):
    """
    Convert a binary mask to a GeoJSON feature.

    Args:
        mask: Binary mask (H, W)
        area_threshold: Minimum area in pixels to keep

    Returns:
        GeoJSON feature dict or None if mask is too small
    """
    # Convert mask to contours
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Check area threshold
    area = cv2.contourArea(largest_contour)
    if area < area_threshold:
        return None

    # Convert contour to polygon
    if len(largest_contour) < 3:
        return None

    # Reshape contour to polygon coordinates
    coords = largest_contour.reshape(-1, 2).tolist()
    # Close the polygon
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    # Create GeoJSON feature
    feature = {
        "type": "Feature",
        "properties": {"area": float(area), "class": "tree"},
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }

    return feature


def process_tile(image_tile, mask_generator, tile_offset_x, tile_offset_y):
    """Process a single tile and return features with adjusted coordinates."""
    masks = mask_generator.generate(image_tile)

    features = []
    for i, mask_dict in enumerate(masks):
        mask = mask_dict["segmentation"]
        feature = mask_to_geojson_feature(mask, area_threshold=50)

        if feature is not None:
            # Adjust coordinates for tile offset
            coords = feature["geometry"]["coordinates"][0]
            adjusted_coords = [
                [x + tile_offset_x, y + tile_offset_y] for x, y in coords
            ]
            feature["geometry"]["coordinates"] = [adjusted_coords]

            # Add properties
            feature["properties"]["predicted_iou"] = float(
                mask_dict.get("predicted_iou", 0)
            )
            feature["properties"]["stability_score"] = float(
                mask_dict.get("stability_score", 0)
            )
            features.append(feature)

    return features


def process_tif_with_sam(
    tif_path, output_dir, checkpoint_path, device="cuda", tile_size=1024, overlap=256
):
    """
    Process a single TIF file with SAM to segment trees.
    Uses tiling for large images to avoid memory issues.

    Args:
        tif_path: Path to input TIF file
        output_dir: Directory to save outputs
        checkpoint_path: Path to SAM checkpoint
        device: Device to run inference on ('cuda', 'cpu', or 'mps')
        tile_size: Size of tiles to process
        overlap: Overlap between tiles
    """
    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading image from {tif_path}...")

    # Read TIF with rasterio to preserve georeferencing
    with rasterio.open(tif_path) as src:
        # Read RGB bands (assuming bands 1,2,3 are RGB)
        image = src.read([1, 2, 3])  # Shape: (3, H, W)
        image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, 3)

        # Store georeferencing info
        crs = src.crs
        bounds = src.bounds

        print(f"  Image shape: {image.shape}")
        print(f"  CRS: {crs}")
        print(f"  Bounds: {bounds}")

    # Normalize image to 0-255 if needed
    if image.max() > 255:
        image = (image / image.max() * 255).astype(np.uint8)

    # Initialize SAM
    print(f"Loading SAM model from {checkpoint_path}...")
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=device)

    # Use automatic mask generator with less aggressive settings
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,  # Reduced for faster processing
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,  # Disable cropping
        min_mask_region_area=50,
    )

    # Calculate tile grid
    h, w = image.shape[:2]
    stride = tile_size - overlap

    print(
        f"Processing image in {tile_size}x{tile_size} tiles with {overlap}px overlap..."
    )

    all_features = []

    # Process tiles
    for y in tqdm(range(0, h, stride), desc="Processing rows"):
        for x in range(0, w, stride):
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)

            tile = image[y:y_end, x:x_end]

            # Skip very small tiles at edges
            if tile.shape[0] < 256 or tile.shape[1] < 256:
                continue

            # Process tile
            try:
                tile_features = process_tile(tile, mask_generator, x, y)
                all_features.extend(tile_features)
            except Exception as e:
                print(f"Error processing tile at ({x},{y}): {e}")
                continue

    print(f"Generated {len(all_features)} features from all tiles")

    # Add IDs to features
    for i, feature in enumerate(all_features):
        feature["properties"]["id"] = i

    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": all_features,
        "crs": {
            "type": "name",
            "properties": {"name": str(crs) if crs else "EPSG:4326"},
        },
    }

    # Save GeoJSON output
    stem = tif_path.stem
    output_geojson_path = output_dir / f"{stem}_sam_trees.geojson"
    print(f"Saving GeoJSON to {output_geojson_path}...")
    with open(output_geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)

    # Create visualization
    print("Creating visualization...")
    vis_image = image.copy()

    # Draw masks on image
    for feature in tqdm(all_features, desc="Drawing features"):
        coords = np.array(feature["geometry"]["coordinates"][0], dtype=np.int32)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.polylines(vis_image, [coords], True, color, 2)

        # Fill with semi-transparent color
        overlay = vis_image.copy()
        cv2.fillPoly(overlay, [coords], color)
        cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)

    # Save visualization
    vis_output_path = output_dir / f"{stem}_sam_visualization.png"
    cv2.imwrite(str(vis_output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    print("✅ Processing complete!")
    print(f"  - GeoJSON: {output_geojson_path}")
    print(f"  - Visualization: {vis_output_path}")
    print(f"  - Total trees detected: {len(all_features)}")


def visualize_geojson(tif_path, geojson_path, output_dir):
    """
    Create visualization from existing GeoJSON file.

    Args:
        tif_path: Path to input TIF file
        geojson_path: Path to existing GeoJSON file
        output_dir: Directory to save visualization
    """
    tif_path = Path(tif_path)
    geojson_path = Path(geojson_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading image from {tif_path}...")

    # Read TIF
    with rasterio.open(tif_path) as src:
        image = src.read([1, 2, 3])
        image = np.transpose(image, (1, 2, 0))

    # Normalize image to 0-255 if needed
    if image.max() > 255:
        image = (image / image.max() * 255).astype(np.uint8)

    # Load GeoJSON
    print(f"Loading GeoJSON from {geojson_path}...")
    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    features = geojson["features"]
    print(f"Loaded {len(features)} features")

    # Create visualization
    print("Creating visualization...")
    vis_image = image.copy()

    # Draw masks on image
    for feature in tqdm(features, desc="Drawing features"):
        coords = np.array(feature["geometry"]["coordinates"][0], dtype=np.int32)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.polylines(vis_image, [coords], True, color, 2)

        # Fill with semi-transparent color
        overlay = vis_image.copy()
        cv2.fillPoly(overlay, [coords], color)
        cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)

    # Save visualization
    stem = tif_path.stem
    vis_output_path = output_dir / f"{stem}_sam_visualization.png"
    cv2.imwrite(str(vis_output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    print("✅ Visualization complete!")
    print(f"  - Output: {vis_output_path}")
    print(f"  - Total trees visualized: {len(features)}")


def parse_args():
    ap = argparse.ArgumentParser(description="SAM Tree Segmentation")
    ap.add_argument(
        "--tif_path",
        default="data/tcd/bin_liang/tcd_tile_WON.tif",
        help="Path to input TIF file",
    )
    ap.add_argument(
        "--output_dir", default="sam_output", help="Output directory for results"
    )
    ap.add_argument(
        "--checkpoint", default="sam_vit_b_01ec64.pth", help="Path to SAM checkpoint"
    )
    ap.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to run inference on",
    )
    ap.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only create visualization from existing GeoJSON (skip SAM inference)",
    )
    ap.add_argument(
        "--geojson-path",
        help="Path to existing GeoJSON file (required with --visualize-only)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = parse_args()

    if args.visualize_only:
        # Visualization-only mode
        if not args.geojson_path:
            # Auto-detect GeoJSON path
            tif_stem = Path(args.tif_path).stem
            geojson_path = Path(args.output_dir) / f"{tif_stem}_sam_trees.geojson"
            if not geojson_path.exists():
                print(f"Error: GeoJSON file not found at {geojson_path}")
                print("Please specify --geojson-path or run full inference first")
                exit(1)
        else:
            geojson_path = args.geojson_path

        visualize_geojson(args.tif_path, geojson_path, args.output_dir)
    else:
        # Full SAM inference mode
        # Auto-detect device if cuda not available
        if args.device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                device = "mps"
                print("CUDA not available, using MPS")
            else:
                device = "cpu"
                print("CUDA not available, using CPU")
        else:
            device = args.device

        process_tif_with_sam(
            args.tif_path, args.output_dir, args.checkpoint, device=device
        )
