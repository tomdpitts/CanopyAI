"""
Modal deployment for SAM3 (Segment Anything 3) tree segmentation pipeline.

Full pipeline: DeepForest detection + SAM3 segmentation runs on Modal GPU.

Usage:
    # From local machine, run foxtrotSAM3.py which calls this remotely
    python foxtrotSAM3.py --image_path data/image.tif --text_prompt "tree canopy"

    # Or deploy and test directly
    modal run modal_foxtrotsam3.py --image-path data/test.tif
"""

import modal
import io
import json
import numpy as np

app = modal.App("canopyai-sam3-pipeline")

# Modal image with SAM3 + DeepForest dependencies
# SAM3 requires Python 3.12, PyTorch 2.7, CUDA 12.6
sam3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
        "libgdal-dev",
        "gdal-bin",
    )
    # Install PyTorch 2.7 with CUDA 12.6
    .pip_install(
        "torch==2.7.0",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    # Install SAM3 from GitHub
    .run_commands(
        "git clone https://github.com/facebookresearch/sam3.git /opt/sam3",
        "cd /opt/sam3 && pip install -e .",
    )
    # Install SAM3 core dependencies (from sam3/pyproject.toml)
    .pip_install(
        "timm>=1.0.17",
        "numpy==1.26",
        "tqdm",
        "ftfy==6.1.1",
        "regex",
        "iopath>=0.1.10",
        "typing_extensions",
        "huggingface_hub",
        # SAM3 notebooks extras (needed for inference)
        "einops",
        "decord",
        "pycocotools",
        "scikit-image",
        "scikit-learn",
    )
    # Install DeepForest and other dependencies
    .pip_install(
        "deepforest==2.0.0",
        "rasterio",
        "geopandas",
        "shapely",
        "opencv-python",
        "pandas",
        "pillow",
        "pytorch-lightning>=2.1.0,<3.0.0",
        "albumentations>=2.0.0",
    )
)


def enhance_contrast(image, factor=1.2):
    """
    Enhance image contrast by a factor.

    Args:
        image: RGB image as numpy array (H, W, C), uint8
        factor: Contrast multiplier (1.2 = 20% increase)

    Returns:
        Contrast-enhanced image as numpy array, uint8
    """
    # Convert to float for manipulation
    img_float = image.astype(np.float32)

    # Calculate mean (per-channel or global)
    mean = img_float.mean()

    # Apply contrast: (pixel - mean) * factor + mean
    enhanced = (img_float - mean) * factor + mean

    # Clip to valid range and convert back to uint8
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced


def enhance_saturation(image, boost_percent=20):
    """
    Enhance image color saturation.

    Args:
        image: RGB image as numpy array (H, W, C), uint8
        boost_percent: Percentage to boost saturation (20 = 20% increase)

    Returns:
        Saturation-enhanced image as numpy array, uint8
    """
    import cv2

    # Convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Boost saturation channel
    factor = 1.0 + (boost_percent / 100.0)
    hsv[:, :, 1] = hsv[:, :, 1] * factor

    # Clip to valid range
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    # Convert back to RGB
    hsv = hsv.astype(np.uint8)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return enhanced


def crop_bbox_with_buffer(image, bbox, buffer_ratio=0.25):
    """
    Crop image region around bbox with buffer.

    Args:
        image: Full image (H, W, C) numpy array
        bbox: [xmin, ymin, xmax, ymax] in pixel coords
        buffer_ratio: Buffer as fraction of bbox size (0.25 = 25% on each side = 150% total)

    Returns:
        crop: Cropped image patch
        crop_coords: (x1, y1, x2, y2) crop coords in original image
    """
    img_h, img_w = image.shape[:2]
    xmin, ymin, xmax, ymax = bbox

    # Calculate buffer in pixels
    bbox_w = xmax - xmin
    bbox_h = ymax - ymin
    buffer_x = bbox_w * buffer_ratio
    buffer_y = bbox_h * buffer_ratio

    # Expand bbox with buffer, clamp to image bounds
    x1 = max(0, int(xmin - buffer_x))
    y1 = max(0, int(ymin - buffer_y))
    x2 = min(img_w, int(xmax + buffer_x))
    y2 = min(img_h, int(ymax + buffer_y))

    crop = image[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2, y2)


def mask_to_polygon(mask, simplify_tolerance=1.0):
    """Convert binary mask to polygon coordinates."""
    import cv2
    from shapely.geometry import Polygon

    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < 50 or len(largest_contour) < 3:
        return None

    coords = largest_contour.reshape(-1, 2).tolist()
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    poly = Polygon(coords)
    if simplify_tolerance > 0:
        poly = poly.simplify(simplify_tolerance, preserve_topology=True)

    return poly


@app.function(
    image=sam3_image,
    gpu="A100-40GB",  # Need 40GB for DeepForest + SAM3 together
    timeout=3600,
    memory=32768,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_pipeline(
    image_bytes: bytes,
    text_prompt: str | None = None,
    deepforest_model_bytes: bytes | None = None,
    deepforest_confidence: float = 0.3,
    enhance_contrast_enabled: bool = False,
    saturation_boost: int = 0,
) -> dict:
    """
    Run full DeepForest + SAM3 pipeline on Modal GPU.

    Args:
        image_bytes: TIF image as bytes
        text_prompt: Optional text prompt for SAM3 (None = bbox-only)
        deepforest_model_bytes: Optional custom DeepForest model weights
        deepforest_confidence: Confidence threshold for DeepForest
        enhance_contrast_enabled: If True, enhance contrast by 20% before SAM3
        saturation_boost: Percentage to boost saturation (0 = disabled)

    Returns:
        dict with keys:
            - geojson: GeoJSON string
            - visualization: PNG bytes
            - stats: dict with counts
    """
    import tempfile
    import cv2
    import torch
    import rasterio
    from pathlib import Path
    import os
    from PIL import Image
    from deepforest import main as deepforest_main
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("🚀 Starting SAM3 pipeline on Modal GPU", flush=True)
    print(f"   Text prompt: {text_prompt or '(none - bbox only)'}", flush=True)

    # =========================================================================
    # 1. Load image from bytes
    # =========================================================================
    print("\n📁 Loading image...", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        f.write(image_bytes)
        tif_path = f.name

    with rasterio.open(tif_path) as src:
        image = src.read([1, 2, 3])
        image = np.transpose(image, (1, 2, 0))
        crs = str(src.crs) if src.crs else "EPSG:4326"

    # Ensure uint8 for DeepForest
    if image.dtype != np.uint8:
        if image.max() > 255:
            image = (image / image.max() * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    print(f"   ✅ Image shape: {image.shape}, dtype: {image.dtype}", flush=True)

    # =========================================================================
    # 2. DeepForest detection
    # =========================================================================
    print("\n🌲 Running DeepForest detection...", flush=True)
    model = deepforest_main.deepforest()

    if deepforest_model_bytes:
        # Load custom model from bytes
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            f.write(deepforest_model_bytes)
            model_path = f.name
        model.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("   Using custom model", flush=True)
        os.unlink(model_path)
    else:
        model.load_model("weecology/deepforest-tree")
        print("   Using pretrained model", flush=True)

    # Move model to GPU
    if torch.cuda.is_available():
        model.to("cuda")
        print("   Model moved to CUDA", flush=True)

    # Convert image to float32 for DeepForest
    if image.dtype == np.uint8:
        image_float = image.astype("float32")
    else:
        image_float = image

    h, w = image.shape[:2]
    tile_size = 400
    tile_overlap = 0.05
    stride = int(tile_size * (1 - tile_overlap))

    # Calculate tile count
    num_tiles_y = max(1, (h - tile_size) // stride + 1) if h > tile_size else 1
    num_tiles_x = max(1, (w - tile_size) // stride + 1) if w > tile_size else 1
    total_tiles = num_tiles_y * num_tiles_x

    print(f"   Image size: {w}x{h}", flush=True)
    print(
        f"   Tile size: {tile_size}px, overlap: {int(tile_overlap * 100)}%", flush=True
    )
    print(f"   Processing {total_tiles} tiles", flush=True)

    import warnings

    warnings.filterwarnings("ignore", message=".*image_path.*")
    warnings.filterwarnings("ignore", message=".*root_dir.*")

    bboxes = []
    scores = []
    tile_count = 0

    # Process tiles (manual loop matching foxtrot.py)
    y_starts = range(0, h, stride) if h > tile_size else [0]
    x_starts = range(0, w, stride) if w > tile_size else [0]

    from tqdm import tqdm

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

            # Run DeepForest prediction
            try:
                tile_preds = model.predict_image(image=tile)
            except Exception:
                continue

            if tile_preds is None or len(tile_preds) == 0:
                continue

            # Filter by confidence (per tile, like foxtrot.py)
            tile_preds = tile_preds[tile_preds["score"] >= deepforest_confidence]
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
                bboxes.append(global_box)
                scores.append(float(row["score"]))

    warnings.filterwarnings("default")

    if len(bboxes) == 0:
        print("   ⚠️ No trees detected", flush=True)
        return {
            "geojson": json.dumps({"type": "FeatureCollection", "features": []}),
            "visualization": b"",
            "stats": {"detected": 0, "segmented": 0, "features": 0},
        }

    print(
        f"\n   ✅ DeepForest: {tile_count} tiles, {len(bboxes)} detections", flush=True
    )

    # =========================================================================
    # 3. SAM3 segmentation with batch processing
    # =========================================================================
    print("\n🎯 Running SAM3 segmentation...", flush=True)

    # Load SAM3 model and create processor
    sam3_model = build_sam3_image_model()
    processor = Sam3Processor(sam3_model)

    # Optionally enhance image before segmentation
    # Keep original image for visualization
    needs_enhancement = enhance_contrast_enabled or saturation_boost > 0
    original_image = image.copy() if needs_enhancement else image

    if enhance_contrast_enabled:
        print("   🔆 Enhancing contrast by 20%...", flush=True)
        image = enhance_contrast(image, factor=1.2)

    if saturation_boost > 0:
        print(f"   🎨 Boosting saturation by {saturation_boost}%...", flush=True)
        image = enhance_saturation(image, boost_percent=saturation_boost)

    # Get image dimensions for mask mapping
    img_h, img_w = image.shape[:2]

    print("   Processing each tree crop independently...", flush=True)

    # Batch processing config
    import gc

    batch_size = 50
    cache_dir = Path(tempfile.mkdtemp(prefix="sam3_cache_"))
    print(f"   Created temp cache: {cache_dir}", flush=True)

    cache_files = []
    valid_bboxes = []
    valid_scores = []

    # Process in batches
    num_batches = (len(bboxes) + batch_size - 1) // batch_size
    print(f"   Processing {len(bboxes)} trees in {num_batches} batches\n", flush=True)

    from tqdm import tqdm

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(bboxes))
        batch_bboxes = bboxes[start_idx:end_idx]
        batch_scores_list = scores[start_idx:end_idx]

        print(
            f"📦 Batch {batch_idx + 1}/{num_batches}: trees {start_idx}-{end_idx - 1}",
            flush=True,
        )

        batch_masks = []
        batch_valid_bboxes = []
        batch_valid_scores = []

        for j, (bbox, score) in enumerate(
            tqdm(
                zip(batch_bboxes, batch_scores_list),
                total=len(batch_bboxes),
                leave=False,
            )
        ):
            tree_idx = start_idx + j
            try:
                # 1. Crop image region with 25% buffer (150% total size)
                crop, (crop_x1, crop_y1, crop_x2, crop_y2) = crop_bbox_with_buffer(
                    image, bbox, buffer_ratio=0.25
                )
                crop_h, crop_w = crop.shape[:2]

                # Skip tiny crops that might cause issues
                if crop_h < 10 or crop_w < 10:
                    print(
                        f"   SKIP tree {tree_idx}: crop too small ({crop_w}x{crop_h})",
                        flush=True,
                    )
                    continue

                # 2. Create new SAM3 processor state for cropped image
                crop_pil = Image.fromarray(crop)
                crop_state = processor.set_image(crop_pil)

                # 3. Calculate local bbox within crop (it will be centered due to buffer)
                local_xmin = bbox[0] - crop_x1
                local_ymin = bbox[1] - crop_y1
                local_xmax = bbox[2] - crop_x1
                local_ymax = bbox[3] - crop_y1

                # 4. Normalize to [cx, cy, w, h] for SAM3
                cx = (local_xmin + local_xmax) / 2 / crop_w
                cy = (local_ymin + local_ymax) / 2 / crop_h
                w = (local_xmax - local_xmin) / crop_w
                h = (local_ymax - local_ymin) / crop_h
                normalized_box = [cx, cy, w, h]

                # 5. Run SAM3 on crop with box prompt
                if text_prompt is not None:
                    state = processor.set_text_prompt(
                        prompt=text_prompt, state=crop_state
                    )
                    state = processor.add_geometric_prompt(
                        box=normalized_box, label=True, state=state
                    )
                else:
                    state = processor.add_geometric_prompt(
                        box=normalized_box, label=True, state=crop_state
                    )

                # 6. Extract mask and map back to original coords
                if "masks" in state and len(state["masks"]) > 0:
                    masks_tensor = state["masks"]
                    num_masks = len(masks_tensor)

                    # Find best mask by overlap with local bbox
                    best_mask = None
                    best_overlap = 0

                    for m_idx in range(min(num_masks, 5)):
                        m = masks_tensor[m_idx].squeeze().cpu().numpy()

                        # Resize mask to crop size if needed
                        if m.shape != (crop_h, crop_w):
                            m_pil = Image.fromarray(m.astype(np.uint8) * 255)
                            m_pil = m_pil.resize((crop_w, crop_h), Image.NEAREST)
                            m = np.array(m_pil) > 127

                        # Calculate overlap with local bbox region
                        local_bbox_mask = np.zeros((crop_h, crop_w), dtype=bool)
                        lx1, ly1 = max(0, int(local_xmin)), max(0, int(local_ymin))
                        lx2, ly2 = (
                            min(crop_w, int(local_xmax)),
                            min(crop_h, int(local_ymax)),
                        )
                        local_bbox_mask[ly1:ly2, lx1:lx2] = True

                        intersection = np.logical_and(m, local_bbox_mask).sum()
                        if intersection > best_overlap:
                            best_overlap = intersection
                            best_mask = m

                    if best_mask is not None and best_overlap > 0:
                        # Map mask back to original image coordinates
                        full_mask = np.zeros((img_h, img_w), dtype=bool)
                        full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = best_mask

                        batch_masks.append(full_mask)
                        batch_valid_bboxes.append(bbox)
                        batch_valid_scores.append(score)
                    else:
                        print(
                            f"   SKIP tree {tree_idx}: {num_masks} masks, "
                            f"best_overlap={best_overlap} pixels",
                            flush=True,
                        )
                else:
                    print(
                        f"   SKIP tree {tree_idx}: SAM3 returned 0 masks",
                        flush=True,
                    )

            except Exception as e:
                print(f"   ⚠️ Failed tree {tree_idx}: {e}", flush=True)
                continue

        # Save batch to disk
        if batch_masks:
            batch_file = cache_dir / f"batch_{batch_idx:04d}.npz"
            np.savez_compressed(
                batch_file,
                masks=np.array(batch_masks, dtype=bool),
                bboxes=np.array(batch_valid_bboxes, dtype=np.float32),
                scores=np.array(batch_valid_scores, dtype=np.float32),
            )
            cache_files.append(batch_file)
            valid_bboxes.extend(batch_valid_bboxes)
            valid_scores.extend(batch_valid_scores)

            file_size_mb = batch_file.stat().st_size / (1024**2)
            print(
                f"   ✓ Saved {len(batch_masks)} masks to {batch_file.name} ({file_size_mb:.1f}MB)",
                flush=True,
            )

        # Clear batch from memory
        del batch_masks
        del batch_valid_bboxes
        del batch_valid_scores
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"\n✅ Successfully segmented {len(valid_bboxes)} trees with SAM3", flush=True
    )
    print(f"   Cached in {len(cache_files)} batch files", flush=True)

    # =========================================================================
    # 4. Create GeoJSON (load masks from cache)
    # =========================================================================
    print("\n📄 Creating GeoJSON...")
    features = []

    # Load masks from cache files one at a time
    mask_idx = 0
    for cache_file in cache_files:
        with np.load(cache_file) as data:
            masks = data["masks"]
            cache_bboxes = data["bboxes"]
            cache_scores = data["scores"]
            for mask, bbox, score in zip(masks, cache_bboxes, cache_scores):
                polygon = mask_to_polygon(mask)
                if polygon is None:
                    mask_idx += 1
                    continue

                feature = {
                    "type": "Feature",
                    "id": mask_idx,
                    "properties": {
                        "tree_id": mask_idx,
                        "deepforest_score": float(score),
                        "area_pixels": float(polygon.area),
                        "bbox": bbox.tolist(),
                        "text_prompt": text_prompt,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [list(polygon.exterior.coords)],
                    },
                }
                features.append(feature)
                mask_idx += 1

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": crs}},
    }

    # =========================================================================
    # 5. Create visualization (load masks from cache)
    # =========================================================================
    print("\n🎨 Creating visualization...")
    vis_image = original_image.copy()

    mask_idx = 0
    for cache_file in cache_files:
        with np.load(cache_file) as data:
            masks = data["masks"]
            cache_bboxes = data["bboxes"]
            for mask, bbox in zip(masks, cache_bboxes):
                color = tuple(np.random.randint(50, 255, 3).tolist())

                # Draw bbox
                xmin, ymin, xmax, ymax = [int(c) for c in bbox]
                cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), color, 1)

                # Draw mask overlay
                mask_colored = np.zeros_like(vis_image)
                mask_colored[mask] = color
                vis_image = cv2.addWeighted(vis_image, 1, mask_colored, 0.05, 0)

                # Draw contour
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis_image, contours, -1, color, 1)
                mask_idx += 1

    # Encode as PNG
    vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    _, png_bytes = cv2.imencode(".png", vis_bgr)

    # Cleanup cache
    import shutil

    shutil.rmtree(cache_dir, ignore_errors=True)

    print("\n✅ Pipeline complete!")
    print(f"   Trees detected: {len(bboxes)}")
    print(f"   Trees segmented: {len(valid_bboxes)}")
    print(f"   Features in GeoJSON: {len(features)}")

    return {
        "geojson": json.dumps(geojson, indent=2),
        "visualization": png_bytes.tobytes(),
        "stats": {
            "detected": len(bboxes),
            "segmented": len(valid_bboxes),
            "features": len(features),
        },
    }


@app.local_entrypoint()
def main(
    image_path: str,
    text_prompt: str = None,
    deepforest_model: str = None,
    deepforest_confidence: float = 0.3,
    output_dir: str = "sam3_output",
    enhance_contrast: bool = False,
    saturation_boost: int = 0,
):
    """
    Test the SAM3 pipeline from command line.
    """
    from pathlib import Path

    print("☁️  Submitting SAM3 pipeline to Modal...")

    # Read image
    image_path = Path(image_path)
    image_bytes = image_path.read_bytes()

    # Read model if provided
    model_bytes = None
    if deepforest_model:
        model_bytes = Path(deepforest_model).read_bytes()

    # Run on Modal
    result = run_pipeline.remote(
        image_bytes=image_bytes,
        text_prompt=text_prompt if text_prompt else None,
        deepforest_model_bytes=model_bytes,
        deepforest_confidence=deepforest_confidence,
        enhance_contrast_enabled=enhance_contrast,
        saturation_boost=saturation_boost,
    )

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    # Save GeoJSON
    geojson_path = output_dir / f"{stem}_sam3.geojson"
    geojson_path.write_text(result["geojson"])
    print(f"✅ Saved GeoJSON: {geojson_path}")

    # Save visualization
    if result["visualization"]:
        vis_path = output_dir / f"{stem}_sam3_visualization.png"
        vis_path.write_bytes(result["visualization"])
        print(f"✅ Saved visualization: {vis_path}")

    # Print stats
    stats = result["stats"]
    print(f"\n📊 Results:")
    print(f"   Detected: {stats['detected']}")
    print(f"   Segmented: {stats['segmented']}")
    print(f"   Features: {stats['features']}")
