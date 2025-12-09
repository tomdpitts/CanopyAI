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
    gpu="A10G",  # Batch processing + disk caching avoids OOM
    timeout=3600,
    memory=32768,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_pipeline(
    image_bytes: bytes,
    text_prompt: str | None = None,
    deepforest_model_bytes: bytes | None = None,
    deepforest_confidence: float = 0.3,
) -> dict:
    """
    Run full DeepForest + SAM3 pipeline on Modal GPU.

    Args:
        image_bytes: TIF image as bytes
        text_prompt: Optional text prompt for SAM3 (None = bbox-only)
        deepforest_model_bytes: Optional custom DeepForest model weights
        deepforest_confidence: Confidence threshold for DeepForest

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
    from deepforest import main as deepforest_main
    from sam3.model_builder import build_sam3_image_model

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
    # 2. DeepForest detection (same patch-based approach as foxtrot.py)
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
    else:
        model.load_model("weecology/deepforest-tree")
        print("   Using pretrained model", flush=True)

    # Move model to GPU for predict_tile
    if torch.cuda.is_available():
        model.to("cuda")
        print("   Model moved to CUDA", flush=True)

    # Use predict_tile with GPU batch strategy instead of manual patch loop
    # dataloader_strategy='batch' loads entire image into GPU for parallel patch processing
    print("   Running predict_tile with GPU batch strategy...", flush=True)

    import warnings

    warnings.filterwarnings("ignore", message=".*image_path.*")
    warnings.filterwarnings("ignore", message=".*root_dir.*")

    # Save image to temp file for predict_tile (it needs a path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        import cv2

        cv2.imwrite(f.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        temp_image_path = f.name

    predictions = model.predict_tile(
        path=temp_image_path,
        patch_size=400,
        patch_overlap=0.05,
        dataloader_strategy="batch",  # GPU batch processing
    )

    # Clean up temp file
    os.unlink(temp_image_path)
    warnings.filterwarnings("default")

    if predictions is None or len(predictions) == 0:
        print("   ⚠️ No trees detected", flush=True)
        return {
            "geojson": json.dumps({"type": "FeatureCollection", "features": []}),
            "visualization": b"",
            "stats": {"detected": 0, "segmented": 0, "features": 0},
        }

    print(f"   Raw predictions: {len(predictions)}", flush=True)
    print(
        f"   Score range: [{predictions['score'].min():.3f}, {predictions['score'].max():.3f}]",
        flush=True,
    )

    # Filter by confidence
    predictions = predictions[predictions["score"] >= deepforest_confidence]
    bboxes = predictions[["xmin", "ymin", "xmax", "ymax"]].values.tolist()
    scores = predictions["score"].values.tolist()

    print(
        f"   ✅ Detected {len(bboxes)} trees (conf >= {deepforest_confidence})",
        flush=True,
    )

    # =========================================================================
    # 3. SAM3 segmentation with batch processing (using SAM1-like API)
    # =========================================================================
    print("\n🎯 Running SAM3 segmentation...", flush=True)

    # Load SAM3 model with interactive predictor enabled (SAM1-like box prompting)
    sam3_model = build_sam3_image_model(enable_inst_interactivity=True)

    # Access the interactive predictor (SAM1-like API)
    predictor = sam3_model.inst_interactive_predictor
    if predictor is None:
        raise RuntimeError("SAM3 inst_interactive_predictor not available")

    img_h, img_w = image.shape[:2]

    # Set image ONCE (expensive backbone forward pass)
    print("   Loading image into SAM3 backbone...", flush=True)
    predictor.set_image(image)  # numpy array in HWC format
    print("   Image loaded, starting segmentation...", flush=True)

    # Batch processing config
    import gc

    batch_size = 50  # Smaller batches for SAM3's higher memory usage
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
            f"📦 Batch {batch_idx + 1}/{num_batches}: Processing trees {start_idx}-{end_idx - 1}",
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
            try:
                # Box prompt in XYXY format (same as SAM1)
                xmin, ymin, xmax, ymax = bbox
                box_array = np.array([xmin, ymin, xmax, ymax])

                # Predict mask for this specific box (SAM1-like behavior)
                masks, iou_scores, low_res_masks = predictor.predict(
                    box=box_array,
                    multimask_output=False,  # Single mask for unambiguous box
                )

                # Debug: show first mask info
                if start_idx + j == 0:
                    print(f"   DEBUG: masks shape: {masks.shape}", flush=True)
                    print(f"   DEBUG: iou_scores: {iou_scores}", flush=True)

                # Take the best mask (first one since multimask_output=False)
                if masks is not None and len(masks) > 0:
                    mask = masks[0] > 0  # Convert to boolean

                    batch_masks.append(mask)
                    batch_valid_bboxes.append(bbox)
                    batch_valid_scores.append(score)

            except Exception as e:
                print(f"   ⚠️ Failed tree {start_idx + j}: {e}", flush=True)
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
    vis_image = image.copy()

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
