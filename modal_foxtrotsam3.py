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
    gpu="A10G",
    timeout=3600,
    memory=32768,
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
    from PIL import Image
    from deepforest import main as deepforest_main
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("🚀 Starting SAM3 pipeline on Modal GPU")
    print(f"   Text prompt: {text_prompt or '(none - bbox only)'}")

    # =========================================================================
    # 1. Load image from bytes
    # =========================================================================
    print("\n📁 Loading image...")
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

    print(f"   ✅ Image shape: {image.shape}, dtype: {image.dtype}")

    # =========================================================================
    # 2. DeepForest detection
    # =========================================================================
    print("\n🌲 Running DeepForest detection...")
    model = deepforest_main.deepforest()

    if deepforest_model_bytes:
        # Load custom model from bytes
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            f.write(deepforest_model_bytes)
            model_path = f.name
        model.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("   Using custom model")
    else:
        model.load_model("weecology/deepforest-tree")
        print("   Using pretrained model")

    # Predict - DeepForest 2.0 expects uint8 for predict_tile, float32 for predict_image
    if image.shape[0] > 2000 or image.shape[1] > 2000:
        print("   Large image, using patch-based prediction...")
        predictions = model.predict_tile(
            image=image,  # uint8 for predict_tile
            patch_size=400,
            patch_overlap=0.05,
        )
    else:
        predictions = model.predict_image(image=image.astype("float32"))

    if predictions is None or len(predictions) == 0:
        print("   ⚠️ No trees detected")
        return {
            "geojson": json.dumps({"type": "FeatureCollection", "features": []}),
            "visualization": b"",
            "stats": {"detected": 0, "segmented": 0},
        }

    # Filter by confidence
    predictions = predictions[predictions["score"] >= deepforest_confidence]
    bboxes = predictions[["xmin", "ymin", "xmax", "ymax"]].values.tolist()
    scores = predictions["score"].values.tolist()

    print(f"   ✅ Detected {len(bboxes)} trees")

    # =========================================================================
    # 3. SAM3 segmentation
    # =========================================================================
    print("\n🎯 Running SAM3 segmentation...")

    # Load SAM3 model
    sam3_model = build_sam3_image_model()
    processor = Sam3Processor(sam3_model)

    # Convert numpy to PIL for SAM3
    pil_image = Image.fromarray(image)
    inference_state = processor.set_image(pil_image)

    all_masks = []
    valid_bboxes = []
    valid_scores = []

    from tqdm import tqdm

    for i, (bbox, score) in enumerate(tqdm(zip(bboxes, scores), total=len(bboxes))):
        try:
            if text_prompt is not None:
                # Use text prompt
                output = processor.set_text_prompt(
                    state=inference_state,
                    prompt=text_prompt,
                )
                masks = output["masks"]
                # Find mask that best overlaps with bbox
                if len(masks) > 0:
                    # Take the first mask for now (could refine with IoU)
                    mask = (
                        masks[0].cpu().numpy() if hasattr(masks[0], "cpu") else masks[0]
                    )
                else:
                    continue
            else:
                # Use bbox-only (like SAM1)
                bbox_array = np.array(bbox)
                output = processor.set_box_prompt(
                    state=inference_state,
                    box=bbox_array,
                )
                masks = output["masks"]
                if len(masks) > 0:
                    mask = (
                        masks[0].cpu().numpy() if hasattr(masks[0], "cpu") else masks[0]
                    )
                else:
                    continue

            all_masks.append(mask)
            valid_bboxes.append(bbox)
            valid_scores.append(score)

        except Exception as e:
            print(f"   ⚠️ Failed to segment tree {i}: {e}")
            continue

    print(f"   ✅ Segmented {len(all_masks)} trees")

    # =========================================================================
    # 4. Create GeoJSON
    # =========================================================================
    print("\n📄 Creating GeoJSON...")
    features = []

    for i, (mask, bbox, score) in enumerate(zip(all_masks, valid_bboxes, valid_scores)):
        polygon = mask_to_polygon(mask)
        if polygon is None:
            continue

        feature = {
            "type": "Feature",
            "id": i,
            "properties": {
                "tree_id": i,
                "deepforest_score": float(score),
                "area_pixels": float(polygon.area),
                "bbox": bbox,
                "text_prompt": text_prompt,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(polygon.exterior.coords)],
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": crs}},
    }

    # =========================================================================
    # 5. Create visualization
    # =========================================================================
    print("\n🎨 Creating visualization...")
    vis_image = image.copy()

    for i, (mask, bbox) in enumerate(zip(all_masks, valid_bboxes)):
        color = tuple(np.random.randint(50, 255, 3).tolist())

        # Draw bbox
        xmin, ymin, xmax, ymax = [int(c) for c in bbox]
        cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), color, 1)

        # Draw mask overlay
        mask_colored = np.zeros_like(vis_image)
        mask_colored[mask] = color
        vis_image = cv2.addWeighted(vis_image, 1, mask_colored, 0.1, 0)

        # Draw contour
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_image, contours, -1, color, 1)

    # Encode as PNG
    vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    _, png_bytes = cv2.imencode(".png", vis_bgr)

    print("\n✅ Pipeline complete!")
    print(f"   Trees detected: {len(bboxes)}")
    print(f"   Trees segmented: {len(all_masks)}")
    print(f"   Features in GeoJSON: {len(features)}")

    return {
        "geojson": json.dumps(geojson, indent=2),
        "visualization": png_bytes.tobytes(),
        "stats": {
            "detected": len(bboxes),
            "segmented": len(all_masks),
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
