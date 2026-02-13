#!/usr/bin/env python3

"""
Two-Stage Tree Detection Pipeline: DeepForest + SAM3 (Modal Cloud)
-------------------------------------------------------------------
This script runs the full detection + segmentation pipeline on Modal GPU.

SAM3 supports text prompts for open-vocabulary segmentation.

Usage:
    # Without text prompt (bbox-only, like SAM1)
    python foxtrotSAM3.py --image_path data/image.tif --output_dir output

    # With text prompt
    python foxtrotSAM3.py --image_path data/image.tif --text_prompt "tree canopy"

Author: CanopyAI
"""

import argparse
import modal
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(
        description="Two-Stage Tree Detection: DeepForest + SAM3 (Modal)",
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
        default="foxtrot_sam3_output",
        help="Output directory for results (default: foxtrot_sam3_output)",
    )

    ap.add_argument(
        "--text_prompt",
        type=str,
        default=None,
        help="Text prompt for SAM3 segmentation (default: None = bbox-only)",
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
        "--enhance_contrast",
        action="store_true",
        help="Enhance image contrast by 20%% before SAM3 segmentation",
    )

    ap.add_argument(
        "--saturation_boost",
        type=int,
        default=0,
        help="Percentage to boost color saturation (e.g., 20 for 20%% increase)",
    )

    ap.add_argument(
        "--shadow_negative_prompts",
        action="store_true",
        help="Detect shadows and inject as negative points to exclude from masks",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    tif_path = Path(args.image_path)
    output_dir = Path(args.output_dir)

    if not tif_path.exists():
        raise FileNotFoundError(f"❌ Image not found: {tif_path}")

    print(f"\n{'=' * 60}")
    print("🚀 Two-Stage Tree Detection: DeepForest + SAM3")
    print(f"{'=' * 60}")
    print(f"Input: {tif_path}")
    print(f"Output: {output_dir}")
    print(f"Text prompt: {args.text_prompt or '(none - bbox only)'}")
    print(f"DeepForest confidence: {args.deepforest_confidence}")
    print(f"{'=' * 60}\n")

    # Read image bytes
    print("📁 Reading image...")
    image_bytes = tif_path.read_bytes()
    print(f"   Size: {len(image_bytes) / (1024 * 1024):.1f} MB")

    # Read custom model if provided
    model_bytes = None
    if args.deepforest_model:
        model_path = Path(args.deepforest_model)
        if not model_path.exists():
            raise FileNotFoundError(f"❌ Model not found: {model_path}")
        print(f"📦 Reading custom model: {model_path}")
        model_bytes = model_path.read_bytes()

    print("\n☁️  Connecting to Modal...")
    try:
        run_pipeline = modal.Function.from_name(
            "canopyai-sam3-pipeline", "run_pipeline"
        )
    except modal.exception.NotFoundError:
        print("❌ Modal app not deployed. Deploy first with:")
        print("   modal deploy modal_foxtrotsam3.py")
        return

    print("🔄 Running pipeline on Modal GPU (this may take a few minutes)...")

    # Enable log streaming from Modal
    with modal.enable_output():
        result = run_pipeline.remote(
            image_bytes=image_bytes,
            text_prompt=args.text_prompt,
            deepforest_model_bytes=model_bytes,
            deepforest_confidence=args.deepforest_confidence,
            enhance_contrast_enabled=args.enhance_contrast,
            saturation_boost=args.saturation_boost,
            shadow_negative_prompts=args.shadow_negative_prompts,
        )

    # Save results locally
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = tif_path.stem

    # Save GeoJSON
    geojson_path = output_dir / f"{stem}_deepforest_sam3.geojson"
    geojson_path.write_text(result["geojson"])
    print(f"\n💾 Saved GeoJSON: {geojson_path}")

    # Save visualization
    if result["visualization"]:
        vis_path = output_dir / f"{stem}_deepforest_sam3_visualization.png"
        vis_path.write_bytes(result["visualization"])
        print(f"🖼️  Saved visualization: {vis_path}")

    # Print summary
    stats = result["stats"]
    print(f"\n{'=' * 60}")
    print("✅ Pipeline Complete!")
    print(f"{'=' * 60}")
    print(f"Trees detected (DeepForest): {stats['detected']}")
    print(f"Trees segmented (SAM3):      {stats['segmented']}")
    print(f"Valid features saved:        {stats['features']}")
    print("\nOutputs:")
    print(f"  📄 GeoJSON:      {geojson_path}")
    if result["visualization"]:
        print(f"  🖼️  Visualization: {vis_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
