#!/usr/bin/env python3
"""
Pre-predict shadow vectors for TCD orthomosaics and generate visualizations.

This script:
1. Loads TCD orthomosaic images
2. Predicts shadow vectors using the trained ResNet-34 model
3. Generates arrow overlay visualizations for manual review
4. Saves predictions to CSV for later use in training

Usage:
    python deepforest/predict_tcd_shadows.py \
        --tcd_dir data/tcd/images \
        --shadow_model solar/shadow_regression/output/shadow_model_combined_best.pth \
        --output_csv deepforest/tcd_shadow_predictions.csv \
        --viz_dir deepforest/tcd_shadow_viz
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import rasterio
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet34
from tqdm import tqdm


# Inline model definition to avoid import dependencies
class ShadowResNet34(nn.Module):
    """ResNet-34 for shadow vector regression."""

    def __init__(self):
        super().__init__()
        # Load pretrained ResNet-34
        self.backbone = resnet34(weights="IMAGENET1K_V1")

        # Replace final FC layer to output 2D vector
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        # Forward through ResNet backbone
        out = self.backbone(x)

        # Normalize to unit vector
        out = out / (torch.norm(out, dim=1, keepdim=True) + 1e-8)
        return out


def load_shadow_model(model_path, device):
    """Load trained shadow prediction model."""
    model = ShadowResNet34()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_shadow_vector(image_path, model, device, n_crops=30):
    """
    Predict shadow vector for an orthomosaic image.

    Returns:
        shadow_vector: (2,) numpy array - unit vector
        angle_deg: Shadow direction in degrees
        stats: Dict with prediction statistics
    """
    # Load image
    with rasterio.open(image_path) as src:
        # Read RGB bands
        image = np.stack([src.read(i) for i in range(1, 4)], axis=-1)

    h, w = image.shape[:2]

    # Extract random crops (avoid edges)
    crop_size = 500
    margin_fraction = 0.2
    margin_w = int(w * margin_fraction)
    margin_h = int(h * margin_fraction)

    safe_min_col = margin_w
    safe_max_col = max(margin_w, w - margin_w - crop_size)
    safe_min_row = margin_h
    safe_max_row = max(margin_h, h - margin_h - crop_size)

    crops = []
    for _ in range(n_crops):
        if safe_max_col <= safe_min_col or safe_max_row <= safe_min_row:
            break

        col_off = np.random.randint(safe_min_col, safe_max_col)
        row_off = np.random.randint(safe_min_row, safe_max_row)

        crop = image[row_off : row_off + crop_size, col_off : col_off + crop_size]
        crops.append(Image.fromarray(crop))

    # Predict on each crop
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    vectors = []
    with torch.no_grad():
        for crop in crops:
            img_tensor = preprocess(crop).unsqueeze(0).to(device)
            pred = model(img_tensor).cpu().squeeze().numpy()
            vectors.append(pred)

    vectors = np.array(vectors)  # (N, 2)

    # Compute circular mean with outlier rejection
    angles = np.arctan2(vectors[:, 0], vectors[:, 1])
    angles_deg = np.degrees(angles) % 360

    # Find circular median
    sin_mean = np.mean(np.sin(np.radians(angles_deg)))
    cos_mean = np.mean(np.cos(np.radians(angles_deg)))
    median_angle = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360

    # Outlier rejection (45° threshold)
    def circular_distance(a1, a2):
        diff = np.abs(a1 - a2)
        return np.minimum(diff, 360 - diff)

    distances = circular_distance(angles_deg, median_angle)
    inlier_mask = distances <= 45.0
    n_inliers = inlier_mask.sum()
    n_outliers = (~inlier_mask).sum()

    # Compute mean of inliers
    inlier_angles_rad = angles[inlier_mask]
    mean_sin = np.mean(np.sin(inlier_angles_rad))
    mean_cos = np.mean(np.cos(inlier_angles_rad))
    mean_angle_rad = np.arctan2(mean_sin, mean_cos)

    # Convert to unit vector
    shadow_vector = np.array([np.sin(mean_angle_rad), np.cos(mean_angle_rad)])
    angle_deg = np.degrees(mean_angle_rad) % 360

    # Circular standard deviation
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    circular_std = np.degrees(np.sqrt(-2 * np.log(R))) if R > 0 else 180.0

    stats = {
        "n_crops": len(crops),
        "n_inliers": n_inliers,
        "n_outliers": n_outliers,
        "consensus_pct": (n_inliers / len(crops)) * 100,
        "circular_std_deg": circular_std,
        "angle_deg": angle_deg,
    }

    return shadow_vector, angle_deg, stats


def create_visualization(image_path, shadow_angle_deg, output_path, thumbnail_size=800):
    """
    Create visualization with arrow overlay showing predicted shadow direction.

    Args:
        image_path: Path to orthomosaic TIF
        shadow_angle_deg: Predicted shadow direction in degrees
        output_path: Where to save visualization
        thumbnail_size: Max dimension for thumbnail
    """
    # Load image
    with rasterio.open(image_path) as src:
        image = np.stack([src.read(i) for i in range(1, 4)], axis=-1)

    # Create thumbnail
    h, w = image.shape[:2]
    scale = thumbnail_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    thumbnail = cv2.resize(image, (new_w, new_h))

    # Convert to BGR for OpenCV
    thumbnail_bgr = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)

    # Draw arrow from center
    center = (new_w // 2, new_h // 2)
    arrow_length = min(new_w, new_h) // 4

    # Convert angle to radians (note: image y-axis points down)
    angle_rad = np.radians(shadow_angle_deg)
    dx = int(arrow_length * np.sin(angle_rad))
    dy = int(arrow_length * np.cos(angle_rad))

    end_point = (center[0] + dx, center[1] + dy)

    # Draw arrow
    cv2.arrowedLine(
        thumbnail_bgr,
        center,
        end_point,
        color=(0, 255, 255),  # Yellow in BGR
        thickness=4,
        tipLength=0.3,
    )

    # Add text with angle
    text = f"Shadow: {shadow_angle_deg:.1f}°"
    cv2.putText(
        thumbnail_bgr,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), thumbnail_bgr)


def main():
    parser = argparse.ArgumentParser(
        description="Predict shadow vectors for TCD orthomosaics"
    )
    parser.add_argument(
        "--tcd_dir",
        type=str,
        required=True,
        help="Directory containing TCD orthomosaic TIF files",
    )
    parser.add_argument(
        "--shadow_model",
        type=str,
        required=True,
        help="Path to trained ResNet-34 shadow model",
    )
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Output CSV with predictions"
    )
    parser.add_argument(
        "--viz_dir", type=str, required=True, help="Directory to save visualizations"
    )
    parser.add_argument(
        "--n_crops",
        type=int,
        default=30,
        help="Number of crops to sample per image (default: 30)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
    )

    args = parser.parse_args()

    # Setup device
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
    print(f"📁 TCD directory: {args.tcd_dir}")
    print(f"🌞 Shadow model: {args.shadow_model}")
    print(f"📊 Crops per image: {args.n_crops}")
    print()

    # Load model
    print("Loading shadow prediction model...")
    model = load_shadow_model(args.shadow_model, device)
    print("✅ Model loaded")
    print()

    # Find TCD images
    tcd_dir = Path(args.tcd_dir)
    tif_files = list(tcd_dir.glob("*.tif")) + list(tcd_dir.glob("*.tiff"))

    if not tif_files:
        print(f"❌ No TIF files found in {tcd_dir}")
        return

    print(f"Found {len(tif_files)} TCD orthomosaics")
    print()

    # Process each image
    results = []
    viz_dir = Path(args.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    for tif_path in tqdm(tif_files, desc="Processing images"):
        print(f"\n📸 {tif_path.name}")

        # Predict shadow vector
        try:
            shadow_vector, angle_deg, stats = predict_shadow_vector(
                tif_path, model, device, n_crops=args.n_crops
            )

            print(f"   Shadow: {angle_deg:.1f}°")
            print(
                f"   Consensus: {stats['n_inliers']}/{stats['n_crops']} "
                f"({stats['consensus_pct']:.0f}%)"
            )
            print(f"   Circular std: {stats['circular_std_deg']:.1f}°")

            # Create visualization
            viz_path = viz_dir / f"{tif_path.stem}_shadow_viz.png"
            create_visualization(tif_path, angle_deg, viz_path)
            print(f"   ✅ Visualization saved: {viz_path}")

            # Store results
            results.append(
                {
                    "image_name": tif_path.name,
                    "image_path": str(tif_path.absolute()),
                    "shadow_x": shadow_vector[0],
                    "shadow_y": shadow_vector[1],
                    "shadow_angle_deg": angle_deg,
                    "n_crops": stats["n_crops"],
                    "n_inliers": stats["n_inliers"],
                    "n_outliers": stats["n_outliers"],
                    "consensus_pct": stats["consensus_pct"],
                    "circular_std_deg": stats["circular_std_deg"],
                }
            )

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback

            traceback.print_exc()

    # Save CSV
    if results:
        df = pd.DataFrame(results)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\n✅ Saved predictions to {args.output_csv}")
        print(f"✅ Visualizations in {args.viz_dir}")
        print(f"\n📊 Summary:")
        print(f"   Total images: {len(results)}")
        print(f"   Mean consensus: {df['consensus_pct'].mean():.1f}%")
        print(f"   Mean circular std: {df['circular_std_deg'].mean():.1f}°")
    else:
        print("\n❌ No results to save")


if __name__ == "__main__":
    main()
