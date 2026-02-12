#!/usr/bin/env python3
"""
Test model on clean 500x500 tiles from unseen orthomosaics.
Extracts tiles from center regions, predicts on CLEAN tiles, then visualizes.
"""

import os
import math
import random
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import rasterio
from rasterio.windows import Window
from train_combined import ShadowResNet34
from torchvision import transforms


def extract_center_tiles(tif_path, tile_size=500, n_tiles=3, margin_fraction=0.2):
    """
    Extract random tiles from center region of TIF (avoiding blank edges).
    Returns list of PIL Images.
    """
    tiles = []
    tile_positions = []

    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width

        # Define safe extraction zone (center, avoiding margins)
        margin_w = int(width * margin_fraction)
        margin_h = int(height * margin_fraction)

        safe_min_col = margin_w
        safe_max_col = width - margin_w - tile_size
        safe_min_row = margin_h
        safe_max_row = height - margin_h - tile_size

        if safe_max_col <= safe_min_col or safe_max_row <= safe_min_row:
            print(f"  Warning: Image too small for safe extraction zone")
            safe_min_col = tile_size
            safe_max_col = width - tile_size
            safe_min_row = tile_size
            safe_max_row = height - tile_size

        # Extract tiles
        for _ in range(n_tiles):
            col_off = random.randint(safe_min_col, safe_max_col)
            row_off = random.randint(safe_min_row, safe_max_row)

            window = Window(col_off, row_off, tile_size, tile_size)
            tile_data = src.read(window=window)

            # Convert to RGB PIL image
            if tile_data.shape[0] >= 3:
                rgb = tile_data[:3].transpose(1, 2, 0)
            else:
                rgb = np.stack([tile_data[0]] * 3, axis=-1)

            # Normalize to 0-255
            if rgb.max() > 255:
                rgb = (rgb / rgb.max() * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

            pil_img = Image.fromarray(rgb)
            tiles.append(pil_img)
            tile_positions.append((col_off, row_off))

    return tiles, tile_positions


def draw_prediction_arrow(draw, cx, cy, azimuth, color, length=100, label=""):
    """Draw arrow for shadow direction."""
    theta_rad = math.radians(azimuth)
    end_x = cx + length * math.sin(theta_rad)
    end_y = cy - length * math.cos(theta_rad)

    # Main line
    draw.line([(cx, cy), (end_x, end_y)], fill=color, width=5)

    # Arrowhead
    head_length = 20
    angle_offset = math.pi / 6
    draw.polygon(
        [
            (end_x, end_y),
            (
                end_x - head_length * math.sin(theta_rad + angle_offset),
                end_y + head_length * math.cos(theta_rad + angle_offset),
            ),
            (
                end_x - head_length * math.sin(theta_rad - angle_offset),
                end_y + head_length * math.cos(theta_rad - angle_offset),
            ),
        ],
        fill=color,
    )

    # Label
    if label:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except:
            font = ImageFont.load_default()

        label_x = end_x + 12
        label_y = end_y - 12
        draw.text((label_x, label_y), label, fill=color, font=font)


def test_on_clean_tiles(
    model_path, input_data_dir, output_dir, tiles_per_ortho=3, device="cpu"
):
    """Extract clean tiles and predict."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {model_path}")
    model = ShadowResNet34()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded successfully\n")

    # Preprocessing (resize to 224 for model input)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Find unseen orthomosaics
    input_path = Path(input_data_dir)
    tif_files = [f for f in input_path.glob("*/*.tif") if "WON003" not in str(f)]

    print(f"Found {len(tif_files)} unseen orthomosaics:")
    for tif in tif_files:
        print(f"  - {tif.parent.name}: {tif.name}")

    print(f"\nExtracting {tiles_per_ortho} clean 500x500 tiles per orthomosaic...\n")

    sample_idx = 0
    all_predictions = []

    for tif_path in tif_files:
        ortho_name = tif_path.parent.name
        print(f"Processing {ortho_name}...")

        try:
            # Extract CLEAN tiles (no modifications)
            tiles, positions = extract_center_tiles(
                str(tif_path), tile_size=500, n_tiles=tiles_per_ortho
            )
        except Exception as e:
            print(f"  Error extracting tiles: {e}")
            continue

        for i, tile_img in enumerate(tiles):
            # PREDICT ON CLEAN TILE (before any visualization)
            img_tensor = preprocess(tile_img).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(img_tensor).cpu().squeeze()

            pred_angle = math.degrees(math.atan2(pred[0], pred[1]))
            pred_angle = (pred_angle + 360) % 360

            all_predictions.append(pred_angle)

            # NOW create visualization (separate from prediction)
            canvas_width = 500 + 400
            canvas = Image.new("RGB", (canvas_width, 500), color=(30, 30, 30))
            canvas.paste(tile_img, (0, 0))

            draw = ImageDraw.Draw(canvas)

            # Draw prediction arrow on original tile
            cx, cy = 250, 250
            draw_prediction_arrow(
                draw, cx, cy, pred_angle, color=(255, 50, 50), label=""
            )

            # Center marker
            draw.ellipse(
                [(cx - 8, cy - 8), (cx + 8, cy + 8)],
                fill=(255, 255, 0),
                outline=(0, 0, 0),
                width=2,
            )

            # Info panel
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
                font_small = ImageFont.truetype(
                    "/System/Library/Fonts/Helvetica.ttc", 16
                )
            except:
                font = ImageFont.load_default()
                font_small = font

            text_x = 520
            y_offset = 60

            draw.text(
                (text_x, y_offset), "Unseen Orthomosaic", fill=(76, 175, 80), font=font
            )
            y_offset += 50

            draw.text(
                (text_x, y_offset),
                f"Site: {ortho_name}",
                fill=(220, 220, 220),
                font=font_small,
            )
            y_offset += 35

            draw.text(
                (text_x, y_offset),
                f"Tile: {i + 1}/{tiles_per_ortho}",
                fill=(180, 180, 180),
                font=font_small,
            )
            y_offset += 35

            draw.text(
                (text_x, y_offset),
                f"Position: ({positions[i][0]}, {positions[i][1]})",
                fill=(150, 150, 150),
                font=font_small,
            )
            y_offset += 50

            draw.text(
                (text_x, y_offset),
                "Predicted Shadow:",
                fill=(200, 200, 200),
                font=font_small,
            )
            y_offset += 30
            draw.text(
                (text_x, y_offset), f"{pred_angle:.1f}°", fill=(255, 50, 50), font=font
            )
            y_offset += 60

            draw.text(
                (text_x, y_offset), "Model:", fill=(150, 150, 150), font=font_small
            )
            y_offset += 25
            draw.text(
                (text_x, y_offset),
                "shadow_model_combined_best.pth",
                fill=(150, 150, 150),
                font=font_small,
            )
            y_offset += 20
            draw.text(
                (text_x, y_offset),
                "(WON003 trained)",
                fill=(150, 150, 150),
                font=font_small,
            )
            y_offset += 50

            draw.text(
                (text_x, y_offset), "Legend:", fill=(180, 180, 180), font=font_small
            )
            y_offset += 25
            draw.text(
                (text_x, y_offset),
                "Red arrow = Prediction",
                fill=(255, 50, 50),
                font=font_small,
            )
            y_offset += 22
            draw.text(
                (text_x, y_offset),
                "Yellow dot = Center",
                fill=(255, 255, 0),
                font=font_small,
            )

            # Save
            output_path = os.path.join(
                output_dir, f"clean_{sample_idx:02d}_{ortho_name}_tile{i + 1}.png"
            )
            canvas.save(output_path)

            print(
                f"  [{i + 1}/{tiles_per_ortho}] Predicted: {pred_angle:6.1f}° | Saved: {Path(output_path).name}"
            )
            sample_idx += 1

    # Summary
    if all_predictions:
        pred_array = np.array(all_predictions)
        print(f"\n{'=' * 60}")
        print(f"SUMMARY: {len(all_predictions)} predictions")
        print(f"  Mean: {pred_array.mean():.1f}°")
        print(f"  Std:  {pred_array.std():.1f}°")
        print(f"  Min:  {pred_array.min():.1f}°")
        print(f"  Max:  {pred_array.max():.1f}°")
        print(f"{'=' * 60}")

    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    test_on_clean_tiles(
        model_path="output/shadow_model_combined_best.pth",
        input_data_dir="../../input_data",
        output_dir="output/predictions_clean_unseen",
        tiles_per_ortho=10,
        device=device,
    )
