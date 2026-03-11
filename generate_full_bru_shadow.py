#!/usr/bin/env python3
"""
Generate the shadow map for the ENTIRE BRU162 orthomosaic using "Saved 1" presets.
This provides a complete overview of the current shadow extraction parameters.
"""
import sys
import os
import json
import cv2
import rasterio
import numpy as np
from pathlib import Path

# Add shadow_tuner to path so we can import its compute_shadow function
script_dir = Path(__file__).parent.resolve()
shadow_tuner_dir = script_dir / "shadow_tuner"
sys.path.insert(0, str(shadow_tuner_dir))

from app import compute_shadow, _valid_mask, _dg, _sm_vectors

def main():
    presets_file = shadow_tuner_dir / "presets.json"
    if not presets_file.exists():
        print(f"Error: {presets_file} not found.")
        sys.exit(1)
        
    with open(presets_file, "r") as f:
        presets = json.load(f)
        
    if "3" not in presets:
        print("Error: Slot '3' not found in presets.")
        sys.exit(1)
        
    p1 = presets["3"]
    p = {k: float(v) for k, v in p1.items()}
    print(f"Loaded preset 3 parameters: {p}")
    
    img_path = script_dir / "input_data/BRU162/BRU162_10cm.tif"
    print(f"Loading full image {img_path}...")
    with rasterio.open(img_path) as src:
        full_image = src.read([1, 2, 3]).transpose(1, 2, 0)
        profile = src.profile
        
    print(f"Image shape: {full_image.shape}")
    
    print("Calculating autonomous image metrics (Norms & Luma)...")
    # Calculate Abs_Luma_Max based on the 6th percentile
    gray_full = cv2.cvtColor(full_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    valid_full = _valid_mask(gray_full)
    valid_pixels = gray_full[valid_full > 0]
    
    abs_luma = np.percentile(valid_pixels, 10)
    print(f"Dynamically Calculated abs_luma_max (10th %ile): {abs_luma:.1f}")
    
    # Calculate DG Norm (99.5th percentile of the full raytrace vector)
    adx, ady, sdx, sdy = _sm_vectors(p['angle'])
    dg_full = _dg(gray_full, valid_full, adx, ady, sdx, sdy, 
                  p['d_short_min'], p['d_short_max'], 8, p['blur_short'])
    mask_full = valid_full > 0
    norm_dg = float(np.percentile(dg_full[mask_full], 99.5)) if mask_full.sum() > 0 else 30.0
    print(f"Dynamically Calculated norm_dg (99.5th %ile): {norm_dg:.2f}")

    del gray_full, valid_full, valid_pixels, dg_full, mask_full
    
    print("\nComputing shadow map for the entire orthomosaic... (This might take a minute)")
    shadow_prob = compute_shadow(
        full_image, 
        angle_deg=p['angle'], 
        d_short_min=p['d_short_min'], d_short_max=p['d_short_max'], blur_short=p['blur_short'],
        d_long_min=p['d_long_min'],   d_long_max=p['d_long_max'],   blur_long=p['blur_long'], 
        short_weight=p['short_weight'],
        abs_luma_max=abs_luma,
        norm_dg=norm_dg,
        otsu_ctr=p['otsu_ctr'], sigmoid_k=p['sigmoid_k'],
        speckle_min_area=p['speckle_min'], activation_floor=p['act_floor'], min_aspect_ratio=p.get('min_ratio', 0.0)
    )
    
    shadow_u8 = (shadow_prob * 255).astype(np.uint8)
    
    # Save as GeoTIFF so the user can overlap it perfectly over the original in QGIS
    out_path = script_dir / "BRU162_shadow_map_saved3.tif"
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='deflate'
    )
    
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(shadow_u8, 1)
        
    print(f"✅ Full shadow map saved to {out_path}")
    
    # Also save a standard PNG for easy immediate viewing
    out_png_path = script_dir / "BRU162_shadow_map_saved3.png"
    cv2.imwrite(str(out_png_path), shadow_u8)
    print(f"✅ PNG version saved to {out_png_path}")

if __name__ == "__main__":
    main()
