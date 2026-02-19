#!/usr/bin/env python3
"""
Batch Inference Script for Foxtrot
----------------------------------
Runs foxtrot.py on all images in a specified directory.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Run foxtrot.py batch inference")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="data/tcd/images/data/tcd/raw_test",
        help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="zebraBRU/runT/TCD",
        help="Directory for output results"
    )
    parser.add_argument(
        "--deepforest_model", 
        type=str, 
        default="phase3_runT_film_aggressive_lr.pth",
        help="Path to DeepForest model"
    )
    parser.add_argument(
        "--shadow_model", 
        type=str, 
        default="solar/shadow_regression/output/shadow_model_combined_best.pth",
        help="Path to shadow model"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Print commands without executing"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
        
    # Find images
    image_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    images = [
        p for p in input_dir.glob("*") 
        if p.suffix.lower() in image_extensions
    ]
    
    if not images:
        print(f"❌ No images found in {input_dir}")
        sys.exit(1)
        
    print(f"🚀 Found {len(images)} images in {input_dir}")
    print(f"   Output: {args.output_dir}")
    print(f"   Model:  {args.deepforest_model}")
    print(f"   Shadow: {args.shadow_model}")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    
    total_images = len(images)
    for i, img_path in enumerate(images, 1):
        print(f"\nProcessing image {i}/{total_images}: {img_path.name}")
        cmd = [
            sys.executable, "foxtrot.py",
            "--image_path", str(img_path),
            "--output_dir", args.output_dir,
            "--deepforest_model", args.deepforest_model,
            "--shadow_model", args.shadow_model
        ]
        
        if args.dry_run:
            print(f"\n[DRY RUN] {' '.join(cmd)}")
            success_count += 1
            continue
            
        try:
            # Run subprocess
            # We capture output to avoid spamming the console too much, 
            # but usually it's better to let it stream if the user wants to see progress.
            # foxtrot.py seems to have its own progress bars (tqdm).
            # To avoid conflicting progress bars, we might want to capture output or not use tqdm here.
            # Let's just run it.
            result = subprocess.run(cmd, check=True, text=True)
            success_count += 1
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to process {img_path.name}")
            print(f"   Error: {e}")
            fail_count += 1
            
    print("-" * 60)
    print(f"✅ Batch processing complete")
    print(f"   Success: {success_count}")
    print(f"   Failed:  {fail_count}")

if __name__ == "__main__":
    main()
