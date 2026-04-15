#!/usr/bin/env python3
"""
compare_models.py — Batch Evaluation Tool for Multiple CanopyAI Models

Automates running inference and evaluation (IoU & IoP) on a dataset for multiple models.

Usage:
    python compare_models.py \
        --models model1.pth model2.pth weecology/deepforest \
        --names model1 model2 weecology \
        --confidence 0.6 \
        --output_root comparison_results
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate multiple DeepForest models on the TCD dataset.")
    p.add_argument("--models", nargs="+", required=True, help="List of model paths (or 'weecology/deepforest' for standard model)")
    p.add_argument("--names", nargs="+", required=True, help="List of names corresponding to the models")
    p.add_argument("--input_dir", type=str, default="data/tcd/images/data/tcd/raw_test", help="Directory with input TIFs")
    p.add_argument("--metadata_dir", type=str, default="data/tcd/images/data/tcd/raw_test", help="Directory with TCD metadata JSONs")
    p.add_argument("--confidence", type=float, default=0.6, help="DeepForest confidence threshold")
    p.add_argument("--output_root", type=str, required=True, help="Root directory for outputs")
    p.add_argument("--shadow_model", type=str, default="solar/shadow_regression/output/shadow_model_combined_best.pth", help="Path to shadow model")
    p.add_argument("--skip_inference", action="store_true", help="Skip running foxtrot.py and go straight to evaluation (requires existing predictions)")
    return p.parse_args()

def parse_evaluation_results(eval_output):
    """Parses the text output of evaluate_batch.py to extract aggregated metrics."""
    metrics = {"trees": {}, "canopy": {}}
    
    # Regex to capture metric names and values
    metric_pattern = re.compile(r"\s+([A-Za-z0-9\s_]+)\s*:\s*([\d\.]+)")
    # Regext to capture "AGGREGATED RESULTS ({n} tiles)"
    agg_pattern = re.compile(r"AGGREGATED RESULTS\s+\((\d+)\s+tiles\)")
    
    current_category = None
    tiles = 0
    for line in eval_output.split("\n"):
        line = line.strip()
        agg_match = agg_pattern.search(line)
        if agg_match:
            tiles = int(agg_match.group(1))

        if "[trees]" in line:
            current_category = "trees"
        elif "[canopy]" in line:
            current_category = "canopy"
        elif current_category and ":" in line:
            match = metric_pattern.search(line)
            if match:
                key = match.group(1).strip()
                val = float(match.group(2))
                metrics[current_category][key] = val
    
    metrics["trees"]["tiles"] = tiles
    metrics["canopy"]["tiles"] = tiles
    return metrics

def run_inference(model_path, model_name, args, out_dir):
    input_dir = Path(args.input_dir)
    image_files = [p for p in input_dir.glob("*") if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}]
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return False

    print(f"\n🚀 Running Inference for '{model_name}' ({len(image_files)} images)...")
    success = 0
    for img_path in tqdm(image_files, desc=f"Inference ({model_name})"):
        cmd = [
            sys.executable, "foxtrot.py",
            "--image_path", str(img_path),
            "--output_dir", str(out_dir),
            "--shadow_model", args.shadow_model,
            "--deepforest_confidence", str(args.confidence),
            "--no_viz"
        ]
        
        # If it's a specific path and not 'weecology/deepforest', pass it.
        # foxtrot.py defaults to 'weecology/deepforest' if --deepforest_model is omitted.
        if model_path.lower() not in ["weecology", "weecology/deepforest", "default"]:
            cmd += ["--deepforest_model", model_path]

        try:
            # Run foxtrot to generate the _canopyai.geojson for the image
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            success += 1
        except subprocess.CalledProcessError as e:
            print(f"\n⚠️ Failed on {img_path.name}: {e.stderr}")
            pass

    return success > 0

def run_evaluation(model_name, args, out_dir):
    print(f"📊 Evaluating '{model_name}'...")
    cmd = [
        sys.executable, "evaluate_batch.py",
        "--predictions", str(out_dir),
        "--images", args.input_dir,
        "--metadata", args.metadata_dir,
        "--no_viz"
    ]
    
    try:
        # We want to show the output to the user as it happens like in the example
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        return parse_evaluation_results(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed for '{model_name}': {e.stderr}")
        return None

def display_comparison_table(results, output_root):
    # Determine the total number of unique tiles evaluated across all models
    # This assumes that the max across all evaluated models is the total number of TCD tiles
    total_tiles = 0
    for res in results.values():
        if res and "trees" in res and "tiles" in res["trees"]:
            total_tiles = max(total_tiles, res["trees"]["tiles"])

    print("\n" + "═"*70)
    print(f"  COMPARISON RESULTS  ({total_tiles} TCD tiles)")
    print("═"*70)
    
    headers = ["Model", "Precision", "Recall", "F1", "Overlap", "Tiles"]
    header_format = "  {:<17}  {:>9}  {:>7}  {:>6}  {:>8}  {:>5}"
    row_format = "  {:<17}  {:9.3f}  {:7.3f}  {:6.3f}  {:8.3f}  {:5d}"
    
    for category, display_name in [("trees", "TREE CROWN (IoU≥0.5)"), ("canopy", "CANOPY (IoP≥0.7)")]:
        print(f"\n  {display_name}")
        print(header_format.format(*headers))
        print("  " + "-"*17 + "  " + "-"*9 + "  " + "-"*7 + "  " + "-"*6 + "  " + "-"*8 + "  " + "-"*5)
        
        for name, res in results.items():
            if not res or category not in res or not res[category]:
                continue
            cat_metrics = res[category]
            
            p = cat_metrics.get("Precision", 0.0)
            r = cat_metrics.get("Recall", 0.0)
            f1 = cat_metrics.get("F1 Score", 0.0)
            ov = cat_metrics.get("Mean Overlap", 0.0)
            t = cat_metrics.get("tiles", 0)
            
            print(row_format.format(name, p, r, f1, ov, t))

    print("\n" + "═"*70 + "\n")
    
    summary_path = Path(output_root) / "comparison_summary.json"
    import json
    with open(summary_path, "w") as f:
         json.dump(results, f, indent=4)
    print(f"💾 Saved detailed results to {summary_path}")

def main():
    args = parse_args()
    
    if len(args.models) != len(args.names):
        print("❌ Error: Number of --models must match number of --names")
        sys.exit(1)
        
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for model_path, name in zip(args.models, args.names):
        out_dir = output_root / name
        out_dir.mkdir(exist_ok=True)
        
        if not args.skip_inference:
            has_preds = run_inference(model_path, name, args, out_dir)
            if not has_preds:
                print(f"⚠️ Skipping evaluation for '{name}' due to inference failure.")
                continue
                
        metrics = run_evaluation(name, args, out_dir)
        results[name] = metrics

    display_comparison_table(results, output_root)

if __name__ == "__main__":
    main()
