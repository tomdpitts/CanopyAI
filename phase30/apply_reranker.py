#!/usr/bin/env python3
"""
apply_reranker.py — Apply a pre-trained CNN reranker to an existing folder
of foxtrot prediction geojsons, writing rescored geojsons to a new folder.

Distinct from `phase30/cnn_reranker.py` (which trains + applies in one go)
and from `foxtrot.py --reranker_checkpoint` (which reranks during fresh
inference).  Use this when you already have a geojson folder and just
want to swap in a different reranker — e.g. comparing two checkpoints
on the same prediction set, or rescoring SAM-H predictions with a
reranker trained on SAM-B predictions.

Usage:
    python phase30/apply_reranker.py \\
        --src benchmark_results_holdout/<predictions> \\
        --dst benchmark_results_holdout/<rescored> \\
        --image-dir data/tcd/images/data/tcd/val \\
        --checkpoint phase30/cnn_reranker_ens3.pt
"""
import argparse, json, sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from shapely.geometry import mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_reranker import load_ensemble


def _load_image(tif_path):
    with rasterio.open(tif_path) as src:
        arr = src.read([1, 2, 3])
    image = np.transpose(arr, (1, 2, 0))
    if image.dtype != np.uint8:
        mx = max(1, image.max())
        image = (image.astype(np.float32) / mx * 255).astype(np.uint8)
    return image


def _rewrite_geojson(gdf, new_scores, out_path):
    feats = []
    for idx, geom in enumerate(gdf.geometry):
        props = {k: v for k, v in gdf.iloc[idx].items() if k != "geometry"}
        clean = {}
        for k, v in props.items():
            if k == "deepforest_score":
                clean[k] = float(new_scores[idx]); continue
            if hasattr(v, "tolist"): clean[k] = v.tolist()
            elif isinstance(v, (int, float, str, bool)) or v is None: clean[k] = v
            else: clean[k] = str(v)
        feats.append({"type": "Feature", "properties": clean,
                      "geometry": mapping(geom)})
    with open(out_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--src", required=True,
                    help="Folder of foxtrot _canopyai.geojson files to rescore.")
    ap.add_argument("--dst", required=True,
                    help="Output folder for rescored geojsons.")
    ap.add_argument("--image-dir", required=True,
                    help="Folder containing the matching <stem>.tif files.")
    ap.add_argument("--checkpoint", required=True,
                    help="Reranker checkpoint .pt saved by cnn_reranker.py.")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading reranker from {args.checkpoint}...")
    rr = load_ensemble(args.checkpoint, device)
    print(f"  ensemble size: {len(rr)}  backbone: {rr.backbone}  "
          f"patch_size: {rr.patch_size}")

    src = Path(args.src); dst = Path(args.dst); image_dir = Path(args.image_dir)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*_canopyai.geojson"))
    print(f"Rescoring {len(files)} tiles...")
    n_ok = 0
    for f in files:
        stem = f.name.replace("_canopyai.geojson", "")
        gdf = gpd.read_file(str(f))
        if gdf.empty or "deepforest_score" not in gdf.columns:
            # Pass through unchanged
            with open(f) as fh: content = fh.read()
            with open(dst / f.name, "w") as fh: fh.write(content)
            n_ok += 1; continue
        tif = image_dir / f"{stem}.tif"
        if not tif.exists():
            print(f"  ⚠ missing tif for {stem}; passing through unchanged")
            with open(f) as fh: content = fh.read()
            with open(dst / f.name, "w") as fh: fh.write(content)
            n_ok += 1; continue
        image = _load_image(tif)
        polys = list(gdf.geometry)
        new_scores = rr.predict(image, polys)
        _rewrite_geojson(gdf, new_scores, dst / f.name)
        n_ok += 1
    print(f"Done.  Wrote {n_ok}/{len(files)} geojsons to {dst}")


if __name__ == "__main__":
    main()
