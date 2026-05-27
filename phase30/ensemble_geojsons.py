#!/usr/bin/env python3
"""
ensemble_geojsons.py — Combine N rescored prediction folders by averaging
per-polygon scores.  Each source must have the same number of features per
tile in the same order (which is the case when they're all derived from
the same source geojsons via cnn_reranker.py).
"""
import argparse, json
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--srcs", nargs="+", required=True,
                    help="Two or more rescored prediction folders.")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--mode", choices=["mean", "rank_mean", "max"], default="mean")
    args = ap.parse_args()

    dst = Path(args.dst); dst.mkdir(parents=True, exist_ok=True)
    srcs = [Path(s) for s in args.srcs]
    files = sorted(srcs[0].glob("*_canopyai.geojson"))

    n_ok = 0
    for f in files:
        # Load all source geojsons for this tile
        gdfs = []
        ok = True
        for s in srcs:
            p = s / f.name
            if not p.exists():
                ok = False; break
            gdfs.append(gpd.read_file(str(p)))
        if not ok or not gdfs:
            continue
        # If any source is empty / missing the score column, pass through the
        # first source unchanged (these are zero-pred tiles).
        if any(g.empty or "deepforest_score" not in g.columns for g in gdfs):
            with open(srcs[0] / f.name) as fh: content = fh.read()
            with open(dst / f.name, "w") as fh: fh.write(content)
            n_ok += 1
            continue
        n = len(gdfs[0])
        if not all(len(g) == n for g in gdfs):
            print(f"  ⚠ {f.name}: pred count mismatch across sources, skipping")
            continue
        scores = np.array([g["deepforest_score"].astype(float).values for g in gdfs])
        if args.mode == "mean":
            new = scores.mean(axis=0)
        elif args.mode == "max":
            new = scores.max(axis=0)
        elif args.mode == "rank_mean":
            ranks = np.zeros_like(scores)
            for i, row in enumerate(scores):
                if len(row) == 0:
                    continue
                ranks[i] = (row.argsort().argsort() + 1) / max(1, len(row))
            new = ranks.mean(axis=0)

        # Write geojson preserving geometry + other props from the FIRST source.
        gdf0 = gdfs[0]
        feats = []
        for idx, geom in enumerate(gdf0.geometry):
            props = {k: v for k, v in gdf0.iloc[idx].items() if k != "geometry"}
            clean = {}
            for k, v in props.items():
                if k == "deepforest_score":
                    clean[k] = float(new[idx]); continue
                if hasattr(v, "tolist"): clean[k] = v.tolist()
                elif isinstance(v, (int, float, str, bool)) or v is None: clean[k] = v
                else: clean[k] = str(v)
            feats.append({"type":"Feature","properties":clean,"geometry":mapping(geom)})
        with open(dst / f.name, "w") as fh:
            json.dump({"type":"FeatureCollection","features":feats}, fh)
        n_ok += 1
    print(f"Ensembled {n_ok}/{len(files)} → {dst}   (mode={args.mode}, n_sources={len(srcs)})")


if __name__ == "__main__":
    main()
