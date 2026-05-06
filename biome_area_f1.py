#!/usr/bin/env python3
"""
Biome-group × model area-F1 evaluation.

Threshold search runs on a stratified subsample (~200 tiles) to keep it fast;
final metrics are reported over all tiles at the found threshold.
Optimal thresholds are cached — subsequent runs skip the sweep automatically.

Usage:
    python biome_area_f1.py \
        --models weecology detectree2 phase21_baseline phase21_B_λ4 \
        --names  weecology detectree2 21base 21l4 \
        --output-root benchmark_results

    # Force re-sweep (ignore cache):
    python biome_area_f1.py ... --sweep
"""
import json
import random
import sys
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from shapely.ops import unary_union
from shapely.validation import make_valid
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_tcd import load_gt, load_predictions, biome_group

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.025), 3).tolist()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models",       nargs="+", required=True,
                    help="Model directory names under --output-root")
    ap.add_argument("--names",        nargs="+", required=True,
                    help="Display names (same order as --models)")
    ap.add_argument("--output-root",  default="benchmark_results")
    ap.add_argument("--tcd-dir",      default="data/tcd/images/data/tcd/raw")
    ap.add_argument("--subsample",    type=int, default=200,
                    help="Tiles used for threshold sweep (stratified by biome, default 200)")
    ap.add_argument("--sweep",        action="store_true",
                    help="Force re-sweep even if cached thresholds exist")
    ap.add_argument("--bad-tiles",    default="tcd_bad_tiles.json")
    ap.add_argument("--tiles",        nargs="+", default=None,
                    help="Restrict to these tile stems (e.g. for holdout-only evaluation)")
    ap.add_argument("--cat",          default="all", choices=["all", "1", "2"],
                    help="GT category filter: all=canopy+trees, 2=individual trees only")
    return ap.parse_args()


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _area_f1(pred_polys, gt_polys):
    if not gt_polys:
        return 0.0, 0.0, 0.0
    gt_u = unary_union([p if p.is_valid else make_valid(p) for p in gt_polys])
    if not pred_polys:
        return 0.0, 0.0, 0.0
    pr_u = unary_union([p if p.is_valid else make_valid(p)
                        for p in pred_polys if p is not None and not p.is_empty])
    if pr_u.is_empty:
        return 0.0, 0.0, 0.0
    inter = gt_u.intersection(pr_u).area
    prec  = inter / pr_u.area if pr_u.area > 0 else 0.0
    rec   = inter / gt_u.area if gt_u.area  > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# ── Data loading ──────────────────────────────────────────────────────────────

def tile_stems(tcd_dir, bad_tiles):
    stems = []
    for p in sorted(Path(tcd_dir).glob("tcd_tile_*_meta.json")):
        stem = p.stem.replace("_meta", "")
        if stem not in bad_tiles:
            stems.append(stem)
    return stems


def load_biome_index(stems, tcd_dir):
    groups = {}
    for stem in stems:
        meta_path = Path(tcd_dir) / f"{stem}_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            groups[stem] = biome_group(meta.get("biome_name", ""))
    return groups


def load_tile_data(stems, tcd_dir, model_dirs, cat_filter="all"):
    """Returns gt_by_tile, preds_by_model, groups_by_tile for the given stems."""
    tcd_dir        = Path(tcd_dir)
    gt_by_tile     = {}
    groups_by_tile = {}
    preds_by_model = {name: {} for name in model_dirs}

    for stem in stems:
        meta_path = tcd_dir / f"{stem}_meta.json"
        tif_path  = tcd_dir / f"{stem}.tif"
        if not meta_path.exists() or not tif_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        with rasterio.open(tif_path) as src:
            meta["width"]  = src.width
            meta["height"] = src.height
            meta["bounds"] = list(src.bounds)

        gt_polys, gt_cats = load_gt(meta, tif_path)
        if cat_filter != "all":
            keep = int(cat_filter)
            gt_polys = [p for p, c in zip(gt_polys, gt_cats) if c == keep]
        if not gt_polys:
            continue

        gt_by_tile[stem]     = gt_polys
        groups_by_tile[stem] = biome_group(meta.get("biome_name", ""))

        for name, out_dir in model_dirs.items():
            pred_path = Path(out_dir) / f"{stem}_canopyai.geojson"
            if not pred_path.exists():
                continue
            polys, scores, _ = load_predictions(pred_path, meta)
            preds_by_model[name][stem] = list(zip(polys, scores))

    return gt_by_tile, preds_by_model, groups_by_tile


# ── Threshold search ──────────────────────────────────────────────────────────

def stratified_sample(all_stems, groups_by_tile, n):
    by_group = defaultdict(list)
    for stem in all_stems:
        by_group[groups_by_tile.get(stem, "unknown")].append(stem)
    total = len(all_stems)
    sample = []
    for stems in by_group.values():
        k = max(1, round(n * len(stems) / total))
        sample.extend(random.sample(stems, min(k, len(stems))))
    # top up to n if rounding left us short
    remaining = [s for s in all_stems if s not in set(sample)]
    if len(sample) < n and remaining:
        sample.extend(random.sample(remaining, min(n - len(sample), len(remaining))))
    return sample[:n]


def mean_area_f1_at_threshold(preds_by_tile, gt_by_tile, threshold):
    f1s = []
    for stem, gt_polys in gt_by_tile.items():
        polys = [g for g, s in preds_by_tile.get(stem, []) if s >= threshold]
        _, _, f = _area_f1(polys, gt_polys)
        f1s.append(f)
    return float(np.mean(f1s)) if f1s else 0.0


def find_optimal_threshold(preds_by_tile, gt_by_tile):
    best_thr, best_f1 = THRESHOLDS[0], 0.0
    for thr in THRESHOLDS:
        f = mean_area_f1_at_threshold(preds_by_tile, gt_by_tile, thr)
        if f > best_f1:
            best_f1, best_thr = f, thr
    return best_thr, best_f1


# ── Full evaluation ───────────────────────────────────────────────────────────

def compute_biome_results(preds_by_tile, gt_by_tile, groups_by_tile, threshold):
    per_group = defaultdict(list)
    for stem, gt_polys in gt_by_tile.items():
        polys = [g for g, s in preds_by_tile.get(stem, []) if s >= threshold]
        _, _, f = _area_f1(polys, gt_polys)
        per_group[groups_by_tile[stem]].append(f)
    return per_group


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    tcd_dir     = Path(args.tcd_dir)
    output_root = Path(args.output_root)
    model_dirs  = {name: output_root / model
                   for name, model in zip(args.names, args.models)}

    bad_tiles = set()
    if Path(args.bad_tiles).exists():
        bad_tiles = set(json.loads(Path(args.bad_tiles).read_text()))

    cache_path = output_root / "area_f1_optimal_thresholds.json"
    cached = json.loads(cache_path.read_text()) if cache_path.exists() and not args.sweep else {}

    needs_sweep = [n for n in args.names if n not in cached] if not args.sweep else list(args.names)

    all_stems = tile_stems(tcd_dir, bad_tiles)
    if args.tiles is not None:
        tile_filter = set(args.tiles)
        all_stems = [s for s in all_stems if s in tile_filter]
        print(f"Restricted to {len(all_stems)} tiles via --tiles")

    # ── Threshold sweep on stratified subsample ───────────────────────────────
    if needs_sweep:
        print(f"Threshold sweep needed for: {needs_sweep}")
        print(f"Building biome index ...")
        groups_index = load_biome_index(all_stems, tcd_dir)
        sample = stratified_sample(all_stems, groups_index, args.subsample)
        print(f"Sweeping {len(THRESHOLDS)} thresholds on {len(sample)}-tile subsample ...")

        sweep_dirs = {n: model_dirs[n] for n in needs_sweep}
        gt_sub, preds_sub, _ = load_tile_data(sample, tcd_dir, sweep_dirs, args.cat)
        print(f"  GT loaded for {len(gt_sub)} subsample tiles.\n")

        for name in needs_sweep:
            thr, f1 = find_optimal_threshold(preds_sub[name], gt_sub)
            cached[name] = {"threshold": float(thr), "subsample_area_f1": round(f1, 4),
                            "subsample_n": len(gt_sub)}
            print(f"  {name:25s}  opt_thr={thr:.3f}  subsample_area_F1={f1:.3f}")

        cache_path.write_text(json.dumps(cached, indent=2))
        print(f"\nThresholds cached → {cache_path}\n")
    else:
        print(f"Using cached thresholds for all {len(args.names)} models.\n")

    # ── Full evaluation at cached thresholds ──────────────────────────────────
    print(f"Evaluating {len(all_stems)} tiles ...")
    gt_full, preds_full, groups_full = load_tile_data(all_stems, tcd_dir, model_dirs, args.cat)
    print(f"  GT loaded for {len(gt_full)} tiles.\n")

    optimal        = {n: cached[n]["threshold"] for n in args.names if n in cached}
    biome_results  = {}
    for name in args.names:
        thr = optimal.get(name, 0.35)
        biome_results[name] = compute_biome_results(
            preds_full[name], gt_full, groups_full, thr)

    # ── Table ─────────────────────────────────────────────────────────────────
    all_groups = sorted({g for g in groups_full.values()})
    group_n    = defaultdict(int)
    for stem, g in groups_full.items():
        if stem in gt_full:
            group_n[g] += 1

    headers = ["Biome Group", "N"] + list(args.names)
    rows    = []
    for group in all_groups:
        row = [group, group_n[group]]
        for name in args.names:
            vals = biome_results[name].get(group, [])
            row.append(round(float(np.mean(vals)), 3) if vals else "—")
        rows.append(row)

    rows.append(["─" * 28, "─" * 4] + ["─" * 8] * len(args.names))

    overall_row = ["OVERALL", sum(group_n.values())]
    for name in args.names:
        all_f1 = [f for vals in biome_results[name].values() for f in vals]
        overall_row.append(round(float(np.mean(all_f1)), 3) if all_f1 else "—")
    rows.append(overall_row)

    thr_row = ["(opt threshold)", ""]
    for name in args.names:
        thr_row.append(f"{optimal[name]:.3f}" if name in optimal else "?")
    rows.append(thr_row)

    print(tabulate(rows, headers=headers, tablefmt="simple", floatfmt=".3f"))


if __name__ == "__main__":
    main()
