#!/usr/bin/env python3
"""
compare_map50.py — Per-tile mAP50 / AR@1000 comparison between two prediction
folders.  Writes a CSV ranked by mAP50 delta and prints the top-K gainers
(and top-K losers) for follow-up visualisation.

Usage:
    python phase30/compare_map50.py \\
        --baseline kunqi5_epoch98 \\
        --candidate kunqi5_score_full \\
        --output-root benchmark_results_holdout \\
        --out phase30/per_tile_map50.csv \\
        --top-k 5
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import _coco_map50, _eval_tile_worker

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOLDOUT_DIR = REPO_ROOT / "data" / "tcd" / "images" / "data" / "tcd" / "val"


def _tile_metric(args):
    """Worker: compute (mAP50, AR@1000) for one (model_name, stem) pair.
    Returns None if either the meta or the prediction file is missing."""
    name, stem, meta_path, pred_path = args
    meta_path, pred_path = Path(meta_path), Path(pred_path)
    if not meta_path.exists() or not pred_path.exists():
        return (name, stem, None, None, 0, 0)
    r = _eval_tile_worker((0, stem, str(meta_path), pred_path, 0.0))
    images_one = [{"id": 0, "file_name": f"{stem}.tif",
                   "width": r["W"], "height": r["H"]}]
    map50, ar = _coco_map50(images_one, r["gt_anns"], r["pred_dets"])
    return (name, stem, map50, ar, r["n_gt_tree"], r["n_pred"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--baseline",  required=True,
                    help="Folder name under --output-root for the baseline run.")
    ap.add_argument("--candidate", required=True,
                    help="Folder name under --output-root for the candidate run.")
    ap.add_argument("--output-root", default="benchmark_results_holdout")
    ap.add_argument("--holdout-dir", default=str(DEFAULT_HOLDOUT_DIR))
    ap.add_argument("--out", default="phase30/per_tile_map50.csv")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    root        = Path(args.output_root)
    holdout_dir = Path(args.holdout_dir)
    metas = sorted(holdout_dir.glob("*_meta.json"))
    stems = [m.name.replace("_meta.json", "") for m in metas]

    # Build worker args for both models in one pool
    work = []
    for stem, meta in zip(stems, metas):
        for name in (args.baseline, args.candidate):
            pred = root / name / f"{stem}_canopyai.geojson"
            work.append((name, stem, str(meta), str(pred)))

    print(f"Computing per-tile mAP50 on {len(stems)} tiles × 2 models "
          f"({len(work)} worker calls)…")
    results = {}
    with ProcessPoolExecutor() as ex:
        for i, r in enumerate(ex.map(_tile_metric, work, chunksize=4)):
            name, stem, m, a, n_gt, n_pred = r
            results.setdefault(stem, {})[name] = {
                "map50": m, "ar": a, "n_gt": n_gt, "n_pred": n_pred,
            }
            if (i + 1) % 100 == 0 or (i + 1) == len(work):
                print(f"  {i + 1}/{len(work)}")

    # Aggregate: keep only stems with both runs present and at least one tree GT
    rows = []
    for stem, by_name in results.items():
        b = by_name.get(args.baseline,  {})
        c = by_name.get(args.candidate, {})
        if b.get("map50") is None or c.get("map50") is None:
            continue
        rows.append({
            "stem":            stem,
            "n_gt":            b["n_gt"],
            "n_pred_baseline": b["n_pred"],
            "n_pred_candidate": c["n_pred"],
            "map50_baseline":  b["map50"],
            "map50_candidate": c["map50"],
            "map50_delta":     c["map50"] - b["map50"],
            "ar_baseline":     b["ar"],
            "ar_candidate":    c["ar"],
            "ar_delta":        c["ar"] - b["ar"],
        })

    # Write CSV sorted by descending mAP50 delta
    rows.sort(key=lambda r: r["map50_delta"], reverse=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n💾 wrote {out_path}  ({len(rows)} tiles compared)")

    # Console summary
    deltas = [r["map50_delta"] for r in rows]
    n_up   = sum(1 for d in deltas if d > 0)
    n_dn   = sum(1 for d in deltas if d < 0)
    print(f"\nmAP50 delta summary  (candidate − baseline):")
    print(f"  mean   = {sum(deltas)/len(deltas):+.4f}")
    print(f"  median = {sorted(deltas)[len(deltas)//2]:+.4f}")
    print(f"  tiles improved: {n_up}   unchanged: {len(deltas)-n_up-n_dn}   "
          f"regressed: {n_dn}")

    K = args.top_k
    print(f"\nTop {K} mAP50 GAINERS (use these for the visualisation):")
    for r in rows[:K]:
        idx = r["stem"].replace("tcd_val_tile_", "")
        print(f"  tile {idx:>4}  n_gt={r['n_gt']:>3}  "
              f"baseline={r['map50_baseline']:.3f} → candidate={r['map50_candidate']:.3f}  "
              f"Δ={r['map50_delta']:+.3f}")

    print(f"\nTop {K} mAP50 LOSERS (for completeness — investigate regressions):")
    for r in rows[-K:][::-1]:
        idx = r["stem"].replace("tcd_val_tile_", "")
        print(f"  tile {idx:>4}  n_gt={r['n_gt']:>3}  "
              f"baseline={r['map50_baseline']:.3f} → candidate={r['map50_candidate']:.3f}  "
              f"Δ={r['map50_delta']:+.3f}")


if __name__ == "__main__":
    main()
