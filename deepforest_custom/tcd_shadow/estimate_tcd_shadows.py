#!/usr/bin/env python3
"""
estimate_tcd_shadows.py — Estimate per-tile shadow vectors for TCD training data.

Runs the ResNet-34 shadow regression model on each 2048×2048 tile in
data/tcd/images/data/tcd/raw/, samples 30 random 500×500 crops, computes a
circular mean with outlier rejection, and writes results to:

    data/tcd/tcd_shadow_vectors.json

Only tiles with consensus_pct >= --min-consensus are included in the JSON.
Excluded tiles get no shadow_x/shadow_y in the training CSV, so
shadow_loss_reweight does not fire for them — they train identically to
baseline. This is the correct behaviour: shadow reweighting is only meaningful
where tree shadows are clearly visible (sparse/open canopy).

The JSON is consumed by prepare_tcd_training.py to add shadow_x/shadow_y
columns to training chips. It is also used by review_tcd_shadows.py for
manual correction of low-confidence estimates.

Usage:
    # Run from project root:
    source venv310/bin/activate
    python deepforest_custom/tcd_shadow/estimate_tcd_shadows.py

    # Skip tiles already in the JSON (resume after interruption):
    python deepforest_custom/tcd_shadow/estimate_tcd_shadows.py --skip-existing

    # Stricter consensus filter:
    python deepforest_custom/tcd_shadow/estimate_tcd_shadows.py --min-consensus 70
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# ── Paths (relative to project root) ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deepforest_custom"))

from predict_tcd_shadows import load_shadow_model, predict_shadow_vector  # noqa: E402

DEFAULT_SHADOW_MODEL = ROOT / "solar/shadow_regression/output/shadow_model_combined_best.pth"
DEFAULT_TCD_DIR      = ROOT / "data/tcd/images/data/tcd/by_id"
DEFAULT_OUTPUT       = ROOT / "data/tcd/tcd_shadow_vectors_by_id.json"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcd-dir",       default=str(DEFAULT_TCD_DIR))
    ap.add_argument("--output",        default=str(DEFAULT_OUTPUT))
    ap.add_argument("--shadow-model",  default=str(DEFAULT_SHADOW_MODEL))
    ap.add_argument("--min-consensus", type=float, default=60.0,
                    help="Min consensus %% to write tile to JSON (default 60)")
    ap.add_argument("--n-crops",       type=int,   default=30,
                    help="Random crops per tile (default 30)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip tiles already present in output JSON")
    return ap.parse_args()


def main():
    args    = parse_args()
    tcd_dir = Path(args.tcd_dir)
    out_path = Path(args.output)

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device : {device}")
    print(f"TCD dir: {tcd_dir}")
    print(f"Output : {out_path}")

    model = load_shadow_model(args.shadow_model, device)

    # Load existing results (JSON stores ALL tiles, including low-confidence ones,
    # so the review app can show them — the min-consensus filter is applied at
    # prepare_tcd_training.py time, not here).
    results = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())
        print(f"Loaded {len(results)} existing entries from {out_path.name}")

    tiles = sorted(
        p.stem for p in tcd_dir.glob("*.tif")
        if (tcd_dir / f"{p.stem}_meta.json").exists()
    )
    print(f"Found {len(tiles)} tiles")

    if args.skip_existing:
        tiles = [t for t in tiles if t not in results]
        print(f"Skipping existing — processing {len(tiles)} remaining tiles")

    for stem in tqdm(tiles, unit="tile", desc="Estimating"):
        tif_path = tcd_dir / f"{stem}.tif"
        try:
            shadow_vec, angle_deg, stats = predict_shadow_vector(
                str(tif_path), model, device, n_crops=args.n_crops
            )
            results[stem] = {
                "shadow_x":         float(shadow_vec[0]),
                "shadow_y":         float(shadow_vec[1]),
                "shadow_angle_deg": float(angle_deg),
                "consensus_pct":    float(stats["consensus_pct"]),
                "circular_std_deg": float(stats["circular_std_deg"]),
                "n_crops":          int(stats["n_crops"]),
                "n_inliers":        int(stats["n_inliers"]),
                "manually_reviewed": False,
            }
        except Exception as e:
            tqdm.write(f"  ⚠  {stem}: {e}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    # Summary
    n_total    = len(results)
    n_accepted = sum(1 for v in results.values()
                     if v["consensus_pct"] >= args.min_consensus)
    n_reviewed = sum(1 for v in results.values() if v.get("manually_reviewed"))
    print(f"\n{'─'*50}")
    print(f"Total tiles estimated : {n_total}")
    print(f"High confidence       : {n_accepted}  (≥{args.min_consensus}% consensus)")
    print(f"Low confidence        : {n_total - n_accepted}  (<{args.min_consensus}%)")
    print(f"Manually reviewed     : {n_reviewed}")
    print(f"\n→ {n_accepted} tiles will receive shadow_x/shadow_y in training CSV")
    print(f"→ {n_total - n_accepted} tiles will train as baseline (no shadow reweight)")
    print(f"\nSaved → {out_path}")
    print(f"\nNext: review low-confidence tiles with:")
    print(f"  python deepforest_custom/tcd_shadow/review_tcd_shadows.py")


if __name__ == "__main__":
    main()
