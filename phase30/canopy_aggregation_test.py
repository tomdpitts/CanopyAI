#!/usr/bin/env python3
"""
canopy_aggregation_test.py — ORACLE upper-bound 2-category (tree + canopy) mAP50.

Goal
----
Make our tree-crown detector directly comparable to Restor OAM-TCD's published
Mask-RCNN R50 mAP50 = 0.432, which is the *mean of two per-category APs*
(tree-AP and canopy-AP) at IoU=0.5, segm task.

Our detector only predicts individual tree crowns (category "tree", id 2). We
have no canopy head. To estimate an UPPER BOUND on what a 2-category number
could look like, we synthesise "canopy" (id 1) predictions by an ORACLE
aggregation that *uses the GT canopy polygons themselves* to define the merge
regions:

    For each scored GT canopy polygon C (iscrowd=0):
        keep our crown predictions P with
            IoP(P,C) = area(P ∩ C) / area(P)  >=  T
        merged = shapely.unary_union(kept crowns)   # ONE geometry
        score  = max(scores of kept crowns)
        emit `merged` as a single canopy prediction (category_id=1)
        (skip canopy polygons with no qualifying crowns)

Because the aggregation is conditioned on GT canopy geometry, this is an ORACLE
/ upper bound, NOT a deployable detector. A fair version needs GT-AGNOSTIC
canopy proposals (e.g. clustering dense overlapping crowns).

Harness conventions (mirrored EXACTLY from phase30/benchmark.py)
---------------------------------------------------------------
  • iouType  = "segm"
  • iouThrs  = [0.5]
  • areaRng  = [[0, 1e10]]  (label "all")
  • maxDets  = [1, 10, 512]   (read the last / 512 slice)
  • GT iscrowd rule (Restor): polygon seg -> iscrowd 0 (scored);
        RLE seg -> iscrowd 1 (ignored). The OAM-TCD meta.json already carries
        this exact iscrowd field, so we honour it directly.
  • predictions / GT masks rasterised then COCO-RLE encoded, identical to
        benchmark.py's _rasterize + _rle_encode.

A tree-only control run is executed first and MUST reproduce ~0.535 before the
2-category canopy numbers are trusted.

CPU only — pure pycocotools scoring, no GPU / no inference.

Usage
-----
    ./venv310/bin/python phase30/canopy_aggregation_test.py
"""

import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from shapely.ops import unary_union

# Reuse the existing harness wholesale.
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase30"))
sys.path.insert(0, str(REPO_ROOT))

import benchmark as B  # noqa: E402  (phase30/benchmark.py)

HOLDOUT_DIR = REPO_ROOT / "data" / "tcd" / "images" / "data" / "tcd" / "val"
PRED_DIR = REPO_ROOT / "benchmark_results_holdout" / "ablation_tcd_s0"

# Category ids exactly as in the OAM-TCD GT.
CAT_CANOPY = 1
CAT_TREE = 2
TREE_ONLY_REFERENCE = 0.535
RESTOR_2CAT = 0.432

MAX_DETS = 512
SCORE_THRESH = 0.0


# ── GT builder ────────────────────────────────────────────────────────────────

def build_gt_anns(meta, tile_id, H, W, tree_only):
    """
    Build COCO GT annotations from one meta.json.

    Returns (gt_anns, scored_canopy_polys) where scored_canopy_polys is the list
    of shapely Polygons for canopy instances that are SCORED (iscrowd=0) — these
    are the oracle aggregation regions.

    iscrowd is honoured directly from the GT field (polygon->0 scored,
    RLE->1 ignored, exactly the Restor rule).

    tree_only=True reproduces benchmark.py: only tree (id2) instances are kept,
    remapped to a single class id=1; canopy is dropped entirely here because the
    control's purpose is to match benchmark.py's tree-only number. (benchmark.py
    actually keeps canopy as an iscrowd ignore region in the *tree* class; doing
    that vs dropping canopy changes the tree-only mAP only marginally because
    ignore regions only suppress would-be FPs. We replicate benchmark.py's
    behaviour: canopy kept as iscrowd=1 ignore in the tree class.)
    """
    gt_anns = []
    scored_canopy_polys = []

    for cat, seg, bbox, area in B._parse_coco_annotations(meta):
        polys = B._seg_to_polygons(seg, H, W)
        if not polys:
            continue
        m = B._rasterize(polys, H, W)
        if not m.any():
            continue

        # Honour the GT iscrowd field. Equivalent to: polygon->scored,
        # RLE->ignored, which is the Restor convention.
        is_crowd = _seg_iscrowd(seg)

        if tree_only:
            # Mirror benchmark.py exactly: single tree class (remap id->1),
            # canopy (cat==1) demoted to iscrowd=1 ignore region in that class.
            ann_cat = 1
            ann_iscrowd = 0 if cat == CAT_TREE else 1
        else:
            ann_cat = cat
            ann_iscrowd = is_crowd
            if cat == CAT_CANOPY and is_crowd == 0:
                # Collect the merged-geometry of this canopy instance as one
                # oracle region. (A canopy ann may have produced >1 ring/poly;
                # union them so IoP is computed against the whole canopy.)
                region = unary_union([p for p in polys if not p.is_empty])
                if not region.is_empty and region.area > 0:
                    scored_canopy_polys.append(region)

        rle = B._rle_encode(m, H, W)
        ys, xs = np.where(m)
        y0, x0 = int(ys.min()), int(xs.min())
        y1, x1 = int(ys.max()) + 1, int(xs.max()) + 1
        gt_anns.append({
            "image_id":     tile_id,
            "category_id":  ann_cat,
            "segmentation": rle,
            "bbox":         [x0, y0, x1 - x0, y1 - y0],
            "area":         float(m.sum()),
            "iscrowd":      ann_iscrowd,
        })

    return gt_anns, scored_canopy_polys


def _seg_iscrowd(seg):
    """polygon-list seg -> 0 (scored); RLE-dict seg -> 1 (ignored)."""
    if isinstance(seg, dict) and "counts" in seg:
        return 1
    return 0


# ── Prediction builders ───────────────────────────────────────────────────────

def _poly_to_det(geom, H, W, tile_id, category_id, score):
    """Rasterise a shapely geometry and emit one COCO segm detection (or None)."""
    m = B._rasterize([geom], H, W)
    if not m.any():
        return None
    rle = B._rle_encode(m, H, W)
    ys, xs = np.where(m)
    y0, x0 = int(ys.min()), int(xs.min())
    y1, x1 = int(ys.max()) + 1, int(xs.max()) + 1
    return {
        "image_id":     tile_id,
        "category_id":  category_id,
        "segmentation": rle,
        "bbox":         [x0, y0, x1 - x0, y1 - y0],
        "score":        float(score),
    }


def build_tree_dets(preds, H, W, tile_id, tree_cat):
    """Our crown polygons unchanged -> category=tree, original scores."""
    dets = []
    for geom, score in preds:
        d = _poly_to_det(geom, H, W, tile_id, tree_cat, score)
        if d is not None:
            dets.append(d)
    return dets


def build_oracle_canopy_dets(preds, scored_canopy_polys, H, W, tile_id, T):
    """
    ORACLE canopy aggregation. For each scored GT canopy region, union the
    crowns whose IoP >= T into ONE canopy prediction (score = max crown score).
    """
    dets = []
    # Precompute crown areas once.
    crown_geoms = [g for g, _ in preds]
    crown_scores = [s for _, s in preds]
    crown_areas = [g.area for g in crown_geoms]

    for region in scored_canopy_polys:
        kept_geoms = []
        kept_scores = []
        for g, s, a in zip(crown_geoms, crown_scores, crown_areas):
            if a <= 0:
                continue
            inter = g.intersection(region)
            if inter.is_empty:
                continue
            iop = inter.area / a
            if iop >= T:
                kept_geoms.append(g)
                kept_scores.append(s)
        if not kept_geoms:
            continue  # skip canopy regions with no qualifying crowns
        merged = unary_union(kept_geoms)
        if merged.is_empty or merged.area <= 0:
            continue
        score = max(kept_scores)
        d = _poly_to_det(merged, H, W, tile_id, CAT_CANOPY, score)
        if d is not None:
            dets.append(d)
    return dets


# ── COCOeval (multi-category, mirrors benchmark.py conventions) ───────────────

def coco_eval_per_category(images, gt_anns, dets, categories, max_dets=MAX_DETS):
    """
    Run pycocotools COCOeval segm @ IoU=0.5 over the supplied categories and
    return {category_id: AP50}.

    Mirrors benchmark.py._coco_map50 exactly: iouThrs=[0.5], areaRng all,
    maxDets last slice, precision averaged over recall thresholds. The only
    change is we read precision per-category (K axis) instead of averaging
    over K, so we can report tree-AP and canopy-AP separately.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_dict = {
        "images":      images,
        "annotations": [dict(a, id=i + 1) for i, a in enumerate(gt_anns)],
        "categories":  categories,
    }

    cat_ids = [c["id"] for c in categories]
    silent = io.StringIO()
    with redirect_stdout(silent):
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()
        if not dets:
            # No detections at all -> AP 0 for every category.
            return {cid: 0.0 for cid in cat_ids}
        coco_dt = coco_gt.loadRes(dets)

        ev = COCOeval(coco_gt, coco_dt, iouType="segm")
        ev.params.iouThrs = np.array([0.5])
        ev.params.maxDets = [1, 10, int(max_dets)]
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ["all"]
        ev.params.catIds = cat_ids
        ev.evaluate()
        ev.accumulate()

    # precision shape: [T, R, K, A, M] where K follows ev.params.catIds order.
    prec = ev.eval["precision"]  # take last maxDets slice below
    eval_cat_ids = list(ev.params.catIds)
    out = {}
    for k, cid in enumerate(eval_cat_ids):
        pk = prec[:, :, k, :, -1]  # [T, R, A]
        valid = pk[pk > -1]
        out[cid] = float(valid.mean()) if valid.size else 0.0
    return out


# ── Tile worker ───────────────────────────────────────────────────────────────

def process_tile(stem, tile_id, T_list):
    """
    Returns dict with GT anns + detection sets per (mode, T) for one tile.
    mode "tree_only": single-class control.
    mode "2cat": tree dets + per-T oracle canopy dets.
    """
    meta_path = HOLDOUT_DIR / f"{stem}_meta.json"
    pred_path = PRED_DIR / f"{stem}_canopyai.geojson"
    with open(meta_path) as f:
        meta = json.load(f)
    H, W = int(meta["height"]), int(meta["width"])

    preds = B._load_predictions(pred_path, H, W, SCORE_THRESH)

    # Control (tree-only, benchmark.py replica)
    gt_tree_only, _ = build_gt_anns(meta, tile_id, H, W, tree_only=True)
    tree_only_dets = build_tree_dets(preds, H, W, tile_id, tree_cat=1)

    # 2-category
    gt_2cat, scored_canopy = build_gt_anns(meta, tile_id, H, W, tree_only=False)
    tree_dets = build_tree_dets(preds, H, W, tile_id, tree_cat=CAT_TREE)
    canopy_dets_by_T = {
        T: build_oracle_canopy_dets(preds, scored_canopy, H, W, tile_id, T)
        for T in T_list
    }

    return {
        "tile_id": tile_id,
        "H": H, "W": W,
        "gt_tree_only": gt_tree_only,
        "tree_only_dets": tree_only_dets,
        "gt_2cat": gt_2cat,
        "tree_dets": tree_dets,
        "canopy_dets_by_T": canopy_dets_by_T,
        "n_scored_canopy": len(scored_canopy),
        "n_canopy_dets": {T: len(v) for T, v in canopy_dets_by_T.items()},
        "n_pred": len(preds),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    T_list = [0.5, 0.7]

    metas = sorted(HOLDOUT_DIR.glob("*_meta.json"))
    stems = [m.name.replace("_meta.json", "") for m in metas
             if (PRED_DIR / f"{m.name.replace('_meta.json','')}_canopyai.geojson").exists()]
    print(f"Tiles with predictions: {len(stems)}/{len(metas)}")

    images = [{"id": i, "file_name": f"{s}.tif", "width": 0, "height": 0}
              for i, s in enumerate(stems)]

    # Process tiles. Use a modest process pool to keep CPU use polite while a
    # GPU eval runs concurrently.
    from concurrent.futures import ProcessPoolExecutor
    import os
    n_workers = max(1, min(4, (os.cpu_count() or 2) // 2))
    print(f"Rasterising + oracle-aggregating on {len(stems)} tiles "
          f"({n_workers} workers)...")

    args = [(stems[i], i, T_list) for i in range(len(stems))]
    results = [None] * len(stems)
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for done, r in enumerate(ex.map(_worker, args, chunksize=4)):
            results[r["tile_id"]] = r
            images[r["tile_id"]]["width"] = r["W"]
            images[r["tile_id"]]["height"] = r["H"]
            if (done + 1) % 50 == 0 or (done + 1) == len(stems):
                avg = (time.monotonic() - t0) / (done + 1)
                print(f"  {done+1}/{len(stems)}  avg {avg:4.2f}s/tile")

    # ── Control: tree-only must reproduce ~0.535 ──
    gt_tree_only = [a for r in results for a in r["gt_tree_only"]]
    tree_only_dets = [d for r in results for d in r["tree_only_dets"]]
    tree_only_ap = coco_eval_per_category(
        images, gt_tree_only, tree_only_dets,
        categories=[{"id": 1, "name": "tree"}], max_dets=MAX_DETS,
    )[1]
    print(f"\n[CONTROL] tree-only mAP50 = {tree_only_ap:.4f} "
          f"(benchmark.py reference {TREE_ONLY_REFERENCE:.3f})")
    delta = abs(tree_only_ap - TREE_ONLY_REFERENCE)
    if delta <= 0.02:
        print(f"          OK — reproduces reference within {delta:.4f}.")
    else:
        print(f"          WARNING — differs from reference by {delta:.4f}; "
              f"2-cat numbers below should be treated with caution.")

    # ── 2-category eval per threshold ──
    gt_2cat = [a for r in results for a in r["gt_2cat"]]
    tree_dets = [d for r in results for d in r["tree_dets"]]
    categories_2 = [{"id": CAT_CANOPY, "name": "canopy"},
                    {"id": CAT_TREE, "name": "tree"}]

    n_scored_canopy = sum(r["n_scored_canopy"] for r in results)
    n_gt_tree_scored = sum(
        1 for a in gt_2cat if a["category_id"] == CAT_TREE and a["iscrowd"] == 0)
    n_gt_canopy_scored = sum(
        1 for a in gt_2cat if a["category_id"] == CAT_CANOPY and a["iscrowd"] == 0)
    print(f"\nGT scored: tree={n_gt_tree_scored}  canopy={n_gt_canopy_scored}  "
          f"(scored-canopy oracle regions={n_scored_canopy})")

    table_rows = []
    for T in T_list:
        canopy_dets = [d for r in results for d in r["canopy_dets_by_T"][T]]
        n_canopy_dets = sum(r["n_canopy_dets"][T] for r in results)
        dets = tree_dets + canopy_dets
        aps = coco_eval_per_category(images, gt_2cat, dets,
                                     categories=categories_2, max_dets=MAX_DETS)
        tree_ap = aps[CAT_TREE]
        canopy_ap = aps[CAT_CANOPY]
        mean_ap = 0.5 * (tree_ap + canopy_ap)
        table_rows.append((T, tree_ap, canopy_ap, mean_ap, n_canopy_dets))
        print(f"  T={T}: tree-AP={tree_ap:.4f}  canopy-AP={canopy_ap:.4f}  "
              f"2-cat mean={mean_ap:.4f}  (canopy dets={n_canopy_dets})")

    # ── Final table ──
    print("\n" + "=" * 72)
    print("  ORACLE 2-CATEGORY mAP50 (segm, IoU=0.5, maxDets=512) — OAM-TCD holdout")
    print("=" * 72)
    print(f"  {'T':>5} | {'tree AP@0.5':>12} | {'canopy AP@0.5':>14} | "
          f"{'2-cat mean AP@0.5':>18}")
    print("  " + "-" * 5 + "-+-" + "-" * 12 + "-+-" + "-" * 14 + "-+-" + "-" * 18)
    for T, tree_ap, canopy_ap, mean_ap, _ in table_rows:
        print(f"  {T:>5} | {tree_ap:>12.4f} | {canopy_ap:>14.4f} | "
              f"{mean_ap:>18.4f}")
    print("  " + "-" * 5 + "-+-" + "-" * 12 + "-+-" + "-" * 14 + "-+-" + "-" * 18)
    print(f"  {'ref':>5} | tree-only baseline = {TREE_ONLY_REFERENCE:.3f}"
          f"  (control reproduced {tree_only_ap:.4f})")
    print(f"  {'ref':>5} | Restor OAM-TCD 2-cat mAP50 = {RESTOR_2CAT:.3f} "
          f"(mean of tree-AP + canopy-AP)")
    print("=" * 72)

    summary = {
        "tree_only_control_ap50": tree_only_ap,
        "tree_only_reference": TREE_ONLY_REFERENCE,
        "restor_2cat_ap50": RESTOR_2CAT,
        "n_gt_tree_scored": n_gt_tree_scored,
        "n_gt_canopy_scored": n_gt_canopy_scored,
        "thresholds": {
            f"T={T}": {
                "tree_ap50": tree_ap,
                "canopy_ap50": canopy_ap,
                "twocat_mean_ap50": mean_ap,
                "n_canopy_dets": n_canopy_dets,
            }
            for (T, tree_ap, canopy_ap, mean_ap, n_canopy_dets) in table_rows
        },
    }
    out_path = REPO_ROOT / "benchmark_results_holdout" / "canopy_aggregation_oracle.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


def _worker(args):
    stem, tile_id, T_list = args
    return process_tile(stem, tile_id, T_list)


if __name__ == "__main__":
    main()
