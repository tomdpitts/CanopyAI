#!/usr/bin/env python3
"""
run_restor.py — Restor `restor/tcd-mask-rcnn-r50` baseline, LOCAL CPU runner.

Goal
----
Read the *per-category* instance AP that the Restor Mask-RCNN produces on our
439-tile OAM-TCD holdout — the one number missing from
``phase30/RESTOR_COMPARISON.md``:

  • overall AP50  (mean of the two categories — sanity target ≈ 0.432, the
    headline they publish),
  • **AP-tree** and **AP-canopy** broken out separately,

and export the **tree-category** predictions per tile as
``{stem}_canopyai.geojson`` so they can later be scored through our own
tree-only convention (``phase30/benchmark.py``) for a like-for-like number
against our 0.535.

This file is deliberately self-contained and isolated under
``phase30/restor_baseline/`` so the Detectron2 stack never touches the
DeepForest/SAM ``venv310`` env. Run it with the sibling ``venv_restor``:

    phase30/restor_baseline/venv_restor/bin/python \
        phase30/restor_baseline/run_restor.py --tiles 3        # smoke test
    phase30/restor_baseline/venv_restor/bin/python \
        phase30/restor_baseline/run_restor.py                  # full 439

Model facts (verified against HF ``restor/tcd-mask-rcnn-r50/config.yaml`` and
the ``Restor-Foundation/tcd`` source — see README for citations)
-----------------------------------------------------------------------------
  • Detectron2 GeneralizedRCNN, Mask-RCNN R50-FPN; standalone ``config.yaml``
    (no model-zoo merge needed — it is a full dump).
  • ``MODEL.WEIGHTS`` in the yaml points at the COCO init checkpoint on
    fbaipublicfiles; we OVERRIDE it with the downloaded ``model.pth``.
  • ``MODEL.DEVICE: cuda`` in the yaml; we OVERRIDE to ``cpu`` (Detectron2's
    MPS coverage is patchy on Apple Silicon).
  • ``ROI_HEADS.NUM_CLASSES = 2``. The class ORDER is ``["canopy", "tree"]``
    (``config/data/default.yaml``) and the ``Vegetation`` IntEnum in
    ``tcd_pipeline/util.py`` fixes it: **CANOPY = 0, TREE = 1**. So:
        detectron2 contiguous class 0  → canopy → OAM-TCD COCO category_id 1
        detectron2 contiguous class 1  → tree   → OAM-TCD COCO category_id 2
  • ``INPUT.FORMAT = BGR``, ``MASK_FORMAT = polygon``, ``MIN_SIZE_TEST = 0``
    (no short-edge resize), ``MAX_SIZE_TEST = 2048``. The holdout val tiles are
    2048x2048 RGB uint8 (native training GSD 0.1 m), so we feed the full tile —
    matching how Restor score their holdout — and do NOT re-tile to 1024.
  • ``SCORE_THRESH_TEST = 0.2``, ``DETECTIONS_PER_IMAGE = 512``. Left at the
    Restor defaults so the AP reproduces their setup.

GT / eval convention (kept consistent with ``phase30/benchmark.py``)
-----------------------------------------------------------------------------
  • meta.json is per-tile COCO GT: trees (cat 2) = polygon-list, iscrowd 0;
    canopy (cat 1) = compressed-RLE dict, iscrowd 1 in the file.
  • For the 2-category COCO eval here we KEEP both categories as real
    positives (iscrowd forced to 0) — that is exactly Restor's framing, whose
    ``COCOEvaluator`` applies no category filter, so the 0.432 = mean(AP-tree,
    AP-canopy). (This differs from benchmark.py, which deliberately demotes
    canopy to an ignore region for its *tree-only* number — a different
    question, scored separately on the exported geojsons.)
  • Masks are turned into compressed RLE; eval is pycocotools ``COCOeval``
    ``segm``, IoU 0.5 only, areaRng "all", maxDets 512.

Geojson export schema (consumed by ``phase30/benchmark._load_predictions``)
-----------------------------------------------------------------------------
  FeatureCollection of Polygon features in PIXEL coords, each with a numeric
  ``score`` property (benchmark.py reads any of deepforest_score/score/...).
  We mark the CRS ``EPSG:3395`` to mirror the existing files, but the coords
  are pixel-space (benchmark.py treats val-tile geojsons as pixel-space).
  Only TREE-category (class 1) predictions are exported.
"""

import argparse
import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import rasterio

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_HOLDOUT_DIR = REPO_ROOT / "data" / "tcd" / "images" / "data" / "tcd" / "val"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark_results_restor"

HF_REPO = "restor/tcd-mask-rcnn-r50"

# detectron2 contiguous class index → OAM-TCD COCO category_id
#   0 = canopy → 1 ;  1 = tree → 2     (Vegetation IntEnum: CANOPY=0, TREE=1)
D2_TO_COCO_CAT = {0: 1, 1: 2}
COCO_CAT_NAME = {1: "canopy", 2: "tree"}
TREE_D2_CLASS = 1  # the class we export to geojson


# ── Model loading ────────────────────────────────────────────────────────────
def build_predictor(device="cpu", score_thresh=None, tta=False):
    """Build a Detectron2 predictor from the Restor HF config + weights.

    The yaml is a complete standalone config; we only override DEVICE and
    WEIGHTS (and optionally the test score threshold). ``set_new_allowed`` is
    required because the dumped config carries a couple of keys not in the
    stock CfgNode schema for this detectron2 build.

    Returns ``(predict_callable, cfg)`` where ``predict_callable(bgr)`` yields
    ``{"instances": Instances}`` regardless of TTA.

    tta=True reproduces Restor's PUBLISHED R50 setting: multi-scale test-time
    augmentation (their ``config/model/detectron2/detectron_mask_rcnn_tta.yaml``
    — ``MIN_SIZES [768,800,900,1024,1500,2048]``, horizontal flip).  Their
    headline holdout mAP50=0.432 uses this; single-scale (tta=False) reproduces
    their *cross-validation* number (~0.418).  This is the empirical test of
    whether TTA explains our 0.402-vs-0.432 gap.
    """
    from huggingface_hub import hf_hub_download
    from detectron2.config import get_cfg

    cfg_path = hf_hub_download(HF_REPO, "config.yaml")
    weights_path = hf_hub_download(HF_REPO, "model.pth")

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = weights_path
    if score_thresh is not None:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)

    if not tta:
        cfg.freeze()
        from detectron2.engine import DefaultPredictor
        return DefaultPredictor(cfg), cfg

    # --- multi-scale TTA (matches Restor's detectron_mask_rcnn_tta.yaml) ---
    cfg.TEST.AUG.ENABLED = True
    cfg.TEST.AUG.MIN_SIZES = (768, 800, 900, 1024, 1500, 2048)
    cfg.TEST.AUG.MAX_SIZE = 4096          # don't cap the 2048 scale
    cfg.TEST.AUG.FLIP = True
    cfg.freeze()

    import torch
    from detectron2.modeling import build_model, GeneralizedRCNNWithTTA
    from detectron2.checkpoint import DetectionCheckpointer

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(weights_path)
    tta_model = GeneralizedRCNNWithTTA(cfg, model)

    def _predict(bgr):
        # DatasetMapperTTA reads dataset_dict["image"] as a CHW tensor in
        # INPUT.FORMAT (BGR here, matching read_tile_bgr) and applies the
        # multi-scale + flip augmentations itself.
        H, W = bgr.shape[:2]
        img = torch.as_tensor(np.ascontiguousarray(bgr.transpose(2, 0, 1)))
        with torch.no_grad():
            return tta_model([{"image": img, "height": H, "width": W}])[0]

    return _predict, cfg


# ── Tile inference ───────────────────────────────────────────────────────────
def read_tile_bgr(tif_path):
    """Read an RGB uint8 tile and return an HxWx3 BGR ndarray (INPUT.FORMAT)."""
    with rasterio.open(tif_path) as ds:
        rgb = ds.read([1, 2, 3]).transpose(1, 2, 0)  # HWC, RGB
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return bgr, rgb.shape[0], rgb.shape[1]


def infer_tile(predictor, bgr):
    """Run the predictor; return (classes, scores, masks_bool) as numpy.

    masks_bool: (N, H, W) bool array of instance masks at tile resolution.
    """
    out = predictor(bgr)
    inst = out["instances"].to("cpu")
    n = len(inst)
    if n == 0:
        H, W = bgr.shape[:2]
        return (np.empty(0, np.int64), np.empty(0, np.float32),
                np.empty((0, H, W), bool))
    classes = inst.pred_classes.numpy().astype(np.int64)
    scores = inst.scores.numpy().astype(np.float32)
    masks = inst.pred_masks.numpy().astype(bool)  # (N, H, W)
    return classes, scores, masks


# ── COCO assembly ────────────────────────────────────────────────────────────
def _rle_from_mask(mask_bool):
    """Compressed-RLE dict (counts decoded to ascii str) from an HxW bool mask."""
    from pycocotools import mask as mask_utils
    rle = mask_utils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _parse_coco_annotations(meta):
    """Yield (category_id, segmentation) from a meta.json (handles stringified)."""
    coco = meta.get("coco_annotations", [])
    if isinstance(coco, str):
        coco = json.loads(coco)
    for ann in coco or []:
        seg = ann.get("segmentation")
        if not seg:
            continue
        yield int(ann.get("category_id", 2)), seg


def gt_anns_from_meta(meta, tile_id, H, W):
    """Build 2-category COCO GT annotations for one tile.

    Both tree (cat 2) and canopy (cat 1) are real positives (iscrowd=0) — the
    Restor 2-category framing whose mean is 0.432. Segmentations are
    re-encoded to compressed RLE so IoU is computed on masks (segm task),
    matching detectron2's COCOEvaluator.
    """
    from pycocotools import mask as mask_utils

    anns = []
    for cat, seg in _parse_coco_annotations(meta):
        # Normalise any segmentation form (polygon list OR RLE dict) to a
        # single binary mask, then to compressed RLE.
        if isinstance(seg, list) and seg and isinstance(seg[0], list):
            rle_objs = mask_utils.frPyObjects(seg, H, W)
            rle = mask_utils.merge(rle_objs)
        elif isinstance(seg, dict) and "counts" in seg:
            r = dict(seg)
            if isinstance(r.get("counts"), str):
                r["counts"] = r["counts"].encode("ascii")
            rle = mask_utils.frPyObjects(r, H, W) if isinstance(
                r.get("counts"), list) else r
        else:
            continue
        area = float(mask_utils.area(rle))
        if area <= 0:
            continue
        bbox = [float(v) for v in mask_utils.toBbox(rle)]
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        anns.append({
            "image_id": tile_id,
            "category_id": cat,           # 1 canopy / 2 tree
            "segmentation": {"size": [H, W], "counts": counts},
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        })
    return anns


def dets_from_inference(classes, scores, masks, tile_id):
    """COCO detection dicts (both categories) for one tile."""
    from pycocotools import mask as mask_utils
    dets = []
    for cls, score, m in zip(classes, scores, masks):
        if not m.any():
            continue
        rle = _rle_from_mask(m)
        bbox = [float(v) for v in mask_utils.toBbox({
            "size": rle["size"], "counts": rle["counts"].encode("ascii")})]
        dets.append({
            "image_id": tile_id,
            "category_id": D2_TO_COCO_CAT[int(cls)],
            "segmentation": rle,
            "bbox": bbox,
            "score": float(score),
        })
    return dets


# ── Geojson export (tree-category only) ──────────────────────────────────────
def export_tree_geojson(out_path, classes, scores, masks):
    """Write tree-category (class 1) masks as a pixel-coord polygon geojson
    that ``phase30/benchmark._load_predictions`` can read (FeatureCollection,
    Polygon geometry, numeric ``score`` property)."""
    import rasterio.features
    from shapely.geometry import shape
    from shapely.validation import make_valid

    features = []
    fid = 0
    for cls, score, m in zip(classes, scores, masks):
        if int(cls) != TREE_D2_CLASS or not m.any():
            continue
        # Polygonise the binary mask. A predicted instance can in principle
        # split into >1 connected component; emit the largest as the crown
        # (benchmark.py rasterises each feature back to a mask, so a single
        # representative polygon per instance preserves the score↔mask pairing).
        best = None
        for geom, val in rasterio.features.shapes(
                m.astype(np.uint8), mask=m):
            if val != 1:
                continue
            poly = shape(geom)
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.is_empty or poly.area <= 0:
                continue
            if best is None or poly.area > best.area:
                best = poly
        if best is None:
            continue
        # Use only the exterior ring (benchmark rasterises the filled polygon).
        geom = best.geoms[0] if best.geom_type == "MultiPolygon" else best
        coords = [[float(x), float(y)] for x, y in geom.exterior.coords]
        features.append({
            "type": "Feature",
            "id": fid,
            "properties": {
                "tree_id": fid,
                "score": float(score),
                "area_pixels": float(geom.area),
            },
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })
        fid += 1

    fc = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": "EPSG:3395"}},
    }
    out_path.write_text(json.dumps(fc))
    return len(features)


# ── Per-category COCO eval ───────────────────────────────────────────────────
def coco_eval_per_category(images, gt_anns, dets, max_dets=512):
    """Run pycocotools COCOeval (segm, IoU 0.5, areaRng all, maxDets given).

    Returns dict {category_id: AP50} plus an "overall" key = mean over the two
    categories (the Restor 0.432 framing). Mirrors the precision-slice readout
    in benchmark.py (take the largest-maxDets slice; don't average over the
    maxDets axis, which would cap recall on dense tiles).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not gt_anns or not dets:
        return {"overall": None, 1: None, 2: None}

    cat_ids = [1, 2]
    gt_dict = {
        "images": images,
        "annotations": [dict(a, id=i + 1) for i, a in enumerate(gt_anns)],
        "categories": [{"id": c, "name": COCO_CAT_NAME[c]} for c in cat_ids],
    }

    silent = io.StringIO()
    with redirect_stdout(silent):
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)

        ev = COCOeval(coco_gt, coco_dt, iouType="segm")
        ev.params.iouThrs = np.array([0.5])
        ev.params.maxDets = [1, 10, int(max_dets)]
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ["all"]
        ev.params.catIds = cat_ids
        ev.evaluate()
        ev.accumulate()

    # precision shape: [T, R, K, A, M]; K indexes catIds in params.catIds order.
    prec = ev.eval["precision"]  # [1, R, K, 1, M]
    out = {}
    per_cat_ap = []
    for k, cat in enumerate(cat_ids):
        sl = prec[:, :, k, :, -1]          # largest maxDets slice
        valid = sl[sl > -1]
        ap = float(valid.mean()) if valid.size else None
        out[cat] = ap
        if ap is not None:
            per_cat_ap.append(ap)
    out["overall"] = float(np.mean(per_cat_ap)) if per_cat_ap else None
    return out


# ── Orchestration ────────────────────────────────────────────────────────────
def run(holdout_dir, output_dir, limit=None, device="cpu",
        max_dets=512, score_thresh=None, skip_existing=False, tta=False):
    holdout_dir = Path(holdout_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metas = sorted(holdout_dir.glob("*_meta.json"))
    if not metas:
        print(f"ERROR: no *_meta.json under {holdout_dir}", file=sys.stderr)
        return 1
    if limit is not None:
        metas = metas[:limit]

    print(f"Holdout dir : {holdout_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Tiles       : {len(metas)}"
          + (f"  (limited from {len(list(holdout_dir.glob('*_meta.json')))})"
             if limit is not None else ""))
    print(f"Device      : {device}   maxDets={max_dets}   TTA={tta}")
    print("Building predictor (loading Restor weights)...", flush=True)
    t0 = time.time()
    predictor, cfg = build_predictor(device=device, score_thresh=score_thresh,
                                     tta=tta)
    print(f"  predictor ready in {time.time() - t0:.1f}s  "
          f"(NUM_CLASSES={cfg.MODEL.ROI_HEADS.NUM_CLASSES}, "
          f"SCORE_THRESH={cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}, "
          f"DETS={cfg.TEST.DETECTIONS_PER_IMAGE}, FORMAT={cfg.INPUT.FORMAT})",
          flush=True)

    images, all_gt, all_dets = [], [], []
    times = []
    n_tree_exported = 0
    for tile_id, meta_path in enumerate(metas):
        stem = meta_path.name.replace("_meta.json", "")
        tif_path = holdout_dir / f"{stem}.tif"
        if not tif_path.exists():
            print(f"  ! missing tif for {stem}, skipping")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        H, W = int(meta["height"]), int(meta["width"])
        images.append({"id": tile_id, "file_name": tif_path.name,
                       "height": H, "width": W})

        geo_out = output_dir / f"{stem}_canopyai.geojson"

        t1 = time.time()
        bgr, h2, w2 = read_tile_bgr(tif_path)
        classes, scores, masks = infer_tile(predictor, bgr)
        times.append(time.time() - t1)

        all_gt.extend(gt_anns_from_meta(meta, tile_id, H, W))
        all_dets.extend(dets_from_inference(classes, scores, masks, tile_id))

        if not (skip_existing and geo_out.exists()):
            n = export_tree_geojson(geo_out, classes, scores, masks)
            n_tree_exported += n

        n_tree = int((classes == TREE_D2_CLASS).sum())
        n_can = int((classes == 0).sum())
        avg = np.mean(times)
        eta = avg * (len(metas) - tile_id - 1)
        print(f"  [{tile_id + 1}/{len(metas)}] {stem}: "
              f"{len(classes)} inst (tree={n_tree} canopy={n_can}) "
              f"{times[-1]:.1f}s  eta {eta / 60:.1f}m", flush=True)

    # ── Per-category COCO eval ──
    print("\nRunning per-category COCO eval (segm, IoU 0.5, "
          f"maxDets {max_dets})...", flush=True)
    res = coco_eval_per_category(images, all_gt, all_dets, max_dets=max_dets)

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else "n/a"

    print("\n" + "=" * 56)
    print("RESTOR tcd-mask-rcnn-r50  —  instance AP50 (segm)")
    print("=" * 56)
    print(f"  n_tiles            : {len(images)}")
    print(f"  n_gt anns          : {len(all_gt)}")
    print(f"  n_predictions      : {len(all_dets)}")
    print(f"  tree crowns export : {n_tree_exported}  -> {output_dir}")
    print("-" * 56)
    print(f"  AP-canopy (cat 1)  : {fmt(res.get(1))}")
    print(f"  AP-tree   (cat 2)  : {fmt(res.get(2))}")
    print(f"  AP50 overall (mean): {fmt(res.get('overall'))}   "
          f"(sanity target ~0.432)")
    print("=" * 56)

    # Persist a small summary alongside the geojsons.
    summary = {
        "model": HF_REPO,
        "device": device,
        "tta": tta,
        "max_dets": max_dets,
        "n_tiles": len(images),
        "n_gt_anns": len(all_gt),
        "n_predictions": len(all_dets),
        "n_tree_exported": n_tree_exported,
        "AP_canopy_cat1": res.get(1),
        "AP_tree_cat2": res.get(2),
        "AP50_overall_mean": res.get("overall"),
    }
    (output_dir / "restor_ap_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nSummary written: {output_dir / 'restor_ap_summary.json'}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Restor tcd-mask-rcnn-r50 per-category AP on the OAM-TCD "
                    "holdout (local CPU).")
    p.add_argument("--holdout-dir", default=str(DEFAULT_HOLDOUT_DIR),
                   help=f"val split dir (default: {DEFAULT_HOLDOUT_DIR})")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                   help=f"where geojsons + summary go (default: {DEFAULT_OUTPUT_DIR})")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--tiles", type=int, default=None,
                   help="Run on only the first N tiles (smoke test).")
    g.add_argument("--limit", type=int, default=None,
                   help="Alias for --tiles.")
    p.add_argument("--device", default="cpu",
                   help="torch device (default cpu; MPS coverage is patchy).")
    p.add_argument("--max-dets", type=int, default=512,
                   help="COCOeval detections-per-image cap (Restor uses 512).")
    p.add_argument("--score-thresh", type=float, default=None,
                   help="Override ROI_HEADS.SCORE_THRESH_TEST (default: keep "
                        "Restor's 0.2).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Do not overwrite an existing per-tile geojson.")
    p.add_argument("--tta", action="store_true",
                   help="Multi-scale test-time augmentation (MIN_SIZES "
                        "768..2048 + flip) — Restor's PUBLISHED 0.432 setting. "
                        "Use a separate --output-dir; CPU-slow.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    limit = args.tiles if args.tiles is not None else args.limit
    return run(args.holdout_dir, args.output_dir, limit=limit,
               device=args.device, max_dets=args.max_dets,
               score_thresh=args.score_thresh,
               skip_existing=args.skip_existing, tta=args.tta)


if __name__ == "__main__":
    raise SystemExit(main())
