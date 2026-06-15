#!/usr/bin/env python3
"""
run_stock_detectron2.py — STOCK COCO-pretrained Detectron2 Mask-RCNN, the
"off-the-shelf floor" zero-shot baseline on our 439-tile OAM-TCD holdout.

Why this exists
---------------
``run_restor.py`` scores Restor's *TCD-trained* Mask-RCNN (the published 0.432
two-category number; their tree head is purpose-built). This file scores the
exact same architecture but with **vanilla COCO weights** straight from the
detectron2 model zoo — a network that has never seen a tree-crown label. It
establishes how far an out-of-the-box instance segmenter gets on tree crowns
with zero domain adaptation, i.e. the floor that any "tree" model must beat.

The headline comparison is against our **tree-only mAP50 = 0.535** and against
Restor's tree-only number, using the *identical* tree-only convention coded in
``phase30/benchmark.py`` (single class "tree" = COCO cat 2 scored as positives,
canopy = COCO cat 1 demoted to an ``iscrowd=1`` ignore region). So this number
is apples-to-apples with the 0.535.

Model facts
-----------
  • ``COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`` from
    ``detectron2.model_zoo`` — same R50-FPN Mask-RCNN backbone/heads as Restor,
    but 80 COCO classes. Config + COCO checkpoint both come from the zoo
    (``get_config_file`` + ``get_checkpoint_url``); the ``.pkl`` weights
    auto-download to the detectron2 cache (~178 MB) on first run.
  • ``MODEL.DEVICE = cpu`` (Apple-Silicon MPS coverage in detectron2 is patchy;
    matches run_restor).
  • ``ROI_HEADS.SCORE_THRESH_TEST = 0.05`` and ``DETECTIONS_PER_IMAGE = 512`` —
    a *low* score floor + the same 512 cap as Restor / benchmark.py, so we keep
    as many candidate masks as the eval can use (maxDets 512). The model-zoo
    default thresh is 0.05 already, but we set it explicitly.
  • ``INPUT.FORMAT`` is whatever the zoo config ships (BGR for these R50 COCO
    models); we read tiles in that format from cfg so we never hard-code it.
    Stock COCO models DO short-edge resize (MIN_SIZE_TEST=800) — we leave the
    zoo default; the point is the off-the-shelf behaviour, not a tuned one.

Class-agnostic ("all detections → tree") scoring — and why
----------------------------------------------------------
None of the 80 COCO classes is "tree". The fair zero-shot question is: treat
the model purely as a *class-agnostic instance proposer* — every predicted
mask, regardless of which COCO class fired, is a candidate tree crown. We remap
**every** detection to category "tree" and keep its (per-class) confidence as
the crown score. This is the most generous reading of an off-the-shelf model
(it gets credit for any well-localised blob that happens to land on a crown).
We deliberately do NOT filter to "vegetation-ish" COCO classes — there is no
principled mapping, and filtering would only lower recall.

  • Should there be a confidence floor? COCOeval is rank-based: AP integrates
    precision over the full recall sweep, so adding low-score detections can
    only *lower or maintain* AP, never inflate it — extra FPs at the bottom of
    the ranking depress precision at high recall. A floor of 0.05 (the zoo
    default) keeps essentially all proposals; raising it would discard recall
    and could only hurt mAP50. So "all detections, no extra floor" is the right,
    most-charitable scoring for a *floor* baseline. (We expose --score-thresh in
    case one wants the precision-at-a-threshold view, but it is not used for the
    headline AP.)

Everything downstream (GT build, RLE encoding, COCOeval params, the
``{stem}_canopyai.geojson`` export schema) mirrors ``run_restor.py`` /
``benchmark.py`` so the numbers line up.

Usage
-----
    OMP_NUM_THREADS=2 phase30/restor_baseline/venv_restor/bin/python \
        phase30/restor_baseline/run_stock_detectron2.py --tiles 4   # smoke
    phase30/restor_baseline/venv_restor/bin/python \
        phase30/restor_baseline/run_stock_detectron2.py             # full 439
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark_results_stock_detectron2"

MODEL_ZOO_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


# ── Model loading ────────────────────────────────────────────────────────────
def build_predictor(device="cpu", score_thresh=0.05, max_dets=512):
    """Build a Detectron2 DefaultPredictor from the STOCK COCO model zoo.

    The COCO weights (.pkl) auto-download to the detectron2 cache on first use.
    We force CPU, a low test score threshold (keep candidate masks), and a 512
    detections-per-image cap to match the eval.
    """
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_ZOO_CONFIG))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_ZOO_CONFIG)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.TEST.DETECTIONS_PER_IMAGE = int(max_dets)
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    return predictor, cfg


# ── Tile inference ───────────────────────────────────────────────────────────
def read_tile(tif_path, input_format):
    """Read an RGB uint8 tile in the predictor's expected channel order.

    detectron2's DefaultPredictor handles RGB->BGR conversion only when given
    an "RGB" cfg.INPUT.FORMAT; the stock R50 COCO configs use "BGR", so we
    deliver BGR (as run_restor does). Returns (img, H, W).
    """
    with rasterio.open(tif_path) as ds:
        rgb = ds.read([1, 2, 3]).transpose(1, 2, 0)  # HWC, RGB
    if str(input_format).upper() == "BGR":
        img = np.ascontiguousarray(rgb[:, :, ::-1])
    else:
        img = np.ascontiguousarray(rgb)
    return img, rgb.shape[0], rgb.shape[1]


def infer_tile(predictor, img):
    """Run the predictor; return (scores, masks_bool) as numpy.

    Class-agnostic: we DISCARD the predicted COCO class entirely and keep every
    instance mask + its score as a candidate tree crown. masks_bool: (N, H, W).
    """
    out = predictor(img)
    inst = out["instances"].to("cpu")
    n = len(inst)
    if n == 0:
        H, W = img.shape[:2]
        return np.empty(0, np.float32), np.empty((0, H, W), bool)
    scores = inst.scores.numpy().astype(np.float32)
    masks = inst.pred_masks.numpy().astype(bool)  # (N, H, W)
    return scores, masks


# ── GT build (TREE-ONLY convention, identical to benchmark.py) ───────────────
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


def gt_anns_tree_only(meta, tile_id, H, W):
    """Build TREE-ONLY COCO GT annotations for one tile (benchmark.py rules).

    Every annotation is remapped to single class category_id=1 ("tree").
    cat==2 (real trees) -> iscrowd=0 positives; cat==1 (canopy blobs) ->
    iscrowd=1 ignore regions so detections inside indistinct canopy are not
    counted as false positives. Segmentations (polygon-list OR RLE-dict) are
    normalised to a single binary mask then compressed RLE (segm task).
    """
    from pycocotools import mask as mask_utils

    anns = []
    n_tree = 0
    for cat, seg in _parse_coco_annotations(meta):
        if isinstance(seg, list) and seg and isinstance(seg[0], list):
            rle_objs = mask_utils.frPyObjects(seg, H, W)
            rle = mask_utils.merge(rle_objs)
        elif isinstance(seg, dict) and "counts" in seg:
            r = dict(seg)
            if isinstance(r.get("counts"), str):
                r["counts"] = r["counts"].encode("ascii")
            rle = (mask_utils.frPyObjects(r, H, W)
                   if isinstance(r.get("counts"), list) else r)
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
            "category_id": 1,                      # single class: "tree"
            "segmentation": {"size": [H, W], "counts": counts},
            "bbox": bbox,
            "area": area,
            "iscrowd": 0 if cat == 2 else 1,       # cat=1 canopy -> ignore
        })
        if cat == 2:
            n_tree += 1
    return anns, n_tree


def dets_class_agnostic(scores, masks, tile_id):
    """COCO detection dicts — EVERY instance remapped to tree (category_id=1)."""
    from pycocotools import mask as mask_utils
    dets = []
    for score, m in zip(scores, masks):
        if not m.any():
            continue
        rle = _rle_from_mask(m)
        bbox = [float(v) for v in mask_utils.toBbox({
            "size": rle["size"], "counts": rle["counts"].encode("ascii")})]
        dets.append({
            "image_id": tile_id,
            "category_id": 1,                      # class-agnostic -> tree
            "segmentation": rle,
            "bbox": bbox,
            "score": float(score),
        })
    return dets


# ── Geojson export (all detections as candidate crowns) ──────────────────────
def export_geojson(out_path, scores, masks):
    """Write each predicted mask as a pixel-coord polygon geojson that
    ``phase30/benchmark._load_predictions`` can read (FeatureCollection,
    Polygon geometry, numeric ``score`` property). Class-agnostic: every
    instance is exported."""
    import rasterio.features
    from shapely.geometry import shape
    from shapely.validation import make_valid

    features = []
    fid = 0
    for score, m in zip(scores, masks):
        if not m.any():
            continue
        best = None
        for geom, val in rasterio.features.shapes(m.astype(np.uint8), mask=m):
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


# ── Tree-only COCO eval (identical params to benchmark.py) ───────────────────
def coco_map50_tree_only(images, gt_anns, dets, max_dets=512):
    """pycocotools COCOeval (segm, IoU 0.5, areaRng all, maxDets given),
    single class "tree". Returns (mAP50, AR). Mirrors benchmark._coco_map50:
    take the largest-maxDets slice (do NOT average over the maxDets axis)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not gt_anns or not dets:
        return None, None

    gt_dict = {
        "images": images,
        "annotations": [dict(a, id=i + 1) for i, a in enumerate(gt_anns)],
        "categories": [{"id": 1, "name": "tree"}],
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
        ev.evaluate()
        ev.accumulate()

    prec = ev.eval["precision"][:, :, :, :, -1]   # largest maxDets slice
    rec = ev.eval["recall"][:, :, :, -1]
    valid_p = prec[prec > -1]
    valid_r = rec[rec > -1]
    map50 = float(valid_p.mean()) if valid_p.size else None
    ar = float(valid_r.mean()) if valid_r.size else None
    return map50, ar


# ── Orchestration ────────────────────────────────────────────────────────────
def run(holdout_dir, output_dir, limit=None, device="cpu",
        max_dets=512, score_thresh=0.05, skip_existing=False):
    holdout_dir = Path(holdout_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metas = sorted(holdout_dir.glob("*_meta.json"))
    if not metas:
        print(f"ERROR: no *_meta.json under {holdout_dir}", file=sys.stderr)
        return 1
    n_total = len(metas)
    if limit is not None:
        metas = metas[:limit]

    print(f"Holdout dir : {holdout_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Tiles       : {len(metas)}"
          + (f"  (limited from {n_total})" if limit is not None else ""))
    print(f"Model       : STOCK {MODEL_ZOO_CONFIG} (COCO-pretrained)")
    print(f"Device      : {device}   maxDets={max_dets}  "
          f"score_thresh={score_thresh}")
    print("Building predictor (downloading COCO weights on first run)...",
          flush=True)
    t0 = time.time()
    predictor, cfg = build_predictor(device=device, score_thresh=score_thresh,
                                     max_dets=max_dets)
    input_format = cfg.INPUT.FORMAT
    print(f"  predictor ready in {time.time() - t0:.1f}s  "
          f"(NUM_CLASSES={cfg.MODEL.ROI_HEADS.NUM_CLASSES}, "
          f"SCORE_THRESH={cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}, "
          f"DETS={cfg.TEST.DETECTIONS_PER_IMAGE}, FORMAT={input_format}, "
          f"MIN_SIZE_TEST={cfg.INPUT.MIN_SIZE_TEST})", flush=True)

    images, all_gt, all_dets = [], [], []
    times = []
    n_pred_exported = 0
    n_gt_tree_total = 0
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
        img, _, _ = read_tile(tif_path, input_format)
        scores, masks = infer_tile(predictor, img)
        times.append(time.time() - t1)

        gt, n_tree = gt_anns_tree_only(meta, tile_id, H, W)
        all_gt.extend(gt)
        n_gt_tree_total += n_tree
        all_dets.extend(dets_class_agnostic(scores, masks, tile_id))

        if not (skip_existing and geo_out.exists()):
            n_pred_exported += export_geojson(geo_out, scores, masks)

        avg = np.mean(times)
        eta = avg * (len(metas) - tile_id - 1)
        print(f"  [{tile_id + 1}/{len(metas)}] {stem}: "
              f"{len(scores)} inst (gt_tree={n_tree}) "
              f"{times[-1]:.1f}s  eta {eta / 60:.1f}m", flush=True)

    # ── Tree-only COCO eval ──
    print(f"\nRunning tree-only COCO eval (segm, IoU 0.5, maxDets {max_dets}, "
          "class-agnostic dets -> tree)...", flush=True)
    map50, ar = coco_map50_tree_only(images, all_gt, all_dets, max_dets=max_dets)

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else "n/a"

    print("\n" + "=" * 60)
    print("STOCK COCO Mask-RCNN R50-FPN  —  tree-only mAP50 (segm)")
    print("=" * 60)
    print(f"  model              : {MODEL_ZOO_CONFIG}")
    print(f"  n_tiles            : {len(images)}")
    print(f"  n_gt tree anns     : {n_gt_tree_total}")
    print(f"  n_predictions      : {len(all_dets)}  (all -> tree)")
    print(f"  crowns exported    : {n_pred_exported}  -> {output_dir}")
    print("-" * 60)
    print(f"  tree-only mAP50    : {fmt(map50)}   (vs ours 0.535)")
    print(f"  tree-only AR@{max_dets:<4}  : {fmt(ar)}")
    print("=" * 60)

    summary = {
        "model": MODEL_ZOO_CONFIG,
        "pretrained": "COCO (stock model zoo, NOT TCD-trained)",
        "scoring": "class-agnostic: all detections remapped to category tree",
        "device": device,
        "max_dets": max_dets,
        "score_thresh": score_thresh,
        "n_tiles": len(images),
        "n_gt_tree_anns": n_gt_tree_total,
        "n_predictions": len(all_dets),
        "n_crowns_exported": n_pred_exported,
        "tree_only_mAP50": map50,
        "tree_only_AR": ar,
    }
    (output_dir / "stock_detectron2_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nSummary written: {output_dir / 'stock_detectron2_summary.json'}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="STOCK COCO-pretrained Detectron2 Mask-RCNN tree-only mAP50 "
                    "on the OAM-TCD holdout (local CPU, zero-shot floor).")
    p.add_argument("--holdout-dir", default=str(DEFAULT_HOLDOUT_DIR),
                   help=f"val split dir (default: {DEFAULT_HOLDOUT_DIR})")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                   help=f"geojsons + summary (default: {DEFAULT_OUTPUT_DIR})")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--tiles", type=int, default=None,
                   help="Run on only the first N tiles (smoke test).")
    g.add_argument("--limit", type=int, default=None, help="Alias for --tiles.")
    p.add_argument("--device", default="cpu",
                   help="torch device (default cpu; MPS coverage is patchy).")
    p.add_argument("--max-dets", type=int, default=512,
                   help="COCOeval detections-per-image cap (default 512, "
                        "matches benchmark.py apples-to-apples).")
    p.add_argument("--score-thresh", type=float, default=0.05,
                   help="ROI_HEADS.SCORE_THRESH_TEST (default 0.05, the zoo "
                        "default — keeps ~all candidate masks).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Do not overwrite an existing per-tile geojson.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    limit = args.tiles if args.tiles is not None else args.limit
    return run(args.holdout_dir, args.output_dir, limit=limit,
               device=args.device, max_dets=args.max_dets,
               score_thresh=args.score_thresh, skip_existing=args.skip_existing)


if __name__ == "__main__":
    raise SystemExit(main())
