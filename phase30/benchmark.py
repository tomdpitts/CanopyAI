#!/usr/bin/env python3
"""
benchmark.py — Restor-comparable evaluation on the OAM-TCD holdout split.

Computes the two metric flavours reported in the Restor TCD paper:

  • Binary semantic segmentation (micro-averaged pixel IoU / F1 / Acc)
        comparable to Restor Table 1 (UNet / SegFormer holdout column).
  • Instance mAP50 via pycocotools segm task (cat=2 trees only)
        comparable to Restor's Mask-RCNN R50 baseline (mAP50 = 43.22).

Reuses the existing per-tile inference contract from benchmark_tcd.py:
each model produces  {stem}_canopyai.geojson  in pixel space with one
of the known confidence columns (deepforest_score / score / etc.).

Inference is delegated to:
  • foxtrot.py            — phase30 .pth checkpoints
  • infer_detectree2.py   — detectree2 baseline
  • infer_segformer.py    — Restor SegFormer mit-b5

Usage:
    python phase30/benchmark.py \\
        --models phase30_L4.pth detectree2 segformer \\
        --names  phase30_L4    detectree2 segformer_b5 \\
        --output-root benchmark_results_holdout \\
        --skip-existing
"""

import argparse
import csv
import datetime
import io
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
from shapely.geometry import Polygon, shape
from shapely.validation import make_valid

# Project root: phase30/ sibling scripts (foxtrot.py etc.) live one level up.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_HOLDOUT_DIR = REPO_ROOT / "data" / "tcd" / "images" / "data" / "tcd" / "val"


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Restor-comparable OAM-TCD holdout benchmark."
    )
    p.add_argument("--models", nargs="+", required=True,
                   help="Specifiers: 'detectree2', 'segformer', or path to .pth checkpoint.")
    p.add_argument("--names", nargs="+", required=True,
                   help="Display name for each model (same length as --models).")
    p.add_argument("--holdout-dir", default=str(DEFAULT_HOLDOUT_DIR),
                   help=f"OAM-TCD val split with *.tif + *_meta.json (default: {DEFAULT_HOLDOUT_DIR}).")
    p.add_argument("--output-root", default="benchmark_results_holdout",
                   help="Where per-model prediction folders + summary files are written.")
    p.add_argument("--shadow-model",
                   default="solar/shadow_regression/output/shadow_model_combined_best.pth")
    p.add_argument("--abs-luma-max", type=float, default=None,
                   help="Shadow map luma ceiling passed to foxtrot (None → foxtrot default).")
    p.add_argument("--df-confidence", type=float, default=None,
                   help="DeepForest detection confidence threshold passed to foxtrot "
                        "(None → foxtrot default of 0.35). Lower it (e.g. 0.05) to "
                        "diagnose models that under-predict.")
    p.add_argument("--df-tile-overlap", type=float, default=None,
                   help="DeepForest tile overlap fraction. None → foxtrot default (0.5).")
    p.add_argument("--bbox-pad", type=float, default=None,
                   help="Pad each detection bbox by this fraction before "
                        "passing to SAM. None → foxtrot default (0.0).")
    p.add_argument("--skip-nms", action="store_true",
                   help="Skip foxtrot's NMS + containment stages entirely. "
                        "Diagnostic only — usually produces huge FP counts.")
    p.add_argument("--containment-threshold", type=float, default=None,
                   help="Box-level containment: drop box ≥this fraction inside "
                        "a larger box. None → foxtrot default (0.8). 0 disables.")
    p.add_argument("--poly-containment-threshold", type=float, default=None,
                   help="Polygon-level containment: drop polygon ≥this fraction "
                        "inside a larger polygon. None → foxtrot default (0.9). "
                        "0 disables.")
    p.add_argument("--reranker-checkpoint", default=None,
                   help="Path to a CNN reranker checkpoint (.pt) saved by "
                        "phase30/cnn_reranker.py --save-checkpoint.  Passed "
                        "through to foxtrot so each polygon's deepforest_score "
                        "is replaced with the reranker's TP-probability at "
                        "inference time.")
    p.add_argument("--sam-model", default=None,
                   choices=["vit_b", "vit_l", "vit_h"],
                   help="SAM backbone passed to foxtrot (default: foxtrot's "
                        "vit_b).  Pair with --sam-checkpoint.")
    p.add_argument("--sam-checkpoint", default=None,
                   help="Path to the SAM .pth checkpoint matching --sam-model "
                        "(default: foxtrot's sam_vit_b_01ec64.pth).")
    p.add_argument("--tiles", nargs="+", default=None,
                   help="Restrict to these tile stems (e.g. tcd_val_tile_0 tcd_val_tile_1).")
    p.add_argument("--tiles-file", default=None,
                   help="Path to a file with one tile stem per line "
                        "(use when --tiles would be too long for the shell).")
    p.add_argument("--skip-inference", action="store_true",
                   help="Skip Step 1; only re-evaluate existing geojsons under --output-root.")
    p.add_argument("--skip-existing", action="store_true",
                   help="During inference, skip tiles whose geojson is already on disk.")
    p.add_argument("--pred-score-thresh", type=float, default=0.0,
                   help="Drop predictions below this confidence before metric computation.")
    p.add_argument("--max-dets", type=int, default=512,
                   help="Detections-per-image cap for pycocotools mAP (maxDets[2] "
                        "slice).  Default 512 to match Restor's Mask-RCNN "
                        "(DETECTIONS_PER_IMAGE=512 / paper: 'increased predictions to 512').")
    p.add_argument("--max-boxes-sam", type=int, default=0,
                   help="Cap detections fed to SAM to top-N by score (forwarded to "
                        "foxtrot --max_boxes_sam; 0=off). Bounds SAM runtime at low "
                        "--df-confidence; lossless for mAP up to --max-dets.")
    p.add_argument("--df-tta", action="store_true",
                   help="Enable DeepForest multi-scale test-time augmentation "
                        "(forwarded to foxtrot.py --df_tta).")
    p.add_argument("--df-tta-scales", default=None,
                   help="Comma-separated resize factors for --df-tta "
                        "(forwarded to foxtrot.py --df_tta_scales; default there "
                        "is '0.75,1.0,1.25,1.5').")
    return p.parse_args()


def model_type(spec):
    s = spec.lower()
    if s == "detectree2":
        return "detectree2"
    if s == "segformer":
        return "segformer"
    return "checkpoint"


# ── Inference (subprocess wrappers, mirroring benchmark_tcd.py) ───────────────

def _progress_suffix(times, total):
    elapsed = times[-1]
    avg = sum(times) / len(times)
    remaining = avg * (total - len(times))
    eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining)
    return f"  {elapsed:5.1f}s | avg {avg:5.1f}s | ETA {eta:%H:%M:%S}"


def run_foxtrot(model_spec, mtype, image_path, out_dir, shadow_model,
                abs_luma_max=None, df_confidence=None,
                df_tile_overlap=None, bbox_pad=None, skip_nms=False,
                containment_threshold=None, poly_containment_threshold=None,
                reranker_checkpoint=None,
                sam_model=None, sam_checkpoint=None,
                df_tta=False, df_tta_scales=None, max_boxes_sam=0):
    cmd = [sys.executable, str(REPO_ROOT / "foxtrot.py"),
           "--image_path", str(image_path),
           "--output_dir", str(out_dir),
           "--shadow_model", str(shadow_model),
           "--no_viz"]
    if max_boxes_sam and int(max_boxes_sam) > 0:
        cmd += ["--max_boxes_sam", str(int(max_boxes_sam))]
    if df_tta:
        cmd += ["--df_tta"]
        if df_tta_scales is not None:
            cmd += ["--df_tta_scales", str(df_tta_scales)]
    if mtype == "checkpoint":
        cmd += ["--deepforest_model", model_spec]
    if abs_luma_max is not None:
        cmd += ["--abs_luma_max", str(abs_luma_max)]
    if df_confidence is not None:
        cmd += ["--deepforest_confidence", str(df_confidence)]
    if df_tile_overlap is not None:
        cmd += ["--df_tile_overlap", str(df_tile_overlap)]
    if bbox_pad is not None:
        cmd += ["--bbox_pad", str(bbox_pad)]
    if skip_nms:
        cmd += ["--skip_nms"]
    if containment_threshold is not None:
        cmd += ["--containment_threshold", str(containment_threshold)]
    if poly_containment_threshold is not None:
        cmd += ["--poly_containment_threshold", str(poly_containment_threshold)]
    if reranker_checkpoint is not None:
        cmd += ["--reranker_checkpoint", str(reranker_checkpoint)]
    if sam_model is not None:
        cmd += ["--sam_model", str(sam_model)]
    if sam_checkpoint is not None:
        cmd += ["--sam_checkpoint", str(sam_checkpoint)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"      ⚠  foxtrot failed: {r.stderr[-300:]}")
        return False
    return True


def run_segformer(image_path, out_dir):
    cmd = [sys.executable, str(REPO_ROOT / "infer_segformer.py"),
           "--image_path", str(image_path),
           "--output_dir", str(out_dir)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"      ⚠  infer_segformer failed: {r.stderr[-300:]}")
        return False
    return True


def run_detectree2_one(image_path, out_dir):
    cmd = [sys.executable, str(REPO_ROOT / "infer_detectree2.py"),
           "--image_path", str(image_path),
           "--output_dir", str(out_dir),
           "--weights", "finetuned"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"      ⚠  infer_detectree2 failed: {r.stderr[-300:]}")
        return False
    return True


def run_inference(model_spec, mtype, holdout_dir, out_dir, shadow_model,
                  abs_luma_max=None, df_confidence=None,
                  df_tile_overlap=None, bbox_pad=None, skip_nms=False,
                  containment_threshold=None, poly_containment_threshold=None,
                  reranker_checkpoint=None,
                  sam_model=None, sam_checkpoint=None,
                  skip_existing=False, tile_filter=None,
                  df_tta=False, df_tta_scales=None, max_boxes_sam=0):
    tifs = sorted(Path(holdout_dir).glob("*.tif"))
    if tile_filter is not None:
        tifs = [t for t in tifs if t.stem in tile_filter]
    pending = [t for t in tifs
               if not (skip_existing and (Path(out_dir) / f"{t.stem}_canopyai.geojson").exists())]

    ok, skipped, times = 0, 0, []
    for tif in tifs:
        out_file = Path(out_dir) / f"{tif.stem}_canopyai.geojson"
        if skip_existing and out_file.exists():
            skipped += 1
            ok += 1
            continue
        print(f"    {tif.name} ... ", end="", flush=True)
        t0 = time.monotonic()
        if mtype == "segformer":
            success = run_segformer(tif, out_dir)
        elif mtype == "detectree2":
            success = run_detectree2_one(tif, out_dir)
        else:
            success = run_foxtrot(model_spec, mtype, tif, out_dir,
                                  shadow_model, abs_luma_max=abs_luma_max,
                                  df_confidence=df_confidence,
                                  df_tile_overlap=df_tile_overlap,
                                  bbox_pad=bbox_pad,
                                  skip_nms=skip_nms,
                                  containment_threshold=containment_threshold,
                                  poly_containment_threshold=poly_containment_threshold,
                                  reranker_checkpoint=reranker_checkpoint,
                                  sam_model=sam_model,
                                  sam_checkpoint=sam_checkpoint,
                                  df_tta=df_tta, df_tta_scales=df_tta_scales,
                                  max_boxes_sam=max_boxes_sam)
        times.append(time.monotonic() - t0)
        print(("✓" if success else "✗") + _progress_suffix(times, len(pending)))
        ok += success

    if skipped:
        print(f"  {skipped} tiles skipped (already exist)")
    print(f"  {ok}/{len(tifs)} successful")
    return ok > 0


# ── GT parsing (pixel-space, no world transform needed) ───────────────────────

def _parse_coco_annotations(meta):
    """
    Yield (category_id, segmentation_dict_or_polygon_list) from a meta.json.
    Strings (legacy JSON-stringified field) are parsed.
    """
    coco = meta.get("coco_annotations", [])
    if isinstance(coco, str):
        coco = json.loads(coco)
    for ann in coco or []:
        seg = ann.get("segmentation")
        if not seg:
            continue
        yield int(ann.get("category_id", 2)), seg, ann.get("bbox"), ann.get("area")


def _seg_to_polygons(seg, height, width):
    """
    Convert a COCO segmentation into a list of shapely Polygons in pixel space.
    Handles both polygon-list form (cat=2 trees) and RLE-dict form (cat=1 canopy).
    """
    from pycocotools import mask as mask_utils

    polys = []
    if isinstance(seg, list) and seg and isinstance(seg[0], list):
        for ring in seg:
            try:
                coords = np.asarray(ring, dtype=np.float32).reshape(-1, 2)
                if len(coords) < 3:
                    continue
                p = Polygon(coords)
                if not p.is_valid:
                    p = make_valid(p)
                if p.is_valid and p.area > 0:
                    polys.append(p)
            except Exception:
                continue
    elif isinstance(seg, dict) and "counts" in seg:
        try:
            rle = mask_utils.frPyObjects(seg, height, width)
            mask = mask_utils.decode(rle)
            if mask.ndim == 3:
                mask = mask[..., 0]
            for geom, val in rasterio.features.shapes(mask.astype(np.uint8), mask > 0):
                if val != 1:
                    continue
                p = shape(geom)
                if not p.is_valid:
                    p = make_valid(p)
                if p.is_valid and p.area > 0:
                    polys.append(p)
        except Exception:
            pass
    return polys


def _rasterize(polys, height, width):
    if not polys:
        return np.zeros((height, width), dtype=bool)
    shapes = [(p, 1) for p in polys if p is not None and not p.is_empty]
    if not shapes:
        return np.zeros((height, width), dtype=bool)
    mask = rasterio.features.rasterize(
        shapes, out_shape=(height, width), fill=0, dtype=np.uint8, all_touched=False,
    )
    return mask.astype(bool)


def _rle_encode(mask_bool, height, width):
    """Encode a HxW boolean mask as a COCO compressed RLE."""
    from pycocotools import mask as mask_utils
    arr = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = mask_utils.encode(arr)
    # pycocotools returns bytes in 'counts' — JSON-incompatible; decode for transport.
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


# ── Prediction parsing ────────────────────────────────────────────────────────

_SCORE_COLS = ("deepforest_score", "score", "Confidence_score",
               "Confidence", "confidence")


def _load_predictions(pred_path, height, width, score_thresh):
    """
    Read {stem}_canopyai.geojson and return a list of (shapely_polygon, score)
    in pixel space. Defensive against the rare case of stored world coords
    (only foxtrot in TCD-raw mode would do that; val tiles are pixel-space).
    """
    if not pred_path.exists():
        return []
    gdf = gpd.read_file(str(pred_path))
    if gdf.empty:
        return []

    # Score column
    sc = next((c for c in _SCORE_COLS if c in gdf.columns), None)
    scores = gdf[sc].astype(float).values if sc else np.ones(len(gdf))

    out = []
    for geom, score in zip(gdf.geometry, scores):
        if geom is None or geom.is_empty:
            continue
        if score < score_thresh:
            continue
        g = geom if geom.is_valid else make_valid(geom)
        if g.is_empty:
            continue
        out.append((g, float(score)))
    return out


# ── Per-tile worker: rasterise + per-tile counts + COCO dets ──────────────────

def _eval_tile_worker(args):
    """
    One tile's contribution.
    Returns:
      tile_id, stem, n_gt_tree, n_pred,
      tp, fp, fn, tn,                              # binary pixel counts
      gt_anns,                                     # list of coco GT anns (cat=2 only)
      pred_dets                                    # list of coco detections
    """
    (tile_id, stem, meta_path, pred_path, score_thresh) = args
    with open(meta_path) as f:
        meta = json.load(f)
    H, W = int(meta["height"]), int(meta["width"])

    # Build GT polygon lists per metric.
    # For COCO: cat=2 (trees) are normal positives (iscrowd=0); cat=1 (canopy
    # blobs) are emitted as iscrowd=1 ignore regions so tree predictions
    # falling inside indistinct canopy don't get counted as FPs.
    gt_polys_all = []          # cat=1 ∪ cat=2 — for binary semantic mask
    gt_anns      = []          # list of dicts for COCO (trees + canopy-as-ignore)
    n_gt_tree    = 0
    for cat, seg, bbox, area in _parse_coco_annotations(meta):
        polys = _seg_to_polygons(seg, H, W)
        if not polys:
            continue
        gt_polys_all.extend(polys)

        m = _rasterize(polys, H, W)
        if not m.any():
            continue
        rle = _rle_encode(m, H, W)
        ys, xs = np.where(m)
        y0, x0 = int(ys.min()), int(xs.min())
        y1, x1 = int(ys.max()) + 1, int(xs.max()) + 1
        gt_anns.append({
            "image_id":     tile_id,
            "category_id":  1,                       # single class: "tree"
            "segmentation": rle,
            "bbox":         [x0, y0, x1 - x0, y1 - y0],
            "area":         float(m.sum()),
            "iscrowd":      0 if cat == 2 else 1,    # cat=1 canopy → ignore region
        })
        if cat == 2:
            n_gt_tree += 1

    gt_mask_all = _rasterize(gt_polys_all, H, W)

    # Predictions
    preds = _load_predictions(pred_path, H, W, score_thresh)
    pred_polys = [g for g, _ in preds]
    pred_mask = _rasterize(pred_polys, H, W)

    # Binary pixel confusion counts
    gt_flat   = gt_mask_all.ravel()
    pred_flat = pred_mask.ravel()
    tp = int(np.count_nonzero(pred_flat & gt_flat))
    fp = int(np.count_nonzero(pred_flat & ~gt_flat))
    fn = int(np.count_nonzero(~pred_flat & gt_flat))
    tn = int(pred_flat.size - tp - fp - fn)

    # COCO detections (one per predicted polygon, cat=1 in remapped space)
    pred_dets = []
    for poly, score in preds:
        pmask = _rasterize([poly], H, W)
        if not pmask.any():
            continue
        rle = _rle_encode(pmask, H, W)
        ys, xs = np.where(pmask)
        y0, x0 = int(ys.min()), int(xs.min())
        y1, x1 = int(ys.max()) + 1, int(xs.max()) + 1
        pred_dets.append({
            "image_id":     tile_id,
            "category_id":  1,
            "segmentation": rle,
            "bbox":         [x0, y0, x1 - x0, y1 - y0],
            "score":        float(score),
        })

    return {
        "tile_id":   tile_id,
        "stem":      stem,
        "H": H, "W": W,
        "n_gt_all":  len(gt_polys_all),
        "n_gt_tree": n_gt_tree,
        "n_pred":    len(pred_polys),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "gt_anns":   gt_anns,
        "pred_dets": pred_dets,
    }


# ── COCO mAP50 ────────────────────────────────────────────────────────────────

def _coco_map50(images, gt_anns, dets, max_dets=1000):
    """
    Run pycocotools COCOeval segm with IoU=0.5 only.
    Returns (mAP50, AR_at_maxdets).  Returns (None, None) if there is no GT or dets.

    max_dets sets the detections-per-image cap (the COCOeval maxDets[2] slice
    we read).  Default 1000.  Restor's Mask-RCNN reference uses 512
    (DETECTIONS_PER_IMAGE=512) — set max_dets=512 for an apples-to-apples
    comparison with their reported mAP50=43.22.
    """
    if not gt_anns or not dets:
        return None, None
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_dict = {
        "images":      images,
        "annotations": [dict(a, id=i + 1) for i, a in enumerate(gt_anns)],
        "categories":  [{"id": 1, "name": "tree"}],
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
        ev.params.areaRng     = [[0, 1e10]]
        ev.params.areaRngLbl  = ["all"]
        ev.evaluate()
        ev.accumulate()

    # precision shape: [T, R, K, A, M].  Take the largest-maxDets slice (last M).
    # Averaging across maxDets caps recall at 1/27 etc. on dense tiles and
    # turns a perfect-prediction self-test into mAP=0.47 instead of 1.0.
    prec = ev.eval["precision"][:, :, :, :, -1]
    rec  = ev.eval["recall"][:, :, :, -1]
    valid_p = prec[prec > -1]
    valid_r = rec[rec > -1]
    map50 = float(valid_p.mean()) if valid_p.size else None
    ar    = float(valid_r.mean()) if valid_r.size else None
    return map50, ar


# ── Evaluation orchestration ──────────────────────────────────────────────────

def evaluate_model(name, out_dir, holdout_dir, score_thresh, tile_filter=None,
                   max_dets=1000):
    holdout_dir = Path(holdout_dir)
    out_dir = Path(out_dir)
    metas = sorted(holdout_dir.glob("*_meta.json"))
    if tile_filter is not None:
        metas = [m for m in metas
                 if m.name.replace("_meta.json", "") in tile_filter]

    # Auto-scope to tiles that have a prediction file. Missing geojsons would
    # otherwise contribute pure FN and tank the metric — useless during a
    # partial run. On a completed run every tile has a geojson so this is a no-op.
    n_holdout = len(metas)
    metas = [m for m in metas
             if (out_dir / f"{m.name.replace('_meta.json', '')}_canopyai.geojson").exists()]
    if not metas:
        print(f"    ⚠  no predictions found in {out_dir}")
        return None
    if len(metas) < n_holdout:
        print(f"    Evaluating {len(metas)}/{n_holdout} tiles "
              f"({n_holdout - len(metas)} missing geojsons skipped)")

    worker_args = []
    images = []
    for i, mp in enumerate(metas):
        stem = mp.name.replace("_meta.json", "")
        pred_path = out_dir / f"{stem}_canopyai.geojson"
        worker_args.append((i, stem, str(mp), pred_path, score_thresh))
        # populated with H,W after worker runs; pre-fill with placeholder dims
        images.append({"id": i, "file_name": f"{stem}.tif",
                       "width": 0, "height": 0})

    # Run rasterisation + counts in parallel
    print(f"    Rasterising + counting on {len(worker_args)} tiles...")
    t0 = time.monotonic()
    results = []
    with ProcessPoolExecutor() as ex:
        for i, r in enumerate(ex.map(_eval_tile_worker, worker_args, chunksize=4)):
            results.append(r)
            images[r["tile_id"]]["width"]  = r["W"]
            images[r["tile_id"]]["height"] = r["H"]
            if (i + 1) % 50 == 0 or (i + 1) == len(worker_args):
                avg = (time.monotonic() - t0) / (i + 1)
                remaining = avg * (len(worker_args) - (i + 1))
                eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining)
                print(f"      {i+1}/{len(worker_args)}  avg {avg:4.2f}s/tile  ETA {eta:%H:%M:%S}")

    # Binary semantic-seg micro counts
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    tn = sum(r["tn"] for r in results)

    # Match Restor TCD paper Table 1 conventions exactly. They cast binary canopy
    # segmentation as 2-class multiclass and use torchmetrics with:
    #   F1Score (task=multiclass, average="none")  → reports F1_tree (positive-class Dice)
    #   JaccardIndex (task=multiclass)             → reports macro IoU = (IoU_bg + IoU_tree)/2
    #   Accuracy (task=multiclass, average="none") → reports Acc_tree = TP/(TP+FN) = tree recall
    # See src/tcd_pipeline/models/segmentationmodule.py in github.com/Restor-Foundation/tcd.
    iou_tree = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    iou_bg   = tn / (tn + fp + fn) if (tn + fp + fn) else 0.0
    macro_iou   = 0.5 * (iou_tree + iou_bg)
    f1_tree     = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    tree_recall = tp / (tp + fn) if (tp + fn) else 0.0
    pixel_acc   = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    # COCO mAP50 (cat=2 trees only)
    gt_anns = [a for r in results for a in r["gt_anns"]]
    dets    = [d for r in results for d in r["pred_dets"]]
    n_pred_total = sum(r["n_pred"] for r in results)
    n_gt_tree    = sum(r["n_gt_tree"] for r in results)

    map50, ar = _coco_map50(images, gt_anns, dets, max_dets=max_dets)

    return {
        "n_tiles": len(results),
        "n_gt_tree": n_gt_tree,
        "n_pred": n_pred_total,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        # Restor-paper-comparable columns:
        "macro_iou":   macro_iou,
        "f1_tree":     f1_tree,
        "tree_recall": tree_recall,
        # Extras (not in Restor table) for transparency:
        "iou_tree":    iou_tree,
        "iou_bg":      iou_bg,
        "pixel_acc":   pixel_acc,
        "map50": map50, "ar1000": ar,
        "per_tile": [
            {"stem": r["stem"],
             "n_gt_all": r["n_gt_all"], "n_gt_tree": r["n_gt_tree"],
             "n_pred": r["n_pred"],
             "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"]}
            for r in results
        ],
    }


# ── Output ────────────────────────────────────────────────────────────────────

def _fmt(v, w=6, prec=3):
    if v is None:
        return "—".rjust(w)
    return f"{v:{w}.{prec}f}"


def print_table(all_results, holdout_dir):
    n_tiles_total = len(sorted(Path(holdout_dir).glob("*_meta.json")))
    bar = "═" * 90
    print()
    print(bar)
    print(f"  BENCHMARK — OAM-TCD holdout ({n_tiles_total} tiles)   columns match Restor TCD paper Table 1")
    print(f"  IoU  = macro JaccardIndex (avg of bg + tree)")
    print(f"  F1   = per-class F1_tree (positive-class Dice)")
    print(f"  Acc  = per-class Accuracy_tree (= tree recall, TP/(TP+FN))")
    print(f"  mAP50, AR@1000 = pycocotools segm task, cat=tree, canopy as iscrowd")
    print(bar)
    hdr = (f"  {'Model':<18}  {'N':>4}  {'IoU':>6}  {'F1':>6}  {'Acc':>6}  "
           f"{'mAP50':>6}  {'AR@1000':>7}  | {'IoU_tree':>8}")
    sep = "  " + "-"*18 + "  " + "-"*4 + "  " + "  ".join(["-"*6]*4) + "  " + "-"*7 + "  | " + "-"*8
    print(hdr); print(sep)
    for name, r in all_results.items():
        if r is None:
            print(f"  {name:<18}  {'—':>4}  {'—':>6}  {'—':>6}  {'—':>6}  "
                  f"{'—':>6}  {'—':>7}  | {'—':>8}")
            continue
        print(f"  {name:<18}  {r['n_tiles']:>4}  "
              f"{_fmt(r['macro_iou'])}  {_fmt(r['f1_tree'])}  {_fmt(r['tree_recall'])}  "
              f"{_fmt(r['map50'])}  {_fmt(r['ar1000'], 7)}  | {_fmt(r['iou_tree'], 8)}")
    print(bar)
    print("  Restor reference (holdout, mit-b5):  IoU=0.876  F1=0.902  Acc=0.890   Mask-RCNN R50 mAP50=0.432")
    print(bar + "\n")


def save_tile_csv(name, per_tile, out_path):
    if not per_tile:
        return
    fieldnames = ["model"] + list(per_tile[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in per_tile:
            w.writerow({"model": name, **row})
    print(f"  📊 per-tile CSV: {out_path}")


def _now_iso():
    """Current local datetime, ISO-8601 with explicit UTC offset."""
    import datetime
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _metrics_block(r):
    if r is None:
        return None
    return {
        "n_tiles":  r["n_tiles"],
        "n_gt_tree": r["n_gt_tree"],
        "n_pred":   r["n_pred"],
        "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"],
        "macro_iou":   r["macro_iou"],
        "f1_tree":     r["f1_tree"],
        "tree_recall": r["tree_recall"],
        "iou_tree":    r["iou_tree"],
        "iou_bg":      r["iou_bg"],
        "pixel_acc":   r["pixel_acc"],
        "map50":   r["map50"],   "ar1000": r["ar1000"],
    }


def _write_summary_provenance(out_dir, args, model_spec, name,
                              started, ended=None, results=None):
    """Write summary.json beside the model's geojsons so the EXACT pipeline that
    produced them is never lost: FULL resolved args (SAM model, reranker, score
    thresholds, ...), the scored results, start/end datetimes (ISO-8601 with
    timezone) and git SHA.  Written provisionally right after inference
    (results/ended pending) and finalised after scoring, so provenance survives
    a crash in between.  A --skip-inference rescore goes to summary_rescore.json
    instead — it must never overwrite the record of how the geojsons were made."""
    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "unknown"
    doc = {
        "name": name,
        "model": str(model_spec),
        "started": started,
        "ended": ended,
        "git_sha": sha,
        "argv": sys.argv,
        "args": vars(args),
        "results": results,
    }
    fname = "summary_rescore.json" if args.skip_inference else "summary.json"
    path = Path(out_dir) / fname
    try:
        path.write_text(json.dumps(doc, indent=2, default=str))
        print(f"  🧾 provenance: {path}")
    except Exception as e:
        print(f"  ⚠️  could not write {fname}: {e}")


def _rerank_out_dir(out_dir, holdout_dir, ensemble, tile_filter=None):
    """In-place rerank: replace `deepforest_score` in each geojson with the CNN
    ensemble's calibrated TP-probability, using the matching holdout tif.

    Used by `--skip-inference --reranker-checkpoint` so geojsons produced WITHOUT
    the reranker (e.g. DF+SAM-only inference offloaded to Modal) can be reranked
    locally before scoring — the same rescoring foxtrot's 3rd stage does, and the
    same as phase30/apply_reranker.py. Idempotent: the reranker scores from the
    image + polygon geometry, not the current score, so re-running is a no-op."""
    import rasterio
    from shapely.geometry import mapping
    out_dir = Path(out_dir); holdout_dir = Path(holdout_dir)
    files = sorted(out_dir.glob("*_canopyai.geojson"))
    if tile_filter is not None:
        files = [f for f in files
                 if f.name.replace("_canopyai.geojson", "") in tile_filter]
    n_ok = 0
    for f in files:
        stem = f.name.replace("_canopyai.geojson", "")
        gdf = gpd.read_file(str(f))
        if gdf.empty or "deepforest_score" not in gdf.columns:
            continue
        tif = holdout_dir / f"{stem}.tif"
        if not tif.exists():
            print(f"    ⚠  rerank: missing tif for {stem}; left unchanged")
            continue
        with rasterio.open(tif) as src:
            arr = src.read([1, 2, 3])
        img = np.transpose(arr, (1, 2, 0))
        if img.dtype != np.uint8:
            mx = max(1, int(img.max()))
            img = (img.astype(np.float32) / mx * 255).astype(np.uint8)
        new_scores = ensemble.predict(img, list(gdf.geometry))
        feats = []
        for i, geom in enumerate(gdf.geometry):
            clean = {}
            for k, v in gdf.iloc[i].items():
                if k == "geometry":
                    continue
                if k == "deepforest_score":
                    clean[k] = float(new_scores[i]); continue
                if hasattr(v, "tolist"):
                    clean[k] = v.tolist()
                elif isinstance(v, (int, float, str, bool)) or v is None:
                    clean[k] = v
                else:
                    clean[k] = str(v)
            feats.append({"type": "Feature", "properties": clean,
                          "geometry": mapping(geom)})
        with open(f, "w") as fh:
            json.dump({"type": "FeatureCollection", "features": feats}, fh)
        n_ok += 1
    print(f"  🔁 reranked {n_ok} geojsons in {out_dir.name}/ (in place)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if len(args.models) != len(args.names):
        print("❌ --models and --names must match in length"); sys.exit(1)

    holdout_dir = Path(args.holdout_dir)
    if not holdout_dir.is_dir():
        print(f"❌ holdout dir not found: {holdout_dir}"); sys.exit(1)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tile_filter = None
    if args.tiles:
        tile_filter = set(args.tiles)
    if args.tiles_file:
        extra = {s.strip() for s in Path(args.tiles_file).read_text().splitlines() if s.strip()}
        tile_filter = (tile_filter | extra) if tile_filter else extra

    # Step 1: inference per model
    started_at = {}  # name -> ISO start time (inference start, or eval start under --skip-inference)
    for spec, name in zip(args.models, args.names):
        mtype = model_type(spec)
        out_dir = output_root / name
        out_dir.mkdir(exist_ok=True)
        started_at[name] = _now_iso()
        if args.skip_inference:
            continue
        print(f"\n{'─'*60}\n  Inference: {name}  [{mtype}]\n{'─'*60}")
        run_inference(spec, mtype, holdout_dir, out_dir, args.shadow_model,
                      abs_luma_max=args.abs_luma_max,
                      df_confidence=args.df_confidence,
                      df_tile_overlap=args.df_tile_overlap,
                      bbox_pad=args.bbox_pad,
                      skip_nms=args.skip_nms,
                      containment_threshold=args.containment_threshold,
                      poly_containment_threshold=args.poly_containment_threshold,
                      reranker_checkpoint=args.reranker_checkpoint,
                      sam_model=args.sam_model,
                      sam_checkpoint=args.sam_checkpoint,
                      skip_existing=args.skip_existing,
                      tile_filter=tile_filter,
                      df_tta=args.df_tta, df_tta_scales=args.df_tta_scales,
                      max_boxes_sam=args.max_boxes_sam)
        # provisional provenance (results pending) — survives a crash during scoring
        _write_summary_provenance(out_dir, args, spec, name, started_at[name])

    # Optional rerank pass: when scoring pre-made geojsons (--skip-inference) that
    # were produced WITHOUT the reranker (e.g. DF+SAM offloaded to Modal), apply the
    # CNN reranker locally before scoring. (When inference runs here, foxtrot already
    # reranks during inference, so this only fires under --skip-inference.)
    rr_ensemble = None
    if args.skip_inference and args.reranker_checkpoint:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from cnn_reranker import load_ensemble
        import torch
        dev = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"  🔁 reranker: {args.reranker_checkpoint} on {dev} "
              f"(rescoring geojsons before scoring)")
        rr_ensemble = load_ensemble(Path(args.reranker_checkpoint), dev)

    # Step 2: evaluation
    all_results = {}
    for spec, name in zip(args.models, args.names):
        out_dir = output_root / name
        if rr_ensemble is not None:
            _rerank_out_dir(out_dir, holdout_dir, rr_ensemble, tile_filter=tile_filter)
        print(f"\n  Evaluating {name} ...")
        res = evaluate_model(name, out_dir, holdout_dir,
                             args.pred_score_thresh, tile_filter=tile_filter,
                             max_dets=args.max_dets)
        all_results[name] = res
        _write_summary_provenance(out_dir, args, spec, name, started_at[name],
                                  ended=_now_iso(),
                                  results={name: _metrics_block(res)})
        if res is not None:
            save_tile_csv(name, res["per_tile"],
                          output_root / f"{name}_holdout_tiles.csv")
            map_str = "—" if res["map50"] is None else f"{res['map50']:.3f}"
            print(f"    IoU(macro)={res['macro_iou']:.3f}  F1(tree)={res['f1_tree']:.3f}  "
                  f"Acc(tree-recall)={res['tree_recall']:.3f}  mAP50={map_str}")

    # Step 3: output
    print_table(all_results, holdout_dir)

    summary = {
        "args": vars(args),
        "results": {name: _metrics_block(r) for name, r in all_results.items()},
    }
    summary_path = output_root / "benchmark_holdout_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  💾 summary: {summary_path}")


if __name__ == "__main__":
    main()
