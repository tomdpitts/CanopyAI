#!/usr/bin/env python3
"""
cnn_reranker.py — Per-polygon CNN reranker.

For each predicted polygon, classify whether it matches a ground-truth
tree at IoU >= 0.5.  The classifier's TP-probability replaces the
upstream detector's confidence score in the geojson, which is then
used as the score input to pycocotools mAP evaluation.

Architecture
------------
  Image crop (96x96 centred on polygon bbox, 20% padding)
   -> ResNet18 (ImageNet-pretrained, fine-tuned end-to-end)
   -> 3-layer MLP head -> TP probability (sigmoid)

Training
--------
  Labels per polygon:
    * TP (1) iff predicted polygon has IoU >= 0.5 with an unmatched
      cat=2 GT under greedy score-sorted matching (same protocol as
      pycocotools COCOeval).
    * Otherwise FP (0).
    * EXCLUDED from training: predictions falling inside a cat=1 canopy
      region at IoP >= 0.5 with no tree match.  This mirrors
      pycocotools' iscrowd ignore at eval time.
  Loss: BCE-with-logits with positive-class weight.
  Optimiser: AdamW with separate LRs (head 1e-3, backbone 1e-4).
  Augmentation: random horizontal/vertical flip + 90-degree rotation.
  Early-stopping: best validation-loss state restored after training.
  Ensemble: train N runs with different random inits, average the
  output probabilities.

Disjoint train/eval
-------------------
  --train-src points at a foxtrot prediction folder over an independent
  set of training tiles (e.g. OAM-TCD train split).
  --src points at the foxtrot prediction folder for the evaluation
  holdout.  No GT labels from --src are ever seen by the CNN.

Usage
-----
  python phase30/cnn_reranker.py \\
      --src benchmark_results_holdout/<eval-folder> \\
      --holdout-dir data/tcd/images/data/tcd/val \\
      --train-src benchmark_results_train/<train-folder> \\
      --train-holdout-dir data/tcd/images/data/tcd/raw \\
      --dst benchmark_results_holdout/<output-folder> \\
      --epochs 8 --batch-size 128
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from shapely.geometry import mapping
from shapely.validation import make_valid
from torch.utils.data import Dataset, DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import _parse_coco_annotations, _seg_to_polygons


PATCH_SIZE = 96
BBOX_PAD_FRAC = 0.20
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def _safe(g):
    if g is None or g.is_empty: return g
    return g if g.is_valid else make_valid(g)


def _iou(a, b):
    a, b = _safe(a), _safe(b)
    if a is None or b is None or not a.intersects(b): return 0.0
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union > 0 else 0.0


def _iop(pred, region):
    """Intersection over predictor's area — the metric pycocotools uses for
    iscrowd ignore decisions."""
    pred, region = _safe(pred), _safe(region)
    if pred is None or region is None or pred.area <= 0: return 0.0
    if not pred.intersects(region): return 0.0
    return pred.intersection(region).area / pred.area


def label_tile(meta_path, pred_path):
    """For one tile return (gdf, labels, train_mask).
        labels[i] = 1 iff pred i matches a cat=2 tree at IoU>=0.5
                    (greedy by score, descending).
        train_mask[i] = False iff pred i is in cat=1 canopy at IoP>=0.5
                       and didn't match a tree (exclude from training)."""
    meta = json.load(open(meta_path))
    H, W = int(meta["height"]), int(meta["width"])
    tree_gts, canopy_gts = [], []
    for cat, seg, *_ in _parse_coco_annotations(meta):
        for p in _seg_to_polygons(seg, H, W):
            (tree_gts if cat == 2 else canopy_gts).append(_safe(p))
    gdf = gpd.read_file(str(pred_path))
    if gdf.empty:
        return gdf, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=bool)
    # Greedy score-desc matching (same as pycocotools COCOeval)
    triples = sorted(
        [(g, float(s), i) for i, (g, s) in
         enumerate(zip(gdf.geometry, gdf["deepforest_score"]))],
        key=lambda t: -t[1])
    matched = [False] * len(tree_gts)
    labels  = np.zeros(len(gdf), dtype=np.int32)
    train_mask = np.ones(len(gdf), dtype=bool)
    for poly, _score, orig_idx in triples:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(tree_gts):
            if matched[j] or g is None or g.is_empty: continue
            iou = _iou(poly, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= 0.5 and best_j >= 0:
            matched[best_j] = True
            labels[orig_idx] = 1
        else:
            if canopy_gts and any(_iop(poly, c) >= 0.5 for c in canopy_gts):
                train_mask[orig_idx] = False
    return gdf, labels, train_mask


def build_index(holdout_dir, pred_dir):
    """Iterate all (meta, pred) pairs.  Returns a list of dicts per tile."""
    holdout_dir = Path(holdout_dir); pred_dir = Path(pred_dir)
    tiles = []
    for meta_path in sorted(holdout_dir.glob("*_meta.json")):
        stem = meta_path.name.replace("_meta.json", "")
        pred_path = pred_dir / f"{stem}_canopyai.geojson"
        if not pred_path.exists():
            continue
        gdf, y, tm = label_tile(meta_path, pred_path)
        if gdf.empty:
            continue
        tiles.append({"stem": stem, "meta_path": meta_path,
                      "pred_path": pred_path, "gdf": gdf,
                      "y": y, "train_mask": tm})
    return tiles


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def _crop_patch(image, bounds, H, W, pad_frac=BBOX_PAD_FRAC, out_size=PATCH_SIZE):
    minx, miny, maxx, maxy = bounds
    bw = maxx - minx; bh = maxy - miny
    pad_x = bw * pad_frac; pad_y = bh * pad_frac
    x0 = max(0, int(minx - pad_x)); y0 = max(0, int(miny - pad_y))
    x1 = min(W, int(maxx + pad_x) + 1); y1 = min(H, int(maxy + pad_y) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    crop = image[y0:y1, x0:x1]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    return np.array(Image.fromarray(crop).resize((out_size, out_size), Image.BILINEAR))


def extract_patches(tiles, holdout_dir, cache_path):
    """Extract patches for all polygons across tiles, cache to .npz."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Loading cached patches from {cache_path}")
        z = np.load(cache_path, allow_pickle=False)
        return z["patches"], z["offsets"]

    holdout_dir = Path(holdout_dir)
    n_total = sum(len(t["gdf"]) for t in tiles)
    print(f"Extracting {n_total} patches at {PATCH_SIZE}x{PATCH_SIZE}...")
    patches = np.zeros((n_total, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    offsets = [0]
    cursor = 0
    t0 = time.time()
    for ti, t in enumerate(tiles):
        tif_path = holdout_dir / f"{t['stem']}.tif"
        if not tif_path.exists():
            cursor += len(t["gdf"]); offsets.append(cursor); continue
        with rasterio.open(tif_path) as src:
            arr = src.read([1, 2, 3])
        image = np.transpose(arr, (1, 2, 0))
        if image.dtype != np.uint8:
            mx = max(1, image.max())
            image = (image.astype(np.float32) / mx * 255).astype(np.uint8)
        H, W = image.shape[:2]
        for geom in t["gdf"].geometry:
            try:
                bounds = geom.bounds if (geom is not None and not geom.is_empty) else (0, 0, 1, 1)
                patches[cursor] = _crop_patch(image, bounds, H, W)
            except Exception:
                pass
            cursor += 1
        offsets.append(cursor)
        if (ti + 1) % 50 == 0 or (ti + 1) == len(tiles):
            elapsed = time.time() - t0
            eta = elapsed / (ti + 1) * (len(tiles) - ti - 1)
            print(f"  {ti+1}/{len(tiles)} tiles  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")
    offsets = np.asarray(offsets)
    np.savez(cache_path, patches=patches, offsets=offsets)
    print(f"  cached {cache_path}  ({patches.nbytes/1024/1024:.0f} MB)")
    return patches, offsets


# ---------------------------------------------------------------------------
# Dataset / model
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    def __init__(self, patches, labels, mask=None, augment=False):
        self.patches = patches
        self.labels  = labels.astype(np.float32)
        self.mask    = mask if mask is not None else np.ones(len(patches), dtype=bool)
        self.idx_map = np.flatnonzero(self.mask)
        self.augment = augment

    def __len__(self): return len(self.idx_map)

    def __getitem__(self, i):
        idx = self.idx_map[i]
        p = self.patches[idx]
        if self.augment:
            if np.random.rand() < 0.5: p = p[:, ::-1, :]
            if np.random.rand() < 0.5: p = p[::-1, :, :]
            k = np.random.randint(4)
            if k > 0: p = np.rot90(p, k)
        t = torch.from_numpy(p.copy()).permute(2, 0, 1).float() / 255.0
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        return t, self.labels[idx]


class CNNReranker(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        embed_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, patch):
        return self.head(self.backbone(patch)).squeeze(-1)


# ---------------------------------------------------------------------------
# Inference API (used by foxtrot for in-pipeline reranking)
# ---------------------------------------------------------------------------

def extract_polygon_patches(image, polygons, patch_size=PATCH_SIZE,
                            pad_frac=BBOX_PAD_FRAC):
    """Extract image patches for a list of polygons.  Returns an array of
    shape (N, patch_size, patch_size, 3) uint8.  Polygons that fail (None /
    empty / out-of-bounds) get zero patches; caller should rely on the
    polygon's own score handling if the patch is meaningless."""
    if image.dtype != np.uint8:
        mx = max(1, image.max())
        image = (image.astype(np.float32) / mx * 255).astype(np.uint8)
    H, W = image.shape[:2]
    patches = np.zeros((len(polygons), patch_size, patch_size, 3), dtype=np.uint8)
    for i, poly in enumerate(polygons):
        try:
            if poly is None or poly.is_empty:
                continue
            patches[i] = _crop_patch(image, poly.bounds, H, W,
                                     pad_frac=pad_frac, out_size=patch_size)
        except Exception:
            pass
    return patches


class RerankerEnsemble:
    """Loaded reranker.  Wraps one or more `CNNReranker` checkpoints and
    averages their per-polygon TP-probability outputs."""

    def __init__(self, state_dicts, device, patch_size=PATCH_SIZE,
                 pad_frac=BBOX_PAD_FRAC):
        self.device = device
        self.patch_size = int(patch_size)
        self.pad_frac = float(pad_frac)
        self.models = []
        for sd in state_dicts:
            m = CNNReranker().to(device)
            m.load_state_dict(sd)
            m.eval()
            self.models.append(m)

    def __len__(self):
        return len(self.models)

    def predict(self, image, polygons, batch_size=256):
        """For each polygon, extract a patch and return the ensemble-mean
        TP probability (numpy float32 array of length len(polygons))."""
        if len(polygons) == 0:
            return np.zeros(0, dtype=np.float32)
        patches = extract_polygon_patches(image, polygons,
                                          patch_size=self.patch_size,
                                          pad_frac=self.pad_frac)
        probs_sum = np.zeros(len(patches), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                p = patches[i:i+batch_size]
                t = torch.from_numpy(p.copy()).permute(0, 3, 1, 2).float() / 255.0
                t = (t - IMAGENET_MEAN) / IMAGENET_STD
                t = t.to(self.device)
                for m in self.models:
                    probs_sum[i:i+batch_size] += torch.sigmoid(m(t)).cpu().numpy()
        return probs_sum / float(len(self.models))


def save_ensemble(state_dicts, path, patch_size=PATCH_SIZE,
                  pad_frac=BBOX_PAD_FRAC, meta=None):
    """Bundle one or more model state_dicts into a single checkpoint file.

    Format:
        {"state_dicts": [...], "patch_size": int, "pad_frac": float,
         "model_class": "CNNReranker", "meta": dict-or-None}
    """
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dicts": [{k: v.cpu() for k, v in sd.items()} for sd in state_dicts],
        "patch_size": int(patch_size),
        "pad_frac": float(pad_frac),
        "model_class": "CNNReranker",
        "meta": meta or {},
    }, path)


def load_ensemble(path, device):
    """Load a checkpoint saved by `save_ensemble` and return a
    `RerankerEnsemble`."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("model_class") != "CNNReranker":
        raise ValueError(f"Checkpoint {path} has incompatible model_class "
                         f"{blob.get('model_class')!r}")
    return RerankerEnsemble(
        state_dicts=blob["state_dicts"],
        device=device,
        patch_size=blob.get("patch_size", PATCH_SIZE),
        pad_frac=blob.get("pad_frac", BBOX_PAD_FRAC),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, device, epochs,
                lr_head=1e-3, lr_backbone=1e-4, weight_decay=1e-4,
                pos_weight=None):
    optimizer = optim.AdamW([
        {"params": model.head.parameters(),     "lr": lr_head},
        {"params": model.backbone.parameters(), "lr": lr_backbone},
    ], weight_decay=weight_decay)
    pw = torch.tensor([pos_weight], dtype=torch.float32).to(device) if pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        ep_loss, n = 0.0, 0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(patches), labels)
            loss.backward(); optimizer.step()
            ep_loss += loss.item() * patches.size(0); n += patches.size(0)
        model.eval()
        v_loss, vn, correct, total = 0.0, 0, 0, 0
        tp_probs, fp_probs = [], []
        with torch.no_grad():
            for patches, labels in val_loader:
                patches, labels = patches.to(device), labels.to(device)
                logits = model(patches)
                v_loss += loss_fn(logits, labels).item() * patches.size(0); vn += patches.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item(); total += labels.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                lbl   = labels.cpu().numpy()
                tp_probs.extend(probs[lbl == 1]); fp_probs.extend(probs[lbl == 0])
        tp_p = float(np.mean(tp_probs)) if tp_probs else 0.0
        fp_p = float(np.mean(fp_probs)) if fp_probs else 0.0
        print(f"  ep {ep+1}/{epochs}  train_loss={ep_loss/n:.4f}  val_loss={v_loss/vn:.4f}  "
              f"acc={correct/total:.3f}  TP_proba={tp_p:.3f}  FP_proba={fp_p:.3f}")
        if v_loss / vn < best_val:
            best_val = v_loss / vn
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Inference / geojson writing
# ---------------------------------------------------------------------------

def predict_all(model, patches, device, batch_size=256):
    model.eval()
    out = np.zeros(len(patches), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            p = patches[i:i+batch_size]
            t = torch.from_numpy(p.copy()).permute(0, 3, 1, 2).float() / 255.0
            t = (t - IMAGENET_MEAN) / IMAGENET_STD
            t = t.to(device)
            out[i:i+batch_size] = torch.sigmoid(model(t)).cpu().numpy()
    return out


def write_rescored_geojsons(tiles, new_scores, dst):
    """Write each tile's geojson with deepforest_score replaced by new_scores."""
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for t in tiles:
        gdf = t["gdf"]
        feats = []
        s = new_scores[t["stem"]]
        for idx, geom in enumerate(gdf.geometry):
            props = {k: v for k, v in gdf.iloc[idx].items() if k != "geometry"}
            clean = {}
            for k, v in props.items():
                if k == "deepforest_score":
                    clean[k] = float(s[idx]); continue
                if hasattr(v, "tolist"): clean[k] = v.tolist()
                elif isinstance(v, (int, float, str, bool)) or v is None: clean[k] = v
                else: clean[k] = str(v)
            feats.append({"type": "Feature", "properties": clean,
                          "geometry": mapping(geom)})
        with open(dst / f"{t['stem']}_canopyai.geojson", "w") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f)
        n_ok += 1
    # Pass through any source-only tiles missing from our list (e.g. zero-pred).
    seen = {t["stem"] for t in tiles}
    src = Path(tiles[0]["pred_path"]).parent if tiles else None
    if src is not None:
        for f in sorted(src.glob("*_canopyai.geojson")):
            stem = f.name.replace("_canopyai.geojson", "")
            if stem in seen: continue
            with open(f) as fh: content = fh.read()
            with open(dst / f.name, "w") as fh: fh.write(content)
            n_ok += 1
    print(f"Wrote {n_ok} geojsons to {dst}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    REPO = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--src", required=True,
                    help="Folder of foxtrot geojsons to rescore (= inference target).")
    ap.add_argument("--dst", required=True,
                    help="Output folder for rescored geojsons.")
    ap.add_argument("--holdout-dir",
                    default=str(REPO / "data" / "tcd" / "images" / "data" / "tcd" / "val"),
                    help="Meta directory matching --src.")
    ap.add_argument("--train-src", required=True,
                    help="Folder of foxtrot geojsons over an INDEPENDENT training set.")
    ap.add_argument("--train-holdout-dir", required=True,
                    help="Meta directory matching --train-src.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="Fraction of train tiles held out for in-training validation.")
    ap.add_argument("--n-runs", type=int, default=1,
                    help="Train this many random-init runs and ensemble their "
                         "outputs.  Each run is trained sequentially; the final "
                         "geojsons are written with the mean ensemble probability.")
    ap.add_argument("--save-checkpoint", default=None,
                    help="If set, save the trained ensemble's state_dicts to "
                         "this path (.pt).  The file is loadable via "
                         "cnn_reranker.load_ensemble() — e.g. by foxtrot.py "
                         "with --reranker_checkpoint.")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n=== Building train + eval index (labels + IGNORE mask) ===")
    train_tiles = build_index(args.train_holdout_dir, args.train_src)
    eval_tiles  = build_index(args.holdout_dir, args.src)
    print(f"  train: {len(train_tiles)} tiles  ({sum(len(t['gdf']) for t in train_tiles)} polys)")
    print(f"  eval:  {len(eval_tiles)} tiles  ({sum(len(t['gdf']) for t in eval_tiles)} polys)")

    print("\n=== Extracting / loading patches ===")
    train_patches_cache = Path(args.train_src).parent / f"_cnn_patches_{Path(args.train_src).name}.npz"
    eval_patches_cache  = Path(args.src).parent       / f"_cnn_patches_{Path(args.src).name}.npz"
    train_patches, train_offsets = extract_patches(train_tiles, args.train_holdout_dir, train_patches_cache)
    eval_patches,  eval_offsets  = extract_patches(eval_tiles, args.holdout_dir, eval_patches_cache)

    # Stack labels / mask aligned with the patch matrix
    y_train  = np.concatenate([t["y"] for t in train_tiles])
    m_train  = np.concatenate([t["train_mask"] for t in train_tiles])
    assert len(y_train) == len(train_patches)

    # Per-tile val split for in-training validation
    rng = np.random.default_rng(42)
    n_tiles = len(train_tiles)
    val_tile_set = set(rng.choice(n_tiles, size=max(1, int(n_tiles * args.val_frac)), replace=False))
    val_rows = np.zeros(len(y_train), dtype=bool)
    for i in range(n_tiles):
        if i in val_tile_set:
            val_rows[train_offsets[i]:train_offsets[i+1]] = True
    tr_rows = ~val_rows & m_train
    va_rows = val_rows & m_train
    print(f"\nTrain rows after IGNORE filter: {tr_rows.sum()}  TP_rate={y_train[tr_rows].mean():.3f}")
    print(f"Val   rows after IGNORE filter: {va_rows.sum()}  TP_rate={y_train[va_rows].mean():.3f}")

    train_ds = PatchDataset(train_patches, y_train, tr_rows, augment=True)
    val_ds   = PatchDataset(train_patches, y_train, va_rows, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"\n=== Training ResNet18 reranker ===")
    print(f"  epochs={args.epochs}  batch_size={args.batch_size}  device={device}")
    print(f"  n_runs={args.n_runs}")
    pos_weight = float((1 - y_train[tr_rows].mean()) / y_train[tr_rows].mean())
    print(f"  pos_weight={pos_weight:.2f}")

    # Train N ensemble members.  Per-run state_dicts are stashed in CPU
    # memory and the predictions are averaged across them.
    ensemble_state_dicts = []
    probs_sum = np.zeros(len(eval_patches), dtype=np.float32)
    for run in range(args.n_runs):
        print(f"\n--- Run {run+1}/{args.n_runs} ---")
        model = CNNReranker().to(device)
        model = train_model(model, train_loader, val_loader, device,
                            epochs=args.epochs, pos_weight=pos_weight)
        ensemble_state_dicts.append(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f"  inference on holdout patches...")
        probs_sum += predict_all(model, eval_patches, device, batch_size=256)
        del model

    new_probs = probs_sum / float(args.n_runs)
    new_scores = {}
    for i, t in enumerate(eval_tiles):
        s, e = int(eval_offsets[i]), int(eval_offsets[i+1])
        new_scores[t["stem"]] = new_probs[s:e]

    print(f"\n=== Writing rescored geojsons -> {args.dst} ===")
    write_rescored_geojsons(eval_tiles, new_scores, args.dst)

    if args.save_checkpoint:
        save_ensemble(ensemble_state_dicts, args.save_checkpoint,
                      patch_size=PATCH_SIZE, pad_frac=BBOX_PAD_FRAC,
                      meta={"epochs": args.epochs, "n_runs": args.n_runs,
                            "train_src": args.train_src})
        print(f"=== Saved ensemble checkpoint -> {args.save_checkpoint} "
              f"({len(ensemble_state_dicts)} member(s)) ===")
    print("Done.")


if __name__ == "__main__":
    main()
