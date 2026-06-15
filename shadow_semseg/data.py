"""data.py — TCD semantic-segmentation data for the shadow ablation.

- Images + cover masks come from the HF `restor/tcd` dataset (annotation > 0 = tree
  cover, so canopy is a valid positive class for free).
- Manual shadow angles come from tcd_shadow_vectors_by_id.json, keyed `tcd_{image_id}`,
  `manually_reviewed and not excluded` ONLY (predicted angles are deprecated/ignored).
- TRAIN = HF train folds != val_fold; VAL = fold == val_fold (matches phase22, so the
  74 fold-4 manual tiles are not trained on).
- Multiscale: each tile is randomly rescaled in [scale_min, scale_max] then a FIXED
  `crop`×`crop` window is taken — fixed tensor shapes (MPS-graph-cache safe) while still
  letting the head see "beyond 400px" of ground per window.
- Shadow loss weight map (train only, manual tiles only, use_shadow only): pixels the
  phase22 shadow algorithm marks as shadow get loss × shadow_weight.
"""
import json
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from shadow_map import generate_shadow_map


def load_manual_shadow_angles(shadow_json):
    """{int image_id: shadow_angle_deg} for manually-reviewed, non-excluded tiles.
    Returns {} if the file is absent (e.g. no-shadow cloud runs without data/)."""
    from pathlib import Path
    if not Path(shadow_json).exists():
        return {}
    d = json.loads(open(shadow_json).read())
    out = {}
    for k, v in d.items():
        if not (v.get("manually_reviewed") and not v.get("excluded")):
            continue
        if not k.startswith("tcd_"):
            continue
        try:
            out[int(k.split("_", 1)[1])] = float(v["shadow_angle_deg"])
        except (ValueError, KeyError):
            continue
    return out


def rasterize_canopy_region(coco_json, H, W):
    """Binary mask of the cat-1 (CANOPY) class from a tile's coco_annotations.

    The canopy category = closed-canopy regions where individual crowns can't be
    separated (mutually exclusive with cat-2 individual trees). We treat BOTH
    scored polygon canopy (iscrowd=0) and crowd RLE canopy (iscrowd=1) as positive
    — both are visually canopy; the eval ignores the crowd ones either way. Tree
    (cat-2) and background are negative, so the model learns to distinguish
    closed canopy from separable crowns. Returns uint8 HxW."""
    import pycocotools.mask as mask_util
    m = np.zeros((H, W), np.uint8)
    try:
        anns = json.loads(coco_json) if isinstance(coco_json, str) else (coco_json or [])
    except Exception:
        return m
    polys = []
    for a in anns:
        if a.get("category_id") != 1:        # 1 = canopy
            continue
        seg = a.get("segmentation")
        if isinstance(seg, dict) and "counts" in seg:           # RLE
            try:
                r = seg
                if isinstance(r["counts"], list):
                    r = mask_util.frPyObjects(r, r["size"][0], r["size"][1])
                m |= mask_util.decode(r).astype(np.uint8)
            except Exception:
                continue
        elif isinstance(seg, list):                              # polygon(s)
            for part in seg:
                if hasattr(part, "__len__") and len(part) >= 6:
                    pts = np.array(part, np.float64).reshape(-1, 2).round().astype(np.int32)
                    polys.append(pts)
    if polys:
        cv2.fillPoly(m, polys, 1)
    return m


def _normalize(img_u8, mean, std):
    x = torch.from_numpy(img_u8).permute(2, 0, 1).float() / 255.0
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (x - m) / s


class TCDSemanticDataset(Dataset):
    def __init__(self, cfg, split="train"):
        from datasets import load_dataset
        self.cfg = cfg
        self.split = split
        if cfg.smoke:
            # wiring test: reuse the already-downloaded test split as stand-in data
            ds = load_dataset(cfg.hf_dataset, split="test")
            n = len(ds)
            idx = (list(range(min(cfg.limit_train or 6, n))) if split == "train"
                   else list(range(min(cfg.limit_train or 6, n),
                                    min((cfg.limit_train or 6) + (cfg.limit_eval or 4), n))))
            self.ds = ds
            self.idx = idx
            self.angles = load_manual_shadow_angles(cfg.shadow_json)
            print(f"[data:{split}] SMOKE tiles={len(idx)} (test-split stand-in)")
            return
        ds = load_dataset(cfg.hf_dataset, split="train")  # folds live in the train split
        if cfg.all_folds:
            # use ALL train tiles (test holdout is separate); carve 96 for monitoring
            allidx = list(range(len(ds)))
            idx = allidx[96:] if split == "train" else allidx[:96]
        else:
            folds = ds["validation_fold"]
            if split == "train":
                idx = [i for i, f in enumerate(folds) if f != cfg.val_fold]
            else:
                idx = [i for i, f in enumerate(folds) if f == cfg.val_fold]
        if cfg.limit_train and split == "train":
            idx = idx[: cfg.limit_train]
        if cfg.limit_eval and split != "train":
            idx = idx[: cfg.limit_eval]
        self.ds = ds
        self.idx = idx
        self.angles = load_manual_shadow_angles(cfg.shadow_json)
        n_manual = sum(1 for i in idx if int(ds["image_id"][i]) in self.angles)
        print(f"[data:{split}] tiles={len(idx)}  with-manual-shadow={n_manual}")

    def __len__(self):
        return len(self.idx)

    def _scale_and_crop(self, img, mask):
        cfg = self.cfg
        s = random.uniform(cfg.scale_min, cfg.scale_max)
        H, W = img.shape[:2]
        nh, nw = max(1, round(H * s)), max(1, round(W * s))
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        img = cv2.resize(img, (nw, nh), interpolation=interp)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        c = cfg.crop
        # pad if smaller than crop
        ph, pw = max(0, c - nh), max(0, c - nw)
        if ph or pw:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        H2, W2 = img.shape[:2]
        y = random.randint(0, H2 - c)
        x = random.randint(0, W2 - c)
        return img[y:y + c, x:x + c], mask[y:y + c, x:x + c]

    def _geom_aug(self, img, mask, weight):
        """hflip + vflip + rotate, applied identically to all three. Fills: image
        reflect, mask 0 (bg), weight 1.0 (neutral) so rotation padding isn't weighted."""
        if random.random() < 0.5:
            img = img[:, ::-1].copy(); mask = mask[:, ::-1].copy(); weight = weight[:, ::-1].copy()
        if random.random() < 0.5:
            img = img[::-1].copy(); mask = mask[::-1].copy(); weight = weight[::-1].copy()
        if random.random() < 0.5:
            ang = random.uniform(-self.cfg.rotate_deg, self.cfg.rotate_deg)
            h, w = mask.shape
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            weight = cv2.warpAffine(weight, M, (w, h), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
        return img, mask, weight

    def _color_aug(self, img):
        """brightness/contrast + blur + HSV jitter — IMAGE ONLY."""
        f = img.astype(np.float32)
        if random.random() < 0.5:
            f = np.clip(random.uniform(0.8, 1.2) * f + random.uniform(-20, 20), 0, 255)
        if random.random() < 0.3:
            f = cv2.GaussianBlur(f, (0, 0), random.uniform(0.4, 1.2))
        img = f.astype(np.uint8)
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= random.uniform(0.8, 1.2)
            hsv[..., 2] *= random.uniform(0.8, 1.2)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-10, 10)) % 180
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img

    def __getitem__(self, i):
        cfg = self.cfg
        rec = self.ds[self.idx[i]]
        image_id = int(rec["image_id"])
        img = np.array(rec["image"].convert("RGB"), dtype=np.uint8)
        if cfg.target == "canopy_region":
            H0, W0 = img.shape[:2]
            mask = rasterize_canopy_region(rec.get("coco_annotations"), H0, W0)
        else:
            mask = (np.array(rec["annotation"].convert("L")) > 0).astype(np.uint8)

        img, mask = self._scale_and_crop(img, mask)

        # shadow loss weight (train + use_shadow + manual angle only)
        weight = np.ones(mask.shape, dtype=np.float32)
        if self.split == "train" and cfg.use_shadow and image_id in self.angles:
            try:
                sm = generate_shadow_map(img, self.angles[image_id], clahe=True)
                weight = weight + (cfg.shadow_weight - 1.0) * (sm > cfg.shadow_thresh)
            except Exception:
                pass

        # GEOMETRIC augs: applied JOINTLY to img/mask/weight AFTER the weight is built,
        # so the per-pixel weight transforms identically to the image and the manual
        # angle never needs flipping/rotating. COLOR augs are image-only, after, so the
        # shadow-weighting reflects the real scene (not jittered luma).
        if self.split == "train":
            if cfg.aug_geometric:
                img, mask, weight = self._geom_aug(img, mask, weight)
            elif random.random() < 0.5:                      # v1 default: hflip only
                img = img[:, ::-1].copy(); mask = mask[:, ::-1].copy(); weight = weight[:, ::-1].copy()
            if cfg.aug_color:
                img = self._color_aug(img)

        return {
            "image": _normalize(img, cfg.imagenet_mean, cfg.imagenet_std),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "weight": torch.from_numpy(weight.astype(np.float32)),
            "image_id": image_id,
        }
