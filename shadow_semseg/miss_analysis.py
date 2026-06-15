"""miss_analysis.py — where do the semantic recall misses fall?

For a trained model on the holdout, split GT tree-cover into CANOPY regions
(COCO cat 1, closed canopy — detection abstains here) vs individual-TREE regions
(cat 2, crowns — detection is good here), and measure recall in each + where the
false negatives concentrate. Decides whether a multi-task crown-detection head
could plausibly help recall (it can only help in TREE regions, not canopy).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as M

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config
from model import build_model
from train import get_device


def rasterize(seg, H, W):
    try:
        if isinstance(seg, dict):                       # RLE
            r = dict(seg)
            if isinstance(r.get("counts"), str):
                r["counts"] = r["counts"].encode("ascii")
            return M.decode(r).astype(bool)
        if isinstance(seg, list) and seg:               # polygon(s)
            rles = M.frPyObjects(seg, H, W)
            return M.decode(M.merge(rles)).astype(bool)
    except Exception:
        pass
    return np.zeros((H, W), dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="semseg_v2_noshadow")
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()
    dev = get_device()
    run = Path(Config(name=a.name).out_dir) / a.name
    ck = torch.load(run / a.weights, map_location=dev)
    cfg = (Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
           if "cfg" in ck else Config(name=a.name))
    m = build_model(cfg, verbose=False).to(dev).eval()
    m.load_state_dict(ck["model"])
    mean = torch.tensor(cfg.imagenet_mean).view(3, 1, 1)
    std = torch.tensor(cfg.imagenet_std).view(3, 1, 1)

    from datasets import load_dataset
    ds = load_dataset(cfg.hf_dataset, split="test")
    n = min(a.limit, len(ds)) if a.limit else len(ds)

    # pooled counts
    can_area = tree_area = 0          # GT pixels in canopy / tree regions
    can_hit = tree_hit = 0           # predicted-tree pixels inside those regions
    fn_can = fn_tree = fn_other = 0  # missed-cover pixels by region
    cover_area = cover_hit = 0

    t0 = time.time()
    for i in range(n):
        ex = ds[i]
        img = np.array(ex["image"].convert("RGB"), dtype=np.uint8)
        H, W = img.shape[:2]
        cover = (np.array(ex["annotation"].convert("L")) > 0)
        anns = json.loads(ex["coco_annotations"]) if isinstance(ex["coco_annotations"], str) else (ex["coco_annotations"] or [])
        canopy = np.zeros((H, W), dtype=bool); tree = np.zeros((H, W), dtype=bool)
        for an in anns:
            seg = an.get("segmentation")
            if not seg:
                continue
            r = rasterize(seg, H, W)
            if int(an.get("category_id", 2)) == 1:
                canopy |= r
            else:
                tree |= r

        x = ((torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 - mean) / std).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev != "cpu")):
            prob = m(x).float().softmax(1)[0, 1].to("cpu").numpy()
        pred = prob > a.threshold

        cover_area += cover.sum(); cover_hit += (pred & cover).sum()
        can_area += canopy.sum(); tree_area += tree.sum()
        can_hit += (pred & canopy).sum(); tree_hit += (pred & tree).sum()
        fn = cover & ~pred
        fn_can += (fn & canopy).sum()
        fn_tree += (fn & tree).sum()
        fn_other += (fn & ~canopy & ~tree).sum()
        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"{i+1}/{n} {(time.time()-t0)/(i+1):.2f}s/tile", flush=True)

    tot_fn = max(fn_can + fn_tree + fn_other, 1)
    print(f"\n=== miss analysis: {a.name} @ thr {a.threshold} (n={n}) ===")
    print(f"overall recall          : {cover_hit/max(cover_area,1):.4f}")
    print(f"recall INSIDE canopy    : {can_hit/max(can_area,1):.4f}  (canopy = {can_area/1e6:.1f} Mpx)")
    print(f"recall INSIDE tree-crowns: {tree_hit/max(tree_area,1):.4f}  (tree   = {tree_area/1e6:.1f} Mpx)")
    print(f"\nfalse-negative pixels by region:")
    print(f"  in canopy   : {100*fn_can/tot_fn:5.1f}%")
    print(f"  in tree     : {100*fn_tree/tot_fn:5.1f}%")
    print(f"  in neither  : {100*fn_other/tot_fn:5.1f}%")
    print(f"\nVERDICT: multi-task (crown detection) can only recover misses in TREE "
          f"regions. {'PROMISING' if fn_tree > fn_can else 'UNLIKELY TO HELP'} "
          f"(tree-FN {100*fn_tree/tot_fn:.0f}% vs canopy-FN {100*fn_can/tot_fn:.0f}%).")


if __name__ == "__main__":
    main()
