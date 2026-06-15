"""eval.py — faithful semantic eval on Restor's HF test split (the 439 holdout),
full 2048 tiles (no resize), their metric definitions, so the number is directly
comparable to their SegFormer mit-b5 (F1 0.902, macro-IoU 0.876).

Usage:
  python eval.py --name semseg_shadow            # uses runs/<name>/best.pt
  python eval.py --name semseg_shadow --limit 30 # quick subset
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, JaccardIndex

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config
from model import build_model
from train import get_device


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="semseg_shadow")
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tta", action="store_true",
                    help="multi-scale + hflip test-time augmentation (avg softmax)")
    ap.add_argument("--tta-scales", default="0.75,1.0,1.25")
    return ap.parse_args()


def tta_probs(model, x, dev, amp_dtype, scales, flip=True):
    """Average softmax over {scales} x {identity, hflip}, all mapped back to x's HxW."""
    H, W = x.shape[-2:]
    acc = None
    for s in scales:
        xs = x if abs(s - 1.0) < 1e-6 else F.interpolate(
            x, scale_factor=s, mode="bilinear", align_corners=False)
        for fl in ((False, True) if flip else (False,)):
            xin = torch.flip(xs, dims=[3]) if fl else xs
            with torch.no_grad(), torch.autocast(device_type=dev, dtype=amp_dtype,
                                                 enabled=(dev != "cpu")):
                lg = model(xin)
            if fl:
                lg = torch.flip(lg, dims=[3])
            lg = F.interpolate(lg.float(), size=(H, W), mode="bilinear", align_corners=False)
            p = lg.softmax(1)
            acc = p if acc is None else acc + p
    return acc


def main():
    a = parse_args()
    dev = get_device()
    run = Path(Config(name=a.name).out_dir) / a.name
    wpath = run / a.weights
    ck = torch.load(wpath, map_location=dev)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__}) \
        if "cfg" in ck else Config(name=a.name)
    model = build_model(cfg, verbose=False).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(f"loaded {wpath} (epoch {ck.get('epoch','?')}, val_tree_F1 {ck.get('val_tree_F1','?')})", flush=True)

    from datasets import load_dataset
    ds = load_dataset(cfg.hf_dataset, split="test")
    n = len(ds) if not a.limit else min(a.limit, len(ds))
    scales = [float(s) for s in a.tta_scales.split(",")]
    if a.tta:
        print(f"TTA on: scales={scales} + hflip", flush=True)

    mean = torch.tensor(cfg.imagenet_mean).view(3, 1, 1)
    std = torch.tensor(cfg.imagenet_std).view(3, 1, 1)
    f1 = F1Score(task="multiclass", num_classes=2, average="none")
    jac = JaccardIndex(task="multiclass", num_classes=2)   # macro
    jac_fg = JaccardIndex(task="binary")
    accn = Accuracy(task="multiclass", num_classes=2, average="none")

    t0 = time.time()
    for i in range(n):
        ex = ds[i]
        img = np.array(ex["image"].convert("RGB"), dtype=np.uint8)
        gt = (np.array(ex["annotation"].convert("L")) > 0).astype(np.int64)
        x = (torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 - mean) / std
        x = x.unsqueeze(0).to(dev)
        if a.tta:
            probs = tta_probs(model, x, dev, torch.bfloat16, scales)
            pred = probs.argmax(1)[0].to("cpu").flatten()
        else:
            with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.bfloat16,
                                                 enabled=(dev != "cpu")):
                logits = model(x)
            pred = logits.argmax(1)[0].to("cpu").flatten()
        tgt = torch.from_numpy(gt).flatten()
        f1.update(pred, tgt); jac.update(pred, tgt); jac_fg.update(pred, tgt); accn.update(pred, tgt)
        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"{i+1}/{n} {(time.time()-t0)/(i+1):.2f}s/tile "
                  f"f1_tree={f1.compute()[1]:.4f} iou_macro={jac.compute():.4f}", flush=True)

    f1v = f1.compute(); accnv = accn.compute()
    res = {
        "name": cfg.name, "use_shadow": cfg.use_shadow, "n": n,
        "f1_tree": float(f1v[1]), "iou_macro": float(jac.compute()),
        "iou_tree_fg": float(jac_fg.compute()),
        "acc_tree": float(accnv[1]), "acc_bg": float(accnv[0]),
        "restor_segformer_b5": {"f1": 0.902, "iou_macro": 0.876},
    }
    print("\n==== %s (n=%d) ====" % (cfg.name, n))
    print(f"F1 tree        = {res['f1_tree']:.4f}   [Restor SegFormer 0.902]")
    print(f"IoU macro      = {res['iou_macro']:.4f}   [Restor 0.876]")
    print(f"IoU tree (fg)  = {res['iou_tree_fg']:.4f}")
    print(f"Acc tree       = {res['acc_tree']:.4f}")
    out = run / f"eval_{Path(a.weights).stem}{'_tta' if a.tta else ''}.json"
    out.write_text(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
