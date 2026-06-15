"""ensemble_eval.py — average softmax over N models (optional TTA), sweep decision
thresholds in one pass, report F1/P/R/macro-IoU per threshold.

--split test  : the 439 holdout (the number we report)
--split val   : first 96 train tiles (all-folds monitoring set) -> pick t* here

Rigorous use: run --split val to choose t*, then read --split test F1 at that t*.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config
from model import build_model
from train import get_device
from eval import tta_probs


def load_member(name, weights, dev):
    run = Path(Config(name=name).out_dir) / name
    ck = torch.load(run / weights, map_location=dev)
    cfg = (Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
           if "cfg" in ck else Config(name=name))
    m = build_model(cfg, verbose=False).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, cfg, ck.get("val_tree_F1", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="comma-separated run names")
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--tta-scales", default="0.75,1.0,1.25")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = get_device()
    names = [n.strip() for n in a.names.split(",")]
    members = [load_member(n, a.weights, dev) for n in names]
    cfg0 = members[0][1]
    print(f"ensemble {names} (valF1 {[m[2] for m in members]})  tta={a.tta}  split={a.split}", flush=True)

    from datasets import load_dataset
    if a.split == "val":
        ds = load_dataset(cfg0.hf_dataset, split="train"); idxs = list(range(96))
    else:
        ds = load_dataset(cfg0.hf_dataset, split="test"); idxs = list(range(len(ds)))
    if a.limit:
        idxs = idxs[:a.limit]

    mean = torch.tensor(cfg0.imagenet_mean).view(3, 1, 1)
    std = torch.tensor(cfg0.imagenet_std).view(3, 1, 1)
    scales = [float(s) for s in a.tta_scales.split(",")]
    thr = np.round(np.linspace(0.15, 0.60, 10), 3)
    TP = np.zeros(len(thr)); FP = np.zeros(len(thr)); FN = np.zeros(len(thr)); TN = np.zeros(len(thr))

    t0 = time.time()
    for c, i in enumerate(idxs):
        ex = ds[i]
        img = np.array(ex["image"].convert("RGB"), dtype=np.uint8)
        g = (np.array(ex["annotation"].convert("L")) > 0)
        x = ((torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 - mean) / std).unsqueeze(0).to(dev)
        prob = None
        for m, _, _ in members:
            if a.tta:
                p = tta_probs(m, x, dev, torch.bfloat16, scales) / (len(scales) * 2)
            else:
                with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.bfloat16,
                                                     enabled=(dev != "cpu")):
                    p = m(x).float().softmax(1)
            prob = p if prob is None else prob + p
        tp_prob = (prob[0, 1] / len(members)).to("cpu").numpy()
        for j, t in enumerate(thr):
            pr = tp_prob > t
            tp = np.logical_and(pr, g).sum(); fp = np.logical_and(pr, ~g).sum()
            fn = np.logical_and(~pr, g).sum()
            TP[j] += tp; FP[j] += fp; FN[j] += fn; TN[j] += g.size - tp - fp - fn
        if (c + 1) % 25 == 0 or (c + 1) == len(idxs):
            f1n = 2 * TP / (2 * TP + FP + FN)
            print(f"{c+1}/{len(idxs)} {(time.time()-t0)/(c+1):.2f}s/tile "
                  f"bestF1={f1n.max():.4f}@{thr[int(f1n.argmax())]:.2f}", flush=True)

    f1 = 2 * TP / (2 * TP + FP + FN); prec = TP / (TP + FP); rec = TP / (TP + FN)
    treeIoU = TP / (TP + FP + FN); bgIoU = TN / (TN + FP + FN); macroIoU = 0.5 * (treeIoU + bgIoU)
    jb = int(f1.argmax())
    print(f"\n=== ENSEMBLE {names}  split={a.split}  tta={a.tta}  (n={len(idxs)}) ===")
    print(" thr     F1      P       R      macroIoU")
    for j, t in enumerate(thr):
        print(f" {t:.2f}   {f1[j]:.4f}  {prec[j]:.4f}  {rec[j]:.4f}  {macroIoU[j]:.4f}"
              f"{'  <== bestF1' if j == jb else ''}")
    print(f"\nbest: t={thr[jb]:.2f}  F1={f1[jb]:.4f}  macroIoU={macroIoU[jb]:.4f}   [SegFormer 0.902/0.876]")
    out = Path(cfg0.out_dir) / f"ensemble_sweep_{a.split}{'_tta' if a.tta else ''}.json"
    out.write_text(json.dumps({"names": names, "split": a.split, "tta": a.tta,
                               "thr": thr.tolist(), "f1": f1.tolist(),
                               "macroIoU": macroIoU.tolist(), "prec": prec.tolist(),
                               "rec": rec.tolist()}, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
