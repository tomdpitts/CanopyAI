"""threshold_sweep.py — pick the F1-optimal tree-probability threshold on the VAL
set. v2 is precision-heavy / recall-light, so argmax (implicit 0.5) is rarely
F1-optimal. Reports F1/P/R across thresholds; apply the chosen t* at holdout eval.
VAL = first 96 train tiles (the all-folds monitoring set; not train, not test)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config
from model import build_model
from train import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="semseg_v2_noshadow")
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or get_device()
    run = Path(Config(name=a.name).out_dir) / a.name
    ck = torch.load(run / a.weights, map_location=dev)
    cfg = (Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
           if "cfg" in ck else Config(name=a.name))
    m = build_model(cfg, verbose=False).to(dev).eval()
    m.load_state_dict(ck["model"])

    from datasets import load_dataset
    ds = load_dataset(cfg.hf_dataset, split="train")    # all-folds val = first 96
    mean = torch.tensor(cfg.imagenet_mean).view(3, 1, 1)
    std = torch.tensor(cfg.imagenet_std).view(3, 1, 1)
    thr = np.round(np.linspace(0.20, 0.80, 13), 3)
    TP = np.zeros(len(thr)); FP = np.zeros(len(thr)); FN = np.zeros(len(thr))

    t0 = time.time()
    for i in range(a.n):
        ex = ds[i]
        img = np.array(ex["image"].convert("RGB"), dtype=np.uint8)
        g = (np.array(ex["annotation"].convert("L")) > 0)
        x = ((torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 - mean) / std).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev != "cpu")):
            p = m(x).float().softmax(1)[0, 1].to("cpu").numpy()
        for j, t in enumerate(thr):
            pr = p > t
            TP[j] += np.logical_and(pr, g).sum()
            FP[j] += np.logical_and(pr, ~g).sum()
            FN[j] += np.logical_and(~pr, g).sum()
        if (i + 1) % 25 == 0 or (i + 1) == a.n:
            print(f"{i+1}/{a.n} {(time.time()-t0)/(i+1):.2f}s/tile", flush=True)

    f1 = 2 * TP / (2 * TP + FP + FN)
    prec = TP / (TP + FP); rec = TP / (TP + FN)
    j5 = int(np.argmin(np.abs(thr - 0.5))); jb = int(f1.argmax())
    print("\n thr    F1      P       R")
    for j, t in enumerate(thr):
        tag = ("  <- 0.5" if j == j5 else "") + ("  <== best" if j == jb else "")
        print(f" {t:.2f}  {f1[j]:.4f}  {prec[j]:.4f}  {rec[j]:.4f}{tag}")
    print(f"\nbest t*={thr[jb]:.2f}  val_F1={f1[jb]:.4f}   vs 0.5 val_F1={f1[j5]:.4f}   "
          f"gain +{f1[jb]-f1[j5]:.4f}")


if __name__ == "__main__":
    main()
