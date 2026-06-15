"""selfdistill.py — STAGE-2 ceiling-breaker (stub + working stage 1).

Rationale: Restor's own docs say the model "disagrees with the noisy GT in a
probably-correct way", so 0.902 is bounded by *label noise*, not model quality.
An ensemble's consensus is a CLEANER target than the GT — training on it can
EXCEED what's achievable against the noisy labels. This is the established route
to beating a label-noise ceiling.

STAGE 1 (implemented below): run the ensemble on the TRAIN tiles, average softmax,
  save per-tile soft tree-probability maps to a cache dir.
STAGE 2 (TODO — small additions, documented):
  - data.py: if cfg.soft_labels_dir is set, load the soft prob map for each tile
    and return it as the target (float [0,1]) instead of / alongside the hard mask.
  - train.py loss: soft target loss = blend * KL(softmax(logits) || soft) +
    (1-blend) * CE(logits, hard_mask). Keep Lovász/aux on the hard mask.
  - run a fresh 2048 model on the blended targets (blend ~0.5).
Then eval as usual. Expected: pushes PAST the GT-noise ceiling if the ensemble
consensus is genuinely cleaner than the labels.

Usage (stage 1):
  python selfdistill.py gen --names semseg_v3,semseg_2048_s0,semseg_2048_s1 \
      --out /data/soft_labels --tta
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config
from model import build_model
from train import get_device
from eval import tta_probs
from ensemble_eval import load_member


def gen_soft_labels(names, out_dir, weights="best.pt", tta=False,
                    tta_scales="0.5,0.75,1.0,1.25", limit=0):
    """Stage 1: ensemble soft tree-probabilities over the TRAIN tiles -> .npy cache."""
    from datasets import load_dataset
    dev = get_device()
    members = [load_member(n.strip(), weights, dev) for n in names.split(",")]
    cfg0 = members[0][1]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(cfg0.hf_dataset, split="train")
    n = len(ds) if not limit else min(limit, len(ds))
    mean = torch.tensor(cfg0.imagenet_mean).view(3, 1, 1)
    std = torch.tensor(cfg0.imagenet_std).view(3, 1, 1)
    scales = [float(s) for s in tta_scales.split(",")]
    for i in range(n):
        ex = ds[i]
        img = np.array(ex["image"].convert("RGB"), dtype=np.uint8)
        x = ((torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 - mean) / std).unsqueeze(0).to(dev)
        prob = None
        for m, _, _ in members:
            if tta:
                p = tta_probs(m, x, dev, torch.bfloat16, scales) / (len(scales) * 2)
            else:
                with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev != "cpu")):
                    p = m(x).float().softmax(1)
            prob = p if prob is None else prob + p
        tree_prob = (prob[0, 1] / len(members)).to("cpu").numpy().astype(np.float16)
        np.save(out / f"{int(ex['image_id'])}.npy", tree_prob)
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{n} soft labels written", flush=True)
    print(f"done -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("gen")
    g.add_argument("--names", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--weights", default="best.pt")
    g.add_argument("--tta", action="store_true")
    g.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "gen":
        gen_soft_labels(a.names, a.out, a.weights, a.tta, limit=a.limit)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
