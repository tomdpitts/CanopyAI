# shadow_semseg — shadow-aware semantic segmentation (vs Restor SegFormer 0.902)

Isolated experiment: does **shadow conditioning** let a semantic head built on the
phase22_B_L4 backbone match/beat Restor's dedicated SegFormer mit-b5
(**F1 0.902 / macro-IoU 0.876**, verified-reproducible) on the OAM-TCD holdout?

## What this is

- **Architecture:** phase22_B_L4's shadow-trained ResNet-50–FPN backbone (loaded with
  no deepforest dependency) + a Panoptic-FPN-style **semantic head** → per-pixel
  tree-cover logits. Canopy is a valid positive *for free* (it's just cover pixels).
- **"Beyond 400px":** multiscale — each tile is randomly rescaled in `[scale_min,
  scale_max]` then a **fixed 512 crop** is taken (fixed shapes ⇒ MPS-graph-cache safe).
- **Shadow = training-loss reweighter, NOT an input** (corrected mechanism): on the
  ~390 manually-annotated train tiles (folds 0–3; fold-4's 74 are val-held-out),
  pixels the phase22 shadow algorithm marks as shadow get loss × `shadow_weight`
  (4.0). **No predicted angles** (deprecated). **Inference uses no shadow input.**
- **The experiment is the ablation:** identical everything, only `use_shadow` differs.

## Why it's likely to succeed

- Starts from a strong, shadow-trained backbone; a pixel head (unlike box→SAM→union)
  can actually reach ~0.9 pixel-F1.
- Evidence shadow transfers to the pixel metric: zero-shot crown→union F1 went
  0.576→0.614 (phase21→phase22). The head should amplify this.
- Robustness baked in: fixed-shape crops, AMP (bf16 MPS / fp16 CUDA), backbone
  warm-up, **checkpoint + auto-resume**, best-by-val-F1, faithful eval.

## Layout
- `shadow_map.py` — self-contained phase22 shadow-map generator (cv2+numpy only)
- `ckpt.py` — load phase22 ckpt without deepforest (stub-on-demand) + backbone extract
- `data.py` — HF cover masks, fold split, multiscale crop, manual-shadow weight map
- `model.py` — backbone (phase22 weights) + Semantic-FPN head
- `train.py` — shadow-weighted CE+Dice, warm-up, resume
- `eval.py` — faithful holdout eval (full 2048 tiles, their metrics)
- `config.py` — all knobs; the ablation is `use_shadow`

## Env
```
python3.10 -m venv venv && . venv/bin/activate
pip install torch torchvision transformers datasets torchmetrics rasterio opencv-python-headless omegaconf pillow numpy
```
(Smoke used /tmp/segf_venv which already has these.)

## Run
```
# wiring smoke (tiny, CPU/MPS, ~minutes)
bash run_smoke.sh

# full ablation (each arm: train then eval)
python train.py --name semseg_shadow                 # +shadow w=4.0
python eval.py  --name semseg_shadow
python train.py --name semseg_noshadow --no-shadow   # control
python eval.py  --name semseg_noshadow
```

## Compute note
MPS-safe (fixed shapes) so a local `caffeinate` run is viable, but full training
(~3300 tiles × 30 epochs) is slow on MPS — a CUDA GPU is strongly preferred. The
ablation is the result; absolute F1 depends on schedule/compute.

## Result interpretation
Compare `runs/semseg_shadow/eval_best.json` vs `runs/semseg_noshadow/eval_best.json`.
If `f1_tree(shadow) > f1_tree(noshadow)`, shadow conditioning helps semantic — the
contribution — independent of whether either beats 0.902. `iou_macro` is the
bg-inflated headline; `iou_tree_fg` (~0.81 for Restor) is the honest tree IoU.
