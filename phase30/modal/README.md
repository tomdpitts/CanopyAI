# phase30/modal — shadow ablation on Modal (2-stage, shadow held constant)

Reproduce the kunqi5 (0.515 mAP50) lineage on Modal CUDA and ablate `shadow_loss_weight`,
holding the shadow weight **constant across both stages** (cleaner than kunqi5's own 4→2).
Inference is configurable: SAM backbone (default **vit_h**) and reranker on/off.

> Status: scaffold built + path-reconciled against the live volume. **Not yet run on
> Modal** — do the smoke test (Step 4) before any paid training.

## The models (4 new trains; s0/s4 stage-1 already exist on the volume)

| weight | **Stage 1** — pretrain BRU/WON/NEON (`bs16 lr1e-3 50ep p10`) | **Stage 2** — fine-tune 4k TCD from stage-1 (`bs16 lr1e-5 500ep p5`, canopy 1.0) |
|---|---|---|
| **off** | `phase21_baseline` ✓ exists | `ablation_tcd_s0` ← from phase21_baseline, `--shadow 1` |
| **2.0** | `ablation_pre_s2` ← NEW | `ablation_tcd_s2` ← from ablation_pre_s2, `--shadow 2` |
| **4.0** | `phase22_B_L4` ✓ exists | `ablation_tcd_s4` ← from phase22_B_L4, `--shadow 4` |

Stage-2 recipe is the colleague's **actual** kunqi5 history (`bs16 lr1e-5 epochs500 patience5
canopy-loss-scale 1.0`), not the bs32/lr1e-4 we first guessed. `kunqi5` itself was 4→2
(mismatched) and stays the cited 0.515 reference; `tcd_s4` (4→4) / `tcd_s2` (2→2) bracket it.
Evaluate **both regimes**: stage-1 (zero-shot) and stage-2 (fine-tuned), each ×{reranker off,on}.

## Volumes (all already exist)
- `canopyai-deepforest-data` → `/data` — tiles + CSVs (TCD tiles at `images/data/tcd/raw`)
- `canopyai-deepforest-checkpoints` → `/checkpoints` — `phase21_baseline`, `phase22_B_L4`, `sam_vit_b` present
- `canopyai-benchmark-results` → `/results`

## Workflow

**1. Build the TCD CSVs** (paths → `/data/images/data/tcd/raw/<tile>`):
```bash
./venv310/bin/python phase30/modal/prepare_csvs.py
```

**2. Upload (once) — only the new bits; tiles + stage-1 ckpts are already there:**
```bash
modal volume put canopyai-deepforest-data phase30/modal/data_csvs/tcd_train.csv /phase30/tcd_train.csv
modal volume put canopyai-deepforest-data phase30/modal/data_csvs/tcd_val.csv   /phase30/tcd_val.csv
modal volume put canopyai-deepforest-data phase30/phase30_tcd_canopy_polygons.json /phase30/canopy_polygons.json  # ~292 MB
modal volume put canopyai-deepforest-checkpoints phase30/cnn_reranker_ens3.pt /cnn_reranker_ens3.pt              # 136 MB
```

**3. SAM vit_h** (only vit_b is on the volume):
```bash
modal run phase30/modal/infer.py::download_sam --variant vit_h
```

**4. Smoke test** (1 epoch, cents — verify path end-to-end before paid runs):
```bash
modal run phase30/modal/train.py --dataset tcd --shadow 2 --run-name smoke \
  --base-checkpoint /checkpoints/phase22_B_L4/deepforest_final.pth --fast-dev-run
```

**5. Train (4 cells).** `ablation_tcd_s2` depends on `ablation_pre_s2` finishing first; the
other two start immediately. See `ablation_grid.sh` for the exact commands.
```bash
bash phase30/modal/ablation_grid.sh        # prints/launches the 4 trains in dependency order
```

**6. Evaluate** (vit_h + reranker = the 0.515 pipeline; flip `--reranker false` for the ablation):
```bash
modal run --detach phase30/modal/infer.py \
  --models phase21_baseline,ablation_pre_s2,phase22_B_L4,ablation_tcd_s0,ablation_tcd_s2,ablation_tcd_s4 \
  --sam vit_h --reranker true
modal volume get canopyai-benchmark-results /<name> benchmark_results_holdout/<name>
```

## Cost & the lr1e-5 caveat ($30 budget)
The faithful stage-2 recipe is **gentle** (lr1e-5) with up to **500 epochs** (patience-5 early-stops
on val mAP). At ~3 min/epoch on A100 (~$2.5/h), a run that converges at ~100 epochs is ~$12 — so
**3 stage-2 runs at full faithful settings can exceed $30.** De-risk: run `ablation_tcd_s4` *first*,
read its actual per-epoch time + convergence epoch, then budget s0/s2 — capping `--epochs` or
switching `gpu="A10G"` (cheaper; the canopy loss runs partly on CPU so A10G wall-clock is close).
Stage-1 `ablation_pre_s2` is tiny (~minutes). seed=42 is hardwired in `phase30/lib`.
