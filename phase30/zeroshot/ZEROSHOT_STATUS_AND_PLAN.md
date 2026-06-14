# Zero-shot benchmark — STATUS & PLAN (2026-06-14)

Single source of truth for the zero-shot detection comparison. Supersedes the old
`REINFER_PLAN_vitl.md` (its vit_l premise was a red herring — see below).

## 🔑 ROOT CAUSE (proven 2026-06-13)
**The published zero-shot mAP50 numbers (phase22=0.498, etc.) are RERANKED.**
- The CNN reranker (`phase30/cnn_reranker_ens3.pt`, 136 MB) rescores every detection
  with a TP-probability → false positives pushed below threshold, true trees ranked high
  → ~180–200 dets ≥0.05/tile (vs ~520 raw) → well-separated PR curve → ~3× the mAP.
- **Proof (phase22, vit_b, 40-tile sample):**
  | run | mAP50 | scores ≥0.8 | dets ≥0.05/tile |
  |---|---:|---:|---:|
  | existing/published | 0.535 | 8.6% | 182 |
  | **+ reranker (current code)** | **0.526** ✅ reproduces | 7.2% | 198 |
  | no reranker (current) | 0.178 | 2.1% | 524 |
  | no reranker, revert 49f543f | 0.194 | — | 385 |
- The existing geojsons' score distribution is **bimodal** (reranker signature); raw runs are unimodal.

## What is NOT the cause (all tested & EXONERATED)
- **Threshold:** mAP is FLAT 0.535 across `--pred-score-thresh` 0.0→0.05 (the low-conf tail is harmless FPs). Earlier "0.18↔0.54 fragility" claim was WRONG.
- **SAM model:** vit_b = vit_l (0.178 vs 0.179). Negligible. → use **vit_b** (cheapest).
- **Commit 49f543f** (RetinaNet `score_thresh` lowering): reverting it gives 0.194 ≈ 0.178, NOT 0.535. Not the cause.

## Corrections to earlier-this-session claims (were WRONG)
1. "Non-reproducible from committed code" — WRONG. Fully reproducible **with the reranker**.
2. "phase21/22 reranker = off" (in PROVENANCE.md) — WRONG. The output is reranked (bimodal scores + reproduction). PROVENANCE.md reranker column needs fixing to **on**.
3. "mAP threshold-fragile" — WRONG (flat).

## What this INVALIDATES (no-reranker artifacts — must re-run WITH reranker)
- blindzero holdout eval (0.098), the vit_b/vit_l SAM-impact (0.178), revert-49f543f (0.194).
- Canonical `control_run.sh` blind/shadow seeds (zs_sw_2 ≈ zs_blind_2 ≈ 0.141, vit_l, no reranker) — relative shadow≈blind verdict may hold, but absolute numbers are no-reranker.

## ✅ THE CLEAN, REPRODUCIBLE COMMAND (reranker-ON)
```
./venv310/bin/python phase30/benchmark.py \
  --models <ckpt/.pth> --names <name> \
  --sam-model vit_b --sam-checkpoint sam_vit_b_01ec64.pth \
  --df-confidence 0.05 --max-dets 512 --pred-score-thresh 0.0 \
  --reranker-checkpoint phase30/cnn_reranker_ens3.pt \
  --output-root benchmark_results_holdout
```
benchmark.py now auto-writes `summary.json` provenance (cmd, git SHA, SAM, reranker, datetimes).

## Provenance of the EXISTING table (recovered 2026-06-13)
- **phase21 + phase22 = SAM vit_b**, reranked (Jun-3 02:59 run).
- **ablation_pre_s2 = SAM vit_h**, reranked (Jun-4 07:28 run) — different SAM (but SAM doesn't matter).
- Training families A/B/C in `benchmark_results_holdout/PROVENANCE.md` (TRAINING section).
- The published w0→×2→×4 trio is training-fork-confounded (pre_s2 = phase30/lib+photometric vs phase21/22 = deepforest_custom+HFlip). Only **phase21↔phase22** is a clean matched pair.

## PLAN GOING FORWARD
1. **(sanity)** Confirm full-439 `phase22 + reranker` ≈ 0.498 (validates the 40-tile reproduction at scale).
2. **Re-baseline** all zero-shot models WITH the reranker through the one clean command above → a consistent, reproducible, `summary.json`-logged table. SAM = vit_b.
3. **s2 / s6** (matched weight sweep): Modal-CUDA retrain on the **family-A `deepforest_custom`** recipe (`--shadow-loss-weight 2.0` / `6.0`, `--epochs 50 --patience 10 --lr 0.001 --batch-size 16`, weecology base, crop400+HFlip, `phase22_train.csv`); then reranker-ON eval. Kill-gate: phase21/22+reranker must reproduce 0.44/0.50 first.
4. **Re-run blindzero/shadow** checkpoints WITH the reranker for the shadow-specificity verdict (the 0.098/0.141 numbers are no-reranker).
5. **Fix PROVENANCE.md** reranker column (off→on).
6. **Arid/dryland subset** — awaiting Tom's tile pick from the contact sheet (`/tmp/tcd_arid_review/`).

## OPEN DECISIONS (Tom)
- Go-ahead on the reranker-ON re-baseline (step 2)?
- s2/s6 Modal retrain (step 3)?
- Arid subset (step 6)?

## Key paths
- Reranker: `phase30/cnn_reranker_ens3.pt`. SAM vit_b: `sam_vit_b_01ec64.pth` (repo root).
- Harness: `phase30/benchmark.py` (auto summary.json). Engine: `foxtrot.py`.
- Table: `benchmark_results_holdout/ALL_models_holdout_table.md`. Ledger: `benchmark_results_holdout/PROVENANCE.md`.
- Memory (machine-local, not iCloud): `~/.claude/projects/<this-project>/memory/` — `project_zeroshot_comparison`, `project_blindzero_control`, `project_benchmark_paths`.
