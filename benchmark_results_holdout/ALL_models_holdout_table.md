# ALL models on OAM-TCD 439 holdout — one combined table (Restor metrics, 2026-06-10)

Columns = Restor TCD paper Table-1. **IoU** = macro Jaccard (bg+tree avg) · **F1** = tree-class Dice · **Acc** = tree recall · **AR@1000** = pycocotools segm AR · **IoU-tree** = foreground IoU.

**Three mAP50 variants (A & B only):**
- **mAP50 tree-only** — individual-crown AP50, canopy GT kept as **iscrowd ignore** (tree predictions in closed canopy get a free pass). Our headline number.
- **mAP50 oracle 2-cat** — mean(tree-AP, canopy-AP), canopy predictions **synthesised from the GT canopy polygons** (T=0.5). An UPPER BOUND — not deployable. Ceiling *if* we had a perfect canopy proposer.
- **mAP50 non-oracle 2-cat** — same 2-cat but canopy proposed from the model's **OWN crowns** (GT-free, deployable). Directly comparable to Restor's published **0.432**. Collapses toward tree-AP/2 (DeepForest has no canopy head → canopy-AP ≈ 0).

> ⚠️ **Cross-category caveat.** A/B F1·IoU are scored vs **tree-instance** GT; C's F1·IoU are scored vs the **dense cover** GT (Restor's SegFormer target). So F1/IoU are **not** comparable A/B ↔ C — only within a category.

| Model | IoU | F1 | Acc | mAP50 tree-only | mAP50 oracle-2cat | mAP50 nonoracle-2cat | AR@1000 | IoU-tree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **═══ A. ZERO-SHOT DETECTION** — DeepForest→SAM(vit_h) crowns; detectree2 @100 m. *No supervised refs — zero-shot only.* ═══ | | | | | | | | |
| detectree2 (stock, author-spec 100 m) | 0.397 | 0.404 | 0.524 | 0.032 | 0.135 | 0.014 | 0.111 | 0.253 |
| phase21 — DeepForest, **no shadow** (wt0) | 0.590 | 0.576 | 0.519 | 0.440 | 0.321 | 0.135 | 0.503 | 0.405 |
| ablation_pre_s2 — shadow **×2** | 0.610 | 0.611 | 0.575 | 0.493 | 0.416 | 0.153 | 0.561 | 0.440 |
| phase22_B_L4 — shadow **×4** | **0.611** | **0.614** | **0.584** | **0.498** | **0.428** | **0.154** | **0.574** | **0.443** |
| DeepForest stock → SAM vit_h | _pending (44/439)_ | | | | | | | |
| blindzero ×2 / ×4 (shadow-specificity control) | _training now (MPS)_ | | | | | | | |
| **── A′. CLEAN RE-BASELINE** (2026-06-15) — reranker-ON, SAM **vit_b**, `df-conf 0.05 / max-dets 512 / pred-thr 0.0`; matched recipe + harness. mAP50 = tree-only. ── | | | | | | | | |
| phase31_s2 — shadow **×2** (Modal-CUDA) | 0.604 | 0.601 | 0.560 | 0.486 | — | — | 0.567 | 0.430 |
| phase22_B_L4 — shadow **×4** (anchor, clean re-run) | — | — | — | 0.482 | — | — | — | — |
| phase31_s6 — shadow **×6** (Modal-CUDA) | 0.602 | 0.602 | 0.568 | 0.482 | — | — | 0.562 | 0.430 |
| **═══ B. FINE-TUNED ON TCD** (supervised detection). *Restor is a fair ref HERE (also supervised).* ═══ | | | | | | | | |
| ablation_tcd_s0 — wt0 | 0.658 | 0.654 | 0.542 | **0.535** | 0.447 | 0.176 | **0.609** | 0.486 |
| ablation_tcd_s2 — wt2 | **0.664** | **0.665** | **0.562** | 0.500 | — | — | 0.569 | **0.498** |
| ablation_tcd_s4 — wt4 | 0.659 | 0.656 | 0.550 | 0.515 | — | — | 0.585 | 0.489 |
| s0_ttaR — s0 + TTA rerank | 0.717 | 0.742 | 0.714 | 0.404 | — | — | 0.468 | 0.590 |
| _Restor Mask-RCNN R50 (supervised)_ | — | — | — | _0.563‡_ | _n/a_ | _**0.432**_ | — | — |
| **═══ C. DENSE SEMANTIC COVER** — DeepLabV3+; vs cover GT, no instances → mAP50 n/a. ═══ | | | | | | | | |
| semseg v3 (512) | ~0.847 | 0.888 | — | n/a | n/a | n/a | n/a | ~0.78 |
| semseg_v3_1024 (best single) | 0.863 | 0.890 | 0.894 | n/a | n/a | n/a | n/a | 0.801 |
| ensemble {1024, v3} + TTA + thr* | 0.866 | **0.892** | — | n/a | n/a | n/a | n/a | — |
| _Restor SegFormer mit-b5 (supervised)_ | _0.876_ | _0.902_ | _0.890_ | n/a | n/a | n/a | n/a | _~0.81_ |

‡ **0.563 is NOT a Restor publication.** It is OUR re-inference of Restor's released Mask-RCNN on the 439 holdout (`benchmark_results_holdout/restor_mrcnn/`), scored tree-only with canopy=iscrowd (`tree_only_control_ap50`). Restor's **only published instance figure is the 2-cat 0.432**; our faithful 2-cat re-inference of their model is 0.402/0.419 ([[project_restor_verified]]). The 0.535-vs-0.563 comparison is apples-to-apples on our harness, but 0.563 is a measured number, not a Restor claim.

## Takeaways
- **A — shadow weight is monotonic on every metric (zero-shot):** wt0→×2→×4 lifts tree-mAP50 0.440→0.493→0.498, oracle-2cat 0.321→0.416→0.428, F1 0.576→0.611→0.614, IoU-tree 0.405→0.440→0.443. The oracle-2cat trend tracks the tree trend, so shadow's gain is genuine crown localisation, not the iscrowd crutch.
- **The 3 mAP50 columns tell the canopy story:** oracle-2cat (~0.43 for phase22) ≈ Restor 0.432 → the detector localises the right regions; non-oracle-2cat (~0.15) ≪ 0.432 → without a canopy head we can't *propose* canopy from crowns. The oracle→non-oracle gap **is** the canopy-proposal problem.
- **A′ — clean re-baseline (reranker-ON, vit_b, matched harness) is FLAT across shadow weight:** wt2/4/6 = 0.486 / 0.482 / 0.482 tree-only mAP50 (±0.004). This is the apples-to-apples sweep (s2/s6 retrained on Modal-CUDA, same `deepforest_custom` recipe as phase22). It confirms the prior verdict: **shadow's benefit is area-recall, not instance mAP50** — the monotonic A trend above is the older vit-mixed regime. Clean-harness numbers sit ~0.016 below the published vit-mixed ones (phase22 0.482 vs 0.498). ⚠️ s2/s6 `deepforest_final.pth` carried a torch.compile `_orig_mod.` key prefix that silently no-op'd the load (ran base weecology head → 2× over-detection) until fixed in `foxtrot.py` + `_fixed` checkpoints — see `project_orig_mod_load_bug` memory.
- **B — shadow weight does NOT help once fine-tuned on TCD** (non-monotonic: s0 best tree-mAP50, s2 best on IoU/F1/Acc). vs supervised Restor: our s0 tree-only 0.535 vs Restor 0.563 (both our harness); 2-cat we only "reach" 0.432 via the GT oracle.
- **C — semantic cover plateaus at 0.892 vs Restor 0.902** (0.897 on-harness): competitive within GT-noise, no clean win.

## Provenance
- tree-only / IoU / F1 / Acc / AR / IoU-tree: `phase30/benchmark.py --skip-inference` → `benchmark_holdout_summary.json`.
- oracle 2-cat: `phase30/canopy_aggregation_test.py` → `canopy_aggregation_oracle_<dir>.json` (T=0.5 twocat_mean).
- non-oracle 2-cat: `phase30/canopy_fair_proposers.py` → `canopy_fair_proposers.json` (best of density/cluster).
- ⏳ = computing this session (detectree2 oracle+fair; phase21/pre_s2 fair). B via `shadow_semseg/eval.py` vs cover GT.
