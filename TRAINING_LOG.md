# CanopyAI Training Log

Tracks all training runs for replicability. Timestamps are AEST (UTC+11).

---

## Phase 5a — Universal Backbone (WON + BRU, no FiLM)

**Date:** 2026-02-25  
**Goal:** Train a single backbone that generalises across WON and BRU domains, avoiding catastrophic forgetting.

### Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `weecology/deepforest-tree` (pretrained) |
| Dataset | `phase5_train_aug.csv` |
| Train images | 32 WON + 42 BRU = 74 total |
| Val images | 9 WON + 11 BRU = 20 total |
| Train annotations | 1,075 WON + 117 BRU = 1,192 total |
| Val annotations | 258 WON + 43 BRU = 301 total |
| Augmentation | 1 random rotation per image [0°, 360°) |
| Epochs (max) | 50 |
| Patience | 15 |
| Learning rate | 0.001 |
| Batch size | 16 |
| Shadow conditioning | Disabled (backbone only) |
| Hardware | Modal A100 |

### Results

| Metric | Value |
|--------|-------|
| Best val mAP (COCO-style) | **0.206** |
| Best epoch | 22 |
| Early stop epoch | 37 (no improvement for 15 epochs) |

### Checkpoint

```
Modal volume: canopyai-deepforest-checkpoints
Path: /checkpoints/phase5_universal_backbone/deepforest_final.pth
Local: phase5_universal_backbone.pth
```

Download:
```bash
modal volume get canopyai-deepforest-checkpoints \
  /checkpoints/phase5_universal_backbone/deepforest_final.pth \
  ./phase5_universal_backbone.pth
```

### Inference — BRU162 (Phase 5a baseline)

```
Image:  input_data/BRU162/splits2/BRU162_center_80pct.tif  (4600×3648px)
Output: zebra/phase5/A/
Model:  phase5_universal_backbone.pth (standard DeepForest, no FiLM)
```

| Metric | Value |
|--------|-------|
| Detections (post-NMS) | **703** |
| DeepForest time | 25.5 s |
| SAM time | 15.1 s |
| Total pipeline | 48.9 s |

Assessment: "fairly good baseline results". False positives more common than false negatives — model detects some non-tree patches/textures as trees. Expected given no shadow conditioning; Phase 5b FiLM training should suppress these.

**Local checkpoint:** `phase5_universal_backbone.pth`

---

## Phase 5b — FiLM Shadow Conditioning (Frozen Backbone)

**Date:** 2026-02-25  
**Goal:** Freeze the Phase 5a backbone. Train only the FiLM + spatial gating layers on the shadow vector so that the model learns to require a directionally consistent shadow before confirming a detection. Expected to reduce false positives specifically.

**Why this helps false positives:** Bright textures, bare soil patches, and shadows cast by other objects can resemble tree crowns. FiLM teaches the model: "only flag this region as a tree if there is a morphologically consistent shadow in the direction given by the sun vector." Objects without a matching shadow get suppressed.

### Configuration

| Parameter | Value |
|-----------|-------|
| Base checkpoint | `phase5_universal_backbone` (Modal volume) |
| Dataset | `phase5` (same as 5a) |
| Shadow conditioning | **Enabled** (FiLM + spatial gating) |
| Freeze backbone | **Yes** |
| Epochs (max) | 50 |
| Patience | 15 |
| Backbone LR | 0.001 (frozen, no effect) |
| FiLM LR | 1e-4 |

### Command

```bash
modal run deepforest_custom/modal_deepforest.py \
    --dataset phase5 \
    --run-name phase5_film_shadow \
    --epochs 50 --patience 15 --lr 0.001 \
    --shadow-conditioning \
    --freeze-backbone \
    --checkpoint /checkpoints/phase5_universal_backbone/deepforest_final.pth
```
