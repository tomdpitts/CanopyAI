# Training Journal — Detectree2 Fine-tuning

Chronological record of all training experiments, learnings, and architectural decisions.

---

## Detectree2 Models (Mask R-CNN)

### Alpha
- **Data**: TCD (3-20 images)
- **Base**: `230103_randresize_full.pth` (Detectree2 baseline)
- **Config**: `tiny_debug.yaml` → `fast_train.yaml`
- **Iterations**: Various (alpha1-alpha5)
- **Learning**: Initial fine-tuning experiments on TCD data

### Bravo  
- **Data**: WON (local dataset, untiled 2048×2048px images)
- **Base**: `230103_randresize_full.pth` (Detectree2 baseline)
- **Config**: Standard training config
- **Result**: First attempt at WON dataset

### Charlie
- **Data**: WON  
- **Base**: Alpha5 (TCD fine-tuned model)
- **Iterations**: 1000 (cancelled early)
- **Learning**: Insufficient training — stopped too soon

### Delta
- **Data**: WON
- **Base**: Alpha5 (TCD fine-tuned model)  
- **Iterations**: 6000 (early stopping)
- **Config**: Full training config
- **Learning**: Domain transfer TCD→WON, proper early stopping

### Echo
- **Data**: WON
- **Base**: Alpha5 (TCD fine-tuned model)
- **Iterations**: Full run (12,000 max)
- **Config**: `IMS_PER_BATCH: 16` (up from 8)
- **Status**: Current production model (`model_echo29.pth`)
- **Learning**: Larger batch size improved stability

---

## DeepForest Models (RetinaNet)

### Foxtrot
- **Data**: TCD
- **Architecture**: 2-stage (DeepForest detection → SAM segmentation)
- **Base**: NEON baseline (weecology/DeepForest v1.3.0)
- **Training**: DeepForest only, SAM frozen
- **Result**: ❌ Failed — abandoned
- **Reason**: Architecture not suitable for this task

### Golf
- **Data**: TCD
- **Architecture**: 2-stage (DeepForest + SAM)
- **Base**: NEON baseline
- **Training**: Full training (both stages)
- **Result**: Testing in progress

### Hotel
- **Data**: TCD (46 train, 13 val)
- **Architecture**: 2-stage (DeepForest + SAM)
- **Base**: NEON baseline
- **Config**: 30 epochs, patience 5, batch 16
- **Result**: ❌ 0% recall — early stop at epoch 6
- **Issue**: Validation set too small, early stopping too aggressive

### India
- **Data**: TCD (46 train, 13 val)
- **Architecture**: 2-stage (DeepForest + SAM)
- **Base**: NEON baseline
- **Config**: 50 epochs, patience 10, batch 16 (relaxed from Hotel)
- **Result**: ❌ 0% recall despite training to completion
- **Issue**: Model not learning — loss decreases but no predictions

### Juliet
- **Data**: WON (first WON attempt for DeepForest)
- **Architecture**: 2-stage (DeepForest + SAM)
- **Base**: NEON baseline
- **Config**: 50 epochs, patience 10, batch 16
- **Result**: ❌ 0% recall
- **Issue**: Same as India — fundamental learning problem

**🚨 Critical DeepForest Issue (Hotel/India/Juliet)**:
- Training runs, loss decreases normally (~1.8-2.5)
- But 0% validation recall (no predictions above threshold)
- Suspected causes:
  - Image normalization mismatch (DeepForest expects specific format)
  - Pretrained weights not loading correctly
  - Output head dimension mismatch
  - Evaluation pipeline bug
  - All predictions below confidence threshold
- **Status**: DeepForest approach abandoned, focusing on Detectree2

---

## Key Learnings

### Data
- TCD: 2048×2048px tiles, 100m × 100m ground coverage (20.48 px/m)
- Native tile size at 40m = **819 pixels**
- Current config upsamples to 1024px (1.25× upsampling) — may be unnecessary

### Tiling
- **Training**: Fixed 40m × 40m, buffer=30, produces ~819px chips
- **Inference**: Same params but configurable via `--tile_size` flag
- ⚠️ Risk: Changing `--tile_size` at inference creates train/test mismatch

### Detectron2 Config
- **`TEST.DETECTIONS_PER_IMAGE`**: Default 100 was limiting predictions
  - Fixed: Increased to 300 in `full_train.yaml`
  - Symptom: Every tile showing exactly 100/100 predictions
- **`INPUT.MIN_SIZE_TEST: 1024`**: Upsamples from 819px
  - Question: Should train at native ~800px instead?
  - Tradeoff: Speed vs. small object detection quality

### Training Strategy
- Domain transfer works: TCD→WON (alpha5 → delta/echo)
- Batch size matters: 16 > 8 for stability
- Early stopping patience: Need validation set large enough to be meaningful

### Architecture Decisions
- **Detectree2 (Mask R-CNN)**: Working well ✅
- **DeepForest (RetinaNet)**: Fundamental issues, abandoned ❌
- 2-stage detection→segmentation: Not necessary for this task

---

## Open Questions

1. **Native resolution training**: Should we train at 800px instead of 1024px?
   - Pros: Faster, no upsampling artifacts, matches deployment
   - Cons: Smaller receptive field, less context per tree
   
2. **Multi-scale training**: Is it helping or hurting?
   - Currently using (800, 1024, 1280) px
   - Does this improve generalization or just slow training?

3. **Color augmentations**: Are aerial imagery augmentations effective?
   - Random brightness/contrast/saturation/lighting
   - Need ablation study to measure impact

4. **Tiling strategy**: Is 40m optimal for tree detection?
   - Smaller tiles = more boundary artifacts
   - Larger tiles = less GPU memory, slower training

---

## Next Experiments

- [ ] **Kilo**: Native resolution training (800px, no upsampling)
- [ ] **Lima**: Ablation study — color augmentations on/off
- [ ] **Mike**: Multi-scale training ablation
- [ ] **November**: Optimal tile size exploration (30m, 40m, 50m)
