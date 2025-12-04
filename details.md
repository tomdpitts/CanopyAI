

# Models

## Alpha
- TCD data
- starting from baseline

## Bravo
- WON data
- starting from baseline

## Charlie
- Only 1000 epochs - early cancel
- WON data
- Starting from Alpha5

## Delta
- Full training (early stopping at iter 6000)
- WON data
- Starting from Alpha5

## Echo
- Full training
- WON data
- Starting from Alpha5
- Config change: Ims per batch 16 (up from 8)

## Foxtrot
- 2 stage DeepForest + SAM model
- Training on DeepForest only, SAM frozen
- TCD data
- starting from weecology baseline model - https://github.com/weecology/DeepForest/releases/download/v1.3.0/NEON_checkpoint.pl
- failed, moved on to golf

## Golf
- 2 stage DeepForest + SAM model
- Full training
- TCD data
- starting from weecology baseline model - https://github.com/weecology/DeepForest/releases/download/v1.3.0/NEON_checkpoint.pl


## Hotel
- 2 stage DeepForest + SAM model
- Training on TCD data (46 train images, 13 val images)
- 30 epochs, patience 5, batch size 16
- Starting from weecology baseline model
- **Result**: Early stopping at epoch 6, 0% recall (no predictions made)
- **Issue**: Model stopped too early with small validation set, didn't learn

## India
- 2 stage DeepForest + SAM model  
- Training on TCD data (same 46 train, 13 val)
- **50 epochs** (increased from 30), **patience 10** (increased from 5)
- Batch size 16
- Starting from weecology baseline model
- **Goal**: Fix Hotel's early stopping issue by allowing more training time
- **Result**: Trained to completion but 0% recall (no predictions made)
- **Issue**: Model not learning despite running to completion

## Juliet
- 2 stage DeepForest + SAM model
- Training on WON data (first WON experiment)
- 50 epochs, patience 10, batch size 16
- Starting from weecology baseline model
- **Result**: 0% recall (no predictions made)
- **Issue**: Same fundamental problem as Hotel/India - model not learning

## Critical Issue: Models Not Learning
All three experiments (Hotel, India, Juliet) show the same pattern:
- Training runs without errors
- Loss decreases normally (~1.8-2.5)
- **But 0% recall on validation** (literally no predictions above confidence threshold)
- Models output nothing even at low confidence

**Possible causes:**
1. **Image normalization mismatch** - DeepForest expects specific input format
2. **Pretrained weights not loading correctly** - Model might be starting from scratch
3. **Output head mismatch** - Classification/regression heads might be wrong size
4. **Evaluation bug** - Model might be predicting but evaluation fails
5. **Confidence threshold issue** - All predictions below 0.0 confidence

**Next steps:**
- Check if pretrained weights are actually loading
- Verify image normalization in training data
- Test model immediately after loading pretrained weights (before any training)
- Check if raw model outputs exist (even with low confidence)