# Experiment Design Prompt: Shadow-Vector-Driven Geometric Attention for Tree Crown Detection

## Context

This is a DPhil research project on aerial Tree Crown Detection (TCD) using drone/UAV imagery.
The detector is built on top of DeepForest (a Faster R-CNN with ResNet50-FPN backbone, pretrained on
NEON tree crown data). We fine-tune it on two UK forest sites: BRU (broadleaf, darker soil) and WON
(mixed woodland, sandy/pale terrain).

We have a ResNet34 model that predicts the **shadow direction vector** (sin_azimuth, cos_azimuth) from
a given aerial RGB tile. This is a separate pretrained model — treat it as an oracle that provides a
reliable unit vector (sx, sy) per image indicating the sun azimuth.

We also have `generate_shadow_map(img_rgb, angle_deg)` — a classical computer-vision function that
produces a float32 [0,1] shadow probability map from RGB + direction. It uses directional gradients
(brightness drop in the sun direction over 8–150px ranges), a luma gate, lateral edge suppression,
sigmoid thresholding, and speckle removal.

## The Problem We Are Trying to Solve

Trees with clear shadows are the dominant source of missed detections that should have been spotted (false negatives). 


The geometric relationship is deterministic: if there is a tree-cast shadow at pixel position `p`, there
is a tree crown at approximately `p - d * shadow_dir`, where `d` is the shadow length (a function of
tree height and sun elevation angle). Shadow length for trees in the UK at moderate sun elevations
(~30–50°) ranges from ~12 to ~150 pixels at 0.1 m/px resolution.

The research hypothesis is: **explicit geometric reasoning about where the crown is relative to its
shadow should improve recall on shadowed trees**. The shadow direction vector is the key enabler.

## What Has Already Been Tried (and Why It Didn't Work)

### Attempt 1: Shadow 4th Input Channel (phase16_D)
Add the shadow map as a 4th channel to conv1 (zero-init the extra weights). The model sees shadow
as a spatial pattern from the very first layer.

Result: +4.5% mAP over baseline (0.648 vs 0.620).

Ablation (phase16_E): Replace directional shadow map with isotropic darkness map (same pipeline but
averaging over 8 directions instead of using the true sun angle). Result: same +4.5% improvement.

**Conclusion**: the 4th channel helps as an occlusion-awareness signal, but the shadow direction
vector contributes nothing to this improvement. The model cannot exploit the direction when it is
encoded implicitly in a channel's pixel values.

### Attempt 2: Shadow Anticipation (phases 12–14)
At ResNet layer2 (H/8) and layer3 (H/16), sample the shadow map at multiple offsets along the shadow
direction vector: `evidence[p] = shadow_map[p + d * shadow_dir]` for d in {12, 20, 30, 42, 55, 67}px.
This produces a 6-channel "crown evidence" feature map that is mixed into backbone features via a
zero-init 1×1 conv with a learned gate scalar.

Result: no improvement over baseline across all three stages of the ablation. Gate values stalled
at ~0.06–0.2. mAP converged to the same 0.139–0.140 range as baseline in all conditions.

**Conclusion**: mid-backbone residual injection of the evidence feature conflicted with calibrated
RGB features. The gradient path is too long. The gate stalled rather than opening.

### Why Both Attempts Failed to Use the Direction

In the 4th channel approach, the shadow direction is encoded in the *shape* of the blobs in the channel
— but the backbone has no mechanism to interpret that shape as a directional offset. It just sees "dark
region".

In the anticipation approach, the warp is correct geometrically, but injecting the result as a
residual perturbation into mid-backbone features doesn't give the model a clean way to use it — the
features at that scale are already strongly conditioned on RGB, and a small additive perturbation
from a weak gate can't restructure the detection decisions.

## The Proposed New Approaches

Two related ideas that are architecturally cleaner:

---

### Approach A: Shadow-Vector-Driven Deformable Attention Offsets

**Core idea**: Replace the standard region proposal network (RPN) or RoI sampling with deformable
sampling that is *initialised* from the shadow direction vector. Instead of learning sampling offsets
from scratch, provide a geometric prior: for each anchor, also attend to the location offset by
`d * shadow_dir` for a small set of shadow distances d.

**Mechanism sketch**:
- The RPN generates anchor proposals as normal.
- For each anchor at position p, augment the RoI feature extraction (RoIAlign or equivalent) to also
  sample from `p + d * shadow_dir` in the feature map, for d ∈ {0.5, 1.0, 1.5} × expected_shadow_length.
- The expected shadow length can be a global scalar (e.g. 40px) or estimated per-image from a learned
  module taking the shadow map statistics as input.
- The shadow-offset samples are concatenated with the standard RoI features and passed to the
  detection head. A small learned weighting (sigmoid gate, zero-init) controls how much the shadow
  evidence contributes.
- The key difference from Attempt 2: the offset is applied at the *head level* (after RoIAlign), not
  mid-backbone. The backbone is left untouched. The direction only influences which locations are
  sampled during classification/regression, not the feature representations themselves.

**Why this is cleaner**: The gradient chain is short (loss → head → gated offset sampling). The
backbone is not perturbed. The model can learn "if shadow evidence exists at offset d, increase
detection confidence at this anchor" without having to restructure its feature representations.

**Reference architecture**: Deformable Convolutional Networks (Dai et al. 2017) and Deformable DETR
(Zhu et al. 2020) use learned offsets for attention. Here the *initialisation* of those offsets is
provided by the shadow vector rather than learned from scratch.

---

### Approach B: Learned Spatial Warp (STN-style)

**Core idea**: Before the backbone processes the image, apply a differentiable spatial warp that
shifts shadow regions toward the likely crown locations, using the shadow direction vector as input
to the warp parameters.

**Mechanism sketch**:
- A lightweight network (2–3 conv layers) takes the shadow map and shadow direction as input and
  predicts a per-pixel flow field (u, v) — a dense warp.
- The warp is constrained: it should be close to the identity everywhere except at shadow pixels,
  where it shifts content toward `position - d * shadow_dir` (i.e., brings crown-side features
  to the shadow location, or vice versa).
- Initialise the warp network to produce identity transform (zero flow). This ensures the baseline
  is preserved at init.
- The RGB image (and optionally the shadow channel) is warped before being passed to the backbone.
  The backbone is unchanged.
- The warp is differentiable (bilinear sampling, as in Spatial Transformer Networks), so gradients
  flow back through it to the warp network parameters.

**Why this is cleaner**: The entire computation lives before the backbone — there is no mid-backbone
injection. The backbone sees a "corrected" image where shadow evidence has been geometrically aligned
with crown positions. The shadow vector provides a strong inductive bias for what the warp should do.

**Tradeoff**: The warp can blur fine detail. A soft/sparse warp (small magnitude, applied only near
shadow regions) is better than a global warp.

---

## Codebase Details

**Backbone**: `torchvision.models.detection.FasterRCNN` with `ResNet50FPN` backbone, accessed via
`deepforest_main.deepforest()`. The model is at `self.model` (a `FasterRCNN` instance).

**Key model submodules**:
- `self.model.backbone` — `BackboneWithFPN`, wrapping ResNet50
- `self.model.backbone.body` — the ResNet50 body; `.layer1/.layer2/.layer3/.layer4`
- `self.model.rpn` — Region Proposal Network
- `self.model.roi_heads` — RoI heads including box_predictor

**Input pipeline**: Images are float32 tensors in [0,1], shape (C, H, W) with C=3 (or 4 if
shadow channel enabled). Shadow map is computed per-image from RGB + shadow vector in
`_prepend_shadow_channel` before the image enters the model.

**Shadow direction convention**: (sin_azimuth, cos_azimuth) in geographic convention (0° = North,
clockwise). In image coordinates: x = sin_az (right), y = -cos_az (down). Both components stored
in the CSV as `shadow_x`, `shadow_y`.

**Training data**: ~760 training tiles, ~190 val tiles, 400×400px each at ~0.1 m/px. Two domains:
WON (sandy, pale ground, luma_max=150 for shadow gate) and BRU (dark soil/vegetation, luma_max=71).
WON annotations are shrunk by √0.5 per side (50% area) at training time to correct for oversized
ground-truth boxes.

**Metric**: MeanAveragePrecision(iou_thresholds=[0.4]) — torchmetrics. IoU=0.4 is appropriate
for aerial TCD where GT boxes are loosely drawn.

**Baseline performance** (phase16_A, full fine-tune, no shadow):
- map=0.620, map_won=0.637, map_bru=0.539

**4th channel result** (phase16_D, directional shadow channel):
- map=0.648, map_won=0.675, map_bru=0.560 (+4.5%)

**4th channel ablation** (phase16_E, isotropic darkness channel, same pipeline):
- map=0.648, map_won=0.673, map_bru=0.552 (same as D — direction not used)

## What the New Experiment Should Demonstrate

To justify the shadow direction vector as a research contribution:

1. Implement either Approach A or B (or both as sub-ablations).
2. Run three conditions:
   - **F_no_warp**: no shadow input, baseline architecture (same as A, for reference)
   - **F_iso_warp**: warp/offset driven by isotropic darkness map (no direction)
   - **F_dir_warp**: warp/offset driven by directional shadow map (with true sun angle)
3. A statistically significant gap F_dir_warp > F_iso_warp demonstrates that the *direction* adds
   value beyond darkness detection, validating the shadow vector as a contribution.

## Implementation Constraints

- Must integrate with `ShadowConditionedDeepForest(deepforest_main.deepforest)` — a PyTorch Lightning
  module wrapping Faster R-CNN.
- Zero-init all new parameters so the model is identical to baseline at init.
- The shadow map and shadow vector are available at training time (they are in the CSV). At inference
  time they must also be available (the ResNet34 shadow direction predictor provides them).
- Prefer minimal changes to the backbone/FPN — keep the ResNet50-FPN body intact.
- Training runs on Modal (H100 GPU), batch size 16, ~50 epochs with patience=10.
- The codebase uses PyTorch Lightning. The training step, validation step, and data loading are
  managed by DeepForest's parent class. Hooks are available via `on_train_batch_start`,
  `on_before_batch_transfer`, etc.

## Open Questions to Address in the Design

1. For Approach A: at what feature map scale should the shadow-offset RoI sampling happen? (Before
   or after FPN? Which FPN level?)
2. For Approach B: should the warp operate on the raw RGB, the shadow channel, or both? How do you
   prevent the warp from degrading RGB features in non-shadow regions?
3. For both: how do you handle the variable shadow length (different tree heights)? Options: fixed
   set of offsets (as in Attempt 2), learned scalar, or estimated from shadow map statistics.
4. Can the two approaches be combined — a warp to pre-align features, followed by directional RoI
   sampling — without overfitting on a small dataset?
