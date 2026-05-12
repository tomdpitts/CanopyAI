#!/usr/bin/env python3
"""
DeepForest Fine-tuning Script for Tree Detection

Trains DeepForest on custom datasets using DeepForest 2.0 config-based API.

Usage:
    python train_deepforest.py --train_csv train.csv --val_csv val.csv --epochs 10

On Modal:
    Called from modal_deepforest.py
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so utils.py is importable when this
# script is run directly (e.g. python deepforest_custom/train_deepforest.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from deepforest import main as deepforest_main
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Set matmul precision for A100/H100 tensor cores
torch.set_float32_matmul_precision("medium")


# ---------------------------------------------------------------------------
# Monkey-patch: fix deepforest evaluate_boxes index-alignment bug
#
# When an image in ground_df has zero predictions, evaluate_boxes builds a
# placeholder DataFrame mixing:
#   - pd.Series([...] * n) with default index [0, 1, ..., n-1]
#   - group.label / group.geometry with the original DataFrame index (e.g. [5, 6])
# Pandas unions the two index sets → length 2n, then rejects the numpy arrays
# of length n → ValueError: "array length 1 does not match index length 2".
# Fix: reset the group index before building the placeholder.
# ---------------------------------------------------------------------------
def _patched_evaluate_boxes(predictions, ground_df, iou_threshold=0.4):
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import shapely
    from deepforest import evaluate as _ev
    from deepforest.utilities import determine_geometry_type

    if ground_df.empty:
        return {
            "results": None,
            "box_recall": None,
            "box_precision": 0,
            "class_recall": None,
            "predictions": predictions,
            "ground_df": ground_df,
        }

    if not isinstance(predictions, gpd.GeoDataFrame):
        if "geometry" not in predictions.columns and all(
            c in predictions.columns for c in ["xmin", "ymin", "xmax", "ymax"]
        ):
            predictions = predictions.copy()
            predictions["geometry"] = shapely.box(
                predictions["xmin"], predictions["ymin"],
                predictions["xmax"], predictions["ymax"],
            )
        predictions = gpd.GeoDataFrame(predictions, geometry="geometry")

    if not isinstance(ground_df, gpd.GeoDataFrame):
        if "geometry" not in ground_df.columns and all(
            c in ground_df.columns for c in ["xmin", "ymin", "xmax", "ymax"]
        ):
            ground_df = ground_df.copy()
            ground_df["geometry"] = shapely.box(
                ground_df["xmin"], ground_df["ymin"],
                ground_df["xmax"], ground_df["ymax"],
            )
        ground_df = gpd.GeoDataFrame(ground_df, geometry="geometry")

    predictions_by_image = {
        name: group.reset_index(drop=True)
        for name, group in predictions.groupby("image_path")
    }

    results, box_recalls, box_precisions = [], [], []
    for image_path, group in ground_df.groupby("image_path"):
        image_predictions = predictions_by_image.get(image_path, pd.DataFrame())
        if not isinstance(image_predictions, pd.DataFrame) or image_predictions.empty:
            image_predictions = pd.DataFrame()

        if image_predictions.empty:
            # FIX: reset_index so all Series share the same 0-based index
            g = group.reset_index(drop=True)
            n = len(g)
            result = pd.DataFrame({
                "truth_id":       group.index.values,
                "prediction_id":  [None]  * n,
                "IoU":            [0.0]   * n,
                "predicted_label":[None]  * n,
                "score":          [None]  * n,
                "match":          [False] * n,
                "true_label":     g.label.values,
                "geometry":       g.geometry.values,
            })
            box_recalls.append(0)
            results.append(result)
            continue
        else:
            group = group.reset_index(drop=True)
            result = _ev.evaluate_image_boxes(
                predictions=image_predictions, ground_df=group
            )

        result["image_path"] = image_path
        result["match"] = result.IoU > iou_threshold
        result["match"] = result["match"].fillna(False)
        true_positive = sum(result["match"])
        box_recalls.append(true_positive / result.shape[0])
        box_precisions.append(true_positive / image_predictions.shape[0])
        results.append(result)

    if results:
        results = pd.concat(results, ignore_index=True)
    else:
        results = pd.DataFrame(columns=[
            "truth_id", "prediction_id", "IoU", "predicted_label",
            "score", "match", "true_label", "geometry", "image_path",
        ])

    box_recall = np.mean(box_recalls)
    box_precision = np.mean(box_precisions) if box_precisions else np.nan
    class_recall = _ev.compute_class_recall(results[results.match])

    return {
        "results": results,
        "box_precision": box_precision,
        "box_recall": box_recall,
        "class_recall": class_recall,
        "predictions": predictions,
        "ground_df": ground_df,
    }

from deepforest import evaluate as _deepforest_evaluate
_deepforest_evaluate.evaluate_boxes = _patched_evaluate_boxes
# ---------------------------------------------------------------------------


# Import model classes from separate file
try:
    from .models import ShadowConditionedDeepForest
except ImportError:
    from models import ShadowConditionedDeepForest


class _TiledValDataset:
    """Tiles each 2048×2048 val image into 400×400 patches for validation.
    Defined at module level so it can be pickled by multiprocessing workers."""
    def __init__(self, base_ds, patch_size=400, overlap=0.0, min_visibility=0.5):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        self.base        = base_ds
        self.patch_size  = patch_size
        self.step        = max(1, int(patch_size * (1 - overlap)))
        self.min_vis     = min_visibility
        self.collate_fn  = base_ds.collate_fn
        self._val_t      = A.Compose(
            [ToTensorV2()],
            bbox_params=A.BboxParams(format="pascal_voc",
                                     label_fields=["category_ids"],
                                     clip=True, min_visibility=min_visibility),
        )
        _sample = base_ds.load_image(0)
        self._H, self._W = _sample.shape[:2]
        self._build_patch_list()

    def _build_patch_list(self):
        self._patches = []
        for img_idx in range(len(self.base.image_names)):
            for py in range(0, self._H, self.step):
                for px in range(0, self._W, self.step):
                    py2 = min(py + self.patch_size, self._H)
                    px2 = min(px + self.patch_size, self._W)
                    py1 = max(0, py2 - self.patch_size)
                    px1 = max(0, px2 - self.patch_size)
                    self._patches.append((img_idx, px1, py1, px2, py2))

    def __len__(self):
        return len(self._patches)

    def __getitem__(self, idx):
        img_idx, px1, py1, px2, py2 = self._patches[idx]
        img_name = self.base.image_names[img_idx]
        img      = self.base.load_image(img_idx)
        patch    = img[py1:py2, px1:px2]

        gt      = self.base.annotations_for_path(img_name)
        boxes   = gt["boxes"]
        labels  = gt["labels"]

        if len(boxes):
            cx1 = np.maximum(boxes[:, 0], px1) - px1
            cy1 = np.maximum(boxes[:, 1], py1) - py1
            cx2 = np.minimum(boxes[:, 2], px2) - px1
            cy2 = np.minimum(boxes[:, 3], py2) - py1
            bw  = boxes[:, 2] - boxes[:, 0]; bh = boxes[:, 3] - boxes[:, 1]
            vis = np.where(bw * bh > 0,
                           np.maximum(cx2 - cx1, 0) * np.maximum(cy2 - cy1, 0) / (bw * bh),
                           0)
            keep = vis >= self.min_vis
            if keep.any():
                patch_boxes = np.stack(
                    [cx1[keep], cy1[keep], cx2[keep], cy2[keep]],
                    axis=1).astype(np.float32)
                patch_labels = labels[keep]
            else:
                patch_boxes, patch_labels = np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
        else:
            patch_boxes, patch_labels = np.zeros((0, 4), np.float32), np.zeros(0, np.int64)

        aug     = self._val_t(image=patch, bboxes=patch_boxes, category_ids=patch_labels)
        image   = aug["image"]
        boxes_t = torch.from_numpy(np.array(aug["bboxes"], dtype=np.float32)).reshape(-1, 4)
        labels_t= torch.from_numpy(np.array(aug["category_ids"], dtype=np.int64))
        return image, {"boxes": boxes_t, "labels": labels_t}, img_name


class _SingleCropValDataset:
    """One centre-crop per val image — fast, deterministic, good enough for early stopping.
    Defined at module level so multiprocessing workers can pickle it."""
    def __init__(self, base_ds, patch_size=400, min_visibility=0.5):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        self.base       = base_ds
        self.collate_fn = base_ds.collate_fn
        self._val_t     = A.Compose(
            [A.RandomCrop(patch_size, patch_size), ToTensorV2()],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"],
                                     clip=True, min_visibility=min_visibility),
        )

    def __len__(self):
        return len(self.base.image_names)

    def __getitem__(self, idx):
        img_name = self.base.image_names[idx]
        img      = self.base.load_image(idx)
        gt       = self.base.annotations_for_path(img_name)
        boxes    = gt["boxes"] if len(gt["boxes"]) else np.zeros((0, 4), np.float32)
        labels   = gt["labels"] if len(gt["labels"]) else np.zeros(0, np.int64)
        aug      = self._val_t(image=img, bboxes=boxes, category_ids=labels)
        image    = aug["image"]
        boxes_t  = torch.from_numpy(np.array(aug["bboxes"], dtype=np.float32)).reshape(-1, 4)
        labels_t = torch.from_numpy(np.array(aug["category_ids"], dtype=np.int64))
        return image, {"boxes": boxes_t, "labels": labels_t}, img_name


def widen_first_conv_for_shadow_channel(model):
    """
    Replace backbone.body.conv1 ([64,3,7,7]) with a 4-channel version ([64,4,7,7]).
    The first 3 channel slices keep pretrained RGB weights; the 4th is zero-initialised
    so the model starts from identical behaviour to the 3-channel baseline at step 0.
    Also extends the RetinaNet transform's image_mean/image_std to 4 elements so the
    built-in normalizer doesn't broadcast-fail on 4-channel tensors.
    Call this AFTER loading pretrained / checkpoint weights.
    """
    inner = model.model if hasattr(model, 'model') else model
    old_conv = inner.backbone.body.conv1
    new_conv = nn.Conv2d(
        4, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight        # copy RGB weights
        new_conv.weight[:, 3:] = 0                       # 4th channel = 0
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    inner.backbone.body.conv1 = new_conv

    # Extend the RetinaNet transform normalizer to 4 channels.
    # Default ImageNet mean/std are 3-element; broadcasting crashes on 4-ch input.
    # Shadow map is [0,1] float → mean=0.5, std=0.25 keeps it in a similar range.
    transform = inner.transform
    if len(transform.image_mean) == 3:
        transform.image_mean = list(transform.image_mean) + [0.5]
        transform.image_std  = list(transform.image_std)  + [0.25]
        print("   ✅ Normalizer extended to 4 channels (shadow: mean=0.5 std=0.25)")

    print("   ✅ First conv widened to 4 channels (shadow map channel zero-initialised)")


def train_deepforest(
    train_csv,
    val_csv=None,
    output_dir="/checkpoints",
    run_name="default",
    epochs=10,
    batch_size=8,
    lr=0.001,
    patience=5,
    pretrained=True,
    wandb_project=None,
    shadow_angle_deg=None,
    checkpoint=None,          # Optional path to initial weights
    accelerator=None,
    freeze_backbone=False,
    shadow_channel=False,        # Run B/D: shadow as 4th input channel
    shadow_cross_attention=False, # Run C/D: shadow cross-attention after layer4
    shadow_luma_only=False,       # Ablation: replace directional shadow map with luma darkness map
    shadow_input_only=False,      # Ablation F: replace RGB entirely with shadow map (tiled ×3)
    shadow_proposals=False,       # Run F_dir: inject shadow-derived proposals alongside RPN
    shadow_proposals_iso=False,   # Ablation F_iso: scramble shadow direction in proposal generation
    shadow_loss_reweight=False,   # Phase 17: upweight focal loss for shadow-casting GT boxes
    shadow_loss_weight=2.0,       # Multiplier for shadow-casting GT positive anchors
    won_bbox_shrink=True,         # Always apply WON bbox normalisation for consistent evaluation
    augmentations=None,           # If set (list of dicts), overrides default/wrapper augmentation logic
    fast_dev_run=False,           # Lightning fast_dev_run: 1 train + 1 val batch then exit
    precision=None,               # Override Lightning precision (e.g. "16-mixed", "bf16-mixed")
):
    """
    Train a DeepForest model using DeepForest 2.0 config-based API.

    Automatically resumes from existing checkpoints if found (for Modal auto-restarts).

    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV (optional)
        output_dir: Base directory for checkpoints
        run_name: Name of this training run (creates subfolder)
        epochs: Number of epochs to train
        batch_size: Batch size
        lr: Learning rate
        patience: Early stopping patience
        pretrained: Whether to use pretrained weights
        wandb_project: Weights & Biases project name (unused in this version)
        checkpoint: Optional path to DeepForest checkpoint file to load initial weights from
        shadow_channel: If True, add directional shadow map as 4th input channel (Run B/D)
        shadow_cross_attention: If True, graft cross-attention after layer4 (Run C/D)
    """
    # Create run-specific output directory
    run_output_dir = str(Path(output_dir) / run_name)
    Path(run_output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"🌲 DeepForest Training: {run_name}")
    print("=" * 60)
    print("=" * 60)
    print("🌲 DeepForest Fine-tuning")
    print("=" * 60)

    # Initialize model
    print("\n⚙️  Initializing model...")

    # ── Model initialisation ──────────────────────────────────────────────
    use_shadow = shadow_channel or shadow_cross_attention or shadow_proposals or shadow_loss_reweight
    # Always use ShadowConditionedDeepForest when won_bbox_shrink=True so that
    # _maybe_shrink_won_targets is applied consistently across all stages (A/B/C).
    # Using plain deepforest for stage A would train on large WON boxes, then
    # stage B would evaluate against shrunk boxes → mAP collapse at stage B start.
    use_wrapper = use_shadow or won_bbox_shrink

    if use_wrapper:
        # Auto-derive base shadow angle from CSV if not explicitly provided
        if shadow_angle_deg is None:
            _df = pd.read_csv(train_csv)
            if "shadow_angle" in _df.columns:
                shadow_angle_deg = float(_df["shadow_angle"].mode()[0])
                print(f"   Shadow angle auto-derived from CSV: {shadow_angle_deg:.1f} deg")
            else:
                shadow_angle_deg = 215.0
                print(f"   Shadow angle defaulted to {shadow_angle_deg} deg (no shadow_angle column)")
        print(f"   shadow_channel={shadow_channel}  shadow_cross_attention={shadow_cross_attention}  "
              f"shadow_proposals={shadow_proposals}  shadow_proposals_iso={shadow_proposals_iso}  "
              f"won_bbox_shrink={won_bbox_shrink}")
        model = ShadowConditionedDeepForest(
            shadow_angle_deg=shadow_angle_deg,
            train_csv=train_csv,
            val_csv=val_csv,
            freeze_backbone=freeze_backbone,
            shadow_channel=shadow_channel,
            shadow_cross_attention=shadow_cross_attention,
            shadow_luma_only=shadow_luma_only,
            shadow_input_only=shadow_input_only,
            shadow_proposals=shadow_proposals,
            shadow_proposals_iso=shadow_proposals_iso,
            shadow_loss_reweight=shadow_loss_reweight,
            shadow_loss_weight=shadow_loss_weight,
        )
    else:
        print("   Shadow: DISABLED, WON shrink: DISABLED (raw baseline)")
        model = deepforest_main.deepforest()

    # Load weights
    import traceback

    if checkpoint:
        print(f"\n Loading checkpoint: {checkpoint}")
        try:
            state_dict = torch.load(checkpoint, map_location="cpu")
            ck_keys    = list(state_dict.keys())
            has_model_prefix = any(k.startswith("model.") for k in ck_keys)

            # If the checkpoint has a 4-channel conv1, widen the model BEFORE loading
            # so the shapes match (strict=False skips missing/extra keys but not size mismatches).
            _conv1_key = "model.backbone.body.conv1.weight"
            if shadow_channel and _conv1_key in state_dict and state_dict[_conv1_key].shape[1] == 4:
                print("   Checkpoint has 4-ch conv1 — widening model before loading...")
                widen_first_conv_for_shadow_channel(model)

            if has_model_prefix and use_shadow:
                # Full ShadowConditionedDeepForest state dict
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"   Loaded full wrapper checkpoint ({len(ck_keys)} keys)")
                if missing:     print(f"   Missing ({len(missing)}): {missing[:3]}...")
                if unexpected:  print(f"   Unexpected ({len(unexpected)}): {unexpected[:3]}...")
            else:
                # Backbone-only state dict (e.g. oscar50.pth)
                model.model.load_state_dict(state_dict, strict=False)
                print(f"   Loaded backbone-only checkpoint ({len(ck_keys)} keys)")

            print("   Checkpoint loaded successfully")
        except Exception as e:
            print(f"   Failed to load checkpoint: {e}")
            traceback.print_exc()
            raise RuntimeError("Cannot continue without loading checkpoint") from e
    elif pretrained:
        print("\n📦 Loading pretrained weights...")
        try:
            # Matches oscar_archive: call load_model() directly without create_model() first.
            # create_model() re-initialises weights before the load, which can cause subtle
            # mismatches in which layers get overwritten.
            model.load_model("weecology/deepforest-tree")
            print("   ✅ Loaded HuggingFace pretrained weights")
        except Exception as e:
            print(f"   ❌ Failed to load pretrained weights: {e}")
            print(f"   Full traceback:")
            traceback.print_exc()
            raise RuntimeError("Cannot continue without pretrained weights") from e

    # Widen first conv to 4 channels AFTER weights are loaded (shadow_channel mode).
    # Skip if already widened (e.g. when loading a 4-ch checkpoint above).
    if shadow_channel:
        _inner = model.model if hasattr(model, "model") else model
        if _inner.backbone.body.conv1.weight.shape[1] != 4:
            print("\n🔑 Shadow channel mode: widening first conv to 4 channels...")
            widen_first_conv_for_shadow_channel(model)
        else:
            print("\n🔑 Shadow channel mode: conv1 already 4-channel (loaded from checkpoint)")

    # Auto-detect and resume from checkpoint if it exists
    # This handles Modal auto-restarts gracefully
    checkpoint_path = None
    print("\n🔍 Searching for checkpoint to resume from...")
    checkpoint_files = list(Path(run_output_dir).glob("*.ckpt"))
    if checkpoint_files:
        # Get most recent checkpoint by modification time
        checkpoint_path = str(max(checkpoint_files, key=lambda p: p.stat().st_mtime))
        if checkpoint_path:
             print("   🔎 Validating checkpoint...")
             try:
                 ckpt = torch.load(checkpoint_path, map_location="cpu")
                 if "optimizer_states" in ckpt and len(ckpt["optimizer_states"]) > 0:
                     # Check if param_groups exists
                     opt_state = ckpt["optimizer_states"][0]
                     if "param_groups" not in opt_state:
                         print("   ❌ Checkpoint corrupted: optimizer state missing 'param_groups'")
                         checkpoint_path = None
                     else:
                         print("   ✅ Checkpoint valid")
                 else:
                      print("   ⚠️ Checkpoint has no optimizer state (might be okay if fresh finetune but risky for resume)")
             except Exception as e:
                 print(f"   ❌ Checkpoint unreadable: {e}")
                 checkpoint_path = None

        if checkpoint_path:
            print(f"   ✅ Found valid checkpoint: {checkpoint_path}")
            print(f"   🔄 Will resume training from this checkpoint")
        else:
            print(f"   ⚠️  No valid checkpoint found in {run_output_dir} (or verification failed)")
            print("   Starting fresh training")
    else:
        print(f"   ⚠️  No checkpoint found in {run_output_dir}")
        print("   Starting fresh training")

    # Configure training via model.config (DeepForest 2.0 API)
    print("\n⚙️  Configuring training...")
    model.config.train.csv_file = train_csv
    model.config.train.root_dir = ""  # Empty for absolute paths
    model.config.train.epochs = epochs
    model.config.train.lr = lr
    model.config.batch_size = batch_size

    # Augmentation policy:
    # - Explicit augmentations param always wins (e.g. TCD training regime).
    # - Otherwise disable when wrapper is active: spatial shadow features
    #   (ShadowChannel, ShadowCrossAttention, ShadowProposals) embed the shadow
    #   direction in the forward pass, so flips/rotations would misalign it.
    #   shadow_loss_reweight only uses the angle at loss-weighting time, so it
    #   does not strictly require disabling — but we preserve existing behaviour
    #   unless augmentations are explicitly overridden.
    uses_spatial_shadow = (shadow_channel or shadow_cross_attention
                           or shadow_proposals or shadow_luma_only or shadow_input_only)
    from omegaconf import OmegaConf
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Crop 2048×2048 TCD tiles to 400×400 patches — matches the size all prior
    # phases used (pre-chipped tiles) and prevents OOM from full-tile RPN.
    CROP_SIZE    = 400
    MIN_VIS      = 0.5   # min bbox visibility fraction to survive crop
    BBOX_PARAMS  = A.BboxParams(format="pascal_voc", label_fields=["category_ids"],
                                clip=True, min_visibility=MIN_VIS)
    crop_transform = A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE, p=1.0)

    def _build_train_transform(extra_augs):
        aug_list = []
        for a in (extra_augs or []):
            name   = list(a.keys())[0] if isinstance(a, dict) else a
            params = list(a.values())[0] if isinstance(a, dict) else {}
            cls    = getattr(A, name, None)
            if cls is not None:
                aug_list.append(cls(**params) if params else cls())
        return A.Compose([crop_transform] + aug_list + [ToTensorV2()], bbox_params=BBOX_PARAMS)

    if augmentations is not None:
        model._train_transform_override = _build_train_transform(augmentations)
        model.config.train.augmentations = OmegaConf.create([])
        names = [list(a.keys())[0] if isinstance(a, dict) else a for a in augmentations]
        print(f"   🔄 Train transform: crop{CROP_SIZE} + {names} + ToTensorV2")
    elif use_wrapper:
        model._train_transform_override = _build_train_transform(None)
        model.config.train.augmentations = OmegaConf.create([])
        note = "spatial shadow active" if uses_spatial_shadow else "wrapper active"
        print(f"   🔄 Train transform: crop{CROP_SIZE} + ToTensorV2 ({note})")
    else:
        model._train_transform_override = _build_train_transform([{"HorizontalFlip": {"p": 0.5}}])
        print(f"   🔄 Train transform: crop{CROP_SIZE} + HorizontalFlip + ToTensorV2 (default)")

    if val_csv:
        model.config.validation.csv_file = val_csv
        model.config.validation.root_dir = ""  # Empty for absolute paths
        print(f"   Validation enabled: {val_csv}")
    else:
        model.config.validation.csv_file = None
        print("   No validation file provided")

    # ------------------------------------------------------------------
    # Hard-negative support: rows with blank/NaN xmin are confirmed-empty
    # images (no trees). DeepForest's utilities.read_file calls shapely on
    # every row, so we must strip them out before the CSV reaches BoxDataset.
    # We then inject the empty image paths back into image_names after the
    # dataset is built — BoxDataset already handles the zero-box case at
    # __getitem__ line 162 (np.sum(boxes)==0 → torch.zeros).
    # ------------------------------------------------------------------
    # ── Tiled validation dataset ──────────────────────────────────────────────
    # Tiles each full image into CROP_SIZE × CROP_SIZE patches so validation
    # uses the same input size as training (avoids OOM on 2048×2048 tiles).

    train_df = pd.read_csv(train_csv)
    _empty_mask = train_df["xmin"].isna() | (train_df["xmin"].astype(str).str.strip() == "")
    _empty_image_paths = train_df.loc[_empty_mask, "image_path"].unique().tolist()
    import types
    from deepforest.datasets.training import BoxDataset
    from torch.utils.data import DataLoader

    # pin_memory is a CUDA-only feature — disable on MPS and CPU to avoid overhead
    pin_memory = torch.cuda.is_available()

    if _empty_image_paths:
        print(f"\n🔲 Found {len(_empty_image_paths)} confirmed-empty (hard-negative) images — "
              f"will be injected into dataset after CSV loading")
        _clean_train_csv = "/tmp/_clean_train.csv"
        train_df[~_empty_mask].to_csv(_clean_train_csv, index=False)
        model.config.train.csv_file = _clean_train_csv

        _orig_train_dataloader = model.train_dataloader.__func__  # unbound method

        def _train_dataloader_with_empties(self_model):
            dl = _orig_train_dataloader(self_model)
            ds = dl.dataset
            ds.image_names = np.append(ds.image_names, _empty_image_paths)
            if hasattr(self_model, "_train_transform_override"):
                ds.transform = self_model._train_transform_override
            print(f"   ✅ Injected {len(_empty_image_paths)} empty images into training dataset "
                  f"(total: {len(ds.image_names)} images)")
            return DataLoader(
                ds,
                batch_size=dl.batch_size,
                shuffle=True,
                collate_fn=ds.collate_fn,
                num_workers=8,
                pin_memory=pin_memory,
                persistent_workers=True,
            )

        model.train_dataloader = types.MethodType(_train_dataloader_with_empties, model)
    else:
        _clean_train_csv = train_csv
        # No empty images — still need to inject transform override if set
        if hasattr(model, "_train_transform_override"):
            _orig_tl = model.train_dataloader.__func__
            def _train_dataloader_with_transform(self_model):
                dl = _orig_tl(self_model)
                dl.dataset.transform = self_model._train_transform_override
                return dl
            model.train_dataloader = types.MethodType(_train_dataloader_with_transform, model)

    print(f"\n📊 Loading training data from {train_csv}...")
    print(f"   Training samples: {len(train_df[~_empty_mask])} bounding boxes")
    print(f"   Annotated images: {train_df[~_empty_mask]['image_path'].nunique()}")
    print(f"   Empty images    : {len(_empty_image_paths)}")
    print(f"   Total images    : {train_df[~_empty_mask]['image_path'].nunique() + len(_empty_image_paths)}")

    if val_csv:
        val_df = pd.read_csv(val_csv)
        _val_empty_mask  = val_df["xmin"].isna() | (val_df["xmin"].astype(str).str.strip() == "")
        _val_empty_paths = val_df.loc[_val_empty_mask, "image_path"].unique().tolist()
        if _val_empty_paths:
            _clean_val_csv = "/tmp/_clean_val.csv"
            val_df[~_val_empty_mask].to_csv(_clean_val_csv, index=False)
            model.config.validation.csv_file = _clean_val_csv

            import types
            from deepforest.datasets.training import BoxDataset
            from torch.utils.data import DataLoader

            _orig_val_dataloader = model.val_dataloader.__func__

            def _val_dataloader_with_empties(self_model):
                dl  = _orig_val_dataloader(self_model)
                ds  = dl.dataset
                ds.image_names = np.append(ds.image_names, _val_empty_paths)
                tiled_ds = _SingleCropValDataset(ds, patch_size=CROP_SIZE, min_visibility=MIN_VIS)
                return DataLoader(
                    tiled_ds,
                    batch_size=dl.batch_size,
                    shuffle=False,
                    collate_fn=ds.collate_fn,
                    num_workers=8,
                    pin_memory=pin_memory,
                    persistent_workers=True,
                )

            model.val_dataloader = types.MethodType(_val_dataloader_with_empties, model)
        else:
            # No empty val images — still apply tiling
            import types
            from torch.utils.data import DataLoader
            _orig_vl = model.val_dataloader.__func__
            def _val_dataloader_tiled(self_model):
                dl = _orig_vl(self_model)
                tiled_ds = _SingleCropValDataset(dl.dataset, patch_size=CROP_SIZE, min_visibility=MIN_VIS)
                return DataLoader(tiled_ds, batch_size=dl.batch_size, shuffle=False,
                                  collate_fn=dl.dataset.collate_fn,
                                  num_workers=8, pin_memory=pin_memory, persistent_workers=True)
            model.val_dataloader = types.MethodType(_val_dataloader_tiled, model)

        print(f"\n📊 Loading validation data from {val_csv}...")
        print(f"   Validation samples: {len(val_df[~_val_empty_mask])} bounding boxes")
        print(f"   Annotated images  : {val_df[~_val_empty_mask]['image_path'].nunique()}")
        print(f"   Empty images      : {len(_val_empty_paths)}")
    else:
        _val_empty_mask  = pd.Series([], dtype=bool)
        _val_empty_paths = []

    # Print configuration
    print(f"\n⚙️  Training configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Checkpoint dir: {run_output_dir}")

    # Create callbacks
    callbacks = []

    # Monitor DeepForest's native `map` metric, which we've overridden to use
    # IoU=0.4 (better suited to aerial tree crown detection than COCO's 0.5–0.95).
    _monitor = "map"
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_output_dir,
        filename="deepforest-{epoch:02d}-{map:.2f}",
        monitor=_monitor,
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    if val_csv:
        early_stop_callback = EarlyStopping(
            monitor=_monitor,
            patience=patience,
            mode="max",
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    print("\n🚀 Starting training...")
    print("-" * 60)

    # All shadow modes (channel / cross-attention) use ShadowConditionedDeepForest
    # which is a raw LightningModule — always train with the manual Trainer path.
    from pytorch_lightning.loggers import CSVLogger
    logger = CSVLogger(save_dir=output_dir, name=run_name)

    class MetrixPrinter(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            loss = metrics.get("box_loss") or metrics.get("train_loss")
            if loss:
                print(f"   📉 Epoch {trainer.current_epoch} Loss: {loss:.4f}")

        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            mAP = metrics.get("map")
            if mAP:
                print(f"   📈 Epoch {trainer.current_epoch} mAP: {mAP:.4f}")

    callbacks.append(MetrixPrinter())

    trainer_kwargs = {
        "max_epochs": epochs,
        "enable_checkpointing": True,
        "callbacks": callbacks,
        "logger": logger,
        "check_val_every_n_epoch": 3,
        "num_sanity_val_steps": 0,
        "gradient_clip_val": 1.0,
        "fast_dev_run": fast_dev_run,
    }

    if accelerator:
        trainer_kwargs["accelerator"] = accelerator
        if accelerator == "cpu":
            trainer_kwargs["devices"] = 1
    elif torch.backends.mps.is_available():
        trainer_kwargs["accelerator"] = "mps"
        trainer_kwargs["devices"]     = 1
    elif torch.cuda.is_available():
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"]     = 1
    else:
        trainer_kwargs["accelerator"] = "cpu"

    # Mixed precision: explicit override wins; otherwise bf16-mixed on CUDA only.
    if precision:
        trainer_kwargs["precision"] = precision
    elif trainer_kwargs.get("accelerator") == "gpu":
        trainer_kwargs["precision"] = "bf16-mixed"

    trainer = pl.Trainer(**trainer_kwargs)

    # torch.compile: CUDA only — MPS inductor backend is prototype and fails on backward pass.
    if trainer_kwargs.get("accelerator") == "gpu" and torch.cuda.is_available():
        try:
            model.model = torch.compile(model.model)
            print("   ✅ torch.compile applied to detection model")
        except Exception as e:
            print(f"   ⚠️  torch.compile unavailable ({e}) — running uncompiled")

    trainer.fit(model, ckpt_path=checkpoint_path)

    print("\n✅ Training complete!")

    # Save final model
    final_model_path = Path(run_output_dir) / "deepforest_final.pth"
    print(f"💾 Saved final model to {final_model_path}")

    if use_wrapper:
        torch.save(model.state_dict(), str(final_model_path))
    else:
        torch.save(model.model.state_dict(), str(final_model_path))

    return model, None


def main():
    """CLI entrypoint for local testing."""
    parser = argparse.ArgumentParser(description="Train DeepForest model")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--accelerator",
        type=str,
        default=None,
        help="Force accelerator (cpu, gpu, mps, auto)",
    )
    parser.add_argument(
        "--shadow-channel",
        action="store_true",
        help="Run B/D: add shadow map as 4th input channel",
    )
    parser.add_argument(
        "--shadow-cross-attention",
        action="store_true",
        help="Run C/D: shadow cross-attention module after layer4",
    )
    parser.add_argument(
        "--shadow-proposals",
        action="store_true",
        help="Run F_dir: inject shadow-blob-derived proposals alongside RPN output",
    )
    parser.add_argument(
        "--shadow-proposals-iso",
        action="store_true",
        help="Ablation F_iso: use scrambled shadow direction in proposal generation",
    )
    parser.add_argument(
        "--shadow-loss-reweight",
        action="store_true",
        help="Phase 17: upweight focal loss for positive anchors of shadow-casting GT boxes",
    )
    parser.add_argument(
        "--shadow-loss-weight",
        type=float,
        default=2.0,
        help="Focal loss multiplier for shadow-casting GT anchors (default 2.0)",
    )
    parser.add_argument(
        "--shadow-angle-deg",
        type=float,
        default=None,
        help="Base shadow azimuth in degrees. Auto-derived from CSV if not set.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to initial weights (.pth or .ckpt). Auto-resumes if omitted.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze ResNet body, train FPN + head only.",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run 1 train + 1 val batch then exit (sanity check).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Lightning precision override (e.g. 16-mixed, bf16-mixed, 32).",
    )

    args = parser.parse_args()

    train_deepforest(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        pretrained=not args.no_pretrained,
        shadow_angle_deg=args.shadow_angle_deg,
        checkpoint=args.checkpoint,
        accelerator=args.accelerator,
        freeze_backbone=args.freeze_backbone,
        shadow_channel=args.shadow_channel,
        shadow_cross_attention=args.shadow_cross_attention,
        shadow_proposals=args.shadow_proposals,
        shadow_proposals_iso=args.shadow_proposals_iso,
        shadow_loss_reweight=args.shadow_loss_reweight,
        shadow_loss_weight=args.shadow_loss_weight,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
