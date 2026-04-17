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
from pathlib import Path
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

    # Disable default augmentations when using the wrapper (shadow modes or WON shrink).
    # Flips/rotations would misalign the shadow vector stored in the CSV.
    if use_wrapper:
        model.config.train.augmentations = []
        print("   🔄 Default augmentations disabled (wrapper model active)")
    else:
        print("   🔄 Default augmentations: active (raw baseline mode)")

    if val_csv:
        model.config.validation.csv_file = val_csv
        model.config.validation.root_dir = ""  # Empty for absolute paths
        print(f"   Validation enabled: {val_csv}")
    else:
        model.config.validation.csv_file = None
        print("   No validation file provided")

    # Load training data info
    train_df = pd.read_csv(train_csv)
    print(f"\n📊 Loading training data from {train_csv}...")
    print(f"   Training samples: {len(train_df)} bounding boxes")
    print(f"   Unique images: {train_df['image_path'].nunique()}")

    if val_csv:
        val_df = pd.read_csv(val_csv)
        print(f"\n📊 Loading validation data from {val_csv}...")
        print(f"   Validation samples: {len(val_df)} bounding boxes")
        print(f"   Unique images: {val_df['image_path'].nunique()}")

    # Print configuration
    print(f"\n⚙️  Training configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Checkpoint dir: {run_output_dir}")

    # Create callbacks
    callbacks = []

    # Model checkpoint callback - save best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_output_dir,
        filename="deepforest-{epoch:02d}-{map:.2f}",
        monitor="map",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    if val_csv:
        early_stop_callback = EarlyStopping(
            monitor="map",
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
        "check_val_every_n_epoch": 1,
        "num_sanity_val_steps": 0,
        "gradient_clip_val": 1.0,
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

    trainer = pl.Trainer(**trainer_kwargs)
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
    )


if __name__ == "__main__":
    main()
