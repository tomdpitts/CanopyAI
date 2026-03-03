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
    from .models import (
        SolarAttentionBlock,
        ShadowConditionedDeepForest,
    )
except ImportError:
    # Fallback if run as script
    from models import (
        SolarAttentionBlock,
        ShadowConditionedDeepForest,
    )


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
    shadow_conditioning=False,  # Enable FiLM conditioning
    shadow_angle_deg=None,  # Base shadow azimuth (0=North CW); auto-derived from CSV if None
    film_lr=1e-4,  # Learning rate for FiLM blocks (lower than backbone to prevent explosion)
    checkpoint=None,  # Optional checkpoint path for DeepForest weights
    accelerator=None,  # Force accelerator (cpu, gpu, mps)
    freeze_backbone=False,  # If True, freeze backbone weights (train only FiLM/Aux)
    aux_loss_weight=1.0,  # Weight of auxiliary shadow-prediction loss (0.0 = disabled)
    shadow_channel=False,   # If True, add shadow map as 4th input channel (Phase 6)
    use_film=False,         # Defaults to False. If True, use FiLM blocks.
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
        shadow_conditioning: If True, use FiLM conditioning with constant 215° shadow
        checkpoint: Optional path to DeepForest checkpoint file to load initial weights from
        shadow_channel: If True, add directional shadow map as 4th input channel (Phase 6)
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

    if shadow_conditioning or shadow_channel:
        # Auto-derive base shadow angle from CSV if not explicitly provided
        if shadow_angle_deg is None:
            _df = pd.read_csv(train_csv)
            if "shadow_angle" in _df.columns:
                shadow_angle_deg = float(_df["shadow_angle"].mode()[0])
                print(
                    f"   Shadow angle auto-derived from CSV: {shadow_angle_deg:.1f} deg (azimuth 0=North CW)"
                )
            else:
                shadow_angle_deg = 215.0
                print(
                    f"   Shadow angle defaulted to {shadow_angle_deg} deg (no shadow_angle column in CSV)"
                )
        else:
            print(
                f"   Shadow angle (explicit): {shadow_angle_deg} deg (azimuth 0=North CW)"
            )
        print(f"   Shadow conditioning (FiLM): {'ENABLED' if use_film else 'DISABLED'}")

        # Compute global shadow normalization stats from a sample of training images.
        # This is done BEFORE model init so every _prepend_shadow_channel call
        # uses consistent scales rather than renormalizing per-tile.
        shadow_norm_stats = None
        if shadow_channel and shadow_angle_deg is not None:
            try:
                import rasterio, random
                try:
                    from utils import compute_shadow_normalization_stats
                except ImportError:
                    from deepforest_custom.utils import compute_shadow_normalization_stats

                _df_for_stats = pd.read_csv(train_csv)
                _unique_imgs  = _df_for_stats["image_path"].unique().tolist()
                N_SAMPLE = min(5, len(_unique_imgs))
                _sample_imgs = random.sample(_unique_imgs, N_SAMPLE)
                print(f"   📊 Computing shadow normalization stats from {N_SAMPLE} training images...")

                all_dg, all_dark = [], []
                _img_dir = str(Path(train_csv).parent)

                for img_path in _sample_imgs:
                    full_path = img_path if Path(img_path).is_absolute() else str(Path(_img_dir) / img_path)
                    try:
                        with rasterio.open(full_path) as _src:
                            _img = _src.read([1, 2, 3]).transpose(1, 2, 0)
                        _stats = compute_shadow_normalization_stats(_img, shadow_angle_deg)
                        all_dg.append(_stats[0])
                        all_dark.append(_stats[1])
                    except Exception as _e:
                        print(f"      ⚠️  Skipping {img_path}: {_e}")

                if all_dg:
                    # Use 95th pct across sampled images for a robust global scale
                    import numpy as _np
                    g_dg   = float(_np.percentile(all_dg,   95))
                    g_dark = float(_np.percentile(all_dark, 95))
                    # Re-run on the first image to get a representative Otsu ctr
                    with rasterio.open(
                        _sample_imgs[0] if Path(_sample_imgs[0]).is_absolute()
                        else str(Path(_img_dir) / _sample_imgs[0])
                    ) as _src:
                        _img0 = _src.read([1, 2, 3]).transpose(1, 2, 0)
                    _, _, g_otsu = compute_shadow_normalization_stats(_img0, shadow_angle_deg)
                    # Clamp dg/dark to at least the single-image minimums
                    g_dg   = max(g_dg, 30.0)
                    g_dark = max(g_dark, 10.0)
                    shadow_norm_stats = (g_dg, g_dark, g_otsu)
                    print(f"   ✅ shadow_norm_stats: dg={g_dg:.1f}  dark={g_dark:.1f}  otsu={g_otsu:.3f}")
                else:
                    print("   ⚠️  Could not load any training images for shadow stats — falling back to per-tile")
            except Exception as _e:
                print(f"   ⚠️  shadow_norm_stats computation failed ({_e}) — falling back to per-tile")

        model = ShadowConditionedDeepForest(
            shadow_angle_deg=shadow_angle_deg,
            train_csv=train_csv,
            val_csv=val_csv,
            film_lr=film_lr,
            freeze_backbone=freeze_backbone,
            aux_loss_weight=aux_loss_weight,
            shadow_channel=shadow_channel,
            use_film=use_film,
            shadow_norm_stats=shadow_norm_stats,
        )
    else:
        print("   Shadow conditioning: DISABLED (baseline)")
        model = deepforest_main.deepforest()

    # Load weights
    import traceback

    if checkpoint:
        print(f"\n Loading checkpoint: {checkpoint}")
        try:
            state_dict = torch.load(checkpoint, map_location="cpu")
            ck_keys = list(state_dict.keys())
            has_film = any("film_blocks" in k for k in ck_keys)
            has_model_prefix = any(k.startswith("model.") for k in ck_keys)

            if has_film or has_model_prefix:
                # Full ShadowConditionedDeepForest state dict (e.g. oscar50_film_finetune)
                # Keys: model.backbone.body.* and film_blocks.P3.*
                # Load into the wrapper directly (with strict=False in case architecture changed)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"   Loaded full FiLM checkpoint (backbone + FiLM weights)")
                if missing:
                    print(f"   Missing keys ({len(missing)}): {missing[:3]}...")
                if unexpected:
                    print(
                        f"   Unexpected keys ({len(unexpected)}): {unexpected[:3]}..."
                    )
            else:
                # Backbone-only state dict (e.g. oscar50.pth): bare keys like backbone.body.*
                # Load into model.model (the inner DeepForest backbone); FiLM stays random
                model.model.load_state_dict(state_dict, strict=False)
                print(f"   Loaded backbone-only checkpoint ({len(ck_keys)} keys)")
                if use_film:
                    print("   FiLM layers initialized randomly (not in checkpoint)")

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

    # Widen first conv to 4 channels AFTER weights are loaded (shadow_channel mode)
    if shadow_channel:
        print("\n🔑 Shadow channel mode: widening first conv to 4 channels...")
        widen_first_conv_for_shadow_channel(model)

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

    # Configure rotation augmentation for FiLM/shadow channel training
    if shadow_conditioning or shadow_channel:
        print("   🔄 Rotation augmentation: PRE-COMPUTED (in CSV)")
        print(
            "   ℹ️  Disabling default DeepForest augmentations to prevent shadow vector mismatch"
        )
        # Critical: DeepForest's default augmentations (horizontal flip) would flip the image
        # but NOT the shadow vector, causing the model to learn incorrect associations.
        # We perform rotation augmentation by pre-generating images in the CSV instead.
        model.config.train.augmentations = []  # Remove default HorizontalFlip (breaks shadow vector alignment)
    else:
        print("   🔄 Rotation augmentation: DISABLED (baseline mode)")

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

    print(f"\n🚀 Starting training...")
    print("-" * 60)

    if shadow_conditioning or shadow_channel:
        # ── FiLM / Shadow Channel path ─────────────────────────────────────────
        # ShadowConditionedDeepForest is a raw LightningModule, so we must build
        # the trainer manually.  Gradient clipping is kept here because FiLM
        # hooks can cause gradient explosion.
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
            "gradient_clip_val": 1.0,  # FiLM-only: prevents hook-driven explosion
        }

        if accelerator:
            print(f"   🖥️  Forcing accelerator: {accelerator}")
            trainer_kwargs["accelerator"] = accelerator
            if accelerator == "cpu":
                trainer_kwargs["devices"] = 1
        elif torch.backends.mps.is_available():
            trainer_kwargs["accelerator"] = "mps"
            trainer_kwargs["devices"] = 1
        elif torch.cuda.is_available():
            trainer_kwargs["accelerator"] = "gpu"
            trainer_kwargs["devices"] = 1
        else:
            trainer_kwargs["accelerator"] = "cpu"

        model.trainer = pl.Trainer(**trainer_kwargs)
        model.trainer.fit(model, ckpt_path=checkpoint_path)

    else:
        # ── Baseline path ─────────────────────────────────────────────────────
        # Use DeepForest's own create_trainer() so that all internal hooks,
        # metric keys, and data-loader wiring are set up correctly.
        # This matches oscar_archive exactly.
        trainer_kwargs = {
            "callbacks": callbacks,
            "max_epochs": epochs,
            "enable_checkpointing": True,
            # No gradient_clip_val: standard fine-tuning doesn't need it
        }

        if accelerator:
            print(f"   🖥️  Forcing accelerator: {accelerator}")
            trainer_kwargs["accelerator"] = accelerator
            if accelerator == "cpu":
                trainer_kwargs["devices"] = 1
        elif torch.backends.mps.is_available():
            trainer_kwargs["accelerator"] = "mps"
            trainer_kwargs["devices"] = 1
        elif torch.cuda.is_available():
            trainer_kwargs["accelerator"] = "gpu"
            trainer_kwargs["devices"] = 1

        model.create_trainer(**trainer_kwargs)
        model.trainer.fit(model, ckpt_path=checkpoint_path)

    print("\n✅ Training complete!")

    # Save final model
    final_model_path = Path(run_output_dir) / "deepforest_final.pth"
    print(f"💾 Saved final model to {final_model_path}")

    if shadow_conditioning or shadow_channel:
        # Save the full wrapper model (including FiLM weights / shadow conv)
        torch.save(model.state_dict(), str(final_model_path))
    else:
        # Save only the inner DeepForest model (backward compatibility)
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
        "--shadow_conditioning",
        action="store_true",
        help="Enable FiLM shadow conditioning",
    )
    parser.add_argument(
        "--shadow_angle_deg",
        type=float,
        default=None,
        help="Base shadow azimuth in degrees (0=North CW). Auto-derived from CSV shadow_angle column if not set.",
    )
    parser.add_argument(
        "--film_lr",
        type=float,
        default=1e-4,
        help="Learning rate for FiLM blocks (default: 1e-4, lower than backbone lr to prevent explosion)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone weights and only train FiLM/Aux head",
    )
    parser.add_argument(
        "--aux-loss-weight",
        type=float,
        default=1.0,
        help="Weight of auxiliary shadow-prediction loss (default: 1.0, set 0.0 to disable)",
    )
    parser.add_argument(
        "--shadow_channel",
        action="store_true",
        help="Add directional shadow map as 4th input channel (Phase 6)",
    )
    parser.add_argument(
        "--use-film",
        action="store_true",
        help="Enable FiLM conditioning blocks (default is OFF for clean shadow_channel ablation)",
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
        shadow_conditioning=args.shadow_conditioning,
        shadow_angle_deg=args.shadow_angle_deg,
        film_lr=args.film_lr,
        checkpoint=args.checkpoint,
        accelerator=args.accelerator,
        freeze_backbone=args.freeze_backbone,
        aux_loss_weight=args.aux_loss_weight,
        shadow_channel=args.shadow_channel,
        use_film=args.use_film,
    )


if __name__ == "__main__":
    main()
