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

    if shadow_conditioning:
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
        print("   Shadow conditioning: ENABLED (FiLM)")
        model = ShadowConditionedDeepForest(
            shadow_angle_deg=shadow_angle_deg, train_csv=train_csv, film_lr=film_lr
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
                if shadow_conditioning:
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

    # Auto-detect and resume from checkpoint if it exists
    # This handles Modal auto-restarts gracefully
    checkpoint_path = None
    print("\n🔍 Searching for checkpoint to resume from...")
    checkpoint_files = list(Path(run_output_dir).glob("*.ckpt"))
    if checkpoint_files:
        # Get most recent checkpoint by modification time
        checkpoint_path = str(max(checkpoint_files, key=lambda p: p.stat().st_mtime))
        print(f"   ✅ Found checkpoint: {checkpoint_path}")
        print(f"   🔄 Will resume training from this checkpoint")
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

    # Configure rotation augmentation for FiLM training
    if shadow_conditioning:
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

    if shadow_conditioning:
        # ── FiLM path ─────────────────────────────────────────────────────────
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

    if shadow_conditioning:
        # Save the full wrapper model (including FiLM weights)
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
    )


if __name__ == "__main__":
    main()
