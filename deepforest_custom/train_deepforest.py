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
    checkpoint=None,  # Optional checkpoint path for DeepForest weights
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
        print("   🌞 Shadow conditioning: ENABLED (FiLM)")
        model = ShadowConditionedDeepForest(shadow_angle_deg=215.0)
    else:
        print("   🌞 Shadow conditioning: DISABLED (baseline)")
        model = deepforest_main.deepforest()

    # Load weights
    import traceback

    if checkpoint:
        print(f"\n📦 Loading checkpoint: {checkpoint}")
        try:
            state_dict = torch.load(checkpoint, map_location="cpu")
            # strict=False allows FiLM layers to stay random if shadow_conditioning=True
            if shadow_conditioning:
                model.deepforest.model.load_state_dict(state_dict, strict=False)
            else:
                model.model.load_state_dict(state_dict, strict=False)
            print("   ✅ Checkpoint loaded successfully")
            if shadow_conditioning:
                print("   ℹ️  FiLM layers initialized randomly (not in checkpoint)")
        except Exception as e:
            print(f"   ❌ Failed to load checkpoint: {e}")
            print(f"   Full traceback:")
            traceback.print_exc()
            raise RuntimeError("Cannot continue without loading checkpoint") from e
    elif pretrained:
        print("\n📦 Loading pretrained weights...")
        try:
            if shadow_conditioning:
                model.deepforest.create_model()
                model.deepforest.load_model("weecology/deepforest-tree")
            else:
                model.create_model()
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
        print("   🔄 Rotation augmentation: ENABLED")
        print("   ℹ️  Images, bboxes, and shadow vectors will be rotated together")
        # Custom collate will be set in the dataloader creation below
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

    # When using shadow conditioning, model is a wrapper around DeepForest
    # We need to train the inner DeepForest model (which has been patched for rotation)
    # since PyTorch Lightning expects a LightningModule, not our nn.Module wrapper
    training_model = model.deepforest if shadow_conditioning else model

    # Create trainer using DeepForest's API
    training_model.create_trainer(
        callbacks=callbacks,
        max_epochs=epochs,
        enable_checkpointing=True,
    )

    # Train the inner model (patched for rotation + FiLM hooks)
    training_model.trainer.fit(training_model, ckpt_path=checkpoint_path)

    print("\n✅ Training complete!")

    # Save final model
    final_model_path = Path(run_output_dir) / "deepforest_final.pth"
    print(f"💾 Saved final model to {final_model_path}")
    import torch

    if shadow_conditioning:
        # Save the full wrapper model (including FiLM weights)
        torch.save(model.state_dict(), str(final_model_path))
    else:
        # Save only the inner DeepForest model (backward compatibility)
        torch.save(model.model.state_dict(), str(final_model_path))

    # Evaluate on validation set if provided
    # NOTE: Skipping final evaluation due to DeepForest 2.0 pandas bug
    # PyTorch Lightning already logs validation metrics during training
    results = None
    # if val_csv:
    #     print(f"\n📈 Evaluating on validation set...")
    #     results = model.evaluate(
    #         csv_file=val_csv,
    #         root_dir=None,
    #         iou_threshold=0.4,
    #     )
    #
    #     print(f"\n📊 Validation Results:")
    #     for key, value in results.items():
    #         print(f"   {key}: {value}")

    return model, results


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
        "--shadow_conditioning",
        action="store_true",
        help="Enable FiLM shadow conditioning",
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
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
