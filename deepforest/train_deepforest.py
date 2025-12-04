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
from deepforest import main as deepforest_main
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


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
):
    """
    Train a DeepForest model using DeepForest 2.0 config-based API.

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
    print("\n📦 Initializing DeepForest model...")
    model = deepforest_main.deepforest()

    # Load pretrained weights if requested
    if pretrained:
        print("   Loading pretrained weights from Hugging Face...")
        try:
            model.load_model("weecology/deepforest-tree")
            print("   ✅ Loaded pretrained weights from Hugging Face")
        except Exception as e:
            import traceback

            print(f"   ❌ FAILED to load pretrained weights!")
            print(f"   Error: {e}")
            print(f"   Full traceback:")
            traceback.print_exc()
            raise RuntimeError("Cannot continue without pretrained weights") from e

    # Configure training via model.config (DeepForest 2.0 API)
    print("\n⚙️  Configuring training...")
    model.config.train.csv_file = train_csv
    model.config.train.root_dir = ""  # Empty for absolute paths
    model.config.train.epochs = epochs
    model.config.train.lr = lr
    model.config.batch_size = batch_size

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

    # Create trainer using DeepForest's API
    model.create_trainer(
        callbacks=callbacks,
        max_epochs=epochs,
        enable_checkpointing=True,
    )

    # Train the model
    model.trainer.fit(model)

    print("\n✅ Training complete!")

    # Save final model
    final_model_path = Path(run_output_dir) / "deepforest_final.pth"
    print(f"💾 Saved final model to {final_model_path}")
    import torch

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
    )


if __name__ == "__main__":
    main()
