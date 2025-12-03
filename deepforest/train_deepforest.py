#!/usr/bin/env python3
"""
DeepForest Fine-tuning Script for Tree Detection

Trains DeepForest on TCD dataset to improve detection in sparse arid landscapes.

Usage:
    python train_deepforest.py --train_csv train.csv --val_csv val.csv --epochs 10

On Modal:
    Called from modal_deepforest.py
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import torch
from deepforest import main as deepforest_main
import wandb


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
    Train a DeepForest model.

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
        wandb_project: Weights & Biases project name
    """
    # Create run-specific output directory
    run_output_dir = str(Path(output_dir) / run_name)
    Path(run_output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"🌲 DeepForest Training: {run_name}")
    print("=" * 60)

    # Initialize W&B if requested
    if wandb_project:
        wandb.init(
            project=wandb_project,
            name=run_name or f"deepforest_finetune",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "patience": patience,
            },
        )

    print("=" * 60)
    print("🌲 DeepForest Fine-tuning")
    print("=" * 60)

    # Initialize model
    print("\n📦 Initializing DeepForest model...")
    model = deepforest_main.deepforest()

    if pretrained:
        print("   Loading pretrained weights...")
        # Download pretrained weights directly from the release
        import urllib.request
        import tempfile

        weights_url = "https://github.com/weecology/DeepForest/releases/download/v1.3.0/NEON_checkpoint.pl"

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pl") as tmp_file:
                print(f"   Downloading from {weights_url}...")
                urllib.request.urlretrieve(weights_url, tmp_file.name)
                print("   Loading weights...")
                model = model.load_from_checkpoint(tmp_file.name)
                print("   ✅ Loaded pretrained weights")
        except Exception as e:
            print(f"   ⚠️  Could not load pretrained weights: {e}")
            print("   Continuing with random initialization...")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"

    # Enable Tensor Cores for faster training on A10G
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    print(f"   Using device: {device}")
    model.to(device)

    # Load training data
    print(f"\n📊 Loading training data from {train_csv}...")
    train_df = pd.read_csv(train_csv)
    print(f"   Training samples: {len(train_df)} bounding boxes")
    print(f"   Unique images: {train_df['image_path'].nunique()}")

    # Load validation data if provided
    val_df = None
    if val_csv and Path(val_csv).exists():
        print(f"\n📊 Loading validation data from {val_csv}...")
        val_df = pd.read_csv(val_csv)
        print(f"   Validation samples: {len(val_df)} bounding boxes")
        print(f"   Unique images: {val_df['image_path'].nunique()}")

    # Configure training
    model.config["train"]["csv_file"] = train_csv
    model.config["train"]["root_dir"] = ""  # Empty string for absolute paths
    model.config["train"]["epochs"] = epochs
    model.config["train"]["lr"] = lr
    model.config["batch_size"] = batch_size

    if val_df is not None:
        model.config["validation"]["csv_file"] = val_csv
        model.config["validation"]["root_dir"] = ""  # Empty string for absolute paths

    # Set up early stopping
    model.config["train"]["patience"] = patience

    print("\n⚙️  Training configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Checkpoint dir: {run_output_dir}")

    # Train!
    print("\n🚀 Starting training...")
    print("-" * 60)

    try:
        # Create trainer with custom callbacks
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        from pytorch_lightning.loggers import WandbLogger

        callbacks = [
            ModelCheckpoint(
                dirpath=run_output_dir,
                filename="deepforest-{epoch:02d}-{box_recall:.2f}",
                monitor="box_recall"
                if val_df is not None
                else "train_classification_loss",
                mode="max" if val_df is not None else "min",
                save_top_k=3,
            ),
            EarlyStopping(
                monitor="box_recall"
                if val_df is not None
                else "train_classification_loss",
                patience=patience,
                mode="max" if val_df is not None else "min",
            ),
        ]

        logger = None
        if wandb_project:
            logger = WandbLogger(project=wandb_project, name=run_name)

        # Create trainer
        model.create_trainer(
            logger=logger,
            callbacks=callbacks,
        )

        # Train
        model.trainer.fit(model)

        print("\n✅ Training complete!")

        # Save final model
        final_path = Path(run_output_dir) / "deepforest_final.pth"
        torch.save(model.model.state_dict(), final_path)
        print(f"💾 Saved final model to {final_path}")

        # Evaluate on validation set if available
        if val_df is not None:
            print("\n📈 Evaluating on validation set...")
            results = model.evaluate(
                csv_file=val_csv,
                root_dir="",  # Empty string for absolute paths
                iou_threshold=0.4,
            )

            print("\n📊 Validation Results:")
            if isinstance(results, dict):
                for metric, value in results.items():
                    if isinstance(value, (int, float)):
                        print(f"   {metric}: {value:.4f}")
                    else:
                        print(f"   {metric}: {value}")
            else:
                # Results is a DataFrame
                print(results)

            if wandb_project:
                if isinstance(results, dict):
                    wandb.log(results)
                else:
                    wandb.log({"evaluation": results.to_dict()})

        return model, results if val_df else None

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

    finally:
        if wandb_project:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train DeepForest on custom data")

    parser.add_argument(
        "--train_csv", type=str, required=True, help="Path to training CSV"
    )
    parser.add_argument("--val_csv", type=str, help="Path to validation CSV")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="deepforest_outputs",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Train from scratch (don't use pretrained weights)",
    )
    parser.add_argument(
        "--wandb_project", type=str, help="Weights & Biases project name"
    )
    parser.add_argument("--run_name", type=str, help="Custom run name for logging")

    args = parser.parse_args()

    train_deepforest(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        output_dir=args.output_dir,
        pretrained=not args.no_pretrained,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
