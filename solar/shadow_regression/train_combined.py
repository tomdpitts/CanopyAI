#!/usr/bin/env python3
"""
Train shadow regression model on COMBINED WON003 + SJER dataset with rotation augmentation.
"""

import os
import math
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models
from tqdm import tqdm

from local_dataset import LocalShadowDataset, SimpleTransform

import numpy as np
from torchvision import transforms


class RotationTransform:
    """Rotates the image and target vector by the same random angle."""

    def __init__(self, size=224):
        self.size = size
        self.resize = transforms.Resize((size, size))
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, target_vector):
        angle = np.random.uniform(0, 360)
        image = self.resize(image)
        image = transforms.functional.rotate(image, angle)
        image = self.to_tensor(image)

        c = math.cos(math.radians(angle))
        s = math.sin(math.radians(angle))

        x, y = target_vector[0], target_vector[1]
        x_new = x * c - y * s
        y_new = x * s + y * c

        target_vector = torch.tensor([x_new, y_new], dtype=torch.float32)
        target_vector = target_vector / (torch.norm(target_vector) + 1e-6)

        return image, target_vector


class ShadowResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.backbone(x)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=0.001,
    output_dir="output",
    device="cpu",
):
    """Train with rotation augmentation."""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_model_path = os.path.join(output_dir, "shadow_model_combined_best.pth")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False
        )
        for images, targets in train_bar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_angular_errors = []

        with torch.no_grad():
            val_bar = tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False
            )
            for images, targets in val_bar:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

                pred_angle = torch.atan2(outputs[:, 0], outputs[:, 1]) * 180 / math.pi
                true_angle = torch.atan2(targets[:, 0], targets[:, 1]) * 180 / math.pi
                pred_angle = (pred_angle + 360) % 360
                true_angle = (true_angle + 360) % 360
                diff = (pred_angle - true_angle).abs()
                error = torch.min(diff, 360 - diff)
                val_angular_errors.append(error)

        val_loss /= max(len(val_loader), 1)
        mean_angular_error = (
            torch.cat(val_angular_errors).mean().item()
            if val_angular_errors
            else float("nan")
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Angular Error: {mean_angular_error:.2f}°"
        )

        if val_loss < best_val_loss and len(val_loader) > 0:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train on combined WON003 + SJER with rotation augmentation"
    )
    parser.add_argument(
        "--won003_images", type=str, default="../../input_data/WON003/images"
    )
    parser.add_argument("--won003_csv", type=str, default="data/won003_annotations.csv")
    parser.add_argument(
        "--sjer_images", type=str, default="annotation_data/sjer_images"
    )
    parser.add_argument("--sjer_csv", type=str, default="data/sjer_annotations.csv")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.2)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load WON003 dataset
    print("Loading WON003 dataset...")
    won003_dataset = LocalShadowDataset(
        image_dir=args.won003_images,
        annotation_csv=args.won003_csv,
        transform=None,
    )
    print(f"  WON003: {len(won003_dataset)} images")

    # Load SJER dataset
    print("Loading SJER dataset...")
    sjer_dataset = LocalShadowDataset(
        image_dir=args.sjer_images,
        annotation_csv=args.sjer_csv,
        transform=None,
    )
    print(f"  SJER: {len(sjer_dataset)} images")

    # Combine datasets
    combined_dataset = ConcatDataset([won003_dataset, sjer_dataset])
    total_images = len(combined_dataset)
    print(f"\nCombined dataset: {total_images} images")
    print(f"  WON003: {len(won003_dataset)} (shadows ~215°, desert shrubland)")
    print(f"  SJER: {len(sjer_dataset)} (shadows ~0-60° & ~330-360°, oak woodland)\n")

    # Split train/val
    n_val = int(total_images * args.val_split)
    n_train = total_images - n_val

    indices = torch.randperm(total_images).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create train dataset with rotation augmentation
    train_won003 = LocalShadowDataset(
        args.won003_images, args.won003_csv, RotationTransform()
    )
    train_sjer = LocalShadowDataset(
        args.sjer_images, args.sjer_csv, RotationTransform()
    )
    train_combined = ConcatDataset([train_won003, train_sjer])

    # Filter to train indices
    train_dataset = torch.utils.data.Subset(train_combined, train_indices)

    # Create val dataset without rotation
    val_won003 = LocalShadowDataset(
        args.won003_images, args.won003_csv, SimpleTransform()
    )
    val_sjer = LocalShadowDataset(args.sjer_images, args.sjer_csv, SimpleTransform())
    val_combined = ConcatDataset([val_won003, val_sjer])

    # Filter to val indices
    val_dataset = torch.utils.data.Subset(val_combined, val_indices)

    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val\n")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Train
    model = ShadowResNet34()
    train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        device=device,
    )

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(args.output_dir, f"shadow_model_combined_{timestamp}.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
