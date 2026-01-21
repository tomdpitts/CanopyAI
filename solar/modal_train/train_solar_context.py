import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import torchvision
from torch.utils.data import DataLoader, Dataset
import glob
import os
from PIL import Image
import math


class SolarContextTrainer:
    """
    Self-Supervised Trainer for Solar Context Encoder.
    (no annotations required)

    Objective:
    If image is rotated by Theta, the predicted sun vector
    must also rotate by Theta.
    """

    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Cosine Similarity is better for vectors than MSE
        self.criterion = nn.CosineEmbeddingLoss()

    def train_step(self, images, num_rotations=4):
        """
        Multi-rotation training step.

        For each image, apply N random rotations and predict vectors.
        Enforce that all pairs of predictions are correctly rotated.
        This prevents the "constant output" degenerate solution.

        images: Batch of orthomosaic crops (B, 3, 500, 500)
        num_rotations: Number of random rotations per image (default 4)
        """
        batch_size = images.shape[0]
        device = images.device

        # Generate N random angles for this step
        angles_deg = torch.rand(num_rotations) * 360.0  # (N,) random angles
        angles_rad = angles_deg * (math.pi / 180.0)

        # Collect predictions for each rotation
        predictions = []  # List of (B, 2) tensors

        for i, angle_deg in enumerate(angles_deg):
            if i == 0:
                # First "rotation" is identity (original image)
                rotated_images = images
            else:
                rotated_images = TF.rotate(images, angle_deg.item())

            v = self.model(rotated_images)  # (B, 2)
            predictions.append(v)

        # Compute pairwise consistency loss
        # For each pair (i, j), v_j should equal R(theta_j - theta_i) @ v_i
        total_loss = 0.0
        num_pairs = 0

        for i in range(num_rotations):
            for j in range(i + 1, num_rotations):
                # Rotation from i to j
                delta_angle = angles_rad[j] - angles_rad[i]
                c, s = torch.cos(delta_angle), torch.sin(delta_angle)
                R = torch.tensor([[c, -s], [s, c]], device=device)

                # Rotate prediction i to match prediction j
                v_i = predictions[i]  # (B, 2)
                v_j = predictions[j]  # (B, 2)
                v_i_rotated = torch.matmul(v_i, R.t())  # (B, 2)

                # Loss: v_i_rotated should match v_j
                pair_loss = self.criterion(
                    v_i_rotated, v_j, target=torch.ones(batch_size, device=device)
                )
                total_loss += pair_loss
                num_pairs += 1

        # Average over all pairs
        loss = total_loss / num_pairs

        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_val_loss(self, images, num_rotations=4):
        """
        Compute validation loss without gradient updates.
        Same multi-rotation approach as train_step.
        """
        batch_size = images.shape[0]
        device = images.device

        angles_deg = torch.rand(num_rotations) * 360.0
        angles_rad = angles_deg * (math.pi / 180.0)

        predictions = []
        for i, angle_deg in enumerate(angles_deg):
            if i == 0:
                rotated_images = images
            else:
                rotated_images = TF.rotate(images, angle_deg.item())
            v = self.model(rotated_images)
            predictions.append(v)

        total_loss = 0.0
        num_pairs = 0

        for i in range(num_rotations):
            for j in range(i + 1, num_rotations):
                delta_angle = angles_rad[j] - angles_rad[i]
                c, s = torch.cos(delta_angle), torch.sin(delta_angle)
                R = torch.tensor([[c, -s], [s, c]], device=device)

                v_i_rotated = torch.matmul(predictions[i], R.t())
                pair_loss = self.criterion(
                    v_i_rotated,
                    predictions[j],
                    target=torch.ones(batch_size, device=device),
                )
                total_loss += pair_loss.item()
                num_pairs += 1

        return total_loss / num_pairs


# Dataset for pre-tiled training images (already 500x500)
class RawOrthoDataset(Dataset):
    """
    Dataset that loads pre-tiled 500x500 training images.
    No resizing needed - tiles are already at the correct size.
    """

    def __init__(self, image_dir, target_size=500):
        self.target_size = target_size
        patterns = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]
        self.files = []
        for p in patterns:
            self.files.extend(glob.glob(os.path.join(image_dir, p)))

        # Just convert to tensor, no resize needed for pre-tiled data
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),  # Converts to [0, 1] range
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load image using PIL (supports TIF files)
        img_path = self.files[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
            return img_tensor
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy tensor on error to avoid crashing training
            return torch.zeros(3, self.target_size, self.target_size)


if __name__ == "__main__":
    import argparse
    import sys

    # Add parent directory to path to find solar_adapter
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from solar_adapter import GlobalContextEncoder

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="solar_context.pth")
    args = parser.parse_args()

    # Init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlobalContextEncoder().to(device)
    trainer = SolarContextTrainer(model)
    dataset = RawOrthoDataset(args.image_dir)

    # Handle empty dataset for prototype testing
    if len(dataset) == 0:
        print("Warning: No images found. Creating dummy dataset.")
        dataset.files = ["dummy.tif"] * 10

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Starting Self-Supervised Training on {len(dataset)} images...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            loss = trainer.train_step(batch)
            total_loss += loss

        print(f"Epoch {epoch}: Loss {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Saved to {args.save_path}")
