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
    Uses InfoNCE Contrastive Loss to prevent collapse to constant output.

    Objective:
    - POSITIVE: Same image at different rotations should match (after rotation)
    - NEGATIVE: Different images should have different vectors
    """

    def __init__(self, model, learning_rate=1e-4, temperature=0.1):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.temperature = temperature  # Temperature for InfoNCE

    def infonce_loss(self, anchor, positive, negatives):
        """
        InfoNCE contrastive loss.

        anchor: (B, 2) - anchor predictions
        positive: (B, 2) - positive predictions (should match anchor)
        negatives: (B, N, 2) - negative predictions (should NOT match anchor)

        Returns scalar loss
        """
        # Normalize all vectors to unit length
        anchor = torch.nn.functional.normalize(anchor, dim=1)
        positive = torch.nn.functional.normalize(positive, dim=1)
        negatives = torch.nn.functional.normalize(negatives, dim=2)

        # Positive similarity: (B,)
        pos_sim = (anchor * positive).sum(dim=1) / self.temperature

        # Negative similarities: (B, N)
        neg_sim = (
            torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2) / self.temperature
        )

        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = -pos + log(exp(pos) + sum(exp(neg)))
        # = -pos + logsumexp([pos, neg1, neg2, ...])

        # Combine pos and neg for logsumexp
        all_logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+N)

        # Loss = -pos_sim + logsumexp(all_logits)
        loss = -pos_sim + torch.logsumexp(all_logits, dim=1)

        return loss.mean()

    def train_step(self, images, num_rotations=5):
        """
        Contrastive training step with same-image rotation negatives.

        For each image:
        - Apply N random rotations, get predictions
        - For each anchor (rotation i) and positive (rotation j):
          - Anchor_rotated = R(delta) @ prediction[i]
          - Positive = prediction[j] (should match anchor_rotated)
          - Negatives = prediction[k] for k != j (same image, wrong rotations)

        This forces the model to distinguish rotations of the SAME image,
        where rotation is the ONLY varying factor.

        images: Batch of orthomosaic crops (B, 3, 500, 500)
        num_rotations: Number of random rotations per image (default 5)
        """
        batch_size = images.shape[0]
        device = images.device

        # Generate N random angles for this step
        angles_deg = torch.rand(num_rotations) * 360.0
        angles_rad = angles_deg * (math.pi / 180.0)

        # Collect predictions for each rotation: list of (B, 2) tensors
        predictions = []
        for rot_idx, angle_deg in enumerate(angles_deg):
            if rot_idx == 0:
                rotated_images = images
            else:
                rotated_images = TF.rotate(images, angle_deg.item())
            v = self.model(rotated_images)
            predictions.append(v)

        # Stack predictions: (N, B, 2)
        predictions = torch.stack(predictions, dim=0)

        total_loss = 0.0
        num_pairs = 0

        # For each anchor rotation i, positive rotation j
        for i in range(num_rotations):
            for j in range(num_rotations):
                if i == j:
                    continue

                # Rotation matrix from i to j
                delta_angle = angles_rad[j] - angles_rad[i]
                c, s = torch.cos(delta_angle), torch.sin(delta_angle)
                R = torch.tensor([[c, -s], [s, c]], device=device, dtype=torch.float32)

                # Anchor: predictions at rotation i, rotated to match j
                anchor_rotated = torch.matmul(predictions[i], R.t())  # (B, 2)

                # Positive: predictions at rotation j (should match)
                positive = predictions[j]  # (B, 2)

                # Negatives: predictions at ALL OTHER rotations (k != j)
                # Same image, wrong rotation
                neg_indices = [k for k in range(num_rotations) if k != j]
                negatives = predictions[neg_indices]  # (N-1, B, 2)
                # Transpose to (B, N-1, 2) for infonce_loss
                negatives = negatives.permute(1, 0, 2)

                # Compute InfoNCE loss
                loss = self.infonce_loss(anchor_rotated, positive, negatives)
                total_loss += loss
                num_pairs += 1

        # Average loss
        avg_loss = total_loss / num_pairs

        # Optimization
        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()

        return avg_loss.item()

    def compute_val_loss(self, images, num_rotations=5):
        """
        Compute validation loss without gradient updates.
        Same same-image-rotation-negatives approach as train_step.
        """
        device = images.device

        angles_deg = torch.rand(num_rotations) * 360.0
        angles_rad = angles_deg * (math.pi / 180.0)

        predictions = []
        for rot_idx, angle_deg in enumerate(angles_deg):
            if rot_idx == 0:
                rotated_images = images
            else:
                rotated_images = TF.rotate(images, angle_deg.item())
            v = self.model(rotated_images)
            predictions.append(v)

        predictions = torch.stack(predictions, dim=0)  # (N, B, 2)

        total_loss = 0.0
        num_pairs = 0

        for i in range(num_rotations):
            for j in range(num_rotations):
                if i == j:
                    continue

                delta_angle = angles_rad[j] - angles_rad[i]
                c, s = torch.cos(delta_angle), torch.sin(delta_angle)
                R = torch.tensor([[c, -s], [s, c]], device=device, dtype=torch.float32)

                anchor_rotated = torch.matmul(predictions[i], R.t())
                positive = predictions[j]

                neg_indices = [k for k in range(num_rotations) if k != j]
                negatives = predictions[neg_indices].permute(1, 0, 2)

                loss = self.infonce_loss(anchor_rotated, positive, negatives)
                total_loss += loss.item()
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
