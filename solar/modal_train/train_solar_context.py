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

    def train_step(self, images):
        """
        images: Batch of orthomosaic crops (B, 3, 1024, 1024)
        """
        batch_size = images.shape[0]
        device = images.device

        # 1. Forward Pass on Original Images
        # V1 = Predicted Sun Vector for original image
        v1 = self.model(images)  # (B, 2)

        # 2. Rotation Augmentation
        # Pick a random angle for each image in batch (simplification: same angle for batch)
        # Let's rotate by 90 degrees for simplicity (easy tensor swap)
        # Or proper rotation:
        angle_deg = 90.0
        angle_rad = math.radians(angle_deg)

        # Rotate images
        images_rotated = TF.rotate(images, angle_deg)

        # 3. Forward Pass on Rotated Images
        # V2 = Predicted Sun Vector for rotated image
        v2 = self.model(images_rotated)  # (B, 2)

        # 4. Rotate V1 to match V2
        # If the model features are consistent, V1 rotated by 90 should equal V2.
        # Rotation Matrix R for +90 degrees (counter-clockwise)
        # | cos -sin |  | 0 -1 |
        # | sin  cos |  | 1  0 |

        # Generic rotation matrix for batch
        # v_rot = v_original @ R.T
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        # Rotation matrix for (x, y)
        R = torch.tensor([[c, -s], [s, c]], device=device)

        # Apply rotation to V1
        # v1 is (B, 2), R is (2, 2)
        # (B, 2) x (2, 2) -> (B, 2)
        v1_rotated = torch.matmul(v1, R.t())

        # 5. Loss Function
        # We want v1_rotated to be 'close' to v2
        # Target is 1.0 (perfect alignment)
        loss = self.criterion(v1_rotated, v2, target=torch.ones(batch_size).to(device))

        # 6. Optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


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
