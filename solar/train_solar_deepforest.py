import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import glob
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prototypes.solar_deepforest import SolarDeepForest
from prototypes.solar_adapter import GlobalContextEncoder


class SolarDetectionDataset(Dataset):
    """
    Simulated Dataset for Solar Detection Training.
    Returns: image, boxes, shadow_masks, sun_vector (dummy)
    In production, this would load your deepforest CSV + masks.
    """

    def __init__(self, data_dir):
        # Scan for images
        self.image_files = glob.glob(os.path.join(data_dir, "*.png"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Implementation omitted - returns dummy tensors matching DeepForest expected input
        img = torch.randn(3, 400, 400)

        # Targets format for RetinaNet
        target = {
            "boxes": torch.tensor([[50, 50, 100, 100]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }

        # Shadow mask for Level 2 supervision
        shadow_mask = torch.randn(1, 400, 400) > 0.5

        # Pre-computed Sun Vector (from Stage 0)
        sun_vector = torch.randn(2)

        return img, target, shadow_mask.float(), sun_vector


def collate_fn(batch):
    return tuple(zip(*batch))


def train_solar_stage1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Baseline: {args.baseline}")
    # 1. Initialize SolarDeepForest
    solar_df = SolarDeepForest()
    solar_df.create_model()

    # 2. Load Oscar weights (Strict=False to ignore missing solar_gates)
    # Note: DeepForest saves full checkout usually, but we might just have statedict
    try:
        # Assuming args.baseline is a torch checkpoint
        checkpoint = torch.load(args.baseline, map_location=device)
        # DeepForest checkpoints often wrap the state_dict
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Filter out incompatible keys if necessary, or just rely on strict=False
        solar_df.model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded baseline weights into Backbone/FPN/Head.")
        print("Solar Gates are randomly initialized.")

    except Exception as e:
        print(f"Warning: Could not load baseline ({e}). Starting from Scratch.")

    solar_df.model.to(device)
    solar_df.model.train()

    # 3. Freeze Backbone (ResNet) to preserve texture features
    print("Freezing Backbone...")
    for param in solar_df.model.backbone.parameters():
        param.requires_grad = False

    # Only train the new gates and maybe the head
    # For now, let's train gates + head (to adapt to gated features)
    params_to_optimize = list(solar_df.model.solar_gates.parameters()) + list(
        solar_df.model.head.parameters()
    )

    optimizer = optim.AdamW(params_to_optimize, lr=1e-4)

    # Dataset
    dataset = SolarDetectionDataset(args.data_dir)

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    print("Starting Training...")
    for epoch in range(args.epochs):
        epoch_loss = 0
        for images, targets, shadow_masks, sun_vectors in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            sun_vectors = torch.stack(sun_vectors).to(device)
            shadow_masks = torch.stack(shadow_masks).to(device)

            # Forward pass
            loss_dict = solar_df.model(images, sun_vectors, targets, shadow_masks)

            # Sum losses
            # loss_dict contains: classification, bbox_regression, loss_solar_attention
            total_loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        print(f"Epoch {epoch}: Loss {epoch_loss:.4f}")

    torch.save(solar_df.model.state_dict(), args.save_path)
    print(f"Saved SolarDeepForest to {args.save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="solar_deepforest.pth")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    train_solar_stage1(args)
