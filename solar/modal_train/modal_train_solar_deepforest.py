import modal
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Note: These imports are deferred to inside the Modal function
# where sys.path is set up correctly

app = modal.App("solar-deepforest-training")

image = (
    modal.Image.debian_slim()
    # Install dependencies matching DeepForest requirements
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "albumentations",
        "deepforest",
        "opencv-python-headless",
    )
    .add_local_dir(
        "..",
        remote_path="/root/canopyAI",
        ignore=[
            # Large files
            "*.tif",
            "*.tiff",
            "*.pth",
            "*.png",
            "*.jpg",
            "*.jpeg",
            # Directories
            ".git",
            "__pycache__",
            "venv*",
            ".mypy_cache",
            "data/",
            "input_data/",
            "input_data_small/",
            "models/",
            "samples/",
            "quebec/",
            "won_samples/",
            "test_rawortho_output/",
            "test_rawortho_output.png",
            "won003*",
            # Other
            "*.pyc",
            ".DS_Store",
        ],
    )
)

data_volume = modal.Volume.from_name("canopyai-deepforest-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    "canopyai-deepforest-checkpoints", create_if_missing=True
)


@app.function(
    image=image,
    gpu="A100",  # Use A100 for faster training
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    timeout=86400,
)
def train_solar_df_modal(
    baseline_path: str = "/checkpoints/model_oscar50.pth",
    data_dir: str = "/data/won003_train",
    epochs: int = 20,
    patience: int = 5,
    batch_size: int = 8,
    run_name: str = "you_forgot_to_add_run_name2",
):
    os.chdir("/root/canopyAI")
    sys.path.append("/root/canopyAI")
    sys.path.append("/root/canopyAI/solar")
    sys.path.append("/root/canopyAI/solar/modal_train")

    # Deferred imports after sys.path setup
    from solar_deepforest import SolarDeepForest
    from train_solar_deepforest import SolarDetectionDataset, collate_fn

    print(f"🚀 Starting Solar DeepForest Training: {run_name}")

    # 1. Setup Checkpoint Dir
    ckpt_dir = Path(f"/checkpoints/solar_deepforest/{run_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_dir / "last.pth"
    best_ckpt_path = ckpt_dir / "best.pth"

    # 2. Initialize Model
    device = torch.device("cuda")
    solar_df = SolarDeepForest()
    solar_df.create_model()

    # 3. Resume or Load Baseline
    start_epoch = 0
    best_loss = float("inf")
    epochs_no_improve = 0

    optimizer = optim.AdamW(solar_df.model.parameters(), lr=1e-4)  # Re-init logic below

    if last_ckpt_path.exists():
        print(f"🔄 Resuming from {last_ckpt_path}")
        checkpoint = torch.load(last_ckpt_path)
        solar_df.model.load_state_dict(checkpoint["model_state_dict"])
        # Re-create optimizer to match parameters then load state
        params = list(solar_df.model.solar_gates.parameters()) + list(
            solar_df.model.head.parameters()
        )
        optimizer = optim.AdamW(params, lr=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        epochs_no_improve = checkpoint.get("patience_counter", 0)

    else:
        print(f"🆕 Loading Baseline: {baseline_path}")
        if os.path.exists(baseline_path):
            state = torch.load(baseline_path, map_location=device)
            # Handle standard DeepForest checkpoint wrapping
            if "state_dict" in state:
                state = state["state_dict"]
            solar_df.model.load_state_dict(state, strict=False)

            # Freeze setup
            for param in solar_df.model.backbone.parameters():
                param.requires_grad = False

            # Init optimizer for trainable parts only
            params = list(solar_df.model.solar_gates.parameters()) + list(
                solar_df.model.head.parameters()
            )
            optimizer = optim.AdamW(params, lr=1e-4)
        else:
            raise FileNotFoundError(f"Baseline not found: {baseline_path}")

    solar_df.model.to(device)

    # 4. Data Loading
    dataset = SolarDetectionDataset(data_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4
    )
    print(f"📊 Dataset size: {len(dataset)}")

    # 5. Training Loop
    for epoch in range(start_epoch, epochs):
        solar_df.model.train()
        epoch_loss = 0

        for images, targets, masks, sun_vecs in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            sun_vecs = torch.stack(sun_vecs).to(device)
            masks = torch.stack(masks).to(device)

            loss_dict = solar_df.model(images, sun_vecs, targets, masks)
            total_loss = sum(loss_dict.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}")

        # Checkpointing logic
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(solar_df.model.state_dict(), best_ckpt_path)
            print(f"  ⭐ New Best! Saved to {best_ckpt_path}")
        else:
            epochs_no_improve += 1

        # Save 'Last' for resuming
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": solar_df.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "patience_counter": epochs_no_improve,
            },
            last_ckpt_path,
        )
        checkpoint_volume.commit()

        # Early Stopping
        if epochs_no_improve >= patience:
            print(f"⏹️ Early Stopping at epoch {epoch}")
            break

    print("✅ Training Finished.")


@app.local_entrypoint()
def main(run_name: str = "you_forgot_to_add_run_name2"):
    train_solar_df_modal.remote(run_name=run_name)
