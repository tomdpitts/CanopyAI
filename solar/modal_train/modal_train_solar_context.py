import modal
import sys
import os

from pathlib import Path

# Get the project root (solar/modal_train -> solar -> canopyAI)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

app = modal.App("solar-context-training")

# reuse the same image/volumes as deepforest
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "numpy", "pillow", "tqdm")
    .add_local_dir(
        str(PROJECT_ROOT),  # Absolute path to canopyAI
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
            # Other
            "*.pyc",
            ".DS_Store",
        ],
    )
)

# Use existing volumes
data_volume = modal.Volume.from_name("canopyai-deepforest-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    "canopyai-deepforest-checkpoints", create_if_missing=True
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    timeout=86400,
)
def train_context_modal(
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    image_dir: str = "/data/images/data/won/raw",  # Images already on volume
    run_name: str = "you_forgot_to_add_run_name",
    patience: int = 10,  # Early stopping patience
):
    import torch
    from torch.utils.data import DataLoader
    from pathlib import Path

    # 1. Setup Paths
    os.chdir("/root/canopyAI")
    sys.path.append("/root/canopyAI")
    sys.path.append("/root/canopyAI/solar")
    sys.path.append("/root/canopyAI/solar/modal_train")
    from solar_adapter import GlobalContextEncoder
    from train_solar_context import SolarContextTrainer, RawOrthoDataset

    print(f"🚀 Starting Solar Context Training: {run_name}")

    # 2. Check for Resume Checkpoint
    checkpoint_dir = Path(f"/checkpoints/solar_context/{run_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = checkpoint_dir / "latest.pth"
    best_ckpt = checkpoint_dir / "best.pth"

    device = torch.device("cuda")
    model = GlobalContextEncoder().to(device)
    trainer = SolarContextTrainer(model, learning_rate=lr)

    start_epoch = 0
    best_loss = float("inf")
    epochs_without_improvement = 0

    if latest_ckpt.exists():
        print(f"🔄 Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt)
        model.load_state_dict(ckpt["model"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)

    # 3. Data Loading
    # images need to be accessible on the volume at image_dir
    dataset = RawOrthoDataset(image_dir)
    print(f"📊 Training on {len(dataset)} images")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 4. Training Loop with Periodic Checkpointing and Early Stopping
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            loss = trainer.train_step(batch)
            total_loss += loss

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}")

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save({"model": model.state_dict(), "loss": best_loss}, best_ckpt)
            print(f"📈 New best loss: {best_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(
                f"⏳ No improvement for {epochs_without_improvement}/{patience} epochs"
            )

        # Save Checkpoint
        save_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "loss": avg_loss,
            "best_loss": best_loss,
            "epochs_without_improvement": epochs_without_improvement,
        }
        torch.save(save_dict, latest_ckpt)
        # Verify persistence
        checkpoint_volume.commit()

        # Early stopping trigger
        if epochs_without_improvement >= patience:
            print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
            break

    print(f"✅ Training Complete. Best loss: {best_loss:.4f}")


@app.local_entrypoint()
def main(
    epochs: int = 50, run_name: str = "you_forgot_to_add_run_name", patience: int = 10
):
    train_context_modal.remote(epochs=epochs, run_name=run_name, patience=patience)
