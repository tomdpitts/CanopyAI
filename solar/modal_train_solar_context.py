import modal
import sys
import os

app = modal.App("solar-context-training")

# reuse the same image/volumes as deepforest
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "numpy", "pillow", "tqdm")
    .add_local_dir(
        "..",  # Parent directory (canopyAI)
        remote_path="/root/canopyAI",
        ignore=["*.tif", ".git", "__pycache__", "venv*", "data/"],
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
    run_name: str = "romeo_context",
):
    import torch
    from torch.utils.data import DataLoader
    from pathlib import Path

    # 1. Setup Paths
    os.chdir("/root/canopyAI")
    sys.path.append("/root/canopyAI")
    from prototypes.solar_adapter import GlobalContextEncoder
    from prototypes.train_solar_context import SolarContextTrainer, RawOrthoDataset

    print(f"🚀 Starting Solar Context Training: {run_name}")

    # 2. Check for Resume Checkpoint
    checkpoint_dir = Path(f"/checkpoints/solar_context/{run_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = checkpoint_dir / "latest.pth"

    device = torch.device("cuda")
    model = GlobalContextEncoder().to(device)
    trainer = SolarContextTrainer(model, learning_rate=lr)

    start_epoch = 0
    if latest_ckpt.exists():
        print(f"🔄 Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt)
        model.load_state_dict(ckpt["model"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    # 3. Data Loading
    # Note: Assuming images are already on the volume at image_dir
    # If using tarballs, logic to extract would go here (omitted for brevity)
    dataset = RawOrthoDataset(image_dir)
    print(f"📊 Training on {len(dataset)} images")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 4. Training Loop with Periodic Checkpointing
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            loss = trainer.train_step(batch)
            total_loss += loss

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}")

        # Save Checkpoint
        save_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "loss": avg_loss,
        }
        torch.save(save_dict, latest_ckpt)
        # Verify persistence
        checkpoint_volume.commit()

    print(f"✅ Training Complete. Saved to {latest_ckpt}")


@app.local_entrypoint()
def main(epochs: int = 50, run_name: str = "romeo_context"):
    train_context_modal.remote(epochs=epochs, run_name=run_name)
