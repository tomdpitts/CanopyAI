"""
Modal Deployment: Proper SAM Finetuning with Box Prompts.

Freezes SAM's ViT encoder and trains only the mask decoder + prompt encoder.
Supports three training modes:
- unguided: DiceBCE on canopy only
- heuristic: DiceBCE on canopy + shadow penalty
- adaptive: Multi-task with encoder fine-tuning (Phase 2)

Usage:
    modal run modal_train_sam_proper.py --mode unguided --seed 42
    modal run modal_train_sam_proper.py --mode heuristic --seed 42
"""

import modal

app = modal.App("sam-proper-finetuning")

# Image with SAM dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "wget", "git")
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "opencv-python-headless",
        "tqdm",
        "segment-anything @ git+https://github.com/facebookresearch/"
        "segment-anything.git",
    )
)

data_volume = modal.Volume.from_name("shadow-canopy-data", create_if_missing=True)
VOLUME_PATH = "/data"


@app.function(
    image=training_image,
    gpu="A10G",
    timeout=3600 * 6,  # 6 hours
    volumes={VOLUME_PATH: data_volume},
)
def train_model(
    mode: str = "unguided",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    lambda_shadow: float = 0.3,
    seed: int = 42,
):
    """Train SAM decoder on Modal GPU."""
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import cv2
    import random
    import xml.etree.ElementTree as ET
    import json
    import subprocess
    from pathlib import Path
    from tqdm import tqdm

    device = torch.device("cuda")
    torch.manual_seed(seed)
    print(f"Mode: {mode}, Epochs: {epochs}, Seed: {seed}")
    print(f"Device: cuda ({torch.cuda.get_device_name(0)})")

    # =========================================================================
    # Download SAM checkpoint if needed
    # =========================================================================
    sam_checkpoint = Path(VOLUME_PATH) / "sam_vit_b_01ec64.pth"
    if not sam_checkpoint.exists():
        print("📥 Downloading SAM ViT-B checkpoint...")
        subprocess.run(
            [
                "wget",
                "-q",
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "-O",
                str(sam_checkpoint),
            ],
            check=True,
        )
        print(f"✅ Downloaded to {sam_checkpoint}")

    # =========================================================================
    # Load SAM
    # =========================================================================
    from segment_anything import sam_model_registry

    print("🔧 Loading SAM ViT-B...")
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))
    sam.to(device)

    # Freeze image encoder
    print("❄️  Freezing image encoder (86M params)...")
    for param in sam.image_encoder.parameters():
        param.requires_grad = False

    # Count trainable params
    trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    total = sum(p.numel() for p in sam.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} params")

    # =========================================================================
    # Dataset
    # =========================================================================

    def parse_voc(xml_path):
        tree = ET.parse(xml_path)
        boxes = []
        for obj in tree.getroot().findall("object"):
            bbox = obj.find("bndbox")
            boxes.append(
                {
                    "xmin": int(float(bbox.find("xmin").text)),
                    "ymin": int(float(bbox.find("ymin").text)),
                    "xmax": int(float(bbox.find("xmax").text)),
                    "ymax": int(float(bbox.find("ymax").text)),
                }
            )
        return boxes

    class SAMBoxDataset(Dataset):
        """Dataset that returns crops for SAM training."""

        def __init__(self, image_dir, ann_dir, mask_dir, augment=True, seed=42):
            self.image_dir = Path(image_dir)
            self.ann_dir = Path(ann_dir)
            self.mask_dir = Path(mask_dir)
            self.augment = augment
            self.seed = seed

            # Collect all (image_name, box) pairs
            self.samples = []
            for ann in self.ann_dir.glob("*.xml"):
                name = ann.stem
                if (self.image_dir / f"{name}.png").exists():
                    for box in parse_voc(ann):
                        self.samples.append((name, box))

            print(f"   Dataset: {len(self.samples)} box samples")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            name, box = self.samples[idx]

            # Load image and masks
            img = cv2.imread(str(self.image_dir / f"{name}.png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            canopy_path = self.mask_dir / f"canopy_mask_{name}.png"
            shadow_path = self.mask_dir / f"shadow_mask_{name}.png"
            canopy = cv2.imread(str(canopy_path), 0)
            shadow = cv2.imread(str(shadow_path), 0)

            # Binarize masks
            canopy = (canopy > 127).astype(np.float32)
            shadow = (shadow > 127).astype(np.float32)

            x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            h, w = img.shape[:2]

            # Augmentation
            if self.augment:
                rng = random.Random(self.seed + idx)
                np_rng = np.random.RandomState(self.seed + idx)

                # Random flip
                if rng.random() < 0.5:
                    img = np.fliplr(img).copy()
                    canopy = np.fliplr(canopy).copy()
                    shadow = np.fliplr(shadow).copy()
                    x1, x2 = w - x2, w - x1

                if rng.random() < 0.5:
                    img = np.flipud(img).copy()
                    canopy = np.flipud(canopy).copy()
                    shadow = np.flipud(shadow).copy()
                    y1, y2 = h - y2, h - y1

                # Small noise
                sigma = rng.uniform(0.01, 0.03) * 255
                noise = np_rng.normal(0, sigma, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)

            # Resize to SAM's expected 1024x1024
            img_resized = cv2.resize(img, (1024, 1024))
            # SAM output is 256x256
            canopy_resized = cv2.resize(canopy, (256, 256))
            shadow_resized = cv2.resize(shadow, (256, 256))

            # Scale box coordinates
            scale_x = 1024 / w
            scale_y = 1024 / h
            box_scaled = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

            return {
                "image": (
                    torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                ),
                "box": torch.tensor(box_scaled, dtype=torch.float32),
                "canopy_mask": torch.from_numpy(canopy_resized).float(),
                "shadow_mask": torch.from_numpy(shadow_resized).float(),
                "original_size": torch.tensor([h, w]),
            }

    # =========================================================================
    # Loss functions
    # =========================================================================

    def dice_bce_loss(pred, target):
        """Combined Dice + BCE loss."""
        # BCE
        bce = F.binary_cross_entropy(pred, target, reduction="mean")

        # Dice
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        denom = pred_flat.sum() + target_flat.sum() + 1
        dice = 1 - (2 * intersection + 1) / denom

        return bce + dice

    def compute_iou(pred, target):
        """Compute IoU metric."""
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        return (intersection / (union + 1e-6)).item()

    # =========================================================================
    # Training setup
    # =========================================================================

    data_dir = Path(VOLUME_PATH) / "won003"
    output_subdir = f"{mode}_seed{seed}"
    output_dir = Path(VOLUME_PATH) / "sam_proper_outputs" / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    full_ds = SAMBoxDataset(
        data_dir / "images",
        data_dir / "annotations",
        data_dir / "train_masks",
        augment=True,
        seed=seed,
    )

    # Split by image name (not by box)
    all_names = list(set(n for n, _ in full_ds.samples))
    rng = random.Random(seed)
    rng.shuffle(all_names)
    n_val = max(1, int(len(all_names) * 0.2))
    val_names = set(all_names[:n_val])
    train_names = set(all_names[n_val:])

    train_ds = SAMBoxDataset(
        data_dir / "images",
        data_dir / "annotations",
        data_dir / "train_masks",
        augment=True,
        seed=seed,
    )
    train_ds.samples = [(n, b) for n, b in train_ds.samples if n in train_names]

    val_ds = SAMBoxDataset(
        data_dir / "images",
        data_dir / "annotations",
        data_dir / "train_masks",
        augment=False,
        seed=seed,
    )
    val_ds.samples = [(n, b) for n, b in val_ds.samples if n in val_names]

    print(f"📊 Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    # Optimizer (only trainable params)
    trainable_params = [p for p in sam.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # =========================================================================
    # Baseline evaluation (unfinetuned SAM)
    # =========================================================================

    print("\n📏 Evaluating baseline (unfinetuned SAM)...")
    sam.eval()
    baseline_iou = 0
    baseline_count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Baseline"):
            images = batch["image"].to(device)
            boxes = batch["box"].to(device)
            canopy_gt = batch["canopy_mask"].to(device)

            for i in range(images.shape[0]):
                # Get image embedding
                img_embedding = sam.image_encoder(images[i : i + 1])

                # Prepare box prompt
                box_torch = boxes[i : i + 1].unsqueeze(1)  # (1, 1, 4)

                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )

                # Predict mask
                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=img_embedding,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                # Resize to 256x256 (match GT)
                pred = F.interpolate(
                    low_res_masks,
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                )
                pred = torch.sigmoid(pred.squeeze(1))

                baseline_iou += compute_iou(pred[0], canopy_gt[i])
                baseline_count += 1

    baseline_iou /= baseline_count
    print(f"✅ Baseline IoU: {baseline_iou:.4f}")

    # =========================================================================
    # Training loop
    # =========================================================================

    print(f"\n🚀 Training {mode} mode...")
    best_iou = 0
    history = []

    for epoch in range(1, epochs + 1):
        sam.train()
        # Keep encoder in eval mode (frozen)
        sam.image_encoder.eval()

        train_loss = 0
        train_iou = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            images = batch["image"].to(device)
            boxes = batch["box"].to(device)
            canopy_gt = batch["canopy_mask"].to(device)
            shadow_gt = batch["shadow_mask"].to(device)

            optimizer.zero_grad()
            batch_loss = 0
            batch_iou = 0

            for i in range(images.shape[0]):
                # Get image embedding (no grad since frozen)
                with torch.no_grad():
                    img_embedding = sam.image_encoder(images[i : i + 1])

                # Prepare box prompt
                box_torch = boxes[i : i + 1].unsqueeze(1)  # (1, 1, 4)

                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )

                # Predict mask
                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=img_embedding,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                # Resize to 256x256 (match GT)
                pred = F.interpolate(
                    low_res_masks,
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                )
                pred = torch.sigmoid(pred.squeeze(1))

                # Compute loss
                loss = dice_bce_loss(pred[0], canopy_gt[i])

                if mode == "heuristic":
                    # Shadow penalty: penalize predicting shadow as canopy
                    shadow_penalty = (pred[0] * shadow_gt[i]).mean()
                    loss = loss + lambda_shadow * shadow_penalty

                batch_loss += loss
                batch_iou += compute_iou(pred[0], canopy_gt[i])

            # Average over batch
            batch_loss = batch_loss / images.shape[0]
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()
            train_iou += batch_iou / images.shape[0]

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        # Validation
        sam.eval()
        val_loss = 0
        val_iou = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                boxes = batch["box"].to(device)
                canopy_gt = batch["canopy_mask"].to(device)
                shadow_gt = batch["shadow_mask"].to(device)

                for i in range(images.shape[0]):
                    img_embedding = sam.image_encoder(images[i : i + 1])

                    box_torch = boxes[i : i + 1].unsqueeze(1)
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=None,
                        boxes=box_torch,
                        masks=None,
                    )

                    low_res_masks, _ = sam.mask_decoder(
                        image_embeddings=img_embedding,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )

                    pred = F.interpolate(
                        low_res_masks,
                        size=(256, 256),
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred = torch.sigmoid(pred.squeeze(1))

                    loss = dice_bce_loss(pred[0], canopy_gt[i])
                    if mode == "heuristic":
                        shadow_penalty = (pred[0] * shadow_gt[i]).mean()
                        loss = loss + lambda_shadow * shadow_penalty

                    val_loss += loss.item()
                    val_iou += compute_iou(pred[0], canopy_gt[i])

        val_loss /= len(val_ds)
        val_iou /= len(val_ds)
        scheduler.step()

        # Log
        improvement = "📈" if val_iou > best_iou else ""
        print(
            f"Epoch {epoch:3d} | "
            f"Train: {train_loss:.4f} / {train_iou:.4f} | "
            f"Val: {val_loss:.4f} / {val_iou:.4f} {improvement}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_iou": val_iou,
            }
        )

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            # Save only decoder + prompt encoder weights
            checkpoint = {
                "mask_decoder": sam.mask_decoder.state_dict(),
                "prompt_encoder": sam.prompt_encoder.state_dict(),
                "mode": mode,
                "best_iou": best_iou,
                "baseline_iou": baseline_iou,
                "epoch": epoch,
            }
            torch.save(checkpoint, output_dir / "best_decoder.pth")
            print(f"  → Saved best (IoU: {best_iou:.4f}, baseline: {baseline_iou:.4f})")

    # Save final model
    final_checkpoint = {
        "mask_decoder": sam.mask_decoder.state_dict(),
        "prompt_encoder": sam.prompt_encoder.state_dict(),
        "mode": mode,
        "final_iou": val_iou,
        "best_iou": best_iou,
        "baseline_iou": baseline_iou,
    }
    torch.save(final_checkpoint, output_dir / "final_decoder.pth")

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Summary
    summary = {
        "mode": mode,
        "seed": seed,
        "baseline_iou": baseline_iou,
        "best_iou": best_iou,
        "improvement": best_iou - baseline_iou,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    data_volume.commit()

    print("=" * 50)
    print("✅ Training complete!")
    print(f"   Mode: {mode}")
    print(f"   Baseline IoU: {baseline_iou:.4f}")
    print(f"   Best IoU: {best_iou:.4f}")
    print(f"   Improvement: +{(best_iou - baseline_iou):.4f}")
    print(f"   Saved to: {output_dir}")
    print(f"{'=' * 50}")

    return summary


@app.local_entrypoint()
def main(
    mode: str = "unguided",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    lambda_shadow: float = 0.3,
    seed: int = 42,
):
    """Run SAM finetuning on Modal."""
    if mode not in ["unguided", "heuristic", "adaptive"]:
        raise ValueError(f"Invalid mode: {mode}. Use unguided, heuristic, or adaptive")

    if mode == "adaptive":
        print("⚠️  Adaptive mode not yet implemented. Use unguided or heuristic.")
        return

    result = train_model.remote(
        mode=mode,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_shadow=lambda_shadow,
        seed=seed,
    )

    print(f"\n🎯 {result['mode']} (seed={seed})")
    print(f"   Baseline: {result['baseline_iou']:.4f}")
    print(f"   Best: {result['best_iou']:.4f}")
    print(f"   Improvement: +{result['improvement']:.4f}")
