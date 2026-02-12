"""
Modal Deployment: Proper SAM Finetuning with Box Prompts.

Key features:
- Per-box GT extraction: Uses connected components to get the single tree
  crown that intersects each bounding box (fixes prompt-ignoring issue)
- Shadow-canopy dual supervision for heuristic/adaptive modes

Supports three training modes:
- standard_sam: DiceBCE on canopy only (frozen encoder)
- shadow_penalty: DiceBCE on canopy + shadow penalty (frozen encoder)
- lora_adapted_encoder_plus_shadow_penalty: DiceBCE + shadow penalty with LoRA-adapted encoder

Usage:
    modal run modal_train_sam_proper.py --mode standard_sam --seed 42
    modal run modal_train_sam_proper.py --mode shadow_penalty --seed 42
    modal run modal_train_sam_proper.py --mode lora_adapted_encoder_plus_shadow_penalty --seed 42
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
    gpu="A100",
    timeout=3600 * 6,  # 6 hours
    volumes={VOLUME_PATH: data_volume},
)
def train_model(
    mode: str = "standard_sam",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    lambda_shadow: float = 0.3,
    seed: int = 42,
    patience: int = 10,
    run_name: str = "",
):
    """Train SAM decoder on Modal GPU."""
    import math
    import torch
    import torch.nn as nn
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
    # LoRA (Low-Rank Adaptation) for encoder fine-tuning
    # =========================================================================

    class LoRALayer(nn.Module):
        """Low-Rank Adaptation layer.

        Adds trainable rank decomposition: W' = W + BA
        where W is frozen, and B, A are trainable with rank r.
        """

        def __init__(self, frozen_layer, r=8, alpha=16):
            super().__init__()
            self.frozen_layer = frozen_layer
            self.r = r
            self.alpha = alpha

            # Freeze original weights
            for param in frozen_layer.parameters():
                param.requires_grad = False

            d_out, d_in = frozen_layer.weight.shape
            device = frozen_layer.weight.device

            # Low-rank matrices - create on same device as frozen layer
            self.lora_A = nn.Parameter(torch.zeros(r, d_in, device=device))
            self.lora_B = nn.Parameter(torch.zeros(d_out, r, device=device))

            # Initialize: A with Kaiming, B with zeros
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

            self.scaling = alpha / r

        def forward(self, x):
            original = self.frozen_layer(x)
            lora_out = (x @ self.lora_A.T) @ self.lora_B.T
            return original + lora_out * self.scaling

    def add_lora_to_encoder(sam, r=8, alpha=16):
        """Add LoRA adapters to SAM's ViT encoder attention and MLP.

        Returns list of LoRA parameters for the optimizer.
        """
        lora_params = []

        for block in sam.image_encoder.blocks:
            # Adapt QKV projection in attention
            if hasattr(block.attn, "qkv"):
                lora_qkv = LoRALayer(block.attn.qkv, r=r, alpha=alpha)
                block.attn.qkv = lora_qkv
                lora_params.extend([lora_qkv.lora_A, lora_qkv.lora_B])

            # Adapt MLP first layer
            if hasattr(block.mlp, "lin1"):
                lora_lin1 = LoRALayer(block.mlp.lin1, r=r, alpha=alpha)
                block.mlp.lin1 = lora_lin1
                lora_params.extend([lora_lin1.lora_A, lora_lin1.lora_B])

        return lora_params

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

    # Mode-specific encoder configuration
    lora_params = []
    if mode == "lora_adapted_encoder_plus_shadow_penalty":
        # Add LoRA adapters to encoder (keeps pretrained weights frozen)
        print("🔧 Adding LoRA adapters to encoder...")
        lora_params = add_lora_to_encoder(sam, r=8, alpha=16)
        print(f"   Added {len(lora_params)} LoRA parameter tensors")
    else:
        # Freeze image encoder entirely for unguided/heuristic modes
        print("❄️  Freezing image encoder (86M params)...")
        for param in sam.image_encoder.parameters():
            param.requires_grad = False

    # Count trainable params
    trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in lora_params)  # LoRA params
    total = sum(p.numel() for p in sam.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} params")

    # =========================================================================
    # Dataset
    # =========================================================================

    # =========================================================================
    # Helper functions for per-tree mask extraction
    # =========================================================================

    def extract_tree_mask_for_box(canopy_mask, box):
        """Extract the single tree crown with maximum IoU with the bounding box.

        Uses connected component labeling to isolate individual tree crowns,
        then selects the ONE component with highest IoU with the box region.
        This ensures only one tree per box, even with partial overlaps.

        Args:
            canopy_mask: Binary mask (H, W) with all canopy pixels
            box: Dict with xmin, ymin, xmax, ymax

        Returns:
            Binary mask (H, W) containing only the single best-matching tree
        """
        x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        box_area = (x2 - x1) * (y2 - y1)

        # Label connected components in the canopy mask
        num_labels, labels = cv2.connectedComponents(
            (canopy_mask > 0.5).astype(np.uint8)
        )

        # Create box region mask
        box_region = np.zeros_like(canopy_mask, dtype=np.uint8)
        box_region[y1:y2, x1:x2] = 1

        # Find which labels intersect with the box
        labels_in_box = np.unique(labels * box_region)
        labels_in_box = labels_in_box[labels_in_box > 0]  # Exclude background

        if len(labels_in_box) == 0:
            # No tree intersects - return empty mask
            return np.zeros_like(canopy_mask, dtype=np.float32)

        # Calculate IoU for each candidate tree
        best_label = None
        best_iou = -1

        for label in labels_in_box:
            component_mask = (labels == label).astype(np.float32)
            component_area = component_mask.sum()

            # Intersection with box
            intersection = (component_mask * box_region).sum()

            # Union = component + box - intersection
            union = component_area + box_area - intersection

            iou = intersection / (union + 1e-6)

            if iou > best_iou:
                best_iou = iou
                best_label = label

        # Create output mask with only the best tree
        tree_mask = np.zeros_like(canopy_mask, dtype=np.float32)
        if best_label is not None:
            tree_mask[labels == best_label] = 1.0

        return tree_mask

    def extract_shadow_for_tree(shadow_mask, tree_mask, dilation_px=10):
        """Extract shadow region associated with this tree.

        Strategy: Dilate the tree mask and find shadow pixels nearby.
        This captures the cast shadow even if it's not directly connected.

        Args:
            shadow_mask: Binary mask (H, W) with all shadow pixels
            tree_mask: Binary mask (H, W) with this tree's canopy
            dilation_px: Pixels to dilate tree mask for shadow searching

        Returns:
            Binary mask (H, W) containing shadow near this tree
        """
        # Dilate tree mask to create search region
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
        )
        search_region = cv2.dilate(tree_mask.astype(np.uint8), kernel)

        # Find shadow within search region (but not overlapping canopy)
        tree_shadow = shadow_mask * search_region * (1 - tree_mask)

        return tree_shadow.astype(np.float32)

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

            # Load image and full masks
            img = cv2.imread(str(self.image_dir / f"{name}.png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            canopy_full = cv2.imread(str(self.mask_dir / f"canopy_mask_{name}.png"), 0)
            shadow_full = cv2.imread(str(self.mask_dir / f"shadow_mask_{name}.png"), 0)

            # Binarize masks
            canopy_full = (canopy_full > 127).astype(np.float32)
            shadow_full = (shadow_full > 127).astype(np.float32)

            # === Extract tree-specific masks ===
            # Get only the tree crown(s) that intersect with this box
            tree_mask = extract_tree_mask_for_box(canopy_full, box)
            # Get shadow associated with this tree
            tree_shadow = extract_shadow_for_tree(shadow_full, tree_mask)

            x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            h, w = img.shape[:2]

            # Augmentation (apply to tree-specific masks)
            if self.augment:
                rng = random.Random(self.seed + idx)
                np_rng = np.random.RandomState(self.seed + idx)

                # Random flip
                if rng.random() < 0.5:
                    img = np.fliplr(img).copy()
                    tree_mask = np.fliplr(tree_mask).copy()
                    tree_shadow = np.fliplr(tree_shadow).copy()
                    x1, x2 = w - x2, w - x1

                if rng.random() < 0.5:
                    img = np.flipud(img).copy()
                    tree_mask = np.flipud(tree_mask).copy()
                    tree_shadow = np.flipud(tree_shadow).copy()
                    y1, y2 = h - y2, h - y1

                # Small noise
                sigma = rng.uniform(0.01, 0.03) * 255
                noise = np_rng.normal(0, sigma, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)

            # Resize to SAM's expected 1024x1024
            img_resized = cv2.resize(img, (1024, 1024))
            # SAM output is 256x256
            tree_mask_resized = cv2.resize(tree_mask, (256, 256))
            tree_shadow_resized = cv2.resize(tree_shadow, (256, 256))

            # Scale box coordinates
            scale_x = 1024 / w
            scale_y = 1024 / h
            box_scaled = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]

            return {
                "image": (
                    torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                ),
                "box": torch.tensor(box_scaled, dtype=torch.float32),
                "canopy_mask": torch.from_numpy(tree_mask_resized).float(),
                "shadow_mask": torch.from_numpy(tree_shadow_resized).float(),
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
    # Use custom run_name if provided, otherwise default to mode_seed format
    output_subdir = run_name if run_name else f"{mode}_seed{seed}"
    output_dir = Path(VOLUME_PATH) / "sam_proper_outputs" / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

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

    # Optimizer (decoder + prompt encoder, plus LoRA params for adaptive)
    trainable_params = [p for p in sam.parameters() if p.requires_grad]
    if mode == "lora_adapted_encoder_plus_shadow_penalty":
        trainable_params = trainable_params + lora_params
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
    # Helper to get LoRA state dict
    # =========================================================================

    def get_lora_state(sam):
        """Extract LoRA weights from encoder for saving."""
        lora_state = {}
        for idx, block in enumerate(sam.image_encoder.blocks):
            if hasattr(block.attn.qkv, "lora_A"):
                lora_state[f"block{idx}.attn.qkv.lora_A"] = (
                    block.attn.qkv.lora_A.data.cpu()
                )
                lora_state[f"block{idx}.attn.qkv.lora_B"] = (
                    block.attn.qkv.lora_B.data.cpu()
                )
            if hasattr(block.mlp.lin1, "lora_A"):
                lora_state[f"block{idx}.mlp.lin1.lora_A"] = (
                    block.mlp.lin1.lora_A.data.cpu()
                )
                lora_state[f"block{idx}.mlp.lin1.lora_B"] = (
                    block.mlp.lin1.lora_B.data.cpu()
                )
        return lora_state

    def load_lora_state(sam, lora_state, device):
        """Load LoRA weights into encoder."""
        for idx, block in enumerate(sam.image_encoder.blocks):
            if f"block{idx}.attn.qkv.lora_A" in lora_state:
                block.attn.qkv.lora_A.data = lora_state[
                    f"block{idx}.attn.qkv.lora_A"
                ].to(device)
                block.attn.qkv.lora_B.data = lora_state[
                    f"block{idx}.attn.qkv.lora_B"
                ].to(device)
            if f"block{idx}.mlp.lin1.lora_A" in lora_state:
                block.mlp.lin1.lora_A.data = lora_state[
                    f"block{idx}.mlp.lin1.lora_A"
                ].to(device)
                block.mlp.lin1.lora_B.data = lora_state[
                    f"block{idx}.mlp.lin1.lora_B"
                ].to(device)

    # =========================================================================
    # Check for existing checkpoint to resume from
    # =========================================================================

    checkpoint_path = output_dir / "checkpoint.pth"
    start_epoch = 1
    best_iou = 0
    epochs_without_improvement = 0
    history = []

    if checkpoint_path.exists():
        print("\n� Found checkpoint, resuming training...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        sam.mask_decoder.load_state_dict(ckpt["mask_decoder"])
        sam.prompt_encoder.load_state_dict(ckpt["prompt_encoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt["best_iou"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        history = ckpt.get("history", [])
        baseline_iou = ckpt.get("baseline_iou", baseline_iou)
        if mode == "lora_adapted_encoder_plus_shadow_penalty" and "lora" in ckpt:
            load_lora_state(sam, ckpt["lora"], device)
        print(f"   Resuming from epoch {start_epoch}, best IoU: {best_iou:.4f}")
    else:
        print(f"\n🚀 Training {mode} mode (patience={patience})...")

    # =========================================================================
    # Training loop
    # =========================================================================

    for epoch in range(start_epoch, epochs + 1):
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

                if mode in (
                    "shadow_penalty",
                    "lora_adapted_encoder_plus_shadow_penalty",
                ):
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
                    if mode in (
                        "shadow_penalty",
                        "lora_adapted_encoder_plus_shadow_penalty",
                    ):
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
            # Save decoder + prompt encoder + LoRA weights
            checkpoint = {
                "mask_decoder": sam.mask_decoder.state_dict(),
                "prompt_encoder": sam.prompt_encoder.state_dict(),
                "mode": mode,
                "best_iou": best_iou,
                "baseline_iou": baseline_iou,
                "epoch": epoch,
            }
            # For adaptive mode, also save LoRA weights
            if mode == "lora_adapted_encoder_plus_shadow_penalty":
                lora_state = {}
                for idx, block in enumerate(sam.image_encoder.blocks):
                    if hasattr(block.attn.qkv, "lora_A"):
                        lora_state[f"block{idx}.attn.qkv.lora_A"] = (
                            block.attn.qkv.lora_A.data
                        )
                        lora_state[f"block{idx}.attn.qkv.lora_B"] = (
                            block.attn.qkv.lora_B.data
                        )
                    if hasattr(block.mlp.lin1, "lora_A"):
                        lora_state[f"block{idx}.mlp.lin1.lora_A"] = (
                            block.mlp.lin1.lora_A.data
                        )
                        lora_state[f"block{idx}.mlp.lin1.lora_B"] = (
                            block.mlp.lin1.lora_B.data
                        )
                checkpoint["lora"] = lora_state
            torch.save(checkpoint, output_dir / "best_decoder.pth")
            print(f"  → Saved best (IoU: {best_iou:.4f}, baseline: {baseline_iou:.4f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\n⏹️  Early stopping: no improvement for {patience} epochs")
                break

        # Periodic checkpoint (every 5 epochs) for crash recovery
        if epoch % 5 == 0:
            periodic_ckpt = {
                "epoch": epoch,
                "best_iou": best_iou,
                "baseline_iou": baseline_iou,
                "epochs_without_improvement": epochs_without_improvement,
                "mask_decoder": sam.mask_decoder.state_dict(),
                "prompt_encoder": sam.prompt_encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
            }
            if mode == "lora_adapted_encoder_plus_shadow_penalty":
                periodic_ckpt["lora"] = get_lora_state(sam)
            torch.save(periodic_ckpt, checkpoint_path)
            data_volume.commit()  # Persist to volume
            print(f"  💾 Checkpoint saved (epoch {epoch})")

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
    mode: str = "standard_sam",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    lambda_shadow: float = 0.3,
    seed: int = 42,
    patience: int = 10,
    run_name: str = "",
):
    """Run SAM finetuning on Modal."""
    if mode not in [
        "standard_sam",
        "shadow_penalty",
        "lora_adapted_encoder_plus_shadow_penalty",
    ]:
        raise ValueError(
            f"Invalid mode: {mode}. Use standard_sam, shadow_penalty, or lora_adapted_encoder_plus_shadow_penalty"
        )

    result = train_model.remote(
        mode=mode,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_shadow=lambda_shadow,
        seed=seed,
        patience=patience,
        run_name=run_name,
    )

    print(f"\n🎯 {result['mode']} (seed={seed})")
    print(f"   Baseline: {result['baseline_iou']:.4f}")
    print(f"   Best: {result['best_iou']:.4f}")
    print(f"   Improvement: +{result['improvement']:.4f}")
