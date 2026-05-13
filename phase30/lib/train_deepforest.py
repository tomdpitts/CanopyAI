#!/usr/bin/env python3
"""
DeepForest Fine-tuning — phase30 standalone version.

Trains DeepForest on the Restor TCD dataset using shadow loss reweighting.
Based on deepforest_custom/train_deepforest.py; shadow_channel / shadow_cross_attention /
shadow_proposals and WON bbox normalisation have been removed — only shadow_loss_reweight
is retained.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from deepforest import main as deepforest_main
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

torch.set_float32_matmul_precision("medium")


# ---------------------------------------------------------------------------
# Monkey-patch: fix deepforest evaluate_boxes index-alignment bug
# ---------------------------------------------------------------------------
def _patched_evaluate_boxes(predictions, ground_df, iou_threshold=0.4):
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import shapely
    from deepforest import evaluate as _ev
    from deepforest.utilities import determine_geometry_type

    if ground_df.empty:
        return {
            "results": None,
            "box_recall": None,
            "box_precision": 0,
            "class_recall": None,
            "predictions": predictions,
            "ground_df": ground_df,
        }

    if not isinstance(predictions, gpd.GeoDataFrame):
        if "geometry" not in predictions.columns and all(
            c in predictions.columns for c in ["xmin", "ymin", "xmax", "ymax"]
        ):
            predictions = predictions.copy()
            predictions["geometry"] = shapely.box(
                predictions["xmin"], predictions["ymin"],
                predictions["xmax"], predictions["ymax"],
            )
        predictions = gpd.GeoDataFrame(predictions, geometry="geometry")

    if not isinstance(ground_df, gpd.GeoDataFrame):
        if "geometry" not in ground_df.columns and all(
            c in ground_df.columns for c in ["xmin", "ymin", "xmax", "ymax"]
        ):
            ground_df = ground_df.copy()
            ground_df["geometry"] = shapely.box(
                ground_df["xmin"], ground_df["ymin"],
                ground_df["xmax"], ground_df["ymax"],
            )
        ground_df = gpd.GeoDataFrame(ground_df, geometry="geometry")

    predictions_by_image = {
        name: group.reset_index(drop=True)
        for name, group in predictions.groupby("image_path")
    }

    results, box_recalls, box_precisions = [], [], []
    for image_path, group in ground_df.groupby("image_path"):
        image_predictions = predictions_by_image.get(image_path, pd.DataFrame())
        if not isinstance(image_predictions, pd.DataFrame) or image_predictions.empty:
            image_predictions = pd.DataFrame()

        if image_predictions.empty:
            g = group.reset_index(drop=True)
            n = len(g)
            result = pd.DataFrame({
                "truth_id":       group.index.values,
                "prediction_id":  [None]  * n,
                "IoU":            [0.0]   * n,
                "predicted_label":[None]  * n,
                "score":          [None]  * n,
                "match":          [False] * n,
                "true_label":     g.label.values,
                "geometry":       g.geometry.values,
            })
            box_recalls.append(0)
            results.append(result)
            continue
        else:
            group = group.reset_index(drop=True)
            result = _ev.evaluate_image_boxes(
                predictions=image_predictions, ground_df=group
            )

        result["image_path"] = image_path
        result["match"] = result.IoU > iou_threshold
        result["match"] = result["match"].fillna(False)
        true_positive = sum(result["match"])
        box_recalls.append(true_positive / result.shape[0])
        box_precisions.append(true_positive / image_predictions.shape[0])
        results.append(result)

    if results:
        results = pd.concat(results, ignore_index=True)
    else:
        results = pd.DataFrame(columns=[
            "truth_id", "prediction_id", "IoU", "predicted_label",
            "score", "match", "true_label", "geometry", "image_path",
        ])

    box_recall    = np.mean(box_recalls)
    box_precision = np.mean(box_precisions) if box_precisions else np.nan
    class_recall  = _ev.compute_class_recall(results[results.match])

    return {
        "results": results,
        "box_precision": box_precision,
        "box_recall": box_recall,
        "class_recall": class_recall,
        "predictions": predictions,
        "ground_df": ground_df,
    }

from deepforest import evaluate as _deepforest_evaluate
_deepforest_evaluate.evaluate_boxes = _patched_evaluate_boxes
# ---------------------------------------------------------------------------


try:
    from .models import ShadowConditionedDeepForest
except ImportError:
    from models import ShadowConditionedDeepForest


class _TiledValDataset:
    """Tiles each 2048×2048 val image into 400×400 patches for validation.
    Defined at module level so it can be pickled by multiprocessing workers."""
    def __init__(self, base_ds, patch_size=400, overlap=0.0, min_visibility=0.5):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        self.base        = base_ds
        self.patch_size  = patch_size
        self.step        = max(1, int(patch_size * (1 - overlap)))
        self.min_vis     = min_visibility
        self.collate_fn  = base_ds.collate_fn
        self._val_t      = A.Compose(
            [ToTensorV2()],
            bbox_params=A.BboxParams(format="pascal_voc",
                                     label_fields=["category_ids"],
                                     clip=True, min_visibility=min_visibility),
        )
        _sample = base_ds.load_image(0)
        self._H, self._W = _sample.shape[:2]
        self._build_patch_list()

    def _build_patch_list(self):
        self._patches = []
        for img_idx in range(len(self.base.image_names)):
            for py in range(0, self._H, self.step):
                for px in range(0, self._W, self.step):
                    py2 = min(py + self.patch_size, self._H)
                    px2 = min(px + self.patch_size, self._W)
                    py1 = max(0, py2 - self.patch_size)
                    px1 = max(0, px2 - self.patch_size)
                    self._patches.append((img_idx, px1, py1, px2, py2))

    def __len__(self):
        return len(self._patches)

    def __getitem__(self, idx):
        img_idx, px1, py1, px2, py2 = self._patches[idx]
        img_name = self.base.image_names[img_idx]
        img      = self.base.load_image(img_idx)
        patch    = img[py1:py2, px1:px2]

        gt      = self.base.annotations_for_path(img_name)
        boxes   = gt["boxes"]
        labels  = gt["labels"]

        if len(boxes):
            cx1 = np.maximum(boxes[:, 0], px1) - px1
            cy1 = np.maximum(boxes[:, 1], py1) - py1
            cx2 = np.minimum(boxes[:, 2], px2) - px1
            cy2 = np.minimum(boxes[:, 3], py2) - py1
            bw  = boxes[:, 2] - boxes[:, 0]; bh = boxes[:, 3] - boxes[:, 1]
            vis = np.where(bw * bh > 0,
                           np.maximum(cx2 - cx1, 0) * np.maximum(cy2 - cy1, 0) / (bw * bh),
                           0)
            keep = vis >= self.min_vis
            if keep.any():
                patch_boxes = np.stack(
                    [cx1[keep], cy1[keep], cx2[keep], cy2[keep]],
                    axis=1).astype(np.float32)
                patch_labels = labels[keep]
            else:
                patch_boxes, patch_labels = np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
        else:
            patch_boxes, patch_labels = np.zeros((0, 4), np.float32), np.zeros(0, np.int64)

        aug     = self._val_t(image=patch, bboxes=patch_boxes, category_ids=patch_labels)
        image   = aug["image"]
        boxes_t = torch.from_numpy(np.array(aug["bboxes"], dtype=np.float32)).reshape(-1, 4)
        labels_t= torch.from_numpy(np.array(aug["category_ids"], dtype=np.int64))
        return image, {"boxes": boxes_t, "labels": labels_t}, img_name


class _FP32Guard(torch.autograd.Function):
    """fp16 → fp32 forward; fp32 gradient → fp16 backward."""
    @staticmethod
    def forward(ctx, x):
        return x.float()

    @staticmethod
    def backward(ctx, grad):
        return grad.half()


class _FP16Guard(torch.autograd.Function):
    """fp32 → fp16 forward; fp16 gradient → fp32 backward."""
    @staticmethod
    def forward(ctx, x):
        return x.half()

    @staticmethod
    def backward(ctx, grad):
        return grad.float()


class MPSHalfPrecisionCallback(pl.Callback):
    """
    Manual fp16 mixed precision for Apple MPS.

    MPS has no torch.autocast support, so mixed precision is implemented with
    explicit dtype boundary functions (_FP32Guard / _FP16Guard) placed at every
    point where a backward kernel requires float32:

      Backbone convolutions  → fp16  (fast; MPS supports fp16 conv backward)
      BatchNorm layers       → fp32  (wrapped by _FP32Guard / _FP16Guard hooks)
      FPN convolutions       → fp16
      Detection head         → fp32  (focal loss / smooth-L1 backward need fp32)
    """

    _NORM_TYPES = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm,
    )

    def __init__(self):
        self._master: dict[str, torch.Tensor] = {}
        self._overflow = False
        self._hooks: list = []

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._master = {
            name: param.detach().float().clone()
            for name, param in pl_module.named_parameters()
        }
        pl_module.half()

        inner = pl_module.model
        if hasattr(inner, "model"):
            inner = inner.model

        body = None
        if hasattr(inner, "backbone") and hasattr(inner.backbone, "body"):
            body = inner.backbone.body

        n_bn = 0
        if body is not None:
            try:
                import deepforest_custom.mps_ops as mps_ops
                n_bn = mps_ops.patch_batchnorm_for_fp16(body)
            except Exception as e:
                print(f"[MPSHalf] kernel patch failed ({e}), falling back to Python hooks")
                for m in body.modules():
                    if isinstance(m, self._NORM_TYPES):
                        m.float()
                        self._hooks.append(m.register_forward_pre_hook(self._norm_pre_hook))
                        self._hooks.append(m.register_forward_hook(self._norm_post_hook))
                        n_bn += 1

        head = getattr(inner, "head", None)
        if head is not None:
            head.float()
            self._hooks.append(head.register_forward_pre_hook(self._head_pre_hook))

        pl_module._mps_loss_scale = 1.0
        print(
            f"[MPSHalf] backbone fp16 | {n_bn} BN modules patched with MPSGraph kernel"
            f" | detection head → fp32"
        )

    @staticmethod
    def _norm_pre_hook(module, args):
        if args and isinstance(args[0], torch.Tensor) and args[0].dtype == torch.float16:
            return (_FP32Guard.apply(args[0]),) + args[1:]
        return args

    @staticmethod
    def _norm_post_hook(module, args, output):
        if isinstance(output, torch.Tensor) and output.dtype == torch.float32:
            return _FP16Guard.apply(output)
        return output

    @staticmethod
    def _head_pre_hook(module, args):
        if not args:
            return args
        features = args[0]
        def _cast(t):
            return _FP32Guard.apply(t) if isinstance(t, torch.Tensor) and t.dtype == torch.float16 else t
        if isinstance(features, dict):
            features = {k: _cast(v) for k, v in features.items()}
        elif isinstance(features, (list, tuple)):
            features = type(features)(_cast(f) for f in features)
        return (features,) + args[1:]

    def on_before_optimizer_step(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer
    ) -> None:
        overflow = any(
            not torch.isfinite(p.grad).all()
            for p in pl_module.parameters()
            if p.grad is not None
        )
        if overflow:
            self._overflow = True
            for p in pl_module.parameters():
                p.grad = None
            for name, p in pl_module.named_parameters():
                if name in self._master:
                    p.data = self._master[name].data.clone()
            print("[MPSHalf] Overflow in fp16 gradients — step skipped")
            return

        self._overflow = False
        for name, p in pl_module.named_parameters():
            if p.grad is not None:
                self._master[name].grad = p.grad.detach().float()
        for name, p in pl_module.named_parameters():
            if name in self._master:
                p.data = self._master[name].data.clone()

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx
    ) -> None:
        if not self._overflow:
            for name, p in pl_module.named_parameters():
                self._master[name].data.copy_(p.data.float())
        pl_module.half()
        inner = pl_module.model
        if hasattr(inner, "model"):
            inner = inner.model
        head = getattr(inner, "head", None)
        if head is not None:
            head.float()


def train_deepforest(
    train_csv,
    val_csv=None,
    output_dir="/checkpoints",
    run_name="default",
    epochs=10,
    batch_size=8,
    lr=0.001,
    patience=5,
    pretrained=True,
    wandb_project=None,
    checkpoint=None,
    accelerator=None,
    shadow_loss_reweight=False,
    shadow_loss_weight=2.0,
    augmentations=None,           # list of dicts; overrides default augmentation
    fast_dev_run=False,
    precision=None,               # override Lightning precision (e.g. "16-mixed", "16-mps")
):
    """
    Train a DeepForest model on TCD data with optional shadow loss reweighting.

    Args:
        train_csv:            path to training CSV (image_path, xmin, ymin, xmax, ymax, label, ...)
        val_csv:              path to validation CSV (optional; used for early stopping)
        output_dir:           base directory for checkpoints
        run_name:             creates output_dir/run_name/ subfolder
        epochs:               max training epochs
        batch_size:           batch size
        lr:                   learning rate
        patience:             early stopping patience (epochs without mAP improvement)
        pretrained:           load HuggingFace pretrained weights when no checkpoint supplied
        checkpoint:           path to initial .pth or .ckpt weights
        accelerator:          force accelerator (cpu/gpu/mps)
        shadow_loss_reweight: upweight focal loss for shadow-casting GT boxes
        shadow_loss_weight:   multiplier for shadow-casting GT anchors (default 2.0)
        augmentations:        list of albumentations dicts applied after 400px crop
        fast_dev_run:         Lightning fast_dev_run (1 train + 1 val batch then exit)
        precision:            Lightning precision string (e.g. "bf16-mixed", "16-mps")
    """
    run_output_dir = str(Path(output_dir) / run_name)
    Path(run_output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"🌲 DeepForest Training: {run_name}")
    print("=" * 60)

    print("\n⚙️  Initializing model...")

    if shadow_loss_reweight:
        model = ShadowConditionedDeepForest(
            train_csv=train_csv,
            val_csv=val_csv,
            shadow_loss_reweight=True,
            shadow_loss_weight=shadow_loss_weight,
        )
    else:
        print("   Shadow loss reweighting: DISABLED")
        model = deepforest_main.deepforest()

    import traceback

    if checkpoint:
        print(f"\n Loading checkpoint: {checkpoint}")
        try:
            state_dict       = torch.load(checkpoint, map_location="cpu")
            ck_keys          = list(state_dict.keys())
            has_model_prefix = any(k.startswith("model.") for k in ck_keys)

            if has_model_prefix and shadow_loss_reweight:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"   Loaded full wrapper checkpoint ({len(ck_keys)} keys)")
                if missing:    print(f"   Missing ({len(missing)}): {missing[:3]}...")
                if unexpected: print(f"   Unexpected ({len(unexpected)}): {unexpected[:3]}...")
            else:
                model.model.load_state_dict(state_dict, strict=False)
                print(f"   Loaded backbone-only checkpoint ({len(ck_keys)} keys)")

            print("   Checkpoint loaded successfully")
        except Exception as e:
            print(f"   Failed to load checkpoint: {e}")
            traceback.print_exc()
            raise RuntimeError("Cannot continue without loading checkpoint") from e
    elif pretrained:
        print("\n📦 Loading pretrained weights...")
        try:
            model.load_model("weecology/deepforest-tree")
            print("   ✅ Loaded HuggingFace pretrained weights")
        except Exception as e:
            print(f"   ❌ Failed to load pretrained weights: {e}")
            traceback.print_exc()
            raise RuntimeError("Cannot continue without pretrained weights") from e

    # Resume from existing Lightning checkpoint if present (Modal auto-restart support)
    checkpoint_path = None
    print("\n🔍 Searching for checkpoint to resume from...")
    checkpoint_files = list(Path(run_output_dir).glob("*.ckpt"))
    if checkpoint_files:
        checkpoint_path = str(max(checkpoint_files, key=lambda p: p.stat().st_mtime))
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if "optimizer_states" in ckpt and len(ckpt["optimizer_states"]) > 0:
                if "param_groups" not in ckpt["optimizer_states"][0]:
                    print("   ❌ Checkpoint corrupted: optimizer state missing 'param_groups'")
                    checkpoint_path = None
                else:
                    print(f"   ✅ Found valid checkpoint: {checkpoint_path}")
            else:
                print("   ⚠️  Checkpoint has no optimizer state")
        except Exception as e:
            print(f"   ❌ Checkpoint unreadable: {e}")
            checkpoint_path = None
    else:
        print(f"   ⚠️  No checkpoint found in {run_output_dir} — starting fresh")

    print("\n⚙️  Configuring training...")
    model.config.train.csv_file = train_csv
    model.config.train.root_dir = ""
    model.config.train.epochs   = epochs
    model.config.train.lr       = lr
    model.config.batch_size     = batch_size

    from omegaconf import OmegaConf
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    CROP_SIZE   = 400
    MIN_VIS     = 0.5
    BBOX_PARAMS = A.BboxParams(format="pascal_voc", label_fields=["category_ids"],
                               clip=True, min_visibility=MIN_VIS)
    crop_t      = A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE, p=1.0)

    def _build_train_transform(extra_augs):
        aug_list = []
        for a in (extra_augs or []):
            name   = list(a.keys())[0] if isinstance(a, dict) else a
            params = list(a.values())[0] if isinstance(a, dict) else {}
            cls    = getattr(A, name, None)
            if cls is not None:
                aug_list.append(cls(**params) if params else cls())
        return A.Compose([crop_t] + aug_list + [ToTensorV2()], bbox_params=BBOX_PARAMS)

    if augmentations is not None:
        model._train_transform_override = _build_train_transform(augmentations)
        model.config.train.augmentations = OmegaConf.create([])
        names = [list(a.keys())[0] if isinstance(a, dict) else a for a in augmentations]
        print(f"   🔄 Train transform: crop{CROP_SIZE} + {names} + ToTensorV2")
    elif shadow_loss_reweight:
        model._train_transform_override = _build_train_transform(None)
        model.config.train.augmentations = OmegaConf.create([])
        print(f"   🔄 Train transform: crop{CROP_SIZE} + ToTensorV2 (shadow loss reweight)")
    else:
        model._train_transform_override = _build_train_transform([{"HorizontalFlip": {"p": 0.5}}])
        print(f"   🔄 Train transform: crop{CROP_SIZE} + HorizontalFlip + ToTensorV2 (default)")

    if val_csv:
        model.config.validation.csv_file = val_csv
        model.config.validation.root_dir = ""
        print(f"   Validation enabled: {val_csv}")
    else:
        model.config.validation.csv_file = None
        print("   No validation file provided")

    # Hard-negative support: NaN xmin rows → empty-image targets
    train_df    = pd.read_csv(train_csv)
    _empty_mask = train_df["xmin"].isna() | (train_df["xmin"].astype(str).str.strip() == "")
    _empty_image_paths = train_df.loc[_empty_mask, "image_path"].unique().tolist()
    import types
    from deepforest.datasets.training import BoxDataset
    from torch.utils.data import DataLoader

    pin_memory = torch.cuda.is_available()

    if _empty_image_paths:
        print(f"\n🔲 Found {len(_empty_image_paths)} confirmed-empty (hard-negative) images")
        _clean_train_csv = "/tmp/_clean_train.csv"
        train_df[~_empty_mask].to_csv(_clean_train_csv, index=False)
        model.config.train.csv_file = _clean_train_csv

        _orig_train_dataloader = model.train_dataloader.__func__

        def _train_dataloader_with_empties(self_model):
            dl = _orig_train_dataloader(self_model)
            ds = dl.dataset
            ds.image_names = np.append(ds.image_names, _empty_image_paths)
            if hasattr(self_model, "_train_transform_override"):
                ds.transform = self_model._train_transform_override
            print(f"   ✅ Injected {len(_empty_image_paths)} empty images "
                  f"(total: {len(ds.image_names)} images)")
            return DataLoader(ds, batch_size=dl.batch_size, shuffle=True,
                              collate_fn=ds.collate_fn, num_workers=8,
                              pin_memory=pin_memory, persistent_workers=True)

        model.train_dataloader = types.MethodType(_train_dataloader_with_empties, model)
    else:
        if hasattr(model, "_train_transform_override"):
            _orig_tl = model.train_dataloader.__func__
            def _train_dataloader_with_transform(self_model):
                dl = _orig_tl(self_model)
                dl.dataset.transform = self_model._train_transform_override
                return dl
            model.train_dataloader = types.MethodType(_train_dataloader_with_transform, model)

    print(f"\n📊 Training data: {len(train_df[~_empty_mask])} boxes, "
          f"{train_df[~_empty_mask]['image_path'].nunique()} annotated images, "
          f"{len(_empty_image_paths)} empty images")

    if val_csv:
        val_df           = pd.read_csv(val_csv)
        _val_empty_mask  = val_df["xmin"].isna() | (val_df["xmin"].astype(str).str.strip() == "")
        _val_empty_paths = val_df.loc[_val_empty_mask, "image_path"].unique().tolist()

        if _val_empty_paths:
            _clean_val_csv = "/tmp/_clean_val.csv"
            val_df[~_val_empty_mask].to_csv(_clean_val_csv, index=False)
            model.config.validation.csv_file = _clean_val_csv

        _orig_val_dataloader = model.val_dataloader.__func__

        if _val_empty_paths:
            def _val_dataloader_with_empties(self_model):
                dl = _orig_val_dataloader(self_model)
                ds = dl.dataset
                ds.image_names = np.append(ds.image_names, _val_empty_paths)
                tiled_ds = _TiledValDataset(ds, patch_size=CROP_SIZE, min_visibility=MIN_VIS)
                return DataLoader(tiled_ds, batch_size=dl.batch_size, shuffle=False,
                                  collate_fn=ds.collate_fn, num_workers=8,
                                  pin_memory=pin_memory, persistent_workers=True)
            model.val_dataloader = types.MethodType(_val_dataloader_with_empties, model)
        else:
            def _val_dataloader_tiled(self_model):
                dl       = _orig_val_dataloader(self_model)
                tiled_ds = _TiledValDataset(dl.dataset, patch_size=CROP_SIZE, min_visibility=MIN_VIS)
                return DataLoader(tiled_ds, batch_size=dl.batch_size, shuffle=False,
                                  collate_fn=dl.dataset.collate_fn, num_workers=8,
                                  pin_memory=pin_memory, persistent_workers=True)
            model.val_dataloader = types.MethodType(_val_dataloader_tiled, model)

        print(f"   Validation data: {len(val_df[~_val_empty_mask])} boxes, "
              f"{val_df[~_val_empty_mask]['image_path'].nunique()} annotated images, "
              f"{len(_val_empty_paths)} empty images")
    else:
        _val_empty_paths = []

    print(f"\n⚙️  Training configuration:")
    print(f"   Epochs: {epochs}  |  Batch size: {batch_size}  |  LR: {lr}")
    print(f"   Early stopping patience: {patience}  |  Checkpoint dir: {run_output_dir}")

    callbacks = []
    _monitor  = "map"

    callbacks.append(ModelCheckpoint(
        dirpath=run_output_dir,
        filename="deepforest-{epoch:02d}-{map:.2f}",
        monitor=_monitor, mode="max", save_top_k=1, verbose=True,
    ))

    if val_csv:
        callbacks.append(EarlyStopping(
            monitor=_monitor, patience=patience, mode="max", verbose=True,
        ))

    class MetrixPrinter(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            loss = trainer.callback_metrics.get("box_loss") or trainer.callback_metrics.get("train_loss")
            if loss:
                print(f"   📉 Epoch {trainer.current_epoch} Loss: {loss:.4f}")

        def on_validation_epoch_end(self, trainer, pl_module):
            mAP = trainer.callback_metrics.get("map")
            if mAP:
                print(f"   📈 Epoch {trainer.current_epoch} mAP: {mAP:.4f}")

    callbacks.append(MetrixPrinter())

    from pytorch_lightning.loggers import CSVLogger
    logger = CSVLogger(save_dir=output_dir, name=run_name)

    trainer_kwargs = {
        "max_epochs":            epochs,
        "enable_checkpointing":  True,
        "callbacks":             callbacks,
        "logger":                logger,
        "check_val_every_n_epoch": 3,
        "num_sanity_val_steps":  0,
        "gradient_clip_val":     1.0,
        "fast_dev_run":          fast_dev_run,
    }

    if accelerator:
        trainer_kwargs["accelerator"] = accelerator
        if accelerator == "cpu":
            trainer_kwargs["devices"] = 1
    elif torch.backends.mps.is_available():
        trainer_kwargs["accelerator"] = "mps"
        trainer_kwargs["devices"]     = 1
    elif torch.cuda.is_available():
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"]     = 1
    else:
        trainer_kwargs["accelerator"] = "cpu"

    if precision == "16-mps":
        model.mps_fp16 = True
        callbacks.append(MPSHalfPrecisionCallback())
        print("   ✅ MPS fp16: backbone/FPN in fp16 | BN + head in fp32")
    elif precision:
        trainer_kwargs["precision"] = precision
    elif trainer_kwargs.get("accelerator") == "gpu":
        trainer_kwargs["precision"] = "bf16-mixed"
    elif trainer_kwargs.get("accelerator") == "mps":
        trainer_kwargs["precision"] = "32-true"  # MPS has no autocast support

    print("\n🚀 Starting training...")
    print("-" * 60)

    trainer = pl.Trainer(**trainer_kwargs)

    if trainer_kwargs.get("accelerator") == "gpu" and torch.cuda.is_available():
        try:
            model.model = torch.compile(model.model)
            print("   ✅ torch.compile applied")
        except Exception as e:
            print(f"   ⚠️  torch.compile unavailable ({e}) — running uncompiled")

    trainer.fit(model, ckpt_path=checkpoint_path)

    print("\n✅ Training complete!")

    final_model_path = Path(run_output_dir) / "deepforest_final.pth"
    if shadow_loss_reweight:
        torch.save(model.state_dict(), str(final_model_path))
    else:
        torch.save(model.model.state_dict(), str(final_model_path))
    print(f"💾 Saved final model to {final_model_path}")

    return model, None


def main():
    parser = argparse.ArgumentParser(description="Train DeepForest on TCD data")
    parser.add_argument("--train_csv",     required=True)
    parser.add_argument("--val_csv",       default=None)
    parser.add_argument("--output_dir",    default="./checkpoints")
    parser.add_argument("--run_name",      default="default")
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=0.001)
    parser.add_argument("--patience",      type=int,   default=5)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--accelerator",   default=None,
                        help="Force accelerator (cpu, gpu, mps)")
    parser.add_argument("--shadow-loss-reweight", action="store_true",
                        help="Upweight focal loss for shadow-casting GT boxes")
    parser.add_argument("--shadow-loss-weight", type=float, default=2.0,
                        help="Focal loss multiplier for shadow-casting GT anchors (default 2.0)")
    parser.add_argument("--checkpoint",    default=None,
                        help="Path to initial weights (.pth or .ckpt)")
    parser.add_argument("--fast-dev-run",  action="store_true",
                        help="Run 1 train + 1 val batch then exit")
    parser.add_argument("--precision",     default=None,
                        help="Lightning precision override (e.g. 16-mixed, bf16-mixed, 16-mps)")

    args = parser.parse_args()

    train_deepforest(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        pretrained=not args.no_pretrained,
        checkpoint=args.checkpoint,
        accelerator=args.accelerator,
        shadow_loss_reweight=args.shadow_loss_reweight,
        shadow_loss_weight=args.shadow_loss_weight,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
