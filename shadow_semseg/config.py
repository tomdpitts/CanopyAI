"""config.py — single source of truth for the shadow-semseg experiment.

The ±shadow ablation is driven entirely by `use_shadow`: same architecture,
init, data, schedule — only the loss weighting differs. Everything else is
shared so the comparison is clean.
"""
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ── experiment identity ──
    name: str = "semseg_shadow"          # run name (checkpoints/logs subdir)
    use_shadow: bool = True              # the ablation switch
    shadow_weight: float = 4.0          # loss multiplier on shadowed cover pixels
    shadow_thresh: float = 0.35         # shadow-map prob -> "in shadow"

    # ── data ──
    hf_dataset: str = "restor/tcd"
    target: str = "cover"               # "cover" (tree∪canopy binary) | "canopy_region" (cat-1 only)
    shadow_json: str = str(REPO / "data/tcd/tcd_shadow_vectors_by_id.json")
    val_fold: int = 4                   # held out from training (matches phase22)
    crop: int = 512                     # FIXED crop size -> fixed shapes (MPS-safe)
    # multiscale: tile is randomly rescaled by a factor in [scale_min, scale_max]
    # BEFORE the fixed crop, so the head sees "beyond 400px" of ground per window.
    scale_min: float = 0.5
    scale_max: float = 1.25
    num_workers: int = 4

    # ── model ──
    phase22_ckpt: str = str(REPO / "checkpoints/phase22_B_L4/deepforest-epoch=15-map=0.54.ckpt")
    num_classes: int = 2                # background, tree-cover
    head_channels: int = 128
    loss: str = "ce_dice"               # "ce_dice" (v1/v2) | "ce_lovasz" (v3, direct IoU)
    aux_loss: bool = False              # deep-supervision head on layer3 (DeepLab standard)
    aux_weight: float = 0.4
    label_smoothing: float = 0.0       # soft targets — principled for the noisy GT
    aspp_rates: tuple = (12, 24, 36)   # ASPP atrous rates; widen for 2048 global context
    freeze_backbone_epochs: int = 1     # warm up the head before unfreezing backbone
    arch: str = "fpn"                   # "fpn" (v1) | "deeplabv3plus" (v2)

    # ── v2 recipe (levers 5/6) ──
    all_folds: bool = False             # train on all train tiles (test holdout is separate)
    aug_geometric: bool = False         # hflip+vflip+rotate (joint on img/mask/weight)
    aug_color: bool = False             # brightness/contrast/blur/HSV (image only)
    rotate_deg: float = 30.0
    ema: bool = False
    ema_decay: float = 0.999
    warmup_epochs: int = 2              # linear warmup then cosine decay
    freeze_bn: bool = False             # small-batch: reuse phase22 BN stats (no bs2 noise)
    grad_accum: int = 1                # accumulate N micro-batches -> effective batch ×N

    # ── optim ──
    epochs: int = 30
    batch_size: int = 8
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = True

    # ── eval (faithful to Restor's protocol) ──
    eval_full_res: int = 2048           # eval on full tiles, do_resize=False

    # ── io ──
    out_dir: str = "/tmp/shadow_semseg/runs"   # off-iCloud: avoids sync churn on big ckpts
    imagenet_mean: tuple = (0.485, 0.456, 0.406)
    imagenet_std: tuple = (0.229, 0.224, 0.225)
    seed: int = 0

    # smoke knobs (overridden by run_smoke)
    smoke: bool = False                 # use already-downloaded test split as stand-in data
    limit_train: int = 0                # 0 = all; else first N train tiles
    limit_eval: int = 0

    def run_dir(self) -> Path:
        d = Path(self.out_dir) / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d
