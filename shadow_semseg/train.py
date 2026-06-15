"""train.py — train the shadow-semseg head from phase22_B_L4.

Robustness features:
  - fixed-shape crops (MPS-graph-cache safe), AMP (bf16 on MPS, fp16 on CUDA)
  - backbone warm-up (head-only for freeze_backbone_epochs), then joint fine-tune
  - checkpoint + auto-resume (last.pt), best-by-val-tree-F1 (best.pt)
  - shadow ablation is pure config: --no-shadow flips the loss weighting only

Usage:
  python train.py --name semseg_shadow                 # +shadow (weight 4.0)
  python train.py --name semseg_noshadow --no-shadow   # control
  python train.py --smoke                               # tiny wiring test
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import F1Score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config
from data import TCDSemanticDataset
from model import build_model, set_backbone_trainable


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def soft_dice(logits, target, eps=1.0):
    p = logits.softmax(1)[:, 1]
    t = (target == 1).float()
    num = 2 * (p * t).sum(dim=(1, 2))
    den = (p + t).sum(dim=(1, 2))
    return (1 - (num + eps) / (den + eps)).mean()


def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p].clone() - jaccard[0:-1].clone()
    return jaccard


def lovasz_softmax(logits, target):
    """Multiclass Lovász-softmax (Berman et al. 2018), 'present' classes — native
    implementation, no third-party. Directly optimises the IoU surrogate."""
    probs = logits.softmax(1)
    B, C, H, W = probs.shape
    probs = probs.permute(0, 2, 3, 1).reshape(-1, C)
    labels = target.reshape(-1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
    return torch.stack(losses).mean() if losses else probs.sum() * 0.0


def loss_fn(logits, target, weight, kind="ce_dice", label_smoothing=0.0):
    ce = F.cross_entropy(logits, target, reduction="none", label_smoothing=label_smoothing)
    wce = (ce * weight).mean()
    if kind == "ce_lovasz":
        return wce + lovasz_softmax(logits, target)
    return wce + soft_dice(logits, target)


def lr_at(epoch, cfg, base):
    """Linear warmup for warmup_epochs, then cosine decay to 0."""
    if epoch < cfg.warmup_epochs:
        return base * (epoch + 1) / max(cfg.warmup_epochs, 1)
    prog = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
    return 0.5 * base * (1 + math.cos(math.pi * min(prog, 1.0)))


def apply_freeze_bn(model):
    """Put all BatchNorm in eval mode + freeze affine — uses phase22's stable running
    stats instead of noisy small-batch ones. Call after every model.train()."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad_(False)
            if m.bias is not None:
                m.bias.requires_grad_(False)


class EMA:
    """Exponential moving average of parameters; swap in for eval/save, restore after."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.named_parameters()}
        self.backup = {}

    def update(self, model):
        for k, v in model.named_parameters():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.named_parameters()}
        for k, v in model.named_parameters():
            if k in self.shadow:
                v.data.copy_(self.shadow[k])

    def restore(self, model):
        for k, v in model.named_parameters():
            if k in self.backup:
                v.data.copy_(self.backup[k])
        self.backup = {}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="semseg_shadow")
    ap.add_argument("--no-shadow", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--crop", type=int, default=None)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-eval", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore existing last.pt")
    ap.add_argument("--seed", type=int, default=None, help="for ensemble diversity")
    ap.add_argument("--loss", default=None, help="ce_dice | ce_lovasz")
    ap.add_argument("--scale-min", type=float, default=None,
                    help="multiscale floor; 0.25 => a 512 crop = the whole 2048 tile")
    ap.add_argument("--scale-max", type=float, default=None)
    ap.add_argument("--phase22-ckpt", default=None, help="backbone-init ckpt (Modal: volume path)")
    ap.add_argument("--out-dir", default=None, help="output dir (Modal: /checkpoints/...)")
    ap.add_argument("--freeze-bn", action="store_true", help="freeze BatchNorm (small batch)")
    ap.add_argument("--grad-accum", type=int, default=None, help="gradient accumulation steps")
    ap.add_argument("--aux-loss", action="store_true", help="deep-supervision head on layer3")
    ap.add_argument("--label-smoothing", type=float, default=None)
    ap.add_argument("--aspp-rates", default=None, help="comma, e.g. 18,36,54 (wider for 2048)")
    ap.add_argument("--arch", default=None, help="fpn | deeplabv3plus")
    ap.add_argument("--target", default=None, help="cover | canopy_region")
    ap.add_argument("--init-ckpt", default=None,
                    help="warm-start full model from this checkpoint (e.g. v3 cover model)")
    ap.add_argument("--v2", action="store_true",
                    help="DeepLabV3+ + all-folds + full geom/color aug + EMA")
    return ap.parse_args()


def make_cfg(a):
    cfg = Config(name=a.name, use_shadow=not a.no_shadow)
    if a.epochs is not None: cfg.epochs = a.epochs
    if a.batch_size is not None: cfg.batch_size = a.batch_size
    if a.crop is not None: cfg.crop = a.crop
    if a.num_workers is not None: cfg.num_workers = a.num_workers
    cfg.limit_train = a.limit_train
    cfg.limit_eval = a.limit_eval
    if a.seed is not None:
        cfg.seed = a.seed
    if a.loss:
        cfg.loss = a.loss
    if a.scale_min is not None:
        cfg.scale_min = a.scale_min
    if a.scale_max is not None:
        cfg.scale_max = a.scale_max
    if a.phase22_ckpt:
        cfg.phase22_ckpt = a.phase22_ckpt
    if a.out_dir:
        cfg.out_dir = a.out_dir
    if a.freeze_bn:
        cfg.freeze_bn = True
    if a.grad_accum is not None:
        cfg.grad_accum = a.grad_accum
    if a.aux_loss:
        cfg.aux_loss = True
    if a.label_smoothing is not None:
        cfg.label_smoothing = a.label_smoothing
    if a.aspp_rates:
        cfg.aspp_rates = tuple(int(x) for x in a.aspp_rates.split(","))
    if a.arch:
        cfg.arch = a.arch
    if a.target:
        cfg.target = a.target
    if a.v2:
        cfg.arch = "deeplabv3plus"
        cfg.all_folds = True
        cfg.aug_geometric = True
        cfg.aug_color = True
        cfg.ema = True
    if a.smoke:
        cfg.smoke = True
        cfg.epochs = 1; cfg.batch_size = 2; cfg.crop = 256
        cfg.limit_train = 6; cfg.limit_eval = 4; cfg.num_workers = 0
        cfg.freeze_backbone_epochs = 0
        cfg.all_folds = False  # smoke uses test-split stand-in
    return cfg


def main():
    a = parse_args()
    cfg = make_cfg(a)
    torch.manual_seed(cfg.seed)
    dev = get_device()
    run = cfg.run_dir()
    log = open(run / "train.log", "a")

    def say(*m):
        s = "[%s] " % time.strftime("%H:%M:%S") + " ".join(str(x) for x in m)
        print(s, flush=True); log.write(s + "\n"); log.flush()

    say(f"device={dev} name={cfg.name} use_shadow={cfg.use_shadow} "
        f"shadow_weight={cfg.shadow_weight} epochs={cfg.epochs} bs={cfg.batch_size} crop={cfg.crop}")
    (run / "config.json").write_text(json.dumps(cfg.__dict__, default=str, indent=2))

    tr = TCDSemanticDataset(cfg, "train")
    va = TCDSemanticDataset(cfg, "val")
    tl = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, drop_last=True, pin_memory=False)
    vl = DataLoader(va, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=False)

    model = build_model(cfg).to(dev)
    if a.init_ckpt:
        ick = torch.load(a.init_ckpt, map_location=dev)
        sd = ick.get("model", ick)
        miss, unexp = model.load_state_dict(sd, strict=False)
        say(f"warm-started from {a.init_ckpt} (missing={len(miss)} unexpected={len(unexp)})")
    # generic param groups: backbone vs everything-else (works for fpn + deeplab)
    bb_ids = {id(p) for p in model.backbone.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in bb_ids]
    bb_params = [p for p in model.parameters() if id(p) in bb_ids]
    opt = torch.optim.AdamW([
        {"params": head_params, "lr": cfg.lr_head},
        {"params": bb_params, "lr": cfg.lr_backbone},
    ], weight_decay=cfg.weight_decay)

    use_amp = cfg.amp and dev in ("cuda", "mps")
    amp_dtype = torch.float16 if dev == "cuda" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda" and cfg.amp))
    ema = EMA(model, cfg.ema_decay) if cfg.ema else None

    start_epoch, best = 0, -1.0
    last_pt = run / "last.pt"
    if last_pt.exists() and not a.fresh:
        ck = torch.load(last_pt, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1; best = ck.get("best", -1.0)
        if ema and ck.get("ema"):
            for k in ema.shadow:
                if k in ck["ema"]:
                    ema.shadow[k] = ck["ema"][k].to(dev)
        say(f"resumed from epoch {ck['epoch']} (best tree-F1 {best:.4f})")

    f1 = F1Score(task="multiclass", num_classes=2, average="none")
    say(f"steps/epoch={len(tl)}  planned_total_steps={len(tl)*cfg.epochs}  "
        f"val_tiles={len(va)}  (val monitored each epoch)")

    for epoch in range(start_epoch, cfg.epochs):
        set_backbone_trainable(model, epoch >= cfg.freeze_backbone_epochs)
        opt.param_groups[0]["lr"] = lr_at(epoch, cfg, cfg.lr_head)
        opt.param_groups[1]["lr"] = lr_at(epoch, cfg, cfg.lr_backbone)
        model.train()
        if cfg.freeze_bn:
            apply_freeze_bn(model)
        t0 = time.time(); running = 0.0
        opt.zero_grad(set_to_none=True)
        for bi, b in enumerate(tl):
            img = b["image"].to(dev); tgt = b["mask"].to(dev); w = b["weight"].to(dev)
            with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):
                logits = model(img)
                loss = loss_fn(logits.float(), tgt, w, cfg.loss, cfg.label_smoothing)
                aux = getattr(model, "_aux_logits", None)
                if aux is not None:                       # deep-supervision (training only)
                    loss = loss + cfg.aux_weight * loss_fn(
                        aux.float(), tgt, w, cfg.loss, cfg.label_smoothing)
                loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()
            if (bi + 1) % cfg.grad_accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)
            running += loss.item() * cfg.grad_accum
            if (bi + 1) % 20 == 0:
                sit = (time.time() - t0) / (bi + 1)
                say(f"  e{epoch} {bi+1}/{len(tl)} loss={running/(bi+1):.4f} "
                    f"{sit:.2f}s/it  ~epoch {sit*len(tl)/60:.1f}min  "
                    f"~full {sit*len(tl)*cfg.epochs/3600:.1f}h (train-only)")

        # ── validation: tree-F1 on val crops (EMA weights if enabled = what we deploy) ──
        model.eval()
        if ema:
            ema.apply_to(model)
        f1.reset()
        with torch.no_grad():
            for b in vl:
                img = b["image"].to(dev); tgt = b["mask"]
                with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):
                    pred = model(img).argmax(1).cpu()
                f1.update(pred.flatten(), tgt.flatten())
        tree_f1 = float(f1.compute()[1])
        say(f"epoch {epoch} done  train_loss={running/max(len(tl),1):.4f}  val_tree_F1={tree_f1:.4f}")
        if tree_f1 > best:                                  # save EMA weights (in model now)
            best = tree_f1
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_tree_F1": best,
                        "cfg": cfg.__dict__}, run / "best.pt")
            say(f"  ** new best val_tree_F1={best:.4f} -> best.pt")
        if ema:
            ema.restore(model)                             # back to raw for training + resume
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "best": best, "cfg": cfg.__dict__,
                    "ema": (ema.shadow if ema else None)}, last_pt)

    say(f"TRAIN COMPLETE best val_tree_F1={best:.4f}")


if __name__ == "__main__":
    main()
