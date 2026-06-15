"""model.py — Semantic-FPN head on phase22_B_L4's ResNet-50-FPN backbone.

The backbone matches deepforest's torchvision `retinanet_resnet50_fpn` exactly, so
phase22's shadow-trained backbone+FPN weights load directly. A lightweight
Panoptic-FPN-style semantic head turns the FPN features into per-pixel tree-cover
logits. Inference is a plain forward (no shadow input).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import ASPP

from ckpt import load_backbone_state_dict


class SemFPNHead(nn.Module):
    def __init__(self, n_levels, in_ch=256, inter=128, num_classes=2):
        super().__init__()
        g = min(32, inter)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, inter, 3, padding=1),
                          nn.GroupNorm(g, inter), nn.ReLU(inplace=True))
            for _ in range(n_levels)
        ])
        self.classifier = nn.Conv2d(inter, num_classes, 1)

    def forward(self, feats, out_hw):
        fmaps = list(feats.values())
        target = fmaps[0].shape[-2:]            # finest FPN level (P3)
        x = None
        for blk, f in zip(self.blocks, fmaps):
            y = blk(f)
            if y.shape[-2:] != target:
                y = F.interpolate(y, size=target, mode="bilinear", align_corners=False)
            x = y if x is None else x + y
        x = self.classifier(x)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)


class ShadowSemSeg(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats, x.shape[-2:])


class _FCNHead(nn.Module):
    """Standard FCN/DeepLab auxiliary head for deep supervision."""
    def __init__(self, in_ch, num_classes, inter=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, inter, 3, padding=1, bias=False), nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Conv2d(inter, num_classes, 1))

    def forward(self, x):
        return self.block(x)


class DeepLabV3Plus(nn.Module):
    """Stock torchvision dilated ResNet-50 + ASPP (both first-party) with a native
    DeepLabV3+ stride-4 decoder (low-level skip). Backbone init from phase22's body.
    Optional aux deep-supervision head on layer3 (training only)."""
    def __init__(self, num_classes=2, aspp_rates=(12, 24, 36), use_aux=False):
        super().__init__()
        rn = resnet50(weights=None, replace_stride_with_dilation=[False, True, True])
        self.use_aux = use_aux
        layers = {"layer1": "low", "layer4": "out"}
        if use_aux:
            layers["layer3"] = "aux"
        self.backbone = IntermediateLayerGetter(rn, return_layers=layers)
        self.aspp = ASPP(2048, list(aspp_rates), 256)
        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.classifier = nn.Conv2d(256, num_classes, 1)
        self.aux_head = _FCNHead(1024, num_classes) if use_aux else None
        self._aux_logits = None                        # read by train.py for the aux loss

    def forward(self, x):
        H, W = x.shape[-2:]
        f = self.backbone(x)
        low = self.low_proj(f["low"])                  # OS4, 48ch
        hi = self.aspp(f["out"])                        # OS8, 256ch
        hi = F.interpolate(hi, size=low.shape[-2:], mode="bilinear", align_corners=False)
        d = self.decoder(torch.cat([hi, low], 1))
        main = F.interpolate(self.classifier(d), size=(H, W), mode="bilinear", align_corners=False)
        if self.training and self.use_aux:
            self._aux_logits = F.interpolate(self.aux_head(f["aux"]), size=(H, W),
                                             mode="bilinear", align_corners=False)
        else:
            self._aux_logits = None
        return main


def _phase22_body(cfg):
    """phase22 ResNet-50 body weights, keys stripped to torchvision resnet naming."""
    bb = load_backbone_state_dict(cfg.phase22_ckpt)    # {body.*, fpn.*}
    return {k[len("body."):]: v for k, v in bb.items() if k.startswith("body.")}


def build_fpn(cfg, verbose=True):
    det = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=cfg.num_classes)
    backbone = det.backbone
    bb_sd = load_backbone_state_dict(cfg.phase22_ckpt)
    missing, unexpected = backbone.load_state_dict(bb_sd, strict=False)
    if verbose:
        print(f"[model:fpn] loaded {len(bb_sd)-len(unexpected)}/{len(bb_sd)} phase22 "
              f"backbone tensors (missing={len(missing)}, unexpected={len(unexpected)})")
    with torch.no_grad():
        feats = backbone(torch.zeros(1, 3, 64, 64))
    head = SemFPNHead(len(feats), in_ch=list(feats.values())[0].shape[1],
                      inter=cfg.head_channels, num_classes=cfg.num_classes)
    return ShadowSemSeg(backbone, head)


def build_deeplabv3plus(cfg, verbose=True):
    model = DeepLabV3Plus(num_classes=cfg.num_classes,
                          aspp_rates=getattr(cfg, "aspp_rates", (12, 24, 36)),
                          use_aux=getattr(cfg, "aux_loss", False))
    body = _phase22_body(cfg)
    missing, unexpected = model.backbone.load_state_dict(body, strict=False)
    if verbose:
        print(f"[model:deeplabv3plus] loaded {len(body)-len(unexpected)}/{len(body)} "
              f"phase22 ResNet body tensors (missing={len(missing)}, unexpected={len(unexpected)})")
    return model


def build_model(cfg, verbose=True):
    if cfg.arch == "deeplabv3plus":
        return build_deeplabv3plus(cfg, verbose)
    return build_fpn(cfg, verbose)


def set_backbone_trainable(model, trainable: bool):
    # freezes model.backbone only; head/aspp/decoder/classifier stay trainable (warm-up)
    for p in model.backbone.parameters():
        p.requires_grad = trainable
