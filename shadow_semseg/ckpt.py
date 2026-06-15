"""ckpt.py — load the phase22_B_L4 deepforest checkpoint WITHOUT installing
deepforest. The Lightning ckpt pickles deepforest/omegaconf objects in its
hyper-parameters; we stub any missing module on demand so the tensor state_dict
unpickles cleanly, then extract just the ResNet-FPN backbone weights.
"""
import sys
import types

import torch


class _Any:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass
    def __call__(self, *a, **k): return _Any()
    def __getattr__(self, n): return _Any()


def _make_stub(name):
    m = types.ModuleType(name)
    m.__path__ = []

    def _ga(n):
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)
        return _Any
    m.__getattr__ = _ga
    sys.modules[name] = m


def robust_load(path, max_stubs=64):
    """torch.load a checkpoint whose pickle references uninstalled modules."""
    for _ in range(max_stubs):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as e:
            _make_stub(e.name)
    raise RuntimeError(f"Could not load {path}: too many missing modules")


def load_backbone_state_dict(path, prefix="model.backbone."):
    """Return {body.*, fpn.*: tensor} extracted from the phase22 checkpoint,
    ready to load into a torchvision retinanet backbone-with-FPN."""
    ck = robust_load(path)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    if not out:
        raise RuntimeError(
            f"No keys with prefix {prefix!r} in checkpoint {path}; "
            f"sample keys: {list(sd)[:5]}")
    return out
