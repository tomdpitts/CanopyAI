"""
MPS fp16 ops for CanopyAI training.

Importing this module compiles and loads a custom Objective-C++/MPSGraph
extension that implements BatchNorm backward for fp16 tensors on Apple MPS.
PyTorch's built-in native_batch_norm_backward rejects fp16 with an explicit
dtype check; our kernel bypasses that and calls MPSGraph directly.

Usage (done automatically by MPSHalfPrecisionCallback):
    import deepforest_custom.mps_ops as mps_ops
    mps_ops.patch_batchnorm_for_fp16(model.backbone.body)
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import types

# ---------------------------------------------------------------------------
# 1. Compile and load the Objective-C++ extension
# ---------------------------------------------------------------------------

def _load_extension():
    try:
        from .build import build
        build()
        print("[mps_ops] Custom MPS BN backward kernel loaded.")
        return True
    except Exception as e:
        print(f"[mps_ops] WARNING: Failed to compile MPS kernel ({e}). "
              "Falling back to fp32-cast BN backward.")
        return False


_KERNEL_AVAILABLE = _load_extension()


# ---------------------------------------------------------------------------
# 2. autograd.Function: BN forward in fp16, backward via our kernel (or fallback)
# ---------------------------------------------------------------------------

class _MPS_BN_FP16(torch.autograd.Function):
    """
    BatchNorm that runs the forward pass in fp16 on MPS and uses
    either the custom MPSGraph kernel (if compiled) or a pure-Python
    component-op fallback for the backward pass.

    Both paths produce correct fp16 gradients without hitting MPS's
    built-in fp16 restriction on native_batch_norm_backward.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_var,
                training, momentum, eps):
        # BN forward: cast to fp32 so the standard MPS fp32 kernel is used,
        # then cast output back to fp16.  Saves fp16 tensors for backward.
        with torch.no_grad():
            out, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
                input.float(),
                weight.float() if weight is not None else None,
                bias.float()   if bias   is not None else None,
                running_mean,
                running_var,
                training, momentum, eps,
            )
        ctx.save_for_backward(input, weight, save_mean, save_invstd)
        ctx.eps      = eps
        ctx.training = training
        return out.to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, save_mean, save_invstd = ctx.saved_tensors
        eps = ctx.eps

        if _KERNEL_AVAILABLE and input.device.type == "mps":
            # Custom MPSGraph kernel — fp16 backward without casting (MPS only)
            grad_input, grad_weight, grad_bias = torch.ops.canopyai.mps_bn_backward_fp16(
                grad_output.contiguous(),
                input.contiguous(),
                weight.contiguous() if weight is not None else torch.ones(input.size(1), device=input.device, dtype=input.dtype),
                save_mean,
                save_invstd,
                eps,
            )
        else:
            # Pure-Python fallback: BN backward via component ops.
            # All ops used here (sum, multiply, reshape) support fp16 on MPS.
            N, C, H, W = input.shape
            m = float(N * H * W)

            g_f    = grad_output.float()
            in_f   = input.float()
            w_f    = (weight.float() if weight is not None
                      else torch.ones(C, device=input.device))
            mean   = save_mean.float().view(1, C, 1, 1)
            invstd = save_invstd.float().view(1, C, 1, 1)

            x_hat       = (in_f - mean) * invstd
            dy          = g_f * w_f.view(1, C, 1, 1)
            sum_dy      = dy.sum(dim=[0, 2, 3])
            sum_dy_xhat = (dy * x_hat).sum(dim=[0, 2, 3])

            grad_input_f = (dy
                            - sum_dy.view(1, C, 1, 1) / m
                            - x_hat * sum_dy_xhat.view(1, C, 1, 1) / m) * invstd

            grad_input  = grad_input_f.to(input.dtype)
            grad_weight = (g_f * x_hat).sum(dim=[0, 2, 3]).to(
                weight.dtype if weight is not None else input.dtype)
            grad_bias   = g_f.sum(dim=[0, 2, 3]).to(
                weight.dtype if weight is not None else input.dtype)

        return (grad_input, grad_weight, grad_bias,
                None, None, None, None, None)  # no grads for non-tensor args


# ---------------------------------------------------------------------------
# 3. Module patcher: replace BN.forward with our fp16-safe version
# ---------------------------------------------------------------------------

def _make_fp16_forward(module: nn.BatchNorm2d):
    """
    Returns a bound forward function that uses _MPS_BN_FP16 when the input
    is fp16 on MPS, and falls back to the standard BN forward otherwise.
    """
    original_forward = module.__class__.forward  # unbound method

    def fp16_forward(self, input: Tensor) -> Tensor:
        if input.dtype == torch.float16 and input.device.type == "mps":
            return _MPS_BN_FP16.apply(
                input, self.weight, self.bias,
                self.running_mean, self.running_var,
                self.training, self.momentum, self.eps,
            )
        return original_forward(self, input)

    return types.MethodType(fp16_forward, module)


def patch_batchnorm_for_fp16(root: nn.Module) -> int:
    """
    Walk *root* and replace the forward method of every BatchNorm2d /
    SyncBatchNorm with an fp16-safe version.

    Returns the number of modules patched.
    """
    _BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    n = 0
    for m in root.modules():
        if isinstance(m, _BN_TYPES):
            m.forward = _make_fp16_forward(m)
            n += 1
    return n
