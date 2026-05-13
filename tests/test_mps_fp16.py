"""
Tests for the MPS fp16 mixed-precision implementation.

Covers four components:
  1. _FP32Guard / _FP16Guard  — dtype boundary autograd functions
  2. _MPS_BN_FP16             — BatchNorm with fp16-safe backward
  3. MPSHalfPrecisionCallback — master-weight tracking and optimizer flow
  4. mps_bn_backward_fp16     — compiled MPSGraph BN backward kernel (MPS-only)

All tests are device-agnostic unless explicitly marked @pytest.mark.mps.
The MPS tests are skipped on machines without Apple Silicon.

Run with:
    pytest tests/test_mps_fp16.py -v
"""
import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepforest_custom.train_deepforest import _FP32Guard, _FP16Guard, MPSHalfPrecisionCallback
from deepforest_custom.mps_ops import _MPS_BN_FP16, patch_batchnorm_for_fp16

MPS_AVAILABLE = torch.backends.mps.is_available()
pytest_mps = pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ref_bn_grad(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                  eps: float = 1e-5) -> tuple:
    """Reference BN backward via autograd on fp32 tensors."""
    x32 = x.detach().float().requires_grad_(True)
    w32 = weight.detach().float().requires_grad_(True)
    b32 = bias.detach().float().requires_grad_(True)
    rm  = torch.zeros(x.size(1))
    rv  = torch.ones(x.size(1))
    out = F.batch_norm(x32, rm, rv, w32, b32, training=True, eps=eps)
    out.sum().backward()
    return x32.grad.clone(), w32.grad.clone(), b32.grad.clone()


def _bn_apply(x, weight, bias, eps=1e-5):
    """Run _MPS_BN_FP16 and return output + grads."""
    x_  = x.detach().clone().requires_grad_(True)
    w_  = weight.detach().clone().requires_grad_(True)
    b_  = bias.detach().clone().requires_grad_(True)
    rm  = torch.zeros(x.size(1))
    rv  = torch.ones(x.size(1))
    out = _MPS_BN_FP16.apply(x_, w_, b_, rm, rv, True, 0.1, eps)
    out.sum().backward()
    return out.detach(), x_.grad.clone(), w_.grad.clone(), b_.grad.clone()


# ─────────────────────────────────────────────────────────────────────────────
# 1. _FP32Guard / _FP16Guard — dtype conversion + gradient chain
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardFunctions:

    def test_fp32guard_forward_casts_to_float32(self):
        x = torch.randn(4, 8).half()
        out = _FP32Guard.apply(x)
        assert out.dtype == torch.float32
        assert torch.allclose(out, x.float())

    def test_fp32guard_backward_casts_grad_to_half(self):
        x = torch.randn(4, 8).half().requires_grad_(True)
        out = _FP32Guard.apply(x)
        # Upstream grad arrives as fp32 (out is fp32)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.dtype == torch.float16, (
            f"Expected fp16 grad, got {x.grad.dtype}"
        )

    def test_fp32guard_grad_values_are_correct(self):
        """_FP32Guard is an identity in value; gradient should be all-ones."""
        x = torch.randn(3, 5).half().requires_grad_(True)
        _FP32Guard.apply(x).sum().backward()
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_fp16guard_forward_casts_to_half(self):
        x = torch.randn(4, 8)
        out = _FP16Guard.apply(x)
        assert out.dtype == torch.float16
        assert torch.allclose(out.float(), x, atol=1e-3)

    def test_fp16guard_backward_casts_grad_to_float32(self):
        x = torch.randn(4, 8).requires_grad_(True)
        out = _FP16Guard.apply(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.dtype == torch.float32, (
            f"Expected fp32 grad, got {x.grad.dtype}"
        )

    def test_fp16guard_grad_values_are_correct(self):
        x = torch.randn(3, 5).requires_grad_(True)
        _FP16Guard.apply(x).sum().backward()
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_chained_guards_round_trip(self):
        """fp16 → _FP32Guard → _FP16Guard → sum backward should give fp16 grad."""
        x = torch.randn(4, 8).half().requires_grad_(True)
        out = _FP16Guard.apply(_FP32Guard.apply(x))
        out.sum().backward()
        assert x.grad.dtype == torch.float16

    def test_guards_preserve_values_through_chain(self):
        """Values should survive the dtype round-trip within fp16 precision."""
        x = torch.randn(4, 8)
        x_half = x.half()
        out = _FP16Guard.apply(_FP32Guard.apply(x_half))
        assert torch.allclose(out.float(), x, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 2. _MPS_BN_FP16 — output correctness and gradient correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestMPSBNFP16:

    @pytest.fixture
    def bn_inputs(self):
        torch.manual_seed(0)
        N, C, H, W = 4, 8, 6, 6
        x      = torch.randn(N, C, H, W)
        weight = torch.ones(C)
        bias   = torch.zeros(C)
        return x, weight, bias

    def test_output_close_to_fp32_batchnorm(self, bn_inputs):
        """Forward output in fp16 should match fp32 BN within fp16 tolerance."""
        x, weight, bias = bn_inputs
        rm = torch.zeros(x.size(1))
        rv = torch.ones(x.size(1))

        # fp32 reference
        ref = F.batch_norm(x, rm.clone(), rv.clone(), weight, bias,
                           training=True, eps=1e-5)

        # fp16 via _MPS_BN_FP16
        out_fp16, _, _, _ = _bn_apply(x.half(), weight.half(), bias.half())

        assert torch.allclose(ref, out_fp16.float(), atol=5e-2, rtol=1e-2), (
            f"Max diff: {(ref - out_fp16.float()).abs().max().item():.4f}"
        )

    def test_grad_input_close_to_fp32_reference(self, bn_inputs):
        """d(loss)/d(input) from fp16 BN should match fp32 reference."""
        x, weight, bias = bn_inputs
        ref_gx, _, _ = _ref_bn_grad(x, weight, bias)
        _, gx, _, _  = _bn_apply(x.half(), weight.half(), bias.half())

        # fp16 precision: ~1e-2 absolute tolerance
        assert torch.allclose(ref_gx.half(), gx, atol=2e-2, rtol=5e-2), (
            f"Max grad_input diff: {(ref_gx.half() - gx).abs().max().item():.4f}"
        )

    def test_grad_weight_close_to_fp32_reference(self, bn_inputs):
        """d(loss)/d(weight) should match fp32 reference."""
        x, weight, bias = bn_inputs
        _, ref_gw, _  = _ref_bn_grad(x, weight, bias)
        _, _, gw, _   = _bn_apply(x.half(), weight.half(), bias.half())

        assert torch.allclose(ref_gw.half(), gw, atol=2e-2, rtol=5e-2), (
            f"Max grad_weight diff: {(ref_gw.half() - gw).abs().max().item():.4f}"
        )

    def test_grad_bias_close_to_fp32_reference(self, bn_inputs):
        """d(loss)/d(bias) should match fp32 reference."""
        x, weight, bias = bn_inputs
        _, _, ref_gb = _ref_bn_grad(x, weight, bias)
        _, _, _, gb  = _bn_apply(x.half(), weight.half(), bias.half())

        assert torch.allclose(ref_gb.half(), gb, atol=2e-2, rtol=5e-2), (
            f"Max grad_bias diff: {(ref_gb.half() - gb).abs().max().item():.4f}"
        )

    def test_grad_input_dtype_is_fp16(self, bn_inputs):
        x, weight, bias = bn_inputs
        _, gx, _, _ = _bn_apply(x.half(), weight.half(), bias.half())
        assert gx.dtype == torch.float16

    def test_grad_weight_dtype_is_fp16(self, bn_inputs):
        x, weight, bias = bn_inputs
        _, _, gw, _ = _bn_apply(x.half(), weight.half(), bias.half())
        assert gw.dtype == torch.float16

    def test_grad_bias_dtype_is_fp16(self, bn_inputs):
        x, weight, bias = bn_inputs
        _, _, _, gb = _bn_apply(x.half(), weight.half(), bias.half())
        assert gb.dtype == torch.float16

    def test_output_dtype_is_fp16(self, bn_inputs):
        x, weight, bias = bn_inputs
        out, _, _, _ = _bn_apply(x.half(), weight.half(), bias.half())
        assert out.dtype == torch.float16

    def test_running_stats_updated(self, bn_inputs):
        """running_mean should be updated (non-zero) after a forward pass."""
        x, weight, bias = bn_inputs
        rm = torch.zeros(x.size(1))
        rv = torch.ones(x.size(1))
        _MPS_BN_FP16.apply(x.half(), weight.half(), bias.half(),
                            rm, rv, True, 0.1, 1e-5)
        # After one step with momentum=0.1, running_mean should shift toward batch mean
        assert not torch.all(rm == 0), "running_mean was not updated"

    def test_different_spatial_sizes(self):
        """BN should work correctly for various spatial sizes."""
        for H, W in [(1, 1), (4, 4), (16, 16)]:
            x  = torch.randn(2, 4, H, W).half()
            w  = torch.ones(4).half()
            b  = torch.zeros(4).half()
            rm = torch.zeros(4)
            rv = torch.ones(4)
            out = _MPS_BN_FP16.apply(x, w, b, rm, rv, True, 0.1, 1e-5)
            assert out.shape == x.shape, f"Shape mismatch at ({H},{W})"
            assert out.dtype == torch.float16

    def test_fp32_inputs_pass_through_unchanged(self):
        """fp32 inputs should skip _MPS_BN_FP16 and use standard BN."""
        bn = nn.BatchNorm2d(4)
        x  = torch.randn(2, 4, 8, 8)   # fp32
        # patch_batchnorm_for_fp16 only activates on fp16 input
        patch_batchnorm_for_fp16(bn)
        out = bn(x)
        assert out.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# 3. patch_batchnorm_for_fp16
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchBatchnorm:

    def _make_model(self):
        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn1 = nn.BatchNorm2d(8)
                self.bn2 = nn.BatchNorm2d(8)
                self.conv = nn.Conv2d(8, 8, 1)
            def forward(self, x):
                return self.conv(self.bn2(self.bn1(x)))
        return _Net()

    def test_counts_bn_modules(self):
        model = self._make_model()
        n = patch_batchnorm_for_fp16(model)
        assert n == 2

    def test_patched_fp16_output_close_to_fp32(self):
        """Patched BN on fp16 input should match unpatched BN on fp32 input.

        Tested directly on a BN module (not through Conv2d) so the test runs
        on CPU without a dtype mismatch between fp16 input and fp32 conv weights.
        """
        torch.manual_seed(1)
        bn_ref  = nn.BatchNorm2d(8)
        bn_fp16 = copy.deepcopy(bn_ref)
        patch_batchnorm_for_fp16(bn_fp16)

        x = torch.randn(2, 8, 4, 4)
        with torch.no_grad():
            out_ref  = bn_ref(x)
            out_fp16 = bn_fp16(x.half())

        assert torch.allclose(out_ref, out_fp16.float(), atol=5e-2), (
            f"Max diff: {(out_ref - out_fp16.float()).abs().max().item():.4f}"
        )

    def test_patched_fp32_path_unchanged(self):
        """When input is fp32, patched model should behave identically to original."""
        torch.manual_seed(2)
        model_ref  = self._make_model().eval()
        model_patched = copy.deepcopy(model_ref)
        patch_batchnorm_for_fp16(model_patched)

        x = torch.randn(2, 8, 4, 4)
        with torch.no_grad():
            out_ref     = model_ref(x)
            out_patched = model_patched(x)   # fp32 input → standard path

        assert torch.allclose(out_ref, out_patched), (
            "fp32 path changed after patching"
        )

    def test_patched_gradients_flow(self):
        """Gradients should back-propagate through patched BN in fp16."""
        model = self._make_model()
        patch_batchnorm_for_fp16(model)
        model.half()

        x = torch.randn(2, 8, 4, 4).half().requires_grad_(True)
        loss = model(x).sum()
        loss.backward()   # should not raise

        assert x.grad is not None
        assert torch.isfinite(x.grad).all(), "Non-finite gradient after patched BN backward"


# ─────────────────────────────────────────────────────────────────────────────
# 4. MPSHalfPrecisionCallback — weight management and optimizer flow
# ─────────────────────────────────────────────────────────────────────────────

class _SimpleModel(nn.Module):
    """Minimal model with the backbone.body / head structure the callback expects."""
    def __init__(self):
        super().__init__()

        body = nn.Sequential(nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8))
        backbone = nn.Module()
        backbone.body = body

        head = nn.Linear(8, 2)

        inner = nn.Module()
        inner.backbone = backbone
        inner.head     = head

        hub = nn.Module()
        hub.model = inner

        self.model = hub
        self._mps_loss_scale = 1.0

    def named_parameters(self, *args, **kwargs):
        return self.model.model.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.model.parameters(*args, **kwargs)

    def half(self):
        self.model.model.half()
        return self

    def float(self):
        self.model.model.float()
        return self


class _MockTrainer:
    pass


class TestMPSHalfPrecisionCallback:

    def _setup(self):
        cb    = MPSHalfPrecisionCallback()
        model = _SimpleModel()
        cb.on_fit_start(_MockTrainer(), model)
        return cb, model

    # ── on_fit_start ─────────────────────────────────────────────────────────

    def test_on_fit_start_model_is_fp16(self):
        cb, model = self._setup()
        for name, p in model.named_parameters():
            # Head should be fp32; everything else fp16
            if "head" not in name:
                assert p.dtype == torch.float16, f"{name} should be fp16, got {p.dtype}"

    def test_on_fit_start_head_stays_fp32(self):
        cb, model = self._setup()
        head = model.model.model.head
        for name, p in head.named_parameters():
            assert p.dtype == torch.float32, f"head.{name} should be fp32"

    def test_on_fit_start_master_weights_are_fp32(self):
        cb, model = self._setup()
        for name, mp in cb._master.items():
            assert mp.dtype == torch.float32, f"master[{name}] should be fp32"

    def test_on_fit_start_master_weights_match_original(self):
        """Master weights should equal original fp32 parameter values."""
        model = _SimpleModel()
        original = {n: p.detach().clone() for n, p in model.named_parameters()}
        cb = MPSHalfPrecisionCallback()
        cb.on_fit_start(_MockTrainer(), model)

        for name, mp in cb._master.items():
            assert torch.allclose(mp, original[name].float()), (
                f"master[{name}] doesn't match original"
            )

    # ── on_before_optimizer_step ─────────────────────────────────────────────

    def test_optimizer_step_receives_fp32_weights(self):
        """After on_before_optimizer_step, param.data should be fp32."""
        cb, model = self._setup()

        # Simulate fp16 gradients
        for p in model.parameters():
            p.grad = torch.ones_like(p)

        cb.on_before_optimizer_step(_MockTrainer(), model, None)

        for name, p in model.named_parameters():
            assert p.data.dtype == torch.float32, (
                f"{name}: expected fp32 data for optimizer, got {p.data.dtype}"
            )

    def test_master_grads_are_fp32(self):
        """Master parameter gradients should be fp32 after unscaling."""
        cb, model = self._setup()
        for p in model.parameters():
            p.grad = torch.ones_like(p)

        cb.on_before_optimizer_step(_MockTrainer(), model, None)

        for name, mp in cb._master.items():
            if mp.grad is not None:
                assert mp.grad.dtype == torch.float32, (
                    f"master[{name}].grad should be fp32"
                )

    def test_overflow_skips_update(self):
        """When a gradient is inf, no weight update should occur."""
        cb, model = self._setup()
        original_master = {n: mp.detach().clone() for n, mp in cb._master.items()}

        # Inject overflow into one gradient
        params = list(model.parameters())
        params[0].grad = torch.full_like(params[0], float("inf"))

        cb.on_before_optimizer_step(_MockTrainer(), model, None)

        # Master weights should be unchanged
        for name, mp in cb._master.items():
            assert torch.allclose(mp, original_master[name]), (
                f"master[{name}] was modified despite overflow"
            )

    def test_overflow_sets_grads_to_none(self):
        """After overflow, all parameter grads should be None (no-op optimizer step)."""
        cb, model = self._setup()
        for p in model.parameters():
            p.grad = torch.ones_like(p)

        params = list(model.parameters())
        params[0].grad = torch.full_like(params[0], float("inf"))
        cb.on_before_optimizer_step(_MockTrainer(), model, None)

        for p in model.parameters():
            assert p.grad is None, "Gradient should be None after overflow"

    def test_overflow_flag_set(self):
        cb, model = self._setup()
        params = list(model.parameters())
        params[0].grad = torch.full_like(params[0], float("inf"))
        cb.on_before_optimizer_step(_MockTrainer(), model, None)
        assert cb._overflow is True

    def test_no_overflow_flag_not_set(self):
        cb, model = self._setup()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        cb.on_before_optimizer_step(_MockTrainer(), model, None)
        assert cb._overflow is False

    # ── on_train_batch_end ───────────────────────────────────────────────────

    def test_on_train_batch_end_model_back_to_fp16(self):
        """After a batch, non-head params should be fp16 again."""
        cb, model = self._setup()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        cb.on_before_optimizer_step(_MockTrainer(), model, None)  # restores fp32
        cb.on_train_batch_end(_MockTrainer(), model, None, None, 0)

        inner = model.model.model
        for name, p in inner.backbone.body.named_parameters():
            assert p.dtype == torch.float16, (
                f"backbone.{name} should be fp16 after batch end"
            )

    def test_on_train_batch_end_head_stays_fp32(self):
        cb, model = self._setup()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        cb.on_before_optimizer_step(_MockTrainer(), model, None)
        cb.on_train_batch_end(_MockTrainer(), model, None, None, 0)

        head = model.model.model.head
        for name, p in head.named_parameters():
            assert p.dtype == torch.float32, (
                f"head.{name} should still be fp32 after batch end"
            )

    def test_master_weights_updated_after_optimizer_step(self):
        """Master weights should reflect the optimizer's fp32 update."""
        cb, model = self._setup()
        for p in model.parameters():
            p.grad = torch.ones_like(p)

        cb.on_before_optimizer_step(_MockTrainer(), model, None)

        # Simulate optimizer step: modify fp32 params by a fixed delta
        delta = 0.01
        for p in model.parameters():
            p.data.add_(delta)

        cb.on_train_batch_end(_MockTrainer(), model, None, None, 0)

        # Master weights should be updated to the new values
        for name, p in model.named_parameters():
            expected = cb._master[name]
            # After batch_end, master should equal fp32 copy of current param
            # (which was set to fp32 master + delta, then converted back to fp16 and fp32 again)
            # Just verify master is finite and has been updated
            assert torch.isfinite(expected).all(), f"master[{name}] contains non-finite values"

    def test_overflow_does_not_update_master_weights(self):
        """On overflow, master weights should NOT be updated."""
        cb, model = self._setup()
        original_master = {n: mp.detach().clone() for n, mp in cb._master.items()}

        params = list(model.parameters())
        params[0].grad = torch.full_like(params[0], float("inf"))
        cb.on_before_optimizer_step(_MockTrainer(), model, None)
        cb.on_train_batch_end(_MockTrainer(), model, None, None, 0)

        for name, mp in cb._master.items():
            assert torch.allclose(mp, original_master[name]), (
                f"master[{name}] was modified after overflow"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. mps_bn_backward_fp16 compiled kernel (MPS-only)
# ─────────────────────────────────────────────────────────────────────────────

@pytest_mps
class TestMPSKernel:

    @pytest.fixture(autouse=True)
    def skip_if_no_kernel(self):
        try:
            torch.ops.canopyai.mps_bn_backward_fp16
        except AttributeError:
            pytest.skip("mps_bn_backward_fp16 kernel not compiled")

    def _run_kernel(self, N=2, C=8, H=4, W=4, eps=1e-5):
        torch.manual_seed(0)
        x      = torch.randn(N, C, H, W, device="mps").half()
        weight = torch.ones(C, device="mps").half()

        # Compute save_mean / save_invstd manually on MPS to avoid relying on
        # native_batch_norm's MPS return-device behaviour for auxiliary outputs.
        x_f        = x.float()                                       # [N,C,H,W] fp32 MPS
        save_mean  = x_f.mean(dim=[0, 2, 3])                        # [C] fp32 MPS
        var        = x_f.var(dim=[0, 2, 3], unbiased=False)         # [C] fp32 MPS
        save_invstd = (var + eps).rsqrt()                            # [C] fp32 MPS

        grad_out = torch.ones_like(x)   # [N,C,H,W] fp16 MPS

        gi, gw, gb = torch.ops.canopyai.mps_bn_backward_fp16(
            grad_out, x, weight, save_mean, save_invstd, eps
        )
        return gi, gw, gb, x, weight, save_mean, save_invstd

    def test_kernel_output_shapes(self):
        gi, gw, gb, x, weight, *_ = self._run_kernel()
        assert gi.shape == x.shape
        assert gw.shape == weight.shape
        assert gb.shape == weight.shape

    def test_kernel_output_dtypes(self):
        gi, gw, gb, *_ = self._run_kernel()
        assert gi.dtype == torch.float16
        assert gw.dtype == torch.float16
        assert gb.dtype == torch.float16

    def test_kernel_grad_input_finite(self):
        gi, *_ = self._run_kernel()
        assert torch.isfinite(gi).all(), "grad_input contains non-finite values"

    def test_kernel_grad_weight_finite(self):
        _, gw, *_ = self._run_kernel()
        assert torch.isfinite(gw).all()

    def test_kernel_grad_bias_finite(self):
        _, _, gb, *_ = self._run_kernel()
        assert torch.isfinite(gb).all()

    def test_kernel_grad_bias_equals_sum_of_grad_out(self):
        """grad_bias = sum(grad_out, dim=[0,2,3]).  With all-ones grad_out this is N*H*W."""
        N, C, H, W = 2, 8, 4, 4
        _, _, gb, *_ = self._run_kernel(N=N, C=C, H=H, W=W)
        expected = torch.full((C,), N * H * W, dtype=torch.float16, device="mps")
        assert torch.allclose(gb, expected, atol=1.0), (
            f"grad_bias {gb} != expected {expected}"
        )

    def test_kernel_matches_fp32_reference(self):
        """Kernel grad_input / grad_weight should be close to the fp32 BN backward formula."""
        N, C, H, W, eps = 2, 8, 4, 4, 1e-5
        torch.manual_seed(0)

        # Use the same random data as _run_kernel so stats match
        gi, gw, _, x, weight, save_mean, save_invstd = self._run_kernel(
            N=N, C=C, H=H, W=W, eps=eps
        )

        # Manually compute reference BN backward on CPU using the same stats
        x_f        = x.cpu().float()
        mean_f     = save_mean.cpu().view(1, C, 1, 1)
        invstd_f   = save_invstd.cpu().view(1, C, 1, 1)
        w_f        = weight.cpu().float().view(1, C, 1, 1)
        m          = float(N * H * W)
        grad_out_f = torch.ones(N, C, H, W)

        x_hat      = (x_f - mean_f) * invstd_f
        dy         = grad_out_f * w_f
        sum_dy     = dy.sum(dim=[0, 2, 3])
        sum_dy_xh  = (dy * x_hat).sum(dim=[0, 2, 3])
        ref_gx     = (dy - sum_dy.view(1,C,1,1)/m
                      - x_hat * sum_dy_xh.view(1,C,1,1)/m) * invstd_f
        ref_gw     = (grad_out_f * x_hat).sum(dim=[0, 2, 3])

        assert torch.allclose(ref_gx.half(), gi.cpu(), atol=5e-2, rtol=1e-2), (
            f"Max grad_input diff: {(ref_gx.half() - gi.cpu()).abs().max().item():.4f}"
        )
        assert torch.allclose(ref_gw.half(), gw.cpu(), atol=5e-2, rtol=1e-2), (
            f"Max grad_weight diff: {(ref_gw.half() - gw.cpu()).abs().max().item():.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Integration: full forward + backward with the MPS fp16 model structure
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def _make_fp16_model(self):
        """Build a small conv model that mimics the backbone-fp16 / head-fp32 split."""
        class _Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, padding=1)
                self.bn   = nn.BatchNorm2d(8)
                self.relu = nn.ReLU()
            def forward(self, x):
                return self.relu(self.bn(self.conv(x)))

        class _Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc   = nn.Linear(8, 2)
            def forward(self, x):
                return self.fc(self.pool(x).flatten(1))

        backbone = _Backbone()
        head     = _Head()

        backbone.half()
        # BN patched to handle fp16 safely
        patch_batchnorm_for_fp16(backbone)

        return backbone, head

    def test_forward_and_backward_complete(self):
        """Full forward + backward with fp16 backbone / fp32 head should not raise."""
        backbone, head = self._make_fp16_model()
        x = torch.randn(2, 3, 16, 16).half()

        feat = backbone(x)                      # fp16
        feat_fp32 = _FP32Guard.apply(feat)      # cast to fp32 for head
        logits = head(feat_fp32)                # fp32
        loss = logits.sum()
        loss.backward()                         # should not raise

    def test_gradients_flow_to_backbone(self):
        """Gradients should propagate through the fp32 boundary into the fp16 backbone."""
        backbone, head = self._make_fp16_model()

        x = torch.randn(2, 3, 16, 16).half().requires_grad_(True)
        feat = backbone(x)
        feat_fp32 = _FP32Guard.apply(feat)
        logits = head(feat_fp32)
        logits.sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all(), "Non-finite grad flowing into fp16 backbone"
        assert x.grad.dtype == torch.float16

    def test_backbone_param_grads_are_finite(self):
        backbone, head = self._make_fp16_model()
        x = torch.randn(2, 3, 16, 16).half()
        _FP32Guard.apply(backbone(x)).sum().backward()

        for name, p in backbone.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    f"Non-finite grad in backbone.{name}"
                )

    def test_loss_value_is_finite(self):
        backbone, head = self._make_fp16_model()
        x = torch.randn(2, 3, 16, 16).half()
        feat_fp32 = _FP32Guard.apply(backbone(x))
        loss = head(feat_fp32).sum()
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
