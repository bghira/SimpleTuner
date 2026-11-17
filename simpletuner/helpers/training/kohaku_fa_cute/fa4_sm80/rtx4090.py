"""Explicit RTX 4090 shared joint-backward schedules and native fallbacks."""

import math
from functools import lru_cache

import torch

from ._attention import Schedule
from ._host import native_runner


@lru_cache(maxsize=8)
def _short_device(index):
    properties = torch.cuda.get_device_properties(index)
    return (
        torch.cuda.get_device_capability(index) == (8, 9)
        and properties.name == "NVIDIA GeForce RTX 4090"
        and properties.multi_processor_count == 128
        and properties.shared_memory_per_block_optin == 101376
    )


def _short_compatible(q, k, v, mask, dropout, causal, scale, gqa):
    if (
        torch.compiler.is_compiling()
        or not torch.is_grad_enabled()
        or mask is not None
        or dropout != 0
        or gqa
        or torch.are_deterministic_algorithms_enabled()
        or getattr(q, "ndim", None) != 4
        or q.shape != (1, 16, 128, 16)
        or q.shape != k.shape
        or q.shape != v.shape
        or q.device.type != "cuda"
        or q.device != k.device
        or q.device != v.device
        or q.device.index != torch.cuda.current_device()
        or q.dtype not in (torch.float16, torch.bfloat16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or not any(t.requires_grad for t in (q, k, v))
        or not all(t.is_contiguous() for t in (q, k, v))
        or any(t.data_ptr() % 16 for t in (q, k, v))
        or not _short_device(q.device.index)
    ):
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 2**-126 <= scale <= 2e38
    ):
        return False
    from . import available

    return available()


@lru_cache(maxsize=1)
def _short_runner():
    from simpletuner.helpers.training.kohaku_fa_cute.native import extension

    return extension().AttentionRunner("short-optimized", {})


def _short_flags(dtype, causal):
    """Use independently measured layout/statistics choices per input family."""
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Short attention requires BF16 or FP16 inputs.")
    return causal or dtype == torch.float16, not causal


@lru_cache(maxsize=1)
def _metadata_predicate():
    """Cache the function handle, while checking live tensors every call."""
    from simpletuner.helpers.training.kohaku_fa_cute.native import extension

    return extension().rtx4090_metadata_compatible


@lru_cache(maxsize=12)
def schedule(dimension, causal, sequence=1024):
    """Keep the wider dense key tile separate from causal work distribution."""
    if dimension not in (64, 128):
        raise ValueError("The explicit RTX 4090 schedule requires D64 or D128.")
    wide_dense = not causal
    narrow_query = wide_dense and dimension == 128
    wide_d64 = wide_dense and dimension == 64
    long_causal_d64 = causal and dimension == 64 and sequence == 4096
    wide_backward = wide_dense or long_causal_d64
    return Schedule(
        forward_m=128 if dimension == 64 and not causal else 64,
        forward_n=64,
        forward_stages=1,
        query_in_registers=False,
        backward_m=32 if narrow_query else 64,
        backward_n=128 if wide_backward else 64,
        query_stages=1,
        dout_stages=1,
        threads=256,
        score_m_warps=2 if wide_backward else 4,
        kv_m_warps=4 if wide_d64 or long_causal_d64 else 2,
        dq_m_warps=4 if wide_d64 or long_causal_d64 else 2,
        v_in_regs=not narrow_query,
        causal_head_group=16 if causal else 1,
    )


def compatible(q, k, v, mask=None, dropout=0.0, causal=False, scale=None, gqa=False):
    """Keep policy guards live and combine mutable tensor metadata checks."""
    if (
        torch.compiler.is_compiling()
        or mask is not None
        or dropout != 0
        or gqa
        or torch.are_deterministic_algorithms_enabled()
    ):
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 1.1754943508222875e-38 <= scale <= 2e38
    ):
        return False
    from . import available

    return available() and _metadata_predicate()(q, k, v, causal)


def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Use the explicit measured family; retain ordinary CuTe for other calls."""
    if getattr(q, "ndim", None) == 4 and q.shape[-1] == 16:
        if _short_compatible(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
            return _short_runner()(q, k, v, scale=scale, causal=is_causal)
    if compatible(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        return native_runner(schedule(q.shape[-1], is_causal, q.shape[-2]))(q, k, v, scale=scale, causal=is_causal)
    from simpletuner.helpers.training.kohaku_fa_cute import api

    return api.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)


@lru_cache(maxsize=1)
def _automatic_metadata_predicate():
    """Cache the native function handle; its tensor checks remain live."""
    from simpletuner.helpers.training.kohaku_fa_cute.native import extension

    return extension().rtx4090_automatic_metadata_compatible


def _automatic_cold_compatible(q, k, v):
    if (
        not torch.is_grad_enabled()
        or not any(getattr(t, "requires_grad", False) for t in (q, k, v))
        or getattr(q, "ndim", None) != 4
        or q.shape != k.shape
        or q.shape != v.shape
        or q.device.type != "cuda"
        or q.device != k.device
        or q.device != v.device
        or q.device.index != torch.cuda.current_device()
        or q.dtype not in (torch.float16, torch.bfloat16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.shape[:2] != (1, 16)
        or not (q.shape[2:] == (128, 16) or (q.shape[2] == 4096 and q.shape[3] in (64, 128)))
        or not all(t.is_contiguous() and t.data_ptr() % 16 == 0 and not t.is_neg() and not t.is_conj() for t in (q, k, v))
        or not _short_device(q.device.index)
    ):
        return False
    return True


def automatic_compatible(q, k, v, mask, dropout, causal, scale, gqa):
    """Select the measured training families without repeating Python metadata checks."""
    if torch.compiler.is_compiling() or torch.are_deterministic_algorithms_enabled() or mask is not None or dropout != 0:
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 2**-126 <= scale <= 2e38
    ):
        return False
    from . import available

    if not available():
        return False
    if _automatic_metadata_predicate.cache_info().currsize == 0 and not _automatic_cold_compatible(q, k, v):
        return False
    # Equal-shape MHA is valid with either value of enable_gqa.
    return _automatic_metadata_predicate()(q, k, v, causal)


def attention(q, k, v, *, causal=False, scale=None):
    """Execute an already qualified automatic selection."""
    if q.shape[-1] == 16:
        return _short_runner()(q, k, v, scale=scale, causal=causal)
    return native_runner(schedule(q.shape[-1], causal, q.shape[-2]))(q, k, v, scale=scale, causal=causal)


@lru_cache(maxsize=1)
def _s1024_guarded_call():
    from simpletuner.helpers.training.kohaku_fa_cute.native import extension

    return extension().rtx4090_guarded_call


@lru_cache(maxsize=4)
def _s1024_runner(dimension, causal):
    return native_runner(schedule(dimension, causal, 1024))


def _s1024_cold_compatible(q, k, v):
    return (
        any(getattr(t, "requires_grad", False) for t in (q, k, v))
        and q.shape == k.shape == v.shape
        and q.shape[:2] == (1, 16)
        and q.device == k.device == v.device
        and q.device.index == torch.cuda.current_device()
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.dtype == k.dtype == v.dtype
        and all(t.is_contiguous() and t.data_ptr() % 16 == 0 and not t.is_neg() and not t.is_conj() for t in (q, k, v))
    )


def try_s1024_attention(q, k, v, mask, dropout, causal, scale, gqa):
    """Check live policies and execute the S1024 native guard in one call."""
    if (
        torch.compiler.is_compiling()
        or not torch.is_grad_enabled()
        or torch.are_deterministic_algorithms_enabled()
        or mask is not None
        or dropout != 0
    ):
        return None
    if scale is not None and (
        not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 2**-126 <= scale <= 2e38
    ):
        return None
    from . import available

    if not available() or not _short_device(q.device.index):
        return None
    if _s1024_guarded_call.cache_info().currsize == 0 and not _s1024_cold_compatible(q, k, v):
        return None
    dimension = q.shape[-1]
    return _s1024_guarded_call()(_s1024_runner(dimension, causal), q, k, v, dimension, scale, causal)
