"""Explicit full-device GB10 shared schedules with ordinary CuTe fallbacks."""

import math
from functools import lru_cache

import torch

from ._attention import Schedule
from ._host import native_runner


@lru_cache(maxsize=12)
def schedule(sequence, causal, heads=4, dtype=torch.bfloat16):
    """Preserve M64 short/causal work and use the measured wider dense tile."""
    if sequence not in (512, 1024, 4096) or (causal and sequence == 1024):
        raise ValueError("The explicit GB10 schedule requires a qualified sequence family.")
    wide_dense = not causal and ((sequence == 1024 and heads == 4) or (sequence == 4096 and dtype == torch.float16))
    shared_causal = causal and sequence == 4096 and heads == 4 and dtype == torch.float16
    wide_forward = not causal and heads == 16 and sequence == 4096
    return Schedule(
        forward_m=128 if wide_forward else 64,
        forward_n=32 if wide_forward else 64,
        backward_m=32 if wide_dense else 64,
        backward_n=128 if wide_dense else 64,
        query_stages=1 if wide_dense or shared_causal else 2,
        dout_stages=1,
        threads=256,
        score_m_warps=2 if wide_dense else 4,
        kv_m_warps=2,
        dq_m_warps=2,
        v_in_regs=not (wide_dense or shared_causal),
        causal_head_group=(
            4 if shared_causal else 16 if causal and sequence == 512 and heads == 16 and dtype == torch.bfloat16 else 1
        ),
    )


@lru_cache(maxsize=8)
def inference_schedule(sequence, causal):
    """Select only the measured H16 D128 forward-only geometries."""
    if sequence not in (128, 512, 1024, 4096):
        raise ValueError("The explicit GB10 inference schedule requires a qualified sequence family.")
    wide = not causal and sequence in (1024, 4096)
    return Schedule(forward_m=128 if wide else 64, forward_n=32 if wide else 64, query_stages=1, dout_stages=1)


@lru_cache(maxsize=6)
def _d256_inference_configuration(sequence, causal):
    """Describe the qualified D256 register partition and tile."""
    if sequence not in (512, 1024, 4096):
        raise ValueError("The D256 forward-only runner requires a qualified sequence.")
    wide = sequence == 4096 or (sequence == 1024 and not causal)
    selected = Schedule(forward_m=128 if wide else 64, forward_n=32 if wide else 64, query_stages=1, dout_stages=1)
    return selected, 256 if wide else 128


@lru_cache(maxsize=6)
def _long_d256_runner(sequence, causal):
    """Cache immutable configuration while checking live input metadata per call."""
    selected, threads = _d256_inference_configuration(sequence, causal)
    return native_runner(selected, forward_threads=threads)


@lru_cache(maxsize=2)
def _long_d64_runner(causal):
    """Cache the qualified S4096 tile and shared-Q pipeline, never operands."""
    selected = Schedule(forward_m=128, forward_n=64, forward_stages=1 if causal else 2, query_stages=1, dout_stages=1)
    return native_runner(selected)


@lru_cache(maxsize=8)
def _qualified_device(index):
    """CUDA device properties are immutable within the initialized process."""
    properties = torch.cuda.get_device_properties(index)
    return (
        (properties.major, properties.minor) == (12, 1)
        and properties.name == "NVIDIA GB10"
        and properties.multi_processor_count == 48
        and properties.shared_memory_per_block_optin == 101376
    )


def _compatible(q, k, v, mask, dropout, causal, scale, gqa, inference):
    if (
        torch.compiler.is_compiling()
        or mask is not None
        or dropout != 0
        or gqa
        or torch.are_deterministic_algorithms_enabled()
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
        or q.shape[0] != 1
        or q.shape[1] not in ((16,) if inference else (4, 16))
        or q.shape[2] not in ((128, 512, 1024, 4096) if inference else (512, 1024, 4096))
        or (inference and not causal and q.shape[2] == 512 and q.dtype == torch.bfloat16)
        or (not inference and causal and q.shape[2] == 1024)
        or q.shape[-1] != 128
        or not all(t.is_contiguous() for t in (q, k, v))
        or any(t.data_ptr() % 16 for t in (q, k, v))
    ):
        return False
    if not _qualified_device(q.device.index):
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 1.1754943508222875e-38 <= scale <= 2e38
    ):
        return False
    from . import available

    return available()


def compatible(q, k, v, mask=None, dropout=0.0, causal=False, scale=None, gqa=False):
    """Bound the training path to the complete GPU and measured D128 families."""
    return _compatible(q, k, v, mask, dropout, causal, scale, gqa, False)


@lru_cache(maxsize=1)
def _inference_metadata_predicate():
    """Reuse the native function while checking its tensor arguments every call."""
    from simpletuner.helpers.training.kohaku_fa_cute.native import extension

    return extension().gb10_inference_metadata_compatible


def inference_compatible(q, k, v, mask=None, dropout=0.0, causal=False, scale=None, gqa=False):
    """Use native metadata checks within the preceding forward-only boundary."""
    if (
        torch.compiler.is_compiling()
        or mask is not None
        or dropout != 0
        or gqa
        or torch.are_deterministic_algorithms_enabled()
    ):
        return False
    if getattr(q, "device", None) is None or q.device.type != "cuda":
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 1.1754943508222875e-38 <= scale <= 2e38
    ):
        return False
    from . import available

    return available() and _inference_metadata_predicate()(q, k, v, causal)


def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Run the explicit measured family and preserve the public fallback API."""
    if inference_compatible(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        shape = q.shape
        if shape[3] == 64 and shape[2] == 4096:
            selected = _long_d64_runner(is_causal)
        elif shape[3] == 256 and shape[2] != 128:
            selected = _long_d256_runner(shape[2], is_causal)
        else:
            selected = native_runner(inference_schedule(shape[2], is_causal))
        return selected.inference(q, k, v, scale=scale, causal=is_causal)
    if compatible(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        return native_runner(schedule(q.shape[2], is_causal, q.shape[1], q.dtype))(q, k, v, scale=scale, causal=is_causal)
    from simpletuner.helpers.training.kohaku_fa_cute import api

    return api.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)
