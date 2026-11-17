"""Bounded opt-in attention for a full ordinary RTX 3090."""

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=8)
def _device_matches(index):
    properties = torch.cuda.get_device_properties(index)
    return (
        (properties.major, properties.minor) == (8, 6)
        and properties.multi_processor_count == 82
        and properties.name == "NVIDIA GeForce RTX 3090"
    )


def _eligible(q, k, v, mask, dropout, causal, scale, gqa):
    if (
        torch.compiler.is_compiling()
        or mask is not None
        or dropout != 0
        or torch.are_deterministic_algorithms_enabled()
        or getattr(q, "ndim", None) != 4
        or q.shape != k.shape
        or q.shape != v.shape
        or q.device.type != "cuda"
        or q.device != k.device
        or q.device != v.device
        or q.device.index != torch.cuda.current_device()
        or q.shape[0] != 1
        or q.shape[1] != 16
        or not (
            (q.shape[2] == 128 and q.shape[3] in (16, 64, 128)) or (1024 <= q.shape[2] <= 4096 and q.shape[3] in (64, 128))
        )
        or q.dtype not in (torch.bfloat16, torch.float16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or any(not t.is_contiguous() or t.data_ptr() % 16 for t in (q, k, v))
        or not _device_matches(q.device.index)
    ):
        return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 2**-126 <= scale <= 2e38
    ):
        return False
    from .fa4_sm80 import available

    return available()


def _head_group(sequence, causal):
    if not causal:
        return 1
    return 16 if sequence == 1024 else 4


@lru_cache(maxsize=8)
def _runner(dimension, causal, head_group=1):
    from .fa4_sm80._attention import Schedule
    from .fa4_sm80._host import native_runner

    wide = dimension == 64 and not causal
    return native_runner(
        Schedule(
            forward_m=64,
            backward_m=64,
            backward_n=128 if wide else 64,
            query_stages=1,
            dout_stages=1,
            score_m_warps=2 if wide else 4,
            kv_m_warps=4 if wide else 2,
            dq_m_warps=4 if wide else 2,
            v_in_regs=True,
            causal_head_group=head_group,
        )
    )


@lru_cache(maxsize=1)
def _short_runner():
    from .native import extension

    return extension().AttentionRunner("short", {})


@lru_cache(maxsize=1)
def _short_key32_runner():
    from .native import extension

    return extension().AttentionRunner("short_key32", {})


@lru_cache(maxsize=1)
def _short_d128_runner():
    from .native import extension

    return extension().AttentionRunner("short_d128", {})


@lru_cache(maxsize=1)
def _short_d128_stats_runner():
    from .native import extension

    return extension().AttentionRunner("short_d128_stats", {})


def rtx3090_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Use qualified RTX 3090 schedules; other calls retain ordinary CuTe dispatch.

    The opt-in uses the package's normal rounded arithmetic. It requires the
    pinned optional warp-kernel runtime and does not change default dispatch.
    """
    if _eligible(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        if query.shape[-1] == 16:
            return _short_runner()(query, key, value, scale=scale, causal=is_causal)
        if query.shape[-2:] == (128, 64):
            return _short_key32_runner()(query, key, value, scale=scale, causal=is_causal)
        if query.shape[-2:] == (128, 128):
            runner = _short_d128_stats_runner() if is_causal else _short_d128_runner()
            return runner(query, key, value, scale=scale, causal=is_causal)
        return _runner(query.shape[-1], is_causal, _head_group(query.shape[-2], is_causal))(
            query, key, value, scale=scale, causal=is_causal
        )
    from .api import scaled_dot_product_attention

    return scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
