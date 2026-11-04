"""Explicit A40 shared joint-backward schedules and native fallbacks."""

import math
from functools import lru_cache

import torch

from ._attention import Schedule
from ._host import native_runner


@lru_cache(maxsize=1)
def _metadata_predicate():
    """Cache the native function, while checking its tensor arguments each call."""
    from simpletuner.helpers.training.kohaku_fa_cute.native import extension

    return extension().a40_metadata_compatible


@lru_cache(maxsize=32)
def schedule(dimension, causal, sequence=4096):
    """Keep the wider dense key tile separate from causal work distribution."""
    if dimension not in (64, 128, 256):
        raise ValueError("The explicit A40 schedule requires D64, D128 or D256.")
    if dimension == 256:
        return Schedule(
            forward_m=64,
            forward_n=64,
            backward_m=64,
            backward_n=32,
            query_stages=1,
            dout_stages=1,
            score_m_warps=4,
            kv_m_warps=1,
            dq_m_warps=2,
            v_in_regs=True,
        )
    if dimension == 64:
        return Schedule(
            forward_m=64,
            backward_n=128,
            query_stages=2,
            dout_stages=1,
            score_m_warps=2,
            kv_m_warps=4,
            dq_m_warps=4,
            v_in_regs=True,
            causal_head_group=(16 if sequence == 1024 else 4) if causal else 1,
        )
    wide_dense = not causal
    narrow_query = wide_dense and dimension == 128
    return Schedule(
        forward_m=64,
        forward_n=64,
        forward_stages=1,
        query_in_registers=False,
        backward_m=32 if narrow_query else 64,
        backward_n=128 if wide_dense else 64,
        query_stages=1,
        dout_stages=1,
        threads=256,
        score_m_warps=2 if wide_dense else 4,
        kv_m_warps=2,
        dq_m_warps=2,
        v_in_regs=not narrow_query,
        causal_head_group=4 if causal else 1,
    )


def compatible(q, k, v, mask=None, dropout=0.0, causal=False, scale=None, gqa=False):
    """Bound the opt-in path to complete devices and measured shape families."""
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
    if compatible(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        return native_runner(schedule(q.shape[-1], is_causal, q.shape[2]))(q, k, v, scale=scale, causal=is_causal)
    from simpletuner.helpers.training.kohaku_fa_cute import api

    return api.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)
