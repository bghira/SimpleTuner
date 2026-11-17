"""Bounded RTX 5090 training selection for the qualified warp schedule."""

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def target_device(device):
    if torch.cuda.get_device_capability(device) != (12, 0):
        return False
    properties = torch.cuda.get_device_properties(device)
    return (
        properties.multi_processor_count == 170
        and "5090" in properties.name
        and properties.shared_memory_per_block_optin == 101376
    )


@lru_cache(maxsize=1)
def _metadata_predicate():
    from ..native import extension

    return extension().sm120_metadata_compatible


def compatible(q, k, v, mask, dropout, causal, scale, gqa):
    if isinstance(q, torch.Tensor):
        if (
            torch.compiler.is_compiling()
            or torch.are_deterministic_algorithms_enabled()
            or mask is not None
            or dropout != 0
            or q.device.type != "cuda"
            or not target_device(q.device)
            or (
                scale is not None
                and (
                    not isinstance(scale, (float, int))
                    or not math.isfinite(scale)
                    or not 1.1754943508222875e-38 <= scale <= 2e38
                )
            )
        ):
            return False
        return _metadata_predicate()(q, k, v, causal)
    if (
        torch.compiler.is_compiling()
        or torch.are_deterministic_algorithms_enabled()
        or getattr(q, "ndim", None) != 4
        or q.device.type != "cuda"
        or not torch.is_grad_enabled()
        or not all(getattr(t, "requires_grad", False) for t in (q, k, v))
        or not target_device(q.device)
        or mask is not None
        or dropout != 0
        or q.shape != k.shape
        or q.shape != v.shape
        or q.shape[0] != 1
        or q.shape[1] != 16
        or not (
            q.shape[2] == 4096
            or (q.shape[2] == 1024 and (q.shape[3] == 256 or (causal and (q.shape[3] == 64 or q.dtype == torch.bfloat16))))
        )
        or q.shape[3] not in (64, 128, 256)
        or q.device != k.device
        or q.device != v.device
        or q.device.index != torch.cuda.current_device()
        or q.dtype not in (torch.bfloat16, torch.float16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (q, k, v))
    ):
        return False
    return scale is None or (
        isinstance(scale, (float, int)) and math.isfinite(scale) and 1.1754943508222875e-38 <= scale <= 2e38
    )


@lru_cache(maxsize=4)
def schedule(dimension, causal=False):
    from ._attention import Schedule

    if dimension not in (64, 128):
        raise ValueError("The selected SM120 schedule requires D64 or D128.")
    return Schedule(
        forward_m=128 if dimension == 64 else 64,
        forward_n=64,
        forward_stages=1,
        query_in_registers=False,
        backward_m=64,
        backward_n=64,
        query_stages=2,
        dout_stages=1,
        score_m_warps=4,
        kv_m_warps=2,
        dq_m_warps=2,
        v_in_regs=False,
        causal_head_group=16 if causal else 1,
    )


@lru_cache(maxsize=4)
def d256_schedule(causal, sequence):
    from ._attention import Schedule

    if sequence not in (1024, 4096):
        raise ValueError("The selected D256 schedule requires S1024 or S4096.")
    short_dense = sequence == 1024 and not causal
    return Schedule(
        forward_m=128 if short_dense else 64,
        forward_n=32 if short_dense else 64,
        forward_stages=1,
        query_in_registers=False,
        backward_m=64,
        backward_n=32,
        query_stages=1,
        dout_stages=1,
        threads=256,
        score_m_warps=4,
        kv_m_warps=1,
        dq_m_warps=2,
        v_in_regs=True,
        causal_head_group=16 if causal and sequence == 1024 else 1,
    )


@lru_cache(maxsize=4)
def d256_runner(causal, sequence):
    from dataclasses import asdict

    from ..native import extension

    options = asdict(d256_schedule(causal, sequence))
    options["forward_threads"] = 256 if sequence == 1024 and not causal else 128
    return extension().AttentionRunner("warp", options)
