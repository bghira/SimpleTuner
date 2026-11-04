"""Measured A100 policies; explicit experimental Schedule defaults stay stable."""

from dataclasses import replace
from functools import lru_cache

from ._attention import Schedule


@lru_cache(maxsize=32)
def a100_schedule(dimension, sequence, causal, heads=16):
    schedule = Schedule()
    if dimension == 256:
        if sequence == 4096 and heads in (8, 16):
            return replace(
                schedule,
                forward_m=64,
                backward_n=64,
                query_stages=1,
                dout_stages=1,
                score_m_warps=4,
                kv_m_warps=1,
                dq_m_warps=2,
                v_in_regs=False,
                causal_head_group=4 if causal else 1,
            )
        return replace(
            schedule,
            forward_m=64,
            backward_n=32,
            query_stages=1,
            dout_stages=1,
            score_m_warps=4,
            kv_m_warps=1,
            dq_m_warps=2,
            v_in_regs=True,
        )
    if dimension == 64 and sequence >= 2048:
        schedule = replace(
            schedule,
            backward_m=128,
            query_stages=1,
            dout_stages=1,
            score_m_warps=4,
            kv_m_warps=4,
            dq_m_warps=4,
        )
    if dimension == 128 and causal:
        schedule = replace(schedule, forward_m=64)
    if dimension == 128 and causal and sequence == 2048 and heads == 8:
        schedule = replace(schedule, causal_head_group=4)
    if causal and sequence == 4096 and heads in (8, 16):
        schedule = replace(schedule, causal_head_group=4 if heads == 8 else 16)
    return schedule
