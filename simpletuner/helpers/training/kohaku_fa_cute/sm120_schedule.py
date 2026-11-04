"""Qualified RTX 5090 outlier schedules with runtime length fallbacks."""

import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda


def eligible(device, mode, masked, layouts, configuration, kv_configuration, vector):
    """Keep the selected layout family separate from the RTX 4090 policies."""
    if (
        masked
        or vector != 8
        or layouts[0][0] not in (torch.float16, torch.bfloat16)
        or layouts[0][1][:2] != (1, 16)
        or layouts[1][1][1] != 16
        or any(stride >= 0 for i in (0, 1, 2, 7) for stride in layouts[i][2][:-1])
        or torch.cuda.get_device_capability(device) != (12, 0)
        or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5090"
    ):
        return False
    dimension = layouts[0][1][-1]
    if dimension == 16:
        return (
            mode == "forward"
            and configuration == (64, 32, 4, False, False, False, False)
            or mode == "backward"
            and configuration == (32, 32, 4, False, True, False, False)
            and kv_configuration == configuration
        )
    if dimension == 64:
        return (
            mode == "backward"
            and configuration == (32, 64, 4, False, True, True, True)
            and kv_configuration == (64, 64, 4, False, True, True, True)
        )
    if mode == "backward" and dimension in (128, 256):
        return configuration == (
            (32, 128, 4, False, True, False, False) if dimension == 128 else (64, 256, 8, False, True, False, False)
        ) and kv_configuration == (32, dimension, 4, False, True, False, False)
    return mode == "forward" and configuration == {
        128: (32, 128, 4, False, False, True, True),
        256: (16, 256, 4, False, False, True, False),
        512: (32, 512, 4, False, False, False, False),
    }.get(dimension)


class ShortOnly:
    """Select full-key short staging without changing any longer schedule."""

    def __init__(self, fallback, dimension, causal, vector):
        from .short_forward import ShortForward

        self.fallback = fallback
        self.short = ShortForward(causal, dimension, rows=16, chunk=64, output_width=64, vector=vector)

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        if q.shape[2] <= 128 and k.shape[2] <= 128:
            self.short(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        else:
            self.fallback(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)


class ShortBackwardOnly:
    """Keep compact square backward separate from rectangular and long policies."""

    def __init__(self, fallback, dimension, causal):
        from copy import deepcopy

        from .ada import complete_tiles
        from .kernels import Attention

        self.fallback = fallback
        short = deepcopy(fallback)
        common = dict(columns=32, width=dimension, warps=4, centered=True, narrow_n=True)
        short.query = Attention(
            "q",
            causal,
            False,
            **common,
            vector=fallback.query.vector,
            copy_cache_global=fallback.query.copy_cache_global,
            reload_center=fallback.query.reload_center,
            query_column_warps=4,
            warp_centering=True,
        )
        fallback_kv = getattr(fallback.keyvalue, "fallback", fallback.keyvalue)
        short.keyvalue = Attention(
            "kv",
            causal,
            False,
            **common,
            vector=fallback.query.vector,
            copy_cache_global=fallback_kv.copy_cache_global,
            kv_column_warps=4,
        )
        short.base_dot = False
        self.short = short if causal else complete_tiles(short)

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        if q.shape[2] == k.shape[2] and q.shape[2] <= 128:
            self.short(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        else:
            self.fallback(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)


def schedule(program, mode, dimension, causal, vector):
    """Select measured square-short owners and the logical D16 bundle."""
    if dimension == 16:
        from .ada import CompleteTiles
        from .ada import schedule as d16_schedule

        result = d16_schedule(program, mode, dimension, causal)
        if mode == "backward":
            policies = (
                (result.tuned.checked, result.tuned.complete) if isinstance(result.tuned, CompleteTiles) else (result.tuned,)
            )
            for policy in policies:
                policy.query.reload_center = program.query.reload_center
        return result
    if dimension == 64:
        from .sm120_backward import D64Square

        program.keyvalue = D64Square(program.keyvalue)
        return program
    if mode == "backward":
        return ShortBackwardOnly(program, dimension, causal)
    return ShortOnly(program, dimension, causal, vector)
