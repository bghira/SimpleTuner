"""Qualified RTX 3090 Ti schedules with runtime length fallbacks."""

from copy import deepcopy

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch


class ShortBackwardOnly:
    """Preserve the original long and rectangular backward programs."""

    def __init__(self, fallback, compact):
        self.fallback, self.compact = fallback, compact

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        if q.shape[2] == k.shape[2] and q.shape[2] <= 128:
            self.compact(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        else:
            self.fallback(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)


class LongSquareBackward:
    """Keep the original query owner below the measured long-query regime."""

    def __init__(self, fallback, long):
        self.fallback, self.long = fallback, long

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        if q.shape[2] == k.shape[2] and q.shape[2] >= 4096:
            self.long(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        else:
            self.fallback(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)


class D64LongSquare:
    """Use the fixed N2 owner only within the qualified long-square range."""

    def __init__(self, fallback):
        # Reuse the warp-MMA owner with independently qualified Ti bounds.
        from .sm120_backward import D64Values

        self.fallback = fallback
        self.tuned = D64Values(fallback)

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        if q.shape[2] == k.shape[2] and q.shape[2] >= 1024 and q.shape[2] <= 4096:
            self.tuned(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        else:
            self.fallback(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)


def eligible(device, mode, masked, layouts, configuration, kv_configuration, vector):
    dimension = layouts[0][1][-1]
    if (
        masked
        or vector != 8
        or layouts[0][0] not in (torch.float16, torch.bfloat16)
        or layouts[0][1][:2] != (1, 16)
        or layouts[1][1][1] != 16
        or any(stride >= 0 for i in (0, 1, 2, 7) for stride in layouts[i][2][:-1])
        or torch.cuda.get_device_capability(device) != (8, 6)
        or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 3090 Ti"
        or torch.cuda.get_device_properties(device).multi_processor_count != 84
    ):
        return False
    if dimension == 64 and layouts[0][0] != torch.bfloat16:
        return False
    if mode == "forward":
        return configuration == {
            16: (64, 32, 4, False, False, False, False),
            128: (64, 128, 4, False, False, False, False),
            256: (32, 256, 8, False, False, False, False),
            512: (32, 512, 4, False, False, False, False),
        }.get(dimension)
    if mode == "backward" and dimension == 256:
        return configuration == (64, 256, 8, False, True, False, False) and kv_configuration == (
            32,
            256,
            4,
            False,
            True,
            False,
            False,
        )
    return (
        mode == "backward"
        and dimension in (16, 64, 128)
        and configuration == (32, 32 if dimension == 16 else dimension, 4, False, True, False, False)
        and kv_configuration == configuration
    )


def schedule(program, mode, dimension, causal):
    if dimension == 16:
        from .ada import schedule as ada_schedule

        return ada_schedule(program, mode, dimension, causal)
    if mode == "forward":
        from .ada import AdaptiveAttention
        from .short_forward import ShortForward

        short = ShortForward(causal, dimension, rows=16, chunk=64, output_width=64, vector=program.vector)
        return AdaptiveAttention(program, program, short)
    if dimension == 64:
        program.keyvalue = D64LongSquare(program.keyvalue)
        return program
    if dimension == 256:
        from .sm86_backward import QueryMajorQueries

        long = deepcopy(program)
        long.query = QueryMajorQueries(causal, program.query.vector)
        return LongSquareBackward(program, long)
    if dimension == 128:
        from .ada import complete_tiles
        from .kernels import Attention

        compact = deepcopy(program)
        common = dict(columns=32, width=128, warps=4, centered=True, narrow_n=True)
        compact.query = Attention(
            "q",
            causal,
            False,
            **common,
            vector=program.query.vector,
            copy_cache_global=program.query.copy_cache_global,
            query_column_warps=4,
            warp_centering=True,
        )
        compact.keyvalue = Attention(
            "kv",
            causal,
            False,
            **common,
            vector=program.query.vector,
            copy_cache_global=program.query.copy_cache_global,
            kv_column_warps=4,
        )
        compact.base_dot = False
        if not causal:
            compact = complete_tiles(compact)
        return ShortBackwardOnly(program, compact)
    return program
