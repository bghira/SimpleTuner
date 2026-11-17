"""Measured RTX 4090 schedules with runtime sequence-length fallbacks."""

from copy import deepcopy

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


class AdaptiveAttention:
    """Keep both sequence axes in dispatch when native programs are reused."""

    def __init__(self, fallback, tuned, short=None):
        self.fallback, self.tuned = fallback, tuned
        self.short = short

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        if cutlass.const_expr(self.short is not None):
            if q.shape[2] <= 128 and k.shape[2] <= 128:
                self.short(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
            elif q.shape[2] >= 1024 and k.shape[2] >= 1024:
                self.tuned(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
            else:
                self.fallback(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        elif q.shape[2] >= 1024 and k.shape[2] >= 1024:
            self.tuned(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        else:
            self.fallback(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)


class CompleteTiles:
    """Keep checked and complete tiles in the same native cache entry."""

    def __init__(self, checked, complete):
        self.checked, self.complete = checked, complete

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        if q.shape[2] % 64 == 0 and k.shape[2] % 64 == 0:
            self.complete(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        else:
            self.checked(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)


def complete_tiles(program):
    """Elide bounds only behind the runtime guard, preserving centering."""
    from .kernels import Attention

    full = deepcopy(program)
    for owner in (full.query, full.keyvalue):
        owner.full_visibility = True
        owner.full_copies = True
    full.query.warp_centering = True
    if full.query.center_loader is None:
        full.query.center_loader = Attention("forward", False, False, columns=16, warps=1, vector=full.query.vector)
    full.base_dot = False
    return CompleteTiles(program, full)


def schedule(program, mode, dimension, causal):
    """Select measured long schedules and full-key short forward staging."""
    tuned = deepcopy(program)
    short = None
    if mode == "forward":
        if dimension == 16:
            from .kernels import Attention

            tuned = Attention(
                "forward",
                causal,
                False,
                columns=64,
                width=32,
                warps=4,
                vector=program.vector,
                copy_cache_global=program.copy_cache_global,
                logical_width=16,
            )
        elif dimension != 512:
            tuned.forward_overlap = True
            tuned.wide_row_copies = True
            tuned.forward_register_q = causal and dimension in (64, 128)
        if dimension in (128, 256, 512):
            from .short_forward import ShortForward

            short = ShortForward(causal, dimension, rows=16, chunk=64, output_width=64, vector=program.vector)
    elif dimension == 16:
        from .kernels import Attention

        common = dict(columns=64, width=32, warps=4, centered=True, logical_width=16)
        tuned.query = Attention(
            "q",
            causal,
            False,
            **common,
            vector=program.query.vector,
            copy_cache_global=program.query.copy_cache_global,
            warp_centering=True,
        )
        tuned.keyvalue = Attention(
            "kv",
            causal,
            False,
            **common,
            vector=program.keyvalue.vector,
            copy_cache_global=program.keyvalue.copy_cache_global,
        )
        tuned.base_dot = False
    elif dimension == 64:
        if causal:
            tuned.query.backward_register_q = True
            tuned.keyvalue.kv_column_statistics = True
        else:
            tuned.keyvalue.stage_statistics = True
    elif dimension == 128:
        from .kernels import Attention

        short = deepcopy(program)
        common = dict(columns=32, width=128, warps=4, centered=True, narrow_n=True)
        short.query = Attention(
            "q",
            causal,
            False,
            **common,
            vector=program.query.vector,
            copy_cache_global=program.query.copy_cache_global,
            query_column_warps=4,
            warp_centering=True,
        )
        short.keyvalue = Attention(
            "kv",
            causal,
            False,
            **common,
            vector=program.keyvalue.vector,
            copy_cache_global=program.keyvalue.copy_cache_global,
            kv_column_warps=4,
        )
        short.base_dot = False
    if mode == "backward" and not causal:
        tuned = complete_tiles(tuned)
        if short is not None:
            short = complete_tiles(short)
    elif mode == "backward" and dimension == 128:
        tuned = complete_tiles(tuned)
        tuned.complete.query.packed_statistics = True
        tuned.complete.keyvalue.packed_statistics = True
        tuned.complete.keyvalue.kv_column_statistics = True
        tuned.complete.keyvalue.stage_statistics = False
    return AdaptiveAttention(program, tuned, short)
