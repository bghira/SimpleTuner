"""L40S schedules with runtime sequence-length fallbacks."""

from copy import deepcopy

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch


class ShortForwardOnly:
    def __init__(self, fallback, compact):
        self.fallback, self.compact = fallback, compact

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        if q.shape[2] <= 128 and k.shape[2] <= 128:
            self.compact(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        else:
            self.fallback(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)


class ShortBackwardOnly:
    def __init__(self, fallback, compact):
        self.fallback, self.compact = fallback, compact

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        # Rectangles retain their separately qualified value-gradient policy.
        if q.shape[2] == k.shape[2] and q.shape[2] <= 128:
            self.compact(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        else:
            self.fallback(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)


class CompleteForwardTiles:
    """Keep M128 forward staging behind the measured complete-tile guard."""

    def __init__(self, fallback, tuned):
        self.fallback, self.tuned = fallback, tuned

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        if q.shape[2] % 128 == 0 and k.shape[2] % 128 == 0:
            self.tuned(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        else:
            self.fallback(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)


def eligible(device, mode, masked, layouts, configuration, kv_configuration, vector, *, causal=False):
    dimension = layouts[0][1][-1]
    if (
        masked
        or vector != 8
        or layouts[0][0] not in (torch.float16, torch.bfloat16)
        or layouts[0][1][:2] != (1, 16)
        or layouts[1][1][1] != 16
        or dimension not in (16, 64, 128, 256, 512)
        or any(stride >= 0 for i in (0, 1, 2, 7) for stride in layouts[i][2][:-1])
        or torch.cuda.get_device_capability(device) != (8, 9)
        or torch.cuda.get_device_name(device) != "NVIDIA L40S"
    ):
        return False
    if mode == "forward":
        return (
            configuration
            == {
                16: (64, 32, 4, False, False, False, False),
                64: (64, 64, 4, False, False, False, False),
                128: (64, 128, 4, False, False, False, False),
                256: (32, 256, 8, False, False, False, False),
                512: (32, 512, 4, False, False, False, False),
            }[dimension]
        )
    return (
        mode == "backward"
        and dimension in (16, 128)
        and configuration == (32, 32 if dimension == 16 else dimension, 4, False, True, False, False)
        and kv_configuration == configuration
    )


def schedule(program, mode, dimension, causal):
    from .ada import AdaptiveAttention, complete_tiles
    from .ada import schedule as ada_schedule
    from .kernels import Attention
    from .short_forward import ShortForward

    if dimension == 16:
        return ada_schedule(program, mode, dimension, causal)
    if mode == "forward":
        short = (
            ShortForward(causal, dimension, rows=16, chunk=64, output_width=64, vector=program.vector)
            if dimension in (128, 256, 512)
            else None
        )
        if dimension in (64, 128):
            tuned = Attention(
                "forward",
                causal,
                False,
                columns=128,
                width=dimension,
                warps=8,
                vector=program.vector,
                copy_cache_global=program.copy_cache_global,
                forward_overlap=True,
                forward_register_q=causal or dimension == 128,
            )
            return AdaptiveAttention(program, CompleteForwardTiles(program, tuned), short)
        return ShortForwardOnly(program, short)
    if dimension == 128:
        compact = deepcopy(program)
        keyvalue = getattr(program.keyvalue, "fallback", program.keyvalue)
        common = dict(columns=32, width=128, warps=4, centered=True, narrow_n=True)
        compact.query = Attention(
            "q",
            causal,
            False,
            **common,
            vector=program.query.vector,
            copy_cache_global=program.query.copy_cache_global,
            reload_center=program.query.reload_center,
            query_column_warps=4,
            warp_centering=True,
        )
        compact.keyvalue = Attention(
            "kv",
            causal,
            False,
            **common,
            vector=program.query.vector,
            copy_cache_global=keyvalue.copy_cache_global,
            reload_center=keyvalue.reload_center,
            kv_column_warps=4,
        )
        compact.base_dot = False
        if not causal:
            compact = complete_tiles(compact)
        return ShortBackwardOnly(program, compact)
    return program
