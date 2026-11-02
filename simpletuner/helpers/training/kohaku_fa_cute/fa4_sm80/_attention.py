"""Shared warp-MMA forward and joint backward with preserved operand rounding."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from ..determinism import check_atomic_backward
from ._private_api import make_private_api


@dataclass(frozen=True)
class Schedule:
    """Independent pipeline and MMA choices; eligibility is qualified separately."""

    forward_m: int = 128
    forward_n: int = 64
    forward_stages: int = 1
    query_in_registers: bool = False
    backward_m: int = 64
    backward_n: int = 128
    query_stages: int = 2
    dout_stages: int = 2
    threads: int = 256
    score_m_warps: int = 2
    kv_m_warps: int = 2
    dq_m_warps: int = 2
    v_in_regs: bool = False
    causal_head_group: int = 1

    def __post_init__(self):
        if self.causal_head_group not in (1, 2, 4, 8, 16):
            raise ValueError("Causal head groups must be one of 1, 2, 4, 8 or 16.")


@lru_cache(maxsize=8)
def _api(schedule, capacity):
    from ._forward import FlashAttentionForwardSm80
    from ._preprocess import FlashAttentionBackwardPreprocess

    class ConfiguredForward(FlashAttentionForwardSm80):
        def __init__(self, *args, **kwargs):
            kwargs.update(
                num_stages=schedule.forward_stages,
                Q_in_regs=schedule.query_in_registers,
                num_threads=128,
            )
            super().__init__(*args, **kwargs)
            self.shared_capacity = capacity

    return make_private_api(
        FlashAttentionForwardSm80=ConfiguredForward,
        FlashAttentionForwardSm120=ConfiguredForward,
        FlashAttentionBackwardPreprocess=FlashAttentionBackwardPreprocess,
    )


_main_cache = {}


def _backward(q, k, v, out, dout, maximum, logsum, scale, causal, schedule, capacity):
    """Bounded square MHA wrapper; include scratch initialization and conversion."""
    check_atomic_backward()
    import cutlass.cute as cute
    from flash_attn.cute import interface
    from flash_attn.cute.flash_bwd_preprocess import PackGQA
    from flash_attn.cute.utils import AuxData

    from ._backward import FlashAttentionBackwardSm80

    dout = dout.contiguous()
    if dout.data_ptr() % 16:
        dout = dout.clone()
    q, k, v, out, dout = [t.transpose(1, 2).detach() for t in (q, k, v, out, dout)]
    batch, sequence, heads, dimension = q.shape
    rounded = (sequence + schedule.backward_m - 1) // schedule.backward_m * schedule.backward_m
    dtype = interface.torch2cute_dtype_map[q.dtype]
    major, minor = torch.cuda.get_device_capability(q.device)
    architecture = major * 10 + minor
    gradients = tuple(torch.empty_like(t) for t in (q, k, v))
    dq, dk, dv = gradients
    scratch = torch.empty(batch, heads, rounded * dimension, device=q.device, dtype=torch.float32)
    delta = torch.empty(batch, heads, rounded, device=q.device, dtype=torch.float32)
    lognorm = torch.empty_like(delta)
    api = _api(schedule, capacity)
    api._bwd_preprocess(
        out,
        dout,
        delta,
        logsum,
        lognorm,
        scratch,
        None,
        None,
        None,
        dtype,
        dimension,
        dimension,
        schedule.backward_m,
        pack_gqa=PackGQA.OFF,
        fake_mode=False,
    )
    key = (q.device.index, tuple(q.shape), tuple(q.stride()), tuple(dout.stride()), q.dtype, causal, schedule, capacity)
    if key not in _main_cache:
        if len(_main_cache) >= 128:
            _main_cache.clear()
        program = FlashAttentionBackwardSm80(
            dtype,
            dimension,
            dimension,
            1,
            schedule.backward_m,
            schedule.backward_n,
            schedule.query_stages,
            schedule.dout_stages,
            schedule.threads,
            False,
            causal,
            False,
            False,
            False,
            False,
            schedule.score_m_warps,
            schedule.kv_m_warps,
            schedule.dq_m_warps,
            V_in_regs=schedule.v_in_regs,
        )
        program.shared_capacity = capacity
        program.causal_head_group = schedule.causal_head_group
        tensors = [interface.to_cute_tensor(t) for t in (q, k, v, dout, lognorm, delta, scratch, dk, dv)]
        auxiliary = AuxData([interface.to_cute_aux_tensor(maximum)], None)
        _main_cache[key] = cute.compile(
            program,
            *tensors,
            scale,
            *([None] * 9),
            auxiliary,
            None,
            None,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _main_cache[key](
        q, k, v, dout, lognorm, delta, scratch, dk, dv, scale, *([None] * 9), AuxData([maximum], None), None, None
    )
    api._bwd_postprocess_convert(
        scratch,
        dq,
        scale,
        None,
        None,
        architecture,
        dtype,
        dimension,
        schedule.backward_m,
        schedule.threads,
        schedule.dq_m_warps,
        False,
        accum_order=None,
        fake_mode=False,
    )
    return tuple(g.transpose(1, 2) for g in gradients)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale, schedule):
        capacity = torch.cuda.get_device_properties(q.device).shared_memory_per_block_optin
        maximum = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        output = torch.empty_like(q)
        _, logsum, *_ = _api(schedule, capacity)._flash_attn_fwd(
            *(t.transpose(1, 2) for t in (q, k, v)),
            softmax_scale=scale,
            causal=causal,
            tile_mn=(schedule.forward_m, schedule.forward_n),
            pack_gqa=False,
            out=output.transpose(1, 2),
            return_lse=True,
            aux_tensors=[maximum],
        )
        ctx.save_for_backward(q, k, v, output, maximum, logsum)
        ctx.causal, ctx.scale, ctx.schedule, ctx.capacity = causal, scale, schedule, capacity
        return output

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, maximum, logsum = ctx.saved_tensors
        gradients = _backward(q, k, v, out, dout, maximum, logsum, ctx.scale, ctx.causal, ctx.schedule, ctx.capacity)
        return *gradients, None, None, None


def attention(q, k, v, *, causal=False, scale=None, schedule=Schedule()):
    return _Attention.apply(q, k, v, causal, q.shape[-1] ** -0.5 if scale is None else scale, schedule)
