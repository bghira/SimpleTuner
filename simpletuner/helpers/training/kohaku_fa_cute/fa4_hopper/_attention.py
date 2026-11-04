"""Owned Hopper forward and single-rounded joint WGMMA backward."""

from dataclasses import replace
from functools import lru_cache

import torch

from ..determinism import check_atomic_backward
from ._private_api import make_private_api


@lru_cache(maxsize=2)
def _api(persistent=False):
    from flash_attn.cute import interface

    from ._backward import FlashAttentionBackwardSm90
    from ._forward import FlashAttentionForwardSm90
    from ._persistent import PersistentFlashAttentionForwardSm90
    from ._preprocess import FlashAttentionBackwardPreprocess

    selector = interface._tile_size_bwd_sm90

    def select(head_dim, head_dim_v, causal, local, sparse_block_size_q=None):
        configuration = selector(head_dim, head_dim_v, causal, local, sparse_block_size_q)
        if head_dim == 128 and not causal:
            return replace(configuration, m_block_size=64, dQ_swapAB=False)
        return configuration

    return make_private_api(
        FlashAttentionForwardSm90=PersistentFlashAttentionForwardSm90 if persistent else FlashAttentionForwardSm90,
        FlashAttentionBackwardSm90=FlashAttentionBackwardSm90,
        FlashAttentionBackwardPreprocess=FlashAttentionBackwardPreprocess,
        _tile_size_bwd_sm90=select,
    )


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, scale):
        # Persistence qualified a small gain only for long dense D128.
        persistent = not causal and q.shape[-1] == 128 and q.shape[-2] == 4096
        maximum = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        output = torch.empty_like(q)
        _, logsum, *_ = _api(persistent)._flash_attn_fwd(
            *(tensor.transpose(1, 2) for tensor in (q, k, v)),
            softmax_scale=scale,
            causal=causal,
            pack_gqa=False,
            out=output.transpose(1, 2),
            return_lse=True,
            aux_tensors=[maximum],
        )
        ctx.save_for_backward(q, k, v, output, maximum, logsum)
        ctx.causal, ctx.scale = causal, scale
        return output

    @staticmethod
    def backward(ctx, dout):
        check_atomic_backward()
        q, k, v, out, maximum, logsum = ctx.saved_tensors
        padding = (-maximum.shape[-1]) % 640
        if padding:
            maximum = torch.nn.functional.pad(maximum, (0, padding))
        gradients = _api()._flash_attn_bwd(
            *(tensor.transpose(1, 2) for tensor in (q, k, v, out, dout)),
            logsum,
            ctx.scale,
            ctx.causal,
            0.0,
            aux_tensors=[maximum],
        )
        return *(gradient.transpose(1, 2) for gradient in gradients[:3]), None, None


def attention(q, k, v, *, causal=False, scale=None):
    return _Attention.apply(q, k, v, causal, q.shape[-1] ** -0.5 if scale is None else scale)
