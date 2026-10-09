"""Eager and compiler-safe autograd for the local CuTe kernels."""

import torch

from .kernels import centered_backward
from .native import extension


def forward_impl(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, scale: float, causal: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(extension().forward(q, k, v, mask, scale, causal))


forward = torch.library.custom_op("simpletuner_cute_attention::forward", mutates_args=())(forward_impl)


@forward.register_fake
def forward_fake(q, k, v, mask, scale, causal):
    stats = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    blocks = torch.empty(
        (
            1 if mask is None or mask.stride(0) == 0 else q.shape[0],
            1 if mask is None or mask.stride(1) == 0 else q.shape[1],
            1 if mask is None or mask.stride(2) == 0 else (q.shape[2] + 63) // 64,
            1 if mask is None or mask.stride(3) == 0 else (k.shape[2] + 63) // 64,
        ),
        device=q.device,
        dtype=torch.int32,
    )
    return (torch.empty_like(q), stats, torch.empty_like(stats), blocks)


def backward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    maximum: torch.Tensor,
    logsum: torch.Tensor,
    dout: torch.Tensor,
    blocks: torch.Tensor,
    out: torch.Tensor | None,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(extension().backward(q, k, v, mask, maximum, logsum, dout, blocks, out, scale, causal))


backward = torch.library.custom_op("simpletuner_cute_attention::backward", mutates_args=())(backward_impl)


@backward.register_fake
def backward_fake(q, k, v, mask, maximum, logsum, dout, blocks, out, scale, causal):
    return tuple(torch.empty_like(tensor) for tensor in (q, k, v))


def setup_context(ctx, inputs, output):
    q, k, v, mask, ctx.scale, ctx.causal = inputs
    out, maximum, logsum, blocks = output
    ctx.has_mask = mask is not None
    ctx.centered = centered_backward(q)
    ctx.save_for_backward(
        q, k, v, maximum, logsum, blocks, *(() if mask is None else (mask,)), *((out,) if ctx.centered else ())
    )

    ctx.mark_non_differentiable(*output[1:])


def autograd_backward(ctx, dout, dmaximum, dlogsum, dblocks):
    q, k, v, maximum, logsum, blocks, *extras = ctx.saved_tensors
    dq, dk, dv = backward(
        q,
        k,
        v,
        extras[0] if ctx.has_mask else None,
        maximum,
        logsum,
        dout,
        blocks,
        extras[-1] if ctx.centered else None,
        ctx.scale,
        ctx.causal,
    )
    return dq, dk, dv, None, None, None


forward.register_autograd(autograd_backward, setup_context=setup_context)


def attention(q, k, v, *, mask=None, causal=False, scale=None):
    if not torch.compiler.is_compiling():
        return extension().attention(q, k, v, mask, scale, causal)
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("CuTe attention requires 4D Q/K/V tensors.")
    if not q.is_cuda or k.device != q.device or v.device != q.device:
        raise ValueError("CuTe attention requires Q/K/V on the same CUDA device.")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("CuTe attention requires matching FP16/BF16 Q/K/V.")
    if not all(q.shape) or not all(k.shape) or k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("CuTe attention requires nonempty matching batch/head dimensions and equal K/V shapes.")
    if q.shape[1] % k.shape[1] or not 1 <= q.shape[-1] <= 512:
        raise ValueError("CuTe attention requires head dimensions 1..512 and query heads divisible by K/V heads.")
    if mask is not None:
        if mask.dtype != torch.bool or mask.device != q.device:
            raise ValueError("CuTe attention requires a boolean mask on the Q/K/V device.")
        mask = mask.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    arguments = (q, k, v, mask, q.shape[-1] ** -0.5 if scale is None else float(scale), causal)
    return forward(*arguments)[0]


def kohaku_fa_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    if torch.compiler.is_compiling():
        from simpletuner.helpers.training.attention_backend import _kohaku_fa_sdpa

        return _kohaku_fa_sdpa(attention, query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    return extension().sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)


def automatic_kohaku_fa_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    if torch.compiler.is_compiling() or extension.cache_info().currsize == 0:
        from simpletuner.helpers.training.attention_backend import _KOHAKU_FA_AUTO_CAPABILITIES, _automatic_kohaku_fa_sdpa

        return _automatic_kohaku_fa_sdpa(
            kohaku_fa_scaled_dot_product_attention,
            _KOHAKU_FA_AUTO_CAPABILITIES,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
        )
    if query.device.type != "cuda":
        from simpletuner.helpers.training.attention_backend import AttentionBackendController

        return AttentionBackendController._call_original_sdpa(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )
    return extension().automatic_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
