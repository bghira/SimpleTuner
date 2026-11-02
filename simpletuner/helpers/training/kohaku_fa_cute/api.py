"""Strict CuTe attention and automatic per-call native SDPA selection."""

from functools import lru_cache

import torch

from . import _ops, fa4_sm80

_CAPABILITIES = {(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)}


@lru_cache(maxsize=1)
def extension():
    from .native import extension

    return extension()


def _try_rtx4090_s1024(q, k, v, mask, dropout, causal, scale, gqa):
    if (
        torch.compiler.is_compiling()
        or getattr(q, "ndim", None) != 4
        or q.device.type != "cuda"
        or q.shape[2] != 1024
        or q.shape[-1] not in (64, 128)
    ):
        return None
    from .fa4_sm80 import rtx4090

    return rtx4090.try_s1024_attention(q, k, v, mask, dropout, causal, scale, gqa)


def _validate(q, k, v, mask, dropout, causal, gqa, *, combined_mask=False):
    if dropout != 0.0:
        raise ValueError("CuTe attention does not support dropout.")
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("CuTe attention requires 4D [batch, heads, sequence, dim] Q/K/V.")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("CuTe attention requires Q/K/V on the same CUDA device.")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("CuTe attention requires matching FP16/BF16 Q/K/V; FP32 attention is unsupported.")
    if not all(q.shape) or not all(k.shape) or k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3]:
        raise ValueError("CuTe attention requires nonempty matching batch/head dimensions and equal K/V shapes.")
    if q.shape[1] % k.shape[1] or not 1 <= q.shape[3] <= 512:
        raise ValueError("CuTe attention requires head dimensions 1..512 and query heads divisible by K/V heads.")
    if not gqa and q.shape[1] != k.shape[1]:
        raise ValueError("CuTe SDPA requires enable_gqa=True for differing Q and K/V head counts.")
    if mask is not None:
        if mask.dtype != torch.bool or mask.device != q.device:
            raise ValueError("CuTe attention requires a boolean mask on the Q/K/V device.")
        if causal and not combined_mask:
            raise ValueError("CuTe attention requires causal visibility folded into the boolean mask.")
    if torch.cuda.get_device_capability(q.device) not in _CAPABILITIES:
        raise ValueError("CuTe attention requires sm_80, sm_86, sm_89, sm_90, sm_100, sm_103, sm_120 or sm_121.")


def _center_keys(k):
    """Subtract each batch/head's mean key in FP32 and round once to the input dtype."""
    keys = k.float()
    return (keys - keys.mean(dim=-2, keepdim=True)).to(k.dtype)


def attention(q, k, v, *, mask=None, causal=False, scale=None, center_keys=False):
    """Strict CUDA attention; Q/K/V have shape [B, H, S, D], with implicit GQA.

    ``center_keys`` subtracts the mean key per batch and K/V head in FP32 before
    rounding back to the input dtype. Softmax is invariant to this shared logit
    shift in exact arithmetic, including with masks. Autograd differentiates
    through the mean, projecting the key gradient back to the original keys.
    """
    if center_keys or torch.compiler.is_compiling() or extension.cache_info().currsize == 0:
        _validate(q, k, v, mask, 0.0, causal, True, combined_mask=True)
    if center_keys:
        k = _center_keys(k)
    if torch.compiler.is_compiling():
        return _ops.attention(q, k, v, mask=mask, causal=causal, scale=scale)
    selected = _try_rtx4090_s1024(q, k, v, mask, 0, causal, scale, True)
    if selected is not None:
        return selected
    if fa4_sm80.compatible(q, k, v, mask, 0, causal, scale, True):
        return fa4_sm80.attention(q, k, v, causal=causal, scale=scale)
    return extension().attention(q, k, v, mask, scale, causal)


def kohaku_fa_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
):
    """Strict CuTe implementation with the PyTorch SDPA argument convention."""
    selected = _try_rtx4090_s1024(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    if selected is not None:
        return selected
    if not torch.compiler.is_compiling() and fa4_sm80.compatible(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    ):
        return fa4_sm80.attention(query, key, value, causal=is_causal, scale=scale)
    if torch.compiler.is_compiling() or extension.cache_info().currsize == 0:
        _validate(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa)
        if torch.compiler.is_compiling():
            return _ops.attention(query, key, value, mask=attn_mask, causal=is_causal, scale=scale)
    return extension().sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)


scaled_dot_product_attention = kohaku_fa_scaled_dot_product_attention


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
    if query.device.type == "cuda":
        selected = _try_rtx4090_s1024(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        if selected is not None:
            return selected
        if not torch.compiler.is_compiling() and fa4_sm80.compatible(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        ):
            return fa4_sm80.attention(query, key, value, causal=is_causal, scale=scale)
    if torch.compiler.is_compiling() or extension.cache_info().currsize == 0:
        from simpletuner.helpers.training.attention_backend import _automatic_kohaku_fa_sdpa

        return _automatic_kohaku_fa_sdpa(
            kohaku_fa_scaled_dot_product_attention,
            _CAPABILITIES,
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
