"""Strict CuTe attention and automatic per-call native SDPA selection."""

from functools import lru_cache
from math import isfinite

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


def neighborhood_attention(q, k, v, *, spatial_shape, window_size, scale=None):
    """Experimental noncausal 2D attention, stride/dilation one, head dim 64.

    Q/K/V: matching FP16/BF16 CUDA tensors [B, heads, height*width, 64].
    ``spatial_shape`` is (height, width); ``window_size`` is an int or (h, w).
    Each axis starts at clamp(query - window//2, 0, extent - window),
    matching NATTEN's shifted boundary windows, including even windows.
    Normalized dQ and reconstructed dK corrections are enabled by default;
    see https://github.com/SimpleTuner-io/KohakuFA-CuTe/tree/c373d7bf0a7096a9be0fcc7e488191414aac959c/docs/neighborhood
    for the precision contract and qualification details.
    """
    if (
        not isinstance(spatial_shape, (tuple, list))
        or len(spatial_shape) != 2
        or any(type(n) is not int or n <= 0 for n in spatial_shape)
    ):
        raise ValueError("spatial_shape must contain two positive integers.")
    if type(window_size) is int:
        window_size = (window_size, window_size)
    if (
        not isinstance(window_size, (tuple, list))
        or len(window_size) != 2
        or any(type(n) is not int or n <= 0 or n > extent for n, extent in zip(window_size, spatial_shape))
    ):
        raise ValueError("window_size must contain positive integers no larger than spatial_shape.")
    _validate(q, k, v, None, 0, False, False)
    if q.shape != k.shape or q.shape[-1] != 64 or q.shape[2] != spatial_shape[0] * spatial_shape[1]:
        raise ValueError("Neighborhood attention requires matching [B, heads, height*width, 64] Q/K/V.")
    scale = 0.125 if scale is None else float(scale)
    if not isfinite(scale) or abs(scale) > torch.finfo(torch.float32).max:
        raise ValueError("Neighborhood attention requires a finite FP32 scale.")
    from .neighborhood_ops import forward

    # DLPack describes physical storage. Resolve logical view flags while
    # autograd can still map the logical operands back to their base tensors.
    q, k, v = (tensor.resolve_conj().resolve_neg() for tensor in (q, k, v))
    return forward(q, k, v, *spatial_shape, *window_size, scale)[0]


def _autocast_sdpa_inputs(query, key, value):
    if query.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        return tuple(t.to(dtype) if t.is_floating_point() and t.dtype != torch.float64 else t for t in (query, key, value))
    return query, key, value


def kohaku_fa_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
):
    """Strict CuTe implementation with the PyTorch SDPA argument convention."""
    query, key, value = _autocast_sdpa_inputs(query, key, value)
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
    query, key, value = _autocast_sdpa_inputs(query, key, value)
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
