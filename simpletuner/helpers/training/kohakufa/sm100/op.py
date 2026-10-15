# Vendored KohakuFA; SimpleTuner namespace changes. See NOTICE.
"""The sm_100 kernels as torch custom ops with autograd: ``attention(q, k, v, scale,
block_causal)``.

    q [B, H, Sq, D], k / v [B, H, Skv, D] fp16 (views of [B, S, H, D] projections are
    fine: any strides with a contiguous last dim and 16-byte aligned strides), D = 64.
    scale: softmax scale (None -> D ** -0.5). block_causal: frame size P (query i sees
    key j iff j // P <= i // P), 0 for dense attention.
    Returns out [B, H, Sq, D] fp16, a view of [B, Sq, H, D] storage (so
    ``out.transpose(1, 2).flatten(2)`` is free).

Forward and backward are Gluon kernels (fwd.py, bwd.py) registered as torch custom ops
(``kohakufa::sm100_forward`` / ``_backward``) with autograd, so
torch.compile keeps them in the graph (no graph break) and CUDA-graph capture works:
no host synchronization, all buffers allocated by the caching allocator.

Extra memory: forward lse (row max and log2 sum, fp32 [B, H, 2, Sq]); backward
(-m, -log2 l, -delta) per query (fp32,
padded to 128 rows) and an fp32 dQ accumulator [B, H, Sq, D] while it runs.
"""

import torch
from torch import Tensor

KERNEL_DIM_STEP = 16  # kernel head dims: multiples of 16 (chunks of 64 / 32 / 16)
MAX_KERNEL_DIM = 512  # beyond 128: one-half / value-split forward, sliced streamed backward

from simpletuner.helpers.training.kohakufa.mask import PackedMask
from simpletuner.helpers.training.kohakufa.sm100.bwd import (
    attention_backward,
    attention_backward_varlen,
    varlen_backward_tables,
)
from simpletuner.helpers.training.kohakufa.sm100.fwd import (
    attention_forward,
    attention_forward_varlen,
    forward_rows,
    varlen_tiles,
)


def _check(q: Tensor, k: Tensor, v: Tensor) -> None:
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("attention: q, k, v must be fp16 or bf16 (all the same)")
    dim = q.shape[-1]
    if dim % KERNEL_DIM_STEP or not 0 < dim <= MAX_KERNEL_DIM:
        raise ValueError(f"attention: kernel head_dim {dim}: a multiple of 16 up to 512")
    if k.shape != v.shape or k.shape[0] != q.shape[0] or k.shape[-1] != q.shape[-1]:
        raise ValueError(f"attention: k {tuple(k.shape)} / v {tuple(v.shape)} vs q {tuple(q.shape)}")
    if q.shape[1] % k.shape[1]:
        raise ValueError("attention: query heads must be a multiple of K / V heads (GQA)")
    for t in (q, k, v):
        if t.stride(-1) != 1 or any((s * 2) % 16 for s in t.stride()[:-1]):
            raise ValueError("attention: inputs need a contiguous last dim and 16-byte strides")


@torch.library.custom_op("simpletuner_kohakufa::sm100_forward", mutates_args=())
def attention_fwd_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: float,
    block: int,
    mask_words: Tensor | None,
    mask_words_t: Tensor | None,
    list_start: Tensor | None,
    list_count: Tensor | None,
    list_entries: Tensor | None,
    list_classes: Tensor | None,
    list_start_t: Tensor | None,
    list_count_t: Tensor | None,
    list_entries_t: Tensor | None,
    list_classes_t: Tensor | None,
) -> tuple[Tensor, Tensor]:
    lists = None
    if list_start is not None:
        q_tiles = list_start.numel() // (q.shape[0] * q.shape[1])
        lists = (list_start, list_count, list_entries, list_classes, q_tiles)
    return attention_forward(q, k, v, scale, block, mask_words, lists)


@attention_fwd_op.register_fake
def _(q, k, v, scale, block, mask_words, mask_words_t, *lists):
    batch, heads, seq_q, dim = q.shape
    out = q.new_empty(batch, seq_q, heads, dim).transpose(1, 2)
    lse = q.new_empty(batch, heads, 2, seq_q, dtype=torch.float32)
    return out, lse


@torch.library.custom_op("simpletuner_kohakufa::sm100_backward", mutates_args=())
def attention_bwd_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    dout: Tensor,
    lse: Tensor,
    scale: float,
    block: int,
    mask_words_t: Tensor | None,
    list_start_t: Tensor | None,
    list_count_t: Tensor | None,
    list_entries_t: Tensor | None,
    list_classes_t: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    lists_t = None
    if list_start_t is not None:
        rows = list_start_t.numel()
        k_tiles = rows // (q.shape[0] * q.shape[1])
        lists_t = (list_start_t, list_count_t, list_entries_t, list_classes_t, k_tiles, rows)
    return attention_backward(q, k, v, out, dout, lse, scale, block, mask_words_t, lists_t)


@attention_bwd_op.register_fake
def _(q, k, v, out, dout, lse, scale, block, mask_words_t, *lists_t):
    def like(t):
        batch, heads, seq, dim = t.shape
        return t.new_empty(batch, seq, heads, dim).transpose(1, 2)

    return like(q), like(k), like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, scale, block, _, mask_words_t, *_ = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, mask_words_t, *inputs[11:15])
    ctx.scale = scale
    ctx.block = block
    ctx.mark_non_differentiable(lse)


def _backward(ctx, dout, dlse):
    q, k, v, out, lse, mask_words_t, *lists_t = ctx.saved_tensors
    if dout.stride(-1) != 1 or any((s * 2) % 16 for s in dout.stride()[:-1]):
        dout = dout.contiguous()
    dq, dk, dv = attention_bwd_op(q, k, v, out, dout, lse, ctx.scale, ctx.block, mask_words_t, *lists_t)
    return (dq, dk, dv) + (None,) * 12


attention_fwd_op.register_autograd(_backward, setup_context=_setup_context)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: float | None = None,
    block_causal: int = 0,
    mask: Tensor | None = None,
) -> Tensor:
    """``softmax(scale q k^T) v`` over ``[B, H, S, D]``; ``block_causal > 0``: tokens of
    block ``i`` (``block_causal`` consecutive tokens) see blocks ``0..i`` only; ``mask``:
    a bool mask broadcastable to ``[B, H, Sq, Skv]`` (True = visible; a row with no
    visible key outputs 0)."""
    _check(q, k, v)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    words = words_t = None
    lists = lists_t = (None,) * 4
    if mask is not None:
        if block_causal:
            raise ValueError("attention: give a mask or block_causal, not both (fold it in)")
        shape = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
        packed = mask if isinstance(mask, PackedMask) else PackedMask(mask, *shape)
        if packed.shape != shape:
            raise ValueError(f"packed mask for {packed.shape}, inputs are {shape}")
        words, words_t = packed.words, packed.words_t
        lists, lists_t = packed.lists, packed.lists_t
    out, _ = attention_fwd_op(q, k, v, float(scale), int(block_causal), words, words_t, *lists, *lists_t)
    return out


# --------------------------------------------------------------------------------------
# Variable-length (packed) sequences
# --------------------------------------------------------------------------------------


@torch.library.custom_op("simpletuner_kohakufa::sm100_varlen_forward", mutates_args=())
def varlen_fwd_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    fwd_tiles: Tensor,
    key_tiles: Tensor,
    q_tiles: Tensor,
    cu_qt: Tensor,
    scale: float,
    block: int,
) -> tuple[Tensor, Tensor]:
    return attention_forward_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, fwd_tiles, scale, block)


@varlen_fwd_op.register_fake
def _(q, k, v, cu_seqlens_q, cu_seqlens_k, fwd_tiles, key_tiles, q_tiles, cu_qt, scale, block):
    return torch.empty_like(q), q.new_empty(1, q.shape[1], 2, q.shape[0], dtype=torch.float32)


@torch.library.custom_op("simpletuner_kohakufa::sm100_varlen_backward", mutates_args=())
def varlen_bwd_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    dout: Tensor,
    lse: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    key_tiles: Tensor,
    q_tiles: Tensor,
    cu_qt: Tensor,
    scale: float,
    block: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return attention_backward_varlen(
        q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, key_tiles, q_tiles, cu_qt,
        scale, block,
    )  # fmt: skip


@varlen_bwd_op.register_fake
def _(q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, key_tiles, q_tiles, cu_qt, scale, block):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _varlen_setup(ctx, inputs, output):
    q, k, v, cu_q, cu_k, _, key_tiles, q_tiles, cu_qt, scale, block = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k, key_tiles, q_tiles, cu_qt)
    ctx.scale, ctx.block = scale, block
    ctx.mark_non_differentiable(lse)


def _varlen_backward(ctx, dout, dlse):
    q, k, v, out, lse, cu_q, cu_k, key_tiles, q_tiles, cu_qt = ctx.saved_tensors
    dq, dk, dv = varlen_bwd_op(
        q, k, v, out, dout.contiguous(), lse, cu_q, cu_k, key_tiles, q_tiles, cu_qt,
        ctx.scale, ctx.block,
    )  # fmt: skip
    return dq, dk, dv, None, None, None, None, None, None, None, None


varlen_fwd_op.register_autograd(_varlen_backward, setup_context=_varlen_setup)


def varlen_plan(
    cu_seqlens_q: Tensor, cu_seqlens_k: Tensor, heads: int, kv_heads: int, block: int = 0
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The tile tables of a batch of packed sequences (reads ``cu_seqlens`` on the host
    once): build it outside CUDA-graph capture and reuse it for every call with the same
    lengths. The forward's tables come in 256- and 128-row tiles (the latter for wide
    heads, whose forward runs one 128-row half per CTA)."""
    fwd_tiles = {rows: varlen_tiles(cu_seqlens_q, heads, rows) for rows in (256, 128)}
    key_tiles, q_tiles, cu_qt = varlen_backward_tables(cu_seqlens_q, cu_seqlens_k, kv_heads, block)
    return fwd_tiles[256], fwd_tiles[128], key_tiles, q_tiles, cu_qt


def attention_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    scale: float | None = None,
    block_causal: int = 0,
    plan: tuple | None = None,
) -> Tensor:
    """Packed sequences, each its own attention problem: ``q [Tq, H, D]``, ``k / v
    [Tk, Hkv, D]``; sequence ``b`` is rows ``cu_seqlens[b] : cu_seqlens[b + 1]``
    (int32 [B + 1] on the device). ``plan`` from ``varlen_plan`` (built here if None)."""
    _check(*(t.unsqueeze(0).transpose(1, 2) for t in (q, k, v)))
    if scale is None:
        scale = q.shape[-1] ** -0.5
    cu_seqlens_q, cu_seqlens_k = cu_seqlens_q.int(), cu_seqlens_k.int()
    if plan is None:
        plan = varlen_plan(cu_seqlens_q, cu_seqlens_k, q.shape[1], k.shape[1], block_causal)
    tiles_256, tiles_128, *tables = plan
    rows = forward_rows(q.shape[-1], q.element_size(), block_causal > 0)
    fwd_tiles = tiles_256 if rows == 256 else tiles_128
    out, _ = varlen_fwd_op(q, k, v, cu_seqlens_q, cu_seqlens_k, fwd_tiles, *tables, float(scale), int(block_causal))
    return out
