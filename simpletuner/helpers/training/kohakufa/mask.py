"""Boolean attention masks packed for the kernels: 32 keys per int32 word along each
query row (forward), and 32 queries per word along each key row (backward), padded to
whole 128-wide tiles so a tile's words never run past a row."""

# SimpleTuner modifications: fixed-size tile lists, SDPA broadcasting, custom-op mask packing.

import torch
import triton
import triton.language as tl
from torch import Tensor

TILE = 128


def normalize(mask: Tensor, batch: int, heads: int, s_q: int, s_kv: int) -> Tensor:
    """A bool mask broadcastable to ``[B, H, Sq, Skv]`` (True = visible) as 4D with
    batch / head dimensions of size 1 or B / H."""
    if mask.dtype != torch.bool:
        raise TypeError("mask: a boolean tensor (True = the key is visible)")
    while mask.dim() < 4:
        mask = mask.unsqueeze(0)
    if (
        mask.shape[-2] not in (1, s_q)
        or mask.shape[-1] not in (1, s_kv)
        or mask.shape[0] not in (1, batch)
        or mask.shape[1] not in (1, heads)
    ):
        raise ValueError(f"mask {tuple(mask.shape)} does not broadcast to {(batch, heads, s_q, s_kv)}")
    return mask.expand(mask.shape[0], mask.shape[1], s_q, s_kv)


@triton.jit
def _pack_kernel(
    mask_ptr, words_ptr, words_t_ptr, any_ptr, all_ptr, s_q, s_kv, heads_m,
    stride_b, stride_h, stride_r, stride_c, q_blocks, k_blocks,
    BLOCK: tl.constexpr,
):  # fmt: skip
    """One (mask slice, 128-query block, 128-key block): the rows' words (bit c % 32 of
    word c // 32 = key c), the transposed words (per key, over queries), and whether
    any / every key of the block is visible (rows past S_q count as seeing all; keys past
    S_kv as hidden)."""
    slice_ = tl.program_id(0)
    qb = tl.program_id(1)
    kb = tl.program_id(2)
    b = slice_ // heads_m
    h = slice_ % heads_m
    rows = qb * BLOCK + tl.arange(0, BLOCK)
    cols = kb * BLOCK + tl.arange(0, BLOCK)
    row_ok = rows < s_q
    col_ok = cols < s_kv
    ptrs = mask_ptr + b * stride_b + h * stride_h + rows[:, None] * stride_r + cols[None, :] * stride_c
    bits = tl.load(ptrs, mask=row_ok[:, None] & col_ok[None, :], other=0).to(tl.int32)
    shifts = tl.arange(0, 32)
    # rows' words: [BLOCK, BLOCK / 32] (distinct bits: their sum is their OR)
    words = tl.sum(tl.reshape(bits, (BLOCK, BLOCK // 32, 32)) << shifts[None, None, :], axis=2)
    word_cols = kb * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    out = words_ptr + (slice_ * s_q + rows[:, None]).to(tl.int64) * (k_blocks * (BLOCK // 32))
    tl.store(out + word_cols[None, :], words, mask=row_ok[:, None])
    # keys' words over queries: the transpose
    bits_t = tl.trans(bits)
    words_t = tl.sum(tl.reshape(bits_t, (BLOCK, BLOCK // 32, 32)) << shifts[None, None, :], axis=2)
    word_rows = qb * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    out_t = words_t_ptr + (slice_ * s_kv + cols[:, None]).to(tl.int64) * (q_blocks * (BLOCK // 32))
    tl.store(out_t + word_rows[None, :], words_t, mask=col_ok[:, None])
    # block classes
    full = tl.where(row_ok[:, None], bits, 1)
    full = tl.where(col_ok[None, :], full, 0)
    cell = (slice_ * q_blocks + qb) * k_blocks + kb
    tl.store(any_ptr + cell, tl.max(tl.max(bits, axis=1), axis=0).to(tl.int8))
    tl.store(all_ptr + cell, tl.min(tl.min(full, axis=1), axis=0).to(tl.int8))


def _mask_buffers(dense: Tensor, s_q: int, s_kv: int):
    bm, hm = dense.shape[:2]
    nq, nk = -(-s_q // TILE), -(-s_kv // TILE)
    words = dense.new_empty(bm, hm, s_q, nk * 4, dtype=torch.int32)
    words_t = dense.new_empty(bm, hm, s_kv, nq * 4, dtype=torch.int32)
    any_ = dense.new_empty(bm, hm, nq, nk, dtype=torch.int8)
    return words, words_t, any_, torch.empty_like(any_)


@torch.library.custom_op("simpletuner_kohakufa::pack_mask_words", mutates_args=())
def _pack_words(dense: Tensor, s_q: int, s_kv: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    words, words_t, any_, all_ = _mask_buffers(dense, s_q, s_kv)
    bm, hm, nq, nk = any_.shape
    _pack_kernel[(bm * hm, nq, nk)](
        dense.view(torch.uint8), words, words_t, any_, all_, s_q, s_kv, hm,
        *dense.stride(), nq, nk, BLOCK=TILE,
    )  # fmt: skip
    return words, words_t, any_, all_


_pack_words.register_fake(_mask_buffers)


def _csr(listed: Tensor, classes: Tensor, batch: int, heads: int):
    """CSR over ``listed [Bm, Hm, R, C]`` (bool) per (batch * head, row), with the per-entry
    ``classes``; a row with nothing listed keeps column 0, so every row runs one step."""
    empty = ~listed.any(dim=-1, keepdim=True)
    first = torch.zeros_like(listed)
    first[..., 0] = True
    listed = listed | (empty & first)
    n_rows, n_cols = listed.shape[-2:]
    listed = listed.expand(batch, heads, n_rows, n_cols).reshape(-1, n_cols)
    classes = classes.expand(batch, heads, n_rows, n_cols).reshape(-1, n_cols)
    count = listed.sum(dim=1).to(torch.int32)
    columns = torch.arange(n_cols, device=listed.device).expand_as(listed)
    entries = torch.where(listed, columns, columns + n_cols).argsort(dim=1)
    start = torch.arange(listed.shape[0], device=listed.device, dtype=torch.int32) * n_cols
    return (
        start.contiguous(),
        count.contiguous(),
        entries.flatten().to(torch.int32),
        classes.gather(1, entries).flatten().to(torch.int32),
    )


class PackedMask:
    """A boolean mask packed once for the kernels (words, transposed words, tile lists),
    reusable across calls with the same mask and shape (e.g. every layer of a model):
    packing a large mask costs more than the attention it gates."""

    def __init__(self, mask: Tensor, batch: int, heads: int, s_q: int, s_kv: int) -> None:
        dense = normalize(mask, batch, heads, s_q, s_kv)
        self.shape = (batch, heads, s_q, s_kv)
        nq = -(-s_q // TILE)
        self.words, self.words_t, any_, all_ = _pack_words(dense, s_q, s_kv)
        any_, all_ = any_.bool(), all_.bool()
        # forward: 256-row query tiles (two 128-row parts) per key tile
        pad = nq % 2
        any_q = torch.nn.functional.pad(any_, (0, 0, 0, pad)).unflatten(2, (-1, 2))
        all_q = torch.nn.functional.pad(all_, (0, 0, 0, pad), value=True).unflatten(2, (-1, 2))
        classes = all_q[:, :, :, 0].int() | (all_q[:, :, :, 1].int() << 1)
        self.lists = _csr(any_q.any(dim=3), classes, batch, heads)
        # backward: 128-query tiles per key tile (the transpose)
        self.lists_t = _csr(any_.transpose(-1, -2), all_.transpose(-1, -2).int(), batch, heads)


def pack_mask(mask: Tensor, batch: int, heads: int, s_q: int, s_kv: int) -> PackedMask:
    """Pack ``mask`` (bool, broadcastable to ``[B, H, Sq, Skv]``, True = visible) once;
    pass the result as ``attention(..., mask=packed)``."""
    return PackedMask(mask, batch, heads, s_q, s_kv)
