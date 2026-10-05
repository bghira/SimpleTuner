# Vendored KohakuFA; SimpleTuner namespace changes. See NOTICE.
"""Attention backward as one warp-specialized Gluon kernel (Blackwell, tcgen05 + TMEM).

Given q, k, v, out, dout [B, H, S, D] fp16 and the forward's softmax statistics (row
max m of the raw scores and log2 l), with P = softmax(scale q k^T) = exp2(c (q k^T - m)
- log2 l), c = scale log2(e), and delta = rowsum(dout * out):

    dV = P^T dO,   dS = P * (dO V^T - delta),   dK = scale dS^T Q,   dQ = scale dS K

Work decomposition. A CTA tile is 128 keys of one (batch, head); it walks the query
tiles (128 rows, "steps") that see any of its keys and accumulates dK and dV in TMEM.
Everything is computed transposed (keys on the TMEM lanes): S^T = K Q^T, dP^T = V dO^T,
so P^T is the A operand of dV += P^T dO straight from TMEM, and dS^T (written once to
smem) is the A operand of dK += dS^T Q and, transposed, of dQ_i = dS K. dQ_i is a
128 x D fp32 partial per (key tile, query tile), added to an fp32 dQ accumulator in
global memory with TMA bulk reduce-add (cp.reduce.async.bulk.tensor .add).

Head dim. D = NCH x DC: Q / K / V / dO are staged as NCH separate [128, DC] boxes (DC in
64 / 32 / 16: a 128-byte TMA row), S^T and dP^T accumulate over the chunks, and dV / dK
/ dQ are NCH power-of-two [128, DC] TMEM blocks.

Block-causal masking (block size P) by block sparsity: a key tile only visits query
tiles at or after the first frame that sees it (earlier ones are fully hidden: never
loaded or computed); query tiles whose first row sees the whole key tile run unmasked;
only the few query tiles on the frame boundary apply an elementwise mask (key c is
visible to query r iff r >= (c // P) * P, one division per key row and tile). Query
rows past S_q get -m = -inf (P = 0); keys past S_kv are masked (P = 0) in the last key
tile: their score is 0, and exp2(-c m - log2 l) overflows when a row's scores are all
negative (dQ += dS K would then be inf * 0).

Partitions:

    load       1 warp   TMA: K, V of the tile (KV_BUFS buffers: with two, the next
                        tile's load overlaps); Q_i with the step's (-m, -log2 l, -delta)
                        into Q_SLOTS rotating slots, dO_i into DO_SLOTS
    mma        1 warp   per step i, in issue order:
                          [P(i) in TMEM]        dV += P^T(i) dO_i, S^T(i+1) = K Q^T
                          [dP(i) read out]      dP^T(i+1) = V dO^T
                          [dS(i) in smem]       dK += dS^T(i) Q_i, dQ(i) = dS(i) K
    p          4 warps  P^T(i) = exp2(c (S^T - m) - log2 l) for all 128 queries ->
                        TMEM (fp16); runs a step ahead of the dS warpgroup
    ds         4 warps  dP^T(i) -> registers (frees its TMEM slot for dP^T(i+1) at
                        once), dS^T = P^T (dP^T - delta) -> smem (fp16)
    reduce     4 warps  dQ(i) TMEM -> smem -> TMA reduce-add; at the end of a tile
                        dK (scaled), dV -> fp16 -> TMA store

TMEM (512 columns): S^T 128 | dP^T 128 | P^T 64 (fp16 x 128) | dV D | dK D | dQ D fits
up to D = 64. Above that (ALIAS, FA4's layout) P^T lives in the S^T columns and dQ in
the dP^T columns: S^T 128 | dP^T 128 | dV D | dK D, up to D = 128. The aliases cost
ordering: S^T(i+1) waits until the dS warpgroup has read P^T(i) (it reads all of P^T
first for that reason), and dP^T(i+1) waits until dQ(i) has been read out, so the mma
warp issues dK(i), dQ(i) before dP^T(i+1).
"""

import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._core import builtin
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    float2,
    get_tmem_reg_layout,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
    tensor_memory_descriptor,
    tma,
)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.core import _aggregate as aggregate

from simpletuner.helpers.training.kohakufa.sm100.config import backward_config
from simpletuner.helpers.training.kohakufa.sm100.fwd import LOG2E, _join_columns, mask_arguments, sm_count, tma_descriptor

TILE_ROWS = 128
TILE = gl.constexpr(TILE_ROWS)  # keys per CTA tile = query rows per step
STATS = gl.constexpr(3)  # per query row: -m, -log2 l, -delta
CHUNK = gl.constexpr(32)  # score columns a compute thread holds at once


@aggregate
class Problem:
    qk_scale: gl.tensor  # softmax scale * log2(e)
    heads: gl.tensor  # query heads
    group: gl.tensor  # query heads per K / V head (GQA; 1 = multi-head)
    seq_q: gl.tensor
    seq_kv: gl.tensor
    block: gl.tensor  # block-causal block size (unused when not CAUSAL)
    num_kv_tiles: gl.tensor
    num_tiles: gl.tensor
    mask_ptr: gl.tensor  # MASK: transposed words, bit c of word w of a key row = query 32 w + c
    mask_b_stride: gl.tensor
    mask_h_stride: gl.tensor
    mask_words: gl.tensor  # words per key row (a multiple of TILE / 32)
    stats_tiles: gl.tensor  # query tiles per (batch, head) slice of the stats
    tiles_ptr: gl.tensor  # VARLEN: int32 [num_tiles, 3] = (sequence, K / V head, first key)
    cu_q_ptr: gl.tensor  # VARLEN: int32 [B + 1] packed query offsets
    cu_k_ptr: gl.tensor  # VARLEN: int32 [B + 1] packed key offsets
    cu_qt_ptr: gl.tensor  # VARLEN: int32 [B + 1] first stats query tile of each sequence
    dk_ptr: gl.tensor  # VARLEN: dK / dV, for row-masked stores at sequence ends
    dv_ptr: gl.tensor
    kv_row_stride: gl.tensor  # VARLEN: elements between packed K / V rows (Hkv * D)
    list_start_ptr: gl.tensor  # MASK: per (batch * query head, key tile) list start,
    list_count_ptr: gl.tensor  # ... and length, into
    list_ptr: gl.tensor  # MASK: the query tiles a key tile visits (hidden ones skipped)
    list_cls_ptr: gl.tensor  # MASK: per listed tile, 1 = every query sees every key
    k_tiles: gl.tensor  # MASK: key tiles per (batch, query head)
    list_rows: gl.tensor  # MASK: entries of list_start (clamp for look-ahead tiles)
    HEAD_DIM: gl.constexpr  # D = NCH * DC
    DC: gl.constexpr  # head-dim chunk: tiles are staged and multiplied per chunk
    NCH: gl.constexpr
    ALIAS: gl.constexpr  # P^T in the S^T columns, dQ in the dP^T columns
    KV_BUFS: gl.constexpr
    Q_SLOTS: gl.constexpr  # (STREAM: statistics slots)
    DO_SLOTS: gl.constexpr  # (STREAM: ring slots)
    CAUSAL: gl.constexpr
    MASK: gl.constexpr
    VARLEN: gl.constexpr
    STREAM: gl.constexpr  # wide heads: output slices, operands streamed by chunk
    NSL: gl.constexpr  # output slices per key tile (CTAs sharing it)
    NW: gl.constexpr  # head-dim chunks per output slice (the last may have fewer)
    K_RES: gl.constexpr  # STREAM: all of K resident (else only the slice's chunks)

    @gluon.constexpr_function
    def __init__(
        self, qk_scale, heads, group, seq_q, seq_kv, block, num_kv_tiles, num_tiles,
        mask_ptr, mask_b_stride, mask_h_stride, mask_words, stats_tiles, tiles_ptr,
        cu_q_ptr, cu_k_ptr, cu_qt_ptr, dk_ptr, dv_ptr, kv_row_stride, list_start_ptr,
        list_count_ptr, list_ptr, list_cls_ptr, k_tiles, list_rows,
        DC, NCH, ALIAS, KV_BUFS, Q_SLOTS, DO_SLOTS, CAUSAL, MASK, VARLEN, STREAM, NSL, NW,
        K_RES,
    ):  # fmt: skip
        self.qk_scale = qk_scale
        self.heads = heads
        self.group = group
        self.seq_q = seq_q
        self.seq_kv = seq_kv
        self.block = block
        self.num_kv_tiles = num_kv_tiles
        self.num_tiles = num_tiles
        self.mask_ptr = mask_ptr
        self.mask_b_stride = mask_b_stride
        self.mask_h_stride = mask_h_stride
        self.mask_words = mask_words
        self.stats_tiles = stats_tiles
        self.tiles_ptr = tiles_ptr
        self.cu_q_ptr = cu_q_ptr
        self.cu_k_ptr = cu_k_ptr
        self.cu_qt_ptr = cu_qt_ptr
        self.dk_ptr = dk_ptr
        self.dv_ptr = dv_ptr
        self.kv_row_stride = kv_row_stride
        self.list_start_ptr = list_start_ptr
        self.list_count_ptr = list_count_ptr
        self.list_ptr = list_ptr
        self.list_cls_ptr = list_cls_ptr
        self.k_tiles = k_tiles
        self.list_rows = list_rows
        self.HEAD_DIM = gl.constexpr(DC * NCH)
        self.DC = gl.constexpr(DC)
        self.NCH = gl.constexpr(NCH)
        self.ALIAS = gl.constexpr(ALIAS)
        self.KV_BUFS = gl.constexpr(KV_BUFS)
        self.Q_SLOTS = gl.constexpr(Q_SLOTS)
        self.DO_SLOTS = gl.constexpr(DO_SLOTS)
        self.CAUSAL = gl.constexpr(CAUSAL)
        self.MASK = gl.constexpr(MASK)
        self.VARLEN = gl.constexpr(VARLEN)
        self.STREAM = gl.constexpr(STREAM)
        self.NSL = gl.constexpr(NSL)
        self.NW = gl.constexpr(NW)
        self.K_RES = gl.constexpr(K_RES)

    @gluon.jit
    def tile(self, tile_id):
        """CTA tile ``tile_id``: key tile ``tile_id // NSL`` and output slice ``tile_id %
        NSL`` (head-dim chunks NW slice .. + NW - 1 of dK, dV, dQ)."""
        info = self._key_tile(tile_id // self.NSL)
        chunk0 = (tile_id % self.NSL) * self.NW
        nw = gl.minimum(self.NCH - chunk0, self.NW)
        return TileInfo(info.batch, info.head, info.kv_start, info.first_q, info.first_full,
                        info.end_q, info.seq_q, info.seq_kv, info.q_off, info.k_off,
                        info.qt_base, chunk0, nw)  # fmt: skip

    @gluon.jit
    def _key_tile(self, tile_id):
        """Key tile ``tile_id``: its keys and the query tiles it visits."""
        if self.VARLEN:
            # the producer warps look ahead past the last tile: clamp (values unused)
            last = self.num_tiles // self.NSL - 1
            entry = self.tiles_ptr + gl.minimum(tile_id, last) * 3
            seq, head, kv_start = gl.load(entry), gl.load(entry + 1), gl.load(entry + 2)
            q_off = gl.load(self.cu_q_ptr + seq)
            seq_q = gl.load(self.cu_q_ptr + seq + 1) - q_off
            k_off = gl.load(self.cu_k_ptr + seq)
            seq_kv = gl.load(self.cu_k_ptr + seq + 1) - k_off
            qt_base = gl.load(self.cu_qt_ptr + seq)
            return self._info(seq * 0, head, kv_start, seq_q, seq_kv, q_off, k_off, qt_base)
        else:
            # a head's key tiles are consecutive: CTAs running together share its Q / dO
            # in L2 (block-causal: within a head, the longest key tiles come first)
            kv_tile = tile_id % self.num_kv_tiles
            batch_kv_head = tile_id // self.num_kv_tiles  # batch * kv_heads + kv_head
            kv_heads = self.heads // self.group
            zero = tile_id * 0
            return self._info(batch_kv_head // kv_heads, batch_kv_head % kv_heads,
                              kv_tile * TILE, self.seq_q, self.seq_kv, zero, zero, zero)  # fmt: skip

    @gluon.jit
    def _info(self, batch, head, kv_start, seq_q, seq_kv, q_off, k_off, qt_base):
        """The query tiles a key tile at ``kv_start`` of a (seq_q, seq_kv) problem visits."""
        num_q_tiles = gl.cdiv(seq_q, TILE)
        if self.CAUSAL:
            # first query row that sees key kv_start: the start of its frame
            first_q = (kv_start // self.block) * self.block // TILE
            # first query tile whose first row sees every key of the tile
            kv_end = gl.minimum(kv_start + TILE, seq_kv)
            full_row = (gl.cdiv(kv_end, self.block) - 1) * self.block
            first_full = gl.minimum(gl.cdiv(full_row, TILE), num_q_tiles)
            first_full = gl.maximum(first_full, first_q)
        else:
            first_q = kv_start * 0
            first_full = first_q
        # the key tile holding padded keys runs every step masked (see _p_partition)
        if kv_start + TILE > seq_kv:
            first_full = num_q_tiles
        zero = kv_start * 0
        return TileInfo(batch, head, kv_start, first_q, first_full, num_q_tiles,
                        seq_q, seq_kv, q_off, k_off, qt_base, zero, zero)  # fmt: skip

    @gluon.jit
    def q_head(self, info, g):
        """Query head ``g`` (0 .. group - 1) of the tile's K / V head."""
        return info.head * self.group + g

    @gluon.jit
    def q_range(self, info, g):
        """(start, count) of the query tiles query head ``g`` of the tile visits: a list
        under MASK (hidden tiles skipped), else the contiguous first_q .. end_q."""
        if self.MASK:
            entry = (info.batch * self.heads + self.q_head(info, g)) * self.k_tiles
            entry = gl.minimum(entry + info.kv_start // TILE, self.list_rows - 1)
            return gl.load(self.list_start_ptr + entry), gl.load(self.list_count_ptr + entry)
        else:
            return info.first_q, info.end_q - info.first_q

    @gluon.jit
    def q_tile(self, start, index):
        """The ``index``-th query tile of a ``q_range``."""
        if self.MASK:
            return gl.load(self.list_ptr + start + index)
        else:
            return start + index

    @gluon.jit
    def visit(self, info, start, count, index):
        """The query tile of a key tile's ``index``-th step of a ``q_range``. The
        unmasked tiles are visited rotated by the key tile's index (masked ones first, in
        order), so the CTAs of one head reduce-add into different dQ rows at a time
        instead of all into the same ones."""
        rot = info.kv_start // TILE
        if self.MASK:
            return self.q_tile(start, (index + rot) % count)
        else:
            masked = info.first_full - info.first_q
            unmasked = gl.maximum(info.end_q - info.first_full, 1)
            rotated = info.first_full + (index - masked + rot) % unmasked
            return gl.where(index < masked, info.first_q + index, rotated)

    @gluon.jit
    def q_full(self, start, index):
        """MASK: the ``index``-th listed query tile sees every key of the tile."""
        return (gl.load(self.list_cls_ptr + start + index) & 1) != 0

    @gluon.jit
    def steps_of(self, info):
        """Query steps of a tile over all of its query heads."""
        if self.MASK:
            n = info.kv_start * 0
            for g in range(self.group):
                _, count = self.q_range(info, g)
                n += count
            return n
        else:
            return (info.end_q - info.first_q) * self.group


@aggregate
class TileInfo:
    batch: gl.tensor
    head: gl.tensor  # the K / V head (query heads head * group .. + group - 1)
    kv_start: gl.tensor
    first_q: gl.tensor  # first query tile visited
    first_full: gl.tensor  # first query tile computed without a mask (end_q: none)
    end_q: gl.tensor  # one past the last query tile
    seq_q: gl.tensor  # the tile's sequence: queries, keys, packed row offsets, and the
    seq_kv: gl.tensor  # first stats query tile (dense: the problem's lengths, offsets 0)
    q_off: gl.tensor
    k_off: gl.tensor
    qt_base: gl.tensor
    chunk0: gl.tensor  # first head-dim chunk of the output slice
    nw: gl.tensor  # head-dim chunks in the output slice

    @gluon.constexpr_function
    def __init__(
        self, batch, head, kv_start, first_q, first_full, end_q, seq_q, seq_kv, q_off,
        k_off, qt_base, chunk0, nw,
    ):  # fmt: skip
        self.seq_q = seq_q
        self.seq_kv = seq_kv
        self.q_off = q_off
        self.k_off = k_off
        self.qt_base = qt_base
        self.chunk0 = chunk0
        self.nw = nw
        self.batch = batch
        self.head = head
        self.kv_start = kv_start
        self.first_q = first_q
        self.first_full = first_full
        self.end_q = end_q


@aggregate
class Smem:
    k: gl.shared_memory_descriptor  # [KV_BUFS * NCH, 1, 1, TILE, DC]: buffer * NCH + chunk
    v: gl.shared_memory_descriptor  # [KV_BUFS * NCH, 1, 1, TILE, DC]
    q: gl.shared_memory_descriptor  # [Q_SLOTS * NCH, 1, 1, TILE, DC]
    do: gl.shared_memory_descriptor  # [DO_SLOTS * NCH, 1, 1, TILE, DC]
    ds: gl.shared_memory_descriptor  # [TILE, TILE]: dS^T (keys x queries), fp16
    out: gl.shared_memory_descriptor  # [RSTAGES, 1, 1, TILE, REDUCE] fp32: dQ / dK / dV staging
    stats: gl.shared_memory_descriptor  # [STATS Q_SLOTS, TILE] fp32: -m, -log2 l, -delta

    @gluon.constexpr_function
    def __init__(self, k, v, q, do, ds, out, stats):
        self.k = k
        self.v = v
        self.q = q
        self.do = do
        self.ds = ds
        self.out = out
        self.stats = stats


@aggregate
class Bars:
    kv_ready: gl.shared_memory_descriptor  # [KV_BUFS] K and V of the tile loaded
    kv_free: gl.shared_memory_descriptor  # [KV_BUFS] all MMAs of the tile done
    q_ready: gl.shared_memory_descriptor  # [Q_SLOTS] Q_i and its statistics loaded
    q_free: gl.shared_memory_descriptor  # [Q_SLOTS]
    do_ready: gl.shared_memory_descriptor  # [DO_SLOTS] dO_i loaded
    do_free: gl.shared_memory_descriptor  # [DO_SLOTS]
    s_ready: gl.shared_memory_descriptor  # [1] S^T(i) computed
    p_ready: gl.shared_memory_descriptor  # [1] P^T(i) in TMEM, S^T(i) read
    p_read: gl.shared_memory_descriptor  # [1] dS warpgroup done reading P^T(i)
    s_read: gl.shared_memory_descriptor  # [1] ALIAS: compute half 0 read its S^T(i)
    dp_ready: gl.shared_memory_descriptor  # [1] dP^T(i) computed
    dp_read: gl.shared_memory_descriptor  # [1] dS warpgroup holds dP^T(i) in registers
    ds_ready: gl.shared_memory_descriptor  # [1] dS^T(i) in smem
    ds_free: gl.shared_memory_descriptor  # [1] dK(i), dQ(i) done reading dS^T(i)
    dq_ready: gl.shared_memory_descriptor  # [1] dQ(i) in TMEM
    dq_free: gl.shared_memory_descriptor  # [1] reduce partition read dQ(i)
    dkv_ready: gl.shared_memory_descriptor  # [1] dK, dV of the tile final
    dkv_free: gl.shared_memory_descriptor  # [1] reduce partition read dK, dV

    @gluon.constexpr_function
    def __init__(
        self, kv_ready, kv_free, q_ready, q_free, do_ready, do_free, s_ready, p_ready,
        p_read, s_read, dp_ready, dp_read, ds_ready, ds_free, dq_ready, dq_free, dkv_ready,
        dkv_free,
    ):  # fmt: skip
        self.kv_ready = kv_ready
        self.kv_free = kv_free
        self.q_ready = q_ready
        self.q_free = q_free
        self.do_ready = do_ready
        self.do_free = do_free
        self.s_ready = s_ready
        self.p_ready = p_ready
        self.p_read = p_read
        self.s_read = s_read
        self.dp_ready = dp_ready
        self.dp_read = dp_read
        self.ds_ready = ds_ready
        self.ds_free = ds_free
        self.dq_ready = dq_ready
        self.dq_free = dq_free
        self.dkv_ready = dkv_ready
        self.dkv_free = dkv_free


@aggregate
class Tmem:
    s: tensor_memory_descriptor  # [TILE, TILE] fp32: S^T
    dp: tensor_memory_descriptor  # [TILE, TILE] fp32: dP^T
    p: tensor_memory_descriptor  # [TILE, TILE] fp16: P^T (ALIAS: in the S^T columns)
    dv: tensor_memory_descriptor  # [TILE, DPAD] fp32 (DPAD: D rounded up to a power of 2)
    dk: tensor_memory_descriptor  # [TILE, DPAD] fp32
    dq: tensor_memory_descriptor  # [TILE, DPAD] fp32 (ALIAS: the dP^T columns)

    @gluon.constexpr_function
    def __init__(self, s, dp, p, dv, dk, dq):
        self.s = s
        self.dp = dp
        self.p = p
        self.dv = dv
        self.dk = dk
        self.dq = dq


@aggregate
class Kernel:
    problem: Problem
    q_desc: tma.tensor_descriptor  # Q / K / V / dO / dK / dV: [TILE, DC] boxes
    k_desc: tma.tensor_descriptor
    v_desc: tma.tensor_descriptor
    do_desc: tma.tensor_descriptor
    dk_desc: tma.tensor_descriptor
    dv_desc: tma.tensor_descriptor
    stats_desc: tma.tensor_descriptor  # [B H num_q_tiles STATS, TILE]
    dq_desc: tma.tensor_descriptor  # fp32 accumulator [B, H, S_q, D], [TILE, REDUCE] box
    smem: Smem
    bars: Bars
    tmem: Tmem

    @gluon.constexpr_function
    def __init__(
        self, problem, q_desc, k_desc, v_desc, do_desc, dk_desc, dv_desc, stats_desc,
        dq_desc, smem, bars, tmem,
    ):  # fmt: skip
        self.problem = problem
        self.q_desc = q_desc
        self.k_desc = k_desc
        self.v_desc = v_desc
        self.do_desc = do_desc
        self.dk_desc = dk_desc
        self.dv_desc = dv_desc
        self.stats_desc = stats_desc
        self.dq_desc = dq_desc
        self.smem = smem
        self.bars = bars
        self.tmem = tmem


@gluon.jit
def _wait_ready(bar, count):
    """Wait for the ``count``-th (0-based) completion of ``bar``."""
    mbarrier.wait(bar, count & 1)


@gluon.jit
def _wait_free(bar, count):
    """Producer side: wait until the consumer released the buffer for use ``count``
    (passes immediately for the first use)."""
    mbarrier.wait(bar, (count & 1) ^ 1)


@gluon.jit
def _matrix(buffer, rows: gl.constexpr, cols: gl.constexpr):
    """A [1, 1, rows, cols] TMA box in smem as the [rows, cols] MMA operand."""
    return buffer.reshape([rows, cols])


@gluon.jit
def _chunk(buffers, slot, c, nch: gl.constexpr, cols: gl.constexpr):
    """Head-dim chunk ``c`` of buffer ``slot`` as a [TILE, cols] MMA operand."""
    return _matrix(buffers.index(slot * nch + c), TILE, cols)


@gluon.constexpr_function
def _pow2(n):
    return n & (n - 1) == 0


@gluon.jit
def _whole(buffers, slot, p):
    """All NCH head-dim chunks of buffer ``slot`` as one [TILE, D] MMA operand: the
    chunks are consecutive [TILE, DC] swizzle atoms, which is how a [TILE, D] tile with a
    DC-element swizzle is laid out (one MMA of N = D instead of NCH of N = DC: the A
    operand is read from smem once, not NCH times)."""
    layout: gl.constexpr = gl.NVMMASharedLayout(p.DC * 2, 16, rank=2)
    tiles: gl.constexpr = buffers.shape[0] // p.NCH
    whole = buffers._reinterpret(buffers.dtype, [tiles, TILE, p.HEAD_DIM], layout)
    return whole.index(slot)


# --------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------


@gluon.jit
def _load_partition(k):
    if k.problem.STREAM:
        _load_stream(k)
    else:
        _load_resident(k)


@gluon.jit
def _load_resident(k):
    """K, V once per tile; Q_i (+ statistics) and dO_i per step into rotating slots."""
    p, sm, bars = k.problem, k.smem, k.bars
    NCH: gl.constexpr = p.NCH
    DC: gl.constexpr = p.DC
    box_bytes: gl.constexpr = k.k_desc.block_type.nbytes
    q_bytes: gl.constexpr = NCH * box_bytes + STATS * k.stats_desc.block_type.nbytes
    tiles = 0
    steps = 0
    for tile_id in range(gl.program_id(0), p.num_tiles, gl.num_programs(0)):
        info = p.tile(tile_id)
        kv_slot = tiles % p.KV_BUFS
        _wait_free(bars.kv_free.index(kv_slot), tiles // p.KV_BUFS)
        ready = bars.kv_ready.index(kv_slot)
        mbarrier.expect(ready, 2 * NCH * box_bytes)
        for c in gl.static_range(NCH):
            coords = [info.batch, info.head, info.k_off + info.kv_start, c * DC]
            dst = kv_slot * NCH + c
            tma.async_copy_global_to_shared(k.k_desc, coords, ready, sm.k.index(dst))
            tma.async_copy_global_to_shared(k.v_desc, coords, ready, sm.v.index(dst))
        tiles += 1  # noqa: SIM113 (Gluon has no enumerate)
        for g in range(p.group):
            q_head = p.q_head(info, g)
            q_start, q_count = p.q_range(info, g)
            for index in range(q_count):
                i = p.visit(info, q_start, q_count, index)
                row = info.q_off + i * TILE
                slot = steps % p.Q_SLOTS
                _wait_free(bars.q_free.index(slot), steps // p.Q_SLOTS)
                ready = bars.q_ready.index(slot)
                mbarrier.expect(ready, q_bytes)
                for c in gl.static_range(NCH):
                    tma.async_copy_global_to_shared(
                        k.q_desc, [info.batch, q_head, row, c * DC], ready,
                        sm.q.index(slot * NCH + c),
                    )  # fmt: skip
                stats_row = (info.batch * p.heads + q_head) * p.stats_tiles + info.qt_base + i
                stats_at = stats_row * STATS * TILE
                for which in gl.static_range(STATS):
                    tma.async_copy_global_to_shared(
                        k.stats_desc, [stats_at + which * TILE], ready,
                        sm.stats.index(STATS * slot + which),
                    )  # fmt: skip
                slot = steps % p.DO_SLOTS
                _wait_free(bars.do_free.index(slot), steps // p.DO_SLOTS)
                ready = bars.do_ready.index(slot)
                mbarrier.expect(ready, NCH * box_bytes)
                for c in gl.static_range(NCH):
                    tma.async_copy_global_to_shared(
                        k.do_desc, [info.batch, q_head, row, c * DC], ready,
                        sm.do.index(slot * NCH + c),
                    )  # fmt: skip
                steps += 1


# --------------------------------------------------------------------------------------
# MMA
# --------------------------------------------------------------------------------------


@gluon.jit
def _issue_s(k, kv_slot, step):
    """S^T(step) = K Q^T into TMEM, summed over the head-dim chunks."""
    p, sm, bars = k.problem, k.smem, k.bars
    slot = step % p.Q_SLOTS
    _wait_ready(bars.q_ready.index(slot), step // p.Q_SLOTS)
    for c in gl.static_range(p.NCH):
        q_tile = _chunk(sm.q, slot, c, p.NCH, p.DC).permute((1, 0))
        k_tile = _chunk(sm.k, kv_slot, c, p.NCH, p.DC)
        signal = [bars.s_ready.index(0)] if c == p.NCH - 1 else None
        tcgen05_mma(k_tile, q_tile, k.tmem.s, use_acc=c > 0, mbarriers=signal)


@gluon.jit
def _issue_dp(k, kv_slot, step):
    """dP^T(step) = V dO^T into TMEM, summed over the head-dim chunks."""
    p, sm, bars = k.problem, k.smem, k.bars
    slot = step % p.DO_SLOTS
    _wait_ready(bars.do_ready.index(slot), step // p.DO_SLOTS)
    for c in gl.static_range(p.NCH):
        do_tile = _chunk(sm.do, slot, c, p.NCH, p.DC).permute((1, 0))
        v_tile = _chunk(sm.v, kv_slot, c, p.NCH, p.DC)
        signal = [bars.dp_ready.index(0)] if c == p.NCH - 1 else None
        tcgen05_mma(v_tile, do_tile, k.tmem.dp, use_acc=c > 0, mbarriers=signal)


@gluon.jit
def _issue_dv(k, step, acc):
    """dV += P^T(step) dO_step; releases the dO slot (dP^T(step) came first)."""
    p, sm, tm = k.problem, k.smem, k.tmem
    slot = step % p.DO_SLOTS
    done = [k.bars.do_free.index(slot)]
    if _pow2(p.HEAD_DIM):
        tcgen05_mma(tm.p, _whole(sm.do, slot, p), tm.dv, use_acc=acc, mbarriers=done)
    else:  # TMEM blocks are powers of two: one MMA per head-dim chunk
        for c in gl.static_range(p.NCH):
            do_tile = _chunk(sm.do, slot, c, p.NCH, p.DC)
            signal = done if c == p.NCH - 1 else None
            tcgen05_mma(tm.p, do_tile, tm.dv.slice(c * p.DC, p.DC), use_acc=acc, mbarriers=signal)


@gluon.jit
def _issue_dk_dq(k, kv_slot, step, acc):
    """dK += dS^T Q_step and dQ(step) = dS K; releases dS^T and the Q slot and signals
    dQ(step)."""
    p, sm, bars, tm = k.problem, k.smem, k.bars, k.tmem
    slot = step % p.Q_SLOTS
    DC: gl.constexpr = p.DC
    done = [bars.dq_ready.index(0), bars.ds_free.index(0), bars.q_free.index(slot)]
    ds_t = sm.ds.permute((1, 0))
    if _pow2(p.HEAD_DIM):
        tcgen05_mma(sm.ds, _whole(sm.q, slot, p), tm.dk, use_acc=acc)
        dq = tm.dq.slice(0, p.HEAD_DIM)  # (ALIAS: the first D of the 128 dP^T columns)
        tcgen05_mma(ds_t, _whole(sm.k, kv_slot, p), dq, use_acc=False, mbarriers=done)
    else:  # TMEM blocks are powers of two: one MMA per head-dim chunk
        for c in gl.static_range(p.NCH):
            q_tile = _chunk(sm.q, slot, c, p.NCH, DC)
            tcgen05_mma(sm.ds, q_tile, tm.dk.slice(c * DC, DC), use_acc=acc)
        for c in gl.static_range(p.NCH):
            k_tile = _chunk(sm.k, kv_slot, c, p.NCH, DC)
            signal = done if c == p.NCH - 1 else None
            tcgen05_mma(ds_t, k_tile, tm.dq.slice(c * DC, DC), use_acc=False, mbarriers=signal)


@gluon.jit
def _mma_partition(k):
    if k.problem.STREAM:
        _mma_stream(k)
    else:
        _mma_resident(k)


@gluon.jit
def _mma_resident(k):
    """Per tile: S^T(0), dP^T(0); then per step i (see module docstring) dV(i),
    S^T(i+1) once P(i) is in TMEM, and dP^T(i+1), dK(i), dQ(i) once dS(i) is in smem.
    ALIAS reorders: S^T(i+1) also waits for the dS warpgroup to have read P^T(i), and
    dP^T(i+1) comes after dQ(i) has been read out of the dP^T columns."""
    p, bars = k.problem, k.bars
    tiles = 0
    steps = 0  # query steps of all tiles so far
    dq_count = 0
    for tile_id in range(gl.program_id(0), p.num_tiles, gl.num_programs(0)):
        info = p.tile(tile_id)
        n = p.steps_of(info)  # every query head of the K / V head
        kv_slot = tiles % p.KV_BUFS
        _wait_ready(bars.kv_ready.index(kv_slot), tiles // p.KV_BUFS)
        _issue_s(k, kv_slot, steps)
        if p.ALIAS:
            _wait_free(bars.dq_free.index(0), dq_count)  # dQ(steps - 1) read out
        _issue_dp(k, kv_slot, steps)
        for i in range(n):
            step = steps + i
            # P^T(i) ready: dV += P^T dO
            _wait_ready(bars.p_ready.index(0), step)
            if i == 0:
                _wait_free(bars.dkv_free.index(0), tiles)  # last tile's dK, dV read
            _issue_dv(k, step, i > 0)
            if p.ALIAS:
                # S^T(i + 1) overwrites P^T(i): after dV(i) in issue order (the compute
                # warpgroups keep P^T(i) in registers for dS^T(i))
                if i + 1 < n:
                    _issue_s(k, kv_slot, step + 1)
                # dS^T(i) ready (so dP^T(i) was read): dK, dQ(i) into the dP^T columns
                _wait_ready(bars.ds_ready.index(0), step)
                _wait_free(bars.dq_free.index(0), dq_count)
                _issue_dk_dq(k, kv_slot, step, i > 0)
                dq_count += 1
                if i + 1 < n:
                    _wait_free(bars.dq_free.index(0), dq_count)  # dQ(i) read out
                    _issue_dp(k, kv_slot, step + 1)
            else:
                # the S^T slot is free for S^T(i + 1)
                if i + 1 < n:
                    _issue_s(k, kv_slot, step + 1)
                # dP^T(i) read out: dP^T(i + 1) into the freed slot
                if i + 1 < n:
                    _wait_ready(bars.dp_read.index(0), step)
                    _issue_dp(k, kv_slot, step + 1)
                # dS^T(i) ready in smem: dK += dS^T Q, dQ(i) = dS K
                _wait_ready(bars.ds_ready.index(0), step)
                if i + 1 == n:
                    _wait_ready(bars.dp_read.index(0), step)  # keep the barrier count
                _wait_free(bars.dq_free.index(0), dq_count)
                _issue_dk_dq(k, kv_slot, step, i > 0)
                dq_count += 1
        tcgen05_commit(bars.dkv_ready.index(0))
        tcgen05_commit(bars.kv_free.index(kv_slot))
        steps += n
        tiles += 1  # noqa: SIM113 (Gluon has no enumerate)


# --------------------------------------------------------------------------------------
# STREAM (wide heads): output slices, operands streamed through a ring of chunks
# --------------------------------------------------------------------------------------
#
# A CTA owns one key tile and one output slice (NW head-dim chunks of dK, dV, dQ); the
# NSL slices of a key tile are separate CTAs, each recomputing S^T and dP^T over the
# whole head dim. Tensor memory is the D = 128 ALIAS layout (S^T, dP^T, dV and dK of the
# slice). K is resident (all of it, or the slice's chunks); everything else streams
# through a ring of [TILE, DC] chunks in exactly the order the MMA warp consumes it:
#
#     per tile:  [S^T(0): (K_c), Q_c]  [dP^T(0): V_c, dO_c]
#     per step:  dO_s(i)  [S^T(i+1): (K_c), Q_c]  Q_s(i)  [dP^T(i+1): V_c, dO_c]
#
# (dO_s, Q_s: the slice's chunks, loaded again for dV, dK). Every chunk is released
# by the MMA that reads it, so the ring cannot deadlock at any depth. In the smem /
# barrier aggregates: ``k`` holds the resident K, ``v`` (= ``q`` = ``do``) is the ring;
# q_ready / q_free guard the statistics slots and do_ready / do_free the ring slots.


@gluon.jit
def _put(k, desc, coords, ring):
    """Load one [TILE, DC] chunk into the next ring slot."""
    RING: gl.constexpr = k.smem.v.shape[0]
    slot = ring % RING
    _wait_free(k.bars.do_free.index(slot), ring // RING)
    ready = k.bars.do_ready.index(slot)
    mbarrier.expect(ready, desc.block_type.nbytes)
    tma.async_copy_global_to_shared(desc, coords, ready, k.smem.v.index(slot))
    return ring + 1


@gluon.jit
def _take(k, ring):
    """The ring slot of the next chunk, once loaded."""
    RING: gl.constexpr = k.smem.v.shape[0]
    slot = ring % RING
    _wait_ready(k.bars.do_ready.index(slot), ring // RING)
    return slot, ring + 1


@gluon.jit
def _free(k, slot):
    """Release a ring slot when the MMAs issued so far (the one reading it) are done."""
    tcgen05_commit(k.bars.do_free.index(slot))


@gluon.jit
def _step_of(p, info, g, start, count, index):
    """(query head, query tile) of position ``index`` of query head ``g``'s range."""
    return p.q_head(info, g), p.visit(info, start, count, index)


@gluon.jit
def _advance(p, info, g, start, count, index):
    """The step after (g, index): the next index, or the next query head's first."""
    index += 1
    while (index == count) & (g + 1 < p.group):
        g += 1
        start, count = p.q_range(info, g)
        index = 0
    return g, start, count, index


@gluon.jit
def _load_s(k, info, q_head, i, step, ring):
    """Statistics of query tile ``i`` and the chunks of S^T = K Q^T (K_c if not
    resident, Q_c)."""
    p, sm, bars = k.problem, k.smem, k.bars
    slot = step % p.Q_SLOTS
    _wait_free(bars.q_free.index(slot), step // p.Q_SLOTS)
    ready = bars.q_ready.index(slot)
    mbarrier.expect(ready, STATS * k.stats_desc.block_type.nbytes)
    stats_row = (info.batch * p.heads + q_head) * p.stats_tiles + info.qt_base + i
    for which in gl.static_range(STATS):
        tma.async_copy_global_to_shared(
            k.stats_desc, [(stats_row * STATS + which) * TILE], ready,
            sm.stats.index(STATS * slot + which),
        )  # fmt: skip
    for c in gl.static_range(p.NCH):
        if not p.K_RES:
            coords = [info.batch, info.head, info.k_off + info.kv_start, c * p.DC]
            ring = _put(k, k.k_desc, coords, ring)
        coords = [info.batch, q_head, info.q_off + i * TILE, c * p.DC]
        ring = _put(k, k.q_desc, coords, ring)
    return ring


@gluon.jit
def _load_dp(k, info, q_head, i, ring):
    """The chunks of dP^T = V dO^T (V_c, dO_c)."""
    p = k.problem
    for c in gl.static_range(p.NCH):
        coords = [info.batch, info.head, info.k_off + info.kv_start, c * p.DC]
        ring = _put(k, k.v_desc, coords, ring)
        coords = [info.batch, q_head, info.q_off + i * TILE, c * p.DC]
        ring = _put(k, k.do_desc, coords, ring)
    return ring


@gluon.jit
def _load_slice(k, desc, info, q_head, i, ring):
    """The output slice's NW chunks of Q or dO (past the head dim: zeros)."""
    p = k.problem
    for c in gl.static_range(p.NW):
        coords = [info.batch, q_head, info.q_off + i * TILE, (info.chunk0 + c) * p.DC]
        ring = _put(k, desc, coords, ring)
    return ring


@gluon.jit
def _load_stream(k):
    """STREAM loader: resident K once per tile, then the ring in consumption order."""
    p, sm, bars = k.problem, k.smem, k.bars
    K_CH: gl.constexpr = sm.k.shape[0]
    tiles = 0
    steps = 0
    ring = 0
    for tile_id in range(gl.program_id(0), p.num_tiles, gl.num_programs(0)):
        info = p.tile(tile_id)
        _wait_free(bars.kv_free.index(0), tiles)
        ready = bars.kv_ready.index(0)
        mbarrier.expect(ready, K_CH * k.k_desc.block_type.nbytes)
        first = 0 if p.K_RES else info.chunk0
        for c in gl.static_range(K_CH):
            coords = [info.batch, info.head, info.k_off + info.kv_start, (first + c) * p.DC]
            tma.async_copy_global_to_shared(k.k_desc, coords, ready, sm.k.index(c))
        tiles += 1  # noqa: SIM113 (Gluon has no enumerate)
        n = p.steps_of(info)
        g = info.kv_start * 0
        start, count = p.q_range(info, 0)
        index = g
        while (index == count) & (g + 1 < p.group):  # (MASK: heads with no listed tile)
            g += 1
            start, count = p.q_range(info, g)
        q_head, i = _step_of(p, info, g, start, count, index)
        if n > 0:
            ring = _load_s(k, info, q_head, i, steps, ring)
            ring = _load_dp(k, info, q_head, i, ring)
        for t in range(n):
            g, start, count, index = _advance(p, info, g, start, count, index)
            next_head, next_i = _step_of(p, info, g, start, count, index)
            ring = _load_slice(k, k.do_desc, info, q_head, i, ring)  # dO_s(t)
            if t + 1 < n:
                ring = _load_s(k, info, next_head, next_i, steps + 1, ring)
            ring = _load_slice(k, k.q_desc, info, q_head, i, ring)  # Q_s(t)
            if t + 1 < n:
                ring = _load_dp(k, info, next_head, next_i, ring)
            q_head = next_head
            i = next_i
            steps += 1


@gluon.jit
def _mma_s(k, step, ring):
    """S^T(step) = sum_c K_c Q_c^T."""
    p, sm, bars = k.problem, k.smem, k.bars
    DC: gl.constexpr = p.DC
    slot = step % p.Q_SLOTS
    _wait_ready(bars.q_ready.index(slot), step // p.Q_SLOTS)  # statistics, for the P warps
    for c in gl.static_range(p.NCH):
        if p.K_RES:
            k_tile = _matrix(sm.k.index(c), TILE, DC)
        else:
            k_slot, ring = _take(k, ring)
            k_tile = _matrix(sm.v.index(k_slot), TILE, DC)
        q_slot, ring = _take(k, ring)
        q_tile = _matrix(sm.v.index(q_slot), TILE, DC).permute((1, 0))
        signal = [bars.s_ready.index(0)] if c == p.NCH - 1 else None
        tcgen05_mma(k_tile, q_tile, k.tmem.s, use_acc=c > 0, mbarriers=signal)
        _free(k, q_slot)
        if not p.K_RES:
            _free(k, k_slot)
    return ring


@gluon.jit
def _mma_dp(k, ring):
    """dP^T = sum_c V_c dO_c^T."""
    p, sm, bars = k.problem, k.smem, k.bars
    DC: gl.constexpr = p.DC
    for c in gl.static_range(p.NCH):
        v_slot, ring = _take(k, ring)
        do_slot, ring = _take(k, ring)
        v_tile = _matrix(sm.v.index(v_slot), TILE, DC)
        do_tile = _matrix(sm.v.index(do_slot), TILE, DC).permute((1, 0))
        signal = [bars.dp_ready.index(0)] if c == p.NCH - 1 else None
        tcgen05_mma(v_tile, do_tile, k.tmem.dp, use_acc=c > 0, mbarriers=signal)
        _free(k, v_slot)
        _free(k, do_slot)
    return ring


@gluon.jit
def _mma_dv(k, ring, acc):
    """dV_s += P^T dO_s, chunk by chunk."""
    p, sm, tm = k.problem, k.smem, k.tmem
    DC: gl.constexpr = p.DC
    for c in gl.static_range(p.NW):
        slot, ring = _take(k, ring)
        tcgen05_mma(tm.p, _matrix(sm.v.index(slot), TILE, DC), tm.dv.slice(c * DC, DC), use_acc=acc)
        _free(k, slot)
    return ring


@gluon.jit
def _mma_dk_dq(k, info, ring, step, acc):
    """dK_s += dS^T Q_s and dQ_s(step) = dS K_s; releases dS^T and the statistics slot
    and signals dQ_s(step)."""
    p, sm, bars, tm = k.problem, k.smem, k.bars, k.tmem
    DC: gl.constexpr = p.DC
    for c in gl.static_range(p.NW):
        slot, ring = _take(k, ring)
        tcgen05_mma(sm.ds, _matrix(sm.v.index(slot), TILE, DC), tm.dk.slice(c * DC, DC), use_acc=acc)
        _free(k, slot)
    done = [bars.dq_ready.index(0), bars.ds_free.index(0), bars.q_free.index(step % p.Q_SLOTS)]
    for c in gl.static_range(p.NW):
        if p.K_RES:
            k_tile = _matrix(sm.k.index(info.chunk0 + c), TILE, DC)
        else:
            k_tile = _matrix(sm.k.index(c), TILE, DC)
        signal = done if c == p.NW - 1 else None
        tcgen05_mma(sm.ds.permute((1, 0)), k_tile, tm.dq.slice(c * DC, DC), use_acc=False,
                    mbarriers=signal)  # fmt: skip
    return ring


@gluon.jit
def _mma_stream(k):
    """STREAM MMA issuer: the ALIAS order of _mma_resident with streamed operands."""
    p, bars = k.problem, k.bars
    tiles = 0
    steps = 0
    ring = 0
    dq_count = 0
    for tile_id in range(gl.program_id(0), p.num_tiles, gl.num_programs(0)):
        info = p.tile(tile_id)
        n = p.steps_of(info)
        _wait_ready(bars.kv_ready.index(0), tiles)
        if n > 0:
            ring = _mma_s(k, steps, ring)
            _wait_free(bars.dq_free.index(0), dq_count)  # dQ(steps - 1) read out
            ring = _mma_dp(k, ring)
        for t in range(n):
            step = steps + t
            _wait_ready(bars.p_ready.index(0), step)
            if t == 0:
                _wait_free(bars.dkv_free.index(0), tiles)  # last tile's dK, dV read
            ring = _mma_dv(k, ring, t > 0)
            if t + 1 < n:
                ring = _mma_s(k, step + 1, ring)
            _wait_ready(bars.ds_ready.index(0), step)
            _wait_free(bars.dq_free.index(0), dq_count)
            ring = _mma_dk_dq(k, info, ring, step, t > 0)
            dq_count += 1
            if t + 1 < n:
                _wait_free(bars.dq_free.index(0), dq_count)  # dQ(t) read out
                ring = _mma_dp(k, ring)
        tcgen05_commit(bars.dkv_ready.index(0))
        tcgen05_commit(bars.kv_free.index(0))
        steps += n
        tiles += 1  # noqa: SIM113 (Gluon has no enumerate)


# --------------------------------------------------------------------------------------
# Compute: P^T and dS^T
# --------------------------------------------------------------------------------------


@gluon.jit
def _p_partition(k, HALF: gl.constexpr):
    """P warpgroup: per step, P^T = exp2(qk_scale (S^T - m) - log2 l) for all 128 query
    columns -> TMEM (fp16): the forward's shift-then-scale (S - m exact near the row
    max, whatever the scores' magnitude), then log2 l in the same fma. Thread t owns key
    row t. It runs a step ahead of the dS warpgroup: P^T(i + 1) is written once the dS
    warpgroup has read P^T(i)."""
    p = k.problem
    chunk_tmem: gl.constexpr = TensorMemoryLayout([TILE, CHUNK], col_stride=1)
    regs: gl.constexpr = get_tmem_reg_layout(gl.float32, [TILE, CHUNK], chunk_tmem, gl.num_warps())
    row_layout: gl.constexpr = gl.SliceLayout(1, regs)
    steps = 0
    for tile_id in range(gl.program_id(0), p.num_tiles, gl.num_programs(0)):
        info = p.tile(tile_id)
        # key c is visible to query r iff r >= first query of c's frame
        keys = info.kv_start + gl.arange(0, TILE, layout=row_layout)
        if p.CAUSAL:
            first_query = (keys // p.block) * p.block
        else:
            first_query = keys * 0
        # padded keys (past S_kv, zero K / V rows) are visible to no query: their score
        # is 0, so P = exp2(-c m - log2 l) overflows for rows whose scores are all
        # negative, and dQ += dS K would be inf * 0. The key tile holding them runs
        # every step masked; full tiles keep the unmasked path.
        key_ok = keys < info.seq_kv
        first_query = gl.where(key_ok, first_query, 2**30)
        for g in range(p.group):  # the same keys for every query head of the group
            if p.MASK:  # the key rows' words for query head g (padded keys: key_ok)
                start = info.batch.to(gl.int64) * p.mask_b_stride
                start += p.q_head(info, g).to(gl.int64) * p.mask_h_stride
                rows = gl.minimum(keys, info.seq_kv - 1).to(gl.int64)
                words = p.mask_ptr + start + rows * p.mask_words
                q_start, q_count = p.q_range(info, g)
                for index in range(q_count):  # listed tiles: hidden ones are skipped
                    entry = (index + info.kv_start // TILE) % q_count  # as in p.visit
                    i = p.q_tile(q_start, entry)
                    if p.q_full(q_start, entry):
                        steps = _p_step(k, words, key_ok, i, steps, regs, False, HALF)
                    else:
                        steps = _p_step(k, words, key_ok, i, steps, regs, True, HALF)
            else:
                masked = info.first_full - info.first_q
                count = info.end_q - info.first_q
                for index in range(masked):
                    i = info.first_q + index
                    steps = _p_step(k, first_query, key_ok, i, steps, regs, True, HALF)
                for index in range(masked, count):
                    i = p.visit(info, info.first_q, count, index)
                    steps = _p_step(k, first_query, key_ok, i, steps, regs, False, HALF)


@gluon.jit
def _p_step(
    k, first_query, key_ok, i, steps, regs: gl.constexpr, MASKED: gl.constexpr,
    HALF: gl.constexpr,
):  # fmt: skip
    """P^T of query tile ``i`` (the CTA's ``steps``-th step) -> TMEM: all 128 query
    columns (HALF = -1), or columns 64 HALF .. + 63, and then dS^T of those columns from
    the P^T still in registers (ALIAS: both compute warpgroups run this)."""
    p, sm, bars, tm = k.problem, k.smem, k.bars, k.tmem
    FIRST: gl.constexpr = 0 if HALF < 0 else HALF * (TILE // CHUNK // 2)
    CHUNKS: gl.constexpr = TILE // CHUNK if HALF < 0 else TILE // CHUNK // 2
    col_layout: gl.constexpr = gl.SliceLayout(0, regs)
    offsets = gl.arange(0, CHUNK, layout=col_layout)
    neg_m = sm.stats.index(STATS * (steps % p.Q_SLOTS))
    neg_log_l = sm.stats.index(STATS * (steps % p.Q_SLOTS) + 1)
    _wait_ready(bars.s_ready.index(0), steps)
    probs = ()
    for c in gl.static_range(FIRST, FIRST + CHUNKS):
        s2 = float2.pack(tm.s.slice(c * CHUNK, CHUNK).load(regs), axis=1)
        m = neg_m.slice(c * CHUNK, CHUNK).load(col_layout)
        m2 = float2.pack(m[None, :].broadcast_to([TILE, CHUNK]), axis=1)
        log_l = neg_log_l.slice(c * CHUNK, CHUNK).load(col_layout)
        log_l2 = float2.pack(log_l[None, :].broadcast_to([TILE, CHUNK]), axis=1)
        scale2 = float2.full_like(s2, p.qk_scale)
        x2 = float2.fma(s2 + m2, scale2, log_l2)
        prob = gl.exp2(float2.unpack(x2, axis=1))
        if MASKED:
            if p.MASK:  # CHUNK = 32 queries: one word per key row
                word = gl.load(first_query + (i * (TILE // CHUNK) + c))
                visible = (((word[:, None] >> offsets[None, :]) & 1) != 0) & key_ok[:, None]
            else:
                queries = i * TILE + c * CHUNK + offsets
                visible = queries[None, :] >= first_query[:, None]
            prob = gl.where(visible, prob, 0.0)
        probs = probs + (prob.to(k.q_desc.block_type.element_ty),)
    if HALF < 0:
        _wait_free(bars.p_read.index(0), steps)  # the dS warpgroup read P^T(i - 1)
        tm.p.store(_join_columns(probs))
        mbarrier.arrive(bars.p_ready.index(0), count=1)
    else:
        # 16-bit P^T packs two columns per TMEM column: half 1's P^T lands on S^T
        # columns 32 .. 63, which half 0 reads, so half 1 stores once half 0 has read
        if HALF == 0:
            mbarrier.arrive(bars.s_read.index(0), count=1)
        else:
            _wait_ready(bars.s_read.index(0), steps)
        tm.p.slice(FIRST * CHUNK, CHUNKS * CHUNK).store(_join_columns(probs))
        mbarrier.arrive(bars.p_ready.index(0), count=1)  # (count 2: both halves)
        _ds_half(k, probs, steps, regs, FIRST, CHUNKS)
    return steps + 1


@gluon.jit
def _ds_half(k, probs, steps, regs: gl.constexpr, FIRST: gl.constexpr, CHUNKS: gl.constexpr):
    """ALIAS: dS^T = P^T (dP^T - delta) for query columns of chunks FIRST .. + CHUNKS - 1,
    with P^T from registers -> smem."""
    p, sm, bars, tm = k.problem, k.smem, k.bars, k.tmem
    col_layout: gl.constexpr = gl.SliceLayout(0, regs)
    neg_delta = sm.stats.index(STATS * (steps % p.Q_SLOTS) + 2)
    _wait_ready(bars.dp_ready.index(0), steps)
    _wait_free(bars.ds_free.index(0), steps)
    for c in gl.static_range(FIRST, FIRST + CHUNKS):
        dp = tm.dp.slice(c * CHUNK, CHUNK).load(regs)
        delta = neg_delta.slice(c * CHUNK, CHUNK).load(col_layout)
        delta2 = float2.pack(delta[None, :].broadcast_to([TILE, CHUNK]), axis=1)
        prob2 = float2.pack(probs[c - FIRST].to(gl.float32), axis=1)
        ds2 = (float2.pack(dp, axis=1) + delta2) * prob2
        ds = float2.unpack(ds2, axis=1).to(k.q_desc.block_type.element_ty)
        sm.ds.slice(c * CHUNK, CHUNK, dim=1).store(ds)
    fence_async_shared()
    mbarrier.arrive(bars.ds_ready.index(0), count=1)  # (count 2: both halves)


@gluon.jit
def _ds_partition(k):
    """dS warpgroup (without ALIAS): per step, dS^T = P^T (dP^T - delta) for all 128 query
    columns -> smem (fp16, the A operand of the dK and dQ MMAs). It reads all of dP^T
    first: its TMEM slot is then free for dP^T(i + 1)."""
    p, sm, bars, tm = k.problem, k.smem, k.bars, k.tmem
    NUM_CHUNKS: gl.constexpr = TILE // CHUNK
    chunk_tmem: gl.constexpr = TensorMemoryLayout([TILE, CHUNK], col_stride=1)
    regs: gl.constexpr = get_tmem_reg_layout(gl.float32, [TILE, CHUNK], chunk_tmem, gl.num_warps())
    p_regs: gl.constexpr = get_tmem_reg_layout(k.q_desc.block_type.element_ty, [TILE, CHUNK], chunk_tmem, gl.num_warps())
    col_layout: gl.constexpr = gl.SliceLayout(0, regs)
    steps = 0
    for tile_id in range(gl.program_id(0), p.num_tiles, gl.num_programs(0)):
        info = p.tile(tile_id)
        for _ in range(p.steps_of(info)):  # every query head's steps
            neg_delta = sm.stats.index(STATS * (steps % p.Q_SLOTS) + 2)
            _wait_ready(bars.dp_ready.index(0), steps)
            dps = ()
            for c in gl.static_range(NUM_CHUNKS):
                dps = dps + (tm.dp.slice(c * CHUNK, CHUNK).load(regs),)
            mbarrier.arrive(bars.dp_read.index(0), count=1)
            _wait_ready(bars.p_ready.index(0), steps)
            _wait_free(bars.ds_free.index(0), steps)
            for c in gl.static_range(NUM_CHUNKS):
                prob = tm.p.slice(c * CHUNK, CHUNK).load(p_regs)
                prob = gl.convert_layout(prob.to(gl.float32), regs)
                delta = neg_delta.slice(c * CHUNK, CHUNK).load(col_layout)
                delta2 = float2.pack(delta[None, :].broadcast_to([TILE, CHUNK]), axis=1)
                ds2 = (float2.pack(dps[c], axis=1) + delta2) * float2.pack(prob, axis=1)
                ds = float2.unpack(ds2, axis=1).to(k.q_desc.block_type.element_ty)
                sm.ds.slice(c * CHUNK, CHUNK, dim=1).store(ds)
            mbarrier.arrive(bars.p_read.index(0), count=1)
            fence_async_shared()
            mbarrier.arrive(bars.ds_ready.index(0), count=1)
            steps += 1


# --------------------------------------------------------------------------------------
# Reduce: dQ atomics, dK / dV epilogue
# --------------------------------------------------------------------------------------


@gluon.jit
def _reduce_partition(k):
    """Default partition. Per step: dQ(i) TMEM -> registers -> smem (fp32) -> TMA
    reduce-add into the dQ accumulator ([TILE, REDUCE] boxes). Per tile: dV and dK
    (scaled) -> fp16 -> TMA store per head-dim chunk, staged in the same smem."""
    p, sm, bars, tm = k.problem, k.smem, k.bars, k.tmem
    DC: gl.constexpr = p.DC
    RED: gl.constexpr = k.dq_desc.block_type.shape[3]  # dQ columns per reduce-add
    red_tmem: gl.constexpr = TensorMemoryLayout([TILE, RED], col_stride=1)
    red_regs: gl.constexpr = get_tmem_reg_layout(gl.float32, [TILE, RED], red_tmem, gl.num_warps())
    acc_tmem: gl.constexpr = TensorMemoryLayout([TILE, DC], col_stride=1)
    regs: gl.constexpr = get_tmem_reg_layout(gl.float32, [TILE, DC], acc_tmem, gl.num_warps())
    # the staging buffer seen as one fp16 [TILE, DC] box for the dK / dV stores
    out16 = sm.out.index(0)._reinterpret(k.dk_desc.block_type.element_ty, [1, 1, TILE, DC], k.dk_desc.layout)
    tiles = 0
    dq_count = 0
    reduces = 0  # TMA reduce-adds issued (staging ring position)
    scale = p.qk_scale / 1.4426950408889634
    for tile_id in range(gl.program_id(0), p.num_tiles, gl.num_programs(0)):
        info = p.tile(tile_id)
        for g in range(p.group):
            q_head = p.q_head(info, g)
            q_start, q_count = p.q_range(info, g)
            for index in range(q_count):
                i = p.visit(info, q_start, q_count, index)
                _wait_ready(bars.dq_ready.index(0), dq_count)
                row = info.q_off + i * TILE
                # all of dQ(i) to registers first: its TMEM (under ALIAS the dP^T
                # columns) is free for the next step at once
                parts = ()
                for c in gl.static_range(p.NW):
                    for r in gl.static_range(DC // RED):
                        parts = parts + (tm.dq.slice(c * DC + r * RED, RED).load(red_regs),)
                mbarrier.arrive(bars.dq_free.index(0), count=1)
                for c in gl.static_range(p.NW):
                    if c < info.nw:  # (the last slice may be narrower)
                        for r in gl.static_range(DC // RED):
                            column = (info.chunk0 + c) * DC + r * RED
                            coords = [info.batch, q_head, row, column]
                            _stage_reduce(k, parts[c * (DC // RED) + r], coords, reduces)
                            reduces += 1
                dq_count += 1

        _wait_ready(bars.dkv_ready.index(0), tiles)
        tiles += 1  # noqa: SIM113 (Gluon has no enumerate)
        for c in gl.static_range(p.NW):
            dv = tm.dv.slice(c * DC, DC).load(regs)
            dk = tm.dk.slice(c * DC, DC).load(regs) * scale
            if c == p.NW - 1:
                mbarrier.arrive(bars.dkv_free.index(0), count=1)
            column = (info.chunk0 + c) * DC
            if c >= info.nw:
                pass  # past the end of a narrower last slice
            elif p.VARLEN and info.kv_start + TILE > info.seq_kv:
                # the sequence ends inside this key tile: a full TMA box would overwrite
                # the next sequence's rows, so store row-masked from registers
                keys = info.kv_start + gl.arange(0, TILE, layout=gl.SliceLayout(1, regs))
                cols = column + gl.arange(0, DC, layout=gl.SliceLayout(0, regs))
                packed = (info.k_off + keys).to(gl.int64) * p.kv_row_stride
                offsets = packed[:, None] + (info.head * p.HEAD_DIM + cols)[None, :]
                ok = (keys < info.seq_kv)[:, None]
                gl.store(p.dv_ptr + offsets, dv.to(p.dv_ptr.dtype.element_ty), mask=ok)
                gl.store(p.dk_ptr + offsets, dk.to(p.dk_ptr.dtype.element_ty), mask=ok)
            else:
                coords = [info.batch, info.head, info.k_off + info.kv_start, column]
                tma.store_wait(pendings=0)
                _matrix(out16, TILE, DC).store(dv.to(k.dk_desc.block_type.element_ty))
                fence_async_shared()
                tma.async_copy_shared_to_global(k.dv_desc, coords, out16)
                tma.store_wait(pendings=0)
                _matrix(out16, TILE, DC).store(dk.to(k.dk_desc.block_type.element_ty))
                fence_async_shared()
                tma.async_copy_shared_to_global(k.dk_desc, coords, out16)
    tma.store_wait(pendings=0)


@gluon.jit
def _stage_reduce(k, values, coords, count):
    """global[coords box] += values through the ``count``-th staging buffer of the ring
    (TMA reduce-add); up to RSTAGES - 1 earlier reduce-adds stay in flight."""
    RSTAGES: gl.constexpr = k.smem.out.shape[0]
    staging = k.smem.out.index(count % RSTAGES)
    tma.store_wait(pendings=RSTAGES - 1)  # that buffer's previous TMA op read it
    _matrix(staging, TILE, k.dq_desc.block_type.shape[3]).store(values)
    fence_async_shared()
    _tma_reduce_add(k.dq_desc, coords, staging)


@builtin
def _tma_reduce_add(tensor_desc, coord, src, _semantic=None):
    """TMA bulk reduce: global[box at coord] += smem ``src`` (cp.reduce.async.bulk
    .tensor .add). Completion is tracked like a TMA store (``tma.store_wait``)."""
    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    _semantic.builder.create_async_tma_reduce(ir.DESCRIPTOR_REDUCE_KIND.ADD, tensor_desc.handle, coord, src.handle)


# --------------------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------------------


@gluon.jit
def _barriers(n: gl.constexpr, count: gl.constexpr = 1):
    bars = gl.allocate_shared_memory(gl.int64, [n, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(n):
        mbarrier.init(bars.index(i), count=count)
    return bars


@gluon.jit
def attention_bwd_kernel(
    q_desc, k_desc, v_desc, do_desc, dk_desc, dv_desc, stats_desc, dq_desc, qk_scale,
    heads, group, seq_q, seq_kv, block, num_kv_tiles, num_tiles,
    mask_ptr, mask_b_stride, mask_h_stride, mask_words, stats_tiles, tiles_ptr, cu_q_ptr,
    cu_k_ptr, cu_qt_ptr, dk_ptr, dv_ptr, kv_row_stride, list_start_ptr, list_count_ptr,
    list_ptr, list_cls_ptr, k_tiles, list_rows,
    DC: gl.constexpr, NCH: gl.constexpr, ALIAS: gl.constexpr, KV_BUFS: gl.constexpr,
    Q_SLOTS: gl.constexpr, DO_SLOTS: gl.constexpr, P_REGS: gl.constexpr,
    DS_REGS: gl.constexpr, SMALL_REGS: gl.constexpr, RSTAGES: gl.constexpr, CAUSAL: gl.constexpr,
    MASK: gl.constexpr, VARLEN: gl.constexpr, STREAM: gl.constexpr, NSL: gl.constexpr,
    NW: gl.constexpr, K_RES: gl.constexpr,
):  # fmt: skip
    problem = Problem(
        gl.to_tensor(qk_scale), gl.to_tensor(heads), gl.to_tensor(group),
        gl.to_tensor(seq_q),
        gl.to_tensor(seq_kv), gl.to_tensor(block), gl.to_tensor(num_kv_tiles),
        gl.to_tensor(num_tiles), mask_ptr, gl.to_tensor(mask_b_stride),
        gl.to_tensor(mask_h_stride), gl.to_tensor(mask_words), gl.to_tensor(stats_tiles),
        tiles_ptr, cu_q_ptr, cu_k_ptr, cu_qt_ptr, dk_ptr, dv_ptr, gl.to_tensor(kv_row_stride),
        list_start_ptr, list_count_ptr, list_ptr, list_cls_ptr, gl.to_tensor(k_tiles),
        gl.to_tensor(list_rows),
        DC, NCH, ALIAS, KV_BUFS, Q_SLOTS, DO_SLOTS, CAUSAL, MASK, VARLEN, STREAM, NSL, NW,
        K_RES,
    )  # fmt: skip
    # head-dim chunks are separate buffers: [slot * NCH + chunk, 1, 1, TILE, DC]
    box: gl.constexpr = [1, 1, TILE, DC]
    # 64-byte swizzle: compute threads store dS^T in 32-column slices
    ds_layout: gl.constexpr = gl.NVMMASharedLayout(64, 16, rank=2)
    dtype: gl.constexpr = q_desc.block_type.element_ty  # fp16 or bf16 operands
    if STREAM:  # resident K (all, or the slice's chunks) and the chunk ring
        k_res = gl.allocate_shared_memory(dtype, [NCH if K_RES else NW] + box, k_desc.layout)
        ring = gl.allocate_shared_memory(dtype, [DO_SLOTS] + box, v_desc.layout)
        smem_k, smem_v, smem_q, smem_do = k_res, ring, ring, ring
    else:
        smem_k = gl.allocate_shared_memory(dtype, [KV_BUFS * NCH] + box, k_desc.layout)
        smem_v = gl.allocate_shared_memory(dtype, [KV_BUFS * NCH] + box, v_desc.layout)
        smem_q = gl.allocate_shared_memory(dtype, [Q_SLOTS * NCH] + box, q_desc.layout)
        smem_do = gl.allocate_shared_memory(dtype, [DO_SLOTS * NCH] + box, do_desc.layout)
    smem = Smem(
        smem_k,
        smem_v,
        smem_q,
        smem_do,
        gl.allocate_shared_memory(dtype, [TILE, TILE], ds_layout),
        gl.allocate_shared_memory(gl.float32, [RSTAGES] + dq_desc.block_type.shape, dq_desc.layout),
        gl.allocate_shared_memory(gl.float32, [STATS * Q_SLOTS, TILE], stats_desc.layout),
    )
    halves: gl.constexpr = 2 if ALIAS else 1  # warpgroups producing P^T and dS^T
    bars = Bars(
        _barriers(KV_BUFS), _barriers(KV_BUFS), _barriers(Q_SLOTS), _barriers(Q_SLOTS),
        _barriers(DO_SLOTS), _barriers(DO_SLOTS), _barriers(1), _barriers(1, halves),
        _barriers(1), _barriers(1), _barriers(1), _barriers(1), _barriers(1, halves),
        _barriers(1), _barriers(1), _barriers(1), _barriers(1), _barriers(1),
    )  # fmt: skip
    fence_async_shared()
    scores: gl.constexpr = TensorMemoryLayout([TILE, TILE], col_stride=1)
    DPAD: gl.constexpr = triton.next_power_of_2(DC * NW)  # TMEM blocks: powers of two
    acc: gl.constexpr = TensorMemoryLayout([TILE, DPAD], col_stride=1)
    s_tmem = allocate_tensor_memory(gl.float32, [TILE, TILE], scores)
    dp_tmem = allocate_tensor_memory(gl.float32, [TILE, TILE], scores)
    if ALIAS:
        # P^T (16-bit, two per column) in the first half of the S^T columns; dQ's NCH
        # [TILE, DC] blocks in the dP^T columns
        p_tmem = s_tmem.slice(0, TILE // 2)._reinterpret(dtype, [TILE, TILE], scores)
        dq_tmem = dp_tmem  # D (STREAM: the slice) <= 128
    else:
        p_tmem = allocate_tensor_memory(dtype, [TILE, TILE], scores)
        dq_tmem = allocate_tensor_memory(gl.float32, [TILE, DPAD], acc)
    tmem = Tmem(
        s_tmem, dp_tmem, p_tmem,
        allocate_tensor_memory(gl.float32, [TILE, DPAD], acc),
        allocate_tensor_memory(gl.float32, [TILE, DPAD], acc),
        dq_tmem,
    )  # fmt: skip
    k = Kernel(
        problem, q_desc, k_desc, v_desc, do_desc, dk_desc, dv_desc, stats_desc, dq_desc,
        smem, bars, tmem,
    )  # fmt: skip
    if ALIAS:  # two compute warpgroups, each P^T then dS^T of 64 query columns
        gl.warp_specialize(
            [
                (_reduce_partition, (k,)),  # default partition: 4 warps
                (_p_partition, (k, 0)),
                (_p_partition, (k, 1)),
                (_mma_partition, (k,)),
                (_load_partition, (k,)),
            ],
            [4, 4, 1, 1],
            [P_REGS, DS_REGS, SMALL_REGS, SMALL_REGS],  # the reduce warpgroup gets the rest
        )
    else:  # a P warpgroup running a step ahead of a dS warpgroup
        gl.warp_specialize(
            [
                (_reduce_partition, (k,)),
                (_p_partition, (k, -1)),
                (_ds_partition, (k,)),
                (_mma_partition, (k,)),
                (_load_partition, (k,)),
            ],
            [4, 4, 1, 1],
            [P_REGS, DS_REGS, SMALL_REGS, SMALL_REGS],
        )


# --------------------------------------------------------------------------------------
# Small kernels around the main one
# --------------------------------------------------------------------------------------


@triton.jit
def _prepare_kernel(
    out_ptr, dout_ptr, lse_ptr, stats_ptr, dq_ptr, seq, heads, num_q_tiles,
    stride_ob, stride_oh, stride_os, stride_db, stride_dh, stride_ds,
    D: tl.constexpr, D_POW2: tl.constexpr, TILE_Q: tl.constexpr,
):  # fmt: skip
    """One query tile of one (batch, head): stats[bh, tile] = (-m, -log2 l, -delta) with
    delta = rowsum(dout * out) (rows past seq: -m = -inf, so P = 0); zeroes the tile's
    rows of the fp32 dQ accumulator."""
    bh = tl.program_id(0) // num_q_tiles
    tile = tl.program_id(0) % num_q_tiles
    b = bh // heads
    h = bh % heads
    s = tile * TILE_Q + tl.arange(0, TILE_Q)
    valid = s < seq
    cols = tl.arange(0, D_POW2)
    both = valid[:, None] & (cols < D)[None, :]
    o_off = b * stride_ob + h * stride_oh + s * stride_os
    d_off = b * stride_db + h * stride_dh + s * stride_ds
    o = tl.load(out_ptr + o_off[:, None] + cols[None, :], mask=both, other=0.0)
    do = tl.load(dout_ptr + d_off[:, None] + cols[None, :], mask=both, other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    row_stats = lse_ptr + bh.to(tl.int64) * 2 * seq + s
    m = tl.load(row_stats, mask=valid, other=float("inf"))
    log_l = tl.load(row_stats + seq, mask=valid, other=0.0)
    stats = stats_ptr + (bh.to(tl.int64) * num_q_tiles + tile) * 3 * TILE_Q
    tl.store(stats + tl.arange(0, TILE_Q), -m)
    tl.store(stats + TILE_Q + tl.arange(0, TILE_Q), -log_l)
    tl.store(stats + 2 * TILE_Q + tl.arange(0, TILE_Q), -delta)
    dq_rows = dq_ptr + (bh.to(tl.int64) * seq + s[:, None]) * D + cols[None, :]
    tl.store(dq_rows, tl.zeros([TILE_Q, D_POW2], dtype=tl.float32), mask=both)


@triton.jit
def _dq_convert_kernel(
    acc_ptr, dq_ptr, rows, seq, heads, scale,
    stride_b, stride_h, stride_s, D: tl.constexpr, D_POW2: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """dq = scale * accumulator, fp16, into the [B, H, S, D] view (any strides)."""
    pid = tl.program_id(0)
    row = pid * BLOCK + tl.arange(0, BLOCK)
    valid = row < rows
    s = row % seq
    bh = row // seq
    b = bh // heads
    h = bh % heads
    cols = tl.arange(0, D_POW2)
    both = valid[:, None] & (cols < D)[None, :]
    acc = tl.load(acc_ptr + row[:, None].to(tl.int64) * D + cols[None, :], mask=both)
    off = b * stride_b + h * stride_h + s * stride_s
    tl.store(dq_ptr + off[:, None] + cols[None, :], (acc * scale).to(dq_ptr.dtype.element_ty), mask=both)


def attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    block: int,
    mask_words_t: torch.Tensor | None = None,
    mask_lists_t: tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients of ``attention_forward``: dq, dk, dv as [B, H, S, D] views of
    [B, S, H, D] storage. Extra memory: (-m, -log2 l, -delta) per row (fp32) and the fp32 dQ
    accumulator ([B, H, Sq, D])."""
    batch, heads, seq_q, dim = q.shape
    seq_kv = k.shape[2]
    device = q.device
    if dout.stride(-1) != 1 or (dout.stride(2) * 2) % 16:
        dout = dout.contiguous()

    rows = batch * heads * seq_q
    num_q_tiles = triton.cdiv(seq_q, TILE_ROWS)
    stats = torch.empty(batch * heads * num_q_tiles * 3 * TILE_ROWS, device=device, dtype=torch.float32)
    dq_acc = torch.empty(batch, heads, seq_q, dim, device=device, dtype=torch.float32)
    _prepare_kernel[(batch * heads * num_q_tiles,)](
        out, dout, lse, stats, dq_acc, seq_q, heads, num_q_tiles,
        out.stride(0), out.stride(1), out.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        D=dim, D_POW2=triton.next_power_of_2(dim), TILE_Q=TILE_ROWS,
    )  # fmt: skip
    stats_layout = gl.NVMMASharedLayout(0, 32, rank=1)
    stats_desc = TensorDescriptor.from_tensor(stats, [TILE_ROWS], stats_layout)
    cfg = backward_config(dim, q.element_size(), block > 0, max(seq_q, seq_kv))
    dq_desc = _dq_descriptor(dq_acc, cfg.reduce_cols)

    # Keys no query sees (block-causal with Sq < Skv: key c is first seen by query
    # (c // P) * P) get zero dK / dV; only key tiles some query sees are launched (a
    # tile with no query step would never signal its dK / dV).
    seen_kv = seq_kv if block == 0 else min(seq_kv, ((seq_q - 1) // block + 1) * block)
    if mask_words_t is not None:  # every key tile runs (unseen keys get P = 0, dK = dV = 0)
        seen_kv = seq_kv
    alloc = torch.empty if seen_kv == seq_kv else torch.zeros
    kv_heads = k.shape[1]  # GQA: dK / dV per K / V head, summed over its query heads
    dk = alloc(batch, seq_kv, kv_heads, dim, device=device, dtype=q.dtype)
    dv = alloc(batch, seq_kv, kv_heads, dim, device=device, dtype=q.dtype)
    dk, dv = dk.transpose(1, 2), dv.transpose(1, 2)
    num_kv_tiles = triton.cdiv(seen_kv, TILE_ROWS)
    num_tiles = batch * kv_heads * num_kv_tiles * cfg.nsl
    grid = (min(num_tiles, sm_count(device)),)
    attention_bwd_kernel[grid](
        *(tma_descriptor(t, TILE_ROWS, cfg.dc) for t in (q, k, v, dout, dk, dv)),
        stats_desc, dq_desc, scale * LOG2E, heads, heads // kv_heads, seq_q, seq_kv,
        max(block, 1),
        num_kv_tiles, num_tiles, *mask_arguments(mask_words_t, device), num_q_tiles,
        *_no_varlen(device), *(mask_lists_t or _no_lists(device)),
        **_constexprs(cfg), CAUSAL=block > 0, MASK=mask_words_t is not None, VARLEN=False,
        num_warps=4,
    )  # fmt: skip

    dq = torch.empty(batch, seq_q, heads, dim, device=device, dtype=q.dtype)
    dq = dq.transpose(1, 2)
    _dq_convert_kernel[(triton.cdiv(rows, 64),)](
        dq_acc, dq, rows, seq_q, heads, scale,
        dq.stride(0), dq.stride(1), dq.stride(2), D=dim, D_POW2=triton.next_power_of_2(dim),
        BLOCK=64,
    )  # fmt: skip
    return dq, dk, dv


def _dq_descriptor(dq_acc: torch.Tensor, cols: int) -> TensorDescriptor:
    """The fp32 dQ accumulator, reduce-added in [TILE, cols] boxes."""
    box = [1, 1, TILE_ROWS, cols]
    layout = gl.NVMMASharedLayout.get_default_for(box, gl.float32)
    return TensorDescriptor(dq_acc, list(dq_acc.shape), list(dq_acc.stride()), box, layout)


def _constexprs(cfg) -> dict:
    """The kernel's compile-time configuration from a ``BackwardConfig``."""
    return {
        "DC": cfg.dc,
        "NCH": cfg.nch,
        "ALIAS": cfg.alias,
        "KV_BUFS": cfg.kv_bufs,
        "Q_SLOTS": cfg.q_slots,
        "DO_SLOTS": cfg.do_slots,
        "P_REGS": cfg.p_regs,
        "DS_REGS": cfg.ds_regs,
        "SMALL_REGS": cfg.small_regs,
        "RSTAGES": cfg.reduce_stages,
        "STREAM": cfg.stream,
        "NSL": cfg.nsl,
        "NW": cfg.nw,
        "K_RES": cfg.k_res,
    }


def _no_lists(device) -> tuple:
    """Dummy MASK tile lists (start, count, list, classes, key tiles, list rows)."""
    dummy = torch.zeros(1, dtype=torch.int32, device=device)
    return dummy, dummy, dummy, dummy, 1, 1


def _no_varlen(device) -> tuple:
    """Dummy VARLEN arguments for the dense kernel."""
    dummy = torch.zeros(1, dtype=torch.int32, device=device)
    return dummy, dummy, dummy, dummy, dummy, dummy, 0


@triton.jit
def _prepare_varlen_kernel(
    out_ptr, dout_ptr, lse_ptr, stats_ptr, dq_ptr, qtiles_ptr, cu_q_ptr, total_q,
    total_q_tiles, heads, D: tl.constexpr, D_POW2: tl.constexpr, TILE_Q: tl.constexpr,
):  # fmt: skip
    """VARLEN ``_prepare_kernel``: one (head, query tile of a sequence) per program, from
    the query-tile table (sequence, tile); rows past the sequence's end get -m = -inf
    (P = 0: they add nothing to dQ, dK, dV although their TMA boxes read the next
    sequence's rows). Packed out / dout [Tq, H, D], lse [1, H, 2, Tq]."""
    head = tl.program_id(0) // total_q_tiles
    index = tl.program_id(0) % total_q_tiles
    seq = tl.load(qtiles_ptr + 2 * index)
    tile = tl.load(qtiles_ptr + 2 * index + 1)
    q_off = tl.load(cu_q_ptr + seq)
    length = tl.load(cu_q_ptr + seq + 1) - q_off
    s = tile * TILE_Q + tl.arange(0, TILE_Q)
    valid = s < length
    rows = (q_off + s).to(tl.int64)
    cols = tl.arange(0, D_POW2)
    both = valid[:, None] & (cols < D)[None, :]
    off = rows[:, None] * (heads * D) + head * D + cols[None, :]
    o = tl.load(out_ptr + off, mask=both, other=0.0)
    do = tl.load(dout_ptr + off, mask=both, other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    row_stats = lse_ptr + head.to(tl.int64) * 2 * total_q + rows
    m = tl.load(row_stats, mask=valid, other=float("inf"))
    log_l = tl.load(row_stats + total_q, mask=valid, other=0.0)
    stats = stats_ptr + (head.to(tl.int64) * total_q_tiles + index) * 3 * TILE_Q
    tl.store(stats + tl.arange(0, TILE_Q), -m)
    tl.store(stats + TILE_Q + tl.arange(0, TILE_Q), -log_l)
    tl.store(stats + 2 * TILE_Q + tl.arange(0, TILE_Q), -delta)
    dq_rows = dq_ptr + (head.to(tl.int64) * total_q + rows[:, None]) * D + cols[None, :]
    tl.store(dq_rows, tl.zeros([TILE_Q, D_POW2], dtype=tl.float32), mask=both)


def varlen_backward_tables(
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, kv_heads: int, block: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(key-tile table int32 [tiles, 3] = (sequence, K / V head, first key), query-tile
    table int32 [query tiles, 2] = (sequence, tile), first query tile of each sequence
    int32 [B + 1]). Key tiles no query sees (block-causal, Sq < Skv) are left out."""
    len_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    len_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()
    device = cu_seqlens_q.device
    key_tiles, q_tiles, cu_qt = [], [], [0]
    for seq in range(len(len_q)):
        seen = len_k[seq]
        if block and len_q[seq]:
            seen = min(seen, ((len_q[seq] - 1) // block + 1) * block)
        if len_q[seq] == 0:
            seen = 0
        for start in range(0, seen, TILE_ROWS):
            key_tiles += [(seq, head, start) for head in range(kv_heads)]
        n = triton.cdiv(len_q[seq], TILE_ROWS)
        q_tiles += [(seq, t) for t in range(n)]
        cu_qt.append(cu_qt[-1] + n)

    def as_int(rows, width):
        return torch.tensor(rows or [(0,) * width], dtype=torch.int32, device=device)

    return (
        as_int(key_tiles, 3),
        as_int(q_tiles, 2),
        torch.tensor(cu_qt, dtype=torch.int32, device=device),
    )


def attention_backward_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    key_tiles: torch.Tensor,
    q_tiles: torch.Tensor,
    cu_qt: torch.Tensor,
    scale: float,
    block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients of ``attention_forward_varlen``: dq [Tq, H, D], dk / dv [Tk, Hkv, D];
    the tables from ``varlen_backward_tables`` (no host synchronization here)."""
    total_q, heads, dim = q.shape
    total_k, kv_heads, _ = k.shape
    device = q.device
    dout = dout.contiguous()
    total_q_tiles = q_tiles.shape[0]
    stats = torch.empty(heads * max(total_q_tiles, 1) * 3 * TILE_ROWS, device=device, dtype=torch.float32)
    dq_acc = torch.zeros(1, heads, total_q, dim, device=device, dtype=torch.float32)
    if total_q_tiles:
        _prepare_varlen_kernel[(heads * total_q_tiles,)](
            out, dout, lse, stats, dq_acc, q_tiles, cu_seqlens_q, total_q, total_q_tiles,
            heads, D=dim, D_POW2=triton.next_power_of_2(dim), TILE_Q=TILE_ROWS,
        )  # fmt: skip
    stats_desc = TensorDescriptor.from_tensor(stats, [TILE_ROWS], gl.NVMMASharedLayout(0, 32, rank=1))
    cfg = backward_config(dim, q.element_size(), block > 0)
    dq_desc = _dq_descriptor(dq_acc, cfg.reduce_cols)
    dk = torch.zeros_like(k)  # keys no query sees keep 0
    dv = torch.zeros_like(v)
    q4, k4, v4, do4, dk4, dv4 = (t.unsqueeze(0).transpose(1, 2) for t in (q, k, v, dout, dk, dv))
    num_tiles = key_tiles.shape[0] * cfg.nsl
    if num_tiles and total_q_tiles:
        attention_bwd_kernel[(min(num_tiles, sm_count(device)),)](
            *(tma_descriptor(t, TILE_ROWS, cfg.dc) for t in (q4, k4, v4, do4, dk4, dv4)),
            stats_desc, dq_desc, scale * LOG2E, heads, heads // kv_heads, total_q, total_k,
            max(block, 1), 1, num_tiles, *mask_arguments(None, device), total_q_tiles,
            key_tiles, cu_seqlens_q, cu_seqlens_k, cu_qt, dk, dv, kv_heads * dim,
            *_no_lists(device),
            **_constexprs(cfg), CAUSAL=block > 0, MASK=False, VARLEN=True, num_warps=4,
        )  # fmt: skip
    dq = torch.empty_like(q)
    dq4 = dq.unsqueeze(0).transpose(1, 2)
    _dq_convert_kernel[(triton.cdiv(heads * total_q, 64),)](
        dq_acc, dq4, heads * total_q, total_q, heads, scale,
        dq4.stride(0), dq4.stride(1), dq4.stride(2), D=dim, D_POW2=triton.next_power_of_2(dim),
        BLOCK=64,
    )  # fmt: skip
    return dq, dk, dv
