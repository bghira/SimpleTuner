"""Derived FA4 SM80 methods preserving separate unscaled attention statistics.

Derived from Dao-AILab/flash-attention revision
4616f5116d33e3ba3cbb71d582caffcfcb49649c, BSD-3-Clause.
See LICENSES/FlashAttention-BSD-3-Clause.txt and FlashAttention-AUTHORS.txt.
Unchanged methods use the pinned optional dependency.
"""

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang,
# Vijay Thakkar, Pradeep Ramani, Tri Dao.

import enum
import math
import operator
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import Arch, BaseDSL
from flash_attn.cute import utils
from flash_attn.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess as _Original
from flash_attn.cute.flash_bwd_preprocess import PackGQA, offset_rows
from flash_attn.cute.pack_gqa import pack_gqa_layout
from flash_attn.cute.seqlen_info import SeqlenInfo, sequence_row_offset
from flash_attn.cute.tile_scheduler import SingleTileScheduler, SingleTileVarlenScheduler, TileSchedulerArguments
from quack import copy_utils, layout_utils
from quack.cute_dsl_utils import ParamsBase


class FlashAttentionBackwardPreprocess(_Original):

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mOlo: Optional[cute.Tensor],
        mPdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mdLSE: Optional[cute.Tensor],
        mRowMax: Optional[cute.Tensor],
        mScaleP: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        # This kernel is launched with use_pdl=True, so the GPU may start executing it in
        # "prologue" mode while the previous stream kernel is still running. We must wait
        # before touching any upstream GMEM outputs (mO, mdO, mLSE); otherwise we risk
        # reading a partially-written dout, which silently corrupts dpsum = sum(O * dO) and
        # propagates to dQ/dK via dS = P * (dP - dpsum).
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        if work_tile.is_valid_tile:
            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////
            seqlen_static = mO.shape[0] if const_expr(self.pack_gqa == PackGQA.OFF) else mO.shape[0][1]
            seqlen = SeqlenInfo.create(batch_idx, seqlen_static, mCuSeqlensQ, mSeqUsedQ, tile=self.tile_m)
            # varlen offset of this sequence's dPsum / LSElog2 / dQaccum rows: with
            # PACKED_ROW_OUTPUTS they are per KV head, qhead_per_kvhead rows per token (with
            # PER_QHEAD_OUTPUTS they keep the packed (qh, t) row mode, see offset_rows)
            rows_per_token = self.qhead_per_kvhead if const_expr(self.pack_gqa == PackGQA.PACKED_ROW_OUTPUTS) else 1
            row_offset = sequence_row_offset(
                seqlen.offset,
                batch_idx,
                rows_per_token,
                self.tile_m,
                padded=self.use_padded_offsets,
            )
            # (seqlen, dv)
            mO_cur, mdO_cur = [seqlen.offset_batch(mX, batch_idx, dim=3)[None, None, head_idx] for mX in (mO, mdO)]
            mOlo_cur = None
            if const_expr(mOlo is not None):
                mOlo_cur = seqlen.offset_batch(mOlo, batch_idx, dim=3)[None, None, head_idx]
            mPdPsum_cur = (
                mPdPsum[None, head_idx, batch_idx]
                if const_expr(not seqlen.has_cu_seqlens)
                else offset_rows(mPdPsum[None, head_idx], row_offset)
            )
            headdim_v = mO_cur.shape[1]
            seqlen_q = seqlen.seqlen if const_expr(self.pack_gqa == PackGQA.OFF) else seqlen.seqlen * self.qhead_per_kvhead
            seqlen_q_rounded = cute.round_up(seqlen_q, self.tile_m)
            seqlen_limit = seqlen_q - m_block * self.tile_m

            lse = None
            if const_expr(mLSE is not None):
                mLSE_cur = seqlen.offset_batch(mLSE, batch_idx, dim=2)[None, head_idx]
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                lse = Float32.inf
                if tidx < seqlen_limit:
                    lse = gLSE[tidx]

            blk_shape = (self.tile_m, self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, blk_shape, (m_block, 0))
            gdO = cute.local_tile(mdO_cur, blk_shape, (m_block, 0))
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            # (CPY_Atom, CPY_M, CPY_K)
            tOgO = gmem_thr_copy_O.partition_S(gO)
            tOgdO = gmem_thr_copy_O.partition_S(gdO)
            tOgOlo = None
            if const_expr(mOlo is not None):
                gOlo = cute.local_tile(mOlo_cur, blk_shape, (m_block, 0))
                tOgOlo = gmem_thr_copy_O.partition_S(gOlo)
            cO = cute.make_identity_tensor(blk_shape)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_thr_copy_O.get_slice(0).partition_S(cO)
            tOpO = None
            if const_expr(self.check_hdim_v_oob):
                tOpO = copy_utils.predicate_k(tOcO, limit=headdim_v)
            # Each copy uses the predicate sliced to the current m (see loop below).
            copy = copy_utils.copy

            tOrO = cute.make_rmem_tensor_like(tOgO)
            tOrdO = cute.make_rmem_tensor_like(tOgdO)
            if const_expr(self.check_hdim_v_oob):
                tOrO.fill(0.0)
                tOrdO.fill(0.0)
            assert tOgO.shape == tOgdO.shape
            if const_expr(mOlo is None):
                for m in cutlass.range(cute.size(tOrO.shape[1]), unroll_full=True):
                    # Instead of using tOcO, we using t0OcO and subtract the offset from the limit.
                    # This is bc the entries of t0OcO are known at compile time.
                    if t0OcO[0, m, 0][0] < seqlen_limit - tOcO[0][0]:
                        # The predicate carries a CPY_M mode (broadcast over m); slice it to the
                        # current m so its shape matches the m-sliced copy source. Otherwise the
                        # vectorized copy atom's predicate-shape verification fails when head_dim_v
                        # is not a multiple of the copy-atom width (e.g. 72, 104).
                        tOpO_cur = tOpO[None, m, None] if const_expr(tOpO is not None) else None
                        copy(tOgO[None, m, None], tOrO[None, m, None], pred=tOpO_cur)
                        copy(tOgdO[None, m, None], tOrdO[None, m, None], pred=tOpO_cur)
                # O and dO loads are done; signal that the next kernel can start.
                # Correctness is ensured by griddepcontrol_wait() in bwd_sm90 before it reads our outputs.
                if const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_launch_dependents()
                # Sum across the "k" dimension
                pdpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(
                    cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1)
                )
            else:
                # O residual: O + O_lo recovers the forward's fp32 O to ~2^-16 relative, so
                # dpsum = rowsum(dO * O) no longer carries the bf16 output rounding (which
                # dominates dS = P * (dP - dpsum) for peaked attention rows). Three tiles do
                # not fit the register budget the (O, dO) pair already fills, so this variant
                # streams one row-slice at a time: load O/dO/O_lo for slice m, reduce over
                # the head dimension, keep only the per-row partial sums.
                tOrOlo = cute.make_rmem_tensor_like(tOgOlo)
                if const_expr(self.check_hdim_v_oob):
                    tOrOlo.fill(0.0)
                PdP_part = cute.make_rmem_tensor(cute.size(tOrO, mode=[1]), Float32)
                for m in cutlass.range(cute.size(tOrO.shape[1]), unroll_full=True):
                    if t0OcO[0, m, 0][0] < seqlen_limit - tOcO[0][0]:
                        tOpO_cur = tOpO[None, m, None] if const_expr(tOpO is not None) else None
                        copy(tOgO[None, m, None], tOrO[None, m, None], pred=tOpO_cur)
                        copy(tOgdO[None, m, None], tOrdO[None, m, None], pred=tOpO_cur)
                        copy(tOgOlo[None, m, None], tOrOlo[None, m, None], pred=tOpO_cur)
                    # Rows past seqlen_limit hold stale registers here, exactly as in the
                    # non-residual path; their partial sums are masked at the gmem write.
                    o_lo_f32 = tOrOlo[None, m, None].load().to(Float32)
                    o_f32 = tOrO[None, m, None].load().to(Float32) + o_lo_f32
                    PdP_part[m] = (o_f32 * tOrdO[None, m, None].load().to(Float32)).reduce(
                        cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
                    )
                if const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_launch_dependents()
                pdpsum = PdP_part.load()
            threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
            assert cute.arch.WARP_SIZE % threads_per_row == 0
            pdpsum = utils.warp_reduce(pdpsum, operator.add, width=threads_per_row)
            PdP_sum = cute.make_rmem_tensor(cute.size(tOrO, mode=[1]), Float32)
            PdP_sum.store(pdpsum)

            # If dLSE is provided, compute D' = D - dLSE (see module docstring for derivation).
            gdLSE = None
            if const_expr(mdLSE is not None):
                mdLSE_cur = seqlen.offset_batch(mdLSE, batch_idx, dim=2)[None, head_idx]
                gdLSE = cute.local_tile(mdLSE_cur, (self.tile_m,), (m_block,))

            # Write PdPsum from rmem -> gmem
            gPdPsum = cute.local_tile(mPdPsum_cur, (self.tile_m,), (m_block,))
            # Only the thread corresponding to column 0 writes out the PdPsum to gmem
            if tOcO[0, 0, 0][1] == 0:
                for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
                    row = tOcO[0, m, 0][0]
                    PdPsum_val = 0.0
                    if row < seqlen_limit:
                        PdPsum_val = PdP_sum[m]
                        if const_expr(mdLSE is not None):
                            PdPsum_val -= gdLSE[row]
                    # Without padded per-sequence offsets there is no slack past seqlen: the
                    # zero-filled tail would alias the next sequence (or fall off the buffer).
                    if const_expr(self.use_padded_offsets) or row < seqlen_limit:
                        gPdPsum[row] = PdPsum_val

            # Clear dQaccum
            if const_expr(mdQaccum is not None):
                mdQaccum_cur = (
                    mdQaccum[None, head_idx, batch_idx]
                    if const_expr(not seqlen.has_cu_seqlens)
                    else offset_rows(mdQaccum[None, head_idx], row_offset * self.head_dim_padded)
                )
                blkdQaccum_shape = (self.tile_m * self.head_dim_padded,)
                gdQaccum = cute.local_tile(mdQaccum_cur, blkdQaccum_shape, (m_block,))
                gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
                tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
                zero = cute.make_rmem_tensor_like(tdQgdQaccum)
                zero.fill(0.0)
                cute.copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum)

            LOG2_E = math.log2(math.e)
            lse_log2 = lse if lse != -Float32.inf else 0.0
            if const_expr(mLSElog2 is not None):
                mLSElog2_cur = (
                    mLSElog2[None, head_idx, batch_idx]
                    if const_expr(not seqlen.has_cu_seqlens)
                    else offset_rows(mLSElog2[None, head_idx], row_offset)
                )
                gLSElog2 = cute.local_tile(mLSElog2_cur, (self.tile_m,), (m_block,))
                # Without padded per-sequence offsets there is no slack past seqlen: the
                # +inf tail would alias the next sequence's rows (or fall off the buffer).
                seqlen_q_stored = seqlen_q_rounded if const_expr(self.use_padded_offsets) else seqlen_q
                if tidx < seqlen_q_stored - m_block * self.tile_m:
                    gLSElog2[tidx] = lse_log2

            if const_expr(mRowMax is not None):
                assert mLSE is not None
                # (s, n)
                mRowMax_cur, mScaleP_cur = [
                    seqlen.offset_batch(mX, batch_idx, dim=3)[None, None, head_idx] for mX in (mRowMax, mScaleP)
                ]
                # (tile_m, n)
                gRowMax, gScaleP = [
                    cute.local_tile(mX, (self.tile_m,), (m_block, None)) for mX in (mRowMax_cur, mScaleP_cur)
                ]

                assert self.tile_m <= self.num_threads
                # Rows past seqlen_limit are outside row_max/scale_p (partial last tile).
                if (const_expr(self.tile_m == self.num_threads) or tidx < self.tile_m) and tidx < seqlen_limit:
                    for n in cutlass.range(gRowMax.shape[1], unroll=4):
                        row_max = gRowMax[tidx, n]
                        scale = 0.0
                        if row_max != -Float32.inf and lse != -Float32.inf:
                            scale = softmax_scale_log2 * row_max - lse_log2
                            scale = cute.math.exp2(scale, fastmath=True)
                        gScaleP[tidx, n] = scale
