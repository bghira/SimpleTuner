"""Derived FA4 SM80 methods preserving separate unscaled attention statistics.

Derived from Dao-AILab/flash-attention revision
4616f5116d33e3ba3cbb71d582caffcfcb49649c, BSD-3-Clause.
See LICENSES/FlashAttention-BSD-3-Clause.txt and FlashAttention-AUTHORS.txt.
Unchanged methods use the pinned optional dependency.
"""

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang,
# Vijay Thakkar, Pradeep Ramani, Tri Dao.

import math
from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils_basic
from cutlass import Int32
from cutlass.cute.nvgpu import cpasync, warp
from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute import utils
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.flash_bwd import FlashAttentionBackwardSm80 as _Original
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import call_score_mod, call_score_mod_bwd
from flash_attn.cute.tile_scheduler import SingleTileScheduler, SingleTileVarlenScheduler, TileSchedulerArguments
from flash_attn.cute.utils import AuxData
from quack import layout_utils
from quack.cute_dsl_utils import ParamsBase


class FlashAttentionBackwardSm80(_Original):
    causal_head_group = 1

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout, self.sdO_layout)
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]
        sLSE_struct, sdPsum_struct = [
            cute.struct.Align[cute.struct.MemRange[cutlass.Float32, cute.cosize(layout)], 128]
            for layout in (self.sLSE_layout, self.sLSE_layout)
        ]
        sP_struct, sdS_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 128]
            for layout in (self.sPdS_layout, self.sPdS_layout)
        ]

        @cute.struct
        class SharedStorageSeparateQV:
            sK: sK_struct
            sV: sV_struct
            sQ: sQ_struct
            sdO: sdO_struct
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sP: sP_struct
            sdS: sdS_struct
            # TODO: the case where there's no sP

        @cute.struct
        class SharedStorageSharedQV:
            sK: sK_struct
            sQ: sQV_struct
            sdO: sdO_struct
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sP: sP_struct
            sdS: sdS_struct

        storage = SharedStorageSeparateQV if cutlass.const_expr(not self.share_QV_smem) else SharedStorageSharedQV
        if storage.size_in_bytes() > self.shared_capacity:
            raise ValueError("Backward storage exceeds the actual device limit.")
        return storage

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sLSEMma_layout: cute.Layout,
        gmem_tiled_copy_QK: cute.TiledCopy,
        gmem_tiled_copy_VdO: cute.TiledCopy,
        gmem_tiled_copy_dK: cute.TiledCopy,
        gmem_tiled_copy_dV: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dkv: cute.TiledMma,
        tiled_mma_dq: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()

        n_block, head_idx, batch_idx, _ = work_tile.tile_idx

        if cutlass.const_expr(self.is_causal and self.causal_head_group > 1):
            assert mCuSeqlensQ is None and mCuSeqlensK is None
            assert mSeqUsedQ is None and mSeqUsedK is None
            heads = mQ.shape[2]
            total_heads = heads * mQ.shape[0]
            blocks = cute.ceil_div(mK.shape[1], self.n_block_size)
            group = self.causal_head_group
            head_batch = batch_idx * heads + head_idx
            section = head_batch // group
            within_section = n_block + blocks * (head_batch % group)
            mapped_block = within_section // group
            mapped_head = within_section % group
            if section >= total_heads // group:
                remainder = cutlass.max(total_heads % group, 1)
                mapped_block = within_section // remainder
                mapped_head = within_section % remainder
            n_block = mapped_block
            mapped_head_batch = section * group + mapped_head
            batch_idx = mapped_head_batch // heads
            head_idx = mapped_head_batch % heads

        if work_tile.is_valid_tile:
            seqlen = SeqlenInfoQK.create(
                batch_idx,
                mQ.shape[1],
                mK.shape[1],
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=mSeqUsedK,
                tile_m=self.m_block_size,
                tile_n=self.n_block_size,
            )

            block_info = BlockInfo(
                self.m_block_size,
                self.n_block_size,
                self.is_causal,
                self.is_local,
                False,
                window_size_left,
                window_size_right,
            )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            # TODO: return early if m_block_max == 0

            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////
            blkQ_shape = (self.m_block_size, self.head_dim_padded)
            blkK_shape = (self.n_block_size, self.head_dim_padded)
            blkV_shape = (self.n_block_size, self.head_dim_v_padded)
            blkdO_shape = (self.m_block_size, self.head_dim_v_padded)

            if cutlass.const_expr(not seqlen.has_cu_seqlens_q):
                mQ_cur = mQ[batch_idx, None, head_idx, None]
                mLSE_cur = mLSE[batch_idx, head_idx, None]
                mdO_cur = mdO[batch_idx, None, head_idx, None]
                mdPsum_cur = mdPsum[batch_idx, head_idx, None]
                mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
            else:
                padded_offset_q = seqlen.padded_offset_q
                mQ_cur = cute.domain_offset((seqlen.offset_q, 0), mQ[None, head_idx, None])
                mLSE_cur = cute.domain_offset((padded_offset_q,), mLSE[head_idx, None])
                mdO_cur = cute.domain_offset((seqlen.offset_q, 0), mdO[None, head_idx, None])
                mdPsum_cur = cute.domain_offset((padded_offset_q,), mdPsum[head_idx, None])
                mdQaccum_cur = cute.domain_offset((padded_offset_q * self.head_dim_padded,), mdQaccum[head_idx, None])
            head_idx_kv = head_idx // self.qhead_per_kvhead if cutlass.const_expr(not self.pack_gqa) else head_idx

            if cutlass.const_expr(not seqlen.has_cu_seqlens_k):
                mK_cur, mV_cur = [t[batch_idx, None, head_idx_kv, None] for t in (mK, mV)]
            else:
                mK_cur, mV_cur = [cute.domain_offset((seqlen.offset_k, 0), t[None, head_idx_kv, None]) for t in (mK, mV)]

            # (m_block_size, head_dim, m_block)
            gQ = cute.local_tile(mQ_cur, blkQ_shape, (None, 0))
            # (n_block_size, head_dim)
            gK = cute.local_tile(mK_cur, blkK_shape, (n_block, 0))
            # (n_block_size, head_dim_v)
            gV = cute.local_tile(mV_cur, blkV_shape, (n_block, 0))
            # (m_block_size, head_dim_v, m_block)
            gdO = cute.local_tile(mdO_cur, blkdO_shape, (None, 0))
            gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (None,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.m_block_size,), (None,))
            gdQaccum = cute.local_tile(mdQaccum_cur, (self.m_block_size * self.head_dim_padded,), (None,))

            # ///////////////////////////////////////////////////////////////////////////////
            # Get shared memory buffer
            # ///////////////////////////////////////////////////////////////////////////////
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.sQ.get_tensor(sQ_layout)
            sK = storage.sK.get_tensor(sK_layout)
            if cutlass.const_expr(not self.share_QV_smem):
                sV = storage.sV.get_tensor(sV_layout)
            else:
                sV = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout)
            sdO = storage.sdO.get_tensor(sdO_layout)
            sP = storage.sP.get_tensor(sPdS_layout)
            sdS = storage.sdS.get_tensor(sPdS_layout)
            sLSE = storage.sLSE.get_tensor(sLSE_layout)
            sdPsum = storage.sdPsum.get_tensor(sLSE_layout)
            sLSEMma = storage.sLSE.get_tensor(sLSEMma_layout)
            sdPsumMma = storage.sdPsum.get_tensor(sLSEMma_layout)

            # Transpose view of tensors for tiled mma
            sQt, sdOt, sKt, sPt, sdSt = [layout_utils.transpose_view(t) for t in (sQ, sdO, sK, sP, sdS)]

            gmem_thr_copy_QK = gmem_tiled_copy_QK.get_slice(tidx)
            gmem_thr_copy_VdO = gmem_tiled_copy_VdO.get_slice(tidx)
            gmem_thr_copy_lse = gmem_tiled_copy_LSE.get_slice(tidx)
            gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
            # (CPY_Atom, CPY_M, CPY_K, m_block)
            tQgQ = gmem_thr_copy_QK.partition_S(gQ)
            tQsQ = gmem_thr_copy_QK.partition_D(sQ)
            # (CPY_Atom, CPY_N, CPY_K)
            tKgK = gmem_thr_copy_QK.partition_S(gK)
            tKsK = gmem_thr_copy_QK.partition_D(sK)
            # (CPY_Atom, CPY_N, CPY_K)
            tVgV = gmem_thr_copy_VdO.partition_S(gV)
            tVsV = gmem_thr_copy_VdO.partition_D(sV)
            # (CPY_Atom, CPY_M, CPY_K, m_block)
            tdOgdO = gmem_thr_copy_VdO.partition_S(gdO)
            tdOsdO = gmem_thr_copy_VdO.partition_D(sdO)
            tLSEgLSE = gmem_thr_copy_lse.partition_S(gLSE)
            tLSEsLSE = gmem_thr_copy_lse.partition_D(sLSE)
            tLSEgdPsum = gmem_thr_copy_lse.partition_S(gdPsum)
            tLSEsdPsum = gmem_thr_copy_lse.partition_D(sdPsum)
            tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)

            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            thr_mma_sdp = tiled_mma_sdp.get_slice(tidx)
            thr_mma_dkv = tiled_mma_dkv.get_slice(tidx)
            thr_mma_dq = tiled_mma_dq.get_slice(tidx)
            acc_shape_dK = thr_mma_dkv.partition_shape_C((self.n_block_size, self.head_dim_padded))
            acc_shape_dV = thr_mma_dkv.partition_shape_C((self.n_block_size, self.head_dim_v_padded))
            acc_dK = cute.make_rmem_tensor(acc_shape_dK, cutlass.Float32)
            acc_dV = cute.make_rmem_tensor(acc_shape_dV, cutlass.Float32)
            acc_dK.fill(0.0)
            acc_dV.fill(0.0)

            tSrQ = utils.mma_make_fragment_A(sQ[None, None, 0], thr_mma_sdp, swapAB=self.SdP_swapAB)
            tSrK = utils.mma_make_fragment_B(sK, thr_mma_sdp, swapAB=self.SdP_swapAB)
            tdPrdO = utils.mma_make_fragment_A(sdO[None, None, 0], thr_mma_sdp, swapAB=self.SdP_swapAB)
            tdPrV = utils.mma_make_fragment_B(sV, thr_mma_sdp, swapAB=self.SdP_swapAB)
            tdVrP = utils.mma_make_fragment_A(sPt, thr_mma_dkv, swapAB=self.dKV_swapAB)
            tdVrdO = utils.mma_make_fragment_B(sdOt[None, None, 0], thr_mma_dkv, swapAB=self.dKV_swapAB)
            tdKrdS = utils.mma_make_fragment_A(sdSt, thr_mma_dkv, swapAB=self.dKV_swapAB)
            tdKrQ = utils.mma_make_fragment_B(sQt[None, None, 0], thr_mma_dkv, swapAB=self.dKV_swapAB)
            tdQrdS = utils.mma_make_fragment_A(sdS, thr_mma_dq, swapAB=self.dQ_swapAB)
            tdQrK = utils.mma_make_fragment_B(sKt, thr_mma_dq, swapAB=self.dQ_swapAB)

            LSEslice = (None, 0, None) if cutlass.const_expr(not self.SdP_swapAB) else (0, None, None)
            tSsLSEMma = layout_utils.reshape_acc_to_mn(thr_mma_sdp.partition_C(sLSEMma))[LSEslice]
            tSsdPsumMma = layout_utils.reshape_acc_to_mn(thr_mma_sdp.partition_C(sdPsumMma))[LSEslice]

            # ///////////////////////////////////////////////////////////////////////////////
            # Smem copy atom tiling
            # ///////////////////////////////////////////////////////////////////////////////
            smem_copy_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.dtype,
            )
            smem_copy_atom_transposed = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.dtype,
            )
            smem_thr_copy_QdO = utils.make_tiled_copy_A(smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB).get_slice(
                tidx
            )
            smem_thr_copy_KV = utils.make_tiled_copy_B(smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB).get_slice(tidx)
            # TODO: should this be smem_copy_atom_transposed?
            smem_thr_copy_PdSt = utils.make_tiled_copy_A(
                smem_copy_atom_transposed, tiled_mma_dkv, swapAB=self.dKV_swapAB
            ).get_slice(tidx)
            smem_thr_copy_QdOt = utils.make_tiled_copy_B(
                smem_copy_atom_transposed, tiled_mma_dkv, swapAB=self.dKV_swapAB
            ).get_slice(tidx)
            smem_thr_copy_dS = utils.make_tiled_copy_A(smem_copy_atom, tiled_mma_dq, swapAB=self.dQ_swapAB).get_slice(tidx)
            smem_thr_copy_Kt = utils.make_tiled_copy_B(
                smem_copy_atom_transposed, tiled_mma_dq, swapAB=self.dQ_swapAB
            ).get_slice(tidx)
            # TODO: what's the number of bits? What if SdP_swapAB
            r2s_thr_copy_PdS = cute.make_tiled_copy_C(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=2 * self.dtype.width),
                tiled_mma_sdp,
            ).get_slice(tidx)

            tSsQ = smem_thr_copy_QdO.partition_S(sQ)
            tdPsdO = smem_thr_copy_QdO.partition_S(sdO)
            tSsK = smem_thr_copy_KV.partition_S(sK)
            tdPsV = smem_thr_copy_KV.partition_S(sV)
            tdVsPt = smem_thr_copy_PdSt.partition_S(sPt)
            tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt)
            tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt)
            tdKsQt = smem_thr_copy_QdOt.partition_S(sQt)
            tdQsdS = smem_thr_copy_dS.partition_S(sdS)
            tdQsKt = smem_thr_copy_Kt.partition_S(sKt)
            tPsP = r2s_thr_copy_PdS.partition_D(sP)
            tdSsdS = r2s_thr_copy_PdS.partition_D(sdS)

            # ///////////////////////////////////////////////////////////////////////////////
            # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
            # of tile_shape
            # ///////////////////////////////////////////////////////////////////////////////
            # Construct identity layout for KV
            cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
            tQcQ = gmem_thr_copy_QK.partition_S(cQ)
            t0QcQ = gmem_thr_copy_QK.get_slice(0).partition_S(cQ)
            if cutlass.const_expr(self.head_dim_padded == self.head_dim_v_padded):
                tdOcdO = tQcQ
                t0dOcdO = t0QcQ
            else:
                cdO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
                tdOcdO = gmem_thr_copy_VdO.partition_S(cdO)
                t0dOcdO = gmem_thr_copy_VdO.get_slice(0).partition_S(cdO)
            cLSE = cute.make_identity_tensor((self.m_block_size,))
            tLSEcLSE = gmem_thr_copy_lse.partition_S(cLSE)

            # Allocate predicate tensors for m and n, here we only allocate the tile of k, and
            # use "if" on the mn dimension.
            # This is to reduce register pressure and gets 2-3% performance gain.

            d_head = mQ.shape[cute.rank(mQ) - 1]
            d_head_v = mdO.shape[cute.rank(mdO) - 1]

            tQpQ = utils.predicate_k(tQcQ, limit=d_head)
            if cutlass.const_expr(self.same_hdim_kv):
                tdOpdO = tQpQ
            else:
                tdOpdO = utils.predicate_k(tdOcdO, limit=d_head_v)

            # group parameters for compute_one_m_block
            mma_params = SimpleNamespace(
                thr_mma_sdp=thr_mma_sdp,
                thr_mma_dkv=thr_mma_dkv,
                thr_mma_dq=thr_mma_dq,
                tSrQ=tSrQ,
                tSrK=tSrK,
                tdPrdO=tdPrdO,
                tdPrV=tdPrV,
                tdVrP=tdVrP,
                tdVrdO=tdVrdO,
                tdKrdS=tdKrdS,
                tdKrQ=tdKrQ,
                tdQrdS=tdQrdS,
                tdQrK=tdQrK,
                acc_dK=acc_dK,
                acc_dV=acc_dV,
            )
            smem_copy_params = SimpleNamespace(
                smem_thr_copy_QdO=smem_thr_copy_QdO,
                smem_thr_copy_KV=smem_thr_copy_KV,
                smem_thr_copy_PdSt=smem_thr_copy_PdSt,
                smem_thr_copy_QdOt=smem_thr_copy_QdOt,
                smem_thr_copy_dS=smem_thr_copy_dS,
                smem_thr_copy_Kt=smem_thr_copy_Kt,
                r2s_thr_copy_PdS=r2s_thr_copy_PdS,
                tSsQ=tSsQ,
                tSsK=tSsK,
                tdPsdO=tdPsdO,
                tdPsV=tdPsV,
                tSsLSEMma=tSsLSEMma,
                tSsdPsumMma=tSsdPsumMma,
                tPsP=tPsP,
                tdSsdS=tdSsdS,
                tdVsPt=tdVsPt,
                tdVsdOt=tdVsdOt,
                tdKsdSt=tdKsdSt,
                tdKsQt=tdKsQt,
                tdQsdS=tdQsdS,
                tdQsKt=tdQsKt,
            )
            gmem_copy_params = SimpleNamespace(gmem_thr_copy_dQaccum=gmem_thr_copy_dQaccum, tdQgdQaccum=tdQgdQaccum)
            load_Q_LSE = partial(
                self.load_Q_LSE,
                gmem_tiled_copy_QK,
                gmem_tiled_copy_LSE,
                tQgQ,
                tQsQ,
                tQcQ,
                t0QcQ,
                tQpQ,
                tLSEgLSE,
                tLSEsLSE,
                tLSEcLSE,
                seqlen=seqlen.seqlen_q,
            )
            load_dO_dPsum = partial(
                self.load_dO_dPsum,
                gmem_tiled_copy_VdO,
                gmem_tiled_copy_LSE,
                tdOgdO,
                tdOsdO,
                tdOcdO,
                t0dOcdO,
                tdOpdO,
                tLSEgdPsum,
                tLSEsdPsum,
                tLSEcLSE,
                seqlen=seqlen.seqlen_q,
            )
            compute_one_m_block = partial(
                self.compute_one_m_block,
                mma_params=mma_params,
                smem_copy_params=smem_copy_params,
                gmem_copy_params=gmem_copy_params,
                load_Q_LSE=load_Q_LSE,
                load_dO_dPsum=load_dO_dPsum,
                m_block_max=m_block_max,
                softmax_scale=softmax_scale,
                softmax_scale_log2=softmax_scale_log2,
                aux_data=aux_data,
                batch_idx=batch_idx,
                head_idx=head_idx,
                n_block=n_block,
            )

            if m_block_min < m_block_max:
                # ///////////////////////////////////////////////////////////////////////////////
                # Prologue
                # ///////////////////////////////////////////////////////////////////////////////
                # Start async loads of the last mn-tile, where we take care of the mn residue
                self.load_V(gmem_thr_copy_VdO, tVgV, tVsV, n_block, seqlen=seqlen.seqlen_k, headdim=d_head_v)
                if cutlass.const_expr(self.V_in_regs):
                    cute.arch.cp_async_commit_group()
                self.load_K(gmem_thr_copy_QK, tKgK, tKsK, n_block, seqlen=seqlen.seqlen_k, headdim=d_head)
                cute.arch.cp_async_commit_group()

                if cutlass.const_expr(self.V_in_regs):
                    cute.arch.cp_async_wait_group(1)
                    cute.arch.barrier()
                    tdPrV_copy_view = smem_thr_copy_KV.retile(tdPrV)
                    cute.copy(smem_thr_copy_KV, tdPsV, tdPrV_copy_view)
                    # Sync to avoid loading Q to smem_q, which overlaps with smem_v
                    cute.arch.barrier()

                m_block = m_block_min
                assert self.num_stages_Q >= self.num_stages_dO
                for stage in cutlass.range_constexpr(self.num_stages_Q):
                    if cutlass.const_expr(self.num_stages_Q == 1 or stage < self.num_stages_Q - 1):
                        if stage == 0 or m_block + stage < m_block_max:
                            load_Q_LSE(m_block + stage, smem_pipe_write_q=stage)
                        cute.arch.cp_async_commit_group()
                    if cutlass.const_expr(stage < self.num_stages_dO):
                        if stage == 0 or m_block + stage < m_block_max:
                            load_dO_dPsum(m_block + stage, smem_pipe_write_q=stage)
                        cute.arch.cp_async_commit_group()

                # ///////////////////////////////////////////////////////////////////////////////
                # Mainloop
                # ///////////////////////////////////////////////////////////////////////////////
                # Start processing of the first n-block.
                mask = AttentionMask(
                    self.m_block_size,
                    self.n_block_size,
                    seqlen,
                    window_size_left,
                    window_size_right,
                )
                mask_fn = partial(
                    mask.apply_mask,
                    n_block=n_block,
                    thr_mma=thr_mma_sdp,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    mask_seqlen=True,
                    mask_causal=self.is_causal,
                    mask_local=self.is_local,
                )
                smem_pipe_read_q = cutlass.Int32(0)
                smem_pipe_read_do = cutlass.Int32(0)
                smem_pipe_write_q = cutlass.Int32(self.num_stages_Q - 1)
                smem_pipe_write_do = cutlass.Int32(0)
                for m_tile in cutlass.range(m_block_min, m_block_max, unroll=1):
                    compute_one_m_block(
                        m_tile,
                        smem_pipe_read_q,
                        smem_pipe_read_do,
                        smem_pipe_write_q,
                        smem_pipe_write_do,
                        mask_fn=mask_fn,
                    )
                    smem_pipe_read_q = self.advance_pipeline(smem_pipe_read_q, self.num_stages_Q)
                    smem_pipe_read_do = self.advance_pipeline(smem_pipe_read_do, self.num_stages_dO)
                    smem_pipe_write_q = self.advance_pipeline(smem_pipe_write_q, self.num_stages_Q)
                    smem_pipe_write_do = self.advance_pipeline(smem_pipe_write_do, self.num_stages_dO)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            # If GQA, we scale dK in the postprocessing kernel instead
            if cutlass.const_expr(self.qhead_per_kvhead == 1):
                acc_dK.store(acc_dK.load() * softmax_scale)
            # reuse sK and sV data iterator
            sdK = cute.make_tensor(sK.iterator, sK_layout)
            sdV = cute.make_tensor(sV.iterator, sV_layout)
            self.epilogue(
                acc_dK,
                acc_dV,
                mdK,
                mdV,
                sdK,
                sdV,
                gmem_tiled_copy_dK,
                gmem_tiled_copy_dV,
                tiled_mma_dkv,
                tidx,
                n_block,
                head_idx,
                batch_idx,
                seqlen,
                d_head,
                d_head_v,
            )

    @cute.jit
    def compute_one_m_block(
        self,
        m_block: cutlass.Int32,
        smem_pipe_read_q: cutlass.Int32,
        smem_pipe_read_do: cutlass.Int32,
        smem_pipe_write_q: cutlass.Int32,
        smem_pipe_write_do: cutlass.Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        load_Q_LSE: Callable,
        load_dO_dPsum: Callable,
        m_block_max: cutlass.Int32,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        aux_data: AuxData = AuxData(),
        batch_idx: cutlass.Int32 = 0,
        head_idx: cutlass.Int32 = 0,
        n_block: cutlass.Int32 = 0,
        mask_fn: Optional[Callable] = None,
    ):
        def load_Q_next():
            m_block_next = m_block + (self.num_stages_Q - 1 if cutlass.const_expr(self.num_stages_Q > 1) else 1)
            if m_block_next < m_block_max:
                load_Q_LSE(m_block_next, smem_pipe_write_q)
            cute.arch.cp_async_commit_group()

        def load_dO_next():
            if m_block + self.num_stages_dO < m_block_max:
                load_dO_dPsum(m_block + self.num_stages_dO, smem_pipe_write_do)
            cute.arch.cp_async_commit_group()

        # MMA S
        acc_shape_SdP = mma_params.thr_mma_sdp.partition_shape_C(
            (self.m_block_size, self.n_block_size)
            if cutlass.const_expr(not self.SdP_swapAB)
            else (self.n_block_size, self.m_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_SdP, cutlass.Float32)
        acc_S.fill(0.0)
        cute.arch.cp_async_wait_group(1 if cutlass.const_expr(self.num_stages_Q > 1) else 0)
        cute.arch.barrier()
        sm80_utils.gemm(
            mma_params.thr_mma_sdp,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.tSsQ[None, None, None, smem_pipe_read_q if cutlass.const_expr(self.num_stages_Q > 1) else 0],
            smem_copy_params.tSsK,
            smem_copy_params.smem_thr_copy_QdO,
            smem_copy_params.smem_thr_copy_KV,
            swap_AB=self.SdP_swapAB,
        )
        acc_S_pre = cute.make_fragment_like(acc_S)
        acc_S_pre.store(acc_S.load())
        tLSErLSE = cute.make_fragment_like(smem_copy_params.tSsLSEMma[None, 0])
        cute.autovec_copy(
            smem_copy_params.tSsLSEMma[None, smem_pipe_read_q if cutlass.const_expr(self.num_stages_Q > 1) else 0],
            tLSErLSE,
        )
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        acc_S_pre_mn = layout_utils.reshape_acc_to_mn(acc_S_pre)
        if cutlass.const_expr(self.score_mod is not None):
            for r in cutlass.range(cute.size(acc_S_mn, mode=[0]), unroll_full=True):
                acc_S_mn[r, None].store(
                    call_score_mod(
                        self.score_mod,
                        acc_S_mn[r, None].load() * softmax_scale,
                        0,
                        0,
                        0,
                        0,
                        None,
                        aux_data,
                    )
                )
        # Bounds and causal visibility are applied using exact fragment coordinates below.
        bidx = 0
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_S_mn)
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == 1: cute.print_tensor(tLSErLSE)
        assert cute.size(acc_S_mn, mode=[0]) == cute.size(tLSErLSE)
        probability_coords = layout_utils.reshape_acc_to_mn(
            mma_params.thr_mma_sdp.partition_C(cute.make_identity_tensor((self.m_block_size, self.n_block_size)))
        )
        magnitude = softmax_scale if softmax_scale >= 0 else -softmax_scale
        sign = cutlass.Float32(1) if softmax_scale >= 0 else cutlass.Float32(-1)
        for r in cutlass.range(cute.size(acc_S_mn, mode=[0]), unroll_full=True):
            query_row = m_block * self.m_block_size + probability_coords[r, 0][0]
            maximum = cutlass.Float32(0)
            if query_row < aux_data.tensors[0].shape[2]:
                maximum = aux_data.tensors[0][batch_idx, head_idx, query_row]
            for c in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[1])):
                key_row = n_block * self.n_block_size + probability_coords[r, c][1]
                score = acc_S_mn[r, c]
                probability = cutlass.Float32(0)
                visible = query_row < aux_data.tensors[0].shape[2] and key_row < aux_data.tensors[0].shape[2]
                if cutlass.const_expr(self.is_causal):
                    visible = visible and key_row <= query_row
                if visible and score != -cutlass.Float32.inf:
                    probability = cute.math.exp2(
                        (score * sign - maximum) * magnitude * cutlass.Float32(1.4426950408889634) - tLSErLSE[r],
                        fastmath=True,
                    )
                acc_S_mn[r, c] = probability
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_S_mn)

        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        probability_low_mn = layout_utils.reshape_acc_to_mn(rP)

        # MMA dP
        acc_dP = cute.make_rmem_tensor(acc_shape_SdP, cutlass.Float32)
        acc_dP.fill(0.0)
        cute.arch.cp_async_wait_group(1 if cutlass.const_expr(self.num_stages_dO > 1) else 0)
        cute.arch.barrier()
        sm80_utils.gemm(
            mma_params.thr_mma_sdp,
            acc_dP,
            mma_params.tdPrdO,
            mma_params.tdPrV,
            smem_copy_params.tdPsdO[
                None, None, None, smem_pipe_read_do if cutlass.const_expr(self.num_stages_dO > 1) else 0
            ],
            smem_copy_params.tdPsV,
            smem_copy_params.smem_thr_copy_QdO,
            smem_copy_params.smem_thr_copy_KV,
            hook_fn=load_Q_next if cutlass.const_expr(self.num_stages_Q > 1) else None,
            swap_AB=self.SdP_swapAB,
            B_in_regs=self.V_in_regs,
        )
        tLSErdPsum = cute.make_fragment_like(smem_copy_params.tSsdPsumMma[None, 0])
        cute.autovec_copy(
            smem_copy_params.tSsdPsumMma[None, smem_pipe_read_do if cutlass.const_expr(self.num_stages_dO > 1) else 0],
            tLSErdPsum,
        )
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP)
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_dP_mn)
        assert cute.size(acc_dP_mn, mode=[0]) == cute.size(tLSErdPsum)
        for r in cutlass.range(cute.size(acc_dP_mn, mode=[0]), unroll_full=True):
            grad_val = probability_low_mn[r, None].load().to(cutlass.Float32) * (acc_dP_mn[r, None].load() - tLSErdPsum[r])
            if cutlass.const_expr(self.score_mod_bwd is not None):
                grad_val = call_score_mod_bwd(
                    self.score_mod_bwd,
                    grad_val,
                    acc_S_pre_mn[r, None].load() * softmax_scale,
                    0,
                    0,
                    0,
                    0,
                    None,
                    aux_data,
                )
            acc_dP_mn[r, None].store(grad_val)
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_dP_mn)
        if cutlass.const_expr(not self.Mma_dKV_is_RS):
            tPrP = smem_copy_params.r2s_thr_copy_PdS.retile(rP)  # ((Atom,AtomNum), MMA_N, MMA_N)
            cute.copy(smem_copy_params.r2s_thr_copy_PdS, tPrP, smem_copy_params.tPsP)
        rdS = cute.make_fragment_like(acc_dP, self.dtype)
        rdS.store(acc_dP.load().to(self.dtype))
        if cutlass.const_expr(not self.Mma_dKV_is_RS):
            cute.arch.barrier()  # Make sure P is written
        # For hdim 64, It's faster to write to smem_dS first before the dV gemm
        if cutlass.const_expr(not self.Mma_dKV_is_RS):
            tdSrdS = smem_copy_params.r2s_thr_copy_PdS.retile(rdS)
            cute.copy(smem_copy_params.r2s_thr_copy_PdS, tdSrdS, smem_copy_params.tdSsdS)
        if cutlass.const_expr(self.Mma_dKV_is_RS):
            tdVrP = layout_utils.reshape_acc_to_frgA(rP)
        else:
            tdVrP = mma_params.tdVrP

        # MMA dK
        sm80_utils.gemm(
            mma_params.thr_mma_dkv,
            mma_params.acc_dV,
            tdVrP,
            mma_params.tdVrdO,
            smem_copy_params.tdVsPt,
            smem_copy_params.tdVsdOt[
                None, None, None, smem_pipe_read_do if cutlass.const_expr(self.num_stages_dO > 1) else 0
            ],
            smem_copy_params.smem_thr_copy_PdSt,
            smem_copy_params.smem_thr_copy_QdOt,
            A_in_regs=self.Mma_dKV_is_RS,
            swap_AB=self.dKV_swapAB,
        )
        # if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(mma_params.acc_dV)
        cute.arch.barrier()  # Make sure dS is written

        # MMA dQ
        def dQ_mma(hook_fn):
            acc_shape_dQ = mma_params.thr_mma_dq.partition_shape_C(
                (self.m_block_size, self.head_dim_padded)
                if cutlass.const_expr(not self.dQ_swapAB)
                else (self.head_dim_padded, self.m_block_size)
            )
            acc_dQ = cute.make_rmem_tensor(acc_shape_dQ, cutlass.Float32)
            acc_dQ.fill(0.0)
            sm80_utils.gemm(
                mma_params.thr_mma_dq,
                acc_dQ,
                mma_params.tdQrdS,
                mma_params.tdQrK,
                smem_copy_params.tdQsdS,
                smem_copy_params.tdQsKt,
                smem_copy_params.smem_thr_copy_dS,
                smem_copy_params.smem_thr_copy_Kt,
                swap_AB=self.dQ_swapAB,
                hook_fn=hook_fn,
            )
            # ((1, 1), num_elements)
            acc_dQ_atomic = gmem_copy_params.gmem_thr_copy_dQaccum.retile(acc_dQ)
            tdQgdQaccum_atomic = gmem_copy_params.tdQgdQaccum[None, None, m_block]
            assert cute.size(acc_dQ_atomic) == cute.size(tdQgdQaccum_atomic)
            for i in cutlass.range(cute.size(acc_dQ_atomic), unroll_full=True):
                utils.atomic_add_fp32(acc_dQ_atomic[i], utils.elem_pointer(tdQgdQaccum_atomic, i))
                # utils.atomic_add_fp32(acc_dQ[i], tdQgdQaccum_atomic.iterator + i * tdQgdQaccum_atomic.stride[1])
            # if cute.arch.thread_idx()[0] == 64 and cute.arch.block_idx()[0] == bidx: cute.print_tensor(acc_dQ)

        # If num_stages_Q == 1, we want to do Mma_dK first so we can start loading Q for the next iteration
        if cutlass.const_expr(self.num_stages_Q > 1):
            dQ_mma(load_dO_next)

        # MMA dK
        if cutlass.const_expr(self.Mma_dKV_is_RS):
            tdKrdS = layout_utils.reshape_acc_to_frgA(rdS)
        else:
            tdKrdS = mma_params.tdKrdS
        sm80_utils.gemm(
            mma_params.thr_mma_dkv,
            mma_params.acc_dK,
            tdKrdS,
            mma_params.tdKrQ,
            smem_copy_params.tdKsdSt,
            smem_copy_params.tdKsQt[None, None, None, smem_pipe_read_q if cutlass.const_expr(self.num_stages_Q > 1) else 0],
            smem_copy_params.smem_thr_copy_PdSt,
            smem_copy_params.smem_thr_copy_QdOt,
            A_in_regs=self.Mma_dKV_is_RS,
            swap_AB=self.dKV_swapAB,
            hook_fn=load_dO_next if cutlass.const_expr(self.num_stages_Q == 1) else None,
        )
        # if cute.arch.thread_idx()[0] == 0: cute.print_tensor(mma_params.acc_dK)
        if cutlass.const_expr(self.num_stages_Q == 1):
            cute.arch.barrier()
            dQ_mma(load_Q_next)
