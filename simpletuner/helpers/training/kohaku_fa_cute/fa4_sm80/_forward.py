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
from cutlass import Float32, Int32, const_expr
from cutlass.base_dsl.arch import Arch
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import BaseDSL
from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute import utils
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80 as _Original
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.named_barrier import NamedBarrierFwd
from flash_attn.cute.pack_gqa import PackGQA, pack_gqa_layout
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.tile_scheduler import SingleTileScheduler, SingleTileVarlenScheduler, TileSchedulerArguments
from flash_attn.cute.utils import AuxData
from quack import copy_utils, layout_utils

from ._softmax import Softmax


class FlashAttentionForwardSm80(_Original):

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        mCuTotalSplitsMBlocks: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        assert learnable_sink is None, "Learnable sink is not supported in this kernel"
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)
            )
        )
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads
        self.use_tma_O = Arch.sm_90 <= self.arch < Arch.sm_120
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        assert SharedStorage.size_in_bytes() <= self.shared_capacity, "Forward storage exceeds the actual device limit"
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        # Layout permutation: 4D non-varlen vs 3D varlen
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mQ, mO = [cute.make_tensor(t.iterator, cute.select(t.layout, mode=QO_layout_transpose)) for t in (mQ, mO)]
        mK, mV = [cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose)) for t in (mK, mV)]
        if const_expr(mLSE is not None):
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)
        # TileScheduler for varlen, simple grid for non-varlen
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        num_batch = mCuSeqlensQ.shape[0] - 1 if const_expr(mCuSeqlensQ is not None) else cute.size(mQ.shape[3])
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            num_head=cute.size(mQ.shape[2]),
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=mQ.shape[1],
            headdim_v=mV.shape[1],
            total_q=(
                cute.size(mQ.shape[0])
                if const_expr(mCuSeqlensQ is not None)
                else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3])
            ),
            tile_shape_mn=(self.tile_m, self.tile_n),
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            cu_total_m_blocks_ptr=mCuTotalMBlocks,
            cu_total_splits_m_blocks_ptr=mCuTotalSplitsMBlocks,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)
        fastdiv_mods = utils.compute_fastdiv_mods(mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_data.tensors)

        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
            tile_sched_params,
            TileScheduler,
            aux_data,
            fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()

        if work_tile.is_valid_tile:
            m_block, num_head, batch_size, _ = work_tile.tile_idx

            block_info = BlockInfo(
                self.tile_m,
                self.tile_n,
                self.is_causal,
                self.is_local,
                False,  # is_split_kv
                window_size_left,
                window_size_right,
                qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            )
            seqlen = SeqlenInfoQK.create(
                batch_idx=batch_size,
                seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
                seqlen_k_static=mK.shape[0],
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=mSeqUsedK,
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            # For varlen, wasted grid tiles (where batch_idx >= num_batch) will have
            # seqlen_q=seqlen_k=0 and n_block_max=0.  Clamp to 0 so we don't use a
            # negative block index for K/V loads; the load/store predicates already
            # guard all memory accesses when seqlen is 0.
            n_block = cutlass.max(n_block_max - 1, 0)

            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////
            blkQ_shape = (self.tile_m, self.tile_hdim)
            blkK_shape = (self.tile_n, self.tile_hdim)
            blkV_shape = (self.tile_n, self.tile_hdimv)
            num_head_kv = num_head if const_expr(self.pack_gqa) else num_head // self.qhead_per_kvhead
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_size, dim=3)[None, None, num_head]
            if const_expr(not seqlen.has_cu_seqlens_k):
                mK_cur = mK[None, None, num_head_kv, batch_size]
                mV_cur = mV[None, None, num_head_kv, batch_size]
            else:
                mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, num_head_kv])
                mV_cur = cute.domain_offset((seqlen.offset_k, 0), mV[None, None, num_head_kv])
            if const_expr(not self.pack_gqa):
                gQ = cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))
            gK = cute.local_tile(mK_cur, blkK_shape, (None, 0))
            gV = cute.local_tile(mV_cur, blkV_shape, (None, 0))

            # ///////////////////////////////////////////////////////////////////////////////
            # Get shared memory buffer
            # ///////////////////////////////////////////////////////////////////////////////
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.sQ.get_tensor(sQ_layout)
            sK = storage.sK.get_tensor(sK_layout)
            if const_expr(not self.Q_in_regs):
                sV = storage.sV.get_tensor(sV_layout)
            else:
                sV = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout)
            # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
            sVt = layout_utils.transpose_view(sV)

            gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
            gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
            # (CPY_Atom, CPY_N, CPY_K, n_block)
            tKsK, tKgK = gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK)
            # (CPY_Atom, CPY_N, CPY_K, n_block)
            tVsV, tVgV = gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV)

            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            thr_mma_qk = tiled_mma_qk.get_slice(tidx)
            thr_mma_pv = tiled_mma_pv.get_slice(tidx)
            tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
            tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
            tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
            acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
            acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
            acc_O.fill(0.0)

            # ///////////////////////////////////////////////////////////////////////////////
            # Smem copy atom tiling
            # ///////////////////////////////////////////////////////////////////////////////
            smem_copy_atom_QK = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.dtype,
            )
            smem_copy_atom_V = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.dtype,
            )
            smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
            smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
            smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)

            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSsK = smem_thr_copy_K.partition_S(sK)
            tOsVt = smem_thr_copy_V.partition_S(sVt)

            # ///////////////////////////////////////////////////////////////////////////////
            # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
            # of tile_shape
            # ///////////////////////////////////////////////////////////////////////////////
            # Construct identity layout for KV
            cK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
            tKcK = gmem_thr_copy_K.partition_S(cK)
            t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)
            if const_expr(self.tile_hdim == self.tile_hdimv):
                tVcV = tKcK
                t0VcV = t0KcK
            else:
                cV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
                tVcV = gmem_thr_copy_V.partition_S(cV)
                t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cV)
            # Allocate predicate tensors for m and n, here we only allocate the tile of k, and
            # use "if" on the mn dimension.
            # This is to reduce register pressure and gets 2-3% performance gain.
            tKpK = utils.predicate_k(tKcK, limit=mK.shape[1])
            if const_expr(self.same_hdim_kv):
                tVpV = tKpK
            else:
                tVpV = utils.predicate_k(tVcV, limit=mV.shape[1])

            # shape: (atom_v_m * rest_m)
            softmax = Softmax.create(
                softmax_scale_log2,
                num_rows=acc_O.shape[0][0] * acc_O.shape[1],
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            # group parameters for compute_one_n_block
            mma_params = SimpleNamespace(
                thr_mma_qk=thr_mma_qk,
                thr_mma_pv=thr_mma_pv,
                tSrQ=tSrQ,
                tSrK=tSrK,
                tOrVt=tOrVt,
                acc_O=acc_O,
            )
            smem_copy_params = SimpleNamespace(
                smem_thr_copy_Q=smem_thr_copy_Q,
                smem_thr_copy_K=smem_thr_copy_K,
                smem_thr_copy_V=smem_thr_copy_V,
                tSsQ=tSsQ,
                tSsK=tSsK,
                tOsVt=tOsVt,
            )
            load_K = partial(self.load_K, gmem_tiled_copy_K, tKgK, tKsK, tKcK, t0KcK, tKpK, seqlen=seqlen.seqlen_k)
            load_V = partial(self.load_V, gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV, seqlen=seqlen.seqlen_k)

            compute_one_n_block = partial(
                self.compute_one_n_block,
                mma_params=mma_params,
                smem_copy_params=smem_copy_params,
                softmax=softmax,
                load_K=load_K,
                load_V=load_V,
                score_mod=self.score_mod,
                batch_idx=batch_size,
                head_idx=num_head,
                m_block=m_block,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )

            # ///////////////////////////////////////////////////////////////////////////////
            # Prologue
            # ///////////////////////////////////////////////////////////////////////////////
            # Start async loads of the last mn-tile, where we take care of the mn residue
            gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
            if const_expr(not self.pack_gqa):
                self.load_Q(gmem_thr_copy_Q, gQ, sQ, m_block, seqlen=seqlen.seqlen_q, headdim=mQ.shape[1])
            else:
                pack_gqa = PackGQA(self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead)
                pack_gqa.load_Q(mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q)
            cute.arch.cp_async_commit_group()

            def preprocess_Q():
                cute.arch.cp_async_wait_group(self.num_stages * 2 - 1)
                if const_expr(self.Q_in_regs):
                    cute.arch.barrier()
                    tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                    cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)

            # If Q_in_regs, we load Q, then load 1 stage of K, then (optionally) rotate Q and
            # read from smem_q to registers, then load V.
            # If !Q_in_regs, we load Q, load all stages of K & V, then (optionally) rotate Q.
            if const_expr(self.Q_in_regs):
                load_K(n_block, smem_pipe_write=0, need_predicates=True)
                cute.arch.cp_async_commit_group()
                preprocess_Q()
                cute.arch.barrier()  # Make sure all threads have read smem_q before loading V

            for stage in cutlass.range_constexpr(self.num_stages):
                if const_expr(not self.Q_in_regs or stage > 0):
                    if stage == 0 or n_block - stage >= 0:
                        load_K(n_block - stage, smem_pipe_write=stage, need_predicates=stage == 0)
                    cute.arch.cp_async_commit_group()
                if const_expr(stage < self.num_stages - 1):
                    if stage == 0 or n_block - stage >= 0:
                        load_V(n_block - stage, smem_pipe_write=stage, need_predicates=stage == 0)
                    cute.arch.cp_async_commit_group()
            if const_expr(not self.Q_in_regs):
                preprocess_Q()

            # ///////////////////////////////////////////////////////////////////////////////
            # Mainloop
            # ///////////////////////////////////////////////////////////////////////////////
            # Start processing of the first n-block.
            # For performance reason, we separate out two kinds of iterations:
            # those that need masking on S, and those that don't.
            # We need masking on S for the very last block when K and V has length not multiple of tile_n.
            # We also need masking on S if it's causal, for the last several blocks.
            mask = AttentionMask(
                self.tile_m,
                self.tile_n,
                seqlen,
                window_size_left,
                window_size_right,
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            )
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_size,
                head_idx=num_head,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods if const_expr(self.mask_mod is not None) else None,
            )

            # First iteration with seqlen masking
            smem_pipe_read = Int32(0)
            smem_pipe_write = Int32(self.num_stages - 1)
            compute_one_n_block(
                n_block,
                smem_pipe_read,
                smem_pipe_write,
                is_first_n_block=True,
                seqlen=seqlen,
                mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
            )
            smem_pipe_read = self.advance_pipeline(smem_pipe_read)
            smem_pipe_write = self.advance_pipeline(smem_pipe_write)
            # Next couple of iterations with causal masking
            if const_expr(self.is_causal or self.is_local):
                n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(seqlen, m_block, n_block_min)
                for n_tile in cutlass.range(n_block_max - 1 - n_block_min_causal_local_mask, unroll=1):
                    n_block = n_block_max - 2 - n_tile
                    compute_one_n_block(
                        n_block,
                        smem_pipe_read,
                        smem_pipe_write,
                        seqlen=seqlen,
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                    )
                    smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                    smem_pipe_write = self.advance_pipeline(smem_pipe_write)
            # The remaining iterations have no masking
            for n_tile in cutlass.range(n_block, unroll=1):
                compute_one_n_block(
                    n_block - n_tile - 1,
                    smem_pipe_read,
                    smem_pipe_write,
                    seqlen=seqlen,
                    is_first_n_block=False,
                    mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False),
                )
                smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                smem_pipe_write = self.advance_pipeline(smem_pipe_write)
            # TODO: local

            # normalize acc_O by row_sum and calculate the lse
            row_scale = softmax.finalize(return_lse=mLSE is not None)
            softmax.rescale_O(acc_O, row_scale)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            # reuse sQ's data iterator
            sO = cute.make_tensor(sQ.iterator, sO_layout)
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                None,
                tiled_mma_pv,
                tidx,
                m_block,
                num_head,
                batch_size,
                maximum=aux_data.tensors[0],
                row_max=softmax.row_max,
            )

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        maximum: cute.Tensor,
        row_max: cute.Tensor,
    ):
        # store acc_O
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        # Make sure all threads have finished reading V
        cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue), number_of_threads=self.num_epilogue_threads)
        smem_copy_atom_O = utils.get_smem_store_atom(self.arch.major * 10 + self.arch.minor, self.dtype)
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        # taccOsO = copy_utils.partition_D_position_independent(smem_thr_copy_O, sO)
        # copy acc O from rmem to smem with the smem copy atom
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        pack_gqa = PackGQA(self.tile_m, self.tile_hdimv, self.check_hdim_v_oob, self.qhead_per_kvhead)

        # Write LSE from rmem -> gmem
        if const_expr(mLSE is not None):
            mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx]
            if const_expr(not self.pack_gqa):
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                gLSE_expanded_layout = cute.append(gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,)))
                gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
                thr_mma = tiled_mma.get_slice(tidx)
                taccOgLSE = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gLSE_expanded))
                assert cute.size(taccOgLSE, mode=[0]) == cute.size(lse)
                taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
                t0accOcO = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cO))
                # Only the thread corresponding to column 0 writes out the lse to gmem
                if taccOcO[0][1] == 0:
                    for m in cutlass.range(cute.size(taccOgLSE.shape[1]), unroll_full=True):
                        if t0accOcO[m, 0][0] < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]:
                            taccOgLSE[m, 0] = lse[m]
                            maximum[batch_idx, head_idx, m_block * self.tile_m + taccOcO[m, 0][0]] = row_max[m]
            else:
                pack_gqa.store_LSE(mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q)

        ragged = self.use_tma_O and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
        mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=ragged)[None, None, head_idx]
        # thr_mma = tiled_mma.get_slice(tidx)
        # taccOgO = thr_mma.partition_C(gO)
        # cute.autovec_copy(rO, taccOgO)
        # sync to make sure all smem stores are done
        if const_expr(self.use_tma_O):
            # ensure smem writes are visible to TMA
            cute.arch.fence_view_async_shared()
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
            )
            gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
            store_O, _, _ = copy_utils.tma_get_copy_fn(tma_atom_O, 0, cute.make_layout(1), sO, gO, single_stage=True)
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 4:
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.Epilogue),
                    number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
                )
                store_O()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads,
            )
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOrO = cute.make_fragment_like(tOsO, self.dtype)
            # load acc O from smem to rmem for wider vectorization
            cute.autovec_copy(tOsO, tOrO)
            if const_expr(not self.pack_gqa):
                gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
                tOgO = gmem_thr_copy_O.partition_D(gO)
                tOcO = gmem_thr_copy_O.partition_S(cO)
                t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                # copy acc O from rmem to gmem
                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                    if t0OcO[0, rest_m, 0][0] < seqlen.seqlen_q - m_block * self.tile_m - tOcO[0][0]:
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOrO[None, rest_m, None],
                            tOgO[None, rest_m, None],
                            pred=tOpO[None, rest_m, None] if const_expr(self.check_hdim_v_oob) else None,
                        )
            else:
                pack_gqa.store_O(mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_block, seqlen.seqlen_q)
