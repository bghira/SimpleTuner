"""Derived Hopper methods with separate unscaled attention statistics.

Derived from Dao-AILab/flash-attention revision
4616f5116d33e3ba3cbb71d582caffcfcb49649c, BSD-3-Clause.
See LICENSES/FlashAttention-BSD-3-Clause.txt and FlashAttention-AUTHORS.txt.
Unchanged methods use the pinned optional dependency.
"""

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang,
# Vijay Thakkar, Pradeep Ramani, Tri Dao.

from functools import partial
from types import SimpleNamespace
from typing import Callable, Literal, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, const_expr, pipeline
from cutlass.base_dsl.arch import Arch
from cutlass.cute import FastDivmodDivisorV2
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum, TensorMapManager, TensorMapUpdateMode
from flash_attn.cute import pipeline as pipeline_custom
from flash_attn.cute import utils
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparse_utils import consume_block_sparse_loads, produce_block_sparse_loads
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.config import Sm90Registers
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.named_barrier import NamedBarrierFwd
from flash_attn.cute.pack_gqa import PackGQA, make_packgqa_tiled_tma_atom, pack_gqa_layout
from flash_attn.cute.paged_kv import PagedKVManager
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import apply_score_mod_inner
from flash_attn.cute.tile_scheduler import (
    SingleTileLPTScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    StaticPersistentTileScheduler,
    TileSchedulerArguments,
)
from flash_attn.cute.utils import AuxData
from quack import copy_utils, layout_utils, sm90_utils
from quack.cute_dsl_utils import ParamsBase

from ._forward import FlashAttentionForwardSm90
from ._softmax import Softmax


class PersistentFlashAttentionForwardSm90(FlashAttentionForwardSm90):

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        assert not self.Q_in_regs, "Persistent output storage requires separate Q/V"
        sO_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sO_layout)], self.buffer_align_bytes]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]
        cosize_sP = cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
        # 1 stage * 2 for Q pipeline (full + empty), self.num_stages*2 for K, self.num_stages*2 for V,
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sO: sO_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        mCuTotalSplitsMBlocks: Optional[cute.Tensor] = None,
        tensormap_workspace: Optional[cute.Tensor] = None,  # (num_ctas, 16) Int64, one V descriptor per CTA
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """

        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)
            )
        )

        self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mO = [layout_utils.select(t, QO_layout_transpose) for t in (mQ, mO)]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE = layout_utils.select(mLSE, LSE_layout_transpose) if const_expr(mLSE is not None) else None

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma in [1, 2, 3]
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group  # If not TMA_Q
        self.num_epilogue_threads = self.num_mma_threads
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        self.use_scheduler_barrier = (
            (self.num_wg_mma >= 2 and self.tile_hdim <= 128) if const_expr(self.intra_wg_overlap) else (self.num_wg_mma == 2)
        )
        self.use_tma_Q = self.arch >= Arch.sm_90 and not (self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0)
        self.use_tma_O = self.use_tma_Q
        self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap
        self._setup_attributes()
        # TODO: we prob don't need most of what's in _setup_attributes
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage)
            for mX, shape, stage in [
                (mQ, (self.tile_m, self.tile_hdim), None),
                (mK, (self.tile_n, self.tile_hdim), self.num_stages),
                (mV, (self.tile_n, self.tile_hdimv), self.num_stages),
                (mO, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = sm90_utils.make_smem_layout(mV.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n))

        # Bound V loads by the logical sequence length (see below); decided here because the
        # shared storage needs a descriptor slot for it.
        self.ragged_tma_V = (
            self.use_tma_KV
            and mPageTable is None
            and (mCuSeqlensK is not None or mSeqUsedK is not None)
            and not (isinstance(mV.stride[0], int) and mV.stride[0] == 0)
        )
        if const_expr(self.ragged_tma_V):
            assert tensormap_workspace is not None, "ragged V TMA loads need a tensormap workspace"
        SharedStorage = self._get_shared_storage_cls()
        assert SharedStorage.size_in_bytes() <= 232448, "Forward storage exceeds the actual device limit"

        mQ_og, mO_og = mQ, mO
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
            ]
        }
        make_tiled_tma_atom_fn = (
            partial(make_packgqa_tiled_tma_atom, qhead_per_kvhead=self.qhead_per_kvhead, head_idx=2)
            if const_expr(self.pack_gqa)
            else cpasync.make_tiled_tma_atom
        )
        tma_atom_Q, tma_tensor_Q = None, None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                gmem_tiled_copy_Q,
                mQ_og if const_expr(self.pack_gqa) else mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),  # No mcast
            )
        tma_atom_K, tma_tensor_K = None, None
        tma_atom_V, tma_tensor_V = None, None
        # Bound the V TMA load by the logical seqlen_k (cu_seqlens_k and/or seqused_k) so rows
        # in [seqlen_k, tile_n) arrive zero-filled instead of as whatever the buffer holds. P is
        # exactly zero there, but 0 * NaN = NaN in O = P @ V. K needs no bound: masked scores
        # become -inf before exp. The descriptor is the load-safe form of
        # flash_attn.cute.copy_utils (quack's ptr_shift form is store-only, and its wraparound
        # form needs rank <= 3, i.e. only the packed (t, d, h) layout); it covers both the packed
        # and the dense (s, d, h, b) layout. Paged KV is excluded (finite rows past seqused_k in
        # a page are a caller contract), as are broadcast (seq-stride 0) V views.
        # The load warp rewrites its CTA's copy of the V descriptor per work tile with the
        # sequence's row extent (tensormap.replace), so rows at or past seqlen_k are out of
        # bounds for TMA and arrive zero-filled; a descriptor with a per-launch extent cannot
        # do this (TMA prefetches past the footprint it is given, #3068).
        mV_gmem = mV if const_expr(self.ragged_tma_V) else None
        mV_tma = mV
        if const_expr(self.use_tma_KV):
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,  # No mcast for now
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV_tma,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
                1,  # No mcast for now
            )
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.pack_gqa) else mO
            if const_expr(self.varlen_q):
                mO_tma = copy_utils.create_ragged_tensor_for_tma(mO_tma, ragged_dim=0, ptr_shift=True)
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                gmem_tiled_copy_O,
                mO_tma,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),  # No mcast
            )
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            # TODO: dispatch to DynamicPersistentVarlenScheduler when appropriate
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                StaticPersistentTileScheduler if const_expr(not self.is_causal or self.is_local) else SingleTileLPTScheduler
            )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]) if const_expr(mCuSeqlensQ is None) else cute.size(mCuSeqlensQ.shape[0] - 1),
            1,  # num_splits
            cute.size(mK.shape[0]) if const_expr(mPageTable is None) else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[1],
            total_q=(
                cute.size(mQ.shape[0])
                if const_expr(mCuSeqlensQ is not None)
                else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3])
            ),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype.width // 8,
            is_persistent=not self.is_causal,
            lpt=self.is_causal or self.is_local,
            cu_total_m_blocks_ptr=mCuTotalMBlocks,
            cu_total_splits_m_blocks_ptr=mCuTotalSplitsMBlocks,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None
        fastdiv_mods = utils.compute_fastdiv_mods(mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_data.tensors, mPageTable)

        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K if const_expr(self.use_tma_KV) else mK,
            tma_tensor_V if const_expr(self.use_tma_KV) else mV,
            mV_gmem,
            tensormap_workspace if const_expr(self.ragged_tma_V) else None,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
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
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            aux_data,
            fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mV_gmem: Optional[cute.Tensor],  # gmem V behind tma_atom_V (ragged V only)
        tensormap_workspace: Optional[cute.Tensor],  # (num_ctas, 16) Int64 (ragged V only)
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
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
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier / pipeline init
        mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        if const_expr(self.use_tma_KV):
            pipeline_k = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
            pipeline_v = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["V"],
                defer_sync=True,
            )
        else:
            pipeline_k = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )
            pipeline_v = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type)
        # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
        sVt = layout_utils.transpose_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        # reuse sQ's data iterator
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)

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
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0] if const_expr(mPageTable is None) else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=(blocksparse_tensors.cu_total_m_blocks if blocksparse_tensors is not None else None),
            mCuBlockIdxOffsets=(blocksparse_tensors.cu_block_idx_offsets if blocksparse_tensors is not None else None),
            # Don't need to pass in tile_mn because we won't access offset_padded
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        # Cluster wait before starting
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx < 4:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                mV_gmem,
                tensormap_workspace,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_Q,
                mPageTable,
                blocksparse_tensors,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

        else:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mLSE,
                sQ,
                sK,
                sVt,
                sP,
                sO,
                learnable_sink,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                blocksparse_tensors,
                aux_data,
                fastdiv_mods,
            )
