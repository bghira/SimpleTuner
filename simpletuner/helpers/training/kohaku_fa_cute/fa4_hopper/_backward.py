"""Derived Hopper methods with separate unscaled attention statistics.

Derived from Dao-AILab/flash-attention revision
4616f5116d33e3ba3cbb71d582caffcfcb49649c, BSD-3-Clause.
See LICENSES/FlashAttention-BSD-3-Clause.txt and FlashAttention-AUTHORS.txt.
Unchanged methods use the pinned optional dependency.
"""

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang,
# Vijay Thakkar, Pradeep Ramani, Tri Dao.

import math
from functools import partial
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute import FastDivmodDivisorV2
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum, TensorMapManager, TensorMapUpdateMode
from flash_attn.cute import barrier, pipeline, utils
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparse_utils import (
    consume_block_sparse_mma_bwd_sm90,
    dQaccum_store_block_sparse_bwd_sm90,
    get_total_q_block_count_bwd,
    produce_block_sparse_q_loads_bwd_sm90,
)
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90 as _Original
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.named_barrier import NamedBarrierBwd
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import apply_score_mod_bwd_inner, apply_score_mod_inner
from flash_attn.cute.tile_scheduler import (
    SingleTileLPTBwdScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from flash_attn.cute.utils import AuxData
from quack import copy_utils, layout_utils, sm90_utils
from quack.cute_dsl_utils import ParamsBase
from quack.sm90_utils import gemm_w_idx, gemm_zero_init


class FlashAttentionBackwardSm90(_Original):
    # An explicit experimental subclass may change CTA ownership order without
    # enabling deterministic accumulation or changing the arithmetic policy.
    causal_head_interleave = False
    ordinary_gradient_stores = False
    separate_causal_masking = False

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct, sdQaccum_struct = [
            cute.struct.Align[cute.struct.MemRange[t, cute.cosize(layout)], self.buffer_align_bytes]
            for (layout, t) in [
                (self.sQ_layout, self.dtype),
                (self.sK_layout, self.dtype),
                (self.sV_layout, self.dtype),
                (self.sdO_layout, self.dtype),
                (self.sdQaccum_layout, Float32),
            ]
        ]

        cosize_sdS = cute.cosize(self.sPdS_layout)
        cosize_sP = cute.cosize(self.sPdS_layout) if const_expr(not self.mma_dkv_is_rs) else 0
        sLSE_struct = cute.struct.Align[cute.struct.MemRange[Float32, self.tile_m * self.Q_stage * 2], 128]
        sdPsum_struct = cute.struct.Align[cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.dO_stage], 128]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            mbar_ptr_dO: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sQ: sQ_struct
            sV: sV_struct
            sK: sK_struct
            sdO: sdO_struct
            sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sdS], 1024]
            sdQaccum: sdQaccum_struct

        return SharedStorageQKV

    @cute.jit
    def __call__(
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
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        tensormap_workspace: Optional[cute.Tensor] = None,  # (num_ctas, 2, 16) Int64: K, V descriptors per CTA
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # For GQA (qhead_per_kvhead > 1), multiple Q heads accumulate into the same dK/dV,
        # so we need the float32 accum path + postprocess.
        # For varlen_k with qhead_per_kvhead == 1, we use ragged TMA tensors.
        self.varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None

        self._check_type(
            *(t.element_type if t is not None else None for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV))
        )

        self.is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
        ]

        # Non-varlen inputs are (b, s, n, h), varlen inputs are (s, n, h).
        # We convert both to a seqlen-major view with head-dim second.
        # Each tensor may have different rank when Q is padded (seqused_q) but K/V are unpadded (cu_seqlens_k).
        def _qkv_transpose(t):
            return layout_utils.select(t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1])

        mQ, mK, mV, mdO = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO)]
        if const_expr(self.qhead_per_kvhead == 1):
            mdK, mdV = [_qkv_transpose(t) for t in (mdK, mdV)]
        else:
            # Accum tensors are (b, n, s*h) for non-varlen and (n, s*h) for varlen.
            accum_transpose = [2, 1, 0] if cute.rank(mdK.shape) == 3 else [1, 0]
            mdK, mdV = [layout_utils.select(t, accum_transpose) for t in (mdK, mdV)]
        # Non-varlen stats are (b, n, s), varlen stats are (n, s).
        LSE_dPsum_dQaccum_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
        mLSE, mdPsum, mdQaccum = [layout_utils.select(t, LSE_dPsum_dQaccum_transpose) for t in (mLSE, mdPsum, mdQaccum)]

        tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ = self._get_tiled_mma()
        # (batch, num_head, num_m_blocks, cluster_size) -> (num_m_blocks, cluster_size, num_head, batch)
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=[2, 3, 1, 0])
        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                layout_utils.select(t, mode=[2, 3, 1, 0]) for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self.num_mma_threads = tiled_mma_SdP.size
        assert self.num_mma_threads + 128 == self.num_threads

        self.num_threads_per_warp_group = 128
        self.num_producer_threads = 32

        REG_LIMIT = 504 if self.num_wg_mma == 2 else 512
        if const_expr(self.num_wg_mma == 2):
            if const_expr(self.num_wg_dQ == 1):
                self.num_mma_regs_wg0 = 256
                self.num_mma_regs_wg1 = 224
            else:
                self.num_mma_regs_wg0 = 240
                self.num_mma_regs_wg1 = 240
            self.num_mma_regs = self.num_mma_regs_wg0  # for backward compat
            self.num_producer_regs = 24
            assert self.num_mma_regs_wg0 + self.num_mma_regs_wg1 + self.num_producer_regs <= REG_LIMIT
        else:  # 3 warp groups
            self.num_mma_regs_wg0 = 160
            self.num_mma_regs_wg1 = 160
            self.num_mma_regs = 160
            self.num_producer_regs = 32
            assert self.num_mma_regs_wg0 * self.num_wg_mma + self.num_producer_regs <= REG_LIMIT

        self._setup_attributes()
        # Bound the K/V loads by the logical seqlen_k (see below); decided here because the
        # shared storage needs descriptor slots for it.
        self.ragged_kv_tma = mCuSeqlensK is not None or mSeqUsedK is not None
        if const_expr(self.ragged_kv_tma):
            assert tensormap_workspace is not None, "ragged K/V TMA loads need a tensormap workspace"
        SharedStorage = self._get_shared_storage_cls()

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8 * 2
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = self.tile_m * self.tile_hdim * Float32.width // 8 // self.num_wg_dQ
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.tile_hdim * Float32.width // 8
        self.tma_copy_bytes["dVacc"] = self.tile_n * self.tile_hdimv * Float32.width // 8

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        # Bound the K and V TMA loads by the logical seqlen_k (cu_seqlens_k and/or seqused_k) so
        # rows in [seqlen_k, tile_n) arrive zero-filled instead of as whatever the buffer holds.
        # The masked P / dS are exactly zero there, but 0 * NaN = NaN in dP = dO @ V.T and
        # dQ = dS @ K. The descriptors use the load-safe form of flash_attn.cute.copy_utils
        # (quack's ptr_shift form is store-only, and its wraparound form needs rank <= 3, i.e.
        # only the packed layout); it covers the packed (t, d, h) and the dense (s, d, h, b)
        # layout alike.
        # The load warp rewrites its CTA's copies of the K and V descriptors per work tile with
        # the sequence's row extent (tensormap.replace); a descriptor with a per-launch extent
        # cannot do this (TMA prefetches past the footprint it is given, #3068).
        mK_tma, mV_tma = mK, mV
        mK_gmem, mV_gmem = (mK, mV) if const_expr(self.ragged_kv_tma) else (None, None)
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK_tma,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV_tma,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdimv),
        )
        if const_expr(self.qhead_per_kvhead == 1):
            mdK_tma = copy_utils.create_ragged_tensor_for_tma(mdK, ragged_dim=0, ptr_shift=True) if self.varlen_k else mdK
            mdV_tma = copy_utils.create_ragged_tensor_for_tma(mdV, ragged_dim=0, ptr_shift=True) if self.varlen_k else mdV
            tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdK_tma,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
            )
            tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdV_tma,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
            )
        else:
            tma_atom_dK = tma_atom_dV = tma_tensor_dK = tma_tensor_dV = None

        if const_expr(mCuSeqlensK is not None or mSeqUsedK is not None):
            TileScheduler = SingleTileVarlenScheduler
        elif const_expr(self.deterministic or (self.is_causal and self.causal_head_interleave)):
            TileScheduler = SingleTileLPTBwdScheduler
        else:
            TileScheduler = SingleTileScheduler
        self.spt = (self.is_causal or self.is_local) and self.deterministic
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),
            cute.size(mQ.shape[2]),
            (
                cute.size(mK.shape[3]) if const_expr(mCuSeqlensK is None) else cute.size(mCuSeqlensK.shape[0] - 1)
            ),  # num_batch
            1,  # num_splits
            cute.size(mQ.shape[0]),  # pass seqlen_q or total_q for seqlen_k
            mQ.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=(
                cute.size(mK.shape[0])
                if const_expr(mCuSeqlensK is not None)
                else cute.size(mK.shape[0]) * cute.size(mK.shape[3])
            ),
            tile_shape_mn=(self.tile_n, self.tile_m),  # Swapping the role of Q & K
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.spt,
            head_swizzle=self.deterministic,
            cu_total_m_blocks_ptr=mCuTotalMBlocks,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
        else:
            softmax_scale_log2 = LOG2_E

        fastdiv_mods = None
        if const_expr(aux_data.tensors is not None):
            seqlen_q = cute.size(mQ.shape[0])
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmodDivisorV2(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisorV2(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)

        qhead_per_kvhead_divmod = None
        if const_expr(self.qhead_per_kvhead > 1):
            qhead_per_kvhead_divmod = FastDivmodDivisorV2(self.qhead_per_kvhead)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            mK_gmem,
            mV_gmem,
            tensormap_workspace if const_expr(self.ragged_kv_tma) else None,
            tma_tensor_dO,
            tma_tensor_dK if const_expr(self.qhead_per_kvhead == 1 and not self.ordinary_gradient_stores) else mdK,
            tma_tensor_dV if const_expr(self.qhead_per_kvhead == 1 and not self.ordinary_gradient_stores) else mdV,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dK,
            tma_atom_dV,
            mLSE,
            mdPsum,
            mdQaccum,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sdQaccum_layout,
            self.r2s_tiled_copy_dQaccum,
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
            softmax_scale_log2,
            softmax_scale,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            aux_data,
            fastdiv_mods,
            blocksparse_tensors,
            qhead_per_kvhead_divmod,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            window_size_left,
            window_size_right,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mK_gmem: Optional[cute.Tensor],  # gmem K/V behind tma_atom_K/V (ragged K/V only)
        mV_gmem: Optional[cute.Tensor],
        tensormap_workspace: Optional[cute.Tensor],  # (num_ctas, 2, 16) Int64 (ragged K/V only)
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale_log2,
        softmax_scale,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # prefetch TMA descriptors
        if warp_idx == 0:
            for atom in [tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO, tma_atom_dK, tma_atom_dV]:
                if const_expr(atom is not None):
                    cpasync.prefetch_descriptor(atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
        )
        pipeline_Q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"] + self.tma_copy_bytes["LSE"],
            defer_sync=True,
        )
        pipeline_dO = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_dO.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"] + self.tma_copy_bytes["dPsum"],
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = None
        if const_expr(not self.mma_dkv_is_rs):
            sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sLSE = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.Q_stage),
                stride=(1, self.tile_m * 2),
            )
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.dO_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            # With ragged K/V TMA tensors mode 0 is a sentinel extent, but cu_seqlens_k or
            # seqused_k is then always present and the static length is unused.
            seqlen_k_static=mK.shape[0] if const_expr(not self.ragged_kv_tma) else Int32(0),
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            swap_AB=self.SdP_swapAB,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if warp_idx == 0:
                self.load(
                    mQ,
                    mK,
                    mV,
                    mK_gmem,
                    mV_gmem,
                    tensormap_workspace,
                    mdO,
                    mLSE,
                    mdPsum,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sLSE,
                    sdPsum,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_dO,
                    pipeline_Q,
                    pipeline_dO,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    blocksparse_tensors,
                    qhead_per_kvhead_divmod,
                    maximum=aux_data.tensors[0],
                )
            if warp_idx == 1:
                self.dQaccum_store(
                    mdQaccum,
                    sdQaccum,
                    block_info,
                    TileSchedulerCls,
                    SeqlenInfoCls,
                    blocksparse_tensors,
                    mdQ_semaphore,
                )
        else:
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            mma_args = (
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                tiled_mma_dQ,
                mdK,
                mdV,
                mdK_semaphore,
                mdV_semaphore,
                mdQaccum,
                sQ,
                sK,
                sV,
                sdO,
                sP,
                sdS,
                sLSE,
                sdPsum,
                sdQaccum,
                pipeline_Q,
                pipeline_dO,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                r2s_tiled_copy_dQaccum,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                aux_data,
                fastdiv_mods,
                blocksparse_tensors,
                qhead_per_kvhead_divmod,
            )
            if const_expr(self.num_wg_dQ == self.num_wg_mma):
                # Both WGs compute dQ
                cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                self.mma(*mma_args, is_dQ_wg=True)
            else:
                # WG0 computes dQ, WG1 skips it
                warp_idx_in_mma = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 4
                if warp_idx_in_mma < 4:
                    cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                    self.mma(*mma_args, is_dQ_wg=True)
                else:
                    cute.arch.setmaxregister_increase(self.num_mma_regs_wg1)
                    self.mma(*mma_args, is_dQ_wg=False)

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mK_gmem: Optional[cute.Tensor],
        mV_gmem: Optional[cute.Tensor],
        tensormap_workspace: Optional[cute.Tensor],
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
        maximum: cute.Tensor = None,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        if warp_idx_in_wg == 0:
            producer_state_Q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
            producer_state_dO = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
            )
            tensormap_manager = None
            tensormap_gmem = ()
            desc_ptrs = (None, None)
            if const_expr(self.ragged_kv_tma):
                tensormap_manager = TensorMapManager(TensorMapUpdateMode.GMEM, 128)
                bidx, bidy, bidz = cute.arch.block_idx()
                gdx, gdy, _ = cute.arch.grid_dim()
                cta_linear_idx = bidx + bidy * gdx + bidz * gdx * gdy
                if cta_linear_idx >= cute.size(tensormap_workspace, mode=[0]):
                    # An undersized workspace would silently corrupt memory: fail loudly instead.
                    cute.arch.inline_ptx("trap;", read_only_args=[])
                tensormap_gmem = tuple(
                    tensormap_manager.get_tensormap_ptr(tensormap_workspace[(cta_linear_idx, i, None)].iterator)
                    for i in range(2)
                )
                desc_ptrs = tuple(tensormap_manager.get_tensormap_ptr(p, cute.AddressSpace.generic) for p in tensormap_gmem)
                for atom, gmem_ptr in zip((tma_atom_K, tma_atom_V), tensormap_gmem):
                    tensormap_manager.init_tensormap_from_atom(atom, gmem_ptr, 0)
                tensormap_manager.fence_tensormap_initialization()
            # The K/V extent depends on the batch only: rewrite the descriptors when it changes.
            kv_desc_batch_idx = Int32(-1)
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                n_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                if const_expr(self.ragged_kv_tma) and batch_idx != kv_desc_batch_idx:
                    # Shrink the rows extent (mode 0) of K and V to this sequence's end: packed
                    # layouts count absolute rows, dense layouts per-batch rows. Extents >= 1.
                    kv_desc_batch_idx = batch_idx
                    kv_rows_end = (
                        seqlen.offset_k + seqlen.seqlen_k if const_expr(seqlen.has_cu_seqlens_k) else seqlen.seqlen_k
                    )
                    kv_rows_end = cutlass.max(kv_rows_end, 1)
                    bounded = []
                    for t in (mK_gmem, mV_gmem):
                        shape = list(t.shape)
                        shape[0] = kv_rows_end
                        bounded.append(cute.make_tensor(t.iterator, cute.make_layout(tuple(shape), stride=t.stride)))
                    tensormap_manager.update_tensormap(
                        tuple(bounded), (tma_atom_K, tma_atom_V), tensormap_gmem, 0, (None, None)
                    )
                    for p in tensormap_gmem:
                        tensormap_manager.fence_tensormap_update(p)
                        cute.arch.inline_ptx("prefetch.tensormap [{$r0}];", read_only_args=[p.toint()])
                head_idx_kv = head_idx if const_expr(self.qhead_per_kvhead == 1) else head_idx // qhead_per_kvhead_divmod
                mK_cur, mV_cur = [seqlen.offset_batch_K(t, batch_idx, dim=3)[None, None, head_idx_kv] for t in (mK, mV)]
                kv_block = n_block
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (kv_block, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (kv_block, 0))

                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[None, head_idx]
                mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx]
                mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[None, head_idx]
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
                gdO = cute.local_tile(mdO_cur, (self.tile_m, self.tile_hdimv), (None, 0))
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
                gMAX = cute.local_tile(maximum[batch_idx, head_idx, None], (self.tile_m,), (None,))
                sMAX = cute.make_tensor(sLSE.iterator + self.tile_m, sLSE.layout)
                gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))

                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K,
                    0,
                    cute.make_layout(1),
                    gK,
                    sK,
                    single_stage=True,
                    **({"tma_desc_ptr": desc_ptrs[0]} if const_expr(self.ragged_kv_tma) else {}),
                )
                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V,
                    0,
                    cute.make_layout(1),
                    gV,
                    sV,
                    single_stage=True,
                    **({"tma_desc_ptr": desc_ptrs[1]} if const_expr(self.ragged_kv_tma) else {}),
                )
                load_Q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ)
                load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
                load_dO, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dO, 0, cute.make_layout(1), gdO, sdO)
                load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)
                load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
                load_LSE = copy_utils.tma_producer_copy_fn(load_LSE, pipeline_Q)
                load_MAX = copy_utils.cpasync_bulk_get_copy_fn(gMAX, sMAX)
                load_MAX = copy_utils.tma_producer_copy_fn(load_MAX, pipeline_Q)
                load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)
                load_dPsum = copy_utils.tma_producer_copy_fn(load_dPsum, pipeline_dO)

                m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

                if const_expr(not self.use_block_sparsity):
                    total_m_block_cnt = m_block_max - m_block_min
                    process_tile = const_expr(not self.is_local and not self.is_varlen_q) or m_block_min < m_block_max
                else:
                    total_m_block_cnt = get_total_q_block_count_bwd(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        q_subtile_factor=self.q_subtile_factor,
                        m_block_max=m_block_max,
                    )
                    process_tile = total_m_block_cnt > Int32(0)

                if process_tile:
                    if const_expr(not self.use_block_sparsity):
                        first_m_block = m_block_min
                        pipeline_Q.producer_acquire(producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"])
                        load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
                        load_Q(first_m_block, producer_state=producer_state_Q)
                        # Wait for bwd preprocess to finish writing LSE and dPsum
                        cute.arch.griddepcontrol_wait()
                        load_LSE(first_m_block, producer_state=producer_state_Q)
                        load_MAX(first_m_block, producer_state=producer_state_Q)
                        producer_state_dO_cur = (
                            producer_state_dO if const_expr(self.Q_stage != self.dO_stage) else producer_state_Q
                        )
                        pipeline_dO.producer_acquire(producer_state_dO_cur, extra_tx_count=self.tma_copy_bytes["V"])
                        load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
                        load_dO(first_m_block, producer_state=producer_state_dO_cur)
                        load_dPsum(first_m_block, producer_state=producer_state_dO_cur)
                        producer_state_Q.advance()
                        producer_state_dO.advance()

                        for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                            pipeline_Q.producer_acquire(producer_state_Q)
                            load_Q(m_block, producer_state=producer_state_Q)
                            load_LSE(m_block, producer_state=producer_state_Q)
                            load_MAX(m_block, producer_state=producer_state_Q)
                            producer_state_dO_cur = (
                                producer_state_dO if const_expr(self.Q_stage != self.dO_stage) else producer_state_Q
                            )
                            pipeline_dO.producer_acquire(producer_state_dO_cur)
                            load_dO(m_block, producer_state=producer_state_dO_cur)
                            load_dPsum(m_block, producer_state=producer_state_dO_cur)
                            producer_state_Q.advance()
                            producer_state_dO.advance()
                    else:
                        # The wait for bwd preprocess that the dense arm above does inline sits
                        # in _load_q_do_block_sm90, at the same point in the load order.
                        producer_state_Q, producer_state_dO = produce_block_sparse_q_loads_bwd_sm90(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            n_block,
                            producer_state_Q,
                            producer_state_dO,
                            pipeline_Q,
                            pipeline_dO,
                            load_K,
                            load_V,
                            load_Q,
                            load_dO,
                            load_LSE,
                            load_dPsum,
                            self.tma_copy_bytes["K"],
                            self.tma_copy_bytes["V"],
                            Q_stage_eq_dO_stage=(self.Q_stage == self.dO_stage),
                            q_subtile_factor=self.q_subtile_factor,
                            m_block_max=m_block_max,
                        )

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        mdQaccum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQaccum: cute.Tensor,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
        is_dQ_wg: cutlass.Constexpr[bool] = True,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(self.num_wg_mma, stride=self.num_threads_per_warp_group)
        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        maximum_shape = (self.tile_n, self.tile_m) if self.SdP_swapAB else (self.tile_m, self.tile_n)
        maximum_coords = layout_utils.reshape_acc_to_mn(
            thr_mma_SdP.partition_C(cute.make_identity_tensor(maximum_shape)), transpose=self.SdP_swapAB
        )
        wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dK = tiled_mma_dK.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dQ = None
        if const_expr(is_dQ_wg):
            wg_idx_dQ = warp_group_idx if const_expr(self.num_wg_dQ > 1) else 0
            wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(wg_idx_dQ))
        # S = Q @ K.T
        shape_mnk_S = (self.tile_m, self.tile_n, self.tile_hdim)
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(wg_mma_SdP, shape_mnk_S, sQ, sK, swap_AB=self.SdP_swapAB)
        mma_qk_fn = partial(gemm_zero_init, tiled_mma_SdP, shape_mnk_S[:2], tSrQ, tSrK, swap_AB=self.SdP_swapAB)
        # dP = dO @ V.T
        shape_mnk_dP = (self.tile_m, self.tile_n, self.tile_hdimv)
        _, tdPrdO, tdPrV = sm90_utils.partition_fragment_ABC(wg_mma_SdP, shape_mnk_dP, sdO, sV, swap_AB=self.SdP_swapAB)
        mma_dov_fn = partial(gemm_zero_init, tiled_mma_SdP, shape_mnk_dP[:2], tdPrdO, tdPrV, swap_AB=self.SdP_swapAB)
        # dV += P.T @ dO
        sPt = layout_utils.transpose_view(sP) if sP is not None else None
        sdOt = layout_utils.transpose_view(sdO)
        shape_mnk_dV = (self.tile_n, self.tile_hdimv, self.tile_m)
        acc_dV, tdVrPt, tdVrdOt = sm90_utils.partition_fragment_ABC(
            wg_mma_dV, shape_mnk_dV, sPt, sdOt, swap_AB=self.dKV_swapAB
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tdVrPt, tdVrdOt, swap_AB=self.dKV_swapAB)
        else:
            mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tCrB=tdVrdOt)
        # dK += dS.T @ Q
        sdSt = layout_utils.transpose_view(sdS)
        sQt = layout_utils.transpose_view(sQ)
        shape_mnk_dK = (self.tile_n, self.tile_hdim, self.tile_m)
        acc_dK, tdKrdSt, tdKrQt = sm90_utils.partition_fragment_ABC(
            wg_mma_dK, shape_mnk_dK, sdSt, sQt, swap_AB=self.dKV_swapAB
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tdKrdSt, tdKrQt, swap_AB=self.dKV_swapAB)
        else:
            mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tCrB=tdKrQt)
        # dQ = dS @ K
        sKt = layout_utils.transpose_view(sK)
        shape_mnk_dQ = (self.tile_m, self.tile_hdim, self.tile_n)
        mma_dsk_fn = None
        if const_expr(is_dQ_wg):
            _, tdQrdS, tdQrKt = sm90_utils.partition_fragment_ABC(wg_mma_dQ, shape_mnk_dQ, sdS, sKt, swap_AB=self.dQ_swapAB)
            mma_dsk_fn = partial(
                gemm_zero_init,
                tiled_mma_dQ,
                shape_mnk_dQ[:2],
                tdQrdS,
                tdQrKt,
                swap_AB=self.dQ_swapAB,
            )

        # Smem copy atom tiling for P/dS R2S
        copy_P_r2s = None
        mms_PdS = self.tile_n // (self.num_wg_mma // self.AtomLayoutMSdP)
        if const_expr(sP is not None):
            sP_cpy = sP if const_expr(not self.SdP_swapAB) else sPt
            copy_P_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_SdP,
                sP_cpy,
                tidx,
                transpose=self.SdP_swapAB,
                position_independent=True,
                major_mode_size=mms_PdS,
            )
        sdS_cpy = sdS if const_expr(not self.SdP_swapAB) else sdSt
        copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_SdP,
            sdS_cpy,
            tidx,
            transpose=self.SdP_swapAB,
            position_independent=True,
            major_mode_size=mms_PdS,
        )

        tLSEsLSE = layout_utils.mma_partition_C_vec(
            sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        tLSEsdPsum = layout_utils.mma_partition_C_vec(
            sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        # When shuffle=True, rows are distributed across 8 quads (4 threads each) within a warp.
        # Each thread loads only ceil(num_rows/8) values;
        shfl_copy = copy_utils.tiled_copy_1d(sLSE.element_type, num_threads=8, num_copy_elems=2)
        if const_expr(self.shuffle_LSE):
            tLSEsLSE = shfl_copy.get_slice(cute.arch.lane_idx() // 4).partition_S(tLSEsLSE)
            # ((2, 1), 1, 2) -> (((2, 1), 1), 2)
            tLSEsLSE = cute.group_modes(tLSEsLSE, 0, 2)
        if const_expr(self.shuffle_dPsum):
            tLSEsdPsum = shfl_copy.get_slice(cute.arch.lane_idx() // 4).partition_S(tLSEsdPsum)
            tLSEsdPsum = cute.group_modes(tLSEsdPsum, 0, 2)

        tdQsdQaccum = None
        if const_expr(is_dQ_wg):
            smem_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_slice(tidx)
            tdQsdQaccum = smem_thr_copy_dQaccum.partition_D(sdQaccum)

        PdS_barrier = cutlass.pipeline.NamedBarrier(barrier_id=int(NamedBarrierBwd.PdS), num_threads=self.num_mma_threads)

        mma_one_m_block_all = partial(
            self.mma_one_m_block,
            warp_group_idx=warp_group_idx,
            mma_qk_fn=mma_qk_fn,
            mma_dov_fn=mma_dov_fn,
            mma_pdo_fn=mma_pdo_fn,
            mma_dsq_fn=mma_dsq_fn,
            mma_dsk_fn=mma_dsk_fn,
            copy_P_r2s=copy_P_r2s,
            copy_dS_r2s=copy_dS_r2s,
            pipeline_Q=pipeline_Q,
            pipeline_dO=pipeline_dO,
            tLSEsLSE=tLSEsLSE,
            tLSEsdPsum=tLSEsdPsum,
            tdQsdQaccum=tdQsdQaccum,
            softmax_scale_log2=softmax_scale_log2,
            PdS_barrier=PdS_barrier,
            maximum_coords=maximum_coords,
            maximum=aux_data.tensors[0],
            softmax_scale=softmax_scale,
            # acc_dV=acc_dV,
            # acc_dK=acc_dK,
            is_dQ_wg=is_dQ_wg,
        )

        consumer_state_Q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage)
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            recompute_fastdiv_mods_q = const_expr(
                aux_data.tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = const_expr(
                aux_data.tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )

            if const_expr(fastdiv_mods is not None and fastdiv_mods[0] is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod if not recompute_fastdiv_mods_q else FastDivmodDivisorV2(seqlen.seqlen_q),
                    seqlen_k_divmod if not recompute_fastdiv_mods_k else FastDivmodDivisorV2(seqlen.seqlen_k),
                )

            mask = AttentionMaskCls(seqlen)
            score_mod_fn = partial(
                self.apply_score_mod,
                thr_mma_SdP=thr_mma_SdP,
                softmax_scale=softmax_scale,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
            score_mod_bwd_fn = partial(
                self.apply_score_mod_bwd,
                thr_mma_SdP=thr_mma_SdP,
                softmax_scale=softmax_scale,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
            score_mod_fn_cur = partial(
                score_mod_fn,
                batch_idx=batch_idx,
                head_idx=head_idx,
                n_block=n_block,
                seqlen_info=seqlen,
            )
            score_mod_bwd_fn_cur = partial(
                score_mod_bwd_fn,
                batch_idx=batch_idx,
                head_idx=head_idx,
                n_block=n_block,
                seqlen_info=seqlen,
            )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            if const_expr(not self.use_block_sparsity):
                process_tile = const_expr(not self.is_local and not self.is_varlen_q) or m_block_min < m_block_max
            else:
                total_m_block_cnt = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    q_subtile_factor=self.q_subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = total_m_block_cnt > Int32(0)

            if process_tile:
                if const_expr(not self.use_block_sparsity):
                    mask_fn = partial(
                        mask.apply_mask,
                        batch_idx=batch_idx,
                        head_idx=head_idx,
                        n_block=n_block,
                        thr_mma=thr_mma_SdP,
                        mask_seqlen=True,
                        mask_causal=self.is_causal,
                        mask_local=self.is_local,
                        mask_mod=self.mask_mod,
                        aux_data=aux_data,
                        fastdiv_mods=fastdiv_mods,
                    )
                    dKV_accumulate = False
                    if const_expr(
                        self.separate_causal_masking
                        and self.is_causal
                        and self.tile_hdim == 64
                        and not self.is_local
                        and not self.is_varlen_q
                        and not self.varlen_k
                    ):
                        # Only the diagonal query tiles require a causal
                        # predicate. Keep the key tail predicate in both loops.
                        # This is the same forward query order and pipeline state.
                        masked_end = min(
                            m_block_max,
                            cute.ceil_div(
                                (n_block + 1) * self.tile_n + seqlen.seqlen_q - seqlen.seqlen_k,
                                self.tile_m,
                            ),
                        )
                        for m_block in cutlass.range(m_block_min, masked_end, unroll=1):
                            consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                                m_block,
                                consumer_state_Q,
                                consumer_state_dO,
                                mask_fn=mask_fn,
                                score_mod_fn=score_mod_fn_cur,
                                score_mod_bwd_fn=score_mod_bwd_fn_cur,
                                dKV_accumulate=dKV_accumulate,
                                batch_idx=batch_idx,
                                head_idx=head_idx,
                                seqlen_q=seqlen.seqlen_q,
                            )
                            dKV_accumulate = True
                        m_block_min = masked_end
                        mask_fn = partial(
                            mask.apply_mask,
                            batch_idx=batch_idx,
                            head_idx=head_idx,
                            n_block=n_block,
                            thr_mma=thr_mma_SdP,
                            mask_seqlen=True,
                            mask_causal=False,
                            mask_local=False,
                            mask_mod=self.mask_mod,
                            aux_data=aux_data,
                            fastdiv_mods=fastdiv_mods,
                        )
                    for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                        consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                            m_block,
                            consumer_state_Q,
                            consumer_state_dO,
                            mask_fn=mask_fn,
                            score_mod_fn=score_mod_fn_cur,
                            score_mod_bwd_fn=score_mod_bwd_fn_cur,
                            dKV_accumulate=dKV_accumulate,
                            batch_idx=batch_idx,
                            head_idx=head_idx,
                            seqlen_q=seqlen.seqlen_q,
                        )
                        dKV_accumulate = True
                else:
                    consumer_state_Q, consumer_state_dO = consume_block_sparse_mma_bwd_sm90(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        consumer_state_Q,
                        consumer_state_dO,
                        mma_one_m_block_all,
                        mask,
                        self.mask_mod,
                        is_causal=self.is_causal,
                        is_local=self.is_local,
                        thr_mma_SdP=thr_mma_SdP,
                        score_mod_fn=score_mod_fn_cur,
                        score_mod_bwd_fn=score_mod_bwd_fn_cur,
                        q_subtile_factor=self.q_subtile_factor,
                        m_block_max=m_block_max,
                        aux_data=aux_data,
                        fastdiv_mods=fastdiv_mods,
                    )

                if const_expr(self.qhead_per_kvhead == 1):
                    acc_dK.store(acc_dK.load() * softmax_scale)
                self.epilogue_dKV(
                    acc_dV,
                    mdV,
                    sV,
                    acc_dK,
                    mdK,
                    sK,
                    seqlen,
                    tma_atom_dK,
                    tma_atom_dV,
                    tiled_mma_dK,
                    tiled_mma_dV,
                    tidx,
                    n_block,
                    head_idx,
                    batch_idx,
                    qhead_per_kvhead_divmod,
                    mdK_semaphore,
                    mdV_semaphore,
                )
            else:
                # KV tile with zero Q blocks produces no dK/dV; write zeros.
                if const_expr(self.use_block_sparsity or self.is_local or self.is_varlen_q):
                    acc_dK.fill(0.0)
                    acc_dV.fill(0.0)
                    self.epilogue_dKV(
                        acc_dV,
                        mdV,
                        sV,
                        acc_dK,
                        mdK,
                        sK,
                        seqlen,
                        tma_atom_dK,
                        tma_atom_dV,
                        tiled_mma_dK,
                        tiled_mma_dV,
                        tidx,
                        n_block,
                        head_idx,
                        batch_idx,
                        qhead_per_kvhead_divmod,
                        mdK_semaphore,
                        mdV_semaphore,
                    )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def mma_one_m_block(
        self,
        m_block: Int32,
        consumer_state_Q: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        consumer_state_dO: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        warp_group_idx: Int32,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_pdo_fn: Callable,
        mma_dsq_fn: Callable,
        mma_dsk_fn: Callable,
        copy_P_r2s: Optional[Callable],
        copy_dS_r2s: Callable,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        tdQsdQaccum: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        PdS_barrier: cutlass.pipeline.NamedBarrier,
        maximum_coords: cute.Tensor,
        maximum: cute.Tensor,
        softmax_scale: Float32,
        batch_idx: Int32,
        head_idx: Int32,
        seqlen_q: Int32,
        is_dQ_wg: cutlass.Constexpr[bool] = True,
        mask_fn: Optional[Callable] = None,
        score_mod_fn: Optional[Callable] = None,
        score_mod_bwd_fn: Optional[Callable] = None,
        dKV_accumulate: Boolean = True,
    ):
        consumer_state_dO_cur = consumer_state_Q if const_expr(self.Q_stage == self.dO_stage) else consumer_state_dO
        smem_idx_Q = consumer_state_Q.index
        smem_idx_dO = consumer_state_dO_cur.index if const_expr(self.dO_stage > 1) else 0
        smem_idx_PdS = smem_idx_Q if const_expr(self.PdS_stage > 1) else 0
        # (1) [GEMM 1] S = Q @ K^T
        pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
        acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)
        # If shuffle_LSE, OOB reads are OK since sLSE is already padded
        tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx_Q])
        tMAXsMAX = cute.make_tensor(tLSEsLSE.iterator + self.tile_m, tLSEsLSE.layout)
        tMAXrMAX = copy_utils.load_s2r(tMAXsMAX[None, smem_idx_Q])
        # (2) [GEMM 2] dP = dO @ V.T
        pipeline_dO.consumer_wait(consumer_state_dO_cur, pipeline_dO.consumer_try_wait(consumer_state_dO_cur))
        acc_dP = mma_dov_fn(A_idx=smem_idx_Q, wg_wait=1)

        if const_expr(self.score_mod_bwd is not None):
            acc_S_pre = cute.make_fragment_like(acc_S)
            cute.autovec_copy(acc_S, acc_S_pre)

        if const_expr(self.score_mod is not None):
            score_mod_fn(acc_S, m_block=m_block)

        # (3) [Pointwise 1] P = exp(S - LSE)
        if cutlass.const_expr(mask_fn is not None):
            mask_fn(acc_S, m_block=m_block)
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
        lane_idx = cute.arch.lane_idx()
        magnitude = softmax_scale
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            lse_val = self._get_stat(tLSErLSE, r, lane_idx, shuffle=self.shuffle_LSE)
            query_row = m_block * self.tile_m + maximum_coords[r, 0][1 if const_expr(self.SdP_swapAB) else 0]
            row_maximum = self._get_stat(tMAXrMAX, r, lane_idx, shuffle=self.shuffle_LSE)
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                shifted = acc_S_mn[r, c] - row_maximum
                probability = cute.math.exp2(shifted * softmax_scale_log2 - lse_val, fastmath=True)
                acc_S_mn[r, c] = probability
        tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx_dO])

        # Convert P from f32 -> f16
        tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)
        layout_utils.reshape_acc_to_frgA(acc_S).store(tdVrP.load().to(Float32))
        # R2S for P
        if const_expr(not self.mma_dkv_is_rs):
            # sync to ensure P has already been used in the previous iteration before overwriting
            if const_expr(self.PdS_stage == 1):
                PdS_barrier.arrive_and_wait()
            copy_P_r2s(tdVrP, dst_idx=smem_idx_PdS)

        # (4) [Pointwise 2] dS = P*(dP-dPsum)
        warpgroup.wait_group(0)
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=self.SdP_swapAB)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            dpsum_val = self._get_stat(tLSErdPsum, r, lane_idx, shuffle=self.shuffle_dPsum)
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val)

        if const_expr(self.score_mod_bwd is not None):
            score_mod_bwd_fn(acc_dP, acc_S_pre, m_block=m_block)

        # Convert dS from f32 -> f16
        tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)

        # If there's double buffering on dS, we don't need to sync here.
        # Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
        # But because both WGs have to sync at the end of the loop and double buffering,
        # this race condition is not possible.
        # This sync is to ensure (1) P is written in case of !mma_dkv_is_rs and
        # (2) dS is already read by the Mma in the previous iteration in case of mma_dkv_is_rs.
        if const_expr(not self.mma_dkv_is_rs or (self.PdS_stage == 1 and self.mma_dkv_is_rs)):
            cute.arch.fence_view_async_shared()
            PdS_barrier.arrive_and_wait()

        # R2S for dS
        copy_dS_r2s(tdKrdS, dst_idx=smem_idx_PdS)

        # (5) [GEMM 3] dV += P.T @ dO
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn(A_idx=smem_idx_PdS, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1)
        else:
            mma_pdo_fn(tCrA=tdVrP, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1)

        # smem fence to make sure sdS is written before it's read by WGMMA
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()

        if const_expr(is_dQ_wg):
            # (6) [GEMM 4] dQ = dS @ K
            acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)
            pipeline_dO.consumer_release(consumer_state_dO_cur)  # release dO as dV mma is done

            # (7) [GEMM 5] dK += dS.T @ Q
            if const_expr(not self.mma_dkv_is_rs):
                mma_dsq_fn(A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)
            else:
                mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)

            # dQ R2S: wait for dQaccum_store to free the smem buffer, then write dQ to smem
            # When dQ_single_wg, only WG0 enters here so warp_group_idx == 0
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )
            tdQrdQaccum_flat = cute.make_tensor(acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape))
            cute.autovec_copy(tdQrdQaccum_flat, tdQsdQaccum)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )

            warpgroup.wait_group(0)
            pipeline_Q.consumer_release(consumer_state_Q)
        else:
            # dQ_single_wg: WG1 skips dQ, only does dV wait + dK
            # (7) [GEMM 5] dK += dS.T @ Q
            if const_expr(not self.mma_dkv_is_rs):
                mma_dsq_fn(A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)
            else:
                mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)
            pipeline_dO.consumer_release(consumer_state_dO_cur)
            warpgroup.wait_group(0)
            pipeline_Q.consumer_release(consumer_state_Q)

        consumer_state_Q.advance()
        consumer_state_dO.advance()
        return consumer_state_Q, consumer_state_dO
