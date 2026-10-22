"""Shared-tile attention with unscaled softmax maxima on Ada, Hopper and Blackwell.

The accumulator layout transforms follow NVIDIA's BSD-licensed CuTe Ampere
attention example; see LICENSE. The attention and gradient kernels are local.
"""

from functools import lru_cache
from math import prod

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.memory import SmemAllocator, TmemAllocator

from .blackwell import dot as blackwell_dot


@cute.jit
def row_view(acc: cute.Tensor):
    layout = cute.make_layout(acc.layout.shape)
    if cutlass.const_expr(len(layout.shape[0]) == 3):
        mn = cute.make_layout(
            ((layout.shape[0][1], layout.shape[1]), (layout.shape[0][0], layout.shape[0][2], layout.shape[2])),
            stride=((layout.stride[0][1], layout.stride[1]), (layout.stride[0][0], layout.stride[0][2], layout.stride[2])),
        )
    else:
        mn = cute.make_layout(
            ((layout.shape[0][1], layout.shape[1]), (layout.shape[0][0], layout.shape[2])),
            stride=((layout.stride[0][1], layout.stride[1]), (layout.stride[0][0], layout.stride[2])),
        )
    return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn))


@cute.jit
def operand_a(acc: cute.Tensor, dtype: cutlass.Constexpr):
    values = cute.make_fragment_like(acc, dtype)
    values.store(acc.load().to(dtype))
    divided = cute.logical_divide(values.layout, (None, None, 2))
    layout = cute.make_layout(
        ((divided.shape[0], divided.shape[2][0]), divided.shape[1], divided.shape[2][1]),
        stride=((divided.stride[0], divided.stride[2][0]), divided.stride[1], divided.stride[2][1]),
    )
    return cute.make_tensor(values.iterator, layout)


@cute.jit
def quad_sum(value: cutlass.Float32):
    value += cute.arch.shuffle_sync_bfly(value, offset=2, mask=-1, mask_and_clamp=31)
    value += cute.arch.shuffle_sync_bfly(value, offset=1, mask=-1, mask_and_clamp=31)
    return value


@cute.jit
def quad_max(value: cutlass.Float32):
    value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=2, mask=-1, mask_and_clamp=31))
    value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=1, mask=-1, mask_and_clamp=31))
    return value


@cute.jit
def hopper_operand(acc, operand_layout, dtype: cutlass.Constexpr):
    c, a = acc.layout, operand_layout
    layout = cute.make_layout(
        (a, c.shape[1], (c.shape[2], cute.size(c, mode=[0]) // cute.size(a))),
        stride=(c.stride[0], c.stride[1], (c.stride[2], cute.size(a, mode=[2]) * c.stride[0][2])),
    )
    operand = cute.make_rmem_tensor_like(layout, dtype)
    if cutlass.const_expr(acc.element_type is dtype):
        cute.make_tensor(operand.iterator, acc.layout).store(acc.load())
    else:
        packed_half_conversion(acc, cute.make_tensor(operand.iterator, acc.layout))
    return operand


@dsl_user_op
def pack_half_pair(a, b, dtype, *, loc=None, ip=None):
    instruction = "cvt.rn.bf16x2.f32 $0, $2, $1;" if dtype is cutlass.BFloat16 else "cvt.rn.f16x2.f32 $0, $2, $1;"
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip), cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
            instruction,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def packed_half_conversion(source, destination):
    packed = cute.recast_tensor(destination, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(packed)):
        packed[i] = pack_half_pair(source[2 * i], source[2 * i + 1], destination.element_type)


class Attention:
    def __init__(
        self,
        mode,
        causal,
        masked,
        columns=32,
        width=64,
        warps=4,
        hopper=False,
        centered=False,
        tma=False,
        pipelined=False,
        tcgen=False,
        sm103_rows=0,
        vector=8,
        reload_center=False,
        query_column_warps=1,
        stage_statistics=False,
    ):
        self.mode, self.causal, self.masked = mode, causal, masked
        self.columns, self.width, self.warps = columns, width, warps
        self.hopper, self.centered, self.tma = hopper, centered, tma
        self.pipelined = pipelined
        self.tcgen = tcgen
        self.sm103_rows = sm103_rows
        self.reload_center = reload_center
        self.stage_statistics = stage_statistics
        if stage_statistics and (mode != "kv" or hopper or warps != 4 or not centered):
            raise ValueError("Shared KV statistics require a centered four-warp generic KV kernel.")
        self.pointwise_overlap = hopper and centered and width == 128 and mode != "forward"
        # Only the measured full-output forward geometry opts into N-warp sharing.
        self.forward_columns = (
            mode == "forward"
            and not masked
            and width == 512
            and columns == 32
            and warps == 4
            and not hopper
            and not tma
            and not pipelined
            and not tcgen
            and not sm103_rows
        )
        self.split_kv = (
            mode == "kv"
            and centered
            and not masked
            and width == 512
            and columns == 32
            and warps == 4
            and not (hopper or tma or pipelined or tcgen or sm103_rows)
        )
        self.column_warps = 2 if not hopper and (mode == "kv" and width > 128 or self.forward_columns) else 1
        if self.split_kv:
            self.column_warps = 4
        if mode == "q":
            self.column_warps = query_column_warps
        self.split_query = (
            mode == "q"
            and centered
            and not masked
            and width == 512
            and columns == 32
            and warps == 4
            and self.column_warps == 4
            and not (hopper or tma or pipelined or tcgen or sm103_rows)
        )
        self.score_column_warps = 2 if self.split_query or self.split_kv else self.column_warps
        self.score_warps = 2 if self.split_query or self.split_kv else warps
        self.rows = 64 if hopper else 16 * (warps // self.column_warps)
        if tcgen and (hopper or warps != 4 or self.rows != 64 or columns not in (32, 64)):
            raise ValueError("tcgen05 score tiles require four warps and a 64x32 or 64x64 tile.")
        self.grid_rows = 128 if hopper and mode == "forward" and warps == 8 else self.rows
        self.specialized_load = False
        self.vector = vector
        self.product_width = width // 2 if hopper and mode == "kv" and warps == 8 else width

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        mask: cute.Tensor,
        out: cute.Tensor,
        maximum: cute.Tensor,
        logsum: cute.Tensor,
        dout: cute.Tensor,
        delta: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        blocks: cute.Tensor,
        delta_base: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.hopper):
            mma = cute.make_tiled_mma(
                warpgroup.MmaF16BF16Op(
                    q.element_type,
                    cutlass.Float32,
                    (64, self.columns, 16),
                    warpgroup.OperandSource.SMEM,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.K,
                )
            )
            mma.set(warpgroup.Field.ACCUMULATE, True)
        else:
            mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
                (self.warps // self.column_warps, self.column_warps, 1),
                permutation_mnk=(self.rows, 16 * self.column_warps, 16),
            )
        length = k.shape[2] if cutlass.const_expr(self.mode == "kv") else q.shape[2]
        heads = k.shape[1] if cutlass.const_expr(self.mode == "kv") else q.shape[1]
        widths = cute.ceil_div(q.shape[3], self.width)
        tma_args = (None, None, None, None)
        if cutlass.const_expr(self.tma):
            padded = cute.ceil_div(q.shape[3], self.width) * self.width
            atom_layout = warpgroup.make_smem_layout_atom(
                warpgroup.SmemLayoutAtomKind.K_SW128 if padded % 64 == 0 else warpgroup.SmemLayoutAtomKind.K_SW64,
                q.element_type,
            )
            right_layout = cute.tile_to_shape(atom_layout, (self.columns, padded), (0, 1))
            if cutlass.const_expr(self.mode == "kv"):
                source_a, source_b = q, dout
            else:
                source_a, source_b = k, v
            tma_a, ta = self.create_tma(source_a, right_layout, padded)
            tma_b, tb = self.create_tma(source_b, right_layout, padded)
            tma_args = (tma_a, ta, tma_b, tb)
        self.kernel(
            q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, delta_base, scale, mma, tma_args
        ).launch(
            grid=(cute.ceil_div(length, self.grid_rows), q.shape[0] * heads, widths),
            block=(32 * self.warps, 1, 1),
            stream=stream,
        )

    @cute.jit
    def create_tma(self, source, layout, padded):
        # Keep singleton sequence axes in the TMA descriptor.
        sequence = (
            cutlass.Int32(source.shape[2]) if cute.is_static(source.shape[2]) and source.shape[2] == 1 else source.shape[2]
        )
        tensor = cute.make_tensor(
            source.iterator,
            cute.make_layout(
                (sequence, source.shape[3], (source.shape[1], source.shape[0])),
                stride=(source.stride[2], source.stride[3], (source.stride[1], source.stride[0])),
            ),
        )
        return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), tensor, layout, (self.columns, padded))

    @cute.jit
    def load_tma(self, atom, source, shared, batch, head, start, barrier):
        tile = cute.local_tile(
            source, (self.columns, cute.size(shared, mode=[1])), (start // self.columns, 0, (head, batch))
        )
        dest, src = cpasync.tma_partition(
            atom, 0, cute.make_layout(1), cute.group_modes(shared, 0, 2), cute.group_modes(tile, 0, 2)
        )
        cute.copy(atom, src, dest, tma_bar_ptr=barrier)

    @cute.jit
    def issue_pair(self, stages, other_stages, batch, head, block, ordinal, start, tid, args, barrier, blocks):
        atom_a, ta, atom_b, tb = args
        slot = ordinal % 2
        if cutlass.const_expr(self.mode == "kv"):
            active, _ = self.block_visibility(blocks, batch, head, block * self.columns, start, self.columns, self.rows)
        else:
            active, _ = self.block_visibility(blocks, batch, head, start, block * self.columns, self.rows, self.columns)
        if tid < 32:
            with cute.arch.elect_one():
                size = 4 * self.columns * cute.size(stages, mode=[1]) if active else 0
                cute.arch.mbarrier_arrive_and_expect_tx(barrier + slot, size)
            if active:
                self.load_tma(atom_a, ta, stages[None, None, slot], batch, head, block * self.columns, barrier + slot)
                self.load_tma(atom_b, tb, other_stages[None, None, slot], batch, head, block * self.columns, barrier + slot)

    @cute.jit
    def pipeline_step(self, stages, other_stages, batch, head, block, ordinal, stop, start, tid, args, barrier, blocks):
        slot = ordinal % 2
        cute.arch.mbarrier_wait(barrier + slot, (ordinal // 2) % 2)
        if block + 1 < stop:
            self.issue_pair(stages, other_stages, batch, head, block + 1, ordinal + 1, start, tid, args, barrier, blocks)
        return stages[None, None, slot], other_stages[None, None, slot]

    @cute.jit
    def load_pair(self, source_a, source_b, right, other_right, batch, head, start, tid, tma_args, barrier, phase):
        if cutlass.const_expr(self.tma):
            tma_a, ta, tma_b, tb = tma_args
            if tid < 32:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(barrier, 4 * cute.size(right))
                self.load_tma(tma_a, ta, right, batch, head, start, barrier)
                self.load_tma(tma_b, tb, other_right, batch, head, start, barrier)
            cute.arch.mbarrier_wait(barrier, phase)
            phase = phase ^ 1
        else:
            self.load(source_a, right, batch, head, start, tid)
            self.load(source_b, other_right, batch, head, start, tid)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        return phase

    @cute.jit
    def load(self, source, shared, batch, head, start, tid):
        if cutlass.const_expr(self.vector > 1):
            rows, width = cute.size(shared, mode=[0]), cute.size(shared, mode=[1])
            atom = cute.make_copy_atom(cpasync.CopyG2SOp(), source.element_type, num_bits_per_copy=16 * self.vector)
            column_threads = max(32 // self.vector, 32 * self.warps // rows)
            copy = cute.make_tiled_copy_tv(
                atom,
                cute.make_layout((32 * self.warps // column_threads, column_threads), stride=(column_threads, 1)),
                cute.make_layout((1, self.vector)),
            )
            thread = copy.get_slice(tid)
            tile = cute.local_tile(source[batch, head, None, None], (rows, width), (start // rows, 0))
            coords = cute.local_tile(
                cute.make_identity_tensor((source.shape[2], source.shape[3])), (rows, width), (start // rows, 0)
            )
            g, dest, c = (thread.partition_S(tile), thread.partition_D(shared), thread.partition_S(coords))
            for r in cutlass.range_constexpr(cute.size(dest.shape[1])):
                for d in cutlass.range_constexpr(cute.size(dest.shape[2])):
                    row, dim = c[0, r, d]
                    if row < source.shape[2] and dim < source.shape[3]:
                        cute.copy(copy, g[None, r, d], dest[None, r, d])
                    else:
                        dest[None, r, d].fill(0)
        else:
            for i in cutlass.range(cute.ceil_div(cute.size(shared), 32 * self.warps), unroll=4):
                index = i * 32 * self.warps + tid
                if index < cute.size(shared):
                    row = index // cute.size(shared, mode=[1])
                    d = index % cute.size(shared, mode=[1])
                    value = cutlass.Float32(0.0)
                    if start + row < source.shape[2] and d < source.shape[3]:
                        value = source[batch, head, start + row, d].to(cutlass.Float32)
                    shared[row, d] = value.to(source.element_type)

    @cute.jit
    def store_result(self, result, destination, shared, batch, head, start, offset, thread, tid, scale):
        copy_threads = 128 if self.grid_rows > self.rows else 32 * self.warps
        copy_tid = tid % copy_threads
        coordinates = thread.partition_C(cute.make_identity_tensor((self.rows, self.product_width)))
        group_offset = (
            (tid // 128) * self.product_width
            if cutlass.const_expr(self.hopper and self.mode == "kv" and self.warps == 8)
            else 0
        )
        for i in cutlass.range_constexpr(cute.size(coordinates)):
            row, dim = coordinates[i]
            shared[row, dim + group_offset] = (result[i] * scale).to(destination.element_type)
        if cutlass.const_expr(self.specialized_load):
            cute.arch.barrier(barrier_id=1 + tid // 128, number_of_threads=128)
        else:
            cute.arch.sync_threads()
        column_threads = max(32 // self.vector, copy_threads // self.rows)
        copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), destination.element_type, num_bits_per_copy=16 * self.vector),
            cute.make_layout((copy_threads // column_threads, column_threads), stride=(column_threads, 1)),
            cute.make_layout((1, self.vector)),
        )
        copier = copy.get_slice(copy_tid)
        output_offset = offset - group_offset
        tile = cute.local_tile(
            destination[batch, head, None, None], (self.rows, self.width), (start // self.rows, output_offset // self.width)
        )
        source = cute.local_tile(shared, (self.rows, self.width), (0, 0))
        identity = cute.local_tile(
            cute.make_identity_tensor((destination.shape[2], destination.shape[3])),
            (self.rows, self.width),
            (start // self.rows, output_offset // self.width),
        )
        inputs, outputs, coords = copier.partition_S(source), copier.partition_D(tile), copier.partition_D(identity)
        for r in cutlass.range_constexpr(cute.size(outputs.shape[1])):
            for d in cutlass.range_constexpr(cute.size(outputs.shape[2])):
                row, dim = coords[0, r, d]
                if row < destination.shape[2] and dim < destination.shape[3]:
                    cute.copy(copy, inputs[None, r, d], outputs[None, r, d])
        if cutlass.const_expr(self.specialized_load):
            cute.arch.barrier(barrier_id=1 + tid // 128, number_of_threads=128)
        else:
            cute.arch.sync_threads()

    @cute.jit
    def dot(self, a, b, mma, thread, tid, wait: cutlass.Constexpr = True, tcgen_state=None):
        if cutlass.const_expr(self.tcgen):
            result = blackwell_dot(a, b, thread, tid, self.columns, tcgen_state)
        elif cutlass.const_expr(self.hopper):
            result = self.dot_hopper(a, b, mma, thread, tid, wait)
        else:
            result = cute.make_rmem_tensor(thread.partition_shape_C((self.rows, self.columns)), cutlass.Float32)
            result.fill(0.0)
            ra = thread.make_fragment_A(thread.partition_A(cute.local_tile(a, (self.rows, 32), (0, 0))))
            rb = thread.make_fragment_B(thread.partition_B(cute.local_tile(b, (self.columns, 32), (0, 0))))
            ca = cute.make_tiled_copy_A(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), a.element_type), mma
            )
            cb = cute.make_tiled_copy_B(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), b.element_type), mma
            )
            ta, tb = ca.get_slice(tid), cb.get_slice(tid)
            sa, sb = ta.partition_S(a), tb.partition_S(b)
            da, db = ta.retile(ra), tb.retile(rb)
            cute.copy(ca, sa[None, None, 0], da[None, None, 0])
            cute.copy(cb, sb[None, None, 0], db[None, None, 0])
            for chunk in cutlass.range(cute.size(sa.shape[2]), unroll_full=True):
                slot = chunk % 2
                next_slot = (chunk + 1) % 2
                if chunk + 1 < cute.size(sa.shape[2]):
                    cute.copy(ca, sa[None, None, chunk + 1], da[None, None, next_slot])
                    cute.copy(cb, sb[None, None, chunk + 1], db[None, None, next_slot])
                cute.gemm(mma, result, ra[None, None, slot], rb[None, None, slot], result)
        return result

    @cute.jit
    def product(self, weights, source, offset, mma, thread, tid, result):
        if cutlass.const_expr(self.hopper):
            self.product_hopper(weights, source, offset, mma, thread, tid, result)
        else:
            if cutlass.const_expr(self.column_warps > 1):
                ra = thread.make_fragment_A(thread.partition_A(weights))
                ca = cute.make_tiled_copy_A(
                    cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), source.element_type), mma
                )
                ta = ca.get_slice(tid)
                sa, da = ta.partition_S(weights), ta.retile(ra)
            else:
                ra = operand_a(weights, source.element_type)
            source_tile = cute.local_tile(source, (self.columns, self.width), (0, offset // self.width))
            transposed = cute.composition(
                source_tile, cute.make_layout((self.width, self.columns), stride=(self.columns, 1))
            )
            rb = thread.make_fragment_B(thread.partition_B(transposed))
            cb = cute.make_tiled_copy_B(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
            )
            tb = cb.get_slice(tid)
            sb, db = tb.partition_S(transposed), tb.retile(rb)
            for chunk in cutlass.range_constexpr(self.columns // 16):
                if cutlass.const_expr(self.column_warps > 1):
                    cute.copy(ca, sa[None, None, chunk], da[None, None, chunk])
                cute.copy(cb, sb[None, None, chunk], db[None, None, chunk])
                cute.gemm(mma, result, ra[None, None, chunk], rb[None, None, chunk], result)

    @cute.jit
    def dot_hopper(self, a, b, mma, thread, tid, wait: cutlass.Constexpr = True):
        result = cute.make_rmem_tensor(thread.partition_shape_C((self.rows, self.columns)), cutlass.Float32)
        result.fill(0.0)
        ra = mma.make_fragment_A(thread.partition_A(a))
        rb = mma.make_fragment_B(thread.partition_B(b))
        cute.arch.fence_proxy("async.shared", space="cta")
        warpgroup.fence()
        for chunk in cutlass.range(cute.size(ra.shape[2]), unroll_full=True):
            cute.gemm(mma, result, ra[None, None, chunk], rb[None, None, chunk], result)
        warpgroup.commit_group()
        if cutlass.const_expr(wait):
            warpgroup.wait_group(0)
        return result

    @cute.jit
    def product_hopper(self, weights, source, offset, mma, thread, tid, result):
        cute.arch.fence_proxy("async.shared", space="cta")
        pmma = cute.make_tiled_mma(
            warpgroup.MmaF16BF16Op(
                source.element_type,
                cutlass.Float32,
                (64, self.product_width, 16),
                warpgroup.OperandSource.SMEM if self.mode == "kv" and self.warps == 8 else warpgroup.OperandSource.RMEM,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.MN,
            )
        )
        pt = pmma.get_slice(tid % 128)
        result = cute.make_tensor(result.iterator, cute.make_layout(pt.partition_shape_C((self.rows, self.product_width))))
        source_tile = cute.local_tile(source, (self.columns, self.product_width), (0, offset // self.product_width))
        transposed = cute.composition(
            source_tile, cute.make_layout((self.product_width, self.columns), stride=(self.columns, 1))
        )
        if cutlass.const_expr(self.mode == "kv" and self.warps == 8):
            ra = pmma.make_fragment_A(pt.partition_A(weights))
        else:
            ra = hopper_operand(weights, pmma.tv_layout_A.shape[1], source.element_type)
        rb = pmma.make_fragment_B(pt.partition_B(transposed))
        warpgroup.fence()
        pmma.set(warpgroup.Field.ACCUMULATE, True)
        for chunk in cutlass.range(cute.size(ra.shape[2]), unroll_full=True):
            cute.gemm(pmma, result, ra[None, None, chunk], rb[None, None, chunk], result)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

    @cute.jit
    def kv_weights(
        self,
        left,
        right,
        other_left,
        other_right,
        mask,
        statistics,
        location,
        full,
        magnitude,
        sign,
        mma,
        thread,
        tid,
        shared_statistics=None,
        tcgen_state=None,
    ):
        maximum, logsum, delta, delta_base = statistics
        batch, head, start, query, sq, sk = location
        if cutlass.const_expr(shared_statistics is not None):
            if tid < self.columns:
                for field in cutlass.range_constexpr(4):
                    source = (maximum, logsum, delta, delta_base)[field]
                    if query + tid < sq:
                        source_ptr = source.iterator + cute.crd2idx((batch, head, query + tid), source.layout)
                        target_ptr = shared_statistics.iterator + cute.crd2idx((field, tid), shared_statistics.layout)
                        copy = cute.make_copy_atom(
                            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS), cutlass.Float32, num_bits_per_copy=32
                        )
                        cute.copy(
                            copy,
                            cute.make_tensor(source_ptr, cute.make_layout(1)),
                            cute.make_tensor(target_ptr, cute.make_layout(1)),
                        )
                    else:
                        shared_statistics[field, tid] = cutlass.Float32(0.0)
            cute.arch.cp_async_commit_group()
        coords = thread.partition_C(cute.make_identity_tensor((self.rows, self.columns)))
        scores = self.dot(left, right, mma, thread, tid, wait=not self.pointwise_overlap, tcgen_state=tcgen_state)
        dp = self.dot(other_left, other_right, mma, thread, tid, wait=not self.pointwise_overlap, tcgen_state=tcgen_state)
        if cutlass.const_expr(self.pointwise_overlap):
            warpgroup.wait_group(1)
        if cutlass.const_expr(shared_statistics is not None):
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        probability = cute.make_rmem_tensor(scores.shape, cutlass.Float32)
        for i in cutlass.range_constexpr(cute.size(coords)):
            row, col = coords[i]
            qr = query + col
            p = cutlass.Float32(0.0)
            correction = cutlass.Float32(0.0)
            center = cutlass.Float32(0.0)
            if self.visible(mask, batch, head, qr, start + row, sq, sk, full):
                row_max = cutlass.Float32(0.0)
                row_logsum = cutlass.Float32(0.0)
                if cutlass.const_expr(shared_statistics is not None):
                    row_max = shared_statistics[0, col]
                    row_logsum = shared_statistics[1, col]
                    correction = shared_statistics[2, col]
                    center = shared_statistics[3, col]
                else:
                    row_max = maximum[batch, head, qr]
                    row_logsum = logsum[batch, head, qr]
                    correction = delta[batch, head, qr]
                    if cutlass.const_expr(self.centered):
                        center = delta_base[batch, head, qr]
                p = cute.math.exp2(
                    (scores[i] * sign - row_max) * magnitude * 1.4426950408889634 - row_logsum,
                    fastmath=True,
                )
            probability[i] = p
            if cutlass.const_expr(not self.pointwise_overlap):
                scores[i] = p * ((dp[i] - center) - correction)
        if cutlass.const_expr(self.pointwise_overlap):
            warpgroup.wait_group(0)
            for i in cutlass.range_constexpr(cute.size(coords)):
                row, col = coords[i]
                qr = query + col
                correction = cutlass.Float32(0.0)
                center = cutlass.Float32(0.0)
                if self.visible(mask, batch, head, qr, start + row, sq, sk, full):
                    if cutlass.const_expr(shared_statistics is not None):
                        correction = shared_statistics[2, col]
                        center = shared_statistics[3, col]
                    else:
                        correction = delta[batch, head, qr]
                        center = delta_base[batch, head, qr]
                scores[i] = probability[i] * ((dp[i] - center) - correction)
        return scores, probability

    @cute.jit
    def kv_shared_weights(
        self,
        shared_scores,
        shared_probabilities,
        left,
        right,
        other_left,
        other_right,
        mask,
        statistics,
        location,
        full,
        magnitude,
        sign,
        mma,
        thread,
        tid,
        shared_statistics=None,
        tcgen_state=None,
    ):
        scores, probabilities = self.kv_weights(
            left,
            right,
            other_left,
            other_right,
            mask,
            statistics,
            location,
            full,
            magnitude,
            sign,
            mma,
            thread,
            tid,
            shared_statistics=shared_statistics,
            tcgen_state=tcgen_state,
        )
        coords = thread.partition_C(cute.make_identity_tensor((self.rows, self.columns)))
        for i in cutlass.range_constexpr(cute.size(coords)):
            row, col = coords[i]
            shared_scores[row, col] = scores[i].to(left.element_type)
            shared_probabilities[row, col] = probabilities[i].to(left.element_type)

    @cute.jit
    def dual_product(self, weights, probabilities, source, offset, mma, thread, tid, result, mean):
        if cutlass.const_expr(self.hopper):
            cute.arch.fence_proxy("async.shared", space="cta")
            pmma = cute.make_tiled_mma(
                warpgroup.MmaF16BF16Op(
                    source.element_type,
                    cutlass.Float32,
                    (64, self.width, 16),
                    warpgroup.OperandSource.RMEM,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.MN,
                )
            )
            pt = pmma.get_slice(tid)
            layout = cute.make_layout(pt.partition_shape_C((self.rows, self.width)))
            result = cute.make_tensor(result.iterator, layout)
            mean = cute.make_tensor(mean.iterator, layout)
            tile = cute.local_tile(source, (self.columns, self.width), (0, offset // self.width))
            transposed = cute.composition(tile, cute.make_layout((self.width, self.columns), stride=(self.columns, 1)))
            ra = hopper_operand(weights, pmma.tv_layout_A.shape[1], source.element_type)
            rp = hopper_operand(probabilities, pmma.tv_layout_A.shape[1], source.element_type)
            rb = pmma.make_fragment_B(pt.partition_B(transposed))
            warpgroup.fence()
            pmma.set(warpgroup.Field.ACCUMULATE, True)
            for chunk in cutlass.range(cute.size(ra.shape[2]), unroll_full=True):
                cute.gemm(pmma, result, ra[None, None, chunk], rb[None, None, chunk], result)
                cute.gemm(pmma, mean, rp[None, None, chunk], rb[None, None, chunk], mean)
            warpgroup.commit_group()
            warpgroup.wait_group(0)
        else:
            if cutlass.const_expr(self.column_warps > 1):
                ra = thread.make_fragment_A(thread.partition_A(weights))
                rp = thread.make_fragment_A(thread.partition_A(probabilities))
                ca = cute.make_tiled_copy_A(
                    cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), source.element_type), mma
                )
                ta = ca.get_slice(tid)
                sa, da = ta.partition_S(weights), ta.retile(ra)
                sp, dp = ta.partition_S(probabilities), ta.retile(rp)
            else:
                ra = operand_a(weights, source.element_type)
                rp = operand_a(probabilities, source.element_type)
            tile = cute.local_tile(source, (self.columns, self.width), (0, offset // self.width))
            transposed = cute.composition(tile, cute.make_layout((self.width, self.columns), stride=(self.columns, 1)))
            rb = thread.make_fragment_B(thread.partition_B(transposed))
            cb = cute.make_tiled_copy_B(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
            )
            tb = cb.get_slice(tid)
            sb, db = tb.partition_S(transposed), tb.retile(rb)
            for chunk in cutlass.range_constexpr(self.columns // 16):
                if cutlass.const_expr(self.column_warps > 1):
                    cute.copy(ca, sa[None, None, chunk], da[None, None, chunk])
                    cute.copy(ca, sp[None, None, chunk], dp[None, None, chunk])
                cute.copy(cb, sb[None, None, chunk], db[None, None, chunk])
                cute.gemm(mma, result, ra[None, None, chunk], rb[None, None, chunk], result)
                cute.gemm(mma, mean, rp[None, None, chunk], rb[None, None, chunk], mean)

    @cute.jit
    def split_query_step(
        self,
        operands,
        weights,
        statistics,
        location,
        mask,
        full,
        magnitude,
        sign,
        offset,
        score_mma,
        score_thread,
        mma,
        thread,
        tid,
        totals,
        rounded,
        accum,
        mean,
    ):
        left, right, other_left, other_right = operands
        query_derivatives, query_probabilities = weights
        row_max, row_logsum, base, delta_base = statistics
        batch, head, start, key, sq, sk = location
        coords = score_thread.partition_C(cute.make_identity_tensor((self.rows, self.columns)))
        rc = row_view(coords)
        rounded_scores = cute.make_rmem_tensor(score_thread.partition_shape_C((self.rows, self.columns)), left.element_type)
        rounded_probabilities = cute.make_fragment_like(rounded_scores)
        if tid < 32 * self.score_warps:
            scores = self.dot(left, right, score_mma, score_thread, tid)
            dp = self.dot(other_left, other_right, score_mma, score_thread, tid)
            sv, pv = row_view(scores), row_view(dp)
            for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                center = base[r]
                if cutlass.const_expr(self.reload_center):
                    center = cutlass.Float32(0.0)
                    qr = start + rc[r, 0][0]
                    if qr < sq:
                        center = delta_base[batch, head, qr]
                for c in cutlass.range_constexpr(cute.size(rc.shape[1])):
                    row, col = rc[r, c]
                    probability = cutlass.Float32(0.0)
                    if self.visible(mask, batch, head, start + row, key + col, sq, sk, full):
                        probability = cute.math.exp2(
                            (sv[r, c] * sign - row_max[r]) * magnitude * 1.4426950408889634 - row_logsum[r], fastmath=True
                        )
                    sv[r, c] = probability * (pv[r, c] - center)
                    pv[r, c] = probability
                totals[r] += quad_sum(sv[r, None].load().reduce(cute.ReductionOp.ADD, 0.0, 0))
            packed_half_conversion(scores, rounded_scores)
            packed_half_conversion(dp, rounded_probabilities)
            rounded_view = row_view(rounded_scores)
            for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                rounded[r] += quad_sum(rounded_view[r, None].load().to(cutlass.Float32).reduce(cute.ReductionOp.ADD, 0.0, 0))
        # Both producers finish V reads before any warp can repurpose it.
        cute.arch.sync_threads()
        if tid < 32 * self.score_warps:
            for i in cutlass.range_constexpr(cute.size(coords)):
                row, col = coords[i]
                query_derivatives[row, col] = rounded_scores[i]
                query_probabilities[row, col] = rounded_probabilities[i]
        cute.arch.sync_threads()
        self.dual_product(query_derivatives, query_probabilities, right, offset, mma, thread, tid, accum, mean)

    @cute.jit
    def split_keyvalue_step(
        self,
        operands,
        weights,
        statistics,
        location,
        mask,
        full,
        magnitude,
        sign,
        offset,
        score_mma,
        score_thread,
        mma,
        thread,
        tid,
        accum,
        accv,
    ):
        left, right, other_left, other_right = operands
        shared_scores, shared_probabilities = weights
        coords = score_thread.partition_C(cute.make_identity_tensor((self.rows, self.columns)))
        rounded_scores = cute.make_rmem_tensor(score_thread.partition_shape_C((self.rows, self.columns)), left.element_type)
        rounded_probabilities = cute.make_fragment_like(rounded_scores)
        if tid < 32 * self.score_warps:
            scores, probability = self.kv_weights(
                left,
                right,
                other_left,
                other_right,
                mask,
                statistics,
                location,
                full,
                magnitude,
                sign,
                score_mma,
                score_thread,
                tid,
            )
            packed_half_conversion(scores, rounded_scores)
            packed_half_conversion(probability, rounded_probabilities)
        # All score-warps finish their stationary-V reads before any warp can
        # overwrite that buffer with the two rounded product operands.
        cute.arch.sync_threads()
        if tid < 32 * self.score_warps:
            for i in cutlass.range_constexpr(cute.size(coords)):
                row, col = coords[i]
                shared_scores[row, col] = rounded_scores[i]
                shared_probabilities[row, col] = rounded_probabilities[i]
        cute.arch.sync_threads()
        self.product(shared_scores, right, offset, mma, thread, tid, accum)
        self.product(shared_probabilities, other_right, offset, mma, thread, tid, accv)

    @cute.jit
    def combine_query_rows(self, totals, rounded, coords, staging, tid):
        lane = tid % 32
        column_warp = tid // (32 * (self.warps // self.column_warps))
        if lane % 4 == 0:
            for r in cutlass.range_constexpr(cute.size(coords.shape[0])):
                row = coords[r, 0][0]
                staging[0, column_warp, row] = totals[r]
                staging[1, column_warp, row] = rounded[r]
        cute.arch.sync_threads()
        for r in cutlass.range_constexpr(cute.size(coords.shape[0])):
            row = coords[r, 0][0]
            total, sigma = cutlass.Float32(0), cutlass.Float32(0)
            for column in cutlass.range_constexpr(self.column_warps):
                total += staging[0, column, row]
                sigma += staging[1, column, row]
            totals[r], rounded[r] = total, sigma

    @cute.jit
    def shared(self, allocator, dtype: cutlass.Constexpr, layout):
        if cutlass.const_expr(self.hopper or self.tma or self.tcgen or self.sm103_rows):
            tensor = allocator.allocate_tensor(
                dtype, layout.outer, byte_alignment=1024 if self.tma else 128, swizzle=layout.inner
            )
        else:
            tensor = allocator.allocate_tensor(dtype, layout, byte_alignment=128)
        return tensor

    @cute.jit
    def online_softmax(self, scores, coordinates, accum, maxima, sums, mask, location, full, magnitude, sign):
        batch, head, start, key, sq, sk = location
        rc, av = row_view(coordinates), row_view(accum)
        scores.store(scores.load() * sign)
        sv = row_view(scores)
        for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
            for c in cutlass.range_constexpr(cute.size(rc.shape[1])):
                row, col = rc[r, c]
                if not self.visible(mask, batch, head, start + row, key + col, sq, sk, full):
                    sv[r, c] = -cutlass.Float32.inf
            current = quad_max(sv[r, None].load().reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0))
            new = cute.arch.fmax(maxima[r], current)
            alpha = cutlass.Float32(0.0)
            if maxima[r] != -cutlass.Float32.inf:
                alpha = cute.math.exp2((maxima[r] - new) * magnitude * 1.4426950408889634, fastmath=True)
            for c in cutlass.range_constexpr(cute.size(rc.shape[1])):
                probability = cutlass.Float32(0.0)
                if sv[r, c] != -cutlass.Float32.inf:
                    probability = cute.math.exp2((sv[r, c] - new) * magnitude * 1.4426950408889634, fastmath=True)
                sv[r, c] = probability
            sums[r] = sums[r] * alpha + quad_sum(sv[r, None].load().reduce(cute.ReductionOp.ADD, 0.0, 0))
            maxima[r] = new
            av[r, None] = av[r, None].load() * alpha

    @cute.jit
    def column_softmax_max(self, scores, coordinates, staging, mask, location, full, sign, tid):
        # Store distinct per-N-warp maxima. The caller's first CTA barrier
        # publishes them and makes the consumed K buffer safe for P stores.
        batch, head, start, key, sq, sk = location
        rc, sv = row_view(coordinates), row_view(scores)
        scores.store(scores.load() * sign)
        column = tid // (32 * (self.warps // self.column_warps))
        for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
            for c in cutlass.range_constexpr(cute.size(rc.shape[1])):
                row, col = rc[r, c]
                if not self.visible(mask, batch, head, start + row, key + col, sq, sk, full):
                    sv[r, c] = -cutlass.Float32.inf
            current = quad_max(sv[r, None].load().reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0))
            if tid % 4 == 0:
                staging[0, column, rc[r, 0][0]] = current

    @cute.jit
    def column_softmax_probabilities(self, scores, coordinates, accum, maxima, sums, staging, weights, magnitude, tid):
        # Maxima and partial sums occupy different scratch planes: a fast warp
        # cannot overwrite a max before another N-warp has finished reading it.
        rc, sv, av = row_view(coordinates), row_view(scores), row_view(accum)
        column = tid // (32 * (self.warps // self.column_warps))
        for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
            row = rc[r, 0][0]
            current = -cutlass.Float32.inf
            for nwarp in cutlass.range_constexpr(self.column_warps):
                current = cute.arch.fmax(current, staging[0, nwarp, row])
            new = cute.arch.fmax(maxima[r], current)
            alpha = cutlass.Float32(0.0)
            if maxima[r] != -cutlass.Float32.inf:
                alpha = cute.math.exp2((maxima[r] - new) * magnitude * 1.4426950408889634, fastmath=True)
            for c in cutlass.range_constexpr(cute.size(rc.shape[1])):
                probability = cutlass.Float32(0.0)
                if sv[r, c] != -cutlass.Float32.inf:
                    probability = cute.math.exp2((sv[r, c] - new) * magnitude * 1.4426950408889634, fastmath=True)
                sv[r, c] = probability
                wr, wc = rc[r, c]
                weights[wr, wc] = probability.to(weights.element_type)
            partial = quad_sum(sv[r, None].load().reduce(cute.ReductionOp.ADD, 0.0, 0))
            if tid % 4 == 0:
                staging[1, column, row] = partial
            sums[r] *= alpha
            maxima[r] = new
            av[r, None] = av[r, None].load() * alpha

    @cute.jit
    def column_softmax_sum(self, coordinates, sums, staging):
        rc = row_view(coordinates)
        for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
            row = rc[r, 0][0]
            partial = cutlass.Float32(0.0)
            for nwarp in cutlass.range_constexpr(self.column_warps):
                partial += staging[1, nwarp, row]
            sums[r] += partial

    @cute.jit
    def visible(self, mask, batch, head, row, col, sq, sk, full):
        valid = row < sq and col < sk
        if cutlass.const_expr(self.causal):
            if not full:
                valid = valid and col <= row
        if cutlass.const_expr(self.masked):
            if valid and not full:
                valid = mask[batch, head, row, col]
        return valid

    @cute.jit
    def block_visibility(self, blocks, batch, head, query, key, rows: cutlass.Constexpr, columns: cutlass.Constexpr):
        state = cutlass.Int32(2)
        if cutlass.const_expr(self.masked):
            state = blocks[batch, head, query // 64, key // 64]
        active, full = state != 0, state == 2
        if cutlass.const_expr(self.causal):
            active = active and key < query + rows
            full = full and key + columns <= query + 1
        return active, full

    @cute.kernel
    def kernel(self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, delta_base, scale, mma, tma_args):
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        tile, bh, width = cute.arch.block_idx()
        thread = mma.get_slice(tid % 128 if cutlass.const_expr(self.hopper) else tid)
        batch = bh // q.shape[1]
        head = bh % q.shape[1]
        kvhead = head // (q.shape[1] // k.shape[1])
        start = tile * self.grid_rows
        group_start = start
        if cutlass.const_expr(self.grid_rows > self.rows):
            start += (tid // 128) * self.rows
        offset = width * self.width
        if cutlass.const_expr(self.hopper and self.mode == "kv" and self.warps == 8):
            offset += (tid // 128) * self.product_width
        magnitude = scale if scale >= 0.0 else -scale
        sign = cutlass.Float32(1.0) if scale >= 0.0 else cutlass.Float32(-1.0)
        score_mma, score_thread = mma, thread
        if cutlass.const_expr(self.split_query or self.split_kv):
            score_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
                (self.warps // self.column_warps, self.score_column_warps, 1),
                permutation_mnk=(self.rows, 16 * self.score_column_warps, 16),
            )
            score_thread = score_mma.get_slice(tid % (32 * self.score_warps))
        coords = score_thread.partition_C(cute.make_identity_tensor((self.rows, self.columns)))
        rc = row_view(coords)
        accum = cute.make_rmem_tensor(thread.partition_shape_C((self.rows, self.product_width)), cutlass.Float32)
        accum.fill(0.0)
        av = row_view(accum)
        padded = cute.ceil_div(q.shape[3], self.width) * self.width
        if cutlass.const_expr(self.hopper or self.tma or self.tcgen):
            atom = warpgroup.make_smem_layout_atom(
                warpgroup.SmemLayoutAtomKind.K_SW128 if padded % 64 == 0 else warpgroup.SmemLayoutAtomKind.K_SW64,
                q.element_type,
            )
        else:
            atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        left_layout = cute.tile_to_shape(atom, (self.grid_rows, padded), (0, 1))
        right_layout = cute.tile_to_shape(atom, (self.columns, padded), (0, 1))
        allocator = SmemAllocator()
        tcgen_state = None
        if cutlass.const_expr(self.tcgen):
            mma_barrier = allocator.allocate_array(cutlass.Int64, 1)
            phase_buffer = allocator.allocate_array(cutlass.Int32, 1)
            tmem_buffer = allocator.allocate_array(cutlass.Int32, 1)
            scratch = allocator.allocate_tensor(
                cutlass.Float32, cute.make_layout((64, self.columns), stride=(self.columns, 1)), byte_alignment=128
            )
            tmem = TmemAllocator(
                alloc_result_dst_smem_ptr=tmem_buffer,
                barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=128),
            )
            tmem.allocate(self.columns)
            tmem.wait_for_alloc()
            tmem_pointer = tmem.retrieve_ptr(cutlass.Float32)
            tmem.relinquish_alloc_permit()
            if tid == 0:
                phase_buffer[0] = 0
                cute.arch.mbarrier_init(mma_barrier, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            tcgen_state = (tmem_pointer, mma_barrier, scratch, phase_buffer)
        shared_statistics = None
        if cutlass.const_expr(
            self.mode == "kv" and (self.stage_statistics or self.hopper and self.warps == 4 and self.width == 128)
        ):
            shared_statistics = allocator.allocate_tensor(
                cutlass.Float32, cute.make_layout((4, self.columns), stride=(self.columns, 1)), byte_alignment=128
            )
        barrier = None
        phase_tma = cutlass.Int32(0)
        if cutlass.const_expr(self.tma):
            barrier = allocator.allocate_array(cutlass.Int64, 2 if self.pipelined else 1)
            if tid == 0:
                for stage in cutlass.range_constexpr(2 if self.pipelined else 1):
                    cute.arch.mbarrier_init(barrier + stage, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
        left = self.shared(allocator, q.element_type, left_layout)
        if cutlass.const_expr(self.pipelined):
            staged_layout = cute.tile_to_shape(atom, (self.columns, padded, 2), (0, 1, 2))
            right_stages = self.shared(allocator, q.element_type, staged_layout)
            other_stages = self.shared(allocator, q.element_type, staged_layout)
            right = right_stages[None, None, 0]
            other_right = other_stages[None, None, 0]
        else:
            right = self.shared(allocator, q.element_type, right_layout)
            other_right = self.shared(allocator, q.element_type, right_layout)
        if cutlass.const_expr(self.mode == "kv"):
            batch = bh // k.shape[1]
            kvhead = bh % k.shape[1]
            self.load(k, left, batch, kvhead, start, tid)
        else:
            self.load(q, left, batch, head, group_start, tid)
        if cutlass.const_expr(self.mode != "forward"):
            other_left = self.shared(allocator, q.element_type, left_layout)
            if cutlass.const_expr(self.mode == "kv"):
                self.load(v, other_left, batch, kvhead, start, tid)
            else:
                self.load(dout, other_left, batch, head, start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if cutlass.const_expr(self.grid_rows > self.rows):
            left = cute.local_tile(left, (self.rows, padded), (tid // 128, 0))
        if cutlass.const_expr(self.mode == "forward"):
            forward_statistics = None
            if cutlass.const_expr(self.forward_columns):
                forward_statistics = allocator.allocate_tensor(
                    cutlass.Float32,
                    cute.make_layout(
                        (2, self.column_warps, self.rows), stride=(self.column_warps * self.rows, self.rows, 1)
                    ),
                    byte_alignment=128,
                )
            maxima = cute.make_rmem_tensor((cute.size(rc.shape[0]),), cutlass.Float32)
            sums = cute.make_rmem_tensor(maxima.shape, cutlass.Float32)
            maxima.fill(-cutlass.Float32.inf)
            sums.fill(0.0)
            limit = cutlass.min(k.shape[2], start + self.rows) if cutlass.const_expr(self.causal) else k.shape[2]
            count = cute.ceil_div(limit, self.columns)
            if cutlass.const_expr(self.pipelined):
                self.issue_pair(right_stages, other_stages, batch, kvhead, 0, 0, start, tid, tma_args, barrier, blocks)
            for block in range(count):
                active, full = self.block_visibility(
                    blocks, batch, head, start, block * self.columns, self.rows, self.columns
                )
                if cutlass.const_expr(self.pipelined):
                    right, other_right = self.pipeline_step(
                        right_stages,
                        other_stages,
                        batch,
                        kvhead,
                        block,
                        block,
                        count,
                        start,
                        tid,
                        tma_args,
                        barrier,
                        blocks,
                    )
                if active:
                    if cutlass.const_expr(not self.pipelined):
                        phase_tma = self.load_pair(
                            k,
                            v,
                            right,
                            other_right,
                            batch,
                            kvhead,
                            block * self.columns,
                            tid,
                            tma_args,
                            barrier,
                            phase_tma,
                        )
                    scores = self.dot(left, right, mma, thread, tid, tcgen_state=tcgen_state)
                    if cutlass.const_expr(self.forward_columns):
                        self.column_softmax_max(
                            scores,
                            coords,
                            forward_statistics,
                            mask,
                            (batch, head, start, block * self.columns, q.shape[2], k.shape[2]),
                            full,
                            sign,
                            tid,
                        )
                        cute.arch.sync_threads()
                        weights_layout = cute.tile_to_shape(atom, (self.rows, self.columns), (0, 1))
                        weights = cute.make_tensor(right.iterator, weights_layout)
                        self.column_softmax_probabilities(
                            scores,
                            coords,
                            accum,
                            maxima,
                            sums,
                            forward_statistics,
                            weights,
                            magnitude,
                            tid,
                        )
                        cute.arch.sync_threads()
                        self.column_softmax_sum(coords, sums, forward_statistics)
                        self.product(weights, other_right, offset, mma, thread, tid, accum)
                    else:
                        self.online_softmax(
                            scores,
                            coords,
                            accum,
                            maxima,
                            sums,
                            mask,
                            (batch, head, start, block * self.columns, q.shape[2], k.shape[2]),
                            full,
                            magnitude,
                            sign,
                        )
                        self.product(scores, other_right, offset, mma, thread, tid, accum)
                if cutlass.const_expr(self.pipelined) or active:
                    cute.arch.sync_threads()
            for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                denom = sums[r] if sums[r] > 0.0 else cutlass.Float32(1.0)
                av[r, None] = av[r, None].load() / denom
                row = start + rc[r, 0][0]
                publish = True
                if cutlass.const_expr(self.forward_columns):
                    publish = tid < 32 * (self.warps // self.column_warps)
                if publish and lane % 4 == 0 and row < q.shape[2] and width == 0:
                    maximum[batch, head, row] = maxima[r]
                    logsum[batch, head, row] = cute.math.log2(denom, fastmath=True)
            self.store_result(accum, out, left, batch, head, start, offset, thread, tid, cutlass.Float32(1.0))
        elif cutlass.const_expr(self.mode == "q"):
            totals = cute.make_rmem_tensor((cute.size(rc.shape[0]),), cutlass.Float32)
            row_max = cute.make_rmem_tensor(totals.shape, cutlass.Float32)
            row_logsum = cute.make_rmem_tensor(totals.shape, cutlass.Float32)
            if cutlass.const_expr(self.centered):
                rounded = cute.make_rmem_tensor(totals.shape, cutlass.Float32)
                base = cute.make_rmem_tensor(totals.shape, cutlass.Float32)
                mean = cute.make_rmem_tensor(accum.shape, cutlass.Float32)
                mean.fill(0.0)
                rounded.fill(0.0)
                base.fill(0.0)
            if cutlass.const_expr(self.column_warps > 1):
                weight_layout = cute.tile_to_shape(atom, (self.rows, self.columns), (0, 1))
                if cutlass.const_expr(self.column_warps > 2 or self.warps == 2):
                    # V is dead after dP; reuse its storage for the two MMA operands.
                    query_derivatives = cute.make_tensor(other_right.iterator, weight_layout)
                    query_probabilities = cute.make_tensor(other_right.iterator + self.rows * self.columns, weight_layout)
                else:
                    query_derivatives = self.shared(allocator, q.element_type, weight_layout)
                    query_probabilities = self.shared(allocator, q.element_type, weight_layout)
                query_sums = allocator.allocate_tensor(
                    cutlass.Float32,
                    cute.make_layout(
                        (2, self.column_warps, self.rows), stride=(self.column_warps * self.rows, self.rows, 1)
                    ),
                    byte_alignment=128,
                )
            if cutlass.const_expr(self.centered and self.hopper and q.shape[3] in (64, 128)):
                for part in cutlass.range_constexpr(self.rows // self.columns):
                    self.load(out, other_right, batch, head, start + part * self.columns, tid)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                    diagonal = self.dot(other_left, other_right, mma, thread, tid, tcgen_state=tcgen_state)
                    diagonal_view = row_view(diagonal)
                    for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                        value = cutlass.Float32(0.0)
                        for c in cutlass.range_constexpr(cute.size(rc.shape[1])):
                            if rc[r, c][0] == part * self.columns + rc[r, c][1]:
                                value = diagonal_view[r, c]
                        base[r] += quad_sum(value)
                    cute.arch.sync_threads()
                for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                    qr = start + rc[r, 0][0]
                    if lane % 4 == 0 and qr < q.shape[2] and width == 0:
                        delta_base[batch, head, qr] = base[r]
            totals.fill(0.0)
            for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                qr = start + rc[r, 0][0]
                row_max[r] = cutlass.Float32(0.0)
                row_logsum[r] = cutlass.Float32(0.0)
                if qr < q.shape[2]:
                    if cutlass.const_expr(
                        self.centered and not self.reload_center and not (self.hopper and q.shape[3] in (64, 128))
                    ):
                        base[r] = delta_base[batch, head, qr]
                    row_max[r] = maximum[batch, head, qr]
                    row_logsum[r] = logsum[batch, head, qr]
            limit = cutlass.min(k.shape[2], start + self.rows) if cutlass.const_expr(self.causal) else k.shape[2]
            count = cute.ceil_div(limit, self.columns)
            for phase in cutlass.range_constexpr(1 if self.centered else 2):
                if cutlass.const_expr(self.pipelined):
                    self.issue_pair(
                        right_stages,
                        other_stages,
                        batch,
                        kvhead,
                        0,
                        phase * count,
                        start,
                        tid,
                        tma_args,
                        barrier,
                        blocks,
                    )
                for block in range(count):
                    active, full = self.block_visibility(
                        blocks, batch, head, start, block * self.columns, self.rows, self.columns
                    )
                    if cutlass.const_expr(self.pipelined):
                        right, other_right = self.pipeline_step(
                            right_stages,
                            other_stages,
                            batch,
                            kvhead,
                            block,
                            phase * count + block,
                            count,
                            start,
                            tid,
                            tma_args,
                            barrier,
                            blocks,
                        )
                    if active:
                        if cutlass.const_expr(not self.pipelined):
                            phase_tma = self.load_pair(
                                k,
                                v,
                                right,
                                other_right,
                                batch,
                                kvhead,
                                block * self.columns,
                                tid,
                                tma_args,
                                barrier,
                                phase_tma,
                            )
                        if cutlass.const_expr(self.split_query):
                            self.split_query_step(
                                (left, right, other_left, other_right),
                                (query_derivatives, query_probabilities),
                                (row_max, row_logsum, base, delta_base),
                                (batch, head, start, block * self.columns, q.shape[2], k.shape[2]),
                                mask,
                                full,
                                magnitude,
                                sign,
                                offset,
                                score_mma,
                                score_thread,
                                mma,
                                thread,
                                tid,
                                totals,
                                rounded,
                                accum,
                                mean,
                            )
                        else:
                            scores = self.dot(
                                left, right, mma, thread, tid, wait=not self.pointwise_overlap, tcgen_state=tcgen_state
                            )
                            dp = self.dot(
                                other_left,
                                other_right,
                                mma,
                                thread,
                                tid,
                                wait=not self.pointwise_overlap,
                                tcgen_state=tcgen_state,
                            )
                            if cutlass.const_expr(self.pointwise_overlap):
                                warpgroup.wait_group(1)
                            sv, pv = row_view(scores), row_view(dp)
                            for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                                if cutlass.const_expr(self.centered):
                                    center = base[r]
                                    if cutlass.const_expr(self.reload_center):
                                        # Keep the external center out of the long-lived
                                        # warp-MMA register fragments on Blackwell.
                                        center = cutlass.Float32(0.0)
                                        qr = start + rc[r, 0][0]
                                        if qr < q.shape[2]:
                                            center = delta_base[batch, head, qr]
                                for c in cutlass.range_constexpr(cute.size(rc.shape[1])):
                                    row, col = rc[r, c]
                                    probability = cutlass.Float32(0.0)
                                    if self.visible(
                                        mask,
                                        batch,
                                        head,
                                        start + row,
                                        block * self.columns + col,
                                        q.shape[2],
                                        k.shape[2],
                                        full,
                                    ):
                                        probability = cute.math.exp2(
                                            (sv[r, c] * sign - row_max[r]) * magnitude * 1.4426950408889634 - row_logsum[r],
                                            fastmath=True,
                                        )
                                    if cutlass.const_expr(self.pointwise_overlap):
                                        sv[r, c] = probability
                                    elif cutlass.const_expr(self.centered):
                                        sv[r, c] = probability * (pv[r, c] - center)
                                        pv[r, c] = probability
                                    elif cutlass.const_expr(phase == 0):
                                        sv[r, c] = probability * pv[r, c]
                                    else:
                                        sv[r, c] = probability * (pv[r, c] - totals[r])
                            if cutlass.const_expr(self.pointwise_overlap):
                                warpgroup.wait_group(0)
                                for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                                    for c in cutlass.range_constexpr(cute.size(rc.shape[1])):
                                        probability = sv[r, c]
                                        sv[r, c] = probability * (pv[r, c] - base[r])
                                        pv[r, c] = probability
                            if cutlass.const_expr(phase == 0):
                                for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                                    totals[r] += quad_sum(sv[r, None].load().reduce(cute.ReductionOp.ADD, 0.0, 0))
                            else:
                                self.product(scores, right, offset, mma, thread, tid, accum)
                            if cutlass.const_expr(self.centered):
                                rounded_scores = cute.make_fragment_like(scores, q.element_type)
                                packed_half_conversion(scores, rounded_scores)
                                rounded_view = row_view(rounded_scores)
                                for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                                    rounded[r] += quad_sum(
                                        rounded_view[r, None].load().to(cutlass.Float32).reduce(cute.ReductionOp.ADD, 0.0, 0)
                                    )
                                if cutlass.const_expr(self.column_warps > 1):
                                    if cutlass.const_expr(self.column_warps > 2 or self.warps == 2):
                                        # All warps must finish reading V before its storage is overwritten.
                                        cute.arch.sync_threads()
                                    for i in cutlass.range_constexpr(cute.size(coords)):
                                        row, col = coords[i]
                                        query_derivatives[row, col] = rounded_scores[i]
                                        query_probabilities[row, col] = dp[i].to(q.element_type)
                                    cute.arch.sync_threads()
                                    self.dual_product(
                                        query_derivatives, query_probabilities, right, offset, mma, thread, tid, accum, mean
                                    )
                                else:
                                    self.dual_product(rounded_scores, dp, right, offset, mma, thread, tid, accum, mean)
                    if cutlass.const_expr(self.pipelined) or active:
                        cute.arch.sync_threads()
                if cutlass.const_expr(phase == 0):
                    if cutlass.const_expr(self.column_warps > 1):
                        self.combine_query_rows(totals, rounded, rc, query_sums, tid)
                    for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                        row = start + rc[r, 0][0]
                        publish = tid < 32 * (self.warps // self.column_warps)
                        if lane % 4 == 0 and row < q.shape[2] and width == 0 and publish:
                            delta[batch, head, row] = totals[r]
            if cutlass.const_expr(self.centered):
                mv = row_view(mean)
                for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                    av[r, None] = av[r, None].load() - rounded[r] * mv[r, None].load()
            self.store_result(accum, dq, left, batch, head, start, offset, thread, tid, scale)
        else:
            batch = bh // k.shape[1]
            kvhead = bh % k.shape[1]
            if cutlass.const_expr(self.column_warps > 1 or self.warps == 8):
                weight_atom = atom
                if cutlass.const_expr(self.hopper or self.tma or self.tcgen):
                    weight_atom = warpgroup.make_smem_layout_atom(
                        (
                            warpgroup.SmemLayoutAtomKind.K_SW128
                            if self.columns % 64 == 0
                            else warpgroup.SmemLayoutAtomKind.K_SW64
                        ),
                        q.element_type,
                    )
                weight_layout = cute.tile_to_shape(weight_atom, (self.rows, self.columns), (0, 1))
                if cutlass.const_expr(self.split_kv):
                    # Generic (non-TMA) V iterator has no carried TMA swizzle.
                    # Two M16N32 half operands occupy 2KiB of its 16KiB tile.
                    shared_scores = cute.make_tensor(other_left.iterator, weight_layout)
                    shared_probabilities = cute.make_tensor(other_left.iterator + self.rows * self.columns, weight_layout)
                else:
                    shared_scores = self.shared(allocator, q.element_type, weight_layout)
                    shared_probabilities = self.shared(allocator, q.element_type, weight_layout)
            accv = cute.make_rmem_tensor(accum.shape, cutlass.Float32)
            accv.fill(0.0)
            group = q.shape[1] // k.shape[1]
            for h in range(group):
                head = kvhead * group + h
                first = start // self.columns if cutlass.const_expr(self.causal) else 0
                stop = cute.ceil_div(q.shape[2], self.columns)
                count = stop - first
                if cutlass.const_expr(self.pipelined):
                    if first < stop:
                        self.issue_pair(
                            right_stages,
                            other_stages,
                            batch,
                            head,
                            first,
                            h * count,
                            start,
                            tid,
                            tma_args,
                            barrier,
                            blocks,
                        )
                for block in range(first, stop):
                    active, full = self.block_visibility(
                        blocks, batch, head, block * self.columns, start, self.columns, self.rows
                    )
                    if cutlass.const_expr(self.pipelined):
                        right, other_right = self.pipeline_step(
                            right_stages,
                            other_stages,
                            batch,
                            head,
                            block,
                            h * count + block - first,
                            stop,
                            start,
                            tid,
                            tma_args,
                            barrier,
                            blocks,
                        )
                    if active:
                        if cutlass.const_expr(not self.pipelined):
                            if cutlass.const_expr(self.split_kv):
                                # V was reused as DS/P after the previous dP.
                                # load_pair's existing commit/wait/barrier also
                                # covers this extra asynchronous V reload.
                                self.load(v, other_left, batch, kvhead, start, tid)
                            phase_tma = self.load_pair(
                                q,
                                dout,
                                right,
                                other_right,
                                batch,
                                head,
                                block * self.columns,
                                tid,
                                tma_args,
                                barrier,
                                phase_tma,
                            )
                        if cutlass.const_expr(self.split_kv):
                            self.split_keyvalue_step(
                                (left, right, other_left, other_right),
                                (shared_scores, shared_probabilities),
                                (maximum, logsum, delta, delta_base),
                                (batch, head, start, block * self.columns, q.shape[2], k.shape[2]),
                                mask,
                                full,
                                magnitude,
                                sign,
                                offset,
                                score_mma,
                                score_thread,
                                mma,
                                thread,
                                tid,
                                accum,
                                accv,
                            )
                        else:
                            if cutlass.const_expr(self.column_warps > 1 or self.warps == 8):
                                if cutlass.const_expr(self.mode == "kv" and self.warps == 8):
                                    if tid < 128:
                                        self.kv_shared_weights(
                                            shared_scores,
                                            shared_probabilities,
                                            left,
                                            right,
                                            other_left,
                                            other_right,
                                            mask,
                                            (maximum, logsum, delta, delta_base),
                                            (batch, head, start, block * self.columns, q.shape[2], k.shape[2]),
                                            full,
                                            magnitude,
                                            sign,
                                            mma,
                                            thread,
                                            tid,
                                            shared_statistics=shared_statistics,
                                            tcgen_state=tcgen_state,
                                        )
                                else:
                                    self.kv_shared_weights(
                                        shared_scores,
                                        shared_probabilities,
                                        left,
                                        right,
                                        other_left,
                                        other_right,
                                        mask,
                                        (maximum, logsum, delta, delta_base),
                                        (batch, head, start, block * self.columns, q.shape[2], k.shape[2]),
                                        full,
                                        magnitude,
                                        sign,
                                        mma,
                                        thread,
                                        tid,
                                        shared_statistics=shared_statistics,
                                        tcgen_state=tcgen_state,
                                    )
                                cute.arch.sync_threads()
                                self.product(shared_scores, right, offset, mma, thread, tid, accum)
                                self.product(shared_probabilities, other_right, offset, mma, thread, tid, accv)
                            else:
                                scores, probability = self.kv_weights(
                                    left,
                                    right,
                                    other_left,
                                    other_right,
                                    mask,
                                    (maximum, logsum, delta, delta_base),
                                    (batch, head, start, block * self.columns, q.shape[2], k.shape[2]),
                                    full,
                                    magnitude,
                                    sign,
                                    mma,
                                    thread,
                                    tid,
                                    shared_statistics,
                                    tcgen_state=tcgen_state,
                                )
                                self.product(scores, right, offset, mma, thread, tid, accum)
                                self.product(probability, other_right, offset, mma, thread, tid, accv)
                    if cutlass.const_expr(self.pipelined) or active:
                        cute.arch.sync_threads()
            self.store_result(accum, dk, left, batch, kvhead, start, offset, thread, tid, scale)
            self.store_result(accv, dv, left, batch, kvhead, start, offset, thread, tid, cutlass.Float32(1.0))
        if cutlass.const_expr(self.tcgen):
            cute.arch.sync_threads()
            tmem.free(tmem_pointer)


class ForwardLoadSpecialized(Attention):
    def __init__(self, mode, causal, masked, *configuration, vector):
        super().__init__(mode, causal, masked, *configuration, vector=vector)
        self.loader = Attention(mode, causal, masked, *configuration, vector=vector)
        self.loader.warps = 12
        self.grid_rows = 192
        self.warps = 16
        self.columns = 128
        self.specialized_load = True

    @cute.kernel
    def kernel(self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, delta_base, scale, mma, tma_args):
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        tile, bh, width = cute.arch.block_idx()
        batch = bh // q.shape[1]
        head = bh % q.shape[1]
        kvhead = head // (q.shape[1] // k.shape[1])
        group_start = tile * self.grid_rows
        offset = width * self.width
        padded = cute.ceil_div(q.shape[3], self.width) * self.width
        atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, q.element_type)
        allocator = SmemAllocator()
        full = allocator.allocate_array(cutlass.Int64, 2)
        empty = allocator.allocate_array(cutlass.Int64, 2)
        if tid == 0:
            for slot in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(full + slot, 1)
                cute.arch.mbarrier_init(empty + slot, 3)
        cute.arch.mbarrier_init_fence()
        left = self.shared(allocator, q.element_type, cute.tile_to_shape(atom, (self.grid_rows, padded), (0, 1)))
        stage_layout = cute.tile_to_shape(atom, (self.columns, padded, 2), (0, 1, 2))
        right_stages = self.shared(allocator, q.element_type, stage_layout)
        other_stages = self.shared(allocator, q.element_type, stage_layout)
        if tid < 384:
            self.loader.load(q, left, batch, head, group_start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        count = cute.ceil_div(k.shape[2], self.columns)
        if tid >= 384:
            cute.arch.setmaxregister_decrease(32)
            for block in range(count):
                slot = block % 2
                if block >= 2:
                    cute.arch.mbarrier_wait(empty + slot, (block // 2 - 1) % 2)
                self.issue_pair(
                    right_stages, other_stages, batch, kvhead, block, block, group_start, tid - 384, tma_args, full, blocks
                )
        else:
            cute.arch.setmaxregister_increase(160)
            start = group_start + (tid // 128) * self.rows
            compute_left = cute.local_tile(left, (self.rows, padded), (tid // 128, 0))
            thread = mma.get_slice(tid % 128)
            coords = thread.partition_C(cute.make_identity_tensor((self.rows, self.columns)))
            rc = row_view(coords)
            output_mma = cute.make_tiled_mma(
                warpgroup.MmaF16BF16Op(
                    q.element_type,
                    cutlass.Float32,
                    (64, self.product_width, 16),
                    warpgroup.OperandSource.RMEM,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.MN,
                )
            )
            output_thread = output_mma.get_slice(tid % 128)
            accum = cute.make_rmem_tensor(output_thread.partition_shape_C((self.rows, self.product_width)), cutlass.Float32)
            accum.fill(0.0)
            av = row_view(accum)
            magnitude = scale if scale >= 0.0 else -scale
            sign = cutlass.Float32(1.0) if scale >= 0.0 else cutlass.Float32(-1.0)
            maxima = cute.make_rmem_tensor((cute.size(rc.shape[0]),), cutlass.Float32)
            sums = cute.make_rmem_tensor(maxima.shape, cutlass.Float32)
            maxima.fill(-cutlass.Float32.inf)
            sums.fill(0.0)
            for block in range(count):
                slot = block % 2
                cute.arch.mbarrier_wait(full + slot, (block // 2) % 2)
                right = right_stages[None, None, slot]
                other_right = other_stages[None, None, slot]
                scores = self.dot(compute_left, right, mma, thread, tid)
                self.online_softmax(
                    scores,
                    coords,
                    accum,
                    maxima,
                    sums,
                    mask,
                    (batch, head, start, block * self.columns, q.shape[2], k.shape[2]),
                    True,
                    magnitude,
                    sign,
                )
                self.product(scores, other_right, offset, mma, thread, tid, accum)

                cute.arch.barrier(barrier_id=1 + tid // 128, number_of_threads=128)
                if tid % 128 == 0:
                    cute.arch.mbarrier_arrive(empty + slot)
            for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                denom = sums[r] if sums[r] > 0.0 else cutlass.Float32(1.0)
                av[r, None] = av[r, None].load() / denom
                row = start + rc[r, 0][0]
                if lane % 4 == 0 and row < q.shape[2] and width == 0:
                    maximum[batch, head, row] = maxima[r]
                    logsum[batch, head, row] = cute.math.log2(denom, fastmath=True)
            self.store_result(accum, out, compute_left, batch, head, start, offset, output_thread, tid, cutlass.Float32(1.0))

        cute.arch.sync_threads()


def descriptor(tensor):
    pointer = tensor.data_ptr()
    return (tensor.dtype, tensor.shape, tensor.stride(), min(16, pointer & -pointer))


def copy_vector(descriptors):
    return min(
        (
            next(
                n for n in (8, 4, 2, 1) if alignment >= n * 2 and shape[-1] % n == 0 and all(s % n == 0 for s in stride[:-1])
            )
            if stride[-1] == 1
            else 1
        )
        for _, shape, stride, alignment in descriptors
    )


_SM100_CONFIGURATIONS = {
    # columns, width, warps, WGMMA, centered, TMA, pipelined; optional tcgen05 flag.
    (64, "forward"): (64, 64, 2, False, False, True, True),
    (64, "q"): (32, 64, 1, False, True, True, True),
    (64, "kv"): (64, 64, 2, False, True, True, True),
    (128, "forward"): (64, 128, 4, False, False, True, True),
    (128, "q"): (32, 128, 2, False, True, True, True),
    (128, "kv"): (32, 128, 2, False, True, True, True),
    (256, "forward"): (32, 256, 4, False, False, True, True),
    (256, "q"): (32, 128, 4, False, True, True, False),
    (256, "kv"): (64, 256, 4, False, True, True, False),
    (512, "forward"): (32, 256, 4, False, False, True, True, True),
    (512, "q"): (32, 128, 4, False, True, False, False),
    (512, "kv"): (32, 256, 4, False, True, False, False),
}

_SM103_CONFIGURATIONS = {
    # columns, output width, warps, WGMMA, centered, TMA, pipeline; optional tcgen05 / native row count.
    (64, "forward"): (64, 64, 2, False, False, True, True),
    (64, "q"): (64, 64, 1, False, True, True, False),
    (64, "kv"): (64, 64, 2, False, True, True, True),
    (128, "forward"): (64, 128, 4, False, False, True, True, False, 64),
    (128, "q"): (64, 128, 4, False, True, True, False, False, 64, True),
    (128, "kv"): (64, 128, 4, False, True, True, False, False, 64, True),
    (256, "forward"): (32, 256, 4, False, False, True, True),
    (256, "q"): (64, 256, 4, False, True, True, False, False, 64, True),
    (256, "kv"): (64, 256, 4, False, True, True, False),
    (512, "forward"): (32, 256, 4, False, False, True, True, True),
    (512, "q"): (32, 256, 4, False, True, True, False, False, 64, True),
    (512, "kv"): (32, 256, 4, False, True, False, False),
}

# Shared producer/consumer pipeline; the legacy tables also serve masks and
# layouts that cannot use TMA. Each stage uses 128-row native products.
_BLACKWELL_PIPELINE_CONFIGURATIONS = {
    (dimension, mode): (
        128 if mode == "forward" else 64,
        min(dimension, 256 if mode == "forward" else 128),
        12,
        False,
        mode != "forward",
        True,
        True,
        False,
        128,
    )
    for dimension in (64, 128, 256, 512)
    for mode in ("forward", "q", "kv")
}


# Measured sm_100 path: two 64-column consumer groups each keep FP32 P for
# their own dS columns, while packed P aliases the consumed score tile.
_SM100_FUSED_CONFIGURATIONS = {
    key: (configuration if key[1] == "forward" else (128, *configuration[1:]))
    for key, configuration in _BLACKWELL_PIPELINE_CONFIGURATIONS.items()
}


# B300 MIG validation selects the same 128-column centered backward tiles.
# Keep a distinct table so later full-device tuning can diverge independently.
_SM103_FUSED_CONFIGURATIONS = dict(_SM100_FUSED_CONFIGURATIONS)


_SM120_CONFIGURATIONS = {
    # columns, output width, warps, TMA, double buffering; measured on RTX 5090.
    (64, "forward"): (64, 64, 4, True, True),
    (64, "q"): (32, 64, 4, True, True),
    (64, "kv"): (64, 64, 4, True, True),
    (128, "forward"): (32, 128, 4, True, True),
    (128, "q"): (32, 128, 4, False, False),
    (128, "kv"): (32, 128, 4, False, False),
    (256, "forward"): (16, 256, 4, True, False),
    (256, "q"): (32, 128, 4, False, False),
    (256, "kv"): (32, 256, 4, False, False),
    (512, "forward"): (16, 256, 4, False, False),
    (512, "q"): (16, 128, 2, False, False),
    (512, "kv"): (32, 256, 2, False, False),
}


def _aligned_tiles(descriptors):
    return all(
        alignment >= 16 and stride[-1] == 1 and all(s > 0 and s % 8 == 0 for s in stride[:-1])
        for _, _, stride, alignment in (descriptors[i] for i in (0, 1, 2, 7))
    )


def _tma_compatible(dimension, descriptors):
    return dimension in (64, 128, 256) and _aligned_tiles(descriptors)


def kernel_configuration(device, mode, descriptors):
    capability = torch.cuda.get_device_capability(device)
    dimension = descriptors[0][1][-1]
    columns = 32 if mode == "kv" or mode == "q" and dimension > 128 else 64
    width = min(256, (dimension + 31) // 32 * 32)
    warps = 4 if dimension <= 256 else 2 if mode == "kv" else 1
    if dimension > 128 and capability == (8, 9):
        columns, warps = (32, (4 if mode == "kv" else 2) if dimension <= 256 else (2 if mode == "kv" else 1))
    if dimension > 256:
        columns = 32
    hopper = capability == (9, 0) and (dimension <= 256)
    if hopper:
        columns, warps = 64 if width % 64 == 0 else 32, 4
        if mode == "kv" and dimension == 256:
            columns, warps = 32, 8
        elif mode != "forward" and dimension > 128:
            hopper = False
    centered = capability in ((8, 9), (9, 0), (10, 0), (10, 3), (12, 0)) and mode != "forward"
    if centered and mode == "q":
        width = (dimension + 31) // 32 * 32 if dimension <= 192 else 128
        warps = 4 if dimension <= 256 else 1
        hopper = capability == (9, 0) and dimension <= 256
        columns = 64 if hopper and dimension <= 128 and width % 64 == 0 else 32
    tma = hopper and _tma_compatible(dimension, descriptors)
    if mode == "forward":
        if dimension == 128 and tma and prod(descriptors[0][1][:-1]) >= 32768:
            columns = 32
        elif capability == (8, 9) and dimension == 256:
            warps = 4
    if capability in ((10, 0), (10, 3)):
        configurations = _SM100_CONFIGURATIONS if capability == (10, 0) else _SM103_CONFIGURATIONS
        selected = configurations.get((dimension, mode))
        if selected is not None:
            requires_aligned = selected[5] or any(selected[7:])
            eligible = descriptors[3][0] != torch.bool and _aligned_tiles(descriptors)
            # Small sequences do not amortize the larger producer/consumer CTA.
            threshold = 1024 if dimension == 64 else 512 if dimension == 128 else 256
            sequence = max(descriptors[0][1][2], descriptors[1][1][2])
            if eligible and sequence >= threshold:
                pipelines = _SM100_FUSED_CONFIGURATIONS if capability == (10, 0) else _SM103_FUSED_CONFIGURATIONS
                pipeline = pipelines.get((dimension, mode))
                if pipeline is not None:
                    return pipeline
            if not requires_aligned or eligible:
                return selected
    if capability == (12, 0):
        selected = _SM120_CONFIGURATIONS.get((dimension, mode))
        if (
            mode == "forward"
            and dimension == 512
            and descriptors[3][0] != torch.bool
            and descriptors[0][0] in (torch.float16, torch.bfloat16)
            and _aligned_tiles(descriptors)
            and selected == (16, 256, 4, False, False)
        ):
            return 32, 512, 4, False, False, False, False

        if (
            mode == "kv"
            and dimension == 512
            and descriptors[3][0] != torch.bool
            and descriptors[0][0] in (torch.float16, torch.bfloat16)
            and _aligned_tiles(descriptors)
            and selected == (32, 256, 2, False, False)
        ):
            return 32, 512, 4, False, True, False, False
        if selected is not None:
            cols, output_width, selected_warps, use_tma, pipeline = selected
            # Sparse boolean masks retain the generic tile-skipping path.
            eligible = descriptors[3][0] != torch.bool and _tma_compatible(dimension, descriptors)
            if not use_tma or eligible:
                return cols, output_width, selected_warps, False, mode != "forward", use_tma, pipeline
    return (columns, width, warps, hopper, centered, tma, tma and dimension in (64, 128))


def backward_query_configuration(device, causal, masked, descriptors):
    configuration = kernel_configuration(device, "q", descriptors)
    if (
        torch.cuda.get_device_capability(device) == (12, 0)
        and not masked
        and descriptors[0][0] in (torch.float16, torch.bfloat16)
        and descriptors[0][1][-1] == 512
        and configuration == (16, 128, 2, False, True, False, False)
        and _aligned_tiles(descriptors)
    ):
        return 32, 512, 4, False, True, False, False
    # Share QK/dP across the full output width instead of recomputing them per slice.
    if (
        torch.cuda.get_device_capability(device) == (12, 0)
        and not masked
        and descriptors[0][0] in (torch.float16, torch.bfloat16)
        and descriptors[0][1][-1] == 256
        and configuration == (32, 128, 4, False, True, False, False)
        and _aligned_tiles(descriptors)
    ):
        return 64, 256, 8, False, True, False, False
    return configuration


def compile_kernel(device, mode, causal, masked, descriptors):
    configuration = (
        backward_query_configuration(device, causal, masked, descriptors)
        if mode == "backward"
        else kernel_configuration(device, mode, descriptors)
    )
    specialized_load = (
        mode == "forward"
        and not (causal or masked)
        and configuration[3]
        and configuration[5]
        and descriptors[0][1][-1] == 128
        and descriptors[0][1][2] >= 1536
        and descriptors[1][1][2] >= 1024
    )
    if specialized_load:
        configuration = (64, configuration[1], 8, *configuration[3:])
    kv_configuration = kernel_configuration(device, "kv", descriptors)
    vector = copy_vector(descriptors[index] for index in (0, 1, 2, 7))
    base_vector = copy_vector(descriptors[i] for i in (4, 7))
    # Match native.cpp's cache classes: runtime sequence axes and aligned strides.
    layouts = tuple(
        (
            dtype,
            tuple(
                -1 if size != 1 and (axis == 2 or axis == 3 and (i == 12 or i == 3 and masked)) else size
                for axis, size in enumerate(shape)
            ),
            tuple(s if s <= 1 else -min(8, s & -s) if dtype in (torch.float16, torch.bfloat16) else -1 for s in stride),
            alignment,
        )
        for i, (dtype, shape, stride, alignment) in enumerate(descriptors)
    )
    return _compile_kernel(
        device,
        mode,
        causal,
        masked,
        layouts,
        configuration,
        kv_configuration,
        vector,
        base_vector,
        specialized_load,
        backward_options(device, causal, masked, descriptors, configuration, kv_configuration),
    )


def backward_options(device, causal, masked, descriptors, q_configuration, kv_configuration):
    # These options are part of the Python compilation cache key. Keep shape-dependent
    # choices within the same runtime sequence/stride classes used by native.cpp.
    if (
        torch.cuda.get_device_capability(device) != (12, 0)
        or masked
        or descriptors[0][0] not in (torch.float16, torch.bfloat16)
    ):
        return 1, False
    dimension = descriptors[0][1][-1]
    column_warps = 4 if dimension == 256 and q_configuration == (64, 256, 8, False, True, False, False) else 1
    if dimension == 512 and q_configuration == (32, 256, 2, False, True, False, False):
        column_warps = 2
    if dimension == 512 and q_configuration == (32, 512, 4, False, True, False, False):
        column_warps = 4
    # Prefetch each row's four scalars once while QK/dP execute. Restrict the
    # extra shared memory/barrier to the head/mask classes that benefit.
    statistics = (
        dimension == 64
        and causal
        and kv_configuration == (64, 64, 4, False, True, True, True)
        or dimension == 256
        and not causal
        and kv_configuration == (32, 256, 4, False, True, False, False)
    )
    return column_warps, statistics


@lru_cache(maxsize=128)
def _compile_kernel(
    device,
    mode,
    causal,
    masked,
    layouts,
    configuration,
    kv_configuration,
    vector,
    base_vector,
    specialized_load,
    backward_options,
):
    dtypes = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.bool: cutlass.Boolean,
        torch.int32: cutlass.Int32,
    }
    arguments = [
        make_fake_tensor(
            dtypes[dtype],
            tuple(cute.sym_int() if size == -1 else size for size in shape),
            stride=tuple(cute.sym_int64(divisibility=-s) if s < 0 else s for s in stride),
            memspace=cute.AddressSpace.gmem,
            assumed_align=alignment,
        )
        for dtype, shape, stride, alignment in layouts
    ]
    if mode == "backward":
        base_dot = configuration[4] and not (configuration[3] and layouts[0][1][-1] in (64, 128))
        program = JointBackward(
            causal,
            masked,
            configuration,
            kv_configuration,
            vector,
            base_vector,
            base_dot,
            reload_center=torch.cuda.get_device_capability(device) in ((10, 0), (10, 3), (12, 0)),
            hardware_reductions=torch.cuda.get_device_capability(device) == (10, 3),
            blackwell_operand_chunk=128 if layouts[0][1][-1] == 128 else 64,
            query_column_warps=backward_options[0],
            stage_statistics=backward_options[1],
        )
        if (
            not masked
            and layouts[0][0] == torch.bfloat16
            and layouts[0][1][-1] == 512
            and torch.cuda.get_device_capability(device) == (10, 3)
            and configuration == _SM103_FUSED_CONFIGURATIONS[(512, "q")]
            and kv_configuration == _SM103_FUSED_CONFIGURATIONS[(512, "kv")]
            and layouts[0][1][1] == layouts[1][1][1]
            and layouts[0][1][0] * layouts[0][1][1] >= 8
        ):
            from .blackwell_pipeline import Backward as PipelineBackward

            # Preserve explicit pipeline sources supplied by profiling/tuning tools.
            if isinstance(program.query, PipelineBackward) and isinstance(program.keyvalue, PipelineBackward):
                from .blackwell_register_aux import AdaptiveBackward

                program = AdaptiveBackward(causal, program, base_vector)
    elif mode == "forward" and masked:
        program = JointForward(causal, configuration, vector)
    else:
        if specialized_load:
            program = ForwardLoadSpecialized(mode, causal, masked, *configuration, vector=vector)
        elif len(configuration) > 8 and configuration[8]:
            if configuration[2] == 12:
                from .blackwell_pipeline import Forward
            else:
                from .sm103 import Forward

            program = Forward(
                mode,
                causal,
                masked,
                *configuration,
                vector=vector,
                hardware_reductions=torch.cuda.get_device_capability(device) == (10, 3),
                **({"head_dimension": layouts[0][1][-1]} if configuration[2] == 12 else {}),
            )
        else:
            program = Attention(
                mode,
                causal,
                masked,
                *configuration,
                vector=vector,
                reload_center=torch.cuda.get_device_capability(device) in ((10, 0), (10, 3), (12, 0)),
            )
    with torch.cuda.device(device):
        return cute.compile(
            program,
            *arguments,
            cutlass.Float32(1.0),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def centered_backward(q):
    return torch.cuda.get_device_capability(q.device.index) in ((8, 9), (9, 0), (10, 0), (10, 3), (12, 0))


class PackMask:
    @cute.jit
    def __call__(self, mask: cute.Tensor, blocks: cute.Tensor, stream: cuda.CUstream):
        self.kernel(mask, blocks).launch(
            grid=(cute.ceil_div(blocks.shape[2] * blocks.shape[3], 4), blocks.shape[0] * blocks.shape[1], 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mask, blocks):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, _ = cute.arch.block_idx()
        tile = tile * 4 + tid // 32
        qr, kc = tile // blocks.shape[3], tile % blocks.shape[3]
        batch, head = bh // blocks.shape[1], bh % blocks.shape[1]
        present = cutlass.Int32(0)
        missing = cutlass.Int32(0)
        if tile < blocks.shape[2] * blocks.shape[3]:
            for i in cutlass.range(
                cute.ceil_div(cutlass.min(64, mask.shape[2]) * cutlass.min(64, mask.shape[3]), 32), unroll=4
            ):
                index = i * 32 + tid % 32
                row, col = (
                    qr * 64 + index // cutlass.min(64, mask.shape[3]),
                    kc * 64 + index % cutlass.min(64, mask.shape[3]),
                )
                if row < mask.shape[2] and col < mask.shape[3]:
                    if mask[batch, head, row, col]:
                        present += 1
                    else:
                        missing += 1
        present = cute.arch.warp_redux_sync(present, "add")
        missing = cute.arch.warp_redux_sync(missing, "add")
        if tid % 32 == 0 and tile < blocks.shape[2] * blocks.shape[3]:
            blocks[batch, head, qr, kc] = (
                cutlass.Int32(0) if present == 0 else cutlass.Int32(2) if missing == 0 else cutlass.Int32(1)
            )


class OutputDelta:
    def __init__(self, vector):
        self.loader = Attention("forward", False, False, columns=16, warps=1, vector=vector)

    @cute.jit
    def __call__(self, out: cute.Tensor, dout: cute.Tensor, delta: cute.Tensor, stream: cuda.CUstream):
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(out.element_type, cutlass.Float32, (16, 8, 16)), (1, 1, 1), permutation_mnk=(16, 16, 16)
        )
        self.kernel(out, dout, delta, mma).launch(
            grid=(cute.ceil_div(out.shape[2], 16), out.shape[0] * out.shape[1], 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, out, dout, delta, mma):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, _ = cute.arch.block_idx()
        batch, head = bh // out.shape[1], bh % out.shape[1]
        padded = cute.ceil_div(out.shape[3], 32) * 32
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        layout = cute.tile_to_shape(atom, (16, padded), (0, 1))
        allocator = SmemAllocator()
        left = allocator.allocate_tensor(out.element_type, layout, byte_alignment=128)
        right = allocator.allocate_tensor(out.element_type, layout, byte_alignment=128)
        self.loader.load(dout, left, batch, head, tile * 16, tid)
        self.loader.load(out, right, batch, head, tile * 16, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        thread = mma.get_slice(tid)
        scores = self.loader.dot(left, right, mma, thread, tid)
        coords = row_view(thread.partition_C(cute.make_identity_tensor((16, 16))))
        sv = row_view(scores)
        for r in cutlass.range_constexpr(cute.size(coords.shape[0])):
            value = cutlass.Float32(0.0)
            row = coords[r, 0][0]
            for c in cutlass.range_constexpr(cute.size(coords.shape[1])):
                if coords[r, c][0] == coords[r, c][1]:
                    value = sv[r, c]
            value = quad_sum(value)
            if tid % 4 == 0 and tile * 16 + row < out.shape[2]:
                delta[batch, head, tile * 16 + row] = value


class JointBackward:
    def __init__(
        self,
        causal,
        masked,
        q_configuration,
        kv_configuration,
        vector,
        base_vector,
        base_dot,
        reload_center=False,
        hardware_reductions=True,
        blackwell_operand_chunk=64,
        query_column_warps=1,
        stage_statistics=False,
    ):
        if len(q_configuration) > 8 and q_configuration[8]:
            if q_configuration[2] == 12:
                from .blackwell_pipeline import Query
            else:
                from .sm103 import Query

            self.query = Query(
                "q",
                causal,
                masked,
                *q_configuration[:9],
                vector=vector,
                reload_center=reload_center,
                stream_aux=len(q_configuration) > 9 and q_configuration[9],
                hardware_reductions=hardware_reductions,
                **({"operand_chunk": blackwell_operand_chunk} if q_configuration[2] == 12 else {}),
            )
        else:
            self.query = Attention(
                "q",
                causal,
                masked,
                *q_configuration,
                vector=vector,
                reload_center=reload_center,
                query_column_warps=query_column_warps,
            )
        if len(kv_configuration) > 8 and kv_configuration[8]:
            if kv_configuration[2] == 12:
                from .blackwell_pipeline import KeyValue
            else:
                from .sm103 import KeyValue

            self.keyvalue = KeyValue(
                "kv",
                causal,
                masked,
                *kv_configuration[:9],
                vector=vector,
                stream_aux=len(kv_configuration) > 9 and kv_configuration[9],
                hardware_reductions=hardware_reductions,
                **({"operand_chunk": blackwell_operand_chunk} if kv_configuration[2] == 12 else {}),
            )
        else:
            self.keyvalue = Attention(
                "kv", causal, masked, *kv_configuration, vector=vector, stage_statistics=stage_statistics
            )
        self.base_dot = base_dot
        self.centering = OutputDelta(base_vector)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        mask: cute.Tensor,
        out: cute.Tensor,
        maximum: cute.Tensor,
        logsum: cute.Tensor,
        dout: cute.Tensor,
        delta: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        blocks: cute.Tensor,
        delta_base: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.base_dot):
            self.centering(out, dout, delta_base, stream)
        self.query(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, delta_base, scale, stream)
        self.keyvalue(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, delta_base, scale, stream)


class JointForward:
    def __init__(self, causal, configuration, vector):
        self.classify = PackMask()
        self.attention = Attention("forward", causal, True, *configuration, vector=vector)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        mask: cute.Tensor,
        out: cute.Tensor,
        maximum: cute.Tensor,
        logsum: cute.Tensor,
        dout: cute.Tensor,
        delta: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        blocks: cute.Tensor,
        delta_base: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        source = cute.make_tensor(
            mask.iterator,
            cute.make_layout(
                tuple(1 if cute.is_static(mask.stride[i]) and mask.stride[i] == 0 else mask.shape[i] for i in range(4)),
                stride=mask.stride,
            ),
        )
        self.classify(source, blocks, stream)
        expanded = cute.make_tensor(
            blocks.iterator,
            cute.make_layout(
                (q.shape[0], q.shape[1], cute.ceil_div(q.shape[2], 64), cute.ceil_div(k.shape[2], 64)),
                stride=tuple(
                    0 if cute.is_static(blocks.shape[i]) and blocks.shape[i] == 1 else blocks.stride[i] for i in range(4)
                ),
            ),
        )
        self.attention(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, expanded, delta_base, scale, stream)
