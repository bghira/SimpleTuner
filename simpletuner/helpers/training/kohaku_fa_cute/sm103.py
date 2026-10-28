"""Native Blackwell products with architecture-specific TMEM reductions."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05, warpgroup
from cutlass.memory import SmemAllocator, TmemAllocator

from .blackwell import fence_after_sync, fence_before_sync
from .kernels import Attention


class Forward(Attention):
    def __init__(self, *args, hardware_reductions=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.hardware_reductions = hardware_reductions
        if self.mode != "forward" or self.masked or self.hopper or self.tcgen or self.warps != 4:
            raise ValueError("The sm_103 forward path requires four warps and no explicit mask.")
        self.configure_rows()

    @cute.jit
    def issue(self, mma, a, b, accum, accumulate=False):
        fence_after_sync()
        # Each issue owns its mutable MMA fields; staged branches must not share
        # SSA operands left behind by another issue.
        mma = cute.make_tiled_mma(mma.op)
        mma.set(tcgen05.Field.ACCUMULATE, accumulate)
        part = mma.get_slice(0)
        ra = mma.make_fragment_A(part.partition_A(a))
        rb = mma.make_fragment_B(part.partition_B(b))
        for chunk in cutlass.range(cute.size(ra.shape[2]), unroll_full=True):
            cute.gemm(mma, accum, ra[None, None, chunk], rb[None, None, chunk], accum)
            mma.set(tcgen05.Field.ACCUMULATE, True)

    def configure_rows(self):
        rows = self.sm103_rows
        if rows not in (64, 128) or self.columns not in (32, 64, 128) or self.width not in (64, 128, 256):
            raise ValueError("Native sm_103 tiles require 64/128 rows and supported column/output widths.")
        if self.pipelined and not self.tma:
            raise ValueError("Native double buffering requires TMA.")
        if self.mode != "forward" and not self.centered:
            raise ValueError("Native backward requires centered softmax statistics.")
        if self.mode != "forward":
            columns = self.width + max(self.width, 2 * self.columns) if self.stream_aux else 2 * (self.columns + self.width)
            if columns > 512:
                raise ValueError("Native backward accumulators exceed the TMEM allocation limit.")
        self.rows = self.grid_rows = rows

    @cute.jit
    def row_index(self, tid):
        # M=64 assigns two column halves to lanes separated by 16 in each warp.
        return tid if cutlass.const_expr(self.rows == 128) else tid // 32 * 16 + tid % 16

    @cute.jit
    def row_sum(self, value):
        if cutlass.const_expr(self.rows == 64):
            value += cute.arch.shuffle_sync_bfly(value, offset=16, mask=-1, mask_and_clamp=31)
        return value

    @cute.jit
    def row_maximum(self, value):
        if cutlass.const_expr(self.rows == 64):
            value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=16, mask=-1, mask_and_clamp=31))
        return value

    @cute.jit
    def copying(self, accumulator, tid, columns: cutlass.Constexpr, reduction: cutlass.Constexpr = None):
        view = accumulator[(None, None), 0, 0]
        if cutlass.const_expr(self.rows == 64 and self.hardware_reductions):
            red = tcgen05.copy.TmemLoadRedOp.MAX if reduction is None else reduction
            op = tcgen05.LdRed16x32bx2Op(tcgen05.Repetition(columns // 2), redOp=red, half_split_off=columns // 2)
        elif cutlass.const_expr(self.rows == 64):
            op = tcgen05.Ld16x32bx2Op(tcgen05.Repetition(columns // 2))
        elif cutlass.const_expr(reduction is not None and self.hardware_reductions):
            op = tcgen05.LdRed32x32bOp(tcgen05.Repetition(columns), redOp=reduction)
        else:
            op = tcgen05.Ld32x32bOp(tcgen05.Repetition(min(columns, 128)))
        atom = cute.make_copy_atom(op, cutlass.Float32)
        copy = tcgen05.make_tmem_copy(atom, view)
        part = copy.get_slice(tid)
        return copy, part.partition_S(view), part.partition_D(cute.make_identity_tensor((self.rows, columns)))

    @cute.jit
    def read(self, accumulator, tid, columns: cutlass.Constexpr):
        fence_after_sync()
        copy, src, coords = self.copying(accumulator, tid, columns)
        values = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
        reduced = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
        if cutlass.const_expr(self.rows == 64 and self.hardware_reductions):
            cute.copy(copy, src, [values, reduced])
        else:
            cute.copy(copy, src, values)
        cute.arch.fence_view_async_tmem_load()
        fence_before_sync()
        return values, coords, reduced

    @cute.jit
    def read_scores(self, accumulator, tid, scale):
        if cutlass.const_expr(not self.hardware_reductions):
            values, coords, _ = self.read(accumulator, tid, self.columns)
            sign = cutlass.Float32(1) if scale >= 0 else cutlass.Float32(-1)
            current = (values.load() * sign).reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
        else:
            fence_after_sync()
            maximum_copy, src, coords = self.copying(accumulator, tid, self.columns, tcgen05.copy.TmemLoadRedOp.MAX)
            minimum_copy, _, _ = self.copying(accumulator, tid, self.columns, tcgen05.copy.TmemLoadRedOp.MIN)
            values = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
            reduced = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
            if scale >= 0:
                cute.copy(maximum_copy, src, [values, reduced])
            else:
                cute.copy(minimum_copy, src, [values, reduced])
            cute.arch.fence_view_async_tmem_load()
            fence_before_sync()
            current = reduced[0] if scale >= 0 else -reduced[0]
        return values, coords, current

    @cute.jit
    def store(self, source, target, batch, head, start, offset, tid):
        copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), target.element_type, num_bits_per_copy=16 * self.vector),
            cute.make_layout((32, 4), stride=(4, 1)),
            cute.make_layout((1, self.vector)),
        )
        part = copy.get_slice(tid)
        src = cute.local_tile(source, (self.rows, self.width), (0, 0))
        dst = cute.local_tile(
            target[batch, head, None, None], (self.rows, self.width), (start // self.rows, offset // self.width)
        )
        ids = cute.local_tile(
            cute.make_identity_tensor((target.shape[2], target.shape[3])),
            (self.rows, self.width),
            (start // self.rows, offset // self.width),
        )
        inputs, outputs, coords = part.partition_S(src), part.partition_D(dst), part.partition_D(ids)
        for r in cutlass.range_constexpr(cute.size(outputs.shape[1])):
            for d in cutlass.range_constexpr(cute.size(outputs.shape[2])):
                row, dim = coords[0, r, d]
                if row < target.shape[2] and dim < target.shape[3]:
                    cute.copy(copy, inputs[None, r, d], outputs[None, r, d])

    @cute.kernel
    def kernel(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, delta_base, scale, unused, tma_args
    ):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, wi = self.grid_coordinates(q, k)
        batch, head = bh // q.shape[1], bh % q.shape[1]
        kvhead = head // (q.shape[1] // k.shape[1])
        start, offset = tile * self.rows, wi * self.width
        padded = cute.ceil_div(q.shape[3], self.width) * self.width
        allocator = SmemAllocator()
        mbar = allocator.allocate_array(cutlass.Int64, 1)
        load_bar = allocator.allocate_array(cutlass.Int64, 2)
        holding = allocator.allocate_array(cutlass.Int32, 1)
        atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128 if padded % 64 == 0 else warpgroup.SmemLayoutAtomKind.K_SW64,
            q.element_type,
        )
        left = self.shared(allocator, q.element_type, cute.tile_to_shape(atom, (self.rows, padded), (0, 1)))
        stages = 2 if cutlass.const_expr(self.pipelined) else 1
        right_stages = self.shared(
            allocator, q.element_type, cute.tile_to_shape(atom, (self.columns, padded, stages), (0, 1, 2))
        )
        other_stages = self.shared(
            allocator, q.element_type, cute.tile_to_shape(atom, (self.columns, padded, stages), (0, 1, 2))
        )
        weight_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128 if self.columns % 64 == 0 else warpgroup.SmemLayoutAtomKind.K_SW64,
            q.element_type,
        )
        probability = self.shared(
            allocator, q.element_type, cute.tile_to_shape(weight_atom, (self.rows, self.columns), (0, 1))
        )
        tmem = TmemAllocator(
            alloc_result_dst_smem_ptr=holding,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=128),
        )
        tmem.allocate(128 if self.columns + self.width <= 128 else 256 if self.columns + self.width <= 256 else 512)
        tmem.wait_for_alloc()
        pointer = tmem.retrieve_ptr(cutlass.Float32)
        tmem.relinquish_alloc_permit()
        if tid == 0:
            cute.arch.mbarrier_init(mbar, 1)
            for slot in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(load_bar + slot, 1)
        cute.arch.mbarrier_init_fence()
        self.load(q, left, batch, head, start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        score_mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                cutlass.Float32,
                (self.rows, self.columns, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        product_mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                cutlass.Float32,
                (self.rows, self.width, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.MN,
            )
        )
        sf = score_mma.make_fragment_C(score_mma.get_slice(0).partition_shape_C((self.rows, self.columns)))
        pf = product_mma.make_fragment_C(product_mma.get_slice(0).partition_shape_C((self.rows, self.width)))
        score_acc = cute.make_tensor(pointer, sf.layout)
        product_acc = cute.make_tensor(pointer + self.columns, pf.layout)
        _, _, output_coords = self.copying(product_acc, tid, self.width)
        result = cute.make_rmem_tensor((self.width * self.rows // 128,), cutlass.Float32)
        result.fill(0.0)
        row_max, row_sum = -cutlass.Float32.inf, cutlass.Float32(0)
        magnitude = scale if scale >= 0 else -scale
        sign = cutlass.Float32(1) if scale >= 0 else cutlass.Float32(-1)
        limit = cutlass.min(k.shape[2], start + self.rows) if cutlass.const_expr(self.causal) else k.shape[2]
        count = cute.ceil_div(limit, self.columns)
        phase, load_phase = cutlass.Int32(0), cutlass.Int32(0)
        if cutlass.const_expr(self.pipelined):
            self.issue_pair(right_stages, other_stages, batch, kvhead, 0, 0, start, tid, tma_args, load_bar, blocks)
        for block in range(count):
            if cutlass.const_expr(self.pipelined):
                right, other_right = self.pipeline_step(
                    right_stages, other_stages, batch, kvhead, block, block, count, start, tid, tma_args, load_bar, blocks
                )
            else:
                right, other_right = right_stages[None, None, 0], other_stages[None, None, 0]
                load_phase = self.load_pair(
                    k, v, right, other_right, batch, kvhead, block * self.columns, tid, tma_args, load_bar, load_phase
                )
            cute.arch.fence_view_async_shared()
            if tid < 32:
                self.issue(score_mma, left, right, score_acc)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar)
            cute.arch.mbarrier_wait(mbar, phase)
            phase = phase ^ 1
            scores, coords, current = self.read_scores(score_acc, tid, scale)
            full = block * self.columns + self.columns <= k.shape[2]
            if cutlass.const_expr(self.causal):
                full = full and block * self.columns + self.columns <= start + 1
            for i in cutlass.range_constexpr(cute.size(scores)):
                row, col = coords[i]
                valid = start + row < q.shape[2] and block * self.columns + col < k.shape[2]
                if cutlass.const_expr(self.causal):
                    valid = valid and block * self.columns + col <= start + row
                scores[i] = scores[i] * sign if valid else -cutlass.Float32.inf
            if not full:
                current = scores.load().reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            current = self.row_maximum(current)
            new = cute.arch.fmax(row_max, current)
            alpha = cutlass.Float32(0)
            if row_max != -cutlass.Float32.inf:
                alpha = cute.math.exp2((row_max - new) * magnitude * 1.4426950408889634, fastmath=True)
            for i in cutlass.range_constexpr(cute.size(scores)):
                row, col = coords[i]
                p = cutlass.Float32(0)
                if scores[i] != -cutlass.Float32.inf:
                    p = cute.math.exp2((scores[i] - new) * magnitude * 1.4426950408889634, fastmath=True)
                scores[i] = p
                probability[row, col] = p.to(q.element_type)
            row_sum = row_sum * alpha + self.row_sum(scores.load().reduce(cute.ReductionOp.ADD, 0.0, 0))
            row_max = new
            cute.arch.sync_threads()
            cute.arch.fence_view_async_shared()
            source = cute.local_tile(other_right, (self.columns, self.width), (0, offset // self.width))
            transposed = cute.composition(source, cute.make_layout((self.width, self.columns), stride=(self.columns, 1)))
            if tid < 32:
                self.issue(product_mma, probability, transposed, product_acc)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar)
            cute.arch.mbarrier_wait(mbar, phase)
            phase = phase ^ 1
            values, _, _ = self.read(product_acc, tid, self.width)
            for i in cutlass.range_constexpr(cute.size(result)):
                result[i] = result[i] * alpha + values[i]
            cute.arch.sync_threads()
        denominator = row_sum if row_sum > 0 else cutlass.Float32(1)
        for i in cutlass.range_constexpr(cute.size(result)):
            row, col = output_coords[i]
            left[row, col] = (result[i] / denominator).to(q.element_type)
        if start + self.row_index(tid) < q.shape[2] and wi == 0 and (self.rows == 128 or tid % 32 < 16):
            maximum[batch, head, start + self.row_index(tid)] = row_max
            logsum[batch, head, start + self.row_index(tid)] = cute.math.log2(denominator, fastmath=True)
        cute.arch.sync_threads()
        self.store(left, out, batch, head, start, offset, tid)
        cute.arch.sync_threads()
        tmem.free(pointer)


class Query(Forward):
    """Centered query backward with persistent native gradient products."""

    def __init__(self, *args, stream_aux=False, hardware_reductions=True, **kwargs):
        Attention.__init__(self, *args, **kwargs)
        self.stream_aux = stream_aux
        self.hardware_reductions = hardware_reductions
        if self.mode != "q" or self.masked or self.hopper or self.tcgen or self.warps != 4:
            raise ValueError("The sm_103 query path requires four warps and no explicit mask.")
        self.configure_rows()

    @cute.kernel
    def kernel(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, delta_base, scale, unused, tma_args
    ):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, wi = self.grid_coordinates(q, k)
        batch, head = bh // q.shape[1], bh % q.shape[1]
        kvhead = head // (q.shape[1] // k.shape[1])
        start, offset = tile * self.rows, wi * self.width
        padded = cute.ceil_div(q.shape[3], self.width) * self.width
        allocator = SmemAllocator()
        mbar = allocator.allocate_array(cutlass.Int64, 1)
        load_bar = allocator.allocate_array(cutlass.Int64, 2)
        holding = allocator.allocate_array(cutlass.Int32, 1)
        atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128 if padded % 64 == 0 else warpgroup.SmemLayoutAtomKind.K_SW64,
            q.element_type,
        )
        layout = cute.tile_to_shape(atom, (self.rows, padded), (0, 1))
        left = self.shared(allocator, q.element_type, layout)
        other_left = self.shared(allocator, q.element_type, layout)
        stages = 2 if cutlass.const_expr(self.pipelined) else 1
        stage_layout = cute.tile_to_shape(atom, (self.columns, padded, stages), (0, 1, 2))
        right_stages = self.shared(allocator, q.element_type, stage_layout)
        other_stages = self.shared(allocator, q.element_type, stage_layout)
        weight_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128 if self.columns % 64 == 0 else warpgroup.SmemLayoutAtomKind.K_SW64,
            q.element_type,
        )
        weight_layout = cute.tile_to_shape(weight_atom, (self.rows, self.columns), (0, 1))
        ds = self.shared(allocator, q.element_type, weight_layout)
        probability = self.shared(allocator, q.element_type, weight_layout)
        columns = (
            self.width + max(self.width, 2 * self.columns)
            if cutlass.const_expr(self.stream_aux)
            else 2 * (self.columns + self.width)
        )
        tmem = TmemAllocator(
            alloc_result_dst_smem_ptr=holding,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=128),
        )
        tmem.allocate(128 if columns <= 128 else 256 if columns <= 256 else 512)
        tmem.wait_for_alloc()
        pointer = tmem.retrieve_ptr(cutlass.Float32)
        tmem.relinquish_alloc_permit()
        if tid == 0:
            cute.arch.mbarrier_init(mbar, 1)
            for slot in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(load_bar + slot, 1)
        cute.arch.mbarrier_init_fence()
        row_max, row_logsum, center = cutlass.Float32(0), cutlass.Float32(0), cutlass.Float32(0)
        if start + self.row_index(tid) < q.shape[2]:
            row_max = maximum[batch, head, start + self.row_index(tid)]
            row_logsum = logsum[batch, head, start + self.row_index(tid)]
            center = delta_base[batch, head, start + self.row_index(tid)]
        total, rounded_total = cutlass.Float32(0), cutlass.Float32(0)
        self.load(q, left, batch, head, start, tid)
        self.load(dout, other_left, batch, head, start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        score_mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                cutlass.Float32,
                (self.rows, self.columns, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        product_mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                cutlass.Float32,
                (self.rows, self.width, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.MN,
            )
        )
        sf = score_mma.make_fragment_C(score_mma.get_slice(0).partition_shape_C((self.rows, self.columns)))
        pf = product_mma.make_fragment_C(product_mma.get_slice(0).partition_shape_C((self.rows, self.width)))
        score_acc = cute.make_tensor(pointer, sf.layout)
        dp_acc = cute.make_tensor(pointer + self.columns, sf.layout)
        if cutlass.const_expr(self.stream_aux):
            query_acc = cute.make_tensor(pointer + max(self.width, 2 * self.columns), pf.layout)
            mean_acc = cute.make_tensor(pointer, pf.layout)
            auxiliary = cute.make_rmem_tensor((self.width * self.rows // 128,), cutlass.Float32)
            auxiliary.fill(0.0)
        else:
            query_acc = cute.make_tensor(pointer + 2 * self.columns, pf.layout)
            mean_acc = cute.make_tensor(pointer + 2 * self.columns + self.width, pf.layout)
        magnitude = scale if scale >= 0 else -scale
        sign = cutlass.Float32(1) if scale >= 0 else cutlass.Float32(-1)
        limit = cutlass.min(k.shape[2], start + self.rows) if cutlass.const_expr(self.causal) else k.shape[2]
        count = cute.ceil_div(limit, self.columns)
        phase, load_phase = cutlass.Int32(0), cutlass.Int32(0)
        if cutlass.const_expr(self.pipelined):
            self.issue_pair(right_stages, other_stages, batch, kvhead, 0, 0, start, tid, tma_args, load_bar, blocks)
        for block in range(count):
            if cutlass.const_expr(self.pipelined):
                right, other_right = self.pipeline_step(
                    right_stages, other_stages, batch, kvhead, block, block, count, start, tid, tma_args, load_bar, blocks
                )
            else:
                right, other_right = right_stages[None, None, 0], other_stages[None, None, 0]
                load_phase = self.load_pair(
                    k, v, right, other_right, batch, kvhead, block * self.columns, tid, tma_args, load_bar, load_phase
                )
            cute.arch.fence_view_async_shared()
            if tid < 32:
                self.issue(score_mma, left, right, score_acc)
                self.issue(score_mma, other_left, other_right, dp_acc)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar)
            cute.arch.mbarrier_wait(mbar, phase)
            phase = phase ^ 1
            scores, coords, _ = self.read(score_acc, tid, self.columns)
            dps, _, _ = self.read(dp_acc, tid, self.columns)
            for i in cutlass.range_constexpr(cute.size(scores)):
                row, col = coords[i]
                valid = start + row < q.shape[2] and block * self.columns + col < k.shape[2]
                if cutlass.const_expr(self.causal):
                    valid = valid and block * self.columns + col <= start + row
                p = cutlass.Float32(0)
                if valid:
                    p = cute.math.exp2(
                        (scores[i] * sign - row_max) * magnitude * 1.4426950408889634 - row_logsum, fastmath=True
                    )
                value = p * (dps[i] - center)
                rounded = value.to(q.element_type)
                scores[i] = value
                dps[i] = rounded.to(cutlass.Float32)
                ds[row, col] = rounded
                probability[row, col] = p.to(q.element_type)
            total += self.row_sum(scores.load().reduce(cute.ReductionOp.ADD, 0.0, 0))
            rounded_total += self.row_sum(dps.load().reduce(cute.ReductionOp.ADD, 0.0, 0))
            cute.arch.sync_threads()
            cute.arch.fence_view_async_shared()
            source = cute.local_tile(right, (self.columns, self.width), (0, offset // self.width))
            transposed = cute.composition(source, cute.make_layout((self.width, self.columns), stride=(self.columns, 1)))
            if tid < 32:
                self.issue(product_mma, ds, transposed, query_acc, block > 0)
                self.issue(
                    product_mma,
                    probability,
                    transposed,
                    mean_acc,
                    False if cutlass.const_expr(self.stream_aux) else block > 0,
                )
                with cute.arch.elect_one():
                    tcgen05.commit(mbar)
            cute.arch.mbarrier_wait(mbar, phase)
            phase = phase ^ 1
            if cutlass.const_expr(self.stream_aux):
                values, _, _ = self.read(mean_acc, tid, self.width)
                for i in cutlass.range_constexpr(cute.size(auxiliary)):
                    auxiliary[i] += values[i]
            cute.arch.sync_threads()
        result, coords, _ = self.read(query_acc, tid, self.width)
        if cutlass.const_expr(self.stream_aux):
            mean = auxiliary
        else:
            mean, _, _ = self.read(mean_acc, tid, self.width)
        for i in cutlass.range_constexpr(cute.size(result)):
            row, col = coords[i]
            left[row, col] = ((result[i] - rounded_total * mean[i]) * scale).to(q.element_type)
        if start + self.row_index(tid) < q.shape[2] and wi == 0 and (self.rows == 128 or tid % 32 < 16):
            delta[batch, head, start + self.row_index(tid)] = total
        cute.arch.sync_threads()
        self.store(left, dq, batch, head, start, offset, tid)
        cute.arch.sync_threads()
        tmem.free(pointer)


class KeyValue(Forward):
    def __init__(self, *args, stream_aux=False, hardware_reductions=True, **kwargs):
        Attention.__init__(self, *args, **kwargs)
        self.stream_aux = stream_aux
        self.hardware_reductions = hardware_reductions
        if self.mode != "kv" or self.masked or self.hopper or self.tcgen or self.warps != 4 or self.pipelined:
            raise ValueError("Native key/value backward requires four warps, no explicit mask, and single-buffered loads.")
        self.configure_rows()

    @cute.kernel
    def kernel(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, delta_base, scale, unused, tma_args
    ):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, wi = self.grid_coordinates(q, k)
        batch, kvhead = bh // k.shape[1], bh % k.shape[1]
        start, offset = tile * self.rows, wi * self.width
        padded = cute.ceil_div(q.shape[3], self.width) * self.width
        allocator = SmemAllocator()
        mbar = allocator.allocate_array(cutlass.Int64, 1)
        load_bar = allocator.allocate_array(cutlass.Int64, 2)
        holding = allocator.allocate_array(cutlass.Int32, 1)
        atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128 if padded % 64 == 0 else warpgroup.SmemLayoutAtomKind.K_SW64,
            q.element_type,
        )
        layout = cute.tile_to_shape(atom, (self.rows, padded), (0, 1))
        left = self.shared(allocator, q.element_type, layout)
        other_left = self.shared(allocator, q.element_type, layout)
        stages = 2 if cutlass.const_expr(self.pipelined) else 1
        stage_layout = cute.tile_to_shape(atom, (self.columns, padded, stages), (0, 1, 2))
        right_stages = self.shared(allocator, q.element_type, stage_layout)
        other_stages = self.shared(allocator, q.element_type, stage_layout)
        weight_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128 if self.columns % 64 == 0 else warpgroup.SmemLayoutAtomKind.K_SW64,
            q.element_type,
        )
        weight_layout = cute.tile_to_shape(weight_atom, (self.rows, self.columns), (0, 1))
        ds = self.shared(allocator, q.element_type, weight_layout)
        probability = self.shared(allocator, q.element_type, weight_layout)
        columns = (
            self.width + max(self.width, 2 * self.columns)
            if cutlass.const_expr(self.stream_aux)
            else 2 * (self.columns + self.width)
        )
        tmem = TmemAllocator(
            alloc_result_dst_smem_ptr=holding,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=128),
        )
        tmem.allocate(128 if columns <= 128 else 256 if columns <= 256 else 512)
        tmem.wait_for_alloc()
        pointer = tmem.retrieve_ptr(cutlass.Float32)
        tmem.relinquish_alloc_permit()
        if tid == 0:
            cute.arch.mbarrier_init(mbar, 1)
            for slot in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(load_bar + slot, 1)
        cute.arch.mbarrier_init_fence()
        stats = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((4, self.columns), stride=(self.columns, 1)), byte_alignment=128
        )
        self.load(k, left, batch, kvhead, start, tid)
        self.load(v, other_left, batch, kvhead, start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        score_mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                cutlass.Float32,
                (self.rows, self.columns, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        product_mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                cutlass.Float32,
                (self.rows, self.width, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.MN,
            )
        )
        sf = score_mma.make_fragment_C(score_mma.get_slice(0).partition_shape_C((self.rows, self.columns)))
        pf = product_mma.make_fragment_C(product_mma.get_slice(0).partition_shape_C((self.rows, self.width)))
        score_acc = cute.make_tensor(pointer, sf.layout)
        dp_acc = cute.make_tensor(pointer + self.columns, sf.layout)
        if cutlass.const_expr(self.stream_aux):
            query_acc = cute.make_tensor(pointer + max(self.width, 2 * self.columns), pf.layout)
            mean_acc = cute.make_tensor(pointer, pf.layout)
            auxiliary = cute.make_rmem_tensor((self.width * self.rows // 128,), cutlass.Float32)
            auxiliary.fill(0.0)
        else:
            query_acc = cute.make_tensor(pointer + 2 * self.columns, pf.layout)
            mean_acc = cute.make_tensor(pointer + 2 * self.columns + self.width, pf.layout)
        magnitude = scale if scale >= 0 else -scale
        sign = cutlass.Float32(1) if scale >= 0 else cutlass.Float32(-1)
        count = cute.ceil_div(q.shape[2], self.columns)
        begin = cutlass.min(start // self.columns, count - 1) if cutlass.const_expr(self.causal) else 0
        phase, load_phase = cutlass.Int32(0), cutlass.Int32(0)
        for group in range(q.shape[1] // k.shape[1]):
            head = kvhead * (q.shape[1] // k.shape[1]) + group
            if cutlass.const_expr(self.pipelined):
                self.issue_pair(right_stages, other_stages, batch, kvhead, 0, 0, start, tid, tma_args, load_bar, blocks)
            for block in range(begin, count):
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
                        load_bar,
                        blocks,
                    )
                else:
                    right, other_right = right_stages[None, None, 0], other_stages[None, None, 0]
                    load_phase = self.load_pair(
                        q, dout, right, other_right, batch, head, block * self.columns, tid, tma_args, load_bar, load_phase
                    )
                cute.arch.fence_view_async_shared()
                if tid < 32:
                    self.issue(score_mma, left, right, score_acc)
                    self.issue(score_mma, other_left, other_right, dp_acc)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar)
                cute.arch.mbarrier_wait(mbar, phase)
                phase = phase ^ 1
                if tid < self.columns:
                    for field in cutlass.range_constexpr(4):
                        tensor = (maximum, logsum, delta_base, delta)[field]
                        value = cutlass.Float32(0)
                        if block * self.columns + tid < q.shape[2]:
                            value = tensor[batch, head, block * self.columns + tid]
                        stats[field, tid] = value
                cute.arch.sync_threads()
                scores, coords, _ = self.read(score_acc, tid, self.columns)
                dps, _, _ = self.read(dp_acc, tid, self.columns)
                for i in cutlass.range_constexpr(cute.size(scores)):
                    row, col = coords[i]
                    valid = start + row < k.shape[2] and block * self.columns + col < q.shape[2]
                    if cutlass.const_expr(self.causal):
                        valid = valid and block * self.columns + col >= start + row
                    p = cutlass.Float32(0)
                    if valid:
                        p = cute.math.exp2(
                            (scores[i] * sign - stats[0, col]) * magnitude * 1.4426950408889634 - stats[1, col],
                            fastmath=True,
                        )
                    value = p * ((dps[i] - stats[2, col]) - stats[3, col])
                    rounded = value.to(q.element_type)
                    scores[i] = value
                    dps[i] = rounded.to(cutlass.Float32)
                    ds[row, col] = rounded
                    probability[row, col] = p.to(q.element_type)
                cute.arch.sync_threads()
                cute.arch.fence_view_async_shared()
                source = cute.local_tile(right, (self.columns, self.width), (0, offset // self.width))
                transposed = cute.composition(source, cute.make_layout((self.width, self.columns), stride=(self.columns, 1)))
                if tid < 32:
                    self.issue(product_mma, ds, transposed, query_acc, block > begin or group > 0)
                    other_source = cute.local_tile(other_right, (self.columns, self.width), (0, offset // self.width))
                    other_transposed = cute.composition(
                        other_source, cute.make_layout((self.width, self.columns), stride=(self.columns, 1))
                    )
                    self.issue(
                        product_mma,
                        probability,
                        other_transposed,
                        mean_acc,
                        False if cutlass.const_expr(self.stream_aux) else block > begin or group > 0,
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar)
                cute.arch.mbarrier_wait(mbar, phase)
                phase = phase ^ 1
                if cutlass.const_expr(self.stream_aux):
                    values, _, _ = self.read(mean_acc, tid, self.width)
                    for i in cutlass.range_constexpr(cute.size(auxiliary)):
                        auxiliary[i] += values[i]
                cute.arch.sync_threads()
        result, coords, _ = self.read(query_acc, tid, self.width)
        if cutlass.const_expr(self.stream_aux):
            mean = auxiliary
        else:
            mean, _, _ = self.read(mean_acc, tid, self.width)
        for i in cutlass.range_constexpr(cute.size(result)):
            row, col = coords[i]
            left[row, col] = (result[i] * scale).to(q.element_type)
            other_left[row, col] = mean[i].to(q.element_type)
        cute.arch.sync_threads()
        self.store(left, dk, batch, kvhead, start, offset, tid)
        self.store(other_left, dv, batch, kvhead, start, offset, tid)
        cute.arch.sync_threads()
        tmem.free(pointer)
