"""Blackwell producer/consumer attention, adapted from KohakuFA's TMEM stream.

Architectural reference: KohakuFA sm100/{fwd,bwd}.py at
0c5ede6859307bedc0c6089523c16b1ca51b7da9 (Apache-2.0; see NOTICE).
Separate TMA, MMA, softmax/P, and correction/dS workers use mbarriers. Each CTA
has three complete warp groups: two four-warp consumers, one MMA warp, one
loader warp, and two padding warps. Streamed KV backward uses one padding warp
to prefetch row statistics independently of the operand loader. Completion
barriers release a shared slot only after its asynchronous tensor-core reads finish.

D512 forward uses 128-wide transfers and reuses resident Q for final
output staging, after the final product has consumed Q.
Forward packs probabilities over consumed score slots in TMEM. Backward keeps
FP32 P for centered dS: the 64-column path stores P in TMEM, while the fused
128-column path keeps each consumer's 64 probabilities in registers and aliases
packed P over S. D64/D128 backward holds stationary operands in shared memory.
D128 uses 128-wide resident operands at 128 columns. Wide backward heads stream
64-wide A/B pairs; the 128-column path keeps rounded dS in consumed dP TMEM
and uses six transfer slots. Each gradient tile has a single owner.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05, warpgroup
from cutlass.memory import SmemAllocator, TmemAllocator

from .blackwell import fence_after_sync, fence_before_sync
from .kernels import Attention, pack_half_pair
from .launch import block_indices, launch_grid
from .sm103 import Forward as NativeForward


class Forward(NativeForward):
    def __init__(self, *args, hardware_reductions=True, head_dimension=None, **kwargs):
        Attention.__init__(self, *args, **kwargs)
        if self.mode != "forward" or self.masked or self.warps != 12 or self.columns != 128:
            raise ValueError("The warp-specialized forward requires 128 columns and twelve warps.")
        self.hardware_reductions = hardware_reductions
        self.rows = self.grid_rows = 128
        # Wide output slices amortize the larger transfer boxes.
        self.dc = 128 if head_dimension == 512 and self.width == 256 else 64
        self.slots, self.score_slots = (3 if self.dc == 128 else 4), 2

    @cute.jit
    def __call__(self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, delta_base, scale, stream):
        atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, q.element_type)
        layout = cute.tile_to_shape(atom, (128, self.dc), (0, 1))
        qa, qt = self.create_tma(q, layout, self.dc)
        ka, kt = self.create_tma(k, layout, self.dc)
        va, vt = self.create_tma(v, layout, self.dc)
        self.kernel(q, k, v, out, maximum, logsum, scale, (qa, qt, ka, kt, va, vt)).launch(
            grid=launch_grid(cute.ceil_div(q.shape[2], 128), q.shape[0] * q.shape[1], cute.ceil_div(q.shape[3], self.width)),
            block=(384, 1, 1),
            stream=stream,
        )

    @cute.jit
    def box(self, atom, tensor, target, batch, head, row, chunk, barrier):
        tile = cute.local_tile(tensor, (128, self.dc), (row // 128, chunk, (head, batch)))
        dst, src = cpasync.tma_partition(
            atom, 0, cute.make_layout(1), cute.group_modes(target, 0, 2), cute.group_modes(tile, 0, 2)
        )
        cute.copy(atom, src, dst, tma_bar_ptr=barrier)

    @cute.jit
    def enqueue(self, atom, tensor, ring, ready, free, count, batch, head, row, chunk):
        slot = count % self.slots
        cute.arch.mbarrier_wait(free + slot, ((count // self.slots) % 2) ^ 1)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(ready + slot, 128 * self.dc * 2)
        self.box(atom, tensor, ring[None, None, slot], batch, head, row, chunk, ready + slot)

    @cute.jit
    def matrix(self, pointer, columns: cutlass.Constexpr, dtype: cutlass.Constexpr = cutlass.Float32):
        return cute.make_tensor(cute.recast_ptr(pointer, dtype=dtype), cute.make_layout((128, columns), stride=(65536, 1)))

    @cute.jit
    def read_matrix(self, pointer, columns: cutlass.Constexpr, tid):
        fence_after_sync()
        view = self.matrix(pointer, columns)
        atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(columns)), cutlass.Float32)
        copy = tcgen05.make_tmem_copy(atom, view)
        part = copy.get_slice(tid)
        coords = part.partition_D(cute.make_identity_tensor((128, columns)))
        values = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
        cute.copy(copy, part.partition_S(view), values)
        cute.arch.fence_view_async_tmem_load()
        fence_before_sync()
        return values

    @cute.jit
    def write_matrix(self, pointer, values, columns: cutlass.Constexpr, tid):
        fence_after_sync()
        view = self.matrix(pointer, columns, values.element_type)
        atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(columns)), values.element_type)
        copy = tcgen05.make_tmem_copy(atom, view)
        part = copy.get_slice(tid)
        coords = part.partition_S(cute.make_identity_tensor((128, columns)))
        source = cute.make_tensor(values.iterator, cute.make_layout(coords.shape))
        cute.copy(copy, source, part.partition_D(view))
        cute.arch.fence_view_async_tmem_store()
        fence_before_sync()

    @cute.jit
    def issue_pv(self, dtype: cutlass.Constexpr, probability_pointer, v_tile, output_pointer, accumulate):
        fence_after_sync()
        mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                dtype,
                cutlass.Float32,
                (128, self.dc, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.TMEM,
                OperandMajorMode.K,
                OperandMajorMode.MN,
            )
        )
        mma.set(tcgen05.Field.ACCUMULATE, accumulate)
        part = mma.get_slice(0)
        af = mma.make_fragment_A(part.partition_shape_A((128, self.columns)))
        a = cute.make_tensor(cute.recast_ptr(probability_pointer, dtype=dtype), af.layout)
        transposed = cute.composition(v_tile, cute.make_layout((self.dc, self.columns), stride=(self.columns, 1)))
        b = mma.make_fragment_B(part.partition_B(transposed))
        cf = mma.make_fragment_C(part.partition_shape_C((128, self.dc)))
        accum = cute.make_tensor(output_pointer, cf.layout)
        for chunk in cutlass.range(cute.size(a.shape[2]), unroll_full=True):
            cute.gemm(mma, accum, a[None, None, chunk], b[None, None, chunk], accum)
            mma.set(tcgen05.Field.ACCUMULATE, True)

    @cute.jit
    def shifted_scaled_pair(self, values, maxima, magnitude, sign):
        # Keep the scalar rounding order while using Blackwell's paired FP32 ALU.
        values = cute.arch.mul_packed_f32x2(values, (sign, sign))
        values = cute.arch.sub_packed_f32x2(values, maxima)
        values = cute.arch.mul_packed_f32x2(values, (magnitude, magnitude))
        log2e = cutlass.Float32(1.4426950408889634)
        return cute.arch.mul_packed_f32x2(values, (log2e, log2e))

    @cute.jit
    def softmax_step(
        self,
        score,
        pointer,
        block,
        start,
        sq,
        sk,
        scale,
        local,
        row_max,
        row_sum,
        alphas,
        alpha_ready,
        alpha_free,
        p_ready,
        dtype: cutlass.Constexpr,
        masked: cutlass.Constexpr,
    ):
        magnitude = scale if scale >= 0 else -scale
        sign = cutlass.Float32(1) if scale >= 0 else cutlass.Float32(-1)
        if cutlass.const_expr(masked):
            scores, _, _ = self.read(score, local, 128)
            for i in cutlass.range_constexpr(128):
                valid = start + local < sq and block * 128 + i < sk
                if cutlass.const_expr(self.causal):
                    valid = valid and block * 128 + i <= start + local
                scores[i] = scores[i] * sign if valid else -cutlass.Float32.inf
            current = scores.load().reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
        else:
            scores, _, current = self.read_scores(score, local, scale)
        new = cute.arch.fmax(row_max, current)
        alpha = cutlass.Float32(0)
        if row_max != -cutlass.Float32.inf:
            alpha = cute.math.exp2((row_max - new) * magnitude * 1.4426950408889634, fastmath=True)
        if block > 0:
            buffer = (block - 1) % 2
            cute.arch.mbarrier_wait(alpha_free + buffer, (((block - 1) // 2) % 2) ^ 1)
            alphas[buffer, local] = alpha
            cute.arch.barrier(barrier_id=2, number_of_threads=128)
            if local == 0:
                cute.arch.mbarrier_arrive(alpha_ready + buffer)
        packed = cute.make_rmem_tensor((64,), cutlass.Uint32)
        sums = cute.make_rmem_tensor((32,), cutlass.Float32)
        sums.fill(0.0)
        for i in cutlass.range_constexpr(64):
            s0, s1 = scores[2 * i], scores[2 * i + 1]
            if cutlass.const_expr(not masked):
                if cutlass.const_expr(self.width >= 256):
                    pair = self.shifted_scaled_pair((s0, s1), (new, new), magnitude, sign)
                    p0, p1 = cute.math.exp2(pair[0], fastmath=True), cute.math.exp2(pair[1], fastmath=True)
                else:
                    s0, s1 = s0 * sign, s1 * sign
                    p0 = cute.math.exp2((s0 - new) * magnitude * 1.4426950408889634, fastmath=True)
                    p1 = cute.math.exp2((s1 - new) * magnitude * 1.4426950408889634, fastmath=True)
            else:
                p0, p1 = cutlass.Float32(0), cutlass.Float32(0)
                if s0 != -cutlass.Float32.inf:
                    p0 = cute.math.exp2((s0 - new) * magnitude * 1.4426950408889634, fastmath=True)
                if s1 != -cutlass.Float32.inf:
                    p1 = cute.math.exp2((s1 - new) * magnitude * 1.4426950408889634, fastmath=True)
            packed[i] = pack_half_pair(p0, p1, dtype)
            sums[2 * (i % 16)] += p0
            sums[2 * (i % 16) + 1] += p1
        self.write_matrix(pointer + cute.assume(block % 2 * 128, divby=16), packed, 64, local)
        cute.arch.barrier(barrier_id=2, number_of_threads=128)
        if local == 0:
            cute.arch.mbarrier_arrive(p_ready + block % 2)
        return new, row_sum * alpha + sums.load().reduce(cute.ReductionOp.ADD, 0.0, 0)

    @cute.jit
    def store_output(self, staging, out, batch, head, start, offset, tid):
        columns = cute.size(staging.shape[1])
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out.element_type, num_bits_per_copy=128)
        copy = cute.make_tiled_copy_tv(atom, cute.make_layout((32, 4), stride=(4, 1)), cute.make_layout((1, 8)))
        part = copy.get_slice(tid)
        target = cute.local_tile(out[batch, head, None, None], (128, columns), (start // 128, offset // columns))
        ids = cute.local_tile(
            cute.make_identity_tensor((out.shape[2], out.shape[3])), (128, columns), (start // 128, offset // columns)
        )
        src, dst, coords = part.partition_S(staging), part.partition_D(target), part.partition_D(ids)
        for r in cutlass.range_constexpr(cute.size(dst.shape[1])):
            for d in cutlass.range_constexpr(cute.size(dst.shape[2])):
                row, col = coords[0, r, d]
                if row < out.shape[2] and col < out.shape[3]:
                    cute.copy(copy, src[None, r, d], dst[None, r, d])

    @cute.kernel
    def kernel(self, q, k, v, out, maximum, logsum, scale, args):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, wi = block_indices(cute.ceil_div(q.shape[2], 128), q.shape[0] * q.shape[1])
        batch, head = bh // q.shape[1], bh % q.shape[1]
        kvhead = head // (q.shape[1] // k.shape[1])
        start, offset = tile * 128, wi * self.width
        nch = q.shape[3] // self.dc
        nv = self.width // self.dc
        limit = cutlass.min(k.shape[2], start + 128) if cutlass.const_expr(self.causal) else k.shape[2]
        count = cute.ceil_div(limit, 128)
        allocator = SmemAllocator()
        bars = allocator.allocate_array(cutlass.Int64, 20)
        q_ready, kv_ready, kv_free = bars, bars + 1, bars + 5
        s_ready, p_ready = bars + 9, bars + 11
        alpha_ready, alpha_free = bars + 13, bars + 15
        o_ready, o_free, stats_ready = bars + 17, bars + 18, bars + 19
        holding = allocator.allocate_array(cutlass.Int32, 1)
        atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, q.element_type)
        qbuf = self.shared(allocator, q.element_type, cute.tile_to_shape(atom, (128, self.dc, nch), (0, 1, 2)))
        ring = self.shared(allocator, q.element_type, cute.tile_to_shape(atom, (128, self.dc, self.slots), (0, 1, 2)))
        if cutlass.const_expr(self.dc == 128):
            # The final o_ready also follows every QK read of the resident Q.
            # Reuse it for stores to fit three 128-wide transfer slots.
            staging = cute.local_tile(qbuf[None, None, 0], (128, 64), (0, 0))
        else:
            staging = self.shared(allocator, q.element_type, cute.tile_to_shape(atom, (128, self.dc), (0, 1)))
        alphas = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((2, 128), stride=(128, 1)), byte_alignment=128)
        stats = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((2, 128), stride=(128, 1)), byte_alignment=128)
        tmem = TmemAllocator(
            alloc_result_dst_smem_ptr=holding, barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=384)
        )
        tmem.allocate(512)
        tmem.wait_for_alloc()
        pointer = tmem.retrieve_ptr(cutlass.Float32)
        tmem.relinquish_alloc_permit()
        output_pointer = pointer + 256
        if tid == 0:
            for i in cutlass.range_constexpr(20):
                cute.arch.mbarrier_init(bars + i, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        qa, qt, ka, kt, va, vt = args
        score_mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                cutlass.Float32,
                (128, 128, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        sf = score_mma.make_fragment_C(score_mma.get_slice(0).partition_shape_C((128, 128)))

        if tid >= 288 and tid < 320:
            # Loader: the same operand order as the MMA cursor, two scores ahead.
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(q_ready, 128 * q.shape[3] * 2)
            for c in cutlass.range(nch, unroll_full=True):
                self.box(qa, qt, qbuf[None, None, c], batch, head, start, c, q_ready)
            cursor = cutlass.Int32(0)
            for block in range(cutlass.min(count, 2)):
                for c in cutlass.range(nch, unroll_full=True):
                    self.enqueue(ka, kt, ring, kv_ready, kv_free, cursor, batch, kvhead, block * 128, c)
                    cursor += 1
            for block in range(count):
                for c in cutlass.range(nv, unroll_full=True):
                    self.enqueue(va, vt, ring, kv_ready, kv_free, cursor, batch, kvhead, block * 128, offset // self.dc + c)
                    cursor += 1
                if block + 2 < count:
                    for c in cutlass.range(nch, unroll_full=True):
                        self.enqueue(ka, kt, ring, kv_ready, kv_free, cursor, batch, kvhead, (block + 2) * 128, c)
                        cursor += 1
        elif tid >= 256 and tid < 288:
            # MMA: fill score slots, then alternate PV(j) and QK(j+2).
            cute.arch.mbarrier_wait(q_ready, 0)
            cursor = cutlass.Int32(0)
            for block in range(cutlass.min(count, 2)):
                score = cute.make_tensor(pointer + cute.assume(block % 2 * 128, divby=16), sf.layout)
                for c in cutlass.range(nch, unroll_full=True):
                    slot = cursor % self.slots
                    cute.arch.mbarrier_wait(kv_ready + slot, (cursor // self.slots) % 2)
                    cute.arch.fence_view_async_shared()
                    self.issue(score_mma, qbuf[None, None, c], ring[None, None, slot], score, c > 0)
                    with cute.arch.elect_one():
                        tcgen05.commit(kv_free + slot)
                    cursor += 1
                with cute.arch.elect_one():
                    tcgen05.commit(s_ready + block % 2)
            for block in range(count):
                cute.arch.mbarrier_wait(p_ready + block % 2, (block // 2) % 2)
                if block > 0:
                    cute.arch.mbarrier_wait(o_free, (block - 1) % 2)
                for c in cutlass.range(nv, unroll_full=True):
                    slot = cursor % self.slots
                    cute.arch.mbarrier_wait(kv_ready + slot, (cursor // self.slots) % 2)
                    cute.arch.fence_view_async_shared()
                    self.issue_pv(
                        q.element_type,
                        pointer + cute.assume(block % 2 * 128, divby=16),
                        ring[None, None, slot],
                        output_pointer + c * self.dc,
                        block > 0,
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(kv_free + slot)
                    cursor += 1
                with cute.arch.elect_one():
                    tcgen05.commit(o_ready)
                if block + 2 < count:
                    following = block + 2
                    score = cute.make_tensor(pointer + cute.assume(following % 2 * 128, divby=16), sf.layout)
                    for c in cutlass.range(nch, unroll_full=True):
                        slot = cursor % self.slots
                        cute.arch.mbarrier_wait(kv_ready + slot, (cursor // self.slots) % 2)
                        cute.arch.fence_view_async_shared()
                        self.issue(score_mma, qbuf[None, None, c], ring[None, None, slot], score, c > 0)
                        with cute.arch.elect_one():
                            tcgen05.commit(kv_free + slot)
                        cursor += 1
                    with cute.arch.elect_one():
                        tcgen05.commit(s_ready + following % 2)
        elif tid >= 128 and tid < 256:
            # Softmax owns one query row per thread and writes packed P over S.
            local = tid - 128
            row_max, row_sum = -cutlass.Float32.inf, cutlass.Float32(0)
            for block in range(count):
                slot = block % 2
                cute.arch.mbarrier_wait(s_ready + slot, (block // 2) % 2)
                score = cute.make_tensor(pointer + cute.assume(slot * 128, divby=16), sf.layout)
                full = start + 128 <= q.shape[2] and block * 128 + 128 <= k.shape[2]
                if cutlass.const_expr(self.causal):
                    full = full and block * 128 + 128 <= start + 1
                if full:
                    row_max, row_sum = self.softmax_step(
                        score,
                        pointer,
                        block,
                        start,
                        q.shape[2],
                        k.shape[2],
                        scale,
                        local,
                        row_max,
                        row_sum,
                        alphas,
                        alpha_ready,
                        alpha_free,
                        p_ready,
                        q.element_type,
                        False,
                    )
                else:
                    row_max, row_sum = self.softmax_step(
                        score,
                        pointer,
                        block,
                        start,
                        q.shape[2],
                        k.shape[2],
                        scale,
                        local,
                        row_max,
                        row_sum,
                        alphas,
                        alpha_ready,
                        alpha_free,
                        p_ready,
                        q.element_type,
                        True,
                    )
            stats[0, local], stats[1, local] = row_max, row_sum
            cute.arch.barrier(barrier_id=2, number_of_threads=128)
            if local == 0:
                cute.arch.mbarrier_arrive(stats_ready)
        elif tid < 128:
            # Keep 64-column register windows even with 128-wide TMA/PV boxes.
            # Wider correction loads spill at this CTA's register limit.
            for block in range(1, count):
                buffer = (block - 1) % 2
                cute.arch.mbarrier_wait(alpha_ready + buffer, ((block - 1) // 2) % 2)
                alpha = alphas[buffer, tid]
                cute.arch.mbarrier_wait(o_ready, (block - 1) % 2)
                for c in cutlass.range(self.width // 64, unroll_full=True):
                    values = self.read_matrix(output_pointer + c * 64, 64, tid)
                    for i in cutlass.range_constexpr(32):
                        pair = cute.arch.mul_packed_f32x2((values[2 * i], values[2 * i + 1]), (alpha, alpha))
                        values[2 * i], values[2 * i + 1] = pair
                    self.write_matrix(output_pointer + c * 64, values, 64, tid)
                cute.arch.barrier(barrier_id=3, number_of_threads=128)
                if tid == 0:
                    cute.arch.mbarrier_arrive(alpha_free + buffer)
                    cute.arch.mbarrier_arrive(o_free)
            cute.arch.mbarrier_wait(stats_ready, 0)
            denominator = stats[1, tid]
            denominator = denominator if denominator > 0 else cutlass.Float32(1)
            inverse = cutlass.Float32(1) / denominator
            if count > 0:
                cute.arch.mbarrier_wait(o_ready, (count - 1) % 2)
            for c in cutlass.range(self.width // 64, unroll_full=True):
                values = self.read_matrix(output_pointer + c * 64, 64, tid)
                for i in cutlass.range_constexpr(32):
                    pair = cute.arch.mul_packed_f32x2((values[2 * i], values[2 * i + 1]), (inverse, inverse))
                    staging[tid, 2 * i], staging[tid, 2 * i + 1] = pair[0].to(q.element_type), pair[1].to(q.element_type)
                cute.arch.barrier(barrier_id=3, number_of_threads=128)
                self.store_output(staging, out, batch, head, start, offset + c * 64, tid)
                cute.arch.barrier(barrier_id=3, number_of_threads=128)
            if start + tid < q.shape[2] and wi == 0:
                maximum[batch, head, start + tid] = stats[0, tid]
                logsum[batch, head, start + tid] = cute.math.log2(denominator, fastmath=True)
        cute.arch.sync_threads()
        tmem.free(pointer)


class Backward(Forward):
    """Centered backward with separate or fused probability/dS consumers.

    At 64 columns the workers exchange FP32 P through TMEM. At 128 columns each
    worker owns 64 columns, computes P and dS, and shares the packed P operand.
    Two 32-column dP loads limit its live registers; both paths retain FP32
    probabilities for dS and use centered correction.
    """

    def __init__(self, *args, hardware_reductions=True, stream_aux=False, operand_chunk=64, **kwargs):
        Attention.__init__(self, *args, **kwargs)
        if self.mode not in ("q", "kv") or self.masked or self.warps != 12 or self.columns not in (64, 128):
            raise ValueError("Warp-specialized backward requires 64/128 columns and twelve warps.")
        if self.columns == 128 and self.width not in (64, 128):
            raise ValueError("128-column backward currently requires a 64/128-wide output slice.")
        self.hardware_reductions = hardware_reductions
        self.rows = self.grid_rows = 128
        self.dc, self.slots = min(self.width, operand_chunk) if self.columns == 128 else 64, 2
        self.tmem_ds = self.columns == 128 and self.width == 128
        self.stream_slots = (
            (3 if self.dc == 128 else 6) if self.tmem_ds else 2 if self.dc == 128 else 5 if self.columns == 128 else 8
        )
        self.other_slots = 1 if self.columns == 128 and self.width == 128 else 2
        self.ds_slots = 1 if self.columns == 128 and self.width == 128 else 2

    @cute.jit
    def descriptor(self, source, rows: cutlass.Constexpr):
        atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, source.element_type)
        layout = cute.tile_to_shape(atom, (rows, self.dc), (0, 1))
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
        return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), tensor, layout, (rows, self.dc))

    @cute.jit
    def box_rows(self, atom, tensor, target, batch, head, row, chunk, barrier, rows: cutlass.Constexpr):
        tile = cute.local_tile(tensor, (rows, self.dc), (row // rows, chunk, (head, batch)))
        dst, src = cpasync.tma_partition(
            atom, 0, cute.make_layout(1), cute.group_modes(target, 0, 2), cute.group_modes(tile, 0, 2)
        )
        cute.copy(atom, src, dst, tma_bar_ptr=barrier)

    @cute.jit
    def __call__(self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, delta_base, scale, stream):
        left, other_left, right, other_right = (q, dout, k, v) if cutlass.const_expr(self.mode == "q") else (k, v, q, dout)
        la, lt = self.descriptor(left, 128)
        oa, ot = self.descriptor(other_left, 128)
        ra, rt = self.descriptor(right, self.columns)
        da, dt = self.descriptor(other_right, self.columns)
        length, heads = (q.shape[2], q.shape[1]) if cutlass.const_expr(self.mode == "q") else (k.shape[2], k.shape[1])
        self.kernel(
            q,
            k,
            v,
            maximum,
            logsum,
            dout,
            delta,
            delta_base,
            dq,
            dk,
            dv,
            scale,
            (la, lt, oa, ot, ra, rt, da, dt),
            q.shape[3] > 128,
        ).launch(
            grid=launch_grid(
                cute.ceil_div(length, 128)
                * (cute.ceil_div(q.shape[3], self.width) if cutlass.const_expr(q.shape[3] == 512) else 1),
                q.shape[0] * heads,
                1 if cutlass.const_expr(q.shape[3] == 512) else cute.ceil_div(q.shape[3], self.width),
            ),
            block=(384, 1, 1),
            stream=stream,
        )

    @cute.jit
    def stream_pair(
        self, aa, at, ba, bt, a_ring, b_ring, ready, free, cursor, batch, left_head, left_row, right_head, right_row, chunk
    ):
        slot = cursor % self.stream_slots
        cute.arch.mbarrier_wait(free + slot, ((cursor // self.stream_slots) % 2) ^ 1)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(ready + slot, 2 * (128 + self.columns) * self.dc)
        self.box_rows(aa, at, a_ring[None, None, slot], batch, left_head, left_row, chunk, ready + slot, 128)
        self.box_rows(ba, bt, b_ring[None, None, slot], batch, right_head, right_row, chunk, ready + slot, self.columns)

    @cute.jit
    def stream_right(self, atom, tensor, ring, ready, free, cursor, batch, head, row, chunk):
        slot = cursor % self.stream_slots
        cute.arch.mbarrier_wait(free + slot, ((cursor // self.stream_slots) % 2) ^ 1)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(ready + slot, 2 * self.columns * self.dc)
        self.box_rows(atom, tensor, ring[None, None, slot], batch, head, row, chunk, ready + slot, self.columns)

    @cute.jit
    def stream_stats(self, stats, maximum, logsum, delta_base, delta, free, ready, batch, head, block, step, tid):
        slot = step % 2
        cute.arch.mbarrier_wait(free + slot, ((step // 2) % 2) ^ 1)
        for i in cutlass.range_constexpr(self.columns // 32):
            col = tid % 32 + i * 32
            for field in cutlass.range_constexpr(4):
                source = (maximum, logsum, delta_base, delta)[field]
                value = cutlass.Float32(0)
                if block * self.columns + col < maximum.shape[2]:
                    value = source[batch, head, block * self.columns + col]
                stats[field, col, slot] = value
        cute.arch.sync_warp()
        # Score completion does not imply that this independent producer is
        # done. Publish the shared statistics before pointwise workers read them.
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(ready + slot)

    @cute.jit
    def fused_sync(self, half):
        if half == 0:
            cute.arch.barrier(barrier_id=2, number_of_threads=128)
        else:
            cute.arch.barrier(barrier_id=3, number_of_threads=128)

    @cute.jit
    def store_ds(self, values, target, tid):
        columns = cute.size(values.shape)
        template = self.matrix(cute.make_ptr(cutlass.Float32, 0, cute.AddressSpace.tmem), columns)
        tmem_copy = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(columns)), cutlass.Float32), template
        )
        copy = cute.make_tiled_copy_D(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), target.element_type, num_bits_per_copy=128), tmem_copy
        )
        part = copy.get_slice(tid)
        converted = cute.make_rmem_tensor(values.shape, target.element_type)
        converted.store(values.load().to(target.element_type))
        coords = tmem_copy.get_slice(tid).partition_D(cute.make_identity_tensor((128, columns)))
        source = cute.make_tensor(converted.iterator, cute.make_layout(coords.shape))
        cute.copy(copy, part.retile(source), part.partition_D(target))

    @cute.jit
    def reconstruct_probabilities(
        self, values, packed, stats, slot, half, location, normalization, dtype: cutlass.Constexpr
    ):
        # Interior tiles reconstruct P without per-element bounds predicates.
        # Tails and causal boundaries keep the same arithmetic and visibility.
        local, start, block, sq, sk = location
        if cutlass.const_expr(self.mode == "q"):
            full = start + 128 <= sq and (block + 1) * self.columns <= sk
            if cutlass.const_expr(self.causal):
                full = full and (block + 1) * self.columns <= start + 1
        else:
            full = start + 128 <= sk and (block + 1) * self.columns <= sq
            if cutlass.const_expr(self.causal):
                full = full and block * self.columns >= start + 127
        if full:
            self.reconstruct_values(values, packed, stats, slot, half, location, normalization, dtype, False)
        else:
            self.reconstruct_values(values, packed, stats, slot, half, location, normalization, dtype, True)

    @cute.jit
    def reconstruct_values(
        self,
        values,
        packed,
        stats,
        slot,
        half,
        location,
        normalization,
        dtype: cutlass.Constexpr,
        masked: cutlass.Constexpr = True,
    ):
        local, start, block, sq, sk = location
        row_max, row_logsum, magnitude, sign = normalization
        if cutlass.const_expr(not masked):
            for i in cutlass.range_constexpr(32):
                col = half * 64 + 2 * i
                if cutlass.const_expr(self.mode == "q"):
                    maxima, logs = (row_max, row_max), (row_logsum, row_logsum)
                else:
                    maxima = (stats[0, col, slot], stats[0, col + 1, slot])
                    logs = (stats[1, col, slot], stats[1, col + 1, slot])
                pair = self.shifted_scaled_pair((values[2 * i], values[2 * i + 1]), maxima, magnitude, sign)
                pair = cute.arch.sub_packed_f32x2(pair, logs)
                p0, p1 = cute.math.exp2(pair[0], fastmath=True), cute.math.exp2(pair[1], fastmath=True)
                values[2 * i], values[2 * i + 1] = p0, p1
                if cutlass.const_expr(self.columns != 128):
                    packed[i] = pack_half_pair(p0, p1, dtype)
        else:
            for i in cutlass.range_constexpr(32):
                pair = cute.make_rmem_tensor((2,), cutlass.Float32)
                for j in cutlass.range_constexpr(2):
                    index, col = 2 * i + j, half * 64 + 2 * i + j
                    if cutlass.const_expr(self.mode == "q"):
                        row_maximum, row_log_normalizer = row_max, row_logsum
                    else:
                        row_maximum, row_log_normalizer = stats[0, col, slot], stats[1, col, slot]
                    probability = cutlass.Float32(0)
                    if cutlass.const_expr(self.mode == "q"):
                        valid = start + local < sq and block * self.columns + col < sk
                        if cutlass.const_expr(self.causal):
                            valid = valid and block * self.columns + col <= start + local
                    else:
                        valid = start + local < sk and block * self.columns + col < sq
                        if cutlass.const_expr(self.causal):
                            valid = valid and block * self.columns + col >= start + local
                    if valid:
                        probability = cute.math.exp2(
                            (values[index] * sign - row_maximum) * magnitude * 1.4426950408889634 - row_log_normalizer,
                            fastmath=True,
                        )
                    values[index], pair[j] = probability, probability
                if cutlass.const_expr(self.columns != 128):
                    packed[i] = pack_half_pair(pair[0], pair[1], dtype)

    @cute.kernel
    def kernel(
        self, q, k, v, maximum, logsum, dout, delta, delta_base, dq, dk, dv, scale, args, streamed: cutlass.Constexpr
    ):
        tid, _, _ = cute.arch.thread_idx()
        length, heads = (q.shape[2], q.shape[1]) if cutlass.const_expr(self.mode == "q") else (k.shape[2], k.shape[1])
        tiles = cute.ceil_div(length, 128)
        if cutlass.const_expr(q.shape[3] == 512):
            tiles *= cute.ceil_div(q.shape[3], self.width)
        tile, bh, wi = block_indices(tiles, q.shape[0] * heads)
        if cutlass.const_expr(q.shape[3] == 512):
            # Adjacent output slices reuse the inputs of one query/key row tile.
            slices = cute.ceil_div(q.shape[3], self.width)
            tile, wi = tile // slices, tile % slices
        batch, own_head = bh // heads, bh % heads
        if cutlass.const_expr(self.causal and self.mode == "q" and q.shape[3] == 512):
            # Longest causal query tiles first, so the final wave has less work.
            tile = cute.ceil_div(q.shape[2], 128) - 1 - tile
        start, offset = tile * 128, wi * self.width
        nch, nv = q.shape[3] // self.dc, self.width // self.dc
        ratio = q.shape[1] // k.shape[1]
        count = cute.ceil_div(k.shape[2] if cutlass.const_expr(self.mode == "q") else q.shape[2], self.columns)
        begin = cutlass.Int32(0)
        groups = 1 if cutlass.const_expr(self.mode == "q") else ratio
        if cutlass.const_expr(self.causal):
            if cutlass.const_expr(self.mode == "q"):
                count = cute.ceil_div(cutlass.min(k.shape[2], start + 128), self.columns)
            else:
                begin = cutlass.min(start // self.columns, count - 1)
        span = count - begin
        steps = span * groups
        allocator = SmemAllocator()
        bars = allocator.allocate_array(cutlass.Int64, 20)
        left_ready, right_ready, other_ready = bars, bars + 1, bars + 3
        right_free, other_free = bars + 5, bars + 7
        s_ready, dp_ready, p_ready = bars + 9, bars + 10, bars + 11
        s_free, dp_free, p_free = bars + 12, bars + 13, bars + 14
        ds_ready, ds_free, output_ready = bars + 15, bars + 17, bars + 19
        if cutlass.const_expr(streamed):
            right_ready = allocator.allocate_array(cutlass.Int64, self.stream_slots)
            right_free = allocator.allocate_array(cutlass.Int64, self.stream_slots)
        holding = allocator.allocate_array(cutlass.Int32, 1)
        atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, q.element_type)
        if cutlass.const_expr(streamed):
            a_ring = self.shared(
                allocator, q.element_type, cute.tile_to_shape(atom, (128, self.dc, self.stream_slots), (0, 1, 2))
            )
            b_ring = self.shared(
                allocator, q.element_type, cute.tile_to_shape(atom, (self.columns, self.dc, self.stream_slots), (0, 1, 2))
            )
        else:
            resident_layout = cute.tile_to_shape(atom, (128, self.dc, nch), (0, 1, 2))
            left = self.shared(allocator, q.element_type, resident_layout)
            other_left = self.shared(allocator, q.element_type, resident_layout)
            ring_layout = cute.tile_to_shape(atom, (self.columns, self.dc, nch, 2), (0, 1, 2, 3))
            right = self.shared(allocator, q.element_type, ring_layout)
            other_layout = cute.tile_to_shape(atom, (self.columns, self.dc, nch, self.other_slots), (0, 1, 2, 3))
            other_right = self.shared(allocator, q.element_type, other_layout)
        weight_atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, q.element_type)
        if cutlass.const_expr(streamed and self.tmem_ds):
            # Rounded dS reuses the consumed dP allocation in TMEM.
            ds = None
        else:
            ds = self.shared(
                allocator, q.element_type, cute.tile_to_shape(weight_atom, (128, self.columns, self.ds_slots), (0, 1, 2))
            )
        if cutlass.const_expr(streamed):
            # All stream operands have been consumed before the epilogue starts.
            staging = a_ring[None, None, 0]
        elif cutlass.const_expr(self.columns == 128 and self.width == 128):
            staging = left[None, None, 0]
        else:
            staging = self.shared(allocator, q.element_type, cute.tile_to_shape(atom, (128, self.dc), (0, 1)))
        epilogue_staging = cute.local_tile(staging, (128, 64), (0, 0))
        stats = allocator.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((4, self.columns, 2), stride=(self.columns, 1, 4 * self.columns)),
            byte_alignment=128,
        )
        tmem = TmemAllocator(
            alloc_result_dst_smem_ptr=holding, barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=384)
        )
        tmem.allocate(512)
        tmem.wait_for_alloc()
        pointer = tmem.retrieve_ptr(cutlass.Float32)
        tmem.relinquish_alloc_permit()
        # Fused workers consume S into registers before packed P aliases it.
        # This leaves columns 256..511 for the two wide gradient accumulators.
        arena = 2 * self.columns if self.columns == 128 else 5 * self.columns // 2
        grad_pointer, aux_pointer = pointer + arena, pointer + arena + self.width
        probability_pointer = pointer if self.columns == 128 else pointer + 2 * self.columns
        if tid == 0:
            for i in cutlass.range_constexpr(20):
                cute.arch.mbarrier_init(bars + i, 1)
            if cutlass.const_expr(self.columns == 128):
                cute.arch.mbarrier_init(p_ready, 2)
                cute.arch.mbarrier_init(s_free, 2)
                if cutlass.const_expr(not (streamed and self.tmem_ds)):
                    cute.arch.mbarrier_init(dp_free, 2)
                cute.arch.mbarrier_init(ds_ready, 2)
                cute.arch.mbarrier_init(ds_ready + 1, 2)
                if cutlass.const_expr(streamed and self.mode == "kv"):
                    cute.arch.mbarrier_init(other_free, 2)
                    cute.arch.mbarrier_init(other_free + 1, 2)
            if cutlass.const_expr(streamed):
                for i in cutlass.range_constexpr(self.stream_slots):
                    cute.arch.mbarrier_init(right_ready + i, 1)
                    cute.arch.mbarrier_init(right_free + i, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        la, lt, oa, ot, ra, rt, da, dt = args
        score_mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                cutlass.Float32,
                (128, self.columns, 16),
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
                (128, self.dc, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.MN,
            )
        )
        sf = score_mma.make_fragment_C(score_mma.get_slice(0).partition_shape_C((128, self.columns)))
        score = cute.make_tensor(pointer, sf.layout)
        dp = cute.make_tensor(pointer + self.columns, sf.layout)
        pf = product_mma.make_fragment_C(product_mma.get_slice(0).partition_shape_C((128, self.dc)))

        if tid >= 288 and tid < 320:
            # Keep repeated wide-head transfer scheduling compact in the loader.
            if cutlass.const_expr(streamed):
                cursor = cutlass.Int32(0)
                right_head = own_head // ratio if cutlass.const_expr(self.mode == "q") else own_head * ratio
                for c in cutlass.range(nch, unroll=1):
                    self.stream_pair(
                        la,
                        lt,
                        ra,
                        rt,
                        a_ring,
                        b_ring,
                        right_ready,
                        right_free,
                        cursor,
                        batch,
                        own_head,
                        start,
                        right_head,
                        begin * self.columns,
                        c,
                    )
                    cursor += 1
                for c in cutlass.range(nch, unroll=1):
                    self.stream_pair(
                        oa,
                        ot,
                        da,
                        dt,
                        a_ring,
                        b_ring,
                        right_ready,
                        right_free,
                        cursor,
                        batch,
                        own_head,
                        start,
                        right_head,
                        begin * self.columns,
                        c,
                    )
                    cursor += 1
                for step in range(steps):
                    block = begin + step % span
                    right_head = (
                        own_head // ratio if cutlass.const_expr(self.mode == "q") else own_head * ratio + step // span
                    )
                    for c in cutlass.range(nv, unroll_full=True):
                        if cutlass.const_expr(self.mode == "q"):
                            self.stream_right(
                                ra,
                                rt,
                                b_ring,
                                right_ready,
                                right_free,
                                cursor,
                                batch,
                                right_head,
                                block * self.columns,
                                offset // self.dc + c,
                            )
                        else:
                            self.stream_right(
                                da,
                                dt,
                                b_ring,
                                right_ready,
                                right_free,
                                cursor,
                                batch,
                                right_head,
                                block * self.columns,
                                offset // self.dc + c,
                            )
                        cursor += 1
                    if step + 1 < steps:
                        following = step + 1
                        next_block = begin + following % span
                        next_head = (
                            own_head // ratio
                            if cutlass.const_expr(self.mode == "q")
                            else own_head * ratio + following // span
                        )
                        for c in cutlass.range(nch, unroll=1):
                            self.stream_pair(
                                la,
                                lt,
                                ra,
                                rt,
                                a_ring,
                                b_ring,
                                right_ready,
                                right_free,
                                cursor,
                                batch,
                                own_head,
                                start,
                                next_head,
                                next_block * self.columns,
                                c,
                            )
                            cursor += 1
                        if cutlass.const_expr(not self.tmem_ds):
                            for c in cutlass.range(nch, unroll=1):
                                self.stream_pair(
                                    oa,
                                    ot,
                                    da,
                                    dt,
                                    a_ring,
                                    b_ring,
                                    right_ready,
                                    right_free,
                                    cursor,
                                    batch,
                                    own_head,
                                    start,
                                    next_head,
                                    next_block * self.columns,
                                    c,
                                )
                                cursor += 1
                    for c in cutlass.range(nv, unroll_full=True):
                        self.stream_right(
                            ra,
                            rt,
                            b_ring,
                            right_ready,
                            right_free,
                            cursor,
                            batch,
                            right_head,
                            block * self.columns,
                            offset // self.dc + c,
                        )
                        cursor += 1
                    if step + 1 < steps and cutlass.const_expr(self.tmem_ds):
                        following = step + 1
                        next_block = begin + following % span
                        next_head = (
                            own_head // ratio
                            if cutlass.const_expr(self.mode == "q")
                            else own_head * ratio + following // span
                        )
                        for c in cutlass.range(nch, unroll=1):
                            self.stream_pair(
                                oa,
                                ot,
                                da,
                                dt,
                                a_ring,
                                b_ring,
                                right_ready,
                                right_free,
                                cursor,
                                batch,
                                own_head,
                                start,
                                next_head,
                                next_block * self.columns,
                                c,
                            )
                            cursor += 1
            else:
                left_head = own_head
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(left_ready, 4 * 128 * q.shape[3])
                for c in cutlass.range(nch, unroll=1):
                    self.box_rows(la, lt, left[None, None, c], batch, left_head, start, c, left_ready, 128)
                    self.box_rows(oa, ot, other_left[None, None, c], batch, left_head, start, c, left_ready, 128)
                for step in range(steps):
                    slot = step % 2
                    other_slot = step % self.other_slots
                    block = begin + step % span
                    right_head = (
                        own_head // ratio if cutlass.const_expr(self.mode == "q") else own_head * ratio + step // span
                    )
                    cute.arch.mbarrier_wait(right_free + slot, ((step // 2) % 2) ^ 1)
                    if cutlass.const_expr(self.mode == "kv"):
                        for i in cutlass.range_constexpr(self.columns // 32):
                            col = tid % 32 + i * 32
                            for field in cutlass.range_constexpr(4):
                                source = (maximum, logsum, delta_base, delta)[field]
                                value = cutlass.Float32(0)
                                if block * self.columns + col < q.shape[2]:
                                    value = source[batch, right_head, block * self.columns + col]
                                stats[field, col, slot] = value
                        cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(right_ready + slot, 2 * self.columns * q.shape[3])
                    for c in cutlass.range(nch, unroll=1):
                        self.box_rows(
                            ra,
                            rt,
                            right[None, None, c, slot],
                            batch,
                            right_head,
                            block * self.columns,
                            c,
                            right_ready + slot,
                            self.columns,
                        )
                    cute.arch.mbarrier_wait(other_free + other_slot, ((step // self.other_slots) % 2) ^ 1)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(other_ready + other_slot, 2 * self.columns * q.shape[3])
                    for c in cutlass.range(nch, unroll=1):
                        self.box_rows(
                            da,
                            dt,
                            other_right[None, None, c, other_slot],
                            batch,
                            right_head,
                            block * self.columns,
                            c,
                            other_ready + other_slot,
                            self.columns,
                        )
        elif tid >= 320 and tid < 352 and cutlass.const_expr(streamed and self.mode == "kv"):
            # Scalar global loads used to interrupt the operand loader before
            # every look-ahead QK tile. Keep two statistics slots ahead instead.
            for step in range(steps):
                block = begin + step % span
                right_head = own_head * ratio + step // span
                self.stream_stats(
                    stats,
                    maximum,
                    logsum,
                    delta_base,
                    delta,
                    other_free,
                    other_ready,
                    batch,
                    right_head,
                    block,
                    step,
                    tid,
                )
        elif tid >= 256 and tid < 288:
            if cutlass.const_expr(streamed):
                cursor = cutlass.Int32(0)
                for c in cutlass.range(nch, unroll_full=True):
                    slot = cursor % self.stream_slots
                    cute.arch.mbarrier_wait(right_ready + slot, (cursor // self.stream_slots) % 2)
                    cute.arch.fence_view_async_shared()
                    self.issue(score_mma, a_ring[None, None, slot], b_ring[None, None, slot], score, c > 0)
                    with cute.arch.elect_one():
                        tcgen05.commit(right_free + slot)
                    cursor += 1
                with cute.arch.elect_one():
                    tcgen05.commit(s_ready)
                for c in cutlass.range(nch, unroll_full=True):
                    slot = cursor % self.stream_slots
                    cute.arch.mbarrier_wait(right_ready + slot, (cursor // self.stream_slots) % 2)
                    cute.arch.fence_view_async_shared()
                    self.issue(score_mma, a_ring[None, None, slot], b_ring[None, None, slot], dp, c > 0)
                    with cute.arch.elect_one():
                        tcgen05.commit(right_free + slot)
                    cursor += 1
                with cute.arch.elect_one():
                    tcgen05.commit(dp_ready)
                for step in range(steps):
                    cute.arch.mbarrier_wait(p_ready, step % 2)
                    for c in cutlass.range(nv, unroll_full=True):
                        slot = cursor % self.stream_slots
                        cute.arch.mbarrier_wait(right_ready + slot, (cursor // self.stream_slots) % 2)
                        cute.arch.fence_view_async_shared()
                        self.issue_pv(
                            q.element_type,
                            probability_pointer,
                            b_ring[None, None, slot],
                            aux_pointer + c * self.dc,
                            step > 0,
                        )
                        with cute.arch.elect_one():
                            tcgen05.commit(right_free + slot)
                        cursor += 1
                    with cute.arch.elect_one():
                        tcgen05.commit(p_free)
                    if cutlass.const_expr(self.columns == 128):
                        # Packed P readiness already follows both raw-S reads;
                        # its MMA completion releases the aliased score storage.
                        cute.arch.mbarrier_wait(p_free, step % 2)
                    if step + 1 < steps:
                        if cutlass.const_expr(self.columns != 128):
                            cute.arch.mbarrier_wait(s_free, step % 2)
                        for c in cutlass.range(nch, unroll_full=True):
                            slot = cursor % self.stream_slots
                            cute.arch.mbarrier_wait(right_ready + slot, (cursor // self.stream_slots) % 2)
                            cute.arch.fence_view_async_shared()
                            self.issue(score_mma, a_ring[None, None, slot], b_ring[None, None, slot], score, c > 0)
                            with cute.arch.elect_one():
                                tcgen05.commit(right_free + slot)
                            cursor += 1
                        with cute.arch.elect_one():
                            tcgen05.commit(s_ready)
                        # A TMEM dS must be consumed before the next dP overwrites it.
                        if cutlass.const_expr(not self.tmem_ds):
                            cute.arch.mbarrier_wait(dp_free, step % 2)
                            for c in cutlass.range(nch, unroll_full=True):
                                slot = cursor % self.stream_slots
                                cute.arch.mbarrier_wait(right_ready + slot, (cursor // self.stream_slots) % 2)
                                cute.arch.fence_view_async_shared()
                                self.issue(score_mma, a_ring[None, None, slot], b_ring[None, None, slot], dp, c > 0)
                                with cute.arch.elect_one():
                                    tcgen05.commit(right_free + slot)
                                cursor += 1
                            with cute.arch.elect_one():
                                tcgen05.commit(dp_ready)
                    ds_slot = step % self.ds_slots
                    cute.arch.mbarrier_wait(ds_ready + ds_slot, (step // self.ds_slots) % 2)
                    cute.arch.fence_view_async_shared()
                    for c in cutlass.range(nv, unroll_full=True):
                        slot = cursor % self.stream_slots
                        cute.arch.mbarrier_wait(right_ready + slot, (cursor // self.stream_slots) % 2)
                        cute.arch.fence_view_async_shared()
                        transposed = cute.composition(
                            b_ring[None, None, slot], cute.make_layout((self.dc, self.columns), stride=(self.columns, 1))
                        )
                        acc = cute.make_tensor(grad_pointer + c * self.dc, pf.layout)
                        if cutlass.const_expr(self.tmem_ds):
                            self.issue_pv(
                                q.element_type,
                                pointer + self.columns,
                                b_ring[None, None, slot],
                                grad_pointer + c * self.dc,
                                step > 0,
                            )
                        else:
                            self.issue(product_mma, ds[None, None, ds_slot], transposed, acc, step > 0)
                        with cute.arch.elect_one():
                            tcgen05.commit(right_free + slot)
                        cursor += 1
                    with cute.arch.elect_one():
                        if cutlass.const_expr(self.tmem_ds):
                            tcgen05.commit(dp_free)
                        else:
                            tcgen05.commit(ds_free + ds_slot)
                    if step + 1 < steps and cutlass.const_expr(self.tmem_ds):
                        cute.arch.mbarrier_wait(dp_free, step % 2)
                        for c in cutlass.range(nch, unroll_full=True):
                            slot = cursor % self.stream_slots
                            cute.arch.mbarrier_wait(right_ready + slot, (cursor // self.stream_slots) % 2)
                            cute.arch.fence_view_async_shared()
                            self.issue(score_mma, a_ring[None, None, slot], b_ring[None, None, slot], dp, c > 0)
                            with cute.arch.elect_one():
                                tcgen05.commit(right_free + slot)
                            cursor += 1
                        with cute.arch.elect_one():
                            tcgen05.commit(dp_ready)
                with cute.arch.elect_one():
                    tcgen05.commit(output_ready)
            else:
                cute.arch.mbarrier_wait(left_ready, 0)
                cute.arch.mbarrier_wait(right_ready, 0)
                for c in cutlass.range(nch, unroll_full=True):
                    self.issue(score_mma, left[None, None, c], right[None, None, c, 0], score, c > 0)
                with cute.arch.elect_one():
                    tcgen05.commit(s_ready)
                cute.arch.mbarrier_wait(other_ready, 0)
                for c in cutlass.range(nch, unroll_full=True):
                    self.issue(score_mma, other_left[None, None, c], other_right[None, None, c, 0], dp, c > 0)
                with cute.arch.elect_one():
                    tcgen05.commit(dp_ready)
                for step in range(steps):
                    slot = step % 2
                    other_slot = step % self.other_slots
                    ds_slot = step % self.ds_slots
                    cute.arch.mbarrier_wait(p_ready, step % 2)
                    for c in cutlass.range(nv, unroll_full=True):
                        source = (
                            right[None, None, offset // self.dc + c, slot]
                            if cutlass.const_expr(self.mode == "q")
                            else other_right[None, None, offset // self.dc + c, other_slot]
                        )
                        self.issue_pv(q.element_type, probability_pointer, source, aux_pointer + c * self.dc, step > 0)
                    with cute.arch.elect_one():
                        tcgen05.commit(p_free)
                        tcgen05.commit(other_free + other_slot)
                    if cutlass.const_expr(self.columns == 128):
                        # The next score overwrites the aliased packed-P operand.
                        cute.arch.mbarrier_wait(p_free, step % 2)
                    if step + 1 < steps:
                        following = step + 1
                        next_slot = following % 2
                        cute.arch.mbarrier_wait(s_free, step % 2)
                        cute.arch.mbarrier_wait(right_ready + next_slot, (following // 2) % 2)
                        for c in cutlass.range(nch, unroll_full=True):
                            self.issue(score_mma, left[None, None, c], right[None, None, c, next_slot], score, c > 0)
                        with cute.arch.elect_one():
                            tcgen05.commit(s_ready)
                        cute.arch.mbarrier_wait(dp_free, step % 2)
                        cute.arch.mbarrier_wait(
                            other_ready + following % self.other_slots, (following // self.other_slots) % 2
                        )
                        for c in cutlass.range(nch, unroll_full=True):
                            self.issue(
                                score_mma,
                                other_left[None, None, c],
                                other_right[None, None, c, following % self.other_slots],
                                dp,
                                c > 0,
                            )
                        with cute.arch.elect_one():
                            tcgen05.commit(dp_ready)
                    cute.arch.mbarrier_wait(ds_ready + ds_slot, (step // self.ds_slots) % 2)
                    cute.arch.fence_view_async_shared()
                    for c in cutlass.range(nv, unroll_full=True):
                        source = right[None, None, offset // self.dc + c, slot]
                        transposed = cute.composition(
                            source, cute.make_layout((self.dc, self.columns), stride=(self.columns, 1))
                        )
                        acc = cute.make_tensor(grad_pointer + c * self.dc, pf.layout)
                        self.issue(product_mma, ds[None, None, ds_slot], transposed, acc, step > 0)
                    with cute.arch.elect_one():
                        tcgen05.commit(ds_free + ds_slot)
                        tcgen05.commit(right_free + slot)
                with cute.arch.elect_one():
                    tcgen05.commit(output_ready)
        elif tid < 256 and cutlass.const_expr(self.columns == 128):
            local, half = tid % 128, tid // 128
            total, rounded_total = cutlass.Float32(0), cutlass.Float32(0)
            magnitude = scale if scale >= 0 else -scale
            sign = cutlass.Float32(1) if scale >= 0 else cutlass.Float32(-1)
            row_max, row_logsum, center = cutlass.Float32(0), cutlass.Float32(0), cutlass.Float32(0)
            if cutlass.const_expr(self.mode == "q"):
                if start + local < q.shape[2]:
                    row_max = maximum[batch, own_head, start + local]
                    row_logsum = logsum[batch, own_head, start + local]
                    center = delta_base[batch, own_head, start + local]
            for step in range(steps):
                slot = step % 2
                ds_slot = step % self.ds_slots
                block = begin + step % span
                cute.arch.mbarrier_wait(s_ready, step % 2)
                if cutlass.const_expr(streamed and self.mode == "kv"):
                    # The statistics warp publishes separately from score MMA.
                    cute.arch.mbarrier_wait(other_ready + slot, (step // 2) % 2)
                values = self.read_matrix(pointer + cute.assume(half * 64, divby=16), 64, local)
                packed = cute.make_rmem_tensor((32,), cutlass.Uint32)
                self.reconstruct_probabilities(
                    values,
                    packed,
                    stats,
                    slot,
                    half,
                    (local, start, block, q.shape[2], k.shape[2]),
                    (row_max, row_logsum, magnitude, sign),
                    q.element_type,
                )
                # Both halves consume raw S before packed P overwrites its first half.
                cute.arch.barrier(barrier_id=4, number_of_threads=256)
                if cutlass.const_expr(not streamed):
                    if local == 0:
                        cute.arch.mbarrier_arrive(s_free)
                # Pack after the raw-score barrier, keeping only 16 packed
                # columns live alongside the retained FP32 probabilities.
                for part in cutlass.range_constexpr(2):
                    packed_part = cute.make_rmem_tensor((16,), cutlass.Uint32)
                    for i in cutlass.range_constexpr(16):
                        packed_part[i] = pack_half_pair(
                            values[part * 32 + 2 * i], values[part * 32 + 2 * i + 1], q.element_type
                        )
                    self.write_matrix(pointer + cute.assume(half * 32 + part * 16, divby=16), packed_part, 16, local)
                self.fused_sync(half)
                if local == 0:
                    cute.arch.mbarrier_arrive(p_ready)
                cute.arch.mbarrier_wait(dp_ready, step % 2)
                # Streamed TMEM dP readiness follows the preceding dS MMA via
                # dp_free, so it already proves that the dS allocation is free.
                if cutlass.const_expr(not (streamed and self.tmem_ds)):
                    cute.arch.mbarrier_wait(ds_free + ds_slot, ((step // self.ds_slots) % 2) ^ 1)
                if cutlass.const_expr(streamed and self.tmem_ds):
                    ds_packed = cute.make_rmem_tensor((32,), cutlass.Uint32)
                for chunk in cutlass.range_constexpr(2):
                    dps = self.read_matrix(pointer + self.columns + cute.assume(half * 64 + chunk * 32, divby=16), 32, local)
                    for i in cutlass.range_constexpr(16):
                        col = half * 64 + chunk * 32 + 2 * i
                        corrections = (center, center)
                        if cutlass.const_expr(self.mode == "kv"):
                            corrections = (stats[2, col, slot], stats[2, col + 1, slot])
                        pair = cute.arch.sub_packed_f32x2((dps[2 * i], dps[2 * i + 1]), corrections)
                        if cutlass.const_expr(self.mode == "kv"):
                            pair = cute.arch.sub_packed_f32x2(pair, (stats[3, col, slot], stats[3, col + 1, slot]))
                        pair = cute.arch.mul_packed_f32x2(pair, (values[chunk * 32 + 2 * i], values[chunk * 32 + 2 * i + 1]))
                        dps[2 * i], dps[2 * i + 1] = pair
                    if cutlass.const_expr(self.mode == "q"):
                        total += dps.load().reduce(cute.ReductionOp.ADD, 0.0, 0)
                    dps.store(dps.load().to(q.element_type).to(cutlass.Float32))
                    if cutlass.const_expr(self.mode == "q"):
                        rounded_total += dps.load().reduce(cute.ReductionOp.ADD, 0.0, 0)
                    if cutlass.const_expr(streamed and self.tmem_ds):
                        for i in cutlass.range_constexpr(16):
                            ds_packed[chunk * 16 + i] = pack_half_pair(dps[2 * i], dps[2 * i + 1], q.element_type)
                    else:
                        target = cute.local_tile(ds[None, None, ds_slot], (128, 32), (0, half * 2 + chunk))
                        self.store_ds(dps, target, local)
                if cutlass.const_expr(streamed and self.tmem_ds):
                    cute.arch.barrier(barrier_id=4, number_of_threads=256)
                    self.write_matrix(pointer + self.columns + cute.assume(half * 32, divby=16), ds_packed, 32, local)
                self.fused_sync(half)
                if local == 0:
                    if cutlass.const_expr(not (streamed and self.tmem_ds)):
                        cute.arch.mbarrier_arrive(dp_free)
                    cute.arch.mbarrier_arrive(ds_ready + ds_slot)
                    if cutlass.const_expr(streamed and self.mode == "kv"):
                        cute.arch.mbarrier_arrive(other_free + slot)
            cute.arch.mbarrier_wait(output_ready, 0)
            if cutlass.const_expr(self.mode == "q"):
                stats[0, local, half], stats[1, local, half] = total, rounded_total
            cute.arch.barrier(barrier_id=4, number_of_threads=256)
            if half == 0:
                if cutlass.const_expr(self.mode == "q"):
                    total = stats[0, local, 0] + stats[0, local, 1]
                    rounded_total = stats[1, local, 0] + stats[1, local, 1]
                    if start + local < q.shape[2] and wi == 0:
                        delta[batch, own_head, start + local] = total
                # Keep the FP32 correction in 32-column register windows.
                # Loading both 64-column accumulators spills at this CTA size.
                for c in cutlass.range(self.width // 64, unroll_full=True):
                    for part in cutlass.range_constexpr(2):
                        result = self.read_matrix(grad_pointer + c * 64 + part * 32, 32, local)
                        if cutlass.const_expr(self.mode == "q"):
                            auxiliary = self.read_matrix(aux_pointer + c * 64 + part * 32, 32, local)
                        for i in cutlass.range_constexpr(32):
                            value = result[i]
                            if cutlass.const_expr(self.mode == "q"):
                                value -= rounded_total * auxiliary[i]
                            epilogue_staging[local, part * 32 + i] = (value * scale).to(q.element_type)
                    cute.arch.barrier(barrier_id=2, number_of_threads=128)
                    target = dq if cutlass.const_expr(self.mode == "q") else dk
                    self.store_output(epilogue_staging, target, batch, own_head, start, offset + c * 64, local)
                    cute.arch.barrier(barrier_id=2, number_of_threads=128)
                    if cutlass.const_expr(self.mode == "kv"):
                        # K has been stored; V no longer overlaps its live registers.
                        for part in cutlass.range_constexpr(2):
                            auxiliary = self.read_matrix(aux_pointer + c * 64 + part * 32, 32, local)
                            for i in cutlass.range_constexpr(32):
                                epilogue_staging[local, part * 32 + i] = auxiliary[i].to(q.element_type)
                        cute.arch.barrier(barrier_id=2, number_of_threads=128)
                        self.store_output(epilogue_staging, dv, batch, own_head, start, offset + c * 64, local)
                        cute.arch.barrier(barrier_id=2, number_of_threads=128)
        elif cutlass.const_expr(self.columns == 64):
            if tid >= 128 and tid < 256:
                local = tid - 128
                magnitude = scale if scale >= 0 else -scale
                sign = cutlass.Float32(1) if scale >= 0 else cutlass.Float32(-1)
                row_max, row_logsum = cutlass.Float32(0), cutlass.Float32(0)
                if cutlass.const_expr(self.mode == "q"):
                    if start + local < q.shape[2]:
                        row_max = maximum[batch, own_head, start + local]
                        row_logsum = logsum[batch, own_head, start + local]
                for step in range(steps):
                    slot = step % 2
                    block = begin + step % span
                    cute.arch.mbarrier_wait(s_ready, step % 2)
                    if cutlass.const_expr(streamed and self.mode == "kv"):
                        cute.arch.mbarrier_wait(other_ready + slot, (step // 2) % 2)
                    values = self.read_matrix(pointer, self.columns, local)
                    packed = cute.make_rmem_tensor((self.columns // 2,), cutlass.Uint32)
                    self.reconstruct_probabilities(
                        values,
                        packed,
                        stats,
                        slot,
                        0,
                        (local, start, block, q.shape[2], k.shape[2]),
                        (row_max, row_logsum, magnitude, sign),
                        q.element_type,
                    )
                    if step > 0:
                        cute.arch.mbarrier_wait(p_free, (step - 1) % 2)
                    self.write_matrix(pointer, values, self.columns, local)
                    self.write_matrix(pointer + 2 * self.columns, packed, self.columns // 2, local)
                    cute.arch.barrier(barrier_id=3, number_of_threads=128)
                    if local == 0:
                        cute.arch.mbarrier_arrive(p_ready)
            elif tid < 128:
                total, rounded_total = cutlass.Float32(0), cutlass.Float32(0)
                center = cutlass.Float32(0)
                if cutlass.const_expr(self.mode == "q"):
                    if start + tid < q.shape[2]:
                        center = delta_base[batch, own_head, start + tid]
                for step in range(steps):
                    slot = step % 2
                    cute.arch.mbarrier_wait(dp_ready, step % 2)
                    if cutlass.const_expr(streamed and self.mode == "kv"):
                        cute.arch.mbarrier_wait(other_ready + slot, (step // 2) % 2)
                    dps = self.read_matrix(pointer + self.columns, self.columns, tid)
                    cute.arch.barrier(barrier_id=2, number_of_threads=128)
                    if tid == 0:
                        cute.arch.mbarrier_arrive(dp_free)
                    cute.arch.mbarrier_wait(p_ready, step % 2)
                    probabilities = self.read_matrix(pointer, self.columns, tid)
                    cute.arch.barrier(barrier_id=2, number_of_threads=128)
                    if tid == 0:
                        cute.arch.mbarrier_arrive(s_free)
                    for i in cutlass.range_constexpr(self.columns):
                        correction = center
                        if cutlass.const_expr(self.mode == "kv"):
                            correction = stats[2, i, slot]
                        value = dps[i] - correction
                        if cutlass.const_expr(self.mode == "kv"):
                            value -= stats[3, i, slot]
                        value = probabilities[i] * value
                        probabilities[i] = value
                        dps[i] = value.to(q.element_type).to(cutlass.Float32)
                    if cutlass.const_expr(self.mode == "q"):
                        total += probabilities.load().reduce(cute.ReductionOp.ADD, 0.0, 0)
                        rounded_total += dps.load().reduce(cute.ReductionOp.ADD, 0.0, 0)
                    cute.arch.mbarrier_wait(ds_free + slot, ((step // 2) % 2) ^ 1)
                    self.store_ds(dps, ds[None, None, slot], tid)
                    cute.arch.barrier(barrier_id=2, number_of_threads=128)
                    if tid == 0:
                        cute.arch.mbarrier_arrive(ds_ready + slot)
                        if cutlass.const_expr(streamed and self.mode == "kv"):
                            cute.arch.mbarrier_arrive(other_free + slot)
                cute.arch.mbarrier_wait(output_ready, 0)
                if cutlass.const_expr(self.mode == "q"):
                    if start + tid < q.shape[2] and wi == 0:
                        delta[batch, own_head, start + tid] = total
                for c in cutlass.range(self.width // 64, unroll_full=True):
                    result = self.read_matrix(grad_pointer + c * 64, 64, tid)
                    auxiliary = self.read_matrix(aux_pointer + c * 64, 64, tid)
                    for i in cutlass.range_constexpr(64):
                        value = result[i]
                        if cutlass.const_expr(self.mode == "q"):
                            value -= rounded_total * auxiliary[i]
                        epilogue_staging[tid, i] = (value * scale).to(q.element_type)
                    cute.arch.barrier(barrier_id=2, number_of_threads=128)
                    target = dq if cutlass.const_expr(self.mode == "q") else dk
                    self.store_output(epilogue_staging, target, batch, own_head, start, offset + c * 64, tid)
                    cute.arch.barrier(barrier_id=2, number_of_threads=128)
                    if cutlass.const_expr(self.mode == "kv"):
                        for i in cutlass.range_constexpr(64):
                            epilogue_staging[tid, i] = auxiliary[i].to(q.element_type)
                        cute.arch.barrier(barrier_id=2, number_of_threads=128)
                        self.store_output(epilogue_staging, dv, batch, own_head, start, offset + c * 64, tid)
                        cute.arch.barrier(barrier_id=2, number_of_threads=128)
        cute.arch.sync_threads()
        tmem.free(pointer)


class Query(Backward):
    pass


class KeyValue(Backward):
    pass
