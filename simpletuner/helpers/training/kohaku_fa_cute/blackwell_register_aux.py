"""Wide Blackwell backward with an FP32 register-resident auxiliary.

Keeping the P-weighted-key mean / dV sum in registers leaves TMEM for a
256-wide dQ/dK accumulator. D512 output slicing falls from four to two while
FP32 probabilities and the full centered correction are retained.

The measured selector targets dense/causal BF16 MHA on SM103. Other inputs use the
existing backward. Role-local register changes need a launch bound and a CTA
join before restoring registers; vector staging reduces readout register pressure.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05, warpgroup
from cutlass.memory import SmemAllocator, TmemAllocator

from .blackwell_pipeline import Backward
from .kernels import OutputDelta
from .launch import block_indices, launch_grid


class RegisterAux(Backward):

    def __init__(self, mode, causal=False):
        super().__init__(mode, causal, False, columns=128, warps=12, width=128, tma=True)
        self.width, self.stream_slots = (256, 5)

    @cute.jit
    def __call__(self, q, k, v, maximum, logsum, dout, delta, base, dq, dk, dv, scale, stream):
        left, other, right, other_right = (q, dout, k, v) if cutlass.const_expr(self.mode == "q") else (k, v, q, dout)
        la, lt = self.descriptor(left, 128)
        oa, ot = self.descriptor(other, 128)
        ra, rt = self.descriptor(right, 128)
        da, dt = self.descriptor(other_right, 128)
        length, heads = (q.shape[2], q.shape[1]) if cutlass.const_expr(self.mode == "q") else (k.shape[2], k.shape[1])
        self.kernel(q, k, maximum, logsum, dout, delta, base, dq, dk, dv, scale, (la, lt, oa, ot, ra, rt, da, dt)).launch(
            grid=launch_grid(cute.ceil_div(length, 128) * cute.ceil_div(q.shape[3], 256), q.shape[0] * heads),
            block=(512, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.jit
    def product(self, mma, weights, right, pointer, accumulate):
        frag = mma.make_fragment_C(mma.get_slice(0).partition_shape_C((128, 64)))
        transposed = cute.composition(right, cute.make_layout((64, 128), stride=(128, 1)))
        self.issue(mma, weights, transposed, cute.make_tensor(pointer, frag.layout), accumulate)

    @cute.jit
    def subgroup_sync(self, half):
        if half == 0:
            cute.arch.barrier(barrier_id=5, number_of_threads=128)
        else:
            cute.arch.barrier(barrier_id=6, number_of_threads=128)

    @cute.jit
    def probability(self, values, column, stats, local, start, block, sq, sk, row_max, row_log, magnitude, sign):
        full = start + 128 <= (sq if cutlass.const_expr(self.mode == "q") else sk) and (block + 1) * 128 <= (
            sk if cutlass.const_expr(self.mode == "q") else sq
        )
        if cutlass.const_expr(self.causal):
            full = full and (
                (block + 1) * 128 <= start + 1 if cutlass.const_expr(self.mode == "q") else block * 128 >= start + 127
            )
        if full:
            for i in cutlass.range_constexpr(8):
                col = column + 2 * i
                maxima = (row_max, row_max)
                logs = (row_log, row_log)
                if cutlass.const_expr(self.mode == "kv"):
                    maxima = (stats[0, col], stats[0, col + 1])
                    logs = (stats[1, col], stats[1, col + 1])
                pair = self.shifted_scaled_pair((values[2 * i], values[2 * i + 1]), maxima, magnitude, sign)
                pair = cute.arch.sub_packed_f32x2(pair, logs)
                values[2 * i] = cute.math.exp2(pair[0], fastmath=True)
                values[2 * i + 1] = cute.math.exp2(pair[1], fastmath=True)
        else:
            for i in cutlass.range_constexpr(16):
                col = column + i
                valid = start + local < (sq if cutlass.const_expr(self.mode == "q") else sk) and block * 128 + col < (
                    sk if cutlass.const_expr(self.mode == "q") else sq
                )
                if cutlass.const_expr(self.causal):
                    valid = valid and (
                        block * 128 + col <= start + local
                        if cutlass.const_expr(self.mode == "q")
                        else block * 128 + col >= start + local
                    )
                value = cutlass.Float32(0)
                if valid:
                    maximum, logsum = (row_max, row_log)
                    if cutlass.const_expr(self.mode == "kv"):
                        maximum, logsum = (stats[0, col], stats[1, col])
                    value = cute.math.exp2(
                        (values[i] * sign - maximum) * magnitude * cutlass.Float32(1.4426950408889634) - logsum,
                        fastmath=True,
                    )
                values[i] = value

    @cute.kernel
    def kernel(self, q, k, maximum, logsum, dout, delta, base, dq, dk, dv, scale, args):
        tid, _, _ = cute.arch.thread_idx()
        slices = cute.ceil_div(q.shape[3], 256)
        heads = q.shape[1] if cutlass.const_expr(self.mode == "q") else k.shape[1]
        length = q.shape[2] if cutlass.const_expr(self.mode == "q") else k.shape[2]
        tile, bh, _ = block_indices(cute.ceil_div(length, 128) * slices, q.shape[0] * heads)
        tile, wi = (tile // slices, tile % slices)
        batch, own_head = (bh // heads, bh % heads)
        ratio = q.shape[1] // k.shape[1]
        if cutlass.const_expr(self.mode == "q" and self.causal):
            tile = cute.ceil_div(q.shape[2], 128) - 1 - tile
        start, offset = (tile * 128, wi * 256)
        count = cute.ceil_div(k.shape[2] if cutlass.const_expr(self.mode == "q") else q.shape[2], 128)
        begin = cutlass.Int32(0)
        if cutlass.const_expr(self.causal):
            if cutlass.const_expr(self.mode == "q"):
                count = cute.ceil_div(cutlass.min(k.shape[2], start + 128), 128)
            else:
                begin = cutlass.min(start // 128, count - 1)
        span = count - begin
        steps = span * (1 if cutlass.const_expr(self.mode == "q") else ratio)
        nch = q.shape[3] // 64
        alloc = SmemAllocator()
        bars = alloc.allocate_array(cutlass.Int64, 18)
        ready, free = (bars, bars + 5)
        sr, dr, wr, wf, ar, af, final, tr = (
            bars + 10,
            bars + 11,
            bars + 12,
            bars + 13,
            bars + 14,
            bars + 15,
            bars + 16,
            bars + 17,
        )
        stat_bars = alloc.allocate_array(cutlass.Int64, 2)
        stat_ready, stat_free = (stat_bars, stat_bars + 1)
        holding = alloc.allocate_array(cutlass.Int32, 1)
        atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, q.element_type)
        ring_layout = cute.tile_to_shape(atom, (128, 64, 5), (0, 1, 2))
        a = self.shared(alloc, q.element_type, ring_layout)
        b = self.shared(alloc, q.element_type, ring_layout)
        layout = cute.tile_to_shape(atom, (128, 128), (0, 1))
        probabilities = self.shared(alloc, q.element_type, layout)
        derivatives = self.shared(alloc, q.element_type, layout)
        stats = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((4, 128), stride=(128, 1)), byte_alignment=128)
        tm = TmemAllocator(
            alloc_result_dst_smem_ptr=holding, barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=512)
        )
        tm.allocate(512)
        tm.wait_for_alloc()
        pointer = tm.retrieve_ptr(cutlass.Float32)
        tm.relinquish_alloc_permit()
        if tid == 0:
            for i in cutlass.range_constexpr(18):
                cute.arch.mbarrier_init(bars + i, 1)
            cute.arch.mbarrier_init(af, 2)
            cute.arch.mbarrier_init(stat_ready, 1)
            cute.arch.mbarrier_init(stat_free, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        la, lt, oa, ot, ra, rt, da, dt = args
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
        prod_mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                cutlass.Float32,
                (128, 64, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.MN,
            )
        )
        frag = score_mma.make_fragment_C(score_mma.get_slice(0).partition_shape_C((128, 128)))
        score = cute.make_tensor(pointer, frag.layout)
        dp = cute.make_tensor(pointer + 128, frag.layout)
        if tid >= 384:
            cute.arch.setmaxregister_decrease(64)
            if tid >= 416 and tid < 448:
                cursor = cutlass.Int32(0)
                for step in range(steps):
                    block = begin + step % span
                    right_head = (
                        own_head // ratio if cutlass.const_expr(self.mode == "q") else own_head * ratio + step // span
                    )
                    for c in cutlass.range(nch, unroll=1):
                        self.stream_pair(
                            la, lt, ra, rt, a, b, ready, free, cursor, batch, own_head, start, right_head, block * 128, c
                        )
                        cursor += 1
                    for c in cutlass.range(nch, unroll=1):
                        self.stream_pair(
                            oa, ot, da, dt, a, b, ready, free, cursor, batch, own_head, start, right_head, block * 128, c
                        )
                        cursor += 1
                    for c in cutlass.range_constexpr(4):
                        if cutlass.const_expr(self.mode == "q"):
                            self.stream_right(
                                ra, rt, b, ready, free, cursor, batch, right_head, block * 128, offset // 64 + c
                            )
                        else:
                            self.stream_right(
                                da, dt, b, ready, free, cursor, batch, right_head, block * 128, offset // 64 + c
                            )
                        cursor += 1
                    if cutlass.const_expr(self.mode == "kv"):
                        for c in cutlass.range_constexpr(4):
                            self.stream_right(
                                ra, rt, b, ready, free, cursor, batch, right_head, block * 128, offset // 64 + c
                            )
                            cursor += 1
            elif tid >= 448 and tid < 480 and cutlass.const_expr(self.mode == "kv"):
                for step in range(steps):
                    cute.arch.mbarrier_wait(stat_free, step % 2 ^ 1)
                    block = begin + step % span
                    right_head = own_head * ratio + step // span
                    for i in cutlass.range_constexpr(4):
                        col = tid % 32 + i * 32
                        for field in cutlass.range_constexpr(4):
                            source = (maximum, logsum, base, delta)[field]
                            value = cutlass.Float32(0)
                            if block * 128 + col < q.shape[2]:
                                value = source[batch, right_head, block * 128 + col]
                            stats[field, col] = value
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(stat_ready)
            elif tid >= 384 and tid < 416:
                cursor = cutlass.Int32(0)
                for step in range(steps):
                    if step > 0:
                        cute.arch.mbarrier_wait(af, (step - 1) % 2)
                    for c in cutlass.range(nch, unroll=1):
                        slot = cursor % 5
                        cute.arch.mbarrier_wait(ready + slot, cursor // 5 % 2)
                        cute.arch.fence_view_async_shared()
                        self.issue(score_mma, a[None, None, slot], b[None, None, slot], score, c > 0)
                        with cute.arch.elect_one():
                            tcgen05.commit(free + slot)
                        cursor += 1
                    with cute.arch.elect_one():
                        tcgen05.commit(sr)
                    for c in cutlass.range(nch, unroll=1):
                        slot = cursor % 5
                        cute.arch.mbarrier_wait(ready + slot, cursor // 5 % 2)
                        cute.arch.fence_view_async_shared()
                        self.issue(score_mma, a[None, None, slot], b[None, None, slot], dp, c > 0)
                        with cute.arch.elect_one():
                            tcgen05.commit(free + slot)
                        cursor += 1
                    with cute.arch.elect_one():
                        tcgen05.commit(dr)
                    cute.arch.mbarrier_wait(wr, step % 2)
                    cute.arch.fence_view_async_shared()
                    for c in cutlass.range_constexpr(4):
                        slot = cursor % 5
                        cute.arch.mbarrier_wait(ready + slot, cursor // 5 % 2)
                        cute.arch.fence_view_async_shared()
                        self.product(prod_mma, probabilities, b[None, None, slot], pointer + c * 64, False)
                        if cutlass.const_expr(self.mode == "q"):
                            self.product(prod_mma, derivatives, b[None, None, slot], pointer + 256 + c * 64, step > 0)
                        with cute.arch.elect_one():
                            tcgen05.commit(free + slot)
                        cursor += 1
                    if cutlass.const_expr(self.mode == "kv"):
                        for c in cutlass.range_constexpr(4):
                            slot = cursor % 5
                            cute.arch.mbarrier_wait(ready + slot, cursor // 5 % 2)
                            cute.arch.fence_view_async_shared()
                            self.product(prod_mma, derivatives, b[None, None, slot], pointer + 256 + c * 64, step > 0)
                            with cute.arch.elect_one():
                                tcgen05.commit(free + slot)
                            cursor += 1
                    with cute.arch.elect_one():
                        tcgen05.commit(ar)
                        tcgen05.commit(wf)
                with cute.arch.elect_one():
                    tcgen05.commit(final)
        elif tid < 128:
            # Causal KV predicates need more pointwise registers. Vector staging
            # lets its auxiliary consumers fit in the correspondingly smaller pool.
            if cutlass.const_expr(self.mode == "kv" and self.causal):
                cute.arch.setmaxregister_decrease(96)
            else:
                cute.arch.setmaxregister_decrease(80)
            local = tid
            row_max, row_log, center = (cutlass.Float32(0), cutlass.Float32(0), cutlass.Float32(0))
            total, rounded = (cutlass.Float32(0), cutlass.Float32(0))
            if cutlass.const_expr(self.mode == "q"):
                if start + local < q.shape[2]:
                    row_max, row_log, center = (
                        maximum[batch, own_head, start + local],
                        logsum[batch, own_head, start + local],
                        base[batch, own_head, start + local],
                    )
            magnitude = scale if scale >= 0 else -scale
            sign = cutlass.Float32(1) if scale >= 0 else cutlass.Float32(-1)
            for step in range(steps):
                block = begin + step % span
                cute.arch.mbarrier_wait(sr, step % 2)
                cute.arch.mbarrier_wait(wf, step % 2 ^ 1)
                if cutlass.const_expr(self.mode == "kv"):
                    cute.arch.mbarrier_wait(stat_ready, step % 2)
                # Reconstruct P while the independent dP product is running.
                # Reuse the completed score tile for FP32 P, then reread it for
                # dS; the shared matrix operand is the separately rounded copy.
                for early in cutlass.range_constexpr(8):
                    col = early * 16
                    values = self.read_matrix(pointer + col, 16, local)
                    self.probability(
                        values, col, stats, local, start, block, q.shape[2], k.shape[2], row_max, row_log, magnitude, sign
                    )
                    self.store_ds(values, cute.local_tile(probabilities, (128, 16), (0, early)), local)
                    self.write_matrix(pointer + col, values, 16, local)
                cute.arch.mbarrier_wait(dr, step % 2)
                for chunk in cutlass.range_constexpr(8):
                    col = chunk * 16
                    values = self.read_matrix(pointer + col, 16, local)
                    dps = self.read_matrix(pointer + 128 + col, 16, local)
                    for i in cutlass.range_constexpr(8):
                        corrections = (center, center)
                        if cutlass.const_expr(self.mode == "kv"):
                            corrections = (stats[2, col + 2 * i], stats[2, col + 2 * i + 1])
                        pair = cute.arch.sub_packed_f32x2((dps[2 * i], dps[2 * i + 1]), corrections)
                        if cutlass.const_expr(self.mode == "kv"):
                            pair = cute.arch.sub_packed_f32x2(pair, (stats[3, col + 2 * i], stats[3, col + 2 * i + 1]))
                        pair = cute.arch.mul_packed_f32x2(pair, (values[2 * i], values[2 * i + 1]))
                        dps[2 * i], dps[2 * i + 1] = pair
                    if cutlass.const_expr(self.mode == "q"):
                        total += dps.load().reduce(cute.ReductionOp.ADD, 0.0, 0)
                    dps.store(dps.load().to(q.element_type).to(cutlass.Float32))
                    if cutlass.const_expr(self.mode == "q"):
                        rounded += dps.load().reduce(cute.ReductionOp.ADD, 0.0, 0)
                    self.store_ds(dps, cute.local_tile(derivatives, (128, 16), (0, chunk)), local)
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                cute.arch.fence_view_async_shared()
                if local == 0:
                    cute.arch.mbarrier_arrive(wr)
                    if cutlass.const_expr(self.mode == "kv"):
                        cute.arch.mbarrier_arrive(stat_free)
            if cutlass.const_expr(self.mode == "q"):
                stats[0, local] = rounded
                if start + local < q.shape[2] and wi == 0:
                    delta[batch, own_head, start + local] = total
            cute.arch.barrier(barrier_id=2, number_of_threads=128)
            if local == 0:
                cute.arch.mbarrier_arrive(tr)
        else:
            if cutlass.const_expr(self.mode == "kv" and self.causal):
                cute.arch.setmaxregister_increase(176)
            else:
                cute.arch.setmaxregister_increase(184)
            local, half = ((tid - 128) % 128, (tid - 128) // 128)
            acc = cute.make_rmem_tensor((128,), cutlass.Float32)
            acc.fill(0)
            for step in range(steps):
                cute.arch.mbarrier_wait(ar, step % 2)
                for c in cutlass.range_constexpr(8):
                    result = self.read_matrix(pointer + half * 128 + c * 16, 16, local)
                    for i in cutlass.range_constexpr(16):
                        acc[c * 16 + i] += result[i]
                self.subgroup_sync(half)
                if local == 0:
                    cute.arch.mbarrier_arrive(af)
            cute.arch.mbarrier_wait(final, 0)
            cute.arch.mbarrier_wait(tr, 0)
            sigma = cutlass.Float32(0)
            if cutlass.const_expr(self.mode == "q"):
                sigma = stats[0, local]
            staging = a[None, None, half]
            # Export dV before reading dK so its accumulator can die here.
            # The TMEM-compatible copy packs narrow windows into vector shared
            # stores, avoiding the scalar swizzled-address expansion and spills.
            if cutlass.const_expr(self.mode == "kv"):
                for c in cutlass.range_constexpr(2):
                    for part in cutlass.range_constexpr(4):
                        values = cute.make_rmem_tensor((16,), cutlass.Float32)
                        for i in cutlass.range_constexpr(16):
                            values[i] = acc[c * 64 + part * 16 + i]
                        self.store_ds(values, cute.local_tile(staging, (128, 16), (0, part)), local)
                    self.subgroup_sync(half)
                    self.store_output(staging, dv, batch, own_head, start, offset + half * 128 + c * 64, local)
                    self.subgroup_sync(half)
            for c in cutlass.range_constexpr(2):
                for part in cutlass.range_constexpr(4):
                    result = self.read_matrix(pointer + 256 + half * 128 + c * 64 + part * 16, 16, local)
                    for i in cutlass.range_constexpr(16):
                        value = result[i]
                        if cutlass.const_expr(self.mode == "q"):
                            value -= sigma * acc[c * 64 + part * 16 + i]
                        result[i] = value * scale
                    self.store_ds(result, cute.local_tile(staging, (128, 16), (0, part)), local)
                self.subgroup_sync(half)
                target = dq if cutlass.const_expr(self.mode == "q") else dk
                self.store_output(staging, target, batch, own_head, start, offset + half * 128 + c * 64, local)
                self.subgroup_sync(half)
        cute.arch.sync_threads()
        if tid >= 128 and tid < 384:
            cute.arch.setmaxregister_decrease(128)
        else:
            cute.arch.setmaxregister_increase(128)
        cute.arch.sync_threads()
        tm.free(pointer)


class JointBackward:

    def __init__(self, causal=False, base_vector=8, **kwargs):
        self.center = OutputDelta(base_vector)
        self.q = RegisterAux("q", causal)
        self.kv = RegisterAux("kv", causal)

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, workspace, base, scale, stream: cuda.CUstream
    ):
        self.center(out, dout, base, stream)
        self.q(q, k, v, maximum, logsum, dout, delta, base, dq, dk, dv, scale, stream)
        self.kv(q, k, v, maximum, logsum, dout, delta, base, dq, dk, dv, scale, stream)


class AdaptiveBackward:
    """Use wide dense/causal BF16 D512 tiles for sufficiently parallel MHA."""

    def __init__(self, causal, fallback, base_vector):
        self.wide = JointBackward(causal, base_vector)
        self.fallback = fallback

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        if q.shape[2] >= 512 and k.shape[2] == q.shape[2] and k.shape[1] == q.shape[1] and q.shape[0] * q.shape[1] >= 8:
            self.wide(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        else:
            self.fallback(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
