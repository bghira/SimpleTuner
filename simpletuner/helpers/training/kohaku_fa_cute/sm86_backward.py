"""D256 query gradients with query-major scores and remapped row corrections."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from cutlass.memory import SmemAllocator

from .kernels import Attention, packed_half_conversion, quad_sum, row_view
from .launch import block_indices, launch_grid


class QueryMajorQueries(Attention):
    """Keep centered query math while giving the large K operand four N warps."""

    def __init__(self, causal, vector):
        super().__init__(
            "q", causal, False, columns=32, width=256, warps=8, centered=True, vector=vector, query_column_warps=4
        )
        self.rows = self.grid_rows = 64
        self.score = Attention("forward", causal, False, columns=32, width=256, warps=8, vector=vector)
        self.score.rows = self.score.grid_rows = 64
        self.score.column_warps = 2

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        score_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
            (4, 2, 1),
            permutation_mnk=(64, 32, 16),
        )
        gradient_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
            (2, 4, 1),
            permutation_mnk=(64, 64, 16),
        )
        self.kernel_queries(q, k, v, maximum, logsum, dout, delta, base, dq, scale, score_mma, gradient_mma).launch(
            grid=launch_grid(cute.ceil_div(q.shape[2], 64), q.shape[0] * q.shape[1]), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel_queries(self, q, k, v, maximum, logsum, dout, delta, base, dq, scale, score_mma, gradient_mma):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, _ = block_indices(cute.ceil_div(q.shape[2], 64), q.shape[0] * q.shape[1])
        batch, head, start = bh // q.shape[1], bh % q.shape[1], tile * 64
        score_thread, gradient_thread = score_mma.get_slice(tid), gradient_mma.get_slice(tid)
        allocator = SmemAllocator()
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        query_layout = cute.tile_to_shape(atom, (64, 256), (0, 1))
        key_layout = cute.tile_to_shape(atom, (32, 256), (0, 1))
        weight_layout = cute.tile_to_shape(atom, (64, 32), (0, 1))
        queries = self.shared(allocator, q.element_type, query_layout)
        gradients = self.shared(allocator, q.element_type, query_layout)
        keys = self.shared(allocator, q.element_type, key_layout)
        values = self.shared(allocator, q.element_type, key_layout)
        derivatives = cute.make_tensor(values.iterator, weight_layout)
        probabilities = cute.make_tensor(values.iterator + cute.cosize(weight_layout), weight_layout)
        # Score N-warps own separate row sums. The gradient MMA has a different
        # row layout, so its epilogue gathers by coordinate after this reduction.
        shared_rows = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((2, 2, 64), stride=(128, 64, 1)), byte_alignment=128
        )
        self.load(q, queries, batch, head, start, tid)
        self.load(dout, gradients, batch, head, start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        accum = cute.make_rmem_tensor(gradient_thread.partition_shape_C((64, 256)), cutlass.Float32)
        mean = cute.make_fragment_like(accum)
        accum.fill(0.0)
        mean.fill(0.0)
        score_coords = score_thread.partition_C(cute.make_identity_tensor((64, 32)))
        score_rows = row_view(score_coords)
        totals = cute.make_rmem_tensor((cute.size(score_rows.shape[0]),), cutlass.Float32)
        rounded = cute.make_fragment_like(totals)
        row_max, row_logsum, center = (
            cute.make_fragment_like(totals),
            cute.make_fragment_like(totals),
            cute.make_fragment_like(totals),
        )
        totals.fill(0.0)
        rounded.fill(0.0)
        for r in cutlass.range_constexpr(cute.size(score_rows.shape[0])):
            qr = start + score_rows[r, 0][0]
            row_max[r], row_logsum[r], center[r] = 0.0, 0.0, 0.0
            if qr < q.shape[2]:
                row_max[r], row_logsum[r], center[r] = (
                    maximum[batch, head, qr],
                    logsum[batch, head, qr],
                    base[batch, head, qr],
                )
        magnitude = scale if scale >= 0.0 else -scale
        sign = cutlass.Float32(1.0) if scale >= 0.0 else cutlass.Float32(-1.0)
        limit = cutlass.min(k.shape[2], start + 64) if cutlass.const_expr(self.causal) else k.shape[2]
        for block in range(cute.ceil_div(limit, 32)):
            key = block * 32
            self.load(k, keys, batch, head, key, tid)
            self.load(v, values, batch, head, key, tid)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            scores = self.score.dot(queries, keys, score_mma, score_thread, tid)
            dp = self.score.dot(gradients, values, score_mma, score_thread, tid)
            sv, pv = row_view(scores), row_view(dp)
            for r in cutlass.range_constexpr(cute.size(score_rows.shape[0])):
                for c in cutlass.range_constexpr(cute.size(score_rows.shape[1])):
                    row, col = score_rows[r, c]
                    visible = start + row < q.shape[2] and key + col < k.shape[2]
                    if cutlass.const_expr(self.causal):
                        visible = visible and key + col <= start + row
                    probability = cutlass.Float32(0.0)
                    if visible:
                        probability = cute.math.exp2(
                            (sv[r, c] * sign - row_max[r]) * magnitude * 1.4426950408889634 - row_logsum[r], fastmath=True
                        )
                    if cutlass.const_expr(self.baseline_precision):
                        probability = probability.to(q.element_type).to(cutlass.Float32)
                    sv[r, c] = probability * (pv[r, c] - center[r])
                    pv[r, c] = probability
                totals[r] += quad_sum(sv[r, None].load().reduce(cute.ReductionOp.ADD, 0.0, 0))
            rounded_scores = cute.make_fragment_like(scores, q.element_type)
            rounded_probabilities = cute.make_fragment_like(dp, q.element_type)
            packed_half_conversion(scores, rounded_scores)
            packed_half_conversion(dp, rounded_probabilities)
            rounded_view = row_view(rounded_scores)
            for r in cutlass.range_constexpr(cute.size(score_rows.shape[0])):
                rounded[r] += quad_sum(rounded_view[r, None].load().to(cutlass.Float32).reduce(cute.ReductionOp.ADD, 0.0, 0))
            # Release every V reader before any score warp packs P/dS there.
            cute.arch.sync_threads()
            self.store_packed_weights(rounded_scores, derivatives, score_coords)
            if cutlass.const_expr(not self.baseline_precision):
                self.store_packed_weights(rounded_probabilities, probabilities, score_coords)
            cute.arch.sync_threads()
            if cutlass.const_expr(self.baseline_precision):
                self.product(derivatives, keys, 0, gradient_mma, gradient_thread, tid, accum)
            else:
                self.dual_product(derivatives, probabilities, keys, 0, gradient_mma, gradient_thread, tid, accum, mean)
            cute.arch.sync_threads()
        if cutlass.const_expr(not self.baseline_precision):
            if tid % 4 == 0:
                for r in cutlass.range_constexpr(cute.size(score_rows.shape[0])):
                    row = score_rows[r, 0][0]
                    shared_rows[0, tid // 128, row] = totals[r]
                    shared_rows[1, tid // 128, row] = rounded[r]
            cute.arch.sync_threads()
            if tid < 128 and tid % 4 == 0:
                for r in cutlass.range_constexpr(cute.size(score_rows.shape[0])):
                    row = score_rows[r, 0][0]
                    if start + row < q.shape[2]:
                        delta[batch, head, start + row] = shared_rows[0, 0, row] + shared_rows[0, 1, row]
            gradient_rows = row_view(gradient_thread.partition_C(cute.make_identity_tensor((64, 256))))
            av, mv = row_view(accum), row_view(mean)
            for r in cutlass.range_constexpr(cute.size(gradient_rows.shape[0])):
                row = gradient_rows[r, 0][0]
                mass = shared_rows[1, 0, row] + shared_rows[1, 1, row]
                av[r, None] = av[r, None].load() - mass * mv[r, None].load()
        self.store_result(accum, dq, queries, batch, head, start, 0, gradient_thread, tid, scale)
