"""Full-width D256 backward with independent low-component gradient sums."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from cutlass.memory import SmemAllocator

from .kernels import Attention, packed_half_conversion, row_view
from .launch import block_indices, launch_grid


class MediumKeyValue(Attention):
    """Use Q64/K32, eight gradient warps and 96 KiB of shared memory."""

    def __init__(self, causal, *, vector, copy_cache_global):
        super().__init__(
            "kv",
            causal,
            False,
            columns=64,
            width=256,
            warps=8,
            centered=True,
            vector=vector,
            kv_column_warps=8,
            copy_cache_global=copy_cache_global,
        )
        self.baseline_precision = False
        self.compensated_keys = self.compensated_values = True
        self.scaled_key_residual = False
        self.scaled_value_residual = True
        self.rows = self.grid_rows = 32
        self.score = Attention("forward", causal, False, columns=32, width=256, warps=8, vector=vector)
        self.score.rows = self.score.grid_rows = 64
        self.score.column_warps = 2
        self.score.score_b_matrices = 4

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        assert q.shape[3] == 256 and q.shape[1] == k.shape[1]
        score_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)), (4, 2, 1), permutation_mnk=(64, 32, 16)
        )
        gradient_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)), (1, 8, 1), permutation_mnk=(32, 128, 16)
        )
        self.kernel_values(q, k, v, maximum, logsum, dout, delta, base, dk, dv, scale, score_mma, gradient_mma).launch(
            grid=launch_grid(cute.ceil_div(k.shape[2], 32), q.shape[0] * q.shape[1]), block=(256, 1, 1), stream=stream
        )

    @cute.jit
    def transposed_pair(self, weights, low, source, mma, thread, tid, result, low_result):
        transposed_weights = cute.composition(weights, cute.make_layout((32, 64), stride=(64, 1)))
        a = thread.make_fragment_A(thread.partition_A(transposed_weights))
        transposed_low = cute.composition(low, cute.make_layout((32, 64), stride=(64, 1)))
        low_a = thread.make_fragment_A(thread.partition_A(transposed_low))
        copy_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
        )
        copier_a = copy_a.get_slice(tid)
        sa, da = (copier_a.partition_S(transposed_weights), copier_a.retile(a))
        sl, dl = (copier_a.partition_S(transposed_low), copier_a.retile(low_a))
        transposed_source = cute.composition(source, cute.make_layout((256, 64), stride=(64, 1)))
        b = thread.make_fragment_B(thread.partition_B(transposed_source))
        copy_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
        )
        copier_b = copy_b.get_slice(tid)
        sb, db = (copier_b.partition_S(transposed_source), copier_b.retile(b))
        for chunk in cutlass.range_constexpr(4):
            cute.copy(copy_a, sa[None, None, chunk], da[None, None, chunk])
            cute.copy(copy_a, sl[None, None, chunk], dl[None, None, chunk])
            cute.copy(copy_b, sb[None, None, chunk], db[None, None, chunk])
            cute.gemm(mma, result, a[None, None, chunk], b[None, None, chunk], result)
            cute.gemm(mma, low_result, low_a[None, None, chunk], b[None, None, chunk], low_result)

    @cute.kernel
    def kernel_values(self, q, k, v, maximum, logsum, dout, delta, base, dk, dv, scale, score_mma, gradient_mma):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, _ = block_indices(cute.ceil_div(k.shape[2], 32), q.shape[0] * q.shape[1])
        batch, head, start = (bh // q.shape[1], bh % q.shape[1], tile * 32)
        score_thread, gradient_thread = (score_mma.get_slice(tid), gradient_mma.get_slice(tid))
        allocator = SmemAllocator()
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        key_layout = cute.tile_to_shape(atom, (32, 256), (0, 1))
        query_layout = cute.tile_to_shape(atom, (64, 256), (0, 1))
        weight_atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        weight_layout = cute.tile_to_shape(weight_atom, (64, 32), (0, 1))
        assert 4 * cute.cosize(weight_layout) == cute.cosize(key_layout)
        keys = self.shared(allocator, q.element_type, key_layout)
        values = self.shared(allocator, q.element_type, key_layout)
        queries = self.shared(allocator, q.element_type, query_layout)
        gradients = self.shared(allocator, q.element_type, query_layout)
        # Four packed planes exactly fill the released 16 KiB V staging slot.
        derivatives = cute.make_tensor(values.iterator, weight_layout)
        probabilities = cute.make_tensor(values.iterator + cute.cosize(weight_layout), weight_layout)
        derivative_low = cute.make_tensor(values.iterator + 2 * cute.cosize(weight_layout), weight_layout)
        probability_low = cute.make_tensor(values.iterator + 3 * cute.cosize(weight_layout), weight_layout)
        self.load(k, keys, batch, head, start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        accum_k = cute.make_rmem_tensor(gradient_thread.partition_shape_C((32, 256)), cutlass.Float32)
        accum_v = cute.make_rmem_tensor(gradient_thread.partition_shape_C((32, 256)), cutlass.Float32)
        accum_k.fill(0.0)
        accum_v.fill(0.0)
        accum_key_low = cute.make_rmem_tensor(accum_k.shape, cutlass.Float32)
        accum_key_low.fill(0.0)
        accum_value_low = cute.make_rmem_tensor(accum_v.shape, cutlass.Float32)
        accum_value_low.fill(0.0)
        coords = score_thread.partition_C(cute.make_identity_tensor((64, 32)))
        rc = row_view(coords)
        magnitude = scale if scale >= 0.0 else -scale
        sign = cutlass.Float32(1.0) if scale >= 0.0 else cutlass.Float32(-1.0)
        first = start // 64 if cutlass.const_expr(self.causal) else 0
        for block in range(first, cute.ceil_div(q.shape[2], 64)):
            query = block * 64
            self.load(q, queries, batch, head, query, tid)
            self.load(dout, gradients, batch, head, query, tid)
            self.load(v, values, batch, head, start, tid)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            row_max = cute.make_rmem_tensor((cute.size(rc.shape[0]),), cutlass.Float32)
            row_logsum = cute.make_rmem_tensor(row_max.shape, cutlass.Float32)
            row_center = cute.make_rmem_tensor(row_max.shape, cutlass.Float32)
            row_residual = cute.make_rmem_tensor(row_max.shape, cutlass.Float32)
            for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                qr = query + rc[r, 0][0]
                row_max[r], row_logsum[r], row_center[r], row_residual[r] = (0.0, 0.0, 0.0, 0.0)
                if qr < q.shape[2]:
                    row_max[r] = maximum[batch, head, qr]
                    row_logsum[r] = logsum[batch, head, qr]
                    row_center[r] = base[batch, head, qr]
                    row_residual[r] = delta[batch, head, qr]
            scores = self.score.dot(queries, keys, score_mma, score_thread, tid)
            dp = self.score.dot(gradients, values, score_mma, score_thread, tid)
            sv, pv = (row_view(scores), row_view(dp))
            for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                for c in cutlass.range_constexpr(cute.size(rc.shape[1])):
                    row, col = rc[r, c]
                    visible = query + row < q.shape[2] and start + col < k.shape[2]
                    if cutlass.const_expr(self.causal):
                        visible = visible and start + col <= query + row
                    probability = cutlass.Float32(0.0)
                    if visible:
                        probability = cute.math.exp2(
                            (sv[r, c] * sign - row_max[r]) * magnitude * 1.4426950408889634 - row_logsum[r],
                            fastmath=True,
                        )
                    sv[r, c] = probability * (pv[r, c] - row_center[r] - row_residual[r])
                    pv[r, c] = probability
            rounded_scores = cute.make_fragment_like(scores, q.element_type)
            rounded_probabilities = cute.make_fragment_like(dp, q.element_type)
            packed_half_conversion(scores, rounded_scores)
            packed_half_conversion(dp, rounded_probabilities)
            residual = cute.make_fragment_like(scores)
            residual.store(scores.load() - rounded_scores.load().to(cutlass.Float32))
            rounded_residual = cute.make_fragment_like(rounded_scores)
            if cutlass.const_expr(self.scaled_key_residual and q.element_type == cutlass.Float16):
                residual.store(residual.load() * cutlass.Float32(1024.0))
            packed_half_conversion(residual, rounded_residual)
            probability_residual = cute.make_fragment_like(dp)
            probability_residual.store(dp.load() - rounded_probabilities.load().to(cutlass.Float32))
            # P is bounded by one: scaling its low component preserves tiny
            # representable gradients without overflowing the packed operand.
            if cutlass.const_expr(q.element_type == cutlass.Float16):
                probability_residual.store(probability_residual.load() * cutlass.Float32(16777216.0))
            rounded_probability_low = cute.make_fragment_like(rounded_probabilities)
            packed_half_conversion(probability_residual, rounded_probability_low)
            # Every V reader finishes before its staging slot becomes weights.
            cute.arch.sync_threads()
            self.store_packed_weights(rounded_scores, derivatives, coords)
            self.store_packed_weights(rounded_probabilities, probabilities, coords)
            self.store_packed_weights(rounded_residual, derivative_low, coords)
            self.store_packed_weights(rounded_probability_low, probability_low, coords)
            cute.arch.sync_threads()
            self.transposed_pair(
                derivatives, derivative_low, queries, gradient_mma, gradient_thread, tid, accum_k, accum_key_low
            )
            self.transposed_pair(
                probabilities, probability_low, gradients, gradient_mma, gradient_thread, tid, accum_v, accum_value_low
            )
            # Release all weight and Q/dO readers before the next copies.
            cute.arch.sync_threads()
        for component in cutlass.range_constexpr(cute.size(accum_k)):
            if cutlass.const_expr(self.scaled_key_residual and q.element_type == cutlass.Float16):
                accum_k[component] += accum_key_low[component] * cutlass.Float32(0.0009765625)
            else:
                accum_k[component] += accum_key_low[component]
        for component in cutlass.range_constexpr(cute.size(accum_v)):
            if cutlass.const_expr(q.element_type == cutlass.Float16):
                accum_v[component] += accum_value_low[component] * cutlass.Float32(5.960464477539063e-08)
            else:
                accum_v[component] += accum_value_low[component]
        self.store_result(accum_k, dk, keys, batch, head, start, 0, gradient_thread, tid, scale)
        self.store_result(accum_v, dv, keys, batch, head, start, 0, gradient_thread, tid, cutlass.Float32(1.0))
