"""Full-width D512 backward with independent low-component key and value sums."""

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import warp
from cutlass.memory import SmemAllocator

from .kernels import Attention, packed_half_conversion
from .launch import block_indices, launch_grid


class WideKeyValue(Attention):
    """Use 16 key rows, 32 query columns and 99 KiB of shared memory."""

    def __init__(self, causal, *, vector, copy_cache_global, packed_bf16_causal=False):
        super().__init__(
            "kv",
            causal,
            False,
            columns=32,
            width=512,
            warps=8,
            vector=vector,
            centered=True,
            kv_column_warps=8,
            copy_cache_global=copy_cache_global,
        )
        self.rows = self.grid_rows = 16
        self.product_width = 512
        self.baseline_precision = False
        self.compensated_keys = True
        self.scaled_key_residual = True
        self.scaled_value_residual = True
        self.packed_bf16_causal = packed_bf16_causal
        self.compensated_values = True
        self.score = Attention(
            "kv",
            causal,
            False,
            columns=32,
            width=512,
            warps=4,
            vector=vector,
            centered=True,
            narrow_n=True,
            kv_column_warps=4,
            kv_column_statistics=True,
        )
        self.score.rows = self.score.grid_rows = 16
        self.score.split_kv = False
        self.score.score_warps = self.score.score_column_warps = 4
        self.score.score_b_matrices = 2
        self.score.baseline_precision = False

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        assert q.shape[3] == 512 and k.shape[3] == 512
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)), (1, 8, 1), permutation_mnk=(16, 128, 16)
        )
        smma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)), (1, 4, 1), permutation_mnk=(16, 32, 16)
        )
        self.kernel(q, k, v, mask, maximum, logsum, dout, delta, dk, dv, blocks, base, scale, mma, smma).launch(
            grid=launch_grid(cute.ceil_div(k.shape[2], 16), q.shape[0] * k.shape[1], 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, q, k, v, mask, maximum, logsum, dout, delta, dk, dv, blocks, base, scale, mma, smma):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, _ = block_indices(cute.ceil_div(k.shape[2], 16), q.shape[0] * k.shape[1])
        batch, kvhead = bh // k.shape[1], bh % k.shape[1]
        start = tile * 16
        thread = mma.get_slice(tid)
        st = smma.get_slice(tid % 128)
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        kl = cute.tile_to_shape(atom, (16, 512), (0, 1))
        ql = cute.tile_to_shape(atom, (32, 512), (0, 1))
        wl = cute.tile_to_shape(atom, (16, 32), (0, 1))
        alloc = SmemAllocator()
        # K/V are 16 KiB each, Q/dO are 32 KiB each. Three 1 KiB
        # weight planes bring the instantiated dynamic total to 101376 bytes.
        left = self.shared(alloc, q.element_type, kl)
        other_left = self.shared(alloc, q.element_type, kl)
        right = self.shared(alloc, q.element_type, ql)
        other_right = self.shared(alloc, q.element_type, ql)
        ds = self.shared(alloc, q.element_type, wl)
        p = self.shared(alloc, q.element_type, wl)
        low = self.shared(alloc, q.element_type, wl)
        hi_acc = cute.make_rmem_tensor(thread.partition_shape_C((16, 512)), cutlass.Float32)
        lo_acc = cute.make_rmem_tensor(hi_acc.shape, cutlass.Float32)
        v_acc = cute.make_rmem_tensor(hi_acc.shape, cutlass.Float32)
        assert cute.size(hi_acc) == 32
        hi_acc.fill(0.0)
        lo_acc.fill(0.0)
        v_acc.fill(0.0)
        if cutlass.const_expr(self.compensated_values):
            v_low = cute.make_rmem_tensor(hi_acc.shape, cutlass.Float32)
            v_low.fill(0.0)
            probability_low = cute.make_rmem_tensor(st.partition_shape_C((16, 32)), cutlass.Float32)
        self.load(k, left, batch, kvhead, start, tid)
        self.load(v, other_left, batch, kvhead, start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        magnitude = scale if scale >= 0.0 else -scale
        sign = cutlass.Float32(1.0) if scale >= 0.0 else cutlass.Float32(-1.0)
        for h in range(q.shape[1] // k.shape[1]):
            head = kvhead * (q.shape[1] // k.shape[1]) + h
            first = start // 32 if cutlass.const_expr(self.causal) else 0
            for block in range(first, cute.ceil_div(q.shape[2], 32)):
                self.load(q, right, batch, head, block * 32, tid)
                self.load(dout, other_right, batch, head, block * 32, tid)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                if tid < 128:
                    scores, prob = self.score.kv_weights(
                        left,
                        right,
                        other_left,
                        other_right,
                        mask,
                        (maximum, logsum, delta, base),
                        (batch, head, start, block * 32, q.shape[2], k.shape[2]),
                        False,
                        magnitude,
                        sign,
                        smma,
                        st,
                        tid,
                    )
                    coords = st.partition_C(cute.make_identity_tensor((16, 32)))
                    assert cute.size(coords) == 4
                    if cutlass.const_expr(self.packed_bf16_causal and self.causal and q.element_type == cutlass.BFloat16):
                        rounded_scores = cute.make_fragment_like(scores, q.element_type)
                        rounded_probabilities = cute.make_fragment_like(prob, q.element_type)
                        packed_half_conversion(scores, rounded_scores)
                        packed_half_conversion(prob, rounded_probabilities)
                        self.score.store_packed_weights(rounded_scores, ds, coords)
                        self.score.store_packed_weights(rounded_probabilities, p, coords)
                        probability_low.store(prob.load() - rounded_probabilities.load().to(cutlass.Float32))
                        residual = cute.make_fragment_like(scores)
                        residual.store(scores.load() - rounded_scores.load().to(cutlass.Float32))
                        rounded_residual = cute.make_fragment_like(rounded_scores)
                        packed_half_conversion(residual, rounded_residual)
                        self.score.store_packed_weights(rounded_residual, low, coords)
                    else:
                        for i in cutlass.range_constexpr(cute.size(coords)):
                            r, c = coords[i]
                            rounded = scores[i].to(q.element_type)
                            ds[r, c] = rounded
                            p[r, c] = prob[i].to(q.element_type)
                            if cutlass.const_expr(self.compensated_values):
                                probability_low[i] = prob[i] - prob[i].to(q.element_type).to(cutlass.Float32)
                            residual = scores[i] - rounded.to(cutlass.Float32)
                            if cutlass.const_expr(self.scaled_key_residual and q.element_type == cutlass.Float16):
                                # Preserve small FP16 residuals without changing their significand.
                                residual *= 1024.0
                            low[r, c] = residual.to(q.element_type)
                cute.arch.sync_threads()
                self.dual_product(ds, low, right, 0, mma, thread, tid, hi_acc, lo_acc)
                if cutlass.const_expr(not self.compensated_values or not self.causal and q.element_type == cutlass.Float16):
                    self.product(p, other_right, 0, mma, thread, tid, v_acc)
                # Release DS_low only after every gradient warp has consumed it.
                cute.arch.sync_threads()
                if cutlass.const_expr(self.compensated_values):
                    if tid < 128:
                        coords = st.partition_C(cute.make_identity_tensor((16, 32)))
                        if cutlass.const_expr(
                            self.packed_bf16_causal and self.causal and q.element_type == cutlass.BFloat16
                        ):
                            rounded_values = cute.make_fragment_like(probability_low, q.element_type)
                            packed_half_conversion(probability_low, rounded_values)
                            self.score.store_packed_weights(rounded_values, low, coords)
                        else:
                            for i in cutlass.range_constexpr(cute.size(coords)):
                                r, c = coords[i]
                                value_residual = probability_low[i]
                                if cutlass.const_expr(self.scaled_value_residual and q.element_type == cutlass.Float16):
                                    # P is bounded by one: its half residual times 2**24
                                    # stays in range while retaining subnormal-sized weights.
                                    value_residual *= 16777216.0
                                low[r, c] = value_residual.to(q.element_type)
                    # Publish P_low into the released DS_low plane.
                    cute.arch.sync_threads()
                    if cutlass.const_expr(self.causal or q.element_type == cutlass.BFloat16):
                        self.dual_product(p, low, other_right, 0, mma, thread, tid, v_acc, v_low)
                    else:
                        self.product(low, other_right, 0, mma, thread, tid, v_low)
                    # Release all query/dO and weight readers before reuse.
                    cute.arch.sync_threads()
        for i in cutlass.range_constexpr(cute.size(hi_acc)):
            if cutlass.const_expr(self.scaled_key_residual and q.element_type == cutlass.Float16):
                hi_acc[i] += lo_acc[i] * 0.0009765625
            else:
                hi_acc[i] += lo_acc[i]
        self.store_result(hi_acc, dk, left, batch, kvhead, start, 0, thread, tid, scale)
        if cutlass.const_expr(self.compensated_values):
            for i in cutlass.range_constexpr(cute.size(v_acc)):
                if cutlass.const_expr(self.scaled_value_residual and q.element_type == cutlass.Float16):
                    v_acc[i] += v_low[i] * 5.960464477539063e-8
                else:
                    v_acc[i] += v_low[i]
        self.store_result(v_acc, dv, left, batch, kvhead, start, 0, thread, tid, cutlass.Float32(1.0))
