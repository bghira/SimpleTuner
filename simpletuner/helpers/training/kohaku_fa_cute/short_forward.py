"""Full key-tile forward attention with aliased QK and PV staging.

Retain FP32 logits for a full key tile while streaming Q/K feature chunks.
Once QK finishes, reuse its staging area for P and double-buffered V.
Stream output features through a smaller accumulator instead of keeping
the entire head dimension live across repeated softmax updates.
"""

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import warp
from cutlass.memory import SmemAllocator

from .kernels import Attention, row_view
from .launch import block_indices, launch_grid


class ShortForward(Attention):
    def __init__(self, causal, dimension, *, rows=16, chunk=64, output_width=64, vector=8, columns=128):
        if (
            dimension not in (128, 256, 512)
            or rows not in (16, 32, 64)
            or chunk not in (32, 64, 128)
            or output_width not in (32, 64, 128)
            or columns not in (64, 128)
            or rows * output_width < 1024
            or dimension < 2 * max(chunk, output_width)
        ):
            raise ValueError("Short forward requires supported tiles and at least two input/output feature stages.")
        super().__init__(
            "forward", causal, False, columns=columns, width=output_width, warps=4, vector=vector, copy_cache_global=True
        )
        self.rows = self.grid_rows = rows
        self.column_warps = 64 // rows
        self.dimension = dimension
        self.chunk = chunk

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
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
            (self.warps // self.column_warps, self.column_warps, 1),
            permutation_mnk=(self.rows, 16 * self.column_warps, 16),
        )
        self.short_kernel(q, k, v, out, maximum, logsum, scale, mma).launch(
            grid=launch_grid(cute.ceil_div(q.shape[2], self.rows), q.shape[0] * q.shape[1]),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.jit
    def feature_view(self, source, offset, width: cutlass.Constexpr):
        return cute.make_tensor(
            source.iterator + offset,
            cute.make_layout((source.shape[0], source.shape[1], source.shape[2], width), stride=source.stride),
        )

    @cute.jit
    def dot_accumulate(self, a, b, mma, thread, tid, result):
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
        for chunk in cutlass.range_constexpr(self.chunk // 16):
            if chunk + 1 < self.chunk // 16:
                cute.copy(ca, sa[None, None, chunk + 1], da[None, None, (chunk + 1) % 2])
                cute.copy(cb, sb[None, None, chunk + 1], db[None, None, (chunk + 1) % 2])
            cute.gemm(mma, result, ra[None, None, chunk % 2], rb[None, None, chunk % 2], result)

    @cute.kernel
    def short_kernel(self, q, k, v, out, maximum, logsum, scale, mma):
        self.compute_tile(q, k, v, out, maximum, logsum, scale, mma)

    @cute.jit
    def compute_tile(self, q, k, v, out, maximum, logsum, scale, mma, storage=None):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, _ = block_indices(cute.ceil_div(q.shape[2], self.rows), q.shape[0] * q.shape[1])
        batch, head = bh // q.shape[1], bh % q.shape[1]
        kvhead = head // (q.shape[1] // k.shape[1])
        start = tile * self.rows
        thread = mma.get_slice(tid)
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        q_layout = cute.tile_to_shape(atom, (self.rows, self.chunk, 2), (0, 1, 2))
        k_layout = cute.tile_to_shape(atom, (self.columns, self.chunk, 2), (0, 1, 2))
        p_layout = cute.tile_to_shape(atom, (self.rows, self.columns), (0, 1))
        v_layout = cute.tile_to_shape(atom, (self.columns, self.width, 2), (0, 1, 2))
        qk_size = cute.cosize(q_layout) + cute.cosize(k_layout)
        pv_size = cute.cosize(p_layout) + cute.cosize(v_layout)
        allocator = SmemAllocator()
        if cutlass.const_expr(storage is None):
            storage = allocator.allocate_array(q.element_type, max(qk_size, pv_size), byte_alignment=128)
        qs = cute.make_tensor(storage, q_layout)
        ks = cute.make_tensor(storage + cute.cosize(q_layout), k_layout)
        weights = cute.make_tensor(storage, p_layout)
        vs = cute.make_tensor(storage + cute.cosize(p_layout), v_layout)
        stats = None
        if cutlass.const_expr(self.column_warps > 1):
            stats = allocator.allocate_tensor(
                cutlass.Float32,
                cute.make_layout((2, self.column_warps, self.rows), stride=(self.column_warps * self.rows, self.rows, 1)),
                byte_alignment=128,
            )
        scores = cute.make_rmem_tensor(thread.partition_shape_C((self.rows, self.columns)), cutlass.Float32)
        scores.fill(0.0)
        # Keep one future Q/K feature chunk in flight during the current MMA.
        for stage in cutlass.range_constexpr(2):
            self.load(self.feature_view(q, stage * self.chunk, self.chunk), qs[None, None, stage], batch, head, start, tid)
            self.load(self.feature_view(k, stage * self.chunk, self.chunk), ks[None, None, stage], batch, kvhead, 0, tid)
            cute.arch.cp_async_commit_group()
        for feature in cutlass.range_constexpr(self.dimension // self.chunk):
            slot = feature % 2
            if cutlass.const_expr(feature + 1 < self.dimension // self.chunk):
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            self.dot_accumulate(qs[None, None, slot], ks[None, None, slot], mma, thread, tid, scores)
            cute.arch.sync_threads()
            if cutlass.const_expr(feature + 2 < self.dimension // self.chunk):
                self.load(
                    self.feature_view(q, (feature + 2) * self.chunk, self.chunk),
                    qs[None, None, slot],
                    batch,
                    head,
                    start,
                    tid,
                )
                self.load(
                    self.feature_view(k, (feature + 2) * self.chunk, self.chunk), ks[None, None, slot], batch, kvhead, 0, tid
                )
                cute.arch.cp_async_commit_group()
        # QK's final barrier releases the aliased buffers. V transfers can now
        # overlap the full-key softmax, whose scores remain in registers.
        for stage in cutlass.range_constexpr(2):
            self.load(self.feature_view(v, stage * self.width, self.width), vs[None, None, stage], batch, kvhead, 0, tid)
            cute.arch.cp_async_commit_group()
        coords = thread.partition_C(cute.make_identity_tensor((self.rows, self.columns)))
        rc = row_view(coords)
        maxima = cute.make_rmem_tensor((cute.size(rc.shape[0]),), cutlass.Float32)
        sums = cute.make_rmem_tensor(maxima.shape, cutlass.Float32)
        maxima.fill(-cutlass.Float32.inf)
        sums.fill(0.0)
        accum = cute.make_rmem_tensor(thread.partition_shape_C((self.rows, self.width)), cutlass.Float32)
        accum.fill(0.0)
        magnitude = scale if scale >= 0.0 else -scale
        sign = cutlass.Float32(1.0) if scale >= 0.0 else cutlass.Float32(-1.0)
        if cutlass.const_expr(self.column_warps == 1):
            self.online_softmax(
                scores,
                coords,
                accum,
                maxima,
                sums,
                out,
                (batch, head, start, 0, q.shape[2], k.shape[2]),
                False,
                magnitude,
                sign,
            )
            for index in cutlass.range_constexpr(cute.size(coords)):
                row, col = coords[index]
                weights[row, col] = scores[index].to(weights.element_type)
        else:
            self.column_softmax_max(
                scores, coords, stats, out, (batch, head, start, 0, q.shape[2], k.shape[2]), False, sign, tid
            )
            cute.arch.sync_threads()
            self.column_softmax_probabilities(scores, coords, accum, maxima, sums, stats, weights, magnitude, tid)
            cute.arch.sync_threads()
            self.column_softmax_sum(coords, sums, stats)
        # Reuse one FP32 reciprocal across every output feature tile.
        inverse = cute.make_rmem_tensor(sums.shape, cutlass.Float32)
        for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
            denominator = sums[r] if sums[r] > 0.0 else cutlass.Float32(1.0)
            inverse[r] = cutlass.Float32(1.0) / denominator
        for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
            row = start + rc[r, 0][0]
            if tid < 32 * (self.warps // self.column_warps) and tid % 4 == 0 and row < q.shape[2]:
                maximum[batch, head, row] = maxima[r]
                denom = sums[r] if sums[r] > 0.0 else cutlass.Float32(1.0)
                logsum[batch, head, row] = cute.math.log2(denom, fastmath=True)
        for feature in cutlass.range_constexpr(self.dimension // self.width):
            slot = feature % 2
            if cutlass.const_expr(feature + 1 < self.dimension // self.width):
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            accum.fill(0.0)
            self.product(weights, vs[None, None, slot], 0, mma, thread, tid, accum, shared_weights=True)
            # Every warp must release V before its buffer becomes output staging.
            cute.arch.sync_threads()
            av = row_view(accum)
            for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                av[r, None] = av[r, None].load() * inverse[r]
            self.store_result(
                accum, out, vs[None, None, slot], batch, head, start, feature * self.width, thread, tid, cutlass.Float32(1.0)
            )
            if cutlass.const_expr(feature + 2 < self.dimension // self.width):
                self.load(
                    self.feature_view(v, (feature + 2) * self.width, self.width), vs[None, None, slot], batch, kvhead, 0, tid
                )
                cute.arch.cp_async_commit_group()


class CausalPrefixForward(ShortForward):
    """Skip the invisible upper-right key tile within a uniform CTA branch."""

    def __init__(self, dimension, vector=8):
        super().__init__(True, dimension, rows=64, chunk=64, output_width=64, vector=vector)
        self.prefix = ShortForward(True, dimension, rows=64, chunk=64, output_width=64, vector=vector, columns=64)

    @cute.kernel
    def short_kernel(self, q, k, v, out, maximum, logsum, scale, mma):
        tile, _, _ = block_indices(cute.ceil_div(q.shape[2], self.rows), q.shape[0] * q.shape[1])
        qk_size = 2 * (self.rows + self.columns) * self.chunk
        pv_size = self.rows * self.columns + 2 * self.columns * self.width
        # The CTA branches share one allocation to preserve two-block residency.
        storage = SmemAllocator().allocate_array(q.element_type, max(qk_size, pv_size), byte_alignment=128)
        if tile == 0:
            self.prefix.compute_tile(q, k, v, out, maximum, logsum, scale, mma, storage)
        else:
            self.compute_tile(q, k, v, out, maximum, logsum, scale, mma, storage)


class AdaptiveShortForward:
    """Choose full-key staging at runtime so cached programs preserve length dispatch."""

    def __init__(self, fallback, dimension, causal, vector, batch_heads=1):
        small_width = 64 if dimension == 256 else 128
        self.short = ShortForward(causal, dimension, rows=16, chunk=64, output_width=small_width, vector=vector)
        self.parallel = self.short
        self.adaptive = batch_heads > 4
        if self.adaptive:
            if dimension == 256 and batch_heads > 24:
                if causal and batch_heads <= 32:
                    self.parallel = CausalPrefixForward(dimension, vector)
                else:
                    # 32-column output staging admits three CTAs per SM.
                    self.parallel = ShortForward(causal, dimension, rows=64, chunk=32, output_width=32, vector=vector)
            else:
                # Two 64-row query tiles per head fill one 48-SM GB10 wave
                # through 24 heads. Wider batches favor narrower staging.
                rows = 64 if dimension == 256 or batch_heads <= 16 else 32
                width = 128 if dimension == 512 and batch_heads > 32 else 64
                self.parallel = ShortForward(causal, dimension, rows=rows, chunk=64, output_width=width, vector=vector)
        self.fallback = fallback

    @cute.jit
    def __call__(
        self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream: cuda.CUstream
    ):
        if q.shape[2] <= 128 and k.shape[2] <= 128:
            if cutlass.const_expr(self.adaptive):
                if q.shape[2] <= 64:
                    self.short(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
                else:
                    self.parallel(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
            else:
                self.short(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
        else:
            self.fallback(q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, base, scale, stream)
