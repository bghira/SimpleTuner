"""Key-axis parallel forward attention with FP32 partial outputs and softmax merging."""

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.memory import SmemAllocator

from .kernels import Attention, quad_max, quad_sum, row_view
from .launch import block_indices, launch_grid


class Forward(Attention):
    def __init__(self, columns=64, warps=1, splits=8, vector=8, staggered=False, rows=0):
        super().__init__("forward", False, False, columns=columns, width=64, warps=warps, vector=vector)
        if columns not in (32, 64, 128, 256) or warps not in (1, 2, 4, 8):
            raise ValueError("Split forward requires a supported warp-MMA tile.")
        if splits < 1:
            raise ValueError("Split forward requires a positive split count.")
        self.splits = splits
        self.staggered = staggered
        if rows:
            if rows != 128 or warps != 4:
                raise ValueError("A larger query tile requires four warps and 128 rows.")
            self.rows = self.grid_rows = rows

    guarded_load = Attention.load
    guarded_softmax = Attention.online_softmax

    @cute.jit
    def load(self, source, shared, batch, head, start, tid):
        if cutlass.const_expr(self.vector > 1 and self.staggered):
            rows, width = cute.size(shared, mode=[0]), cute.size(shared, mode=[1])
            if start + rows <= source.shape[2]:
                copy = cute.make_tiled_copy_tv(
                    cute.make_copy_atom(
                        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
                        source.element_type,
                        num_bits_per_copy=16 * self.vector,
                    ),
                    cute.make_layout(
                        (
                            32 * self.warps // max(32 // self.vector, 32 * self.warps // rows),
                            max(32 // self.vector, 32 * self.warps // rows),
                        ),
                        stride=(max(32 // self.vector, 32 * self.warps // rows), 1),
                    ),
                    cute.make_layout((1, self.vector)),
                )
                thread = copy.get_slice(tid)
                tile = cute.local_tile(source[batch, head, None, None], (rows, width), (start // rows, 0))
                src, dst = thread.partition_S(tile), thread.partition_D(shared)
                for r in cutlass.range_constexpr(cute.size(dst.shape[1])):
                    for d in cutlass.range_constexpr(cute.size(dst.shape[2])):
                        cute.copy(copy, src[None, r, d], dst[None, r, d])
            else:
                self.guarded_load(source, shared, batch, head, start, tid)
        else:
            self.guarded_load(source, shared, batch, head, start, tid)

    @cute.jit
    def online_softmax(self, scores, coordinates, accum, maxima, sums, mask, location, full, magnitude, sign):
        batch, head, start, key, sq, sk = location
        rc, av = row_view(coordinates), row_view(accum)
        scores.store(scores.load() * sign)
        sv = row_view(scores)
        if not full:
            for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                for c in cutlass.range_constexpr(cute.size(rc.shape[1])):
                    row, col = rc[r, c]
                    if start + row >= sq or key + col >= sk:
                        sv[r, c] = -cutlass.Float32.inf
        for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
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
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        mask: cute.Tensor,
        out: cute.Tensor,
        maximum: cute.Tensor,
        logsum: cute.Tensor,
        partial: cute.Tensor,
        stats: cute.Tensor,
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
            (self.warps, 1, 1),
            permutation_mnk=(self.rows, 16, 16),
        )
        self.kernel(q, k, v, partial, stats, scale, mma).launch(
            grid=launch_grid(cute.ceil_div(q.shape[2], self.rows), q.shape[0] * q.shape[1], self.splits),
            block=(32 * self.warps, 1, 1),
            stream=stream,
        )
        self.merge(partial, stats, out, maximum, logsum, scale).launch(
            grid=launch_grid(cute.ceil_div(q.shape[2], 2), q.shape[0] * q.shape[1]),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, q, k, v, partial, stats, scale, mma):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, split = block_indices(cute.ceil_div(q.shape[2], self.rows), q.shape[0] * q.shape[1])
        batch = bh // q.shape[1]
        head = bh % q.shape[1]
        kvhead = head // (q.shape[1] // k.shape[1])
        start = tile * self.rows
        thread = mma.get_slice(tid)
        coords = thread.partition_C(cute.make_identity_tensor((self.rows, self.columns)))
        rc = row_view(coords)
        accum = cute.make_rmem_tensor(thread.partition_shape_C((self.rows, 64)), cutlass.Float32)
        accum.fill(0.0)
        maxima = cute.make_rmem_tensor((cute.size(rc.shape[0]),), cutlass.Float32)
        maxima.fill(-cutlass.Float32.inf)
        sums = cute.make_rmem_tensor(maxima.shape, cutlass.Float32)
        sums.fill(0.0)
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        allocator = SmemAllocator()
        left = self.shared(allocator, q.element_type, cute.tile_to_shape(atom, (self.rows, 64), (0, 1)))
        right = self.shared(allocator, q.element_type, cute.tile_to_shape(atom, (self.columns, 64), (0, 1)))
        value = self.shared(allocator, q.element_type, cute.tile_to_shape(atom, (self.columns, 64), (0, 1)))
        self.load(q, left, batch, head, start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        blocks = cute.ceil_div(k.shape[2], self.columns)
        blocks_per_split = cute.ceil_div(blocks, self.splits)
        stop = cutlass.min(blocks, (split + 1) * blocks_per_split)
        magnitude = scale if scale >= 0.0 else -scale
        sign = cutlass.Float32(1.0) if scale >= 0.0 else cutlass.Float32(-1.0)
        first = split * blocks_per_split
        if cutlass.const_expr(self.staggered):
            if first < stop:
                self.load(k, right, batch, kvhead, first * self.columns, tid)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        for block in range(first, stop):
            if cutlass.const_expr(not self.staggered):
                self.load(k, right, batch, kvhead, block * self.columns, tid)
            self.load(v, value, batch, kvhead, block * self.columns, tid)
            cute.arch.cp_async_commit_group()
            if cutlass.const_expr(not self.staggered):
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
            scores = self.dot(left, right, mma, thread, tid)
            if cutlass.const_expr(self.staggered):
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                if block + 1 < stop:
                    self.load(k, right, batch, kvhead, (block + 1) * self.columns, tid)
                cute.arch.cp_async_commit_group()
            if cutlass.const_expr(self.staggered):
                self.online_softmax(
                    scores,
                    coords,
                    accum,
                    maxima,
                    sums,
                    q,
                    (batch, head, start, block * self.columns, q.shape[2], k.shape[2]),
                    start + self.rows <= q.shape[2] and (block + 1) * self.columns <= k.shape[2],
                    magnitude,
                    sign,
                )
            else:
                self.guarded_softmax(
                    scores,
                    coords,
                    accum,
                    maxima,
                    sums,
                    q,
                    (batch, head, start, block * self.columns, q.shape[2], k.shape[2]),
                    False,
                    magnitude,
                    sign,
                )
            self.product(scores, value, 0, mma, thread, tid, accum)
            if cutlass.const_expr(self.staggered):
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
            row = start + rc[r, 0][0]
            if tid % 4 == 0 and row < q.shape[2]:
                stats[batch, head, row, split, 0] = maxima[r]
                stats[batch, head, row, split, 1] = sums[r]
        out_coords = thread.partition_C(cute.make_identity_tensor((self.rows, 64)))
        for i in cutlass.range_constexpr(cute.size(out_coords)):
            row, dim = out_coords[i]
            if start + row < q.shape[2]:
                partial[batch, head, start + row, split, dim] = accum[i]

    @cute.kernel
    def merge(self, partial, stats, out, maximum_out, logsum_out, scale):
        tid, _, _ = cute.arch.thread_idx()
        tile, bh, _ = block_indices(cute.ceil_div(out.shape[2], 2), out.shape[0] * out.shape[1])
        dim = tid % 64
        row = tile * 2 + tid // 64
        head = bh % out.shape[1]
        batch = bh // out.shape[1]
        if row < out.shape[2]:
            maximum = -cutlass.Float32.inf
            for split in cutlass.range_constexpr(self.splits):
                maximum = cute.arch.fmax(maximum, stats[batch, head, row, split, 0])
            value = cutlass.Float32(0.0)
            denom = cutlass.Float32(0.0)
            magnitude = scale if scale >= 0.0 else -scale
            for split in cutlass.range_constexpr(self.splits):
                local_max = stats[batch, head, row, split, 0]
                alpha = cutlass.Float32(0.0)
                if local_max != -cutlass.Float32.inf:
                    alpha = cute.math.exp2((local_max - maximum) * magnitude * 1.4426950408889634, fastmath=True)
                denom += alpha * stats[batch, head, row, split, 1]
                value += alpha * partial[batch, head, row, split, dim]
            if denom > 0.0:
                value /= denom
            out[batch, head, row, dim] = value.to(out.element_type)
            if dim == 0:
                maximum_out[batch, head, row] = maximum
                logsum_out[batch, head, row] = cute.math.log2(denom if denom > 0.0 else cutlass.Float32(1.0), fastmath=True)
