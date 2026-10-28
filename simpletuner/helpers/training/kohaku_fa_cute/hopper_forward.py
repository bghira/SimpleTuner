"""Hopper attention with independent K/V staging and ordered softmax warpgroups."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warpgroup
from cutlass.utils import SmemAllocator

from .kernels import Attention, quad_max, quad_sum, row_view


class HopperForwardSpecialized(Attention):
    def __init__(self, mode, causal, masked, *configuration, vector, producer_registers=32):
        super().__init__(mode, causal, masked, *configuration, vector=vector)
        self.loader = Attention(mode, causal, masked, *configuration, vector=vector)
        self.loader.warps = 8
        self.grid_rows = 128
        self.warps = 12
        self.columns = 128
        self.producer_registers = producer_registers
        self.specialized_load = True
        self.loader.wide_row_copies = True

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
        padded = cute.ceil_div(q.shape[3], self.width) * self.width
        atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, q.element_type)
        right_layout = cute.tile_to_shape(atom, (self.columns, padded), (0, 1))
        tma_k, tensor_k = self.create_tma(k, right_layout, padded)
        tma_v, tensor_v = self.create_tma(v, right_layout, padded)
        self.kernel(
            q,
            k,
            v,
            mask,
            out,
            maximum,
            logsum,
            dout,
            delta,
            dq,
            dk,
            dv,
            blocks,
            delta_base,
            scale,
            mma,
            (tma_k, tensor_k, tma_v, tensor_v),
        ).launch(
            grid=(
                q.shape[0] * q.shape[1],
                cute.ceil_div(q.shape[2], self.grid_rows),
                cute.ceil_div(q.shape[3], self.width),
            ),
            block=(32 * self.warps, 1, 1),
            stream=stream,
        )

    @cute.jit
    def online_softmax(
        self,
        scores,
        coordinates,
        accum,
        maxima,
        sums,
        mask,
        location,
        full,
        magnitude,
        sign,
        folded: cutlass.Constexpr,
        positive: cutlass.Constexpr,
    ):
        if cutlass.const_expr(folded):
            magnitude = cutlass.Float32(0.0625)
        batch, head, start, key, sq, sk = location
        rc, av = row_view(coordinates), row_view(accum)
        if cutlass.const_expr(not positive):
            if sign < 0.0:
                scores.store(-scores.load())
        sv = row_view(scores)
        for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
            # Uniform full tiles need no per-score visibility checks.
            if not full or key + self.columns > sk:
                for c in cutlass.range_constexpr(cute.size(rc.shape[1])):
                    row, col = rc[r, c]
                    if not self.visible(mask, batch, head, start + row, key + col, sq, sk, full):
                        sv[r, c] = -cutlass.Float32.inf
            current = quad_max(sv[r, None].load().reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0))
            new = cute.arch.fmax(maxima[r], current)
            alpha = cutlass.Float32(0.0)
            if maxima[r] != -cutlass.Float32.inf:
                alpha = self.probability(maxima[r] - new, magnitude, folded)
            if magnitude > 0.0 and new > -cutlass.Float32.inf and new < cutlass.Float32.inf:
                for c in cutlass.range_constexpr(cute.size(rc.shape[1])):
                    sv[r, c] = self.probability(sv[r, c] - new, magnitude, folded)
            else:
                for c in cutlass.range_constexpr(cute.size(rc.shape[1])):
                    probability = cutlass.Float32(0.0)
                    if sv[r, c] != -cutlass.Float32.inf:
                        probability = self.probability(sv[r, c] - new, magnitude, folded)
                    sv[r, c] = probability
            # Keep denominator partials local until the final quad reduction.
            sums[r] = sums[r] * alpha + sv[r, None].load().reduce(cute.ReductionOp.ADD, 0.0, 0)
            maxima[r] = new
            av[r, None] = av[r, None].load() * alpha

    @cute.jit
    def issue_single(self, index: cutlass.Constexpr, shared, batch, head, start, tid, args, barrier):
        if tid < 32:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, 2 * cute.size(shared))
            self.load_tma(
                args[2 * index],
                args[2 * index + 1],
                shared,
                batch,
                head,
                start,
                barrier,
            )

    @cute.kernel
    def kernel(
        self,
        q,
        k,
        v,
        mask,
        out,
        maximum,
        logsum,
        dout,
        delta,
        dq,
        dk,
        dv,
        blocks,
        delta_base,
        scale,
        mma,
        tma_args,
    ):
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        bh, tile, width = cute.arch.block_idx()
        tile = cute.ceil_div(q.shape[2], self.grid_rows) - 1 - tile
        batch = bh // q.shape[1]
        head = bh % q.shape[1]
        kvhead = head // (q.shape[1] // k.shape[1])
        group_start = tile * self.grid_rows
        offset = width * self.width
        padded = cute.ceil_div(q.shape[3], self.width) * self.width
        atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, q.element_type)
        allocator = SmemAllocator()
        ready = allocator.allocate_array(cutlass.Int64, 2)
        free = allocator.allocate_array(cutlass.Int64, 2)
        math_ready = allocator.allocate_array(cutlass.Int64, 2)
        if tid == 0:
            # Every warp must finish before a peer can reuse the parity token.
            cute.arch.mbarrier_init(math_ready, 4)
            cute.arch.mbarrier_init(math_ready + 1, 4)
            for slot in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(ready + slot, 1)
                cute.arch.mbarrier_init(free + slot, 8)
        cute.arch.mbarrier_init_fence()
        left = self.shared(
            allocator,
            q.element_type,
            cute.tile_to_shape(atom, (self.grid_rows, padded), (0, 1)),
        )
        right = self.shared(
            allocator,
            q.element_type,
            cute.tile_to_shape(atom, (self.columns, padded), (0, 1)),
        )
        other_right = self.shared(
            allocator,
            q.element_type,
            cute.tile_to_shape(atom, (self.columns, padded), (0, 1)),
        )
        if tid < 256:
            self.loader.load(q, left, batch, head, group_start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        limit = cutlass.min(k.shape[2], group_start + self.grid_rows) if cutlass.const_expr(self.causal) else k.shape[2]
        count = cute.ceil_div(limit, self.columns)
        # K and V have independent producer warps and release phases.
        if tid >= 256:
            cute.arch.setmaxregister_decrease(self.producer_registers)
            if tid < 288:
                self.issue_single(0, right, batch, kvhead, 0, tid - 256, tma_args, ready)
                for block in range(count):
                    cute.arch.mbarrier_wait(free, block % 2)
                    if block + 1 < count:
                        self.issue_single(
                            0,
                            right,
                            batch,
                            kvhead,
                            (block + 1) * self.columns,
                            tid - 256,
                            tma_args,
                            ready,
                        )
            elif tid < 320:
                for block in range(count):
                    if block > 0:
                        cute.arch.mbarrier_wait(free + 1, (block - 1) % 2)
                    self.issue_single(
                        1,
                        other_right,
                        batch,
                        kvhead,
                        block * self.columns,
                        tid - 288,
                        tma_args,
                        ready + 1,
                    )
        else:
            cute.arch.setmaxregister_increase(232)
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
            accum = cute.make_rmem_tensor(
                output_thread.partition_shape_C((self.rows, self.product_width)),
                cutlass.Float32,
            )
            accum.fill(0.0)
            av = row_view(accum)
            magnitude = scale if scale >= 0.0 else -scale
            sign = cutlass.Float32(1.0) if scale >= 0.0 else cutlass.Float32(-1.0)
            maxima = cute.make_rmem_tensor((cute.size(rc.shape[0]),), cutlass.Float32)
            sums = cute.make_rmem_tensor(maxima.shape, cutlass.Float32)
            maxima.fill(-cutlass.Float32.inf)
            sums.fill(0.0)
            if magnitude == 0.0625 and sign > 0.0:
                self.compute_loop(
                    q,
                    k,
                    mask,
                    compute_left,
                    right,
                    other_right,
                    mma,
                    thread,
                    tid,
                    accum,
                    coords,
                    maxima,
                    sums,
                    start,
                    batch,
                    head,
                    count,
                    ready,
                    free,
                    math_ready,
                    offset,
                    magnitude,
                    sign,
                    True,
                    True,
                )
            elif sign > 0.0:
                self.compute_loop(
                    q,
                    k,
                    mask,
                    compute_left,
                    right,
                    other_right,
                    mma,
                    thread,
                    tid,
                    accum,
                    coords,
                    maxima,
                    sums,
                    start,
                    batch,
                    head,
                    count,
                    ready,
                    free,
                    math_ready,
                    offset,
                    magnitude,
                    sign,
                    False,
                    True,
                )
            else:
                self.compute_loop(
                    q,
                    k,
                    mask,
                    compute_left,
                    right,
                    other_right,
                    mma,
                    thread,
                    tid,
                    accum,
                    coords,
                    maxima,
                    sums,
                    start,
                    batch,
                    head,
                    count,
                    ready,
                    free,
                    math_ready,
                    offset,
                    magnitude,
                    sign,
                    False,
                    False,
                )
            for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
                total = quad_sum(sums[r])
                denom = total if total > 0.0 else cutlass.Float32(1.0)
                # Share the FP32 reciprocal and correct each quotient with
                # an FMA residual before converting the result to BF16.
                inverse = cutlass.Float32(1.0) / denom
                for c in cutlass.range_constexpr(cute.size(av.shape[1])):
                    value = av[r, c]
                    quotient = value * inverse
                    residual = cute.math.fma(-quotient, denom, value)
                    corrected = cute.math.fma(residual, inverse, quotient)
                    # The residual is NaN for infinite numerators. Preserve the
                    # original division's infinity and NaN propagation.
                    av[r, c] = corrected if quotient < cutlass.Float32.inf and quotient > -cutlass.Float32.inf else quotient
                row = start + rc[r, 0][0]
                if lane % 4 == 0 and row < q.shape[2] and width == 0:
                    maximum[batch, head, row] = maxima[r]
                    logsum[batch, head, row] = cute.math.log2(denom, fastmath=True)
            # Empty consumer groups must not form an output tile: singleton
            # sequence layouts can collapse the identity tensor's row mode.
            if start < q.shape[2]:
                self.store_result(
                    accum,
                    out,
                    compute_left,
                    batch,
                    head,
                    start,
                    offset,
                    output_thread,
                    tid,
                    cutlass.Float32(1.0),
                )

        cute.arch.sync_threads()

    @cute.jit
    def compute_loop(
        self,
        q,
        k,
        mask,
        compute_left,
        right,
        other_right,
        mma,
        thread,
        tid,
        accum,
        coords,
        maxima,
        sums,
        start,
        batch,
        head,
        count,
        ready,
        free,
        math_ready,
        offset,
        magnitude,
        sign,
        folded: cutlass.Constexpr,
        positive: cutlass.Constexpr,
    ):
        for block in range(count):
            cute.arch.mbarrier_wait(ready, block % 2)
            scores = self.dot(compute_left, right, mma, thread, tid, wait=False)
            warpgroup.wait_group(0)
            if tid % 32 == 0:
                cute.arch.mbarrier_arrive(free)
            # Alternate softmax sections so the other warpgroup can issue MMA.
            if tid < 128:
                if block > 0:
                    cute.arch.mbarrier_wait(math_ready, (block - 1) % 2)
            else:
                cute.arch.mbarrier_wait(math_ready + 1, block % 2)
            self.online_softmax(
                scores,
                coords,
                accum,
                maxima,
                sums,
                mask,
                (batch, head, start, block * self.columns, q.shape[2], k.shape[2]),
                block * self.columns + self.columns <= start + 1,
                magnitude,
                sign,
                folded,
                positive,
            )
            if tid % 32 == 0:
                cute.arch.mbarrier_arrive(math_ready + 1 - tid // 128)
            cute.arch.mbarrier_wait(ready + 1, block % 2)
            self.product(scores, other_right, offset, mma, thread, tid, accum)

            if tid % 32 == 0:
                cute.arch.mbarrier_arrive(free + 1)

    @cute.jit
    def probability(self, difference, magnitude, folded: cutlass.Constexpr):
        if cutlass.const_expr(folded):
            return cute.math.exp2(difference * 0.09016844005556021, fastmath=True)
        else:
            return cute.math.exp2(difference * magnitude * 1.4426950408889634, fastmath=True)
