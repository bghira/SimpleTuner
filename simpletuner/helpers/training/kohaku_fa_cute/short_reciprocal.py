"""Bounded D128 forward owner with one FP32 reciprocal per query row."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.memory import SmemAllocator

from .kernels import Attention, row_view
from .short_key_quad_skip_owned import compile_kernel as compile_backward


class ReciprocalForward(Attention):
    def __init__(self, causal):
        super().__init__("forward", causal, False, columns=64, width=128, warps=4, vector=8)

    @cute.kernel
    def kernel(self, q, k, v, mask, out, maximum, logsum, dout, delta, dq, dk, dv, blocks, delta_base, scale, mma, tma_args):
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        tile, bh, width = self.grid_coordinates(q, k)
        thread = mma.get_slice(tid)
        batch, head = bh // q.shape[1], bh % q.shape[1]
        kvhead = head // (q.shape[1] // k.shape[1])
        start, offset = tile * self.grid_rows, width * self.width
        magnitude = scale if scale >= 0.0 else -scale
        sign = cutlass.Float32(1.0) if scale >= 0.0 else cutlass.Float32(-1.0)
        coordinates = thread.partition_C(cute.make_identity_tensor((self.rows, self.columns)))
        rc = row_view(coordinates)
        accum = cute.make_rmem_tensor(thread.partition_shape_C((self.rows, self.product_width)), cutlass.Float32)
        accum.fill(0.0)
        av = row_view(accum)
        padded = cute.ceil_div(q.shape[3], self.width) * self.width
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        left_layout = cute.tile_to_shape(atom, (self.grid_rows, padded), (0, 1))
        right_layout = cute.tile_to_shape(atom, (self.columns, padded), (0, 1))
        allocator = SmemAllocator()
        left = self.shared(allocator, q.element_type, left_layout)
        right = self.shared(allocator, q.element_type, right_layout)
        other_right = self.shared(allocator, q.element_type, right_layout)
        self.load(q, left, batch, head, start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        maxima = cute.make_rmem_tensor((cute.size(rc.shape[0]),), cutlass.Float32)
        sums = cute.make_rmem_tensor(maxima.shape, cutlass.Float32)
        maxima.fill(-cutlass.Float32.inf)
        sums.fill(0.0)
        limit = cutlass.min(k.shape[2], start + self.rows) if cutlass.const_expr(self.causal) else k.shape[2]
        count = cute.ceil_div(limit, self.columns)
        phase = cutlass.Int32(0)
        for block in range(count):
            active, full = self.block_visibility(blocks, batch, head, start, block * self.columns, self.rows, self.columns)
            if active:
                phase = self.load_pair(
                    k, v, right, other_right, batch, kvhead, block * self.columns, tid, tma_args, None, phase
                )
                scores = self.dot(left, right, mma, thread, tid)
                self.online_softmax(
                    scores,
                    coordinates,
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
            if active:
                cute.arch.sync_threads()
        for r in cutlass.range_constexpr(cute.size(rc.shape[0])):
            denominator = sums[r] if sums[r] > 0.0 else cutlass.Float32(1.0)
            inverse = cutlass.Float32(1.0) / denominator
            av[r, None] = av[r, None].load() * inverse
            row = start + rc[r, 0][0]
            if lane % 4 == 0 and row < q.shape[2] and width == 0:
                maximum[batch, head, row] = maxima[r]
                logsum[batch, head, row] = cute.math.log2(denominator, fastmath=True)
        self.store_result(accum, out, left, batch, head, start, offset, thread, tid, cutlass.Float32(1.0))


def compile_kernel(mode, tensors, causal):
    if mode == "backward":
        return compile_backward(mode, tensors, causal)
    if mode != "forward":
        raise ValueError("Select forward or backward.")
    return cute.compile(
        ReciprocalForward(causal),
        *[from_dlpack(t.detach(), assumed_align=16) for t in tensors],
        cutlass.Float32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
