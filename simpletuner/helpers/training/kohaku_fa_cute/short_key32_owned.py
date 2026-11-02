"""Four key-owner CTAs per head with one final FP32 dQ reduction."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.memory import SmemAllocator

from . import kernels
from .kernels import Attention


class ShortKey32Owned(Attention):
    """Each CTA owns 32 key rows; dQ partials are reduced before scaling."""

    def __init__(self, causal, dimension):
        super().__init__("q", causal, False, columns=32, width=dimension, warps=8, query_column_warps=1)
        self.rows = 64
        self.product_width = dimension
        self.score_b_matrices = 4

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        unused_mask: cute.Tensor,
        out: cute.Tensor,
        maximum: cute.Tensor,
        logsum: cute.Tensor,
        dout: cute.Tensor,
        unused_delta: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        unused_blocks: cute.Tensor,
        unused_base: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
            (4, 2, 1) if self.rows == 64 else (2, 4, 1),
            permutation_mnk=(64, 32, 16) if self.rows == 64 else (32, 32, 16),
        )
        kv_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
            (2, 4, 1),
            permutation_mnk=(32, 32, 16),
        )
        self.kernel_owned(q, k, v, out, maximum, logsum, dout, dq, dk, dv, unused_blocks, scale, mma, kv_mma).launch(
            grid=(4 * q.shape[0] * q.shape[1], 1, 1), block=(256, 1, 1), stream=stream
        )
        self.reduce_dq(unused_blocks, dq, scale).launch(
            grid=(cute.ceil_div(cute.size(dq), 256), 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def reduce_dq(self, partials, dq, scale):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        index = block * 256 + tid
        if index < cute.size(dq):
            feature = index % self.width
            row = (index // self.width) % 128
            head = (index // (self.width * 128)) % dq.shape[1]
            batch = index // (self.width * 128 * dq.shape[1])
            value = ((partials[0, index] + partials[1, index]) + (partials[2, index] + partials[3, index])) * scale
            dq[batch, head, row, feature] = value.to(dq.element_type)

    @cute.jit
    def store_direct(self, result, destination, coordinates, batch, head, scale, start):
        for i in cutlass.range_constexpr(cute.size(coordinates)):
            row, column = coordinates[i]
            if column < destination.shape[3]:
                destination[batch, head, start + row, column] = (result[i] * scale).to(destination.element_type)

    @cute.jit
    def transposed_product(self, weights, source, mma, thread, tid, result):
        # The logical transpose requires transposed matrix loads as in Joint.
        left = cute.composition(weights, cute.make_layout((self.columns, self.rows), stride=(self.rows, 1)))
        ra = thread.make_fragment_A(thread.partition_A(left))
        ca = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
        )
        ta = ca.get_slice(tid)
        sa, da = ta.partition_S(left), ta.retile(ra)
        right = cute.composition(source, cute.make_layout((self.width, self.rows), stride=(self.rows, 1)))
        rb = thread.make_fragment_B(thread.partition_B(right))
        cb = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
        )
        tb = cb.get_slice(tid)
        sb, db = tb.partition_S(right), tb.retile(rb)
        for chunk in cutlass.range_constexpr(self.rows // 16):
            cute.copy(ca, sa[None, None, chunk], da[None, None, chunk])
            cute.copy(cb, sb[None, None, chunk], db[None, None, chunk])
            cute.gemm(mma, result, ra[None, None, chunk], rb[None, None, chunk], result)

    @cute.kernel
    def kernel_owned(self, q, k, v, out, maximum, logsum, dout, dq, dk, dv, blocks, scale, mma, kv_mma):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        bh = cta // 4
        key_tile = cta % 4
        key_start = key_tile * self.columns
        batch, head = bh // q.shape[1], bh % q.shape[1]
        allocator = SmemAllocator()
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        inputs = cute.tile_to_shape(atom, (self.rows, self.width), (0, 1))
        keys = cute.tile_to_shape(atom, (self.columns, self.width), (0, 1))
        weights = cute.tile_to_shape(atom, (self.rows, self.columns), (0, 1))
        sq = self.shared(allocator, q.element_type, inputs)
        sk = self.shared(allocator, q.element_type, keys)
        sv = self.shared(allocator, q.element_type, keys)
        sdo = self.shared(allocator, q.element_type, inputs)
        sp = self.shared(allocator, q.element_type, weights)
        sds = self.shared(allocator, q.element_type, weights)
        delta = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((self.rows,)), byte_alignment=128)
        saved_maximum = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((self.rows,)), byte_alignment=128)
        saved_logsum = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((self.rows,)), byte_alignment=128)
        self.load(k, sk, batch, head, key_start, tid)
        self.load(v, sv, batch, head, key_start, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        thread = mma.get_slice(tid)
        kv_thread = kv_mma.get_slice(tid)
        kv_coordinates = kv_thread.partition_C(cute.make_identity_tensor((self.columns, self.width)))
        accum_dk = cute.make_rmem_tensor(kv_thread.partition_shape_C((self.columns, self.width)), cutlass.Float32)
        accum_dv = cute.make_rmem_tensor(kv_thread.partition_shape_C((self.columns, self.width)), cutlass.Float32)
        accum_dk.fill(0.0)
        accum_dv.fill(0.0)
        for query_tile in cutlass.range(128 // self.rows, unroll_full=False):
            start = query_tile * self.rows
            self.load(q, sq, batch, head, start, tid)
            self.load(dout, sdo, batch, head, start, tid)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            if tid < self.rows:
                value = cutlass.Float32(0)
                for d in cutlass.range_constexpr(self.width):
                    value += out[batch, head, start + tid, d].to(cutlass.Float32) * sdo[tid, d].to(cutlass.Float32)
                delta[tid] = value
                saved_maximum[tid] = maximum[batch, head, start + tid]
                saved_logsum[tid] = logsum[batch, head, start + tid]
            cute.arch.sync_threads()
            scores = self.dot(sq, sk, mma, thread, tid)
            derivatives = self.dot(sdo, sv, mma, thread, tid)
            coordinates = thread.partition_C(cute.make_identity_tensor((self.rows, self.columns)))
            magnitude = scale if scale >= 0 else -scale
            sign = cutlass.Float32(1) if scale >= 0 else cutlass.Float32(-1)
            for i in cutlass.range_constexpr(cute.size(coordinates)):
                row, column = coordinates[i]
                probability = cutlass.Float32(0)
                visible = True
                if cutlass.const_expr(self.causal):
                    visible = key_start + column <= start + row
                if visible:
                    probability = cute.math.exp2(
                        (scores[i] * sign - saved_maximum[row]) * magnitude * 1.4426950408889634 - saved_logsum[row],
                        fastmath=True,
                    )
                rounded = probability.to(q.element_type)
                sp[row, column] = rounded
                sds[row, column] = (rounded.to(cutlass.Float32) * (derivatives[i] - delta[row])).to(q.element_type)
            cute.arch.sync_threads()
            output_coordinates = thread.partition_C(cute.make_identity_tensor((self.rows, self.width)))
            result = cute.make_rmem_tensor(thread.partition_shape_C((self.rows, self.width)), cutlass.Float32)
            result.fill(0.0)
            self.product(sds, sk, 0, mma, thread, tid, result, shared_weights=True)
            for i in cutlass.range_constexpr(cute.size(output_coordinates)):
                row, column = output_coordinates[i]
                index = ((batch * q.shape[1] + head) * 128 + start + row) * self.width + column
                blocks[key_tile, index] = result[i]
            self.transposed_product(sds, sq, kv_mma, kv_thread, tid, accum_dk)
            self.transposed_product(sp, sdo, kv_mma, kv_thread, tid, accum_dv)
            # Every warp must finish reading Q/dO/P/dS before the next tile reuses them.
            cute.arch.sync_threads()
        self.store_direct(accum_dk, dk, kv_coordinates, batch, head, scale, key_start)
        self.store_direct(accum_dv, dv, kv_coordinates, batch, head, cutlass.Float32(1), key_start)


def compile_kernel(mode, tensors, causal):
    """Static native entry; preserve the ordinary training forward calculation."""
    if mode == "forward":
        return kernels.compile_kernel(
            tensors[0].device.index, "forward", causal, False, tuple(kernels.descriptor(t) for t in tensors)
        )
    if mode != "backward":
        raise ValueError("Select forward or backward for short attention.")
    program = ShortKey32Owned(causal, tensors[0].shape[-1])
    return cute.compile(
        program,
        *[from_dlpack(t.detach(), assumed_align=16) for t in tensors],
        cutlass.Float32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
