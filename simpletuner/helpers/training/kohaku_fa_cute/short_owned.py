"""Full-row S128 D16 backward with exclusive gradient ownership."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.memory import SmemAllocator

from . import kernels
from .kernels import Attention


class ShortOwned(Attention):
    """One CTA owns every query/key row and all three output gradients."""

    def __init__(self, causal, narrow_product=False, prefetch_statistics=False):
        super().__init__("q", causal, False, columns=128, width=32, warps=8, query_column_warps=1)
        self.rows = 128
        self.product_width = 16 if narrow_product else 32
        self.narrow_product = narrow_product
        self.prefetch_statistics = prefetch_statistics
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
            (4, 2, 1),
            permutation_mnk=(64, 32, 16),
        )
        product_mma = mma
        if cutlass.const_expr(self.narrow_product):
            product_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(q.element_type, cutlass.Float32, (16, 8, 16)),
                (8, 1, 1),
                permutation_mnk=(128, 16, 16),
            )
        self.kernel_owned(q, k, v, out, maximum, logsum, dout, dq, dk, dv, scale, mma, product_mma).launch(
            grid=(q.shape[0] * q.shape[1], 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.jit
    def store_direct(self, result, destination, coordinates, batch, head, scale):
        for i in cutlass.range_constexpr(cute.size(coordinates)):
            row, column = coordinates[i]
            if column < 16:
                destination[batch, head, row, column] = (result[i] * scale).to(destination.element_type)

    @cute.jit
    def transposed_product(self, weights, source, mma, thread, tid, result):
        # The logical transpose requires transposed matrix loads as in Joint.
        left = cute.composition(weights, cute.make_layout((128, 128), stride=(128, 1)))
        ra = thread.make_fragment_A(thread.partition_A(left))
        ca = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
        )
        ta = ca.get_slice(tid)
        sa, da = ta.partition_S(left), ta.retile(ra)
        right = cute.composition(source, cute.make_layout((self.product_width, 128), stride=(128, 1)))
        rb = thread.make_fragment_B(thread.partition_B(right))
        cb = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), source.element_type), mma
        )
        tb = cb.get_slice(tid)
        sb, db = tb.partition_S(right), tb.retile(rb)
        for chunk in cutlass.range_constexpr(8):
            cute.copy(ca, sa[None, None, chunk], da[None, None, chunk])
            cute.copy(cb, sb[None, None, chunk], db[None, None, chunk])
            cute.gemm(mma, result, ra[None, None, chunk], rb[None, None, chunk], result)

    @cute.kernel
    def kernel_owned(self, q, k, v, out, maximum, logsum, dout, dq, dk, dv, scale, mma, product_mma):
        tid, _, _ = cute.arch.thread_idx()
        bh, _, _ = cute.arch.block_idx()
        batch, head = bh // q.shape[1], bh % q.shape[1]
        allocator = SmemAllocator()
        atom = cute.make_composed_layout(cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, 32), stride=(32, 1)))
        inputs = cute.tile_to_shape(atom, (128, 32), (0, 1))
        weights = cute.tile_to_shape(atom, (128, 128), (0, 1))
        sq = self.shared(allocator, q.element_type, inputs)
        sk = self.shared(allocator, q.element_type, inputs)
        sv = self.shared(allocator, q.element_type, inputs)
        sdo = self.shared(allocator, q.element_type, inputs)
        sp = self.shared(allocator, q.element_type, weights)
        sds = self.shared(allocator, q.element_type, weights)
        delta = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((128,)), byte_alignment=128)
        saved_maximum = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((128,)), byte_alignment=128)
        saved_logsum = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((128,)), byte_alignment=128)
        self.load(q, sq, batch, head, 0, tid)
        self.load(k, sk, batch, head, 0, tid)
        self.load(v, sv, batch, head, 0, tid)
        self.load(dout, sdo, batch, head, 0, tid)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if tid < 128:
            value = cutlass.Float32(0)
            for d in cutlass.range_constexpr(16):
                value += out[batch, head, tid, d].to(cutlass.Float32) * sdo[tid, d].to(cutlass.Float32)
            delta[tid] = value
            saved_maximum[tid] = maximum[batch, head, tid]
            saved_logsum[tid] = logsum[batch, head, tid]
        cute.arch.sync_threads()
        thread = mma.get_slice(tid)
        scores = self.dot(sq, sk, mma, thread, tid)
        derivatives = self.dot(sdo, sv, mma, thread, tid)
        coordinates = thread.partition_C(cute.make_identity_tensor((128, 128)))
        # Row statistics are immutable after their publication barrier. Load
        # them before any P/dS stores so shared-store alias analysis cannot
        # prevent common-subexpression reuse of repeated fragment rows.
        if cutlass.const_expr(self.prefetch_statistics):
            row_maximum = cute.make_rmem_tensor(scores.shape, cutlass.Float32)
            row_logsum = cute.make_rmem_tensor(scores.shape, cutlass.Float32)
            row_delta = cute.make_rmem_tensor(scores.shape, cutlass.Float32)
            for i in cutlass.range_constexpr(cute.size(coordinates)):
                row, _ = coordinates[i]
                row_maximum[i] = saved_maximum[row]
                row_logsum[i] = saved_logsum[row]
                row_delta[i] = delta[row]
        magnitude = scale if scale >= 0 else -scale
        sign = cutlass.Float32(1) if scale >= 0 else cutlass.Float32(-1)
        for i in cutlass.range_constexpr(cute.size(coordinates)):
            row, column = coordinates[i]
            probability = cutlass.Float32(0)
            visible = True
            if cutlass.const_expr(self.causal):
                visible = column <= row
            if visible:
                if cutlass.const_expr(self.prefetch_statistics):
                    maximum_value = row_maximum[i]
                    logsum_value = row_logsum[i]
                else:
                    maximum_value = saved_maximum[row]
                    logsum_value = saved_logsum[row]
                probability = cute.math.exp2(
                    (scores[i] * sign - maximum_value) * magnitude * 1.4426950408889634 - logsum_value,
                    fastmath=True,
                )
            rounded = probability.to(q.element_type)
            sp[row, column] = rounded
            if cutlass.const_expr(self.prefetch_statistics):
                delta_value = row_delta[i]
            else:
                delta_value = delta[row]
            sds[row, column] = (rounded.to(cutlass.Float32) * (derivatives[i] - delta_value)).to(q.element_type)
        cute.arch.sync_threads()
        product_thread = product_mma.get_slice(tid)
        output_coordinates = product_thread.partition_C(cute.make_identity_tensor((128, self.product_width)))
        result = cute.make_rmem_tensor(product_thread.partition_shape_C((128, self.product_width)), cutlass.Float32)
        result.fill(0.0)
        self.product(sds, sk, 0, product_mma, product_thread, tid, result, shared_weights=True)
        self.store_direct(result, dq, output_coordinates, batch, head, scale)
        result.fill(0.0)
        self.transposed_product(sds, sq, product_mma, product_thread, tid, result)
        self.store_direct(result, dk, output_coordinates, batch, head, scale)
        result.fill(0.0)
        self.transposed_product(sp, sdo, product_mma, product_thread, tid, result)
        self.store_direct(result, dv, output_coordinates, batch, head, cutlass.Float32(1))


def compile_kernel(mode, tensors, causal, narrow_product=False, prefetch_statistics=False):
    """Static native entry; preserve the ordinary training forward calculation."""
    if mode == "forward":
        return kernels.compile_kernel(
            tensors[0].device.index, "forward", causal, False, tuple(kernels.descriptor(t) for t in tensors)
        )
    if mode != "backward":
        raise ValueError("Select forward or backward for short attention.")
    program = ShortOwned(causal, narrow_product=narrow_product, prefetch_statistics=prefetch_statistics)
    return cute.compile(
        program,
        *[from_dlpack(t.detach(), assumed_align=16) for t in tensors],
        cutlass.Float32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
