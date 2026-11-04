"""Load immutable causal row statistics before writing probability staging."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.memory import SmemAllocator

from . import kernels
from .short_key_quad_skip_owned import ShortKeyQuadSkipOwned


class ShortKeyStatsOwned(ShortKeyQuadSkipOwned):
    """Keep row normalization and delta in registers across shared stores."""

    @cute.kernel
    def kernel_owned(self, q, k, v, out, maximum, logsum, dout, dq, dk, dv, blocks, scale, mma, kv_mma):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        bh = cta // 2
        key_tile = cta % 2
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
        first_query_tile = cutlass.Int32(0)
        if cutlass.const_expr(self.causal):
            first_query_tile = key_start // self.rows
        for query_tile in cutlass.range(first_query_tile, 128 // self.rows, unroll_full=False):
            start = query_tile * self.rows
            self.load(q, sq, batch, head, start, tid)
            self.load(dout, sdo, batch, head, start, tid)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            delta_row, delta_lane = tid // 4, tid % 4
            value = cutlass.Float32(0)
            for d in cutlass.range_constexpr(self.width // 4):
                feature = d * 4 + delta_lane
                value += out[batch, head, start + delta_row, feature].to(cutlass.Float32) * sdo[delta_row, feature].to(
                    cutlass.Float32
                )
            value = kernels.quad_sum(value)
            if delta_lane == 0:
                delta[delta_row] = value
            if tid < self.rows:
                saved_maximum[tid] = maximum[batch, head, start + tid]
                saved_logsum[tid] = logsum[batch, head, start + tid]
            cute.arch.sync_threads()
            scores = self.dot(sq, sk, mma, thread, tid)
            derivatives = self.dot(sdo, sv, mma, thread, tid)
            coordinates = thread.partition_C(cute.make_identity_tensor((self.rows, self.columns)))
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
                    visible = key_start + column <= start + row
                if visible:
                    probability = cute.math.exp2(
                        (scores[i] * sign - row_maximum[i]) * magnitude * 1.4426950408889634 - row_logsum[i],
                        fastmath=True,
                    )
                rounded = probability.to(q.element_type)
                sp[row, column] = rounded
                sds[row, column] = (rounded.to(cutlass.Float32) * (derivatives[i] - row_delta[i])).to(q.element_type)
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
    if mode != "backward" or not causal:
        raise ValueError("Select the causal short backward computation.")
    return cute.compile(
        ShortKeyStatsOwned(causal, tensors[0].shape[-1]),
        *[from_dlpack(t.detach(), assumed_align=16) for t in tensors],
        cutlass.Float32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
