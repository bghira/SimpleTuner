"""Single-CTA tcgen05 score products, returned in the warp-MMA register layout."""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.tensor_utils import LayoutEnum


@dsl_user_op
def fence_after_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def fence_before_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


@cute.jit
def dot(a, b, thread, tid, columns: cutlass.Constexpr, state):
    pointer, barrier, scratch, phase_buffer = state
    mma = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            a.element_type,
            cutlass.Float32,
            (64, columns, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
    )
    partition = mma.get_slice(0)
    ra = mma.make_fragment_A(partition.partition_A(a))
    rb = mma.make_fragment_B(partition.partition_B(b))
    fragment = mma.make_fragment_C(partition.partition_shape_C((64, columns)))
    accumulator = cute.make_tensor(pointer, fragment.layout)
    phase = phase_buffer[0]
    cute.arch.fence_view_async_shared()
    # cute.gemm elects its issuing lane internally; keep the whole warp active.
    if tid < 32:
        fence_after_sync()
        mma.set(tcgen05.Field.ACCUMULATE, False)
        for chunk in cutlass.range(cute.size(ra.shape[2]), unroll_full=True):
            cute.gemm(mma, accumulator, ra[None, None, chunk], rb[None, None, chunk], accumulator)
            mma.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(barrier)
    cute.arch.mbarrier_wait(barrier, phase)
    fence_after_sync()
    view = accumulator[(None, None), 0, 0]
    atom = sm100_utils.get_tmem_load_op(
        (64, columns, 16), LayoutEnum.ROW_MAJOR, cutlass.Float32, cutlass.Float32, (64, columns), False
    )
    copy = tcgen05.make_tmem_copy(atom, view)
    copier = copy.get_slice(tid)
    src, dst = copier.partition_S(view), copier.partition_D(scratch)
    values = cute.make_rmem_tensor(dst.shape, cutlass.Float32)
    cute.copy(copy, src, values)
    cute.arch.fence_view_async_tmem_load()
    fence_before_sync()
    for i in cutlass.range_constexpr(cute.size(dst)):
        dst[i] = values[i]
    cute.arch.sync_threads()
    coordinates = thread.partition_C(cute.make_identity_tensor((64, columns)))
    result = cute.make_rmem_tensor(coordinates.shape, cutlass.Float32)
    for i in cutlass.range_constexpr(cute.size(result)):
        row, col = coordinates[i]
        result[i] = scratch[row, col]
    if tid == 0:
        phase_buffer[0] = phase ^ 1
    cute.arch.sync_threads()
    return result
