"""Derived Hopper methods with separate unscaled attention statistics.

Derived from Dao-AILab/flash-attention revision
4616f5116d33e3ba3cbb71d582caffcfcb49649c, BSD-3-Clause.
See LICENSES/FlashAttention-BSD-3-Clause.txt and FlashAttention-AUTHORS.txt.
Unchanged methods use the pinned optional dependency.
"""

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang,
# Vijay Thakkar, Pradeep Ramani, Tri Dao.

import math
import operator
from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import flash_attn.cute.utils as utils
from cutlass import Boolean, Float32
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.softmax import Softmax as _Original
from flash_attn.cute.utils import AuxData
from quack import layout_utils
from quack.cute_dsl_utils import ParamsBase


class Softmax(_Original):

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
    ):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softmax(scale_log2, num_rows, row_max, row_sum, arch, softmax_scale)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        """Apply online softmax and return the row_scale to rescale O.

        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param is_first: is first n_block
        :type is_first: cutlass.Constexpr
        """
        # Change acc_S to M,N layout view.
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)

        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2
        arch = self.arch

        # Each iteration processes one row of acc_S
        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()  # (n_block_size)

            row_max_cur = utils.fmax_reduce(
                acc_S_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
                arch=arch,
            )

            row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
            # Update row_max before changing row_max_cur to safe value for -inf
            row_max_prev = row_max[r]
            row_max[r] = row_max_cur

            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur

            if cutlass.const_expr(is_first):
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2((acc_S_row - row_max_cur) * scale_log2, fastmath=True)
                acc_S_row_sum = utils.fadd_reduce(acc_S_row_exp, init_val=None, arch=arch)
                row_scale[r] = 1.0
            else:
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = cute.math.exp2((acc_S_row - row_max_cur) * scale_log2, fastmath=True)
                # row_scale[r] = cute.math.exp2(row_max_prev * self.scale_log2 - row_max_cur_scaled)
                row_scale[r] = cute.math.exp2((row_max_prev - row_max_cur) * scale_log2, fastmath=True)
                acc_S_row_sum = utils.fadd_reduce(acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch)

            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)

        return row_scale

    @cute.jit
    def finalize(self, final_scale: Float32 = 1.0, sink_val: Float32 | cute.Tensor | None = None) -> cute.Tensor:
        """Finalize the online softmax by computing the scale and logsumexp."""
        if cutlass.const_expr(sink_val is not None and isinstance(sink_val, cute.Tensor)):
            assert cute.size(sink_val) == cute.size(self.row_sum)
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2

        # quad reduction for row_sum as we didn't do it during each iteration of online softmax
        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(row_max, Float32)

        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            row_max_scaled = row_max[r] * scale_log2
            if cutlass.const_expr(sink_val is not None):
                sink_val_cur = sink_val if not isinstance(sink_val, cute.Tensor) else sink_val[r]
                LOG2_E = math.log2(math.e)
                if row_max[r] == -Float32.inf:
                    row_max_scaled = sink_val_cur * LOG2_E
                row_sum[r] += cute.math.exp2(sink_val_cur * LOG2_E - row_max_scaled, fastmath=True)

            # if row_sum is zero or nan, set acc_O_mn_row to 1.0
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            row_scale[r] = (cute.arch.rcp_approx(row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0)) * final_scale
            row_sum_cur = row_sum[r]
            LN2 = math.log(2.0)
            row_sum[r] = cute.math.log2(row_sum_cur, fastmath=True) if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
        return row_scale
