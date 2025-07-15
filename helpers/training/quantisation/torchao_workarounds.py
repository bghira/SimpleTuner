import torchao, torch

from torch import Tensor
from typing import Optional
from torchao.prototype.quantized_training.int8 import Int8QuantizedTrainingLinearWeight


class _Int8WeightOnlyLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Int8QuantizedTrainingLinearWeight,
        bias: Optional[Tensor] = None,
    ):
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None

        # NOTE: we have to .T before .to(input.dtype) for torch.compile() mixed matmul to work
        out = (input @ weight.int_data.T.to(input.dtype)) * weight.scale
        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        grad_input = (grad_output * weight.scale) @ weight.int_data.to(
            grad_output.dtype
        )
        # print(f"dtypes: grad_output {grad_output.dtype}, input {input.dtype}, weight {weight.dtype}")
        # here is the patch: we will cast the input to the grad_output dtype.
        grad_weight = grad_output.reshape(-1, weight.shape[0]).T @ input.to(
            grad_output.dtype
        ).reshape(-1, weight.shape[1])
        grad_bias = (
            grad_output.reshape(-1, weight.shape[0]).sum(0) if ctx.bias else None
        )
        return grad_input, grad_weight, grad_bias


torchao.prototype.quantized_training.int8._Int8WeightOnlyLinear = _Int8WeightOnlyLinear
