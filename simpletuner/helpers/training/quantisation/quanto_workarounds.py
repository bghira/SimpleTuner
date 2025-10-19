import optimum
import torch

if torch.cuda.is_available():
    # the marlin fp8 kernel needs some help with dtype casting for some reason
    # see: https://github.com/huggingface/optimum-quanto/pull/296#issuecomment-2380719201
    if torch.device("cuda").type == "cuda" and torch.version.cuda:
        from optimum.quanto.library.extensions.cuda import ext as quanto_ext

        # Save the original operator
        original_gemm_f16f8_marlin = torch.ops.quanto.gemm_f16f8_marlin

        def fp8_marlin_gemm_wrapper(
            a: torch.Tensor,
            b_q_weight: torch.Tensor,
            b_scales: torch.Tensor,
            workspace: torch.Tensor,
            num_bits: int,
            size_m: int,
            size_n: int,
            size_k: int,
        ) -> torch.Tensor:
            # Ensure 'a' has the correct dtype
            a = a.to(b_scales.dtype)
            # Call the original operator
            return original_gemm_f16f8_marlin(
                a,
                b_q_weight,
                b_scales,
                workspace,
                num_bits,
                size_m,
                size_n,
                size_k,
            )

        # Monkey-patch the operator
        torch.ops.quanto.gemm_f16f8_marlin = fp8_marlin_gemm_wrapper

    class TinyGemmQBitsLinearFunction(optimum.quanto.tensor.function.QuantizedLinearFunction):
        @staticmethod
        def forward(ctx, input, other, bias):
            ctx.save_for_backward(input, other)
            if type(input) is not torch.Tensor:
                input = input.dequantize()
            in_features = input.shape[-1]
            out_features = other.shape[0]
            output_shape = input.shape[:-1] + (out_features,)
            output = torch._weight_int4pack_mm(
                input.view(-1, in_features).to(dtype=other.dtype),
                other._data._data,
                other._group_size,
                other._scale_shift,
            )
            output = output.view(output_shape)
            if bias is not None:
                output = output + bias
            return output

    from optimum.quanto.tensor.weights import tinygemm

    tinygemm.qbits.TinyGemmQBitsLinearFunction = TinyGemmQBitsLinearFunction


class WeightQBytesLinearFunction(optimum.quanto.tensor.function.QuantizedLinearFunction):
    @staticmethod
    def forward(ctx, input, other, bias=None):
        ctx.save_for_backward(input, other)
        if isinstance(input, optimum.quanto.tensor.QBytesTensor):
            output = torch.ops.quanto.qbytes_mm(input._data, other._data, input._scale * other._scale)
        else:
            in_features = input.shape[-1]
            out_features = other.shape[0]
            output_shape = input.shape[:-1] + (out_features,)
            output = torch.ops.quanto.qbytes_mm(input.reshape(-1, in_features), other._data, other._scale)
            output = output.view(output_shape)
        if bias is not None:
            output = output + bias
        return output


optimum.quanto.tensor.weights.qbytes.WeightQBytesLinearFunction = WeightQBytesLinearFunction


def reshape_qlf_backward(ctx, gO):
    # another one where we need .reshape instead of .view
    input_gO = other_gO = bias_gO = None
    input, other = ctx.saved_tensors
    out_features, in_features = other.shape
    if ctx.needs_input_grad[0]:
        # grad(A@(B.t()) = gO => grad(A) = gO@(B.t().t()) = gO@B
        input_gO = torch.matmul(gO, other)
    if ctx.needs_input_grad[1]:
        # grad(B@A.t()) = gO.t() => grad(B) = gO.t()@(A.t().t()) = gO.t()@A
        other_gO = torch.matmul(
            gO.reshape(-1, out_features).t(),
            input.to(gO.dtype).reshape(-1, in_features),
        )
    if ctx.needs_input_grad[2]:
        # Bias gradient is the sum on all dimensions but the last one
        dim = tuple(range(gO.ndim - 1))
        bias_gO = gO.sum(dim)
    return input_gO, other_gO, bias_gO


optimum.quanto.tensor.function.QuantizedLinearFunction.backward = reshape_qlf_backward
