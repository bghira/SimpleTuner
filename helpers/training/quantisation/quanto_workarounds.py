import torch

if torch.cuda.is_available():
    # the marlin fp8 kernel needs some help with dtype casting for some reason
    # see: https://github.com/huggingface/optimum-quanto/pull/296#issuecomment-2380719201
    import optimum
    from optimum.quanto.library.extensions.cuda import ext as quanto_ext

    @torch.library.custom_op(
        "quanto::fp8_marlin_gemm", mutates_args=(), device_types=["cuda"]
    )
    def fp8_marlin_gemm(
        a: torch.Tensor,
        b_q_weight: torch.Tensor,
        b_scales: torch.Tensor,
        workspace: torch.Tensor,
        num_bits: int,
        size_m: int,
        size_n: int,
        size_k: int,
    ) -> torch.Tensor:
        assert b_scales.dtype == torch.float16 or b_scales.dtype == torch.bfloat16
        assert b_q_weight.dim() == 2
        assert b_q_weight.dtype == torch.int32
        return quanto_ext.lib.fp8_marlin_gemm(
            a.to(b_scales.dtype),
            b_q_weight,
            b_scales,
            workspace,
            num_bits,
            size_m,
            size_n,
            size_k,
        )

    optimum.quanto.library.extensions.cuda.fp8_marlin_gemm = fp8_marlin_gemm

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
                input.view(-1, in_features).to(dtype=other.dtype), other._data._data, other._group_size, other._scale_shift
            )
            output = output.view(output_shape)
            if bias is not None:
                output = output + bias
            return output
        
    from optimum.quanto.tensor.weights import tinygemm
    tinygemm.qbits.TinyGemmQBitsLinearFunction = TinyGemmQBitsLinearFunction