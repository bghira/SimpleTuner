import torch
if torch.cuda.is_available():
    # the marlin fp8 kernel needs some help with dtype casting for some reason
    # see: https://github.com/huggingface/optimum-quanto/pull/296#issuecomment-2380719201
    import optimum
    from optimum.quanto.library.extensions.cuda import ext as quanto_ext

    @torch.library.custom_op("quanto::fp8_marlin_gemm", mutates_args=(), device_types=["cuda"])
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
        return quanto_ext.lib.fp8_marlin_gemm(a.to(b_scales.dtype), b_q_weight, b_scales, workspace, num_bits, size_m, size_n, size_k)

    optimum.quanto.library.extensions.cuda.fp8_marlin_gemm = fp8_marlin_gemm