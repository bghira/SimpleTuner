import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_INPUT_DTYPE = torch.float8_e5m2


def _scaled_mm_supported(x: torch.Tensor) -> bool:
    if not x.is_cuda or not hasattr(torch, "_scaled_mm"):
        return False
    mode = os.environ.get("SIMPLETUNER_FP8_NATIVE_SCALED_MM", "auto").strip().lower()
    if mode in {"0", "false", "off", "no"}:
        return False
    if mode in {"1", "true", "on", "yes"}:
        return True
    return torch.cuda.get_device_capability(x.device)[0] >= 9


def quantize_weight_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight.detach().to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_E4M3_MAX
    q = (w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
    return q, scale.squeeze(1).to(torch.float32)


class _Fp8NativeLinearFn(torch.autograd.Function):
    @staticmethod
    def _dequantized_forward(
        x_2d: torch.Tensor,
        input_shape: torch.Size,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
    ) -> torch.Tensor:
        w = weight.to(x_2d.dtype) * weight_scale.to(x_2d.dtype).unsqueeze(1)
        bias_arg = bias.to(x_2d.dtype) if bias is not None else None
        return torch.nn.functional.linear(x_2d, w, bias_arg).reshape(*input_shape[:-1], out_features)

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
    ) -> torch.Tensor:
        if not _scaled_mm_supported(x):
            raise RuntimeError("fp8-native requires CUDA with torch._scaled_mm support.")

        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        input_max = torch.finfo(FP8_INPUT_DTYPE).max
        input_scale = (input_max / x_2d.detach().abs().amax().clamp(min=1e-12)).clamp(max=input_max)
        x_fp8 = (x_2d * input_scale).clamp(-input_max, input_max).to(FP8_INPUT_DTYPE)

        out_dtype = x.dtype if x.dtype in {torch.bfloat16, torch.float16} else torch.bfloat16
        scale_a = torch.empty((x_2d.shape[0], 1), device=x_2d.device, dtype=torch.float32)
        scale_a.copy_(input_scale.reciprocal().to(torch.float32).expand_as(scale_a))
        scale_b = weight_scale.to(torch.float32).reshape(1, -1).contiguous()
        bias_arg = bias.to(out_dtype) if bias is not None else None

        try:
            out = torch._scaled_mm(
                x_fp8,
                weight.T,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=bias_arg,
                out_dtype=out_dtype,
                use_fast_accum=True,
            )
        except RuntimeError as exc:
            message = str(exc)
            if (
                "CUBLAS_STATUS_NOT_SUPPORTED" not in message
                and "scale_a" not in message
                and "Invalid scaling configuration" not in message
            ):
                raise
            out = _Fp8NativeLinearFn._dequantized_forward(
                x_2d,
                input_shape,
                weight,
                weight_scale,
                bias,
                out_features,
            )
            ctx.save_for_backward(weight, weight_scale)
            ctx.input_shape = input_shape
            ctx.out_features = out_features
            return out
        if isinstance(out, tuple):
            out = out[0]
        if out.dtype != x.dtype:
            out = out.to(x.dtype)

        ctx.save_for_backward(weight, weight_scale)
        ctx.input_shape = input_shape
        ctx.out_features = out_features
        return out.reshape(*input_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight, weight_scale = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_2d = grad_output.reshape(-1, ctx.out_features)
            dequant_weight = weight.to(grad_output.dtype) * weight_scale.to(grad_output.dtype).unsqueeze(1)
            grad_x = grad_2d.matmul(dequant_weight).reshape(ctx.input_shape)
        return grad_x, None, None, None, None


class Fp8NativeLinear(nn.Module):
    weight: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: torch.Tensor | None,
        weight: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        weight_fp8, weight_scale = quantize_weight_to_fp8(weight)
        self.register_buffer("weight", weight_fp8)
        self.register_buffer("weight_scale", weight_scale)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().to(compute_dtype))

    def _apply(self, fn, recurse: bool = True):
        weight = self._buffers.pop("weight")
        weight_scale = self._buffers.pop("weight_scale")
        try:
            super()._apply(fn, recurse=recurse)
            device_probe = fn(torch.empty(0, device=weight.device, dtype=torch.uint8))
        finally:
            self._buffers["weight"] = weight
            self._buffers["weight_scale"] = weight_scale
        self._buffers["weight"] = weight.to(device=device_probe.device)
        self._buffers["weight_scale"] = weight_scale.to(device=device_probe.device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype != FP8_WEIGHT_DTYPE:
            raise RuntimeError(f"Fp8NativeLinear weight must be {FP8_WEIGHT_DTYPE}, got {self.weight.dtype}.")
        if self.weight_scale.dtype != torch.float32:
            raise RuntimeError(f"Fp8NativeLinear weight_scale must be torch.float32, got {self.weight_scale.dtype}.")
        return _Fp8NativeLinearFn.apply(x, self.weight, self.weight_scale, self.bias, self.out_features)


def replace_linear_with_fp8_native(
    module: nn.Module,
    filter_fn,
    compute_dtype: torch.dtype,
    *,
    prefix: str = "",
) -> int:
    converted = 0
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}{name}"
        if isinstance(child, nn.Linear) and filter_fn(child, child_prefix):
            replacement = Fp8NativeLinear(
                child.in_features,
                child.out_features,
                child.bias,
                child.weight,
                compute_dtype,
            )
            replacement.to(device=child.weight.device)
            setattr(module, name, replacement)
            converted += 1
        else:
            converted += replace_linear_with_fp8_native(
                child,
                filter_fn,
                compute_dtype,
                prefix=f"{child_prefix}.",
            )
    return converted


def patch_peft_fp8_native_dispatcher() -> None:
    try:
        import peft.tuners.lora.model as peft_lora_model
        from peft.tuners.lora.layer import Linear
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:
        return

    if hasattr(peft_lora_model, "_simpletuner_original_dispatch_default"):
        return

    peft_lora_model._simpletuner_original_dispatch_default = peft_lora_model.dispatch_default

    def _simpletuner_dispatch_default(target: torch.nn.Module, adapter_name: str, config, **kwargs):
        target_base_layer = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target
        if isinstance(target_base_layer, Fp8NativeLinear):
            if config.fan_in_fan_out:
                config.fan_in_fan_out = False
            return Linear(target, adapter_name, config=config, **kwargs)
        return peft_lora_model._simpletuner_original_dispatch_default(
            target,
            adapter_name,
            config=config,
            **kwargs,
        )

    peft_lora_model.dispatch_default = _simpletuner_dispatch_default


def log_fp8_native_storage_summary(model: nn.Module, model_precision: str) -> None:
    converted = 0
    payload_bytes = 0
    logical_bytes = 0
    for module in model.modules():
        if not isinstance(module, Fp8NativeLinear):
            continue
        converted += 1
        logical_bytes += module.weight.numel() * torch.empty((), dtype=module.compute_dtype).element_size()
        payload_bytes += module.weight.numel() * module.weight.element_size()
        payload_bytes += module.weight_scale.numel() * module.weight_scale.element_size()
        if module.bias is not None:
            payload_bytes += module.bias.numel() * module.bias.element_size()

    logger.info(
        "Native FP8 storage summary for %s: fp8_linear_modules=%s, logical_weight_size=%.2f GB, payload_size=%.2f GB",
        model_precision,
        converted,
        logical_bytes / (1024**3),
        payload_bytes / (1024**3),
    )


def mark_fp8_native_ddp_ignore_params(module: nn.Module) -> int:
    ignored_names = []
    fp8_module_names = {name for name, mod in module.named_modules() if isinstance(mod, Fp8NativeLinear)}
    for name, _buffer in module.named_buffers():
        if any(
            name == f"{module_name}.{suffix}" for module_name in fp8_module_names for suffix in ("weight", "weight_scale")
        ):
            ignored_names.append(name)
    if not ignored_names:
        return 0
    existing = getattr(module, "_ddp_params_and_buffers_to_ignore", set())
    module._ddp_params_and_buffers_to_ignore = set(existing) | set(ignored_names)
    return len(ignored_names)
