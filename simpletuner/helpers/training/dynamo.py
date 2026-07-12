import contextlib
import contextvars
import logging
import os
from collections.abc import Callable, Iterator
from typing import Any

import torch

logger = logging.getLogger(__name__)

_PEFT_LORA_CUDAGRAPH_PATCHED = False
_PEFT_TE_LORA_CUDAGRAPH_PATCHED = False


@torch.compiler.disable
def _clone_outside_compile(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone()


def _coerce_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def dynamo_config_patches(config: Any) -> dict[str, Any]:
    patches: dict[str, Any] = {}
    if _coerce_flag(getattr(config, "dynamo_dynamic", None)):
        patches["capture_dynamic_output_shape_ops"] = True
        patches["force_parameter_static_shapes"] = False
    return patches


@contextlib.contextmanager
def dynamo_config_context(config: Any) -> Iterator[None]:
    patches = dynamo_config_patches(config)
    if not patches:
        yield
        return

    with torch._dynamo.config.patch(patches):
        yield


def run_with_dynamo_config(config: Any, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    with dynamo_config_context(config):
        context = contextvars.copy_context()
        return context.run(fn, *args, **kwargs)


def _normalise_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("_", "-")


def _inductor_cudagraphs_enabled(config: Any = None) -> bool:
    config_backend = _normalise_text(getattr(config, "dynamo_backend", None))
    accelerate_backend = _normalise_text(os.environ.get("ACCELERATE_DYNAMO_BACKEND"))
    backend = accelerate_backend or config_backend
    if backend != "inductor":
        return False

    config_mode = _normalise_text(getattr(config, "dynamo_mode", None))
    accelerate_mode = _normalise_text(os.environ.get("ACCELERATE_DYNAMO_MODE"))
    mode = accelerate_mode or config_mode

    try:
        import torch._inductor.config as inductor_config
    except ImportError:
        return False
    return bool(getattr(inductor_config.triton, "cudagraphs", False)) or (
        mode in {"cudagraphs", "reduce-overhead"} and bool(getattr(inductor_config.triton, "cudagraph_trees", False))
    )


def patch_peft_lora_for_cudagraphs() -> bool:
    global _PEFT_LORA_CUDAGRAPH_PATCHED
    if _PEFT_LORA_CUDAGRAPH_PATCHED:
        return False

    import peft.tuners.lora.layer as peft_lora_layer

    peft_linear = peft_lora_layer.Linear
    if hasattr(peft_linear, "_simpletuner_original_forward"):
        _PEFT_LORA_CUDAGRAPH_PATCHED = True
        return False

    peft_linear._simpletuner_original_forward = peft_linear.forward
    variant_kwarg_keys = peft_lora_layer.VARIANT_KWARG_KEYS

    def _simpletuner_cudagraph_safe_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        variant_kwargs = {key: kwargs.pop(key, None) for key in variant_kwarg_keys}

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **variant_kwargs, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            result_cloned = False
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                if not result_cloned:
                    result = _clone_outside_compile(result)
                    result_cloned = True

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)
                if active_adapter not in self.lora_variant:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    result = self.lora_variant[active_adapter].forward(
                        self,
                        active_adapter=active_adapter,
                        x=x,
                        result=result,
                        **variant_kwargs,
                        **kwargs,
                    )

            result = result.to(torch_result_dtype)

        return result

    peft_linear.forward = _simpletuner_cudagraph_safe_forward
    _PEFT_LORA_CUDAGRAPH_PATCHED = True
    logger.info("Applied PEFT LoRA CUDAGraph clone workaround for compiled training.")
    return True


def patch_peft_te_lora_for_cudagraphs() -> bool:
    global _PEFT_TE_LORA_CUDAGRAPH_PATCHED
    if _PEFT_TE_LORA_CUDAGRAPH_PATCHED:
        return False

    try:
        import peft.tuners.lora.te as peft_te_lora
    except ImportError:
        return False

    te_linear = peft_te_lora.TeLinear
    if hasattr(te_linear, "_simpletuner_original_forward"):
        _PEFT_TE_LORA_CUDAGRAPH_PATCHED = True
        return False

    te_linear._simpletuner_original_forward = te_linear.forward

    def _simpletuner_cudagraph_safe_te_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise ValueError(f"{self.__class__.__name__} does not support mixed_batch_forward yet.")
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            result_cloned = False
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                if not result_cloned:
                    result = _clone_outside_compile(result)
                    result_cloned = True

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)
                result = result + lora_B(lora_A(dropout(x))) * scaling

            result = result.to(torch_result_dtype)

        return result

    te_linear.forward = _simpletuner_cudagraph_safe_te_forward
    _PEFT_TE_LORA_CUDAGRAPH_PATCHED = True
    logger.info("Applied PEFT TransformerEngine LoRA CUDAGraph clone workaround for compiled training.")
    return True


def install_cudagraph_workarounds(config: Any) -> bool:
    if not _inductor_cudagraphs_enabled(config):
        return False
    patched = patch_peft_lora_for_cudagraphs()
    patched |= patch_peft_te_lora_for_cudagraphs()
    return patched


def mark_cudagraph_step_begin(config: Any) -> None:
    cudagraphs_enabled = _inductor_cudagraphs_enabled(config)
    if cudagraphs_enabled:
        install_cudagraph_workarounds(config)
    if not getattr(config, "dynamo_backend", None) and not cudagraphs_enabled:
        return
    mark_step_begin = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if mark_step_begin is not None:
        mark_step_begin()
