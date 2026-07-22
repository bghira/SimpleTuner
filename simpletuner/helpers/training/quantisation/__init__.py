import logging
import os
import sys
from contextlib import nullcontext
from functools import partial
from inspect import Parameter, signature
from typing import Any, Callable, Mapping, Optional

import torch

from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.quantisation.fp8_native import mark_fp8_native_ddp_ignore_params
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel(logging.ERROR)


def _wrap_transformerengine_debug_forward(module: torch.nn.Module, fqn: str) -> None:
    if os.environ.get("SIMPLETUNER_TE_DEBUG_BACKWARD", "") != "1":
        return
    if hasattr(module, "_simpletuner_te_debug_original_forward"):
        return

    module._simpletuner_te_debug_original_forward = module.forward
    original_forward = module.forward

    def _simpletuner_te_debug_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        if not torch.is_tensor(output) or not output.requires_grad:
            return output
        input_tensor = next((arg for arg in args if torch.is_tensor(arg)), None)
        input_shape = tuple(input_tensor.shape) if input_tensor is not None else None
        input_stride = tuple(input_tensor.stride()) if input_tensor is not None else None

        def _log_backward_entry(grad):
            print(
                f"[SimpleTuner TE debug] backward {fqn} "
                f"input_shape={input_shape} input_stride={input_stride} "
                f"output_shape={tuple(output.shape)} grad_shape={tuple(grad.shape)} "
                f"grad_stride={tuple(grad.stride())}",
                flush=True,
            )
            return grad

        output.register_hook(_log_backward_entry)
        return output

    module.forward = _simpletuner_te_debug_forward


PIPELINE_ONLY_PRESETS = {"nf4-bnb", "int4-torchao"}
PIPELINE_QUANTIZATION_PRESETS = PIPELINE_ONLY_PRESETS | {
    "int8-torchao",
    "int8dq-torchao",
    "int8dq-int4-torchao",
    "fp8-torchao",
    "fp8wo-torchao",
    "fp8-int4-torchao",
    "int8-quanto",
    "int4-quanto",
    "int2-quanto",
    "fp8-quanto",
    "fp8uz-quanto",
}
MANUAL_QUANTO_PRESETS = {"int2-quanto", "int4-quanto", "int8-quanto", "fp8-quanto", "fp8uz-quanto"}
MANUAL_TORCHAO_PRESETS = {
    "int8-torchao",
    "int8dq-torchao",
    "int8dq-int4-torchao",
    "fp8-torchao",
    "fp8wo-torchao",
    "fp8-int4-torchao",
}
MANUAL_FP8_NATIVE_PRESETS = {"fp8-native"}
MANUAL_TRANSFORMERENGINE_PRESETS = {"fp8-transformerengine"}
MANUAL_SDNQ_PRESETS = {
    "int8-sdnq",
    "uint8-sdnq",
    "int16-sdnq",
    "uint16-sdnq",
    "fp16-sdnq",
    "fp8-sdnq",
    "int6-sdnq",
    "int5-sdnq",
    "uint5-sdnq",
    "uint4-sdnq",
    "uint3-sdnq",
    "uint2-sdnq",
}
MANUAL_QUANTIZATION_PRESETS = (
    MANUAL_QUANTO_PRESETS
    | MANUAL_TORCHAO_PRESETS
    | MANUAL_SDNQ_PRESETS
    | MANUAL_FP8_NATIVE_PRESETS
    | MANUAL_TRANSFORMERENGINE_PRESETS
)


def _normalize_dtype(weight_dtype: Any):
    if isinstance(weight_dtype, str):
        candidate = getattr(torch, weight_dtype, None)
        if candidate is not None:
            return candidate
    return weight_dtype


def _get_bnb_config_cls(component_type: str):
    if component_type == "transformers":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "Pipeline BitsAndBytes quantization for text encoders requires transformers with BitsAndBytesConfig."
            ) from exc
        return BitsAndBytesConfig
    try:
        from diffusers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError("Pipeline BitsAndBytes quantization requires diffusers with BitsAndBytesConfig support.") from exc
    return BitsAndBytesConfig


def _bnb_nf4_config(weight_dtype=None, overrides: Optional[Mapping[str, Any]] = None, component_type: str = "diffusers"):
    try:
        bitsandbytes_cls = _get_bnb_config_cls(component_type)
    except ImportError as exc:
        raise ImportError(
            "nf4-bnb quantization requires diffusers[torch] with BitsAndBytes support. "
            "Please install diffusers and bitsandbytes."
        ) from exc

    kwargs = {
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": _normalize_dtype(weight_dtype),
    }
    if isinstance(overrides, Mapping):
        kwargs.update(overrides)
    try:
        return bitsandbytes_cls(**kwargs)
    except Exception as exc:
        # BitsAndBytesConfig.post_init() checks for bitsandbytes package metadata,
        # which raises PackageNotFoundError if bitsandbytes is not installed.
        if "bitsandbytes" in str(exc).lower() or "PackageNotFoundError" in type(exc).__name__:
            raise ImportError(
                "nf4-bnb quantization requires bitsandbytes to be installed. "
                "Please install bitsandbytes or choose a different quantization method."
            ) from exc
        raise


def _get_torchao_config_cls(component_type: str):
    if component_type == "transformers":
        try:
            from transformers import TorchAoConfig
        except ImportError as exc:
            raise ImportError(
                "Pipeline TorchAO quantization for text encoders requires transformers with TorchAoConfig."
            ) from exc
        return TorchAoConfig
    try:
        from diffusers import TorchAoConfig
    except ImportError as exc:
        raise ImportError("Pipeline TorchAO quantization requires diffusers with TorchAoConfig support.") from exc
    return TorchAoConfig


TORCHAO_QUANT_TYPE_CONFIG_MAP = {
    "float8_dynamic_activation_float8_weight": "Float8DynamicActivationFloat8WeightConfig",
    "float8_dynamic_activation_int4_weight": "Float8DynamicActivationInt4WeightConfig",
    "int4_weight_only": "Int4WeightOnlyConfig",
    "int8_dynamic_activation_int8_weight": "Int8DynamicActivationInt8WeightConfig",
    "int8_dynamic_activation_intx_weight": "Int8DynamicActivationIntxWeightConfig",
    "int8_weight_only": "Int8WeightOnlyConfig",
    "float8_weight_only": "Float8WeightOnlyConfig",
}


def _normalize_torchao_quant_type_kwargs(quant_type_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(quant_type_kwargs)
    for key, value in list(normalized.items()):
        if key.endswith("_dtype"):
            normalized[key] = _normalize_dtype(value)
    return normalized


def _build_torchao_quant_type(quant_type: Any, quant_type_kwargs: Mapping[str, Any]):
    from torchao.quantization import quant_api
    from torchao.quantization.quant_api import AOBaseConfig

    if isinstance(quant_type, AOBaseConfig):
        if quant_type_kwargs:
            raise ValueError("quant_type_kwargs cannot be used when quant_type is already a TorchAO config instance.")
        return quant_type
    if not isinstance(quant_type, str):
        raise TypeError(
            "quant_type must be a TorchAO quantization name or AOBaseConfig instance, " f"got {type(quant_type).__name__}"
        )

    config_cls_name = TORCHAO_QUANT_TYPE_CONFIG_MAP.get(quant_type)
    if config_cls_name is None:
        raise ValueError(
            f"Unsupported TorchAO quant_type '{quant_type}'. Supported values: "
            f"{', '.join(sorted(TORCHAO_QUANT_TYPE_CONFIG_MAP))}."
        )
    config_cls = getattr(quant_api, config_cls_name)
    return config_cls(**_normalize_torchao_quant_type_kwargs(quant_type_kwargs))


def _split_torchao_config_kwargs(torchao_cls, override_dict: dict[str, Any], quant_type_kwargs: Mapping[str, Any]):
    outer_keys = {
        name
        for name, parameter in signature(torchao_cls.__init__).parameters.items()
        if name not in {"self", "quant_type"} and parameter.kind is not Parameter.VAR_KEYWORD
    }
    outer_kwargs = {key: override_dict.pop(key) for key in list(override_dict) if key in outer_keys}
    inner_kwargs = dict(override_dict)
    inner_kwargs.update(quant_type_kwargs)
    return outer_kwargs, inner_kwargs


def _build_torchao_config(
    weight_dtype=None,
    overrides: Optional[Mapping[str, Any]] = None,
    component_type: str = "diffusers",
    *,
    default_quant_type: str,
):
    try:
        torchao_cls = _get_torchao_config_cls(component_type)
    except ImportError as exc:
        raise ImportError(
            "TorchAO quantization requires torchao and a compatible TorchAoConfig implementation. "
            "Please install torchao and a recent diffusers/transformers."
        ) from exc

    override_dict = dict(overrides) if isinstance(overrides, Mapping) else {}
    quant_type_override = override_dict.pop("quant_type", None)
    quant_type_kwargs = override_dict.pop("quant_type_kwargs", {})
    if quant_type_kwargs is None:
        quant_type_kwargs = {}
    if not isinstance(quant_type_kwargs, Mapping):
        raise TypeError(f"quant_type_kwargs must be a mapping, got {type(quant_type_kwargs).__name__}")
    quant_type_override = quant_type_override or default_quant_type
    if quant_type_override == "int8_dynamic_activation_int8_weight" and "version" not in quant_type_kwargs:
        quant_type_kwargs = {**quant_type_kwargs, "version": 2}
    if quant_type_override == "int8_dynamic_activation_intx_weight" and "weight_dtype" not in quant_type_kwargs:
        quant_type_kwargs = {**quant_type_kwargs, "weight_dtype": "int4"}
    outer_kwargs, inner_kwargs = _split_torchao_config_kwargs(
        torchao_cls,
        override_dict,
        quant_type_kwargs,
    )
    quant_type = _build_torchao_quant_type(quant_type_override, inner_kwargs)
    return torchao_cls(quant_type=quant_type, **outer_kwargs)


TORCHAO_PIPELINE_PRESET_MAP = {
    "int4-torchao": "int4_weight_only",
    "int8-torchao": "int8_weight_only",
    "int8dq-torchao": "int8_dynamic_activation_int8_weight",
    "int8dq-int4-torchao": "int8_dynamic_activation_intx_weight",
    "fp8-torchao": "float8_dynamic_activation_float8_weight",
    "fp8wo-torchao": "float8_weight_only",
    "fp8-int4-torchao": "float8_dynamic_activation_int4_weight",
}


def _get_quanto_config_cls(component_type: str):
    if component_type == "transformers":
        try:
            from transformers import QuantoConfig
        except ImportError as exc:
            raise ImportError(
                "Pipeline Quanto quantization for text encoders requires transformers with QuantoConfig."
            ) from exc
        return QuantoConfig
    try:
        from diffusers import QuantoConfig
    except ImportError as exc:
        raise ImportError("Pipeline Quanto quantization requires diffusers with QuantoConfig support.") from exc
    return QuantoConfig


def _build_quanto_config(
    weight_dtype=None,
    overrides: Optional[Mapping[str, Any]] = None,
    component_type: str = "diffusers",
    *,
    default_weights_dtype: str,
):
    try:
        quanto_cls = _get_quanto_config_cls(component_type)
    except ImportError as exc:
        raise ImportError(
            "Quanto quantization requires diffusers/transformers with QuantoConfig support. "
            "Please install a recent diffusers/transformers and optimum-quanto."
        ) from exc

    override_dict = dict(overrides) if isinstance(overrides, Mapping) else {}
    if component_type == "transformers":
        weight_key = "weights"
        if "weights_dtype" in override_dict and weight_key not in override_dict:
            override_dict[weight_key] = override_dict.pop("weights_dtype")
    else:
        weight_key = "weights_dtype"
        if "weights" in override_dict and weight_key not in override_dict:
            override_dict[weight_key] = override_dict.pop("weights")
    weights_value = override_dict.pop(weight_key, default_weights_dtype)
    return quanto_cls(**{weight_key: weights_value}, **override_dict)


def _build_quanto_fp8uz_config(
    weight_dtype=None,
    overrides: Optional[Mapping[str, Any]] = None,
    component_type: str = "diffusers",
):
    logger.warning(
        "fp8uz-quanto pipeline quantization maps to diffusers float8 weights (qfloat8_e4m3fn). "
        "Use manual quanto quantization if you require the FP8-NUZ variant."
    )
    return _build_quanto_config(
        weight_dtype=weight_dtype,
        overrides=overrides,
        component_type=component_type,
        default_weights_dtype="float8",
    )


QUANTO_PIPELINE_PRESET_MAP = {
    "int8-quanto": "int8",
    "int4-quanto": "int4",
    "int2-quanto": "int2",
    "fp8-quanto": "float8",
}

PIPELINE_PRESET_BUILDERS: dict[str, Callable[[Any, Optional[Mapping[str, Any]], str], Any]] = {
    "nf4-bnb": _bnb_nf4_config,
    **{
        preset: partial(_build_torchao_config, default_quant_type=quant_type)
        for preset, quant_type in TORCHAO_PIPELINE_PRESET_MAP.items()
    },
    **{
        preset: partial(_build_quanto_config, default_weights_dtype=weights_dtype)
        for preset, weights_dtype in QUANTO_PIPELINE_PRESET_MAP.items()
    },
    "fp8uz-quanto": _build_quanto_fp8uz_config,
}


def get_pipeline_quantization_builder(
    preset: Optional[str],
) -> Optional[Callable[[Any, Optional[Mapping[str, Any]], str], Any]]:
    if preset is None:
        return None
    return PIPELINE_PRESET_BUILDERS.get(str(preset))


def build_gguf_quantization_config(model_path: str):
    import inspect

    candidates = []
    try:
        from diffusers import GGUFQuantizationConfig as DiffusersGGUFQuantizationConfig  # type: ignore

        candidates.append(DiffusersGGUFQuantizationConfig)
    except Exception:
        pass

    try:
        from transformers import GGUFQuantizationConfig as TransformersGGUFQuantizationConfig  # type: ignore

        candidates.append(TransformersGGUFQuantizationConfig)
    except Exception:
        pass

    if not candidates:
        raise ImportError(
            "Loading GGUF checkpoints requires GGUFQuantizationConfig from diffusers or transformers. "
            "Please install a recent version of diffusers or transformers with GGUF support."
        )

    last_error: Exception | None = None
    for gguf_cls in candidates:
        try:
            if hasattr(gguf_cls, "from_pretrained"):
                return gguf_cls.from_pretrained(model_path)
        except Exception as exc:  # pragma: no cover - best effort probing
            last_error = exc
        try:
            signature = inspect.signature(gguf_cls)
        except (TypeError, ValueError):
            signature = None
        try:
            if signature and "model_file" in signature.parameters:
                return gguf_cls(model_file=model_path)
            return gguf_cls(model_path)
        except Exception as exc:  # pragma: no cover - best effort probing
            last_error = exc

    if last_error is not None:
        raise RuntimeError(f"Failed to construct GGUF quantization configuration: {last_error}") from last_error

    raise RuntimeError("Unable to construct GGUF quantization configuration for the provided checkpoint.")


def _quanto_type_map(model_precision: str):
    if model_precision == "no_change":
        return None
    from optimum.quanto import qfloat8, qfloat8_e4m3fnuz, qint2, qint4, qint8

    if model_precision == "int2-quanto":
        quant_level = qint2
    elif model_precision == "int4-quanto":
        quant_level = qint4
    elif model_precision == "int8-quanto":
        quant_level = qint8
    elif model_precision == "fp8-quanto" or model_precision == "fp8uz-quanto":
        if torch.backends.mps.is_available():
            logger.warning(
                "MPS doesn't support dtype float8, you must select another precision level such as bf16, int2, int8, or int8."
            )

            return None
        if model_precision == "fp8-quanto":
            quant_level = qfloat8
        elif model_precision == "fp8uz-quanto":
            quant_level = qfloat8_e4m3fnuz
    else:
        raise ValueError(f"Invalid quantisation level: {model_precision}")

    return quant_level


def _quanto_model(
    model,
    model_precision,
    base_model_precision=None,
    quantize_activations: bool = False,
):
    try:
        from optimum.quanto import QTensor, freeze, quantize

        from simpletuner.helpers.training.quantisation import quanto_workarounds
    except ImportError as e:
        raise ImportError(f"To use Quanto, please install the optimum library: `pip install optimum-quanto`: {e}")

    if model_precision is None:
        model_precision = base_model_precision
    if model is None:
        return model
    if model_precision == "no_change" or model_precision is None:
        logger.info(f"...No quantisation applied to {model.__class__.__name__}.")
        return model

    # Check if model has ramtorch modules - skip quantization entirely if so
    # RamTorch keeps weights on CPU and streams to GPU, which is incompatible with quantization
    has_ramtorch = any(getattr(p, "is_ramtorch", False) for p in model.parameters())
    if has_ramtorch:
        logger.info(
            f"Skipping quanto quantization for {model.__class__.__name__} - model uses RamTorch for CPU offloading. "
            "RamTorch and quantization are incompatible approaches to memory management."
        )
        return model

    logger.info(f"Quantising {model.__class__.__name__}. Using {model_precision}.")
    weight_quant = _quanto_type_map(model_precision)
    extra_quanto_args = {}
    if StateTracker.get_args().model_family in ["sd3", "ltxvideo", "wan", "wan_s2v"]:
        extra_quanto_args["exclude"] = [
            # Norm layers of all types
            "*norm*",  # catches *.norm, .norm1, .norm2, .norm_q, .norm_k, .norm_out, etc.
            "*norm1_context*",  # in case any leftover context norm patterns exist
            "*norm_added_q*",  # norm_added_q
            "*norm_added_k*",  # norm_added_k
            # Projection outputs often better left in higher precision
            "proj_out*",
            # Embeddings or positional embeddings
            "*pos_embed*",
            "*patch_embedding*",
            # Feed forward networks
            "*ffn*",
            # Common shift or scale tables
            "*scale_shift_table*",
            # Blocks or final outputs that are tricky in low precision
            "*norm_out*",
            # Text / condition embedder layers
            "*context_embedder*",  # if that appears anywhere
            "*time_text_embed*",
            "*time_proj*",  # time_proj layers
            "*condition_embedder*",  # catches condition_embedder.image_embedder and text_embedder, etc.
        ]
    elif StateTracker.get_args().model_family == "ltxvideo2":
        extra_quanto_args["exclude"] = [
            # Input/output projection layers
            "patchify_proj",
            "audio_patchify_proj",
            "proj_out",
            "audio_proj_out",
            # Timestep embedding layers - int4 tinygemm requires strict bfloat16 input
            # and these receive float32 sinusoidal embeddings that are cast to bfloat16
            "*adaln*",
            "time_proj",
            "timestep_embedder*",
            # Caption/text projection layers
            "caption_projection*",
            "audio_caption_projection*",
            # Normalization layers
            "*norm*",
        ]
    elif StateTracker.get_args().model_family == "flux":
        extra_quanto_args["exclude"] = [
            "*.norm",
            "*.norm1",
            "*.norm2",
            "*.norm2_context",
            "proj_out",
            "x_embedder",
            "norm_out",
            "context_embedder",
        ]
    if quantize_activations:
        logger.info("Quanto: Freezing model weights and activations")
        extra_quanto_args["activations"] = weight_quant
    else:
        logger.info("Quanto: Freezing model weights only")

    try:
        quantize(model, weights=weight_quant, **extra_quanto_args)
        freeze(model)
    except Exception as e:
        if "out of memory" in str(e).lower():
            logger.error("GPU ran out of memory during quantisation. Use --quantize_via=cpu to use the slower CPU method.")
        raise e

    return model


def _torchao_filter_fn(mod: torch.nn.Module, fqn: str):
    # Skip RamTorch-offloaded modules; TorchAO expects GPU-resident weights.
    if any(getattr(p, "is_ramtorch", False) for p in mod.parameters(recurse=False)):
        return False
    if not isinstance(mod, torch.nn.Linear):
        return False
    # Boogu T2I sends empty tensors through reference-image modules; TorchAO fp8
    # cannot dynamically scale empty activations.
    if fqn.startswith("ref_image_"):
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
        return False
    return True


def mark_torchao_ddp_ignore_params(module: torch.nn.Module) -> int:
    try:
        from torchao.utils import TorchAOBaseTensor
    except ImportError:
        return 0

    def _is_torchao_tensor(value: torch.Tensor) -> bool:
        return isinstance(value, TorchAOBaseTensor) or isinstance(getattr(value, "data", None), TorchAOBaseTensor)

    ignored_names = [name for name, param in module.named_parameters() if _is_torchao_tensor(param)]
    ignored_names.extend(name for name, buffer in module.named_buffers() if _is_torchao_tensor(buffer))
    if not ignored_names:
        return 0

    existing = getattr(module, "_ddp_params_and_buffers_to_ignore", set())
    module._ddp_params_and_buffers_to_ignore = set(existing) | set(ignored_names)
    return len(ignored_names)


def _is_transformerengine_float8_tensor(value: Any) -> bool:
    for candidate in (value, getattr(value, "data", None)):
        tensor_type = type(candidate)
        if (
            tensor_type.__name__ == "Float8Tensor"
            and tensor_type.__module__ == "transformer_engine.pytorch.tensor.float8_tensor"
        ):
            return True
    return False


def mark_transformerengine_ddp_ignore_params(module: torch.nn.Module) -> int:
    ignored_names = [
        name
        for name, param in module.named_parameters()
        if not param.requires_grad and _is_transformerengine_float8_tensor(param)
    ]
    ignored_names.extend(name for name, buffer in module.named_buffers() if _is_transformerengine_float8_tensor(buffer))
    if os.environ.get("SIMPLETUNER_TE_DEBUG_DDP_IGNORE", "") == "1":
        print(
            "[SimpleTuner TE debug] DDP ignore candidates " f"count={len(ignored_names)} names={ignored_names[:8]}",
            flush=True,
        )
    if not ignored_names:
        return 0

    existing = getattr(module, "_ddp_params_and_buffers_to_ignore", set())
    module._ddp_params_and_buffers_to_ignore = set(existing) | set(ignored_names)
    return len(ignored_names)


def _log_torchao_storage_summary(model: torch.nn.Module, model_precision: str) -> None:
    try:
        from torchao.utils import TorchAOBaseTensor
    except ImportError:
        return

    torchao_params = 0
    logical_bytes = 0
    payload_bytes = 0
    for param in model.parameters():
        tensor = param.detach()
        logical_bytes += tensor.numel() * tensor.element_size()
        if not isinstance(tensor, TorchAOBaseTensor):
            continue
        torchao_params += 1
        for attr_name in getattr(tensor, "tensor_data_names", ()):
            payload = getattr(tensor, attr_name, None)
            if torch.is_tensor(payload):
                payload_bytes += payload.numel() * payload.element_size()

    logger.info(
        "TorchAO storage summary for %s: tensor_subclass_params=%s, logical_param_size=%.2f GB, payload_size=%.2f GB",
        model_precision,
        torchao_params,
        logical_bytes / (1024**3),
        payload_bytes / (1024**3),
    )


def _torchao_model(
    model,
    model_precision,
    base_model_precision=None,
    quantize_activations: bool = False,
):
    if model_precision is None:
        model_precision = base_model_precision
    if model is None:
        return model
    if model_precision == "no_change" or model_precision is None:
        logger.info(f"...No quantisation applied to {model.__class__.__name__}.")
        return model

    # Check if model has ramtorch modules - skip quantization entirely if so
    # RamTorch keeps weights on CPU and streams to GPU, which is incompatible with quantization
    has_ramtorch = any(getattr(p, "is_ramtorch", False) for p in model.parameters())
    if has_ramtorch:
        logger.info(
            f"Skipping torchao quantization for {model.__class__.__name__} - model uses RamTorch for CPU offloading. "
            "RamTorch and quantization are incompatible approaches to memory management."
        )
        return model

    try:
        import torchao
        from torchao.prototype.quantized_training import int8_weight_only_quantized_training
        from torchao.quantization import quantize_
        from torchao.quantization.quant_api import (
            Float8DynamicActivationFloat8WeightConfig,
            Float8DynamicActivationInt4WeightConfig,
            Float8WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig,
            Int8DynamicActivationIntxWeightConfig,
        )

        from simpletuner.helpers.training.quantisation import torchao_workarounds
    except ImportError as e:
        raise ImportError(f"To use torchao, please install the torchao library: `pip install torchao`: {e}")
    logger.info(f"Quantising {model.__class__.__name__}. Using {model_precision}.")
    if quantize_activations:
        logger.warning("Activation quantisation is not used in TorchAO. This will be ignored.")

    if model_precision == "int8-torchao":
        quantize_(
            model,
            int8_weight_only_quantized_training(),  # , filter_fn=_torchao_filter_fn
        )
    elif model_precision == "int8dq-torchao":
        quantize_(
            model,
            Int8DynamicActivationInt8WeightConfig(version=2),
            filter_fn=_torchao_filter_fn,
        )
    elif model_precision == "int8dq-int4-torchao":
        quantize_(
            model,
            Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4),
            filter_fn=_torchao_filter_fn,
        )
    elif model_precision == "fp8-torchao":
        quantize_(
            model,
            Float8DynamicActivationFloat8WeightConfig(),
            filter_fn=_torchao_filter_fn,
        )
    elif model_precision == "fp8wo-torchao":
        quantize_(
            model,
            Float8WeightOnlyConfig(),
            filter_fn=_torchao_filter_fn,
        )
    elif model_precision == "fp8-int4-torchao":
        quantize_(
            model,
            Float8DynamicActivationInt4WeightConfig(),
            filter_fn=_torchao_filter_fn,
        )

    else:
        raise ValueError(
            f"Invalid quantisation level. model_precision={model_precision}, base_model_precision={base_model_precision}"
        )

    _log_torchao_storage_summary(model, model_precision)
    return model


def _fp8_native_model(
    model,
    model_precision,
    base_model_precision=None,
    quantize_activations: bool = False,
):
    if model_precision is None:
        model_precision = base_model_precision
    if model is None:
        return model
    if model_precision == "no_change" or model_precision is None:
        logger.info(f"...No quantisation applied to {model.__class__.__name__}.")
        return model
    if model_precision != "fp8-native":
        raise ValueError(
            f"Invalid native FP8 quantisation level. model_precision={model_precision}, "
            f"base_model_precision={base_model_precision}"
        )
    if quantize_activations:
        logger.warning("Activation quantisation flag is ignored for fp8-native; activations are scaled inside _scaled_mm.")

    from simpletuner.helpers.training.quantisation.fp8_native import (
        log_fp8_native_storage_summary,
        patch_peft_fp8_native_dispatcher,
        replace_linear_with_fp8_native,
    )

    compute_dtype = next((param.dtype for param in model.parameters() if torch.is_floating_point(param)), torch.bfloat16)
    logger.info("Quantising %s using fp8-native.", model.__class__.__name__)
    patch_peft_fp8_native_dispatcher()
    converted = replace_linear_with_fp8_native(model, _torchao_filter_fn, compute_dtype)
    logger.info("Converted %s Linear layer(s) to native FP8.", converted)
    log_fp8_native_storage_summary(model, model_precision)
    return model


def _transformerengine_autocast_context(te, recipe):
    autocast = getattr(te, "fp8_autocast", None) or getattr(te, "autocast", None)
    if autocast is None:
        raise RuntimeError("TransformerEngine does not expose an FP8 autocast context.")
    parameters = signature(autocast).parameters
    recipe_kwarg = "fp8_recipe" if "fp8_recipe" in parameters else "recipe"
    if recipe_kwarg == "recipe":
        return autocast(enabled=True, recipe=recipe)
    return autocast(enabled=True, fp8_recipe=recipe)


def _make_transformerengine_checkpoint_context_fn(te, recipe):
    def _simpletuner_te_checkpoint_context_fn():
        return (
            _transformerengine_autocast_context(te, recipe),
            _transformerengine_autocast_context(te, recipe),
        )

    return _simpletuner_te_checkpoint_context_fn


def _attach_transformerengine_checkpoint_context(model: torch.nn.Module, te, recipe) -> None:
    context_fn = _make_transformerengine_checkpoint_context_fn(te, recipe)
    for module in model.modules():
        module._simpletuner_te_checkpoint_context_fn = context_fn


def _transformerengine_fp8_output_filter_fn(fqn: str) -> bool:
    if os.environ.get("SIMPLETUNER_TE_FP8_ATTENTION_OUTPUT", "").lower() not in ("1", "true", "yes"):
        return False
    return any(fqn.endswith(f".{target}") for target in ("to_q", "to_k", "to_v"))


def _wrap_transformerengine_fp8_output_forward(module: torch.nn.Module) -> None:
    if hasattr(module, "_simpletuner_te_fp8_output_original_forward"):
        return
    module._simpletuner_te_fp8_output_original_forward = module.forward
    module.forward = partial(module.forward, fp8_output=True)


def _wrap_transformerengine_frozen_weight_cache_forward(module: torch.nn.Module) -> None:
    if hasattr(module, "_simpletuner_te_weight_cache_original_forward"):
        return

    module._simpletuner_te_weight_cache_original_forward = module.forward
    module._simpletuner_te_weight_cache_initialized = False
    original_forward = module.forward

    def _simpletuner_te_weight_cache_forward(*args, **kwargs):
        if "is_first_microbatch" not in kwargs:
            kwargs["is_first_microbatch"] = not module._simpletuner_te_weight_cache_initialized
            module._simpletuner_te_weight_cache_initialized = True
        return original_forward(*args, **kwargs)

    module.forward = _simpletuner_te_weight_cache_forward


def _replace_linears_with_transformerengine(
    module: torch.nn.Module,
    te,
    compute_dtype: torch.dtype,
    recipe=None,
    fp8_model_init_enabled: bool = False,
    prefix: str = "",
) -> int:
    converted = 0
    for name, child in list(module.named_children()):
        fqn = f"{prefix}.{name}" if prefix else name
        if _transformerengine_filter_fn(child, fqn):
            fp8_model_init = getattr(te, "fp8_model_init", None)
            if fp8_model_init_enabled and fp8_model_init is None:
                raise RuntimeError("TransformerEngine does not expose fp8_model_init for FP8 weight storage.")
            context = fp8_model_init(enabled=True) if fp8_model_init_enabled else nullcontext()
            with context:
                te_linear = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    params_dtype=compute_dtype,
                    device=child.weight.device,
                )
            te_linear.train(child.training)
            with torch.no_grad():
                te_linear.weight.copy_(
                    child.weight.detach().to(dtype=te_linear.weight.dtype, device=te_linear.weight.device)
                )
                te_linear.weight.requires_grad_(child.weight.requires_grad)
                if child.bias is not None:
                    te_linear.bias.copy_(child.bias.detach().to(dtype=te_linear.bias.dtype, device=te_linear.bias.device))
                    te_linear.bias.requires_grad_(child.bias.requires_grad)
            if _transformerengine_fp8_output_filter_fn(fqn):
                _wrap_transformerengine_fp8_output_forward(te_linear)
            if not te_linear.weight.requires_grad and os.environ.get("SIMPLETUNER_TE_FROZEN_WEIGHT_CACHE", "").lower() in (
                "1",
                "true",
                "yes",
            ):
                _wrap_transformerengine_frozen_weight_cache_forward(te_linear)
            _wrap_transformerengine_debug_forward(te_linear, fqn)
            setattr(module, name, te_linear)
            converted += 1
            continue
        converted += _replace_linears_with_transformerengine(child, te, compute_dtype, recipe, fp8_model_init_enabled, fqn)
    return converted


def _transformerengine_filter_fn(mod: torch.nn.Module, fqn: str):
    if any(getattr(p, "is_ramtorch", False) for p in mod.parameters(recurse=False)):
        return False
    if not isinstance(mod, torch.nn.Linear):
        return False
    if fqn.startswith("ref_image_"):
        return False
    if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
        return False
    if "audio" in fqn or "timestep_embedder" in fqn or "_adaln" in fqn:
        return False
    args = StateTracker.get_args()
    if "lora" in str(getattr(args, "model_type", "")):
        if fqn == "proj_in":
            return False
        if os.environ.get("SIMPLETUNER_TE_LORA_CONVERT_ALL", "").lower() in ("1", "true", "yes"):
            return True
        if fqn == "proj_out":
            return False
        return any(
            fqn.endswith(f".{target}")
            for target in (
                "to_q",
                "to_k",
                "to_v",
                "to_qkv",
                "to_kv",
                "to_added_qkv",
                "to_out.0",
            )
        )
    return True


def _wrap_transformerengine_fp8_forward(model: torch.nn.Module, te, recipe) -> None:
    if hasattr(model, "_simpletuner_te_original_forward"):
        return
    model._simpletuner_te_original_forward = model.forward
    model._simpletuner_te_fp8_recipe = recipe
    original_forward = model.forward

    def _simpletuner_te_fp8_forward(*args, **kwargs):
        with _transformerengine_autocast_context(te, recipe):
            return original_forward(*args, **kwargs)

    model.forward = _simpletuner_te_fp8_forward


def _transformerengine_model(
    model,
    model_precision,
    base_model_precision=None,
    quantize_activations: bool = False,
):
    if model_precision is None:
        model_precision = base_model_precision
    if model is None:
        return model
    if model_precision == "no_change" or model_precision is None:
        logger.info(f"...No quantisation applied to {model.__class__.__name__}.")
        return model
    if model_precision != "fp8-transformerengine":
        raise ValueError(
            f"Invalid TransformerEngine FP8 quantisation level. model_precision={model_precision}, "
            f"base_model_precision={base_model_precision}"
        )
    if quantize_activations:
        logger.warning(
            "Activation quantisation flag is ignored for fp8-transformerengine; TE autocast owns FP8 activation scaling."
        )

    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format
    except ImportError as e:
        raise ImportError(
            "fp8-transformerengine requires TransformerEngine. Install it with "
            "`pip install 'simpletuner[transformerengine]'`."
        ) from e

    fp8_available = te.is_fp8_available(return_reason=True)
    if isinstance(fp8_available, tuple):
        available, reason = fp8_available
    else:
        available, reason = bool(fp8_available), ""
    if not available:
        raise RuntimeError(f"TransformerEngine FP8 is not available on this system. {reason}".strip())

    compute_dtype = next((param.dtype for param in model.parameters() if torch.is_floating_point(param)), torch.bfloat16)
    recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
    args = StateTracker.get_args()
    fp8_model_init_enabled = getattr(args, "model_type", None) != "full"
    logger.info(
        "Quantising %s using TransformerEngine FP8. fp8_model_init=%s.",
        model.__class__.__name__,
        fp8_model_init_enabled,
    )
    converted = _replace_linears_with_transformerengine(model, te, compute_dtype, recipe, fp8_model_init_enabled)
    if converted == 0:
        raise RuntimeError(f"TransformerEngine FP8 did not find any eligible Linear layers in {model.__class__.__name__}.")
    _attach_transformerengine_checkpoint_context(model, te, recipe)
    _wrap_transformerengine_fp8_forward(model, te, recipe)
    logger.info("Converted %s Linear layer(s) to TransformerEngine FP8.", converted)
    return model


def _default_sdnq_use_quantized_matmul(
    model_precision: str,
    sdnq_use_torch_compile: bool,
    sdnq_fp8_mm_supported: bool,
) -> bool:
    if model_precision == "fp8-sdnq":
        return sdnq_use_torch_compile and sdnq_fp8_mm_supported
    return sdnq_use_torch_compile


def _sdnq_model(
    model,
    model_precision,
    base_model_precision=None,
    quantize_activations: bool = False,
):
    """
    Quantize a model using SDNQ (SD.Next Quantization Engine).

    SDNQ works on AMD, Apple, and NVIDIA hardware without platform-specific gating.

    Precision recommendations from Disty0:
    - Full finetune: uint8, uint16, fp16
    - LoRA training (frozen weights): int8, int6, int5, uint5, uint4, uint3, uint2
    - Below 5 bits: use_svd=True with svd_steps=8 recommended
    """
    if model_precision is None:
        model_precision = base_model_precision
    if model is None:
        return model
    if model_precision == "no_change" or model_precision is None:
        logger.info(f"...No quantisation applied to {model.__class__.__name__}.")
        return model

    # Map precision string to SDNQ weights_dtype before importing SDNQ so invalid
    # user input reports as a configuration error, not as an optional dependency
    # error from the installed SDNQ package.
    sdnq_dtype_map = {
        "int8-sdnq": "int8",
        "uint8-sdnq": "uint8",
        "int16-sdnq": "int16",
        "uint16-sdnq": "uint16",
        "fp16-sdnq": "fp16",
        "fp8-sdnq": "float8_e4m3fn",
        "int6-sdnq": "int6",
        "int5-sdnq": "int5",
        "uint5-sdnq": "uint5",
        "uint4-sdnq": "uint4",
        "uint3-sdnq": "uint3",
        "uint2-sdnq": "uint2",
    }
    weights_dtype = sdnq_dtype_map.get(model_precision)
    if weights_dtype is None:
        raise ValueError(f"Invalid SDNQ precision level: {model_precision}")

    args = StateTracker.get_args()
    sdnq_compile_mode = getattr(args, "sdnq_compile_mode", "auto")
    if sdnq_compile_mode not in (None, "auto"):
        if "sdnq.common" in sys.modules:
            logger.warning(
                "SDNQ was already imported before --sdnq_compile_mode=%s could be applied. "
                "Set SDNQ_USE_TORCH_COMPILE before process startup to force this mode.",
                sdnq_compile_mode,
            )
        else:
            os.environ["SDNQ_USE_TORCH_COMPILE"] = "1" if sdnq_compile_mode == "compile" else "0"

    try:
        # Silence sdnq startup logs
        logging.getLogger("sdnq").setLevel(logging.WARNING)
        import sdnq.common as sdnq_common
        from sdnq.training import sdnq_training_post_load_quant
    except ImportError as e:
        raise ImportError(f"To use SDNQ, please install the sdnq library: `pip install sdnq`: {e}")

    sdnq_fp8_mm_supported = getattr(sdnq_common, "is_fp8_mm_supported", False)
    if callable(sdnq_fp8_mm_supported):
        sdnq_fp8_mm_supported = sdnq_fp8_mm_supported()
    sdnq_tensorwise_fp8_matmul = getattr(sdnq_common, "use_tensorwise_fp8_matmul", False)
    sdnq_use_torch_compile = getattr(sdnq_common, "use_torch_compile", False)

    logger.info(f"Quantising {model.__class__.__name__} using SDNQ. Precision: {model_precision}.")
    weights_dtype = getattr(args, "sdnq_weights_dtype", None) or weights_dtype

    # Determine bit depth for SVD recommendation
    # Below 5 bits: use SVD with 8 steps (per Disty0's recommendation)
    low_bit_dtypes = {"int5", "uint5", "uint4", "uint3", "uint2", "int4", "int3", "int2"}
    use_svd = weights_dtype in low_bit_dtypes
    svd_steps = 8 if use_svd else 2
    use_svd = getattr(args, "sdnq_use_svd", None) if getattr(args, "sdnq_use_svd", None) is not None else use_svd
    svd_rank = getattr(args, "sdnq_svd_rank", None) or 32
    svd_steps = getattr(args, "sdnq_svd_steps", None) or svd_steps

    # Determine quantization device
    # Use GPU for faster quantization: load to CPU, quantize on CUDA, return to CPU
    quantize_via = getattr(args, "quantize_via", "accelerator")
    if quantize_via == "cpu":
        quantization_device = "cpu"
        return_device = "cpu"
    elif quantize_via == "accelerator" and torch.cuda.is_available():
        # Fast GPU quantization: quantize on CUDA, return to original device
        quantization_device = "cuda"
        return_device = model.device if hasattr(model, "device") else None
    else:
        quantization_device = None
        return_device = None

    # SDNQ maintains its own module skip keys list, so we rely on add_skip_keys=True
    # Only add custom exclusions if needed
    # Use "." prefix for root-level modules (per Disty0's recommendation)
    modules_to_not_convert = []
    if args.model_family == "flux":
        # Use ".proj_out" for root level proj_out in Flux (inner layers also have proj_out)
        modules_to_not_convert.append(".proj_out")
    if getattr(args, "sdnq_modules_to_not_convert", None):
        modules_to_not_convert.extend(args.sdnq_modules_to_not_convert)

    # Determine matmul dtype: INT8 preferred for consumer GPUs, FP8 for datacenter
    quantized_matmul_dtype = getattr(args, "sdnq_quantized_matmul_dtype", None)
    if quantized_matmul_dtype in (None, "auto"):
        quantized_matmul_dtype = "float8_e4m3fn" if model_precision == "fp8-sdnq" else "int8"
    group_size = getattr(args, "sdnq_group_size", None)
    if group_size is None:
        group_size = -1 if model_precision == "fp8-sdnq" else 32
    use_quantized_matmul = getattr(args, "sdnq_use_quantized_matmul", None)
    if use_quantized_matmul is None:
        use_quantized_matmul = _default_sdnq_use_quantized_matmul(
            model_precision,
            sdnq_use_torch_compile,
            sdnq_fp8_mm_supported,
        )
    use_hadamard = bool(getattr(args, "sdnq_use_hadamard", False))
    hadamard_group_size = getattr(args, "sdnq_hadamard_group_size", None) or 128
    use_static_quantization = getattr(args, "sdnq_use_static_quantization", None)
    if use_static_quantization is None:
        use_static_quantization = True
    use_stochastic_rounding = getattr(args, "sdnq_use_stochastic_rounding", None)
    if use_stochastic_rounding is None:
        use_stochastic_rounding = True
    dequantize_fp32 = getattr(args, "sdnq_dequantize_fp32", None)
    if dequantize_fp32 is None:
        dequantize_fp32 = True

    # Warn about low-bit precision for full finetune
    if args.model_type == "full" and weights_dtype in low_bit_dtypes:
        logger.warning(
            f"Using {weights_dtype} precision for full finetune is not recommended. "
            f"Consider uint8, uint16, or fp16 for better training stability."
        )

    sdnq_kwargs = {
        "weights_dtype": weights_dtype,
        "quantized_matmul_dtype": quantized_matmul_dtype,
        "group_size": group_size,
        "hadamard_group_size": hadamard_group_size,
        "svd_rank": svd_rank,
        "svd_steps": svd_steps,
        "use_svd": use_svd,
        "use_hadamard": use_hadamard,
        "use_grad_ckpt": getattr(args, "gradient_checkpointing", True),
        "use_quantized_matmul": use_quantized_matmul,
        "use_static_quantization": use_static_quantization,
        "use_stochastic_rounding": use_stochastic_rounding,
        "dequantize_fp32": dequantize_fp32,
        "non_blocking": False,
        "add_skip_keys": True,  # Let SDNQ handle module exclusions
        "quantization_device": quantization_device,
        "return_device": return_device,
        "modules_to_not_convert": modules_to_not_convert,
        "modules_to_not_use_matmul": getattr(args, "sdnq_modules_to_not_use_matmul", None),
        "modules_dtype_dict": getattr(args, "sdnq_modules_dtype_dict", None),
        "modules_quant_config": getattr(args, "sdnq_modules_quant_config", None),
    }
    sdnq_supported_kwargs = set(signature(sdnq_training_post_load_quant).parameters)
    unsupported_requested = []
    if use_hadamard and "use_hadamard" not in sdnq_supported_kwargs:
        unsupported_requested.append("sdnq_use_hadamard")
    if getattr(args, "sdnq_hadamard_group_size", None) is not None and "hadamard_group_size" not in sdnq_supported_kwargs:
        unsupported_requested.append("sdnq_hadamard_group_size")
    if (
        getattr(args, "sdnq_modules_to_not_use_matmul", None) is not None
        and "modules_to_not_use_matmul" not in sdnq_supported_kwargs
    ):
        unsupported_requested.append("sdnq_modules_to_not_use_matmul")
    if getattr(args, "sdnq_modules_dtype_dict", None) is not None and "modules_dtype_dict" not in sdnq_supported_kwargs:
        unsupported_requested.append("sdnq_modules_dtype_dict")
    if getattr(args, "sdnq_modules_quant_config", None) is not None and "modules_quant_config" not in sdnq_supported_kwargs:
        unsupported_requested.append("sdnq_modules_quant_config")
    if unsupported_requested:
        raise ValueError(
            "The installed SDNQ package does not support these SimpleTuner options: " + ", ".join(unsupported_requested)
        )
    sdnq_kwargs = {key: value for key, value in sdnq_kwargs.items() if key in sdnq_supported_kwargs}

    try:
        model = sdnq_training_post_load_quant(model, **sdnq_kwargs)
        logger.info(
            "SDNQ config: weights_dtype=%s, matmul_dtype=%s, group_size=%s, qmm=%s, compile=%s, fp8_mm=%s, tensorwise_fp8=%s, svd=%s, hadamard=%s.",
            weights_dtype,
            quantized_matmul_dtype,
            group_size,
            use_quantized_matmul,
            sdnq_use_torch_compile,
            sdnq_fp8_mm_supported,
            sdnq_tensorwise_fp8_matmul,
            use_svd,
            use_hadamard,
        )
        if use_svd:
            logger.info(f"SDNQ: Using SVD with {svd_steps} steps for {weights_dtype} precision.")
    except Exception as e:
        if "out of memory" in str(e).lower():
            logger.error("GPU ran out of memory during SDNQ quantisation. Use --quantize_via=cpu to use CPU quantisation.")
        raise e

    return model


def get_quant_fn(base_model_precision):
    """
    Determine the quantization function based on the base model precision.

    Args:
        base_model_precision (str): The precision specification for the base model.

    Returns:
        function: The corresponding quantization function.

    Raises:
        ValueError: If the precision specification is unsupported.
    """
    precision = base_model_precision.lower()
    if precision == "no_change":
        return None
    if "quanto" in precision:
        return _quanto_model
    elif precision in MANUAL_FP8_NATIVE_PRESETS:
        return _fp8_native_model
    elif precision in MANUAL_TRANSFORMERENGINE_PRESETS:
        return _transformerengine_model
    elif "torchao" in precision:
        return _torchao_model
    elif "sdnq" in precision:
        return _sdnq_model
    else:
        return None


def quantise_model(
    model=None,
    text_encoders: list = None,
    controlnet=None,
    ema=None,
    args=None,
    return_dict: bool = False,
):
    """
    Quantizes the provided models using the specified precision settings.

    Args:
        model: The base model to quanti
        text_encoders: A list of zero or more text encoders to quantize.
        controlnet: The ControlNet model to quantize.
        ema: An EMAModel to quantize.
        args: An object containing precision settings and other arguments.

    Returns:
        tuple: A tuple containing the quantized models in the order:
               (model, text_encoders, controlnet)
    """
    text_encoder_1, text_encoder_2, text_encoder_3, text_encoder_4 = (
        None,
        None,
        None,
        None,
    )
    if text_encoders is not None:
        if len(text_encoders) > 0:
            text_encoder_1 = text_encoders[0]
        if len(text_encoders) > 1:
            text_encoder_2 = text_encoders[1]
        if len(text_encoders) > 2:
            text_encoder_3 = text_encoders[2]
        if len(text_encoders) > 3:
            text_encoder_4 = text_encoders[3]

    models = [
        (
            model,
            {
                "quant_fn": get_quant_fn(args.base_model_precision),
                "model_precision": args.base_model_precision,
                "quantize_activations": args.quantize_activations,
            },
        ),
        (
            controlnet,
            {
                "quant_fn": get_quant_fn(args.base_model_precision),
                "model_precision": args.base_model_precision,
                "quantize_activations": args.quantize_activations,
            },
        ),
        (
            text_encoder_1,
            {
                "quant_fn": get_quant_fn(args.text_encoder_1_precision),
                "model_precision": args.text_encoder_1_precision,
                "base_model_precision": args.base_model_precision,
            },
        ),
        (
            text_encoder_2,
            {
                "quant_fn": get_quant_fn(args.text_encoder_2_precision),
                "model_precision": args.text_encoder_2_precision,
                "base_model_precision": args.base_model_precision,
            },
        ),
        (
            text_encoder_3,
            {
                "quant_fn": get_quant_fn(args.text_encoder_3_precision),
                "model_precision": args.text_encoder_3_precision,
                "base_model_precision": args.base_model_precision,
            },
        ),
        (
            text_encoder_4,
            {
                "quant_fn": get_quant_fn(args.text_encoder_4_precision),
                "model_precision": args.text_encoder_4_precision,
                "base_model_precision": args.base_model_precision,
            },
        ),
        (
            ema,
            {
                "quant_fn": get_quant_fn(args.base_model_precision),
                "model_precision": args.base_model_precision,
                "quantize_activations": args.quantize_activations,
            },
        ),
    ]

    # Iterate over the models and apply quantization if the model is not None
    for i, (model, quant_args) in enumerate(models):
        quant_fn = quant_args["quant_fn"]
        if quant_fn is None:
            continue
        if model is not None:
            quant_args_combined = {
                "model_precision": quant_args["model_precision"],
                "base_model_precision": quant_args.get("base_model_precision", args.base_model_precision),
                "quantize_activations": quant_args.get("quantize_activations", args.quantize_activations),
            }
            logger.info(f"Quantising {model.__class__.__name__} with {quant_args_combined}")
            models[i] = (quant_fn(model, **quant_args_combined), quant_args)

    # Unpack the quantized models
    (
        model,
        controlnet,
        text_encoder_1,
        text_encoder_2,
        text_encoder_3,
        text_encoder_4,
        ema,
    ) = [model for model, _ in models]

    # repack text encoders
    text_encoders = []
    if text_encoder_1 is not None:
        text_encoders.append(text_encoder_1)
    if text_encoder_2 is not None:
        text_encoders.append(text_encoder_2)
    if text_encoder_3 is not None:
        text_encoders.append(text_encoder_3)
    if text_encoder_4 is not None:
        text_encoders.append(text_encoder_4)
    if len(text_encoders) == 0:
        text_encoders = None

    if return_dict:
        return {
            "model": model,
            "text_encoders": text_encoders,
            "controlnet": controlnet,
            "ema": ema,
        }

    return (
        model,
        text_encoders,
        controlnet,
        ema,
    )
