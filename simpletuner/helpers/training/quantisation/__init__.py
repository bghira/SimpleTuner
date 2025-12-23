import logging
import os
from typing import Any, Callable, Mapping, Optional

import torch

from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel(logging.ERROR)

PIPELINE_QUANTIZATION_PRESETS = {"nf4-bnb", "int4-torchao"}
MANUAL_QUANTO_PRESETS = {"int2-quanto", "int4-quanto", "int8-quanto", "fp8-quanto", "fp8uz-quanto"}
MANUAL_TORCHAO_PRESETS = {"int8-torchao", "fp8-torchao"}
MANUAL_SDNQ_PRESETS = {
    "int8-sdnq",
    "uint8-sdnq",
    "int16-sdnq",
    "uint16-sdnq",
    "fp16-sdnq",
    "int6-sdnq",
    "int5-sdnq",
    "uint5-sdnq",
    "uint4-sdnq",
    "uint3-sdnq",
    "uint2-sdnq",
}
MANUAL_QUANTIZATION_PRESETS = MANUAL_QUANTO_PRESETS | MANUAL_TORCHAO_PRESETS | MANUAL_SDNQ_PRESETS


def _normalize_dtype(weight_dtype: Any):
    if isinstance(weight_dtype, str):
        candidate = getattr(torch, weight_dtype, None)
        if candidate is not None:
            return candidate
    return weight_dtype


def _bnb_nf4_config(weight_dtype=None, overrides: Optional[Mapping[str, Any]] = None):
    try:
        from diffusers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "nf4-bnb quantization requires diffusers[torch] with BitsAndBytes support. Please install diffusers and bitsandbytes."
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
        return BitsAndBytesConfig(**kwargs)
    except Exception as exc:
        # BitsAndBytesConfig.post_init() checks for bitsandbytes package metadata,
        # which raises PackageNotFoundError if bitsandbytes is not installed.
        if "bitsandbytes" in str(exc).lower() or "PackageNotFoundError" in type(exc).__name__:
            raise ImportError(
                "nf4-bnb quantization requires bitsandbytes to be installed. "
                "Please install bitsandbytes or choose a different quantization method."
            ) from exc
        raise


def _torchao_int4_config(weight_dtype=None, overrides: Optional[Mapping[str, Any]] = None):
    try:
        from torchao.quantization import Int4WeightOnlyConfig
        from transformers import TorchAoConfig
    except ImportError as exc:
        raise ImportError(
            "TorchAO int4 quantization requires torchao and transformers with TorchAoConfig. Please install torchao and transformers>=4.39."
        ) from exc

    override_dict = dict(overrides) if isinstance(overrides, Mapping) else {}
    quant_type_override = override_dict.pop("quant_type", None)
    group_size = override_dict.pop("group_size", None)
    if quant_type_override is None:
        quant_kwargs = {}
        if group_size is not None:
            quant_kwargs["group_size"] = group_size
        quant_type_override = Int4WeightOnlyConfig(**quant_kwargs)
    return TorchAoConfig(quant_type=quant_type_override, **override_dict)


PIPELINE_PRESET_BUILDERS: dict[str, Callable[[Any, Optional[Mapping[str, Any]]], Any]] = {
    "nf4-bnb": _bnb_nf4_config,
    "int4-torchao": _torchao_int4_config,
}


def get_pipeline_quantization_builder(preset: Optional[str]) -> Optional[Callable[[Any, Optional[Mapping[str, Any]]], Any]]:
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

    logger.info(f"Quantising {model.__class__.__name__}. Using {model_precision}.")
    weight_quant = _quanto_type_map(model_precision)
    extra_quanto_args = {}
    if StateTracker.get_args().model_family in ["sd3", "ltxvideo", "wan"]:
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
    # don't convert the output module
    if fqn == "proj_out":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True


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

    try:
        import torchao
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        from torchao.prototype.quantized_training import int8_weight_only_quantized_training
        from torchao.quantization import quantize_

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
    elif model_precision == "fp8-torchao":
        model = convert_to_float8_training(
            model,
            module_filter_fn=_torchao_filter_fn,
            config=Float8LinearConfig(pad_inner_dim=True),
        )

    else:
        raise ValueError(
            f"Invalid quantisation level. model_precision={model_precision}, base_model_precision={base_model_precision}"
        )

    return model


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

    try:
        from sdnq.common import use_torch_compile as sdnq_triton_available
        from sdnq.training import sdnq_training_post_load_quant
    except ImportError as e:
        raise ImportError(f"To use SDNQ, please install the sdnq library: `pip install sdnq`: {e}")

    logger.info(f"Quantising {model.__class__.__name__} using SDNQ. Precision: {model_precision}.")

    # Map precision string to SDNQ weights_dtype
    sdnq_dtype_map = {
        "int8-sdnq": "int8",
        "uint8-sdnq": "uint8",
        "int16-sdnq": "int16",
        "uint16-sdnq": "uint16",
        "fp16-sdnq": "fp16",
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

    # Determine bit depth for SVD recommendation
    # Below 5 bits: use SVD with 8 steps (per Disty0's recommendation)
    low_bit_dtypes = {"int5", "uint5", "uint4", "uint3", "uint2", "int4", "int3", "int2"}
    use_svd = weights_dtype in low_bit_dtypes
    svd_steps = 8 if use_svd else 2

    args = StateTracker.get_args()

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

    # Determine matmul dtype: INT8 preferred for consumer GPUs, FP8 for datacenter
    # For now, default to INT8 as it works on more hardware
    quantized_matmul_dtype = "int8"

    # Warn about low-bit precision for full finetune
    if args.model_type == "full" and weights_dtype in low_bit_dtypes:
        logger.warning(
            f"Using {weights_dtype} precision for full finetune is not recommended. "
            f"Consider uint8, uint16, or fp16 for better training stability."
        )

    try:
        model = sdnq_training_post_load_quant(
            model,
            weights_dtype=weights_dtype,
            quantized_matmul_dtype=quantized_matmul_dtype,
            group_size=32,
            svd_rank=32,
            svd_steps=svd_steps,
            use_svd=use_svd,
            use_grad_ckpt=getattr(args, "gradient_checkpointing", True),
            use_quantized_matmul=sdnq_triton_available,
            use_static_quantization=True,
            use_stochastic_rounding=True,
            dequantize_fp32=True,
            non_blocking=False,
            add_skip_keys=True,  # Let SDNQ handle module exclusions
            quantization_device=quantization_device,
            return_device=return_device,
            modules_to_not_convert=modules_to_not_convert,
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
