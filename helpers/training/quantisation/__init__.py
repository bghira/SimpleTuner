from helpers.training.multi_process import should_log
import logging
import torch, os

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel(logging.ERROR)


def _quanto_type_map(model_precision: str):
    if model_precision == "no_change":
        return None
    from optimum.quanto import (
        qfloat8,
        qfloat8_e4m3fnuz,
        qint8,
        qint4,
        qint2,
    )

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

            return model
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
        from helpers.training.quantisation import quanto_workarounds
        from optimum.quanto import (
            freeze,
            quantize,
            QTensor,
        )
    except ImportError as e:
        raise ImportError(
            f"To use Quanto, please install the optimum library: `pip install optimum-quanto`: {e}"
        )

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
    if quantize_activations:
        logger.info("Freezing model weights and activations")
        extra_quanto_args["activations"] = weight_quant
        extra_quanto_args["exclude"] = [
            "*.norm",
            "*.norm1",
            "*.norm2",
            "*.norm2_context",
            "proj_out",
        ]
    else:
        logger.info("Freezing model weights only")

    try:
        quantize(model, weights=weight_quant, **extra_quanto_args)
        freeze(model)
    except Exception as e:
        if "out of memory" in str(e).lower():
            logger.error(
                "GPU ran out of memory during quantisation. Use --quantize_via=cpu to use the slower CPU method."
            )
        raise e

    return model


def _torchao_filter_fn(mod: torch.nn.Module, fqn: str):
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
        from helpers.training.quantisation import torchao_workarounds
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        from torchao.prototype.quantized_training import (
            int8_weight_only_quantized_training,
        )
        import torchao
        from torchao.quantization import quantize_
    except ImportError as e:
        raise ImportError(
            f"To use torchao, please install the torchao library: `pip install torchao`: {e}"
        )
    logger.info(f"Quantising {model.__class__.__name__}. Using {model_precision}.")
    if quantize_activations:
        logger.warning(
            "Activation quantisation is not used in TorchAO. This will be ignored."
        )

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
        raise ValueError(f"Invalid quantisation level: {base_model_precision}")

    return model


def quantise_model(
    unet, transformer, text_encoder_1, text_encoder_2, text_encoder_3, controlnet, args
):
    if "quanto" in args.base_model_precision.lower():
        logger.info("Loading Quanto. This may take a few minutes.")
        quant_fn = _quanto_model
    elif "torchao" in args.base_model_precision.lower():
        logger.info("Loading TorchAO. This may take a few minutes.")
        quant_fn = _torchao_model
    if transformer is not None:
        transformer = quant_fn(
            transformer,
            model_precision=args.base_model_precision,
            quantize_activations=args.quantize_activations,
        )
    if unet is not None:
        unet = quant_fn(
            unet,
            model_precision=args.base_model_precision,
            quantize_activations=args.quantize_activations,
        )
    if controlnet is not None:
        controlnet = quant_fn(
            controlnet,
            model_precision=args.base_model_precision,
            quantize_activations=args.quantize_activations,
        )

    if text_encoder_1 is not None:
        text_encoder_1 = quant_fn(
            text_encoder_1,
            model_precision=args.text_encoder_1_precision,
            base_model_precision=args.base_model_precision,
        )
    if text_encoder_2 is not None:
        text_encoder_2 = quant_fn(
            text_encoder_2,
            model_precision=args.text_encoder_2_precision,
            base_model_precision=args.base_model_precision,
        )
    if text_encoder_3 is not None:
        text_encoder_3 = quant_fn(
            text_encoder_3,
            model_precision=args.text_encoder_3_precision,
            base_model_precision=args.base_model_precision,
        )

    return unet, transformer, text_encoder_1, text_encoder_2, text_encoder_3, controlnet
