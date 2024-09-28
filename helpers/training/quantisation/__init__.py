from helpers.training.multi_process import should_log
import logging
import torch, os

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel(logging.ERROR)


def _quanto_model(model, model_precision, base_model_precision=None):
    try:
        from optimum.quanto import (
            freeze,
            quantize,
            qfloat8,
            qfloat8_e4m3fnuz,
            qint8,
            qint4,
            qint2,
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
    if model_precision == "int2-quanto":
        weight_quant = qint2
    elif model_precision == "int4-quanto":
        if torch.cuda.is_available():
            logger.error(
                "int4-quanto is only supported on A100 and H100 GPUs, but other GPUs would support int2-quanto, int8-quanto or fp8-quanto... waiting 10 seconds for you to cancel."
            )
            import time

            time.sleep(10)
        weight_quant = qint4
    elif model_precision == "int8-quanto":
        weight_quant = qint8
    elif model_precision == "fp8-quanto" or model_precision == "nf4-quanto":
        if torch.backends.mps.is_available():
            logger.warning(
                "MPS doesn't support dtype float8, you must select another precision level such as bf16, int2, int8, or int8."
            )

            return model
        if model_precision == "fp8-quanto":
            weight_quant = qfloat8
        elif model_precision == "nf4-quanto":
            weight_quant = qfloat8_e4m3fnuz
    else:
        raise ValueError(f"Invalid quantisation level: {base_model_precision}")
    quantize(model, weights=weight_quant)
    logger.info("Freezing model.")
    freeze(model)

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


def _torchao_model(model, model_precision, base_model_precision=None):
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

    if model_precision == "int8-torchao":
        quantize_(
            model,
            int8_weight_only_quantized_training(),  # , filter_fn=_torchao_filter_fn
        )
    elif model_precision == "fp8-torchao":
        if not torch.cuda.is_available():
            raise ValueError(
                "fp8-torchao is only supported on CUDA enabled GPUs. int8-quanto can be used everywhere else."
            )
        logger.error(
            "fp8-torchao requires the latest pytorch nightly build, but int8-torchao, int8-quanto, or fp8-quanto may be used instead."
        )
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
        transformer = quant_fn(transformer, args.base_model_precision)
    if unet is not None:
        unet = quant_fn(unet, args.base_model_precision)
    if controlnet is not None:
        controlnet = quant_fn(controlnet, args.base_model_precision)

    if text_encoder_1 is not None:
        text_encoder_1 = quant_fn(
            text_encoder_1, args.text_encoder_1_precision, args.base_model_precision
        )
    if text_encoder_2 is not None:
        text_encoder_2 = quant_fn(
            text_encoder_2, args.text_encoder_2_precision, args.base_model_precision
        )
    if text_encoder_3 is not None:
        text_encoder_3 = quant_fn(
            text_encoder_3, args.text_encoder_3_precision, args.base_model_precision
        )

    return unet, transformer, text_encoder_1, text_encoder_2, text_encoder_3, controlnet
