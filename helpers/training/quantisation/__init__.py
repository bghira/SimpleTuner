from helpers.training.multi_process import should_log
from helpers.training.state_tracker import StateTracker
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
        logger.info("Freezing model weights and activations")
        extra_quanto_args["activations"] = weight_quant
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
        raise ValueError(
            f"Invalid quantisation level. model_precision={model_precision}, base_model_precision={base_model_precision}"
        )

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
                "base_model_precision": quant_args.get(
                    "base_model_precision", args.base_model_precision
                ),
                "quantize_activations": quant_args.get(
                    "quantize_activations", args.quantize_activations
                ),
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
