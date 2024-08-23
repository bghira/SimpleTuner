import os
from accelerate.logging import get_logger
import accelerate
import torch

logger = get_logger(__name__, log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
logger.setLevel(target_level)

is_optimi_available = False
from helpers.training.optimizers.adamw_bfloat16 import AdamWBF16
from helpers.training.optimizers.adamw_schedulefree import AdamWScheduleFreeKahan

try:
    from optimum.quanto import QTensor
except:
    pass

try:
    import optimi

    is_optimi_available = True
except:
    logger.error(
        "Could not load optimi library. Please install `torch-optimi` for better memory efficiency."
    )

optimizer_choices = {
    "adamw_bf16": {
        "precision": "bf16",
        "default_settings": {
            "betas": (0.9, 0.999),
            "weight_decay": 1e-2,
            "eps": 1e-6,
        },
        "class": AdamWBF16,
    },
    "adamw_schedulefree": {
        "precision": "any",
        "override_lr_scheduler": True,
        "can_warmup": True,
        "default_settings": {
            "betas": (0.9, 0.999),
            "weight_decay": 1e-2,
            "eps": 1e-8,
        },
        "class": AdamWScheduleFreeKahan,
    },
    "optimi-stableadamw": {
        "precision": "any",
        "default_settings": {
            "betas": (0.9, 0.99),
            "weight_decay": 1e-2,
            "eps": 1e-6,
            "decouple_lr": False,
            "max_lr": None,
            "kahan_sum": True,
            "foreach": True,
        },
        "class": optimi.StableAdamW,
    },
    "optimi-adamw": {
        "precision": "any",
        "default_settings": {
            "betas": (0.9, 0.99),
            "eps": 1e-6,
            "weight_decay": 0.0,
            "decouple_lr": False,
            "kahan_sum": True,
            "max_lr": None,
        },
        "class": optimi.AdamW,
    },
    "optimi-lion": {
        "precision": "any",
        "default_settings": {
            "betas": (0.9, 0.99),
            "weight_decay": 0.0,
            "decouple_lr": False,
            "max_lr": None,
            "kahan_sum": True,
            "foreach": True,
        },
        "class": optimi.Lion,
    },
    "optimi-radam": {
        "precision": "any",
        "default_settings": {
            "betas": (0.9, 0.99),
            "weight_decay": 0.0,
            "eps": 1e-6,
            "decouple_wd": True,
            "decouple_lr": False,
            "kahan_sum": True,
            "foreach": True,
        },
        "class": optimi.RAdam,
    },
    "optimi-ranger": {
        "precision": "any",
        "default_settings": {
            "betas": (0.9, 0.99),
            "weight_decay": 0.0,
            "eps": 1e-6,
            "k": 6,
            "alpha": 0.5,
            "decouple_wd": True,
            "decouple_lr": False,
            "max_lr": None,
            "kahan_sum": True,
            "foreach": True,
        },
        "class": optimi.Ranger,
    },
    "optimi-adan": {
        "precision": "any",
        "default_settings": {
            "betas": (0.98, 0.92, 0.999),
            "weight_decay": 2e-2,
            "eps": 1e-6,
            "decouple_lr": False,
            "max_lr": None,
            "adam_wd": False,
            "kahan_sum": True,
            "foreach": True,
        },
        "class": optimi.Adan,
    },
    "optimi-adam": {
        "precision": "any",
        "default_settings": {
            "betas": (0.9, 0.99),
            "eps": 1e-6,
            "weight_decay": 0.0,
            "decouple_wd": False,
            "decouple_lr": False,
            "kahan_sum": True,
            "max_lr": None,
        },
        "class": optimi.Adam,
    },
    "optimi-sgd": {
        "precision": "any",
        "default_settings": {
            "momentum": 0,
            "weight_decay": 0.0,
            "dampening": False,
            "decouple_wd": False,
            "decouple_lr": False,
            "max_lr": None,
            "torch_init": False,
            "kahan_sum": True,
            "foreach": True,
        },
        "class": optimi.SGD,
    },
}

args_to_optimizer_mapping = {
    "use_adafactor_optimizer": "adafactor",
    "use_prodigy_optimizer": "prodigy",
    "use_dadaptation_optimizer": "dadaptation",
    "adam_bfloat16": "adamw_bf16",
    "use_8bit_adam": "adamw8bit",
}

deprecated_optimizers = {
    "prodigy": "Prodigy optimiser has been removed due to issues with precision levels and convergence. Please use adamw_schedulefree instead.",
    "dadaptation": "D-adaptation optimiser has been removed due to issues with precision levels and convergence. Please use adamw_schedulefree instead.",
    "adafactor": "Adafactor optimiser has been removed in favour of optimi-stableadamw, which offers improved memory efficiency and convergence.",
    "adamw8bit": "AdamW8Bit has been removed in favour of optimi-adamw optimiser, which offers better low-precision support. Please use this or adamw_bf16 instead.",
}


def convert_arg_to_parameters(args):
    """--optimizer_config can have a format like --optimizer_config=eps=1e-6,weight_decay=0.0"""
    out = {}
    if args.optimizer_config is not None and args.optimizer_config:
        optimizer_params = [
            param.split("=") for param in args.optimizer_config.split(",")
        ]
        for param in optimizer_params:
            if "." in param[1]:
                out[param[0]] = float(param[1])
            elif str(param[1]).isdigit():
                out[param[0]] = int(param[1])
            elif param[1].lower() == "true":
                out[param[0]] = True
            elif param[1].lower() == "false":
                out[param[0]] = False
            elif param[1].lower() == "none":
                out[param[0]] = None
            elif "e-" in param[1]:
                out[param[0]] = float(param[1])
            else:
                out[param[0]] = param[1]
        return out
    if args.optimizer_beta1 is not None and args.optimizer_beta2 is not None:
        # the user has supplied a beta1 and beta2 value
        out["betas"] = tuple([args.optimizer_beta1, args.optimizer_beta2])

    return out


def optimizer_parameters(optimizer, args):
    """Return the parameters for the optimizer"""
    if optimizer in optimizer_choices:
        optimizer_details = optimizer_choices.get(optimizer)
        optimizer_class = optimizer_choices.get(optimizer).get("class")
        optimizer_params = optimizer_choices.get(optimizer).get("default_settings")
        optimizer_params.update(convert_arg_to_parameters(args))
        if args.optimizer_release_gradients and "optimi-" in optimizer:
            optimizer_params["gradient_release"] = True
        optimizer_details["default_settings"] = optimizer_params
        return optimizer_class, optimizer_details
    else:
        raise ValueError(f"Optimizer {optimizer} not found.")


def is_lr_scheduler_disabled(optimizer: str):
    """Check if the optimizer has a built-in LR scheduler"""
    is_disabled = False
    if optimizer in optimizer_choices:
        is_disabled = optimizer_choices.get(optimizer).get(
            "override_lr_scheduler", False
        )
    return is_disabled


def show_optimizer_defaults(optimizer: str = None):
    """we'll print the defaults on a single line, eg. foo=bar, buz=baz"""
    if optimizer is None:
        for key in optimizer_choices:
            print(f"{key}={optimizer_choices[key].get('default_settings')}")
    else:
        print(f"{optimizer}={optimizer_choices.get(optimizer).get('default_settings')}")


def is_optimizer_deprecated(optimizer: str) -> bool:
    if optimizer in deprecated_optimizers:
        raise ValueError(deprecated_optimizers.get(optimizer))


def map_args_to_optimizer(args) -> str:
    for key in args_to_optimizer_mapping:
        if hasattr(args, key) and getattr(args, key):
            print(
                "[ERROR] The use of --use_adafactor_optimizer, --use_prodigy_optimizer, --use_dadaptation_optimizer, --adam_bfloat16, and --use_8bit_adam is deprecated. Please use --optimizer instead."
            )
            return args_to_optimizer_mapping.get(key)
    return getattr(args, "optimizer")


def map_deprecated_optimizer_parameter(optimizer: str) -> str:
    return args_to_optimizer_mapping.get(optimizer, None)


def is_optimizer_bf16(optimizer: str) -> bool:
    optimizer_precision = optimizer_choices.get(optimizer, {}).get("precision", "fp32")
    if optimizer_precision in ["any", "bf16"]:
        return True
    return False


def determine_optimizer_class_with_config(
    args, use_deepspeed_optimizer, is_quantized, enable_adamw_bf16
) -> tuple:
    extra_optimizer_args = {}
    if use_deepspeed_optimizer:
        optimizer_class = accelerate.utils.DummyOptim
        extra_optimizer_args["lr"] = float(args.learning_rate)
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["eps"] = args.adam_epsilon
        extra_optimizer_args["weight_decay"] = args.adam_weight_decay
        default_settings = extra_optimizer_args
    elif is_quantized and not enable_adamw_bf16:
        logger.error(
            f"When --base_model_default_dtype=fp32, AdamWBF16 may not be used. Switching to AdamW."
        )
        optimizer_class, optimizer_details = optimizer_parameters("optimi-adamw", args)
    else:
        optimizer_class, optimizer_details = optimizer_parameters(args.optimizer, args)
        default_settings = optimizer_details.get("default_settings")
    if optimizer_details.get("can_warmup", False):
        logger.info(
            f"Optimizer contains LR scheduler, warmup steps will be set to {args.lr_warmup_steps}."
        )
        default_settings["warmup_steps"] = args.lr_warmup_steps
    logger.info(f"cls: {optimizer_class}, settings: {default_settings}")
    return default_settings, optimizer_class


def determine_params_to_optimize(
    args,
    controlnet,
    unet,
    transformer,
    text_encoder_1,
    text_encoder_2,
    model_type_label,
    lycoris_wrapped_network,
):
    if args.model_type == "full":
        if args.controlnet:
            params_to_optimize = controlnet.parameters()
        elif unet is not None:
            params_to_optimize = list(
                filter(lambda p: p.requires_grad, unet.parameters())
            )
        elif transformer is not None:
            params_to_optimize = list(
                filter(lambda p: p.requires_grad, transformer.parameters())
            )
        if args.train_text_encoder:
            raise ValueError(
                "Full model tuning does not currently support text encoder training."
            )
    elif "lora" in args.model_type:
        if args.controlnet:
            raise ValueError(
                "SimpleTuner does not currently support training a ControlNet LoRA."
            )
        if unet is not None:
            params_to_optimize = list(
                filter(lambda p: p.requires_grad, unet.parameters())
            )
        if transformer is not None:
            params_to_optimize = list(
                filter(lambda p: p.requires_grad, transformer.parameters())
            )
        if args.train_text_encoder:
            if args.sd3 or args.pixart_sigma:
                raise ValueError(
                    f"{model_type_label} does not support finetuning the text encoders, as T5 does not benefit from it."
                )
            else:
                # add the first text encoder's parameters
                params_to_optimize = params_to_optimize + list(
                    filter(lambda p: p.requires_grad, text_encoder_1.parameters())
                )
                # if text_encoder_2 is not None, add its parameters
                if text_encoder_2 is None and not args.flux:
                    # but not flux. it has t5 as enc 2.
                    params_to_optimize = params_to_optimize + list(
                        filter(lambda p: p.requires_grad, text_encoder_2.parameters())
                    )

        if args.lora_type == "lycoris" and lycoris_wrapped_network is not None:
            params_to_optimize = list(
                filter(lambda p: p.requires_grad, lycoris_wrapped_network.parameters())
            )

    return params_to_optimize
