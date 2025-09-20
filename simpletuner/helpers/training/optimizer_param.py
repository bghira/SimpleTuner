import os

import accelerate
import torch
from accelerate.logging import get_logger

logger = get_logger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")

is_optimi_available = False
from simpletuner.helpers.training.optimizers.adamw_bfloat16 import AdamWBF16
from simpletuner.helpers.training.optimizers.adamw_schedulefree import AdamWScheduleFreeKahan
from simpletuner.helpers.training.optimizers.soap import SOAP

try:
    from optimum.quanto import QTensor
except:
    pass

try:
    from torchao.optim import AdamFp8 as AOAdamFp8
    from torchao.optim import AdamW4bit as AOAdamW4Bit
    from torchao.optim import AdamW8bit as AOAdamW8Bit
    from torchao.optim import AdamWFp8 as AOAdamWFp8
    from torchao.optim import CPUOffloadOptimizer as AOCPUOffloadOptimizer

    if torch.backends.mps.is_available():
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
except Exception as e:
    print("You need torchao installed for its low-precision optimizers.")
    raise e

try:
    import optimi

    is_optimi_available = True
except:
    logger.error("Could not load optimi library. Please install `torch-optimi` for better memory efficiency.")

is_bitsandbytes_available = False
try:
    import bitsandbytes

    is_bitsandbytes_available = True
except:
    if torch.cuda.is_available():
        print("Could not load bitsandbytes library. BnB-specific optimisers and other functionality will be unavailable.")

# Some optimizers are not available in multibackend bitsandbytes as of January 2025.
is_ademamix_available = False
if is_bitsandbytes_available:
    if "AdEMAMix" in dir(bitsandbytes.optim):
        is_ademamix_available = True

is_prodigy_available = False
try:
    import prodigyplus

    is_prodigy_available = True
except:
    if torch.cuda.is_available():
        print("Could not load prodigyplus library. Prodigy will not be available.")


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
    "ao-adamw8bit": {
        "gradient_precision": "bf16",
        "precision": "any",
        "default_settings": {
            "betas": (0.9, 0.999),
            "weight_decay": 1e-2,
            "eps": 1e-6,
        },
        "class": AOAdamW8Bit,
    },
    "ao-adamw4bit": {
        "gradient_precision": "bf16",
        "precision": "any",
        "default_settings": {
            "betas": (0.9, 0.999),
            "weight_decay": 1e-2,
            "eps": 1e-6,
        },
        "class": AOAdamW4Bit,
    },
    "ao-adamfp8": {
        "gradient_precision": "bf16",
        "precision": "any",
        "default_settings": {
            "betas": (0.9, 0.999),
            "weight_decay": 1e-2,
            "eps": 1e-6,
        },
        "class": AOAdamFp8,
    },
    "ao-adamwfp8": {
        "gradient_precision": "bf16",
        "precision": "any",
        "default_settings": {
            "betas": (0.9, 0.999),
            "weight_decay": 1e-2,
            "eps": 1e-6,
        },
        "class": AOAdamWFp8,
    },
    "adamw_schedulefree": {
        "precision": "any",
        "override_lr_scheduler": True,
        "is_schedulefree": True,
        "can_warmup": True,
        "default_settings": {
            "betas": (0.9, 0.999),
            "weight_decay": 1e-2,
            "eps": 1e-8,
        },
        "class": AdamWScheduleFreeKahan,
    },
    "adamw_schedulefree+aggressive": {
        "precision": "any",
        "override_lr_scheduler": True,
        "is_schedulefree": True,
        "can_warmup": True,
        "default_settings": {
            "betas": (0.9, 0.999),
            "weight_decay": 1e-3,
            "eps": 1e-6,
        },
        "class": AdamWScheduleFreeKahan,
    },
    "adamw_schedulefree+no_kahan": {
        "precision": "any",
        "override_lr_scheduler": True,
        "is_schedulefree": True,
        "can_warmup": True,
        "default_settings": {
            "betas": (0.9, 0.999),
            "weight_decay": 1e-3,
            "eps": 1e-6,
            "use_kahan": False,
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
    "soap": {
        "precision": "any",
        "default_settings": {
            "betas": (0.95, 0.95),
            "shampoo_beta": -1,
            "eps": 1e-8,
            "weight_decay": 0.01,
            "precondition_frequency": 10,
            "max_precond_dim": 10000,
            "merge_dims": False,
            "precondition_1d": False,
            "normalize_grads": False,
            "data_format": "channels_first",
            "correct_bias": True,
        },
        "class": SOAP,
    },
}

if is_bitsandbytes_available:
    optimizer_choices.update(
        {
            "bnb-adagrad": {
                "precision": "any",
                "default_settings": {
                    "lr_decay": 0,
                    "weight_decay": 0,
                    "initial_accumulator_value": 0,
                    "eps": 1e-10,
                    "min_8bit_size": 4096,
                    "percentile_clipping": 100,
                },
                "class": bitsandbytes.optim.Adagrad,
            },
            "bnb-adagrad8bit": {
                "precision": "any",
                "default_settings": {
                    "lr_decay": 0,
                    "weight_decay": 0,
                    "initial_accumulator_value": 0,
                    "eps": 1e-10,
                    "min_8bit_size": 4096,
                    "percentile_clipping": 100,
                },
                "class": bitsandbytes.optim.Adagrad8bit,
            },
            "bnb-adam": {
                "precision": "any",
                "default_settings": {
                    "betas": (0.9, 0.999),
                    "eps": 1e-08,
                    "weight_decay": 0,
                    "amsgrad": False,
                    "min_8bit_size": 4096,
                    "percentile_clipping": 100,
                },
                "class": bitsandbytes.optim.Adam,
            },
            "bnb-adam8bit": {
                "precision": "any",
                "default_settings": {
                    "betas": (0.9, 0.999),
                    "eps": 1e-08,
                    "weight_decay": 0,
                    "amsgrad": False,
                    "min_8bit_size": 4096,
                    "percentile_clipping": 100,
                },
                "class": bitsandbytes.optim.Adam8bit,
            },
            "bnb-adamw": {
                "precision": "any",
                "default_settings": {
                    "betas": (0.9, 0.999),
                    "weight_decay": 1e-2,
                    "eps": 1e-6,
                },
                "class": bitsandbytes.optim.AdamW,
            },
            "bnb-adamw8bit": {
                "precision": "any",
                "default_settings": {
                    "betas": (0.9, 0.999),
                    "weight_decay": 1e-2,
                    "eps": 1e-6,
                },
                "class": bitsandbytes.optim.AdamW8bit,
            },
            "bnb-adamw-paged": {
                "precision": "any",
                "default_settings": {
                    "betas": (0.9, 0.999),
                    "weight_decay": 1e-2,
                    "eps": 1e-6,
                },
                "class": bitsandbytes.optim.PagedAdamW,
            },
            "bnb-adamw8bit-paged": {
                "precision": "any",
                "default_settings": {
                    "betas": (0.9, 0.999),
                    "weight_decay": 1e-2,
                    "eps": 1e-6,
                },
                "class": bitsandbytes.optim.PagedAdamW8bit,
            },
            "bnb-lion": {
                "precision": "any",
                "default_settings": {
                    "betas": (0.9, 0.99),
                    "weight_decay": 0.0,
                    "min_8bit_size": 4096,
                },
                "class": bitsandbytes.optim.Lion,
            },
            "bnb-lion8bit": {
                "precision": "any",
                "default_settings": {
                    "betas": (0.9, 0.99),
                    "weight_decay": 0.0,
                    "min_8bit_size": 4096,
                },
                "class": bitsandbytes.optim.Lion8bit,
            },
            "bnb-lion-paged": {
                "precision": "any",
                "default_settings": {
                    "betas": (0.9, 0.99),
                    "weight_decay": 0.0,
                    "min_8bit_size": 4096,
                },
                "class": bitsandbytes.optim.PagedLion,
            },
            "bnb-lion8bit-paged": {
                "precision": "any",
                "default_settings": {
                    "betas": (0.9, 0.99),
                    "weight_decay": 0.0,
                    "min_8bit_size": 4096,
                },
                "class": bitsandbytes.optim.PagedLion8bit,
            },
        }
    )

if is_ademamix_available:
    optimizer_choices.update(
        {
            "bnb-ademamix": {
                "precision": "any",
                "default_settings": {
                    "betas": (0.9, 0.999, 0.9999),
                    "alpha": 5.0,
                    "t_alpha": None,
                    "t_beta3": None,
                    "eps": 1e-08,
                    "weight_decay": 0.01,
                    "min_8bit_size": 4096,
                },
                "class": bitsandbytes.optim.AdEMAMix,
            },
            "bnb-ademamix8bit": {
                "precision": "any",
                "default_settings": {
                    "betas": (0.9, 0.999, 0.9999),
                    "alpha": 5.0,
                    "t_alpha": None,
                    "t_beta3": None,
                    "eps": 1e-08,
                    "weight_decay": 0.01,
                    "min_8bit_size": 4096,
                },
                "class": bitsandbytes.optim.AdEMAMix8bit,
            },
            "bnb-ademamix-paged": {
                "precision": "any",
                "default_settings": {
                    "betas": (0.9, 0.999, 0.9999),
                    "alpha": 5.0,
                    "t_alpha": None,
                    "t_beta3": None,
                    "eps": 1e-08,
                    "weight_decay": 0.01,
                    "min_8bit_size": 4096,
                },
                "class": bitsandbytes.optim.PagedAdEMAMix,
            },
            "bnb-ademamix8bit-paged": {
                "precision": "any",
                "default_settings": {
                    "betas": (0.9, 0.999, 0.9999),
                    "alpha": 5.0,
                    "t_alpha": None,
                    "t_beta3": None,
                    "eps": 1e-08,
                    "weight_decay": 0.01,
                    "min_8bit_size": 4096,
                },
                "class": bitsandbytes.optim.PagedAdEMAMix8bit,
            },
        }
    )

if is_prodigy_available:
    optimizer_choices.update(
        {
            "prodigy": {
                "precision": "any",
                "override_lr_scheduler": False,
                "is_schedulefree": True,
                "can_warmup": False,
                "default_settings": {
                    "lr": 1.0,
                    "betas": (0.9, 0.99),
                    "beta3": None,
                    "weight_decay": 0.0,
                    "weight_decay_by_lr": True,
                    "use_bias_correction": False,
                    "d0": 1e-6,
                    "d_coef": 1,
                    "prodigy_steps": 0,
                    "use_speed": False,
                    "eps": 1e-8,
                    "split_groups": True,
                    "split_groups_mean": True,
                    "factored": True,
                    "factored_fp32": True,
                    "fused_back_pass": False,
                    "use_stableadamw": True,
                    "use_muon_pp": False,
                    "use_cautious": False,
                    "use_grams": False,
                    "use_adopt": False,
                    "stochastic_rounding": True,
                },
                "class": prodigyplus.prodigy_plus_schedulefree.ProdigyPlusScheduleFree,
            }
        }
    )

args_to_optimizer_mapping = {
    "use_adafactor_optimizer": "adafactor",
    "use_prodigy_optimizer": "prodigy",
    "use_dadaptation_optimizer": "dadaptation",
    "adam_bfloat16": "adamw_bf16",
    "use_8bit_adam": "adamw8bit",
}

deprecated_optimizers = {
    "dadaptation": "D-adaptation optimiser has been removed due to issues with precision levels and convergence. Please use adamw_schedulefree instead.",
    "adafactor": "Adafactor optimiser has been removed in favour of optimi-stableadamw, which offers improved memory efficiency and convergence.",
    "adamw8bit": "AdamW8Bit has been removed in favour of optimi-adamw optimiser, which offers better low-precision support. Please use this or adamw_bf16 instead.",
}


def convert_arg_to_parameters(args):
    """--optimizer_config can have a format like --optimizer_config=eps=1e-6,weight_decay=0.0"""
    out = {}
    if args.optimizer_config is not None and args.optimizer_config:
        optimizer_params = [param.split("=") for param in args.optimizer_config.split(",")]
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
        if args.optimizer == "prodigy":
            prodigy_steps = args.prodigy_steps
            if prodigy_steps and prodigy_steps > 0:
                optimizer_params["prodigy_steps"] = int(prodigy_steps)
            print(f"Using Prodigy optimiser with {optimizer_params['prodigy_steps']} steps of learning rate adjustment.")
        return optimizer_class, optimizer_details
    else:
        raise ValueError(f"Optimizer {optimizer} not found.")


def is_lr_scheduler_disabled(optimizer: str):
    """Check if the optimizer has a built-in LR scheduler"""
    is_disabled = False
    if optimizer in optimizer_choices:
        is_disabled = optimizer_choices.get(optimizer).get("override_lr_scheduler", False)
    return is_disabled


def is_lr_schedulefree(optimizer: str):
    """
    Check if the optimizer has ScheduleFree logic.

    This is separate from the disabling of LR schedulers, because some optimizers
    that contain ScheduleFree logic (Prodigy) can use an LR scheduler.
    """
    is_schedulefree = False
    if optimizer in optimizer_choices:
        is_schedulefree = optimizer_choices.get(optimizer).get("is_schedulefree", False)
    return is_schedulefree


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


def map_deprecated_optimizer_parameter(optimizer: str) -> str:
    return args_to_optimizer_mapping.get(optimizer, None)


def is_optimizer_bf16(optimizer: str) -> bool:
    optimizer_precision = optimizer_choices.get(optimizer, {}).get("precision", "fp32")
    if optimizer_precision in ["any", "bf16"]:
        return True
    return False


def is_optimizer_grad_fp32(optimizer: str) -> bool:
    optimizer_precision = optimizer_choices.get(optimizer, {}).get("gradient_precision", None)
    if optimizer_precision == "fp32":
        return True
    return False


def cpu_offload_optimizer(
    params_to_optimize,
    optimizer_cls,
    optimizer_parameters: dict,
    offload_gradients: bool = True,
    fused: bool = True,
    offload_mechanism: str = None,
):
    if not offload_mechanism or offload_mechanism == "none":
        return optimizer_cls(params_to_optimize, **optimizer_parameters)
    if offload_mechanism != "torchao":
        raise ValueError(f"Unknown CPU optimiser offload mechanism: {offload_mechanism}")

    if offload_gradients:
        optimizer_parameters["offload_gradients"] = offload_gradients
    if fused:
        optimizer_parameters["fused"] = fused

    optimizer_parameters["optimizer_class"] = optimizer_cls

    return AOCPUOffloadOptimizer(params_to_optimize, **optimizer_parameters)


def determine_optimizer_class_with_config(args, use_deepspeed_optimizer, is_quantized, enable_adamw_bf16) -> tuple:
    extra_optimizer_args = {}
    if use_deepspeed_optimizer:
        optimizer_class = accelerate.utils.DummyOptim
        extra_optimizer_args["lr"] = float(args.learning_rate)
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["eps"] = args.adam_epsilon
        extra_optimizer_args["weight_decay"] = args.adam_weight_decay
        default_settings = extra_optimizer_args
        optimizer_details = {}
    elif is_quantized and not enable_adamw_bf16 and args.optimizer == "adamw_bf16":
        logger.error(f"When --base_model_default_dtype=fp32, AdamWBF16 may not be used. Switching to AdamW.")
        optimizer_class, optimizer_details = optimizer_parameters("optimi-adamw", args)
        default_settings = optimizer_details.get("default_settings")
    else:
        optimizer_class, optimizer_details = optimizer_parameters(args.optimizer, args)
        default_settings = optimizer_details.get("default_settings")
    if optimizer_details.get("can_warmup", False):
        logger.info(f"Optimizer contains LR scheduler, warmup steps will be set to {args.lr_warmup_steps}.")
        default_settings["warmup_steps"] = args.lr_warmup_steps
    logger.info(f"cls: {optimizer_class}, settings: {default_settings}")
    return default_settings, optimizer_class


def determine_params_to_optimize(
    args,
    model,
    model_type_label,
    lycoris_wrapped_network,
):
    params_to_optimize = list(
        filter(
            lambda p: p.requires_grad,
            model.get_trained_component(unwrap_model=False).parameters(),
        )
    )
    if args.train_text_encoder:
        # add the first text encoder's parameters
        for text_encoder in model.text_encoders:
            if "t5" in str(text_encoder.__class__).lower():
                logger.warning(f"{text_encoder.__class__} does not support finetuning, skipping model.")
                continue
            params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, text_encoder.parameters()))

    if args.model_type == "lora" and args.lora_type == "lycoris":
        if lycoris_wrapped_network is not None:
            params_to_optimize = list(filter(lambda p: p.requires_grad, lycoris_wrapped_network.parameters()))
        else:
            raise Exception("Lycoris wrapped network is None, cannot optimize parameters.")

    return params_to_optimize


def create_optimizer_params_with_decay(model, weight_decay=0.01, learning_rate=None):
    """
    Separate model parameters into two groups:
    - Parameters that will experience weight decay (most weights)
    - Parameters that won't (biases, layernorm/embedding weights)

    Args:
        model: The model whose parameters to optimize
        weight_decay: Weight decay value for regularization
        learning_rate: Optional learning rate (if different per group)

    Returns:
        List of parameter group dictionaries for PyTorch optimizer
    """
    decay = []
    no_decay = []

    # Get all named parameters from the model
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Parameters that should NOT have weight decay:
        # - Anything with 'bias' in the name
        # - LayerNorm weights (often named with 'norm', 'ln', or 'layernorm')
        # - Embedding weights
        # - Batch normalization parameters
        if any(nd in name.lower() for nd in ["bias", "norm", "embedding", "embed", "bn"]):
            no_decay.append(param)
        else:
            decay.append(param)

    # Create parameter groups
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    # Add learning rate if specified
    if learning_rate is not None:
        for group in param_groups:
            group["lr"] = learning_rate

    # Log the parameter distribution
    logger.info(f"Parameter groups: {len(decay)} with weight decay, " f"{len(no_decay)} without weight decay")

    return param_groups


def create_optimizer_with_param_groups(
    model,
    optimizer_class,
    optimizer_parameters,
    use_parameter_groups=True,
    cpu_offload_config=None,
):
    """
    Create an optimizer with proper parameter grouping for weight decay.

    Args:
        model: The model to optimize
        optimizer_class: The optimizer class to use
        optimizer_parameters: Dict of optimizer parameters
        use_parameter_groups: Whether to use parameter groups for weight decay
        cpu_offload_config: Optional CPU offload configuration

    Returns:
        Configured optimizer instance
    """
    # Extract weight decay from optimizer parameters
    weight_decay = optimizer_parameters.pop("weight_decay", 0.01)
    learning_rate = optimizer_parameters.get("lr", None)

    if use_parameter_groups and weight_decay > 0:
        # Create parameter groups with appropriate weight decay settings
        param_groups = create_optimizer_params_with_decay(model, weight_decay=weight_decay, learning_rate=learning_rate)

        # Remove lr from optimizer_parameters if it was set in param groups
        if learning_rate is not None:
            optimizer_parameters.pop("lr", None)
    else:
        # Use all parameters with the same weight decay
        param_groups = filter(lambda p: p.requires_grad, model.parameters())
        optimizer_parameters["weight_decay"] = weight_decay

    # Handle CPU offload if configured
    if cpu_offload_config:
        return cpu_offload_optimizer(param_groups, optimizer_class, optimizer_parameters, **cpu_offload_config)

    # Create and return the optimizer
    return optimizer_class(param_groups, **optimizer_parameters)
