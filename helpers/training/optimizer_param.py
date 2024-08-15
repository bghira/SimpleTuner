import os
from accelerate.logging import get_logger
import accelerate
import torch

logger = get_logger(__name__, log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
logger.setLevel(target_level)

def determine_optimizer_class_with_config(args, use_deepspeed_optimizer, is_quantized, enable_adamw_bf16) -> tuple:
    extra_optimizer_args = {}
    if use_deepspeed_optimizer:
        optimizer_class = accelerate.utils.DummyOptim
        extra_optimizer_args["lr"] = float(args.learning_rate)
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["eps"] = args.adam_epsilon
        extra_optimizer_args["weight_decay"] = args.adam_weight_decay
    elif args.use_prodigy_optimizer:
        logger.info("Using Prodigy optimizer. Experimental.")
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
            )

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warn(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        extra_optimizer_args["lr"] = float(args.learning_rate)
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["beta3"] = args.prodigy_beta3
        extra_optimizer_args["weight_decay"] = args.prodigy_weight_decay
        extra_optimizer_args["eps"] = args.prodigy_epsilon
        extra_optimizer_args["decouple"] = args.prodigy_decouple
        extra_optimizer_args["use_bias_correction"] = args.prodigy_use_bias_correction
        extra_optimizer_args["safeguard_warmup"] = args.prodigy_safeguard_warmup
        extra_optimizer_args["d_coef"] = args.prodigy_learning_rate
    elif args.adam_bfloat16:
        if is_quantized and not enable_adamw_bf16:
            logger.error(
                f"When --base_model_default_dtype=fp32, AdamWBF16 may not be used. Reverting to AdamW. You may use other optimizers, such as Adafactor."
            )
            optimizer_class = torch.optim.AdamW
        else:
            logger.info("Using bf16 AdamW optimizer with stochastic rounding.")
            from helpers.training import adam_bfloat16

            optimizer_class = adam_bfloat16.AdamWBF16
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["lr"] = float(args.learning_rate)
    elif args.use_8bit_adam:
        logger.info("Using 8bit AdamW optimizer.")
        try:
            import bitsandbytes as bnb  # type: ignore
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_class = bnb.optim.AdamW8bit
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["lr"] = float(args.learning_rate)
    elif hasattr(args, "use_dadapt_optimizer") and args.use_dadapt_optimizer:
        logger.info("Using D-Adaptation optimizer.")
        try:
            from dadaptation import DAdaptAdam
        except ImportError:
            raise ImportError(
                "Please install the dadaptation library to make use of DaDapt optimizer."
                "You can do so by running `pip install dadaptation`"
            )

        optimizer_class = DAdaptAdam
        if (
            hasattr(args, "dadaptation_learning_rate")
            and args.dadaptation_learning_rate is not None
        ):
            logger.debug(
                f"Overriding learning rate {args.learning_rate} with {args.dadaptation_learning_rate} for D-Adaptation optimizer."
            )
            args.learning_rate = args.dadaptation_learning_rate
            extra_optimizer_args["decouple"] = True
            extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
            extra_optimizer_args["lr"] = float(args.learning_rate)

    elif hasattr(args, "use_adafactor_optimizer") and args.use_adafactor_optimizer:
        logger.info("Using Adafactor optimizer.")
        try:
            from transformers.optimization import Adafactor
        except ImportError:
            raise ImportError(
                "Please install the latest transformers library to make use of Adafactor optimizer."
                "You can do so by running `pip install transformers`, or, `poetry install` from the SimpleTuner directory."
            )

        optimizer_class = Adafactor
        extra_optimizer_args = {}
        if args.adafactor_relative_step:
            extra_optimizer_args["lr"] = None
            extra_optimizer_args["relative_step"] = True
            extra_optimizer_args["scale_parameter"] = False
            extra_optimizer_args["warmup_init"] = True
        else:
            extra_optimizer_args["lr"] = float(args.learning_rate)
            extra_optimizer_args["relative_step"] = False
            extra_optimizer_args["scale_parameter"] = False
            extra_optimizer_args["warmup_init"] = False
    else:
        logger.info("Using AdamW optimizer.")
        optimizer_class = torch.optim.AdamW
        extra_optimizer_args["betas"] = (args.adam_beta1, args.adam_beta2)
        extra_optimizer_args["lr"] = float(args.learning_rate)

    return extra_optimizer_args, optimizer_class


def determine_params_to_optimize(args, controlnet, unet, transformer, text_encoder_1, text_encoder_2, model_type_label, lycoris_wrapped_network):
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

        if args.lora_type == 'lycoris' and lycoris_wrapped_network is not None:
            params_to_optimize = list(
                filter(lambda p: p.requires_grad, lycoris_wrapped_network.parameters())
            )

    return params_to_optimize