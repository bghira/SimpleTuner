import logging, sys
from os import environ
from diffusers.utils import is_wandb_available
from helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
from helpers.training.error_handling import validate_deepspeed_compat_from_args


def safety_check(args, accelerator):
    if accelerator.num_processes > 1:
        # mulit-gpu safety checks & warnings
        if args.model_type == "lora" and args.lora_type == "standard":
            # multi-gpu PEFT checks & warnings
            if "quanto" in args.base_model_precision:
                logger.error(
                    "Quanto is incompatible with multi-GPU training on PEFT adapters. Use LORA_TYPE (--lora_type) lycoris for quantised multi-GPU training of LoKr models."
                )
                sys.exit(1)
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb
    if (
        hasattr(accelerator.state, "deepspeed_plugin")
        and accelerator.state.deepspeed_plugin is not None
    ):
        validate_deepspeed_compat_from_args(accelerator, args)
    if args.controlnet:
        if args.model_family in ["pixart_sigma", "sd3", "kolors", "flux", "smoldit"]:
            raise ValueError(
                f"ControlNet is not yet supported with {args.model_type} models. Please disable --controlnet, or switch model types."
            )
    if "lora" in args.model_type and "standard" == args.lora_type.lower():
        if args.model_family == "pixart_sigma":
            raise Exception(f"{args.model_type} does not support LoRA model training.")

    if "lora" in args.model_type and args.train_text_encoder:
        if args.lora_type.lower() == "lycoris":
            logger.error(
                "LyCORIS training is not meant to be combined with --train_text_encoder. It is powerful enough on its own!"
            )
            sys.exit(1)
