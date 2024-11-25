import logging, sys, os
from os import environ
from diffusers.utils import is_wandb_available
from helpers.training.multi_process import _get_rank as get_rank
from helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)
if get_rank() == 0:
    logger.setLevel(environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel(logging.ERROR)
from helpers.training.error_handling import validate_deepspeed_compat_from_args


def safety_check(args, accelerator):
    if accelerator is not None and accelerator.num_processes > 1:
        # mulit-gpu safety checks & warnings
        if args.model_type == "lora" and args.lora_type == "standard":
            # multi-gpu PEFT checks & warnings
            if args.base_model_precision in ["fp8-quanto"]:
                logger.error(
                    f"{args.base_model_precision} is incompatible with multi-GPU training on PEFT LoRA."
                    " Use LORA_TYPE (--lora_type) lycoris for quantised multi-GPU training of LoKr models in FP8."
                )
                args.base_model_precision = "int8-quanto"

    if (
        args.base_model_precision in ["fp8-quanto", "int4-quanto"]
        or (args.base_model_precision != "no_change" and args.quantize_activations)
    ) and (
        accelerator is not None
        and accelerator.state.dynamo_plugin.backend.lower() == "inductor"
    ):
        logger.warning(
            f"{args.base_model_precision} is not supported with Dynamo backend. Disabling Dynamo."
        )
        from accelerate.utils import DynamoBackend

        accelerator.state.dynamo_plugin.backend = DynamoBackend.NO
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb
    if accelerator is not None and (
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
    if args.user_prompt_library and not os.path.exists(args.user_prompt_library):
        raise FileNotFoundError(
            f"User prompt library not found at {args.user_prompt_library}. Please check the path and try again."
        )

    # optimizer memory limit check for SOAP w/ 24G
    if (
        accelerator is not None
        and accelerator.device.type == "cuda"
        and accelerator.is_main_process
    ):
        import subprocess

        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ]
        ).split(b"\n")[get_rank()]
        total_memory = int(output.decode().strip()) / 1024
        from math import ceil

        total_memory_gb = ceil(total_memory)
        if total_memory_gb < 32 and total_memory_gb > 16 and args.optimizer == "soap":
            logger.warning(
                f"Your GPU has {total_memory_gb}GB of memory. The SOAP optimiser may require more than this. Setting --accelerator_cache_clear_interval=10 may help to eliminate OOM."
            )
        elif total_memory_gb < 24 and args.optimizer == "soap":
            logger.error(
                f"Your GPU has {total_memory_gb}GB of memory. The SOAP optimiser requires a GPU with at least 24G of memory."
            )
            sys.exit(1)

    if (
        args.model_type != "lora"
        and not args.controlnet
        and args.base_model_precision != "no_change"
        and not args.i_know_what_i_am_doing
    ):
        logger.error(
            f"{args.model_type} tuning is not compatible with quantisation. Please set --base_model_precision to 'no_change' or train LyCORIS/LoRA."
        )
        sys.exit(1)

    if (
        args.flux_schedule_shift is not None
        and args.flux_schedule_shift > 0
        and args.flux_schedule_auto_shift
    ):
        logger.error(
            f"--flux_schedule_auto_shift cannot be combined with --flux_schedule_shift. Please set --flux_schedule_shift to 0 if you want to train with --flux_schedule_auto_shift."
        )
        sys.exit(1)