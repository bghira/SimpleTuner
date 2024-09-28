import os
import sys
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level=os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
logger.setLevel(target_level)


def validate_deepspeed_compat_from_args(accelerator, args):
    if "lora" in args.model_type:
        logger.error(
            "LoRA can not be trained with DeepSpeed. Please disable DeepSpeed via 'accelerate config' before reattempting."
        )
        sys.exit(1)
    if (
        "gradient_accumulation_steps"
        in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        args.gradient_accumulation_steps = (
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "gradient_accumulation_steps"
            ]
        )
        logger.info(
            f"Updated gradient_accumulation_steps to the value provided by DeepSpeed: {args.gradient_accumulation_steps}"
        )
