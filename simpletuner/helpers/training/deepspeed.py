import contextlib
import logging
import os

import accelerate
import transformers
from accelerate.state import AcceleratorState
from transformers.integrations import HfDeepSpeedConfig

logger = logging.getLogger("DeepSpeed")
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")

from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)


@contextlib.contextmanager
def temporarily_disable_deepspeed_zero3():
    # https://github.com/huggingface/transformers/issues/28106
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        print("DeepSpeed was not enabled.")
        return []

    if deepspeed_plugin and is_deepspeed_zero3_enabled():
        print("DeepSpeed being disabled.")
        _hf_deepspeed_config_weak_ref = transformers.integrations.deepspeed._hf_deepspeed_config_weak_ref
        unset_hf_deepspeed_config()
        yield
        print("DeepSpeed being enabled.")
        set_hf_deepspeed_config(HfDeepSpeedConfig(deepspeed_plugin.deepspeed_config))
        transformers.integrations.deepspeed._hf_deepspeed_config_weak_ref = _hf_deepspeed_config_weak_ref
    else:
        print(f"Doing nothing, deepspeed zero3 was not enabled?")
        yield


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        logger.debug("DeepSpeed context manager disabled, no DeepSpeed detected.")
        return []

    logger.debug(f"DeepSpeed context manager enabled, DeepSpeed detected: {deepspeed_plugin}")
    return [
        deepspeed_plugin.zero3_init_context_manager(enable=False),
        temporarily_disable_deepspeed_zero3(),
    ]


def prepare_model_for_deepspeed(accelerator, args):
    use_deepspeed_optimizer = False
    use_deepspeed_scheduler = False
    if (
        hasattr(accelerator, "state")
        and hasattr(accelerator.state, "deepspeed_plugin")
        and getattr(accelerator.state, "deepspeed_plugin") is not None
    ):
        offload_param = accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_param"]
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_param"]["pin_memory"] = True
        if offload_param["device"] == "nvme":
            if offload_param["nvme_path"] == "none":
                if args.offload_param_path is None:
                    raise ValueError(
                        f"DeepSpeed is using {offload_param['device']} but nvme_path is not specified. The configuration has '{offload_param['nvme_path']}' for 'nvme_path'."
                    )
                else:
                    offload_buffer = 100000000.0
                    if args.model_family in ["flux"]:
                        # flux is big
                        offload_buffer = 131600000.0
                    logger.info(f"Attempting to allocate {offload_buffer} size byte buffer.")
                    accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_param"][
                        "buffer_size"
                    ] = offload_buffer
                    accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["offload_param"][
                        "nvme_path"
                    ] = args.offload_param_path
            logger.info(
                f"Using DeepSpeed NVMe offload at {accelerator.state.deepspeed_plugin.deepspeed_config['zero_optimization']['offload_param']['nvme_path']}."
            )

        use_deepspeed_optimizer = True
        if "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config:
            logger.info("Using DeepSpeed optimizer (AdamW).")
            accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"] = {
                "type": "AdamW",
                "params": {
                    "lr": args.learning_rate,
                    "betas": [args.adam_beta1, args.adam_beta2],
                    "eps": args.adam_epsilon,
                    "weight_decay": args.adam_weight_decay,
                },
            }

        use_deepspeed_scheduler = True
        if "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config:
            logger.info("Using DeepSpeed scheduler (WarmupLR).")
            accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"] = {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": args.learning_rate,
                    "warmup_num_steps": args.lr_warmup_steps,
                },
            }

    return use_deepspeed_optimizer, use_deepspeed_scheduler
