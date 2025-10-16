import contextlib
import logging
import os

import accelerate
import transformers
from accelerate.state import AcceleratorState
from transformers.integrations import HfDeepSpeedConfig

logger = logging.getLogger("DeepSpeed")
from simpletuner.helpers.training.deepspeed_optimizers import DEFAULT_OPTIMIZER as DS_DEFAULT_OPTIMIZER
from simpletuner.helpers.training.deepspeed_optimizers import normalize_optimizer_name
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
        zero_config = accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization") or {}
        offload_param = zero_config.get("offload_param")

        if isinstance(offload_param, dict):
            offload_param.setdefault("pin_memory", True)
            if offload_param.get("device") == "nvme":
                if str(offload_param.get("nvme_path", "none")).lower() == "none":
                    if args.offload_param_path is None:
                        raise ValueError(
                            f"DeepSpeed is using {offload_param['device']} but nvme_path is not specified. "
                            f"The configuration has '{offload_param['nvme_path']}' for 'nvme_path'."
                        )
                    offload_buffer = 131600000.0 if args.model_family in ["flux"] else 100000000.0
                    logger.info(f"Attempting to allocate {offload_buffer} size byte buffer.")
                    offload_param["buffer_size"] = offload_buffer
                    offload_param["nvme_path"] = args.offload_param_path

                logger.info(
                    "Using DeepSpeed NVMe offload at %s.",
                    offload_param.get("nvme_path", "<unspecified>"),
                )
        else:
            logger.debug("DeepSpeed config missing zero_optimization.offload_param; skipping NVMe pin configuration.")

        use_deepspeed_optimizer = True
        optimizer_config = accelerator.state.deepspeed_plugin.deepspeed_config.get("optimizer")

        user_supplied_optimizer = bool(optimizer_config)
        if optimizer_config:
            normalized_name, fallback_source = normalize_optimizer_name(
                optimizer_config.get("type") or optimizer_config.get("name")
            )
            if normalized_name:
                if fallback_source:
                    logger.warning(
                        "Unsupported DeepSpeed optimizer '%s'; replacing with '%s'.",
                        fallback_source,
                        normalized_name,
                    )
                optimizer_config["type"] = normalized_name
                optimizer_config.pop("name", None)
                accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"] = optimizer_config

        if not user_supplied_optimizer:
            optimizer_offload = zero_config.get("offload_optimizer", {})
            offload_device = str(optimizer_offload.get("device", "")).lower()
            optimizer_type = DS_DEFAULT_OPTIMIZER
            if offload_device == "cpu":
                logger.info("Using DeepSpeed optimizer (%s with CPU offload).", optimizer_type)
            else:
                logger.info("Using DeepSpeed optimizer (%s).", optimizer_type)
            accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"] = {
                "type": optimizer_type,
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
