import ast
import copy
import glob
import hashlib
import json
import logging
import math
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import huggingface_hub
import wandb

from simpletuner.helpers import log_format  # noqa
from simpletuner.helpers.caching.memory import reclaim_memory
from simpletuner.helpers.configuration.cli_utils import mapping_to_cli_args
from simpletuner.helpers.configuration.loader import load_config
from simpletuner.helpers.data_backend.factory import BatchFetcher, configure_multi_databackend, random_dataloader_iterator
from simpletuner.helpers.models.all import model_families
from simpletuner.helpers.publishing.huggingface import HubManager
from simpletuner.helpers.training import trainable_parameter_count
from simpletuner.helpers.training.custom_schedule import get_lr_scheduler
from simpletuner.helpers.training.deepspeed import prepare_model_for_deepspeed
from simpletuner.helpers.training.deepspeed_optimizers import DEFAULT_OPTIMIZER as DS_DEFAULT_OPTIMIZER
from simpletuner.helpers.training.deepspeed_optimizers import sanitize_optimizer_block
from simpletuner.helpers.training.default_settings.safety_check import safety_check
from simpletuner.helpers.training.evaluation import ModelEvaluator
from simpletuner.helpers.training.min_snr_gamma import compute_snr
from simpletuner.helpers.training.multi_process import _get_rank as get_rank
from simpletuner.helpers.training.optimizer_param import (
    cpu_offload_optimizer,
    create_optimizer_with_param_groups,
    determine_optimizer_class_with_config,
    determine_params_to_optimize,
    is_lr_schedulefree,
    is_lr_scheduler_disabled,
)
from simpletuner.helpers.training.peft_init import init_lokr_network_with_perturbed_normal
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.validation import Validation, prepare_validation_prompt_list
from simpletuner.helpers.training.wrappers import unwrap_model
from simpletuner.helpers.utils.checkpoint_manager import CheckpointManager
from simpletuner.helpers.webhooks.events import (
    attach_timestamp,
    checkpoint_event,
    error_event,
    lifecycle_stage_event,
    notification_event,
    training_status_event,
)
from simpletuner.simpletuner_sdk.api_state import APIState


def _setup_logger(name: str, *, env_var: str | None = None, default_level: str = "INFO") -> logging.Logger:
    level_value = (
        os.environ.get(env_var, os.environ.get("SIMPLETUNER_LOG_LEVEL", default_level))
        if env_var
        else os.environ.get("SIMPLETUNER_LOG_LEVEL", default_level)
    )
    try:
        numeric_level = logging._nameToLevel.get(level_value.upper(), None)
        if numeric_level is None:
            numeric_level = int(level_value)
    except Exception:
        numeric_level = logging.INFO

    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(numeric_level)
    if not logger_instance.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger_instance.addHandler(handler)
    logger_instance.propagate = False
    return logger_instance


logger = _setup_logger("SimpleTuner")
filelock_logger = _setup_logger("filelock", default_level="WARNING")
connection_logger = _setup_logger("urllib3.connectionpool", default_level="WARNING")
training_logger = _setup_logger("training-loop", env_var="SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL")


import accelerate
import diffusers
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DeepSpeedPlugin, FullyShardedDataParallelPlugin
from accelerate.utils import (
    DistributedType,
    DynamoBackend,
    ParallelismConfig,
    TorchContextParallelConfig,
    TorchDynamoPlugin,
    set_seed,
)
from torch.distributions import Beta

from simpletuner.configure import model_classes

try:
    from lycoris import LycorisNetwork
except:
    print("[ERROR] Lycoris not available. Please install.")

try:
    from peft_singlora import update_singlora_global_step
except:
    pass
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    UniPCMultistepScheduler,
)
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm

from simpletuner.helpers.models.common import ImageModelFoundation, VideoModelFoundation
from simpletuner.helpers.training.ema import EMAModel

is_optimi_available = False
try:
    from optimi import prepare_for_gradient_release

    is_optimi_available = True
except:
    pass

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

SCHEDULER_NAME_MAP = {
    "euler": EulerDiscreteScheduler,
    "euler-a": EulerAncestralDiscreteScheduler,
    "unipc": UniPCMultistepScheduler,
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
}
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
if hasattr(transformers.utils, "logging"):
    transformers.utils.logging.set_verbosity_warning()
if hasattr(diffusers.utils, "logging"):
    diffusers.utils.logging.set_verbosity_warning()


class Trainer:
    def __init__(
        self,
        config: dict = None,
        disable_accelerator: bool = False,
        job_id: str = None,
        exit_on_error: bool = False,
    ):
        if job_id in (None, ""):
            job_id_env = os.environ.get("SIMPLETUNER_JOB_ID")
            if job_id_env:
                job_id = job_id_env

        self.accelerator = None
        self.model = None
        self.checkpoint_manager = None
        self.lycoris_wrapped_network = None
        self.lycoris_config = None
        self.lr_scheduler = None
        self.webhook_handler = None
        self.should_abort = False
        self._external_abort_checker = None
        self.ema_model = None
        self.job_id = job_id
        StateTracker.set_job_id(job_id)
        try:
            self.parse_arguments(
                args=config,
                disable_accelerator=disable_accelerator,
                exit_on_error=exit_on_error,
            )
        except Exception as e:
            self._send_webhook_msg(f"Error: {e}", message_level="critical")
            raise e

        if getattr(self, "config", None) is not None and self.config.model_family in model_families:
            self.model = model_families[self.config.model_family](self.config, self.accelerator)
            self.model.check_user_config()
            StateTracker.set_model(self.model)
        self._misc_init()
        self.validation = None
        # this updates self.config further, so we will run it here.
        self.init_noise_schedule()

    def _config_to_obj(self, config):
        if not config:
            return None
        return type("Config", (object,), config)

    def _update_grad_metrics(
        self, target_logs: Dict[str, float], *, require_value_method: bool = False, clone_norm_value: bool = False
    ):
        if self.grad_norm is None:
            return

        if self.config.grad_clip_method == "norm":
            grad_value = self.grad_norm
            if clone_norm_value:
                grad_value = float(self.grad_norm.clone().detach())
            target_logs["grad_norm"] = grad_value
        elif (
            not require_value_method or self.config.grad_clip_method == "value"
        ) and not self.config.use_deepspeed_optimizer:
            target_logs["grad_absmax"] = self.grad_norm

    def parse_arguments(self, args=None, disable_accelerator: bool = False, exit_on_error: bool = False):
        skip_config_fallback = False
        args_payload = args

        if isinstance(args, dict):
            args_payload = dict(args)
        elif hasattr(args, "__dict__"):
            args_payload = dict(vars(args))

        if isinstance(args_payload, dict):
            skip_config_fallback = bool(args_payload.pop("__skip_config_fallback__", False))
            # Strip any internal metadata entries that shouldn't be forwarded to the CLI parser.
            metadata_keys = [key for key in list(args_payload.keys()) if isinstance(key, str) and key.startswith("__")]
            for key in metadata_keys:
                args_payload.pop(key, None)

        self.configure_webhook(raw_config=args_payload if isinstance(args_payload, dict) else args)
        self.config = load_config(args_payload, exit_on_error=exit_on_error)
        if self.config is None and args_payload and not skip_config_fallback:
            # Fallback to the user's persisted defaults when ad-hoc CLI args are incomplete.
            # This mirrors historical behaviour where we would silently load the active
            # environment when parsing failed, while still surfacing an explicit failure if
            # no configuration can be resolved at all.
            self.config = load_config(None, exit_on_error=exit_on_error)

        if self.config is None:
            raise ValueError("Training configuration could not be parsed")

        accelerate_config_path = getattr(self.config, "accelerate_config", None)
        if accelerate_config_path not in (None, "", "None"):
            os.environ["ACCELERATE_CONFIG_PATH"] = os.path.expanduser(str(accelerate_config_path))

        accelerate_extra_args = getattr(self.config, "accelerate_extra_args", None)
        if accelerate_extra_args not in (None, ""):
            os.environ["ACCELERATE_EXTRA_ARGS"] = str(accelerate_extra_args)

        num_processes_value = getattr(self.config, "num_processes", None)
        if num_processes_value not in (None, ""):
            os.environ["TRAINING_NUM_PROCESSES"] = str(num_processes_value)

        num_machines_value = getattr(self.config, "num_machines", None)
        if num_machines_value not in (None, ""):
            os.environ["TRAINING_NUM_MACHINES"] = str(num_machines_value)

        def _resolve_dynamo_backend(candidate: object) -> DynamoBackend | None:
            if isinstance(candidate, DynamoBackend):
                return candidate
            if candidate is None:
                return None
            raw_was_string_like = isinstance(candidate, (str, bytes))
            if isinstance(candidate, bytes):
                try:
                    text = candidate.decode("utf-8", "strict").strip()
                except UnicodeDecodeError:
                    logger.debug(
                        "Unable to decode Torch Dynamo backend bytes value %r; defaulting to no dynamo.",
                        candidate,
                    )
                    return None
            else:
                text = str(candidate).strip()
            if not text:
                return None
            normalised = text.replace("-", "_").upper()
            if normalised in {"DISABLED", "NONE"}:
                normalised = "NO"
            try:
                return DynamoBackend[normalised]
            except KeyError as exc:
                if raw_was_string_like:
                    raise ValueError(f"Unsupported Torch Dynamo backend '{candidate}'.") from exc
                logger.debug(
                    "Ignoring non-string Torch Dynamo backend value %r (%s); defaulting to no dynamo.",
                    candidate,
                    type(candidate).__name__,
                )
                return None

        def _coerce_flag(value: object) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return False

        dynamo_backend_value = getattr(self.config, "dynamo_backend", None)
        try:
            resolved_dynamo_backend = _resolve_dynamo_backend(dynamo_backend_value)
        except ValueError:
            raise

        dynamo_backend_env = "no"
        if resolved_dynamo_backend and resolved_dynamo_backend != DynamoBackend.NO:
            dynamo_backend_env = resolved_dynamo_backend.value.lower()
        elif isinstance(dynamo_backend_value, str) and dynamo_backend_value.strip():
            dynamo_backend_env = dynamo_backend_value.strip().lower()
        os.environ["TRAINING_DYNAMO_BACKEND"] = dynamo_backend_env

        report_to_value = getattr(self.config, "report_to", None)
        if report_to_value is None or (isinstance(report_to_value, str) and not report_to_value.strip()):
            report_to_value = "none"
            setattr(self.config, "report_to", report_to_value)

        if isinstance(report_to_value, str):
            normalized_report = report_to_value.strip().lower()
            report_to = None if normalized_report == "none" else report_to_value.strip()
        else:
            report_to = report_to_value
        if not disable_accelerator:
            accelerator_custom_config = [self.config.process_group_kwargs]
            if self.config.mixed_precision == "fp8":
                # we'll set up a TorchAO config for Accelerator, since otherwise it uses MS-AMP which
                # is clunky and proprietary third party accelerator that is typically unavailable.
                from accelerate.utils import AORecipeKwargs

                accelerator_custom_config.append(AORecipeKwargs())

            should_override_bf16 = self._should_force_bf16_override()
            if should_override_bf16:
                logging.getLogger("SimpleTuner").info("Applying Accelerate bf16 capability override for this platform.")
                self._enable_bf16_override()

            accelerator_kwargs = dict(
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                mixed_precision=self.config.mixed_precision,
                log_with=report_to,
                project_config=self.config.accelerator_project_config,
                kwargs_handlers=accelerator_custom_config,
                dynamo_backend=dynamo_backend_env,
            )

            fsdp_plugin = None
            if getattr(self.config, "fsdp_enable", False):
                fsdp_plugin = self._load_fsdp_plugin()
            deepspeed_plugin = None
            if fsdp_plugin is None and hasattr(self.config, "deepspeed_config"):
                deepspeed_plugin = self._load_deepspeed_plugin()
            if fsdp_plugin is not None and deepspeed_plugin is not None:
                raise ValueError("Cannot enable both FSDP and DeepSpeed in the same run.")
            if fsdp_plugin is not None:
                fsdp_plugin = self._prepare_fsdp_plugin_for_accelerator(fsdp_plugin)
                accelerator_kwargs["fsdp_plugin"] = fsdp_plugin
            if deepspeed_plugin is not None:
                accelerator_kwargs["deepspeed_plugin"] = deepspeed_plugin

            dynamo_plugin = None
            if resolved_dynamo_backend and resolved_dynamo_backend != DynamoBackend.NO:
                plugin_kwargs: Dict[str, object] = {"backend": resolved_dynamo_backend}

                mode_value = getattr(self.config, "dynamo_mode", None)
                if isinstance(mode_value, str):
                    mode_text = mode_value.strip().lower()
                    if mode_text and mode_text not in {"auto", "default"}:
                        plugin_kwargs["mode"] = mode_text
                    elif mode_text == "default":
                        plugin_kwargs["mode"] = "default"
                elif mode_value:
                    plugin_kwargs["mode"] = str(mode_value)

                if _coerce_flag(getattr(self.config, "dynamo_fullgraph", None)):
                    plugin_kwargs["fullgraph"] = True
                if _coerce_flag(getattr(self.config, "dynamo_dynamic", None)):
                    plugin_kwargs["dynamic"] = True
                if _coerce_flag(getattr(self.config, "dynamo_use_regional_compilation", None)):
                    plugin_kwargs["use_regional_compilation"] = True

                dynamo_plugin = TorchDynamoPlugin(**plugin_kwargs)
                accelerator_kwargs["dynamo_plugin"] = dynamo_plugin

                flag_details = []
                if "mode" in plugin_kwargs:
                    flag_details.append(f"mode={plugin_kwargs['mode']}")
                if plugin_kwargs.get("fullgraph"):
                    flag_details.append("fullgraph")
                if plugin_kwargs.get("dynamic"):
                    flag_details.append("dynamic")
                if plugin_kwargs.get("use_regional_compilation"):
                    flag_details.append("regional")
                extras = f" ({', '.join(flag_details)})" if flag_details else ""
                logger.info("Torch Dynamo enabled (backend=%s)%s.", resolved_dynamo_backend.value.lower(), extras)

            context_parallel_size_raw = getattr(self.config, "context_parallel_size", None)
            context_parallel_strategy_raw = getattr(self.config, "context_parallel_comm_strategy", None)
            context_parallel_size = None
            if context_parallel_size_raw not in (None, "", "None"):
                try:
                    context_parallel_size = int(context_parallel_size_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Context parallel size must be an integer, got {context_parallel_size_raw!r}."
                    ) from exc
                setattr(self.config, "context_parallel_size", context_parallel_size)

            if context_parallel_size is not None and context_parallel_size <= 0:
                raise ValueError("Context parallel size must be greater than 0 when specified.")

            enable_context_parallel = context_parallel_size is not None and context_parallel_size > 1
            if enable_context_parallel:
                if not getattr(self.config, "fsdp_enable", False) or fsdp_plugin is None:
                    raise ValueError(
                        "Context parallelism requires FSDP2. Enable FSDP in Hardware > Accelerate before setting a context parallel size."
                    )
                fsdp_version_value = getattr(self.config, "fsdp_version", 2)
                try:
                    fsdp_version = int(fsdp_version_value)
                except (TypeError, ValueError):
                    fsdp_version = 2
                if fsdp_version != 2:
                    raise ValueError("Context parallelism currently only supports FSDP version 2.")

                context_parallel_strategy = (context_parallel_strategy_raw or "allgather").strip().lower()
                if context_parallel_strategy not in {"allgather", "alltoall"}:
                    raise ValueError(
                        f"Unsupported context parallel rotation '{context_parallel_strategy_raw}'. "
                        "Valid options are 'allgather' and 'alltoall'."
                    )
                setattr(self.config, "context_parallel_comm_strategy", context_parallel_strategy)

                accelerator_kwargs["parallelism_config"] = ParallelismConfig(
                    cp_size=context_parallel_size,
                    cp_handler=TorchContextParallelConfig(cp_comm_strategy=context_parallel_strategy),
                )
                logger.info(
                    "Context parallelism enabled (size=%s, rotation=%s).",
                    context_parallel_size,
                    context_parallel_strategy,
                )
            elif context_parallel_size is not None:
                # Normalise stored value when users explicitly set 1 to disable sharding
                strategy_fallback = context_parallel_strategy_raw or "allgather"
                strategy_text = strategy_fallback.strip().lower() if isinstance(strategy_fallback, str) else "allgather"
                setattr(self.config, "context_parallel_comm_strategy", strategy_text)

            try:
                self.accelerator = Accelerator(**accelerator_kwargs)
            except ValueError as err:
                if not should_override_bf16 and self._should_force_bf16_override(err):
                    logging.getLogger("SimpleTuner").warning(
                        "Retrying Accelerator initialisation with bf16 capability override."
                    )
                    self._enable_bf16_override()
                    self.accelerator = Accelerator(**accelerator_kwargs)
                else:
                    raise
            if self.accelerator:
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
            self._setup_accelerator_barrier_guard()
        fsdp_active = False
        if self.accelerator and hasattr(self.accelerator, "state"):
            fsdp_active = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        setattr(self.config, "use_fsdp", fsdp_active)
        safety_check(args=self.config, accelerator=self.accelerator)

        if self.config.lr_scale:
            lr_cur = self.config.learning_rate
            lr_scale_bsz = self.config.train_batch_size
            lr_scale_ga = self.config.gradient_accumulation_steps
            lr_scale_np = getattr(self.accelerator, "num_processes", 1)
            lr_scale_mul = lr_scale_ga * lr_scale_bsz * lr_scale_np
            lr_new = lr_cur * (math.sqrt(lr_scale_mul) if self.config.lr_scale_sqrt else lr_scale_mul)
            logger.info(
                f"Scaling learning rate from {lr_cur:.1e} to {lr_new:.1e}"
                f" due to {'--lr-scale and --lr-scale-sqrt' if self.config.lr_scale_sqrt else '--lr-scale'}"
                f" (bsz: {lr_scale_bsz}, ga: {lr_scale_ga}, nprocs: {lr_scale_np})"
            )
            self.config.learning_rate = lr_new

        StateTracker.set_accelerator(self.accelerator)
        StateTracker.set_args(self.config)
        StateTracker.set_weight_dtype(self.config.weight_dtype)
        self.set_model_family()

    def _should_force_bf16_override(self, error: Exception | None = None) -> bool:
        if self.config.mixed_precision != "bf16":
            return False

        if torch.backends.mps.is_available():
            return True

        if error is None:
            return False

        return "bf16 mixed precision requires" in str(error).lower()

    @staticmethod
    def _enable_bf16_override() -> None:
        def _always_true(*_args, **_kwargs):
            return True

        try:
            import accelerate.accelerator as accelerate_accelerator
            import accelerate.utils as accelerate_utils

            accelerate_utils.is_bf16_available = _always_true
            accelerate_accelerator.is_bf16_available = _always_true
        except ImportError:
            logger.warning("Failed to patch Accelerate bf16 guard; proceeding without override.")

    def _setup_accelerator_barrier_guard(self) -> None:
        if not self.accelerator:
            return

        if getattr(self.accelerator, "_simpletuner_safe_wait_installed", False):
            return

        original_wait_for_everyone = self.accelerator.wait_for_everyone

        def _safe_wait_for_everyone(*fn_args, **fn_kwargs):
            state = getattr(self.accelerator, "state", None)
            distributed_type = getattr(state, "distributed_type", None)
            if distributed_type in (
                DistributedType.MULTI_GPU,
                DistributedType.MULTI_MLU,
                DistributedType.MULTI_SDAA,
                DistributedType.MULTI_MUSA,
                DistributedType.MULTI_NPU,
                DistributedType.MULTI_XPU,
                DistributedType.MULTI_CPU,
                DistributedType.MULTI_HPU,
                DistributedType.DEEPSPEED,
                DistributedType.FSDP,
            ):
                if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
                    logger.debug("Skipping accelerator.wait_for_everyone(); distributed process group not initialised.")
                    return
            return original_wait_for_everyone(*fn_args, **fn_kwargs)

        self.accelerator.wait_for_everyone = _safe_wait_for_everyone
        setattr(self.accelerator, "_simpletuner_safe_wait_installed", True)

    def _load_fsdp_plugin(self):
        if not getattr(self.config, "fsdp_enable", False):
            return None

        deepspeed_config = getattr(self.config, "deepspeed_config", None)
        if deepspeed_config not in (None, "", "None", False):
            raise ValueError("FSDP and DeepSpeed cannot be enabled simultaneously.")

        fsdp_version = getattr(self.config, "fsdp_version", 2) or 2
        reshard_after_forward = getattr(self.config, "fsdp_reshard_after_forward", True)
        cpu_ram_efficient_loading = getattr(self.config, "fsdp_cpu_ram_efficient_loading", False)
        state_dict_type = getattr(self.config, "fsdp_state_dict_type", None)
        auto_wrap_policy = getattr(self.config, "fsdp_auto_wrap_policy", None)
        transformer_cls = getattr(self.config, "fsdp_transformer_layer_cls_to_wrap", None)

        plugin_kwargs = {
            "fsdp_version": fsdp_version,
            "reshard_after_forward": reshard_after_forward,
            "cpu_ram_efficient_loading": cpu_ram_efficient_loading,
        }

        resolved_state_dict_display = None
        if state_dict_type:
            state_dict_mapping = {
                "SHARDED_STATE_DICT": "sharded_state_dict",
                "FULL_STATE_DICT": "full_state_dict",
            }
            resolved_state_dict = state_dict_mapping.get(state_dict_type.upper(), state_dict_type)
            resolved_state_dict_display = resolved_state_dict
            plugin_kwargs["state_dict_type"] = resolved_state_dict

        if auto_wrap_policy:
            plugin_kwargs["auto_wrap_policy"] = auto_wrap_policy

        if transformer_cls:
            plugin_kwargs["transformer_cls_names_to_wrap"] = transformer_cls

        logger.info(
            "FSDP v%s configuration detected; enabling FullyShardedDataParallelPlugin%s.",
            fsdp_version,
            f" (reshard_after_forward={reshard_after_forward})" if fsdp_version >= 2 else "",
        )
        plugin = FullyShardedDataParallelPlugin(**plugin_kwargs)

        if resolved_state_dict_display:
            setattr(plugin, "_state_dict_type_enum", plugin.state_dict_type)
            setattr(plugin, "state_dict_type_display", resolved_state_dict_display)
            plugin.state_dict_type = resolved_state_dict_display

        if auto_wrap_policy and callable(getattr(plugin, "auto_wrap_policy", None)):
            setattr(plugin, "_auto_wrap_policy_callable", plugin.auto_wrap_policy)
            setattr(plugin, "auto_wrap_policy_display", auto_wrap_policy)
            plugin.auto_wrap_policy = auto_wrap_policy

        return plugin

    @staticmethod
    def _prepare_fsdp_plugin_for_accelerator(fsdp_plugin: FullyShardedDataParallelPlugin) -> FullyShardedDataParallelPlugin:
        if hasattr(fsdp_plugin, "_state_dict_type_enum"):
            fsdp_plugin.state_dict_type = getattr(fsdp_plugin, "_state_dict_type_enum")
        elif isinstance(getattr(fsdp_plugin, "state_dict_type", None), str):
            fsdp_plugin.set_state_dict_type(fsdp_plugin.state_dict_type)

        if hasattr(fsdp_plugin, "_auto_wrap_policy_callable"):
            fsdp_plugin.auto_wrap_policy = getattr(fsdp_plugin, "_auto_wrap_policy_callable")

        return fsdp_plugin

    def _finalize_deepspeed_config_auto_values(self, model) -> None:
        if not self.accelerator or model is None:
            return

        if getattr(self.accelerator.state, "distributed_type", None) != DistributedType.DEEPSPEED:
            return

        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if not deepspeed_plugin:
            return

        hidden_size_keys = (
            "zero_optimization.reduce_bucket_size",
            "zero_optimization.stage3_prefetch_bucket_size",
            "zero_optimization.stage3_param_persistence_threshold",
        )
        pending_auto_keys = [key for key in hidden_size_keys if deepspeed_plugin.is_auto(key)]
        if not pending_auto_keys:
            return

        inferred_hidden_size = self._infer_model_hidden_size(model)
        if not inferred_hidden_size or inferred_hidden_size <= 0:
            keys_text = ", ".join(pending_auto_keys)
            raise ValueError(
                "Unable to infer a representative hidden size from the training model's configuration. "
                "Please populate the DeepSpeed config with explicit numeric values for "
                f"{keys_text} or ensure the model config exposes a hidden-size attribute."
            )

        zero_config = deepspeed_plugin.deepspeed_config.setdefault("zero_optimization", {})
        if deepspeed_plugin.is_auto("zero_optimization.reduce_bucket_size"):
            zero_config["reduce_bucket_size"] = int(inferred_hidden_size * inferred_hidden_size)
        if deepspeed_plugin.is_auto("zero_optimization.stage3_prefetch_bucket_size"):
            zero_config["stage3_prefetch_bucket_size"] = int(0.9 * inferred_hidden_size * inferred_hidden_size)
        if deepspeed_plugin.is_auto("zero_optimization.stage3_param_persistence_threshold"):
            zero_config["stage3_param_persistence_threshold"] = int(10 * inferred_hidden_size)

    @staticmethod
    def _normalize_numeric_candidate(value):
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, (list, tuple, set)):
            numeric_values = [int(v) for v in value if isinstance(v, (int, float))]
            return max(numeric_values) if numeric_values else None
        if isinstance(value, dict):
            numeric_values = [int(v) for v in value.values() if isinstance(v, (int, float))]
            return max(numeric_values) if numeric_values else None
        return None

    def _infer_model_hidden_size(self, model) -> Optional[int]:
        if model is None:
            return None

        config = getattr(model, "config", None)
        if config is not None:
            candidate_attributes = (
                "hidden_size",
                "hidden_sizes",
                "encoder_hidden_sizes",
                "d_model",
                "model_dim",
                "embed_dim",
                "embedding_dim",
                "cross_attention_dim",
                "encoder_embed_dim",
                "inner_dim",
                "projection_dim",
                "block_out_channels",
                "channels",
            )
            for attr in candidate_attributes:
                if hasattr(config, attr):
                    candidate = self._normalize_numeric_candidate(getattr(config, attr))
                    if candidate and candidate > 0:
                        return candidate

            # Some configs expose dimensions via dict-style access
            if hasattr(config, "to_dict"):
                config_dict = config.to_dict()
            elif hasattr(config, "__dict__"):
                config_dict = config.__dict__
            else:
                config_dict = {}
            if isinstance(config_dict, dict):
                for attr in candidate_attributes:
                    if attr in config_dict:
                        candidate = self._normalize_numeric_candidate(config_dict[attr])
                        if candidate and candidate > 0:
                            return candidate

        direct_attributes = (
            "hidden_size",
            "d_model",
            "model_dim",
            "embed_dim",
            "inner_dim",
        )
        for attr in direct_attributes:
            if hasattr(model, attr):
                candidate = self._normalize_numeric_candidate(getattr(model, attr))
                if candidate and candidate > 0:
                    return candidate

        try:
            first_param = next(model.parameters())
            if first_param is not None and hasattr(first_param, "shape"):
                dims = [int(dim) for dim in first_param.shape if isinstance(dim, int) and dim > 1]
                if dims:
                    return max(dims)
        except (StopIteration, TypeError, AttributeError):
            pass

        return None

    def _load_deepspeed_plugin(self):
        raw_config = getattr(self.config, "deepspeed_config", None)
        if raw_config in (None, "", "None", False):
            return None

        try:
            normalized = self._normalize_deepspeed_config(raw_config)
        except ValueError as error:
            logger.error(str(error))
            raise

        if normalized is None:
            return None

        normalized, fallback_optimizer = sanitize_optimizer_block(normalized)
        if fallback_optimizer:
            logger.warning(
                "Unsupported DeepSpeed optimizer '%s'; replacing with '%s'.",
                fallback_optimizer,
                DS_DEFAULT_OPTIMIZER,
            )
        self._apply_deepspeed_overrides(normalized)
        self.config.deepspeed_config = normalized

        plugin_kwargs = {
            "hf_ds_config": normalized,
            "gradient_accumulation_steps": getattr(self.config, "gradient_accumulation_steps", 1),
        }

        zero_stage = self._extract_zero_stage(normalized)
        if zero_stage is not None:
            plugin_kwargs["zero_stage"] = zero_stage

        offload_param_path = getattr(self.config, "offload_param_path", None)
        if offload_param_path:
            expanded_offload_path = os.path.abspath(os.path.expanduser(str(offload_param_path)))
            self._cleanup_deepspeed_offload_artifacts(expanded_offload_path, zero_stage)
            zero_config = normalized.get("zero_optimization")
            if isinstance(zero_config, dict):
                if self._uses_nvme(zero_config.get("offload_param")):
                    plugin_kwargs.setdefault("offload_param_nvme_path", expanded_offload_path)
                if self._uses_nvme(zero_config.get("offload_optimizer")):
                    plugin_kwargs.setdefault("offload_optimizer_nvme_path", expanded_offload_path)

        logger.info(
            "DeepSpeed configuration detected; enabling DeepSpeedPlugin%s.",
            f" (ZeRO stage {zero_stage})" if zero_stage is not None else "",
        )
        return DeepSpeedPlugin(**plugin_kwargs)

    @staticmethod
    def _uses_nvme(section):
        if not isinstance(section, dict):
            return False
        device = section.get("device")
        if isinstance(device, str):
            return device.lower() == "nvme"
        return False

    def _apply_deepspeed_overrides(self, config_dict: dict) -> None:
        if not isinstance(config_dict, dict):
            return

        if "deepspeed_config_file" in config_dict:
            # User is delegating configuration to an external DeepSpeed JSON file.
            # Respect their settings and avoid mutating the inline dictionary.
            return

        train_batch_size = getattr(self.config, "train_batch_size", None)
        if train_batch_size is not None and "train_micro_batch_size_per_gpu" not in config_dict:
            config_dict["train_micro_batch_size_per_gpu"] = train_batch_size

        grad_accum = getattr(self.config, "gradient_accumulation_steps", None)
        if grad_accum is not None and "gradient_accumulation_steps" not in config_dict:
            config_dict["gradient_accumulation_steps"] = grad_accum

        offload_param_path = getattr(self.config, "offload_param_path", None)
        if not offload_param_path:
            return

        zero_config = config_dict.get("zero_optimization")
        if not isinstance(zero_config, dict):
            return

        expanded_path = os.path.expanduser(str(offload_param_path))
        for key in ("offload_param", "offload_optimizer"):
            section = zero_config.get(key)
            if isinstance(section, dict) and self._uses_nvme(section):
                section.setdefault("nvme_path", expanded_path)

    def _cleanup_deepspeed_offload_artifacts(self, offload_root: str, zero_stage: Optional[int]) -> None:
        """Remove stale DeepSpeed offload swap directories to avoid corrupted resume state."""

        if not offload_root:
            return
        try:
            resolved_root = os.path.abspath(offload_root)
        except Exception:
            logger.debug("Unable to resolve DeepSpeed offload path %s", offload_root, exc_info=True)
            return

        try:
            if not os.path.isdir(resolved_root):
                return

            candidates: List[str] = []
            if zero_stage is not None:
                candidates.append(os.path.join(resolved_root, f"zero_stage_{zero_stage}"))
            else:
                for entry in os.listdir(resolved_root):
                    if entry.startswith("zero_stage_"):
                        candidates.append(os.path.join(resolved_root, entry))

            for candidate in candidates:
                if not os.path.isdir(candidate):
                    continue
                try:
                    if os.path.commonpath([resolved_root, candidate]) != resolved_root:
                        logger.debug("Skipping DeepSpeed offload cleanup outside root: %s", candidate)
                        continue
                except Exception:
                    logger.debug("Failed to validate DeepSpeed offload path %s", candidate, exc_info=True)
                    continue
                try:
                    shutil.rmtree(candidate, ignore_errors=True)
                    logger.info("Cleared DeepSpeed offload artefacts at %s", candidate)
                except Exception:
                    logger.debug("Failed to clear DeepSpeed offload directory %s", candidate, exc_info=True)
        except Exception:
            logger.debug("Unexpected error while cleaning DeepSpeed offload path %s", resolved_root, exc_info=True)

    @staticmethod
    def _extract_zero_stage(config_dict: dict) -> Optional[int]:
        if not isinstance(config_dict, dict):
            return None

        candidates = []
        if "zero_stage" in config_dict:
            candidates.append(config_dict.get("zero_stage"))

        zero_config = config_dict.get("zero_optimization")
        if isinstance(zero_config, dict):
            candidates.append(zero_config.get("stage"))

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue

        return None

    @staticmethod
    def _normalize_deepspeed_config(raw_config):
        if raw_config in (None, "", "None", False):
            return None

        if isinstance(raw_config, (dict, list)):
            return raw_config

        if isinstance(raw_config, Path):
            raw_config = str(raw_config)

        if isinstance(raw_config, str):
            candidate = raw_config.strip()
            if not candidate:
                return None

            if candidate.startswith("{") or candidate.startswith("["):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as json_error:
                    try:
                        return ast.literal_eval(candidate)
                    except (ValueError, SyntaxError) as ast_error:
                        raise ValueError(f"Could not parse --deepspeed_config JSON payload: {ast_error}") from ast_error

            expanded_path = os.path.expanduser(candidate)
            if os.path.isfile(expanded_path):
                try:
                    with open(expanded_path, "r", encoding="utf-8") as handle:
                        return json.load(handle)
                except json.JSONDecodeError as file_error:
                    raise ValueError(f"DeepSpeed config at {expanded_path} is invalid JSON: {file_error}") from file_error

            raise ValueError(f"DeepSpeed config value '{raw_config}' is neither valid JSON nor an existing file.")

        raise ValueError(f"Unsupported type for --deepspeed_config: {type(raw_config)}")

    def run(self):
        try:
            self._exit_on_signal()
            # Initialize essential configurations and schedules
            self.init_noise_schedule()
            self._exit_on_signal()
            self.init_seed()
            self._exit_on_signal()
            self.init_huggingface_hub()
            self._exit_on_signal()

            # Core initialization steps with signal checks after each step
            self._initialize_components_with_signal_check(
                [
                    self.init_preprocessing_models,
                    self.init_data_backend,
                    self.init_validation_prompts,
                    self.init_unload_text_encoder,
                    self.init_unload_vae,
                    self.init_load_base_model,
                    self.init_precision,
                    self.init_controlnet_model,
                    self.init_tread_model,
                    self.init_freeze_models,
                    self.init_trainable_peft_adapter,
                    self.init_ema_model,
                ]
            )

            # Model movement and validation setup
            self.move_models(destination="accelerator")
            self._exit_on_signal()
            self.init_distillation()
            self._exit_on_signal()
            self.init_validations()
            self._exit_on_signal()
            self.enable_sageattention_inference()
            self.init_benchmark_base_model()
            self.disable_sageattention_inference()
            self._exit_on_signal()
            self.resume_and_prepare()
            self._exit_on_signal()
            self.init_trackers()

            # Start the training process
            self.train()

        except Exception as e:
            import traceback

            logger.error(f"Failed to run training: {e}, traceback: {traceback.format_exc()}")

            # Ensure webhook handler is configured before sending error message
            if not self.webhook_handler:
                logger.warning("Webhook handler not initialized, attempting to configure it now")
                try:
                    self.configure_webhook(send_startup_message=False)
                except Exception as webhook_err:
                    logger.error(f"Failed to configure webhook handler: {webhook_err}")

            self._send_webhook_msg(
                message=f"Failed to run training: {e}",
            )
            event = error_event(
                message=f"Failed to run training: {e}",
                title="Fatal Error",
                job_id=self.job_id,
                data={"status": "error"},
            )
            self._emit_event(event, message_level="critical")

            status_event = training_status_event(
                status="failed",
                message=f"Training failed: {e}",
                job_id=self.job_id,
                severity="error",
            )
            self._emit_event(status_event, message_level="critical")

            raise e

    def _initialize_components_with_signal_check(self, initializers):
        """
        Runs a list of initializer functions with signal checks after each.

        Args:
            initializers (list): A list of initializer functions to run sequentially.
        """
        for initializer in initializers:
            initializer()
            self._exit_on_signal()

    def _get_noise_schedule(self):
        self.config, scheduler = self.model.setup_training_noise_schedule()

        return scheduler

    def init_noise_schedule(self):
        if self.model is None:
            return
        from simpletuner.helpers.models.common import PredictionTypes

        self.config.flow_matching = True if self.model.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING else False
        self.noise_scheduler = self._get_noise_schedule()
        self.lr = 0.0

    def configure_webhook(self, send_startup_message: bool = True, raw_config: str = None):
        self.webhook_handler = None
        if raw_config is not None:
            # Handle both dict and argparse.Namespace
            if hasattr(raw_config, "get"):
                webhook_config = raw_config.get("webhook_config", raw_config.get("--webhook_config", None))
            else:
                # argparse.Namespace - use getattr
                webhook_config = getattr(raw_config, "webhook_config", getattr(raw_config, "__webhook_config", None))
            logging.info(f"Creating webhook: {webhook_config}")
        else:
            webhook_config = getattr(getattr(self, "config", None), "webhook_config", None)
        if webhook_config is None:
            return
        from simpletuner.helpers.webhooks.handler import WebhookHandler

        self.webhook_handler = WebhookHandler(
            self.accelerator,
            (
                f"{getattr(self.config, 'tracker_project_name', 'unknown')} {getattr(self.config, 'tracker_run_name', 'unknown')}"
                if hasattr(self, "config")
                else "unknown"
            ),
            send_video=(True if isinstance(self.model, VideoModelFoundation) else False),
            video_framerate=getattr(self.config, "framerate", None) if hasattr(self, "config") else None,
            webhook_config=webhook_config,
        )
        StateTracker.set_webhook_handler(self.webhook_handler)
        if send_startup_message:
            self._send_webhook_msg(
                message="SimpleTuner has launched. Hold onto your butts!",
                store_response=True,
            )
        event = notification_event(
            message="Training job has started, configuration has begun.",
            job_id=self.job_id,
        )
        self._emit_event(event)

    def _misc_init(self):
        """things that do not really need an order."""
        torch.set_num_threads(self.config.torch_num_threads)
        self.state = {}
        self.state["lr"] = 0.0
        # Initialize CheckpointManager with output directory
        self.checkpoint_manager = None
        if hasattr(self, "config") and getattr(self.config, "output_dir", None):
            self.checkpoint_manager = CheckpointManager(self.config.output_dir)
        # Global step represents the most recently *completed* optimization step, which means it
        #  takes into account the number of gradient_accumulation_steps. If we use 1 gradient_accumulation_step,
        #  then global_step and step will be the same throughout training. However, if we use
        #  2 gradient_accumulation_steps, then global_step will be twice as large as step, and so on.
        self.state["global_step"] = 0
        self.state["global_resume_step"] = 0
        self.state["first_epoch"] = 1
        self.state["args"] = self.config.__dict__
        self.timesteps_buffer = []
        self.guidance_values_list = []
        self.train_loss = 0.0
        self.bf = None
        self.grad_norm = None
        self.extra_lr_scheduler_kwargs = {}
        StateTracker.set_global_step(self.state["global_step"])
        self.config.use_deepspeed_optimizer, self.config.use_deepspeed_scheduler = prepare_model_for_deepspeed(
            self.accelerator, self.config
        )
        self.config.base_weight_dtype = self.config.weight_dtype
        self.config.is_quanto = False
        self.config.is_torchao = False
        self.config.is_bnb = False
        if "quanto" in self.config.base_model_precision:
            self.config.is_quanto = True
        elif "torchao" in self.config.base_model_precision:
            self.config.is_torchao = True
        elif "bnb" in self.config.base_model_precision:
            self.config.is_bnb = True
        # if text_encoder_1_precision -> text_encoder_4_precision has quanto we'll mark that as well
        for i in range(1, 5):
            if isinstance(getattr(self.config, f"text_encoder_{i}_precision", None), str) and getattr(
                self.config, f"text_encoder_{i}_precision", None
            ):
                if "quanto" in getattr(self.config, f"text_encoder_{i}_precision"):
                    if self.config.is_torchao:
                        raise ValueError(
                            "Cannot enable Quanto and TorchAO together. One quant engine must be used for all precision levels."
                        )
                    self.config.is_quanto = True
                elif "torchao" in getattr(self.config, f"text_encoder_{i}_precision"):
                    if self.config.is_quanto:
                        raise ValueError(
                            "Cannot enable Quanto and TorchAO together. One quant engine must be used for all precision levels."
                        )
                    self.config.is_torchao = True
                elif "bnb" in getattr(self.config, f"text_encoder_{i}_precision"):
                    self.config.is_bnb = True
        if self.config.is_quanto or self.config.is_torchao:
            from simpletuner.helpers.training.quantisation import quantise_model

            self.quantise_model = quantise_model

    def set_model_family(self, model_family: str = None):
        model_family = getattr(self.config, "model_family", model_family)
        if model_family not in model_classes["full"]:
            raise ValueError(f"Invalid model family specified: {model_family}")

        from simpletuner.helpers.models.all import model_families

        model_implementation = model_families.get(model_family)
        StateTracker.set_model_family(model_family)
        self.config.model_type_label = getattr(model_implementation, "NAME", None)
        if StateTracker.is_sdxl_refiner():
            self.config.model_type_label = "SDXL Refiner"

    def init_clear_backend_cache(self):
        if self.config.output_dir is not None:
            os.makedirs(self.config.output_dir, exist_ok=True)
        if self.config.preserve_data_backend_cache:
            return
        if not self.accelerator.is_local_main_process:
            return
        StateTracker.delete_cache_files(preserve_data_backend_cache=self.config.preserve_data_backend_cache)

    def init_seed(self):
        if self.config.seed is not None and self.config.seed != 0:
            set_seed(self.config.seed, self.config.seed_for_each_device)

    def init_huggingface_hub(self, access_token: str = None):
        # Handle the repository creation
        self.hub_manager = None
        if not self.accelerator.is_main_process:
            return
        if access_token is None:
            access_token = getattr(self, "hf_access_token", None)
        if access_token:
            huggingface_hub.login(token=access_token)
        self.hub_manager = HubManager(config=self.config, model=self.model)
        if self.config.push_to_hub:
            try:
                StateTracker.set_hf_user(huggingface_hub.whoami())
                logger.info(f"Logged into Hugging Face Hub as '{StateTracker.get_hf_username()}'")
            except Exception as e:
                logger.error(f"Failed to log into Hugging Face Hub: {e}")
                raise e

    def init_preprocessing_models(self, move_to_accelerator: bool = True):
        # image embeddings
        self.init_vae(move_to_accelerator=move_to_accelerator)
        # text embeds
        self.init_text_encoder(move_to_accelerator=move_to_accelerator)

    def init_vae(self, move_to_accelerator: bool = True):
        if getattr(self.model, "AUTOENCODER_CLASS", None) is None:
            logger.debug(f"Model {self.model.NAME} does not have a VAE.")
            return
        logger.info(f"Load VAE: {self.config.pretrained_vae_model_name_or_path}")
        self.model.load_vae(move_to_device=move_to_accelerator)
        StateTracker.set_vae_dtype(self.model.vae.dtype)
        StateTracker.set_vae(self.model.vae)

    def init_text_encoder(self, move_to_accelerator: bool = True):
        self.model.load_text_encoder(move_to_device=move_to_accelerator)

    def init_freeze_models(self):
        self.model.freeze_components()
        self.accelerator.wait_for_everyone()

    def init_load_base_model(self):
        webhook_msg = f"Loading model: `{self.config.pretrained_model_name_or_path}`..."
        # Send to Discord but don't emit event (lifecycle event below will handle that)
        self._send_webhook_msg(message=webhook_msg, emit_event=False)
        self._emit_event(
            lifecycle_stage_event(
                key="init_load_base_model",
                label="Loading base model",
                status="running",
                message=webhook_msg,
                job_id=self.job_id,
            )
        )
        self.model.load_model(move_to_device=False)
        self.accelerator.wait_for_everyone()
        self._emit_event(
            lifecycle_stage_event(
                key="init_load_base_model",
                label="Loading base model",
                status="completed",
                message="Base model has loaded.",
                current=1,
                total=1,
                percent=100,
                job_id=self.job_id,
            ),
        )

    def init_data_backend(self):
        try:
            self.init_clear_backend_cache()
            # Send to Discord but don't emit event (lifecycle event below will handle that)
            self._send_webhook_msg(message="Configuring data backends... (this may take a while!)", emit_event=False)
            self._emit_event(
                lifecycle_stage_event(
                    key="init_data_backend",
                    label="Configuring data backends",
                    status="running",
                    message="Configuring data backends... (this may take a while!)",
                    job_id=self.job_id,
                )
            )
            configure_multi_databackend(
                self.config,
                accelerator=self.accelerator,
                text_encoders=self.model.text_encoders,
                tokenizers=self.model.tokenizers,
                model=self.model,
            )
            self._emit_event(
                lifecycle_stage_event(
                    key="init_data_backend",
                    label="Configuring data backends",
                    status="completed",
                    message="Completed configuring data backends.",
                    current=1,
                    total=1,
                    percent=100,
                    job_id=self.job_id,
                )
            )
        except Exception as e:
            import traceback

            logger.error(f"{e}, traceback: {traceback.format_exc()}")

            # Ensure webhook handler is configured before sending error message
            if not self.webhook_handler:
                logger.warning("Webhook handler not initialized, attempting to configure it now")
                try:
                    self.configure_webhook(send_startup_message=False)
                except Exception as webhook_err:
                    logger.error(f"Failed to configure webhook handler: {webhook_err}")

            self._send_webhook_msg(
                message=f"Failed to load data backends: {e}",
                message_level="critical",
            )
            self._emit_event(
                lifecycle_stage_event(
                    key="init_data_backend",
                    label="Configuring data backends",
                    status="failed",
                    message=f"Failed to load data backends: {e}",
                    job_id=self.job_id,
                ),
                message_level="critical",
            )

            raise e

        try:
            self.init_validation_prompts()
        except Exception as e:
            logger.error("Could not generate validation prompts.")

            logger.exception("Could not generate validation prompts")
            raise e

        # We calculate the number of steps per epoch by dividing the number of images by the effective batch divisor.
        # Gradient accumulation steps mean that we only update the model weights every /n/ steps.
        collected_data_backend_keys = list(StateTracker.get_data_backends().keys())
        if self.hub_manager is not None and self.accelerator.is_main_process:
            self.hub_manager.collected_data_backend_str = collected_data_backend_keys
            validation_prompt_metadata = self.validation_prompt_metadata if self.validation_prompt_metadata else {}
            self.hub_manager.set_validation_prompts(validation_prompt_metadata)
            logger.debug(f"Collected validation prompts: {validation_prompt_metadata}")
        self._recalculate_training_steps()
        logger.info(f"Collected the following data backends: {collected_data_backend_keys}")
        # Send to Discord but don't emit event (notification event below will handle that)
        self._send_webhook_msg(
            message=f"Collected the following data backends: {collected_data_backend_keys}", emit_event=False
        )
        self._emit_event(
            notification_event(
                message=f"Collected the following data backends: {collected_data_backend_keys}",
                severity="info",
                job_id=self.job_id,
            )
        )
        self.accelerator.wait_for_everyone()

    def init_validation_prompts(self):
        if (
            hasattr(self.accelerator, "state")
            and hasattr(self.accelerator.state, "deepspeed_plugin")
            and getattr(self.accelerator.state.deepspeed_plugin, "deepspeed_config", {})
            .get("zero_optimization", {})
            .get("stage")
            == 3
        ):
            logger.warning("Cannot run validations with DeepSpeed ZeRO stage 3. Disabling validation.")
            self.config.validation_disable = True

        if self.accelerator.is_main_process and not self.config.validation_disable:
            self.validation_prompt_metadata = prepare_validation_prompt_list(
                args=self.config,
                embed_cache=StateTracker.get_default_text_embed_cache(),
                model=self.model,
            )
        else:
            self.validation_prompt_metadata = None
            self.validation_shortnames = None
            self.validation_negative_prompt_embeds = None
            self.validation_negative_pooled_embeds = None
        self.accelerator.wait_for_everyone()

    def stats_memory_used(self):
        # Grab GPU memory used:
        if torch.cuda.is_available():
            curent_memory_allocated = torch.cuda.memory_allocated() / 1024**3
        elif torch.backends.mps.is_available():
            curent_memory_allocated = torch.mps.current_allocated_memory() / 1024**3
        else:
            logger.warning("CUDA, ROCm, or Apple MPS not detected here. We cannot report VRAM reductions.")
            curent_memory_allocated = 0

        return curent_memory_allocated

    def init_unload_text_encoder(self):
        if self.config.model_type != "full" and self.config.train_text_encoder:
            return
        memory_before_unload = self.stats_memory_used()
        if self.accelerator.is_main_process:
            logger.info("Unloading text encoders, as they are not being trained.")
        self.model.unload_text_encoder()
        for backend_id, backend in StateTracker.get_data_backends().items():
            if "text_embed_cache" in backend:
                backend["text_embed_cache"].text_encoders = None
                backend["text_embed_cache"].pipeline = None
        reclaim_memory()
        memory_after_unload = self.stats_memory_used()
        memory_saved = memory_after_unload - memory_before_unload
        logger.info(f"After nuking text encoders from orbit, we freed {abs(round(memory_saved, 2))} GB of VRAM.")

    def init_precision(self, preprocessing_models_only: bool = False, ema_only: bool = False):
        self.config.enable_adamw_bf16 = True if self.config.weight_dtype == torch.bfloat16 else False
        quantization_device = "cpu" if self.config.quantize_via == "cpu" else self.accelerator.device

        if "bnb" in self.config.base_model_precision:
            # can't cast or move bitsandbytes models
            return

        if not self.config.disable_accelerator and self.config.is_quantized:
            if self.config.base_model_default_dtype == "fp32":
                self.config.base_weight_dtype = torch.float32
                self.config.enable_adamw_bf16 = False
            elif self.config.base_model_default_dtype == "bf16":
                self.config.base_weight_dtype = torch.bfloat16
                self.config.enable_adamw_bf16 = True
            elif self.config.base_model_default_dtype == "fp16":
                raise ValueError("fp16 mixed precision training is not supported.")
            if not preprocessing_models_only:
                logger.info(
                    f"Moving {self.model.MODEL_TYPE.value} to dtype={self.config.base_weight_dtype}, device={quantization_device}"
                )
                self.model.model.to(quantization_device, dtype=self.config.base_weight_dtype)
                if self.config.controlnet:
                    logger.info(f"Moving ControlNet to dtype={self.config.base_weight_dtype}, device={quantization_device}")
                    self.model.controlnet.to(quantization_device, dtype=self.config.base_weight_dtype)
        if self.config.is_quanto:
            with self.accelerator.local_main_process_first():
                if ema_only:
                    self.quantise_model(ema=self.ema_model, args=self.config)

                    return
                if self.config.controlnet:
                    # we'll do the base model first
                    self.quantise_model(
                        model=(self.model.unwrap_model(model=self.model.model) if not preprocessing_models_only else None),
                        text_encoders=None,
                        controlnet=None,
                        ema=self.ema_model,
                        args=self.config,
                    )

                self.quantise_model(
                    model=(self.model.get_trained_component() if not preprocessing_models_only else None),
                    text_encoders=self.model.text_encoders,
                    controlnet=None,
                    ema=None,
                    args=self.config,
                )
        elif self.config.is_torchao:
            with self.accelerator.local_main_process_first():
                if ema_only:
                    self.ema_model = self.quantise_model(ema=self.ema_model, args=self.config, return_dict=True)["ema"]

                    return
                (
                    q_model,
                    self.model.text_encoders,
                    self.controlnet,
                    self.ema_model,
                ) = self.quantise_model(
                    model=(self.model.get_trained_component(base_model=True) if not preprocessing_models_only else None),
                    text_encoders=self.model.text_encoders,
                    controlnet=None,
                    ema=self.ema_model,
                    args=self.config,
                )
                self.model.set_prepared_model(q_model, base_model=True)
                if self.config.controlnet:
                    (
                        q_model,
                        _,
                        _,
                        _,
                    ) = self.quantise_model(
                        model=(
                            self.model.get_trained_component(base_model=False) if not preprocessing_models_only else None
                        ),
                        args=self.config,
                    )
                    self.model.set_prepared_model(q_model, base_model=False)

    def init_controlnet_model(self):
        if not self.config.controlnet:
            return
        self.model.controlnet_init()
        self.accelerator.wait_for_everyone()

    def init_tread_model(self):
        if not self.config.tread_config:
            return
        self.model.tread_init()
        self.accelerator.wait_for_everyone()

    def init_trainable_peft_adapter(self):
        if "lora" not in self.config.model_type:
            return
        if "standard" == self.config.lora_type.lower():
            lora_info_msg = f"Using LoRA training mode (rank={self.config.lora_rank})"
            logger.info(lora_info_msg)
            self._send_webhook_msg(message=lora_info_msg)
            addkeys, misskeys = self.model.add_lora_adapter()
            if addkeys:
                logger.warning(
                    "The following keys were found in %s, but are not part of the model and are ignored:\n %s.\nThis is most likely an error"
                    % (self.config.init_lora, str(addkeys))
                )
            if misskeys:
                logger.warning(
                    "The following keys were part of the model but not found in %s:\n %s.\nThese keys will be initialized according to the lora weight initialisation. This could be an error, or intended behaviour in case a lora is finetuned with additional keys."
                    % (self.config.init_lora, str(misskeys))
                )

            logger.info(
                f"LoRA network has been initialized with {trainable_parameter_count(self._get_trainable_parameters())} parameters"
            )
        elif "lycoris" == self.config.lora_type.lower():
            from lycoris import create_lycoris

            with open(self.config.lycoris_config, "r") as f:
                self.lycoris_config = json.load(f)
            multiplier = int(self.lycoris_config.get("multiplier", 1))
            linear_dim = int(self.lycoris_config.get("linear_dim", 4))
            linear_alpha = int(self.lycoris_config.get("linear_alpha", 1))
            apply_preset = self.lycoris_config.get("apply_preset", None)
            if apply_preset is not None and apply_preset != {}:
                LycorisNetwork.apply_preset(apply_preset)

            # Remove the positional arguments we extracted.
            keys_to_remove = ["multiplier", "linear_dim", "linear_alpha"]
            for key in keys_to_remove:
                if key in self.lycoris_config:
                    del self.lycoris_config[key]

            logger.info("Using lycoris training mode")
            self._send_webhook_msg(message="Using lycoris training mode.")

            if self.config.init_lora is not None:
                from lycoris import create_lycoris_from_weights

                self.lycoris_wrapped_network = create_lycoris_from_weights(
                    multiplier,
                    self.config.init_lora,
                    self.model.get_trained_component(),
                    weights_sd=None,
                    **self.lycoris_config,
                )[0]
            else:
                self.lycoris_wrapped_network = create_lycoris(
                    self.model.get_trained_component(),
                    multiplier,
                    linear_dim,
                    linear_alpha,
                    **self.lycoris_config,
                )

                if self.config.init_lokr_norm is not None:
                    init_lokr_network_with_perturbed_normal(
                        self.lycoris_wrapped_network,
                        scale=self.config.init_lokr_norm,
                    )

            self.lycoris_wrapped_network.apply_to()
            setattr(
                self.accelerator,
                "_lycoris_wrapped_network",
                self.lycoris_wrapped_network,
            )
            logger.info(
                f"LyCORIS network has been initialized with {trainable_parameter_count(self.lycoris_wrapped_network.parameters())} parameters"
            )
        self.accelerator.wait_for_everyone()

    def init_post_load_freeze(self):
        if self.config.layer_freeze_strategy == "bitfit":
            from simpletuner.helpers.training.model_freeze import apply_bitfit_freezing

            if self.model.get_trained_component() is not None:
                logger.info(f"Applying BitFit freezing strategy to the {self.model.MODEL_TYPE.value}.")
                self.model.model = apply_bitfit_freezing(unwrap_model(self.accelerator, self.model.model), self.config)
        self.enable_gradient_checkpointing()

    def init_distillation(self):
        """Initialize distillation using the factory pattern."""
        from simpletuner.helpers.distillation.factory import DistillerFactory

        self.distiller = None

        if self.config.distillation_method is None:
            return

        # Get prediction type from model
        prediction_type = None
        if hasattr(self.model, "PREDICTION_TYPE"):
            prediction_type = self.model.PREDICTION_TYPE.value

        try:
            # Create distiller using factory
            self.distiller = DistillerFactory.create_distiller(
                method=self.config.distillation_method,
                teacher_model=self.model,
                noise_scheduler=self.noise_scheduler,
                config=vars(self.config),  # Convert config object to dict
                model_type=self.config.model_type,
                model_family=getattr(self.config, "model_family", None),
                prediction_type=prediction_type,
                student_model=None,  # Set this if using separate student model
            )

            if self.distiller:
                logger.info(f"Successfully initialized {self.config.distillation_method.upper()} distiller")
        except Exception as e:
            logger.error(f"Failed to initialize distillation: {e}")
            raise

    def enable_gradient_checkpointing(self):
        if self.config.gradient_checkpointing:
            logger.debug("Enabling gradient checkpointing.")
            if hasattr(self.model.get_trained_component(), "enable_gradient_checkpointing"):
                unwrap_model(self.accelerator, self.model.get_trained_component()).enable_gradient_checkpointing()
            if hasattr(self.config, "train_text_encoder") and self.config.train_text_encoder:
                for text_encoder in self.model.text_encoders:
                    if text_encoder is not None:
                        unwrap_model(self.accelerator, text_encoder).gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self):
        if self.config.gradient_checkpointing:
            logger.debug("Disabling gradient checkpointing.")
            if hasattr(self.model.get_trained_component(), "disable_gradient_checkpointing"):
                unwrap_model(
                    self.accelerator, self.model.get_trained_component(base_model=True)
                ).disable_gradient_checkpointing()
            if self.config.controlnet:
                unwrap_model(self.accelerator, self.model.get_trained_component()).disable_gradient_checkpointing()
            if hasattr(self.config, "train_text_encoder") and self.config.train_text_encoder:
                unwrap_model(self.accelerator, self.text_encoder_1).gradient_checkpointing_disable()
                unwrap_model(self.accelerator, self.text_encoder_2).gradient_checkpointing_disable()

    def _get_trainable_parameters(self):
        # Return just a list of the currently trainable parameters.
        if self.config.model_type == "lora":
            if self.config.lora_type == "lycoris":
                return self.lycoris_wrapped_network.parameters()
        return [param for param in self.model.get_trained_component(unwrap_model=False).parameters() if param.requires_grad]

    def _recalculate_training_steps(self):
        # Scheduler and math around the number of training steps.
        if not hasattr(self.config, "overrode_max_train_steps"):
            self.config.overrode_max_train_steps = False
        self.config.total_num_batches = sum(
            [
                len(backend["metadata_backend"] if "metadata_backend" in backend else [])
                for _, backend in StateTracker.get_data_backends().items()
            ]
        )
        self.config.num_update_steps_per_epoch = math.ceil(
            self.config.total_num_batches / max(self.config.gradient_accumulation_steps or 1, 1)
        )
        if getattr(self.config, "overrode_max_train_steps", False):
            self.config.max_train_steps = self.config.num_train_epochs * self.config.num_update_steps_per_epoch
            # Afterwards we recalculate our number of training epochs
            self.config.num_train_epochs = math.ceil(self.config.max_train_steps / self.config.num_update_steps_per_epoch)
            logger.info(
                "After removing any undesired samples and updating cache entries, we have settled on"
                f" {self.config.num_train_epochs} epochs and {self.config.num_update_steps_per_epoch} steps per epoch."
            )
        if self.config.max_train_steps is None or self.config.max_train_steps == 0:
            if self.config.num_train_epochs is None or self.config.num_train_epochs == 0:
                raise ValueError("You must specify either --max_train_steps or --num_train_epochs with a value > 0")
            self.config.max_train_steps = self.config.num_train_epochs * self.config.num_update_steps_per_epoch
            logger.info(
                f"Calculated our maximum training steps at {self.config.max_train_steps} because we have"
                f" {self.config.num_train_epochs} epochs and {self.config.num_update_steps_per_epoch} steps per epoch."
            )
            self.config.overrode_max_train_steps = True
        elif self.config.num_train_epochs is None or self.config.num_train_epochs == 0:
            if self.config.max_train_steps is None or self.config.max_train_steps == 0:
                raise ValueError("You must specify either --max_train_steps or --num_train_epochs with a value > 0")
            self.config.num_train_epochs = math.ceil(
                self.config.max_train_steps / max(self.config.num_update_steps_per_epoch, 1)
            )
            logger.info(
                f"Calculated our maximum training steps at {self.config.max_train_steps} because we have"
                f" {self.config.num_train_epochs} epochs and {self.config.num_update_steps_per_epoch} steps per epoch."
            )
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, "num_update_steps_per_epoch"):
            self.lr_scheduler.num_update_steps_per_epoch = self.config.num_update_steps_per_epoch
        self.config.total_batch_size = (
            self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps
        )

    def init_optimizer(self):
        logger.info(f"Learning rate: {self.config.learning_rate}")
        extra_optimizer_args = {"lr": self.config.learning_rate}
        # Initialize the optimizer
        optimizer_args_from_config, optimizer_class = determine_optimizer_class_with_config(
            args=self.config,
            use_deepspeed_optimizer=self.config.use_deepspeed_optimizer,
            is_quantized=self.config.is_quantized,
            enable_adamw_bf16=self.config.enable_adamw_bf16,
        )
        extra_optimizer_args.update(optimizer_args_from_config)

        self.params_to_optimize = determine_params_to_optimize(
            args=self.config,
            model=self.model,
            model_type_label=self.config.model_type_label,
            lycoris_wrapped_network=self.lycoris_wrapped_network,
        )
        logger.info(f"Connecting optimizer to {trainable_parameter_count(self.params_to_optimize)} trainable parameters")

        if self.config.use_deepspeed_optimizer:
            logger.info(
                f"DeepSpeed Optimizer arguments, weight_decay={self.config.adam_weight_decay} eps={self.config.adam_epsilon}, extra_arguments={extra_optimizer_args}"
            )
            self.optimizer = create_optimizer_with_param_groups(
                self.model.get_trained_component(unwrap_model=False),
                optimizer_class,
                optimizer_parameters={
                    **extra_optimizer_args,
                    "weight_decay": self.config.adam_weight_decay,
                    "eps": self.config.adam_epsilon,
                },  # Not sure which should have priority
                use_parameter_groups=True,  # Enable weight decay separation
                cpu_offload_config=(
                    {"offload_mechanism": self.config.optimizer_offload_mechanism}
                    if hasattr(self.config, "optimizer_offload_mechanism") and self.config.optimizer_offload_mechanism
                    else None
                ),
            )
        else:
            logger.info(f"Optimizer arguments={extra_optimizer_args}")
            if self.config.train_text_encoder and self.config.text_encoder_lr:
                # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
                # --learning_rate
                self.params_to_optimize[1]["lr"] = float(self.config.learning_rate)
                if self.text_encoder_2 is not None:
                    self.params_to_optimize[2]["lr"] = float(self.config.learning_rate)

            self.optimizer = cpu_offload_optimizer(
                params_to_optimize=self.params_to_optimize,
                optimizer_cls=optimizer_class,
                optimizer_parameters=extra_optimizer_args,
                fused=self.config.fuse_optimizer,
                offload_gradients=self.config.optimizer_offload_gradients,
                offload_mechanism=self.config.optimizer_cpu_offload_method,
            )

        if is_optimi_available and self.config.optimizer_release_gradients and "optimi" in self.config.optimizer:
            logger.warning(
                "Marking model for gradient release. This feature is experimental, and may use more VRAM or not work."
            )
            prepare_for_gradient_release(self.model.get_trained_component(), self.optimizer)

    def init_lr_scheduler(self):
        self.config.is_schedulefree = is_lr_schedulefree(self.config.optimizer)
        self.config.is_lr_scheduler_disabled = (
            is_lr_scheduler_disabled(self.config.optimizer) or self.config.use_deepspeed_scheduler
        )
        if self.config.is_schedulefree:
            logger.info("Using experimental ScheduleFree optimiser..")
        if self.config.is_lr_scheduler_disabled:
            # we don't use LR schedulers with schedulefree optimisers
            logger.info("Optimiser cannot use an LR scheduler, so we are disabling it.")
            lr_scheduler = None
            logger.info(f"Using dummy learning rate scheduler")
            if torch.backends.mps.is_available():
                lr_scheduler = None
            else:
                lr_scheduler = accelerate.utils.DummyScheduler(
                    self.optimizer,
                    total_num_steps=self.config.max_train_steps,
                    warmup_num_steps=self.config.lr_warmup_steps,
                )
            return lr_scheduler

        logger.info(
            f"Loading {self.config.lr_scheduler} learning rate scheduler with {self.config.lr_warmup_steps} warmup steps"
        )
        lr_scheduler = get_lr_scheduler(
            self.config,
            self.optimizer,
            self.accelerator,
            logger,
            global_step=self.state["global_step"],
            use_deepspeed_scheduler=False,
        )
        if hasattr(lr_scheduler, "num_update_steps_per_epoch"):
            lr_scheduler.num_update_steps_per_epoch = self.config.num_update_steps_per_epoch
        if hasattr(lr_scheduler, "last_step"):
            lr_scheduler.last_step = self.state.get("global_resume_step", 0)

        return lr_scheduler

    def init_ema_model(self):
        # Create EMA for the model.
        self.ema_model = None
        if not self.config.use_ema:
            return
        # this runs on all processes to ensure shapes are aligned.
        self.model.pre_ema_creation()
        if self.accelerator.is_main_process:
            logger.info("Using EMA. Creating EMAModel.")

            ema_model_cls = None
            ema_model_config = None
            if self.config.controlnet:
                ema_model_cls = self.controlnet.__class__
                ema_model_config = self.controlnet.config
            elif self.model.get_trained_component() is not None:
                ema_model_cls = self.model.get_trained_component().__class__
                ema_model_config = self.model.get_trained_component().config
            else:
                raise ValueError(
                    f"Please open a bug report or disable EMA. Unknown EMA model family: {self.config.model_family}"
                )

            self.ema_model = EMAModel(
                self.config,
                self.accelerator,
                parameters=self._get_trainable_parameters(),
                model_cls=ema_model_cls,
                model_config=ema_model_config,
                decay=self.config.ema_decay,
                foreach=not self.config.ema_foreach_disable,
            )
            logger.info(f"EMA model creation completed with {self.ema_model.parameter_count():,} parameters")

        self.accelerator.wait_for_everyone()
        # same about running on all processes to ensure alignment.
        self.model.post_ema_creation()

    def init_hooks(self):
        from simpletuner.helpers.training.save_hooks import SaveHookManager

        self.model_hooks = SaveHookManager(
            args=self.config,
            model=self.model,
            ema_model=self.ema_model,
            accelerator=self.accelerator,
            use_deepspeed_optimizer=self.config.use_deepspeed_optimizer,
        )
        self.accelerator.register_save_state_pre_hook(self.model_hooks.save_model_hook)
        self.accelerator.register_load_state_pre_hook(self.model_hooks.load_model_hook)

    def init_prepare_models(self, lr_scheduler):
        # Prepare everything with our `accelerator`.
        logger.info("Preparing models..")

        # TODO: Review if this is still required - added January 2024 as temporary solution.
        self.train_dataloaders = []
        for _, backend in StateTracker.get_data_backends().items():
            if "train_dataloader" not in backend:
                continue
            self.train_dataloaders.append(backend["train_dataloader"])
            break
        if len(self.train_dataloaders) == 0:
            logger.error("For some reason, no dataloaders were configured.")
            sys.exit(0)
        if self.config.disable_accelerator:
            logger.warning("Because SIMPLETUNER_DISABLE_ACCELERATOR is set, we will not prepare the accelerator.")
            return
        logger.info("Loading our accelerator...")
        if torch.backends.mps.is_available():
            self.accelerator.native_amp = False
        # Send to Discord but don't emit event (lifecycle event below will handle that)
        self._send_webhook_msg(message="Preparing model components...", emit_event=False)
        self._emit_event(
            lifecycle_stage_event(
                key="init_prepare_models",
                label="Preparing model components",
                status="running",
                message="Preparing model components",
                job_id=self.job_id,
            )
        )
        primary_model = self.model.get_trained_component(unwrap_model=False)
        self._finalize_deepspeed_config_auto_values(primary_model)
        results = self.accelerator.prepare(primary_model, lr_scheduler, self.optimizer, self.train_dataloaders[0])
        self.model.set_prepared_model(results[0])

        self.lr_scheduler = results[1]
        self.optimizer = results[2]
        # The rest of the entries are dataloaders:
        self.train_dataloaders = [results[3:]]
        if self.config.use_ema and self.ema_model is not None:
            if self.config.ema_device == "accelerator":
                logger.info("Moving EMA model weights to accelerator...")
            print(f"EMA model: {self.ema_model}")
            self.ema_model.to(
                (self.accelerator.device if self.config.ema_device == "accelerator" else "cpu"),
                dtype=self.config.weight_dtype,
            )

            if self.config.ema_device == "cpu" and not self.config.ema_cpu_only:
                logger.info("Pinning EMA model weights to CPU...")
                try:
                    self.ema_model.pin_memory()
                except Exception as e:
                    self._emit_event(
                        notification_event(
                            message=f"Failed to pin EMA to CPU: {e}",
                            severity="warning",
                            job_id=self.job_id,
                        )
                    )
                    logger.error(f"Failed to pin EMA model to CPU: {e}")

        idx_count = 0
        for _, backend in StateTracker.get_data_backends().items():
            if idx_count == 0 or "train_dataloader" not in backend:
                continue
            self.train_dataloaders.append(self.accelerator.prepare(backend["train_dataloader"]))
        idx_count = 0

        if "lora" in self.config.model_type and self.config.train_text_encoder:
            logger.info("Preparing text encoders for training.")
            if self.config.model_family == "sd3":
                logger.info("NOTE: The third text encoder is not trained for SD3.")
            self.text_encoder_1, self.text_encoder_2 = self.accelerator.prepare(self.text_encoder_1, self.text_encoder_2)
        self._recalculate_training_steps()
        self.accelerator.wait_for_everyone()
        self._emit_event(
            lifecycle_stage_event(
                key="init_prepare_models",
                label="Preparing model components",
                status="completed",
                message="Completed preparing model components",
                current=1,
                total=1,
                percent=100,
                job_id=self.job_id,
            )
        )

    def init_unload_vae(self):
        if self.config.keep_vae_loaded or self.config.vae_cache_ondemand:
            return
        memory_before_unload = self.stats_memory_used()
        self.model.unload_vae()
        for _, backend in StateTracker.get_data_backends().items():
            if "vaecache" in backend:
                backend["vaecache"].vae = None
        reclaim_memory()
        memory_after_unload = self.stats_memory_used()
        memory_saved = memory_after_unload - memory_before_unload
        logger.info(f"After nuking the VAE from orbit, we freed {abs(round(memory_saved, 2)) * 1024} MB of VRAM.")

    def init_validations(self):
        if (
            hasattr(self.accelerator, "state")
            and hasattr(self.accelerator.state, "deepspeed_plugin")
            and getattr(self.accelerator.state.deepspeed_plugin, "deepspeed_config", {})
            .get("zero_optimization", {})
            .get("stage")
            == 3
        ):
            logger.warning("Cannot run validations with DeepSpeed ZeRO stage 3. Disabling validation.")
            self.config.validation_disable = True
        if getattr(self.config, "use_fsdp", False) and getattr(self.config, "fsdp_reshard_after_forward", True):
            logger.warning("Cannot run validations with FSDP reshard-after-forward enabled. Disabling validation.")
            self.config.validation_disable = True
        self.evaluation = None
        if self.config.validation_disable:
            return
        if self.config.eval_steps_interval is not None and self.config.eval_steps_interval > 0:
            from simpletuner.helpers.training.validation import Evaluation

            self.evaluation = Evaluation(accelerator=self.accelerator)
        model_evaluator = ModelEvaluator.from_config(args=self.config)
        self.validation = Validation(
            trainable_parameters=self._get_trainable_parameters,
            accelerator=self.accelerator,
            model=self.model,
            distiller=self.distiller,
            args=self.config,
            validation_prompt_metadata=self.validation_prompt_metadata,
            vae_path=self.config.vae_path,
            weight_dtype=self.config.weight_dtype,
            embed_cache=StateTracker.get_default_text_embed_cache(),
            ema_model=self.ema_model,
            model_evaluator=model_evaluator,
            is_deepspeed=self.config.use_deepspeed_optimizer,
            is_fsdp=self.config.use_fsdp,
        )
        if not self.config.train_text_encoder and self.validation is not None:
            self.validation.clear_text_encoders()
        self.init_benchmark_base_model()
        self.accelerator.wait_for_everyone()

    def init_benchmark_base_model(self):
        if self.config.disable_benchmark or self.validation is None or self.validation.benchmark_exists("base_model"):
            # if we've disabled it or the benchmark exists, we will not do it again.
            # deepspeed zero3 can't do validations at all.
            return
        if not self.accelerator.is_main_process:
            return
        logger.info("Benchmarking base model for comparison. Supply `--disable_benchmark: true` to disable this behaviour.")
        self._emit_event(
            lifecycle_stage_event(
                key="benchmark_base_model",
                label="Benchmarking base model",
                status="running",
                message="Base model benchmark begins",
                job_id=self.job_id,
            )
        )
        # we'll run validation on base model if it hasn't already.
        self.validation.run_validations(validation_type="base_model", step=0)
        self.validation.save_benchmark("base_model")
        self._emit_event(
            lifecycle_stage_event(
                key="benchmark_base_model",
                label="Benchmarking base model",
                status="completed",
                message="Base model benchmark completed",
                current=1,
                total=1,
                percent=100,
                job_id=self.job_id,
            )
        )

    def init_resume_checkpoint(self, lr_scheduler):
        # Potentially load in the weights and states from a previous save
        self.config.total_steps_remaining_at_start = self.config.max_train_steps
        self.state["current_epoch"] = self.state["first_epoch"]
        self.state["global_resume_step"] = self.state["global_step"] = StateTracker.get_global_step()
        StateTracker.set_global_resume_step(self.state["global_resume_step"])
        if not self.config.resume_from_checkpoint:
            logger.info(f"Not resuming from checkpoint.")
            return lr_scheduler
        if self.config.resume_from_checkpoint != "latest":
            path = os.path.basename(self.config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            path = self.checkpoint_state_latest(self.config.output_dir)
            logger.info(f"Checking {path} for latest checkpoint.")

        if path is None:
            logger.info(f"Checkpoint '{self.config.resume_from_checkpoint}' does not exist. Starting a new training run.")
            event = lifecycle_stage_event(
                key="init_resume_checkpoint",
                label="Resume Checkpoint",
                status="completed",
                message="No model to resume. Beginning fresh training run.",
                job_id=self.job_id,
            )
            self._emit_event(event)

            self.config.resume_from_checkpoint = None
            return lr_scheduler

        checkpoint_dir = os.path.join(self.config.output_dir, path)
        logger.info(f"Resuming from checkpoint {checkpoint_dir}")
        self.accelerator.load_state(checkpoint_dir)
        if getattr(self, "distiller", None) is not None:
            logger.info(f"Loading DCM checkpoint states..")
            self.distiller.on_load_checkpoint(checkpoint_dir)
        try:
            if "constant" == self.config.lr_scheduler and not self.config.is_schedulefree:
                for g in self.optimizer.param_groups:
                    if "lr" in g:
                        g["lr"] = self.config.learning_rate
                for k, v in lr_scheduler.state_dict().items():
                    if k in ("base_lrs", "_last_lr"):
                        v[0] = self.config.learning_rate
        except Exception as e:
            event = notification_event(
                message="Could not update learning rate scheduler LR value.",
                severity="warning",
                job_id=self.job_id,
            )
            self._emit_event(event)
            logger.error(
                f"Could not update lr_scheduler {self.config.lr_scheduler} learning rate to {self.config.learning_rate} upon resume: {e}"
            )

        event = lifecycle_stage_event(
            key="init_resume_checkpoint",
            label="Resume Checkpoint",
            status="running",
            message=f"Resuming model: {path}",
            job_id=self.job_id,
        )
        self._emit_event(event)
        training_state_filename = f"training_state.json"
        if get_rank() > 0:
            training_state_filename = f"training_state-{get_rank()}.json"
        for _, backend in StateTracker.get_data_backends().items():
            if "sampler" in backend:
                backend["sampler"].load_states(
                    state_path=os.path.join(
                        self.config.output_dir,
                        path,
                        training_state_filename,
                    ),
                )
        self.state["global_resume_step"] = self.state["global_step"] = StateTracker.get_global_step()
        StateTracker.set_global_resume_step(self.state["global_resume_step"])
        training_state_in_ckpt = StateTracker.get_training_state()
        event = lifecycle_stage_event(
            key="init_resume_checkpoint",
            label="Resume Checkpoint",
            status="completed",
            message=f"Resumed from global_step {self.state['global_resume_step']}",
            job_id=self.job_id,
            extra=training_state_in_ckpt,
        )
        self._emit_event(event)
        logger.debug(f"Training state inside checkpoint: {training_state_in_ckpt}")
        if hasattr(lr_scheduler, "last_step"):
            lr_scheduler.last_step = self.state["global_resume_step"]
        logger.info(f"Resuming from global_step {self.state['global_resume_step']}.")

        # Log the current state of each data backend.
        for _, backend in StateTracker.get_data_backends().items():
            if "sampler" in backend:
                backend["sampler"].log_state()
        # We store the number of dataset resets that have occurred inside the checkpoint.
        self.state["first_epoch"] = StateTracker.get_epoch()
        if self.state["first_epoch"] > 1 or self.state["global_resume_step"] > 1:
            self.config.total_steps_remaining_at_start -= self.state["global_resume_step"]
            logger.debug(
                f"Resuming from epoch {self.state['first_epoch']}, which leaves us with {self.config.total_steps_remaining_at_start}."
            )
        self.state["current_epoch"] = self.state["first_epoch"]
        StateTracker.set_epoch(self.state["current_epoch"])
        if hasattr(lr_scheduler, "last_epoch"):
            lr_scheduler.last_epoch = (
                training_state_in_ckpt.get("epoch_step", self.state.get("global_resume_step", 1))
                * self.accelerator.num_processes
            )

        if self.state["current_epoch"] > self.config.num_train_epochs + 1 and not self.config.ignore_final_epochs:
            logger.info(
                f"Reached the end ({self.state['current_epoch']} epochs) of our training run ({self.config.num_train_epochs} epochs). This run will do zero steps."
            )
        self.accelerator.wait_for_everyone()

        if self.optimizer is not None and self.config.optimizer == "prodigy":
            # fix the device assignment for the prodigy optimizer parameters
            for group in (
                self.optimizer.param_groups if self.optimizer.optimizer.split_groups else self.optimizer.param_groups[:1]
            ):
                p = group["params"][0]
                group["running_d_numerator"] = group["running_d_numerator"].to(p.device)
                group["running_d_denom"] = group["running_d_denom"].to(p.device)
                if "use_focus" not in group:
                    group["use_focus"] = False

        # Emit completion event for resume checkpoint stage
        if self.config.resume_from_checkpoint:
            completion_event = lifecycle_stage_event(
                key="init_resume_checkpoint",
                label="Resuming checkpoint",
                status="completed",
                percent=100,
                current=1,
                total=1,
                job_id=self.job_id,
                severity="info",
            )
            self._emit_event(completion_event, message_level="info")

        return lr_scheduler

    def init_trackers(self):
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        self.guidance_values_table = None
        if self.accelerator.is_main_process:
            # Copy args into public_args:
            public_args = copy.deepcopy(self.config)
            delattr(public_args, "accelerator_project_config")
            delattr(public_args, "process_group_kwargs")
            delattr(public_args, "webhook_config")
            delattr(public_args, "weight_dtype")
            delattr(public_args, "base_weight_dtype")
            if hasattr(public_args, "vae_kwargs"):
                delattr(public_args, "vae_kwargs")
            if hasattr(public_args, "sana_complex_human_instruction"):
                delattr(public_args, "sana_complex_human_instruction")

            # Hash the contents of public_args to reflect a deterministic ID for a single set of params:
            public_args_hash = hashlib.md5(json.dumps(vars(public_args), sort_keys=True).encode("utf-8")).hexdigest()
            project_name = self.config.tracker_project_name or "simpletuner-training"
            tracker_run_name = self.config.tracker_run_name or "simpletuner-training-run"
            try:
                self.accelerator.init_trackers(
                    project_name,
                    config=vars(public_args),
                    init_kwargs={
                        "wandb": {
                            "name": tracker_run_name,
                            "id": f"{public_args_hash}",
                            "resume": "allow",
                            "allow_val_change": True,
                        }
                    },
                )
            except Exception as e:
                if "Object has no attribute 'disabled'" in repr(e):
                    logger.warning("WandB is disabled, and Accelerate was not quite happy about it.")
                    self.accelerator.trackers = []
                else:
                    logger.error(f"Could not initialize trackers: {e}")
                    event = error_event(
                        message=f"Could not initialize trackers. Continuing without. {e}",
                        job_id=self.job_id,
                    )
                    self._emit_event(event)
            event = notification_event(
                message="Training configuration initialized",
                title="Training Config",
                job_id=self.job_id,
                data=public_args.__dict__,
            )
            self._emit_event(event)

    def resume_and_prepare(self):
        self.init_optimizer()
        lr_scheduler = self.init_lr_scheduler()
        self.init_hooks()
        self.init_prepare_models(lr_scheduler=lr_scheduler)
        lr_scheduler = self.init_resume_checkpoint(lr_scheduler=lr_scheduler)
        self.init_post_load_freeze()

    def enable_sageattention_inference(self):
        # if the sageattention is inference-only, we'll enable it.
        # if it's training only, we'll disable it.
        # if it's inference+training, we leave it alone.
        if "sageattention" not in self.config.attention_mechanism or self.config.sageattention_usage == "training+inference":
            return
        if self.config.sageattention_usage == "inference":
            self.enable_sageattention()
        if self.config.sageattention_usage == "training":
            self.disable_sageattention()

    def disable_sageattention_inference(self):
        # if the sageattention is inference-only, we'll disable it.
        # if it's training only, we'll enable it.
        # if it's inference+training, we leave it alone.
        if "sageattention" not in self.config.attention_mechanism or self.config.sageattention_usage == "training+inference":
            return
        if self.config.sageattention_usage == "inference":
            self.disable_sageattention()
        if self.config.sageattention_usage == "training":
            self.enable_sageattention()

    def disable_sageattention(self):
        if "sageattention" not in self.config.attention_mechanism:
            return

        if (
            hasattr(torch.nn.functional, "scaled_dot_product_attention_sdpa")
            and torch.nn.functional != torch.nn.functional.scaled_dot_product_attention_sdpa
        ):
            logger.info("Disabling SageAttention.")
            setattr(
                torch.nn.functional,
                "scaled_dot_product_attention",
                torch.nn.functional.scaled_dot_product_attention_sdpa,
            )

    def enable_sageattention(self):
        if "sageattention" not in self.config.attention_mechanism:
            return

        # we'll try and load SageAttention and overload pytorch's sdpa function.
        try:
            logger.info("Enabling SageAttention.")
            from sageattention import (
                sageattn,
                sageattn_qk_int8_pv_fp8_cuda,
                sageattn_qk_int8_pv_fp16_cuda,
                sageattn_qk_int8_pv_fp16_triton,
            )

            sageattn_functions = {
                "sageattention": sageattn,
                "sageattention-int8-fp16-triton": sageattn_qk_int8_pv_fp16_triton,
                "sageattention-int8-fp16-cuda": sageattn_qk_int8_pv_fp16_cuda,
                "sageattention-int8-fp8-cuda": sageattn_qk_int8_pv_fp8_cuda,
            }
            # store the old SDPA for validations to use during VAE decode
            if not hasattr(torch.nn.functional, "scaled_dot_product_attention_sdpa"):
                setattr(
                    torch.nn.functional,
                    "scaled_dot_product_attention_sdpa",
                    torch.nn.functional.scaled_dot_product_attention,
                )

            def sageattn_wrapper_for_torch_sdpa_with_fallback(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
                enable_gqa=False,
            ) -> torch.Tensor:
                try:
                    return sageattn_functions[self.config.attention_mechanism](query, key, value, is_causal=is_causal)
                except:
                    logger.error(
                        f"Could not run SageAttention with {self.config.attention_mechanism}. Falling back to Pytorch SDPA."
                    )
                    return torch.nn.functional.scaled_dot_product_attention_sdpa(
                        query,
                        key,
                        value,
                        attn_mask,
                        dropout_p,
                        is_causal,
                        scale,
                        enable_gqa,
                    )

            torch.nn.functional.scaled_dot_product_attention = sageattn_wrapper_for_torch_sdpa_with_fallback
            if not hasattr(torch.nn.functional, "scaled_dot_product_attention_sage"):
                setattr(
                    torch.nn.functional,
                    "scaled_dot_product_attention_sage",
                    torch.nn.functional.scaled_dot_product_attention,
                )

            if "training" in self.config.sageattention_usage:
                logger.warning(
                    f"Using {self.config.attention_mechanism} for attention calculations during training. Your attention layers will not be trained. To disable SageAttention, remove or set --attention_mechanism to a different value."
                )
        except ImportError as e:
            logger.error(
                "Could not import SageAttention. Please install it to use this --attention_mechanism=sageattention."
            )
            logger.error(repr(e))
            sys.exit(1)

    def move_models(self, destination: str = "accelerator"):
        target_device = "cpu"
        if destination == "accelerator":
            target_device = self.accelerator.device
        logger.info(
            f"Moving the {str(self.model.get_trained_component().__class__)} to GPU in {self.config.weight_dtype if not self.config.is_quantized else self.config.base_model_precision} precision."
        )
        if self.model.get_trained_component() is not None:
            if self.config.is_quantized:
                self.model.get_trained_component(unwrap_model=False).to(target_device)
            else:
                self.model.get_trained_component(unwrap_model=False).to(target_device, dtype=self.config.weight_dtype)
        if getattr(self.accelerator, "_lycoris_wrapped_network", None) is not None:
            self.accelerator._lycoris_wrapped_network = self.accelerator._lycoris_wrapped_network.to(
                target_device, dtype=self.config.weight_dtype
            )

        if "sageattention" in self.config.attention_mechanism and "training" in self.config.sageattention_usage:
            logger.info("Using SageAttention for training. This is an unsupported, experimental configuration.")
            self.enable_sageattention()
        elif self.config.attention_mechanism == "xformers":
            if is_xformers_available():
                import xformers  # type: ignore # noqa

                if hasattr(
                    self.model.get_trained_component(),
                    "enable_xformers_memory_efficient_attention",
                ):
                    logger.info("Enabling xformers memory-efficient attention.")
                    self.model.get_trained_component().enable_xformers_memory_efficient_attention()
                else:
                    self.config.enable_xformers_memory_efficient_attention = False
                    self.config.attention_mechanism = "diffusers"
                    logger.warning(
                        "xformers is not enabled, as it is incompatible with this model type."
                        " Falling back to diffusers attention mechanism (Pytorch SDPA)."
                        " Alternatively, provide --attention_mechanism=sageattention for a more efficient option on CUDA systems."
                    )
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if self.config.controlnet:
            self.model.get_trained_component(unwrap_model=False).train()
            self.model.unwrap_model(self.model.model).to(device=target_device, dtype=self.config.weight_dtype)
            if self.config.train_text_encoder:
                logger.warning("Unknown results will occur when finetuning the text encoder alongside ControlNet.")

    def mark_optimizer_train(self):
        if is_lr_schedulefree(self.config.optimizer) and hasattr(self.optimizer, "train"):
            # we typically have to call train() on the optim for schedulefree.
            logger.debug("Setting optimiser into train() mode.")
            self.optimizer.train()

    def mark_optimizer_eval(self):
        if is_lr_schedulefree(self.config.optimizer) and hasattr(self.optimizer, "eval"):
            # we typically have to call eval() on the optim for schedulefree before saving or running validations.
            logger.debug("Setting optimiser into eval() mode.")
            self.optimizer.eval()

    def _send_webhook_msg(
        self, message: str, message_level: str = "info", store_response: bool = False, emit_event: bool = True
    ):
        if type(message) is not str:
            logger.error(f"_send_webhook_msg received {type(message)} type message instead of str.")
            return False
        if self.webhook_handler is None or not self.webhook_handler:
            return
        self.webhook_handler.send(message=message, message_level=message_level, store_response=store_response)
        # Only emit event if requested (to avoid duplicates when a structured event is also being sent)
        if emit_event:
            event = notification_event(message=message, severity=message_level, job_id=self.job_id)
            self._emit_event(event, message_level=message_level)

    def _emit_event(self, event: dict, message_level: str = "info"):
        if not isinstance(event, dict):
            logger.error(f"_emit_event expected dict payload, received {type(event)}.")
            return False
        if not self.webhook_handler:
            logger.warning("No webhook handler is configured.")
            return False
        if self.job_id and event.get("job_id") is None:
            event["job_id"] = self.job_id
        attach_timestamp(event)
        if "severity" not in event:
            event["severity"] = message_level
        self.webhook_handler.send_raw(event, message_level=message_level, job_id=self.job_id)
        return True

    def _train_initial_msg(self):
        initial_msg = "\n***** Running training *****"
        initial_msg += f"\n-  Trainable parameters: {trainable_parameter_count(self._get_trainable_parameters())}"
        initial_msg += f"\n-  Num batches = {self.config.total_num_batches}"
        initial_msg += f"\n-  Num Epochs = {self.config.num_train_epochs}"
        initial_msg += f"\n  - Current Epoch = {self.state['first_epoch']}"
        initial_msg += (
            f"\n-  Total train batch size (w. parallel, distributed & accumulation) = {self.config.total_batch_size}"
        )
        initial_msg += f"\n  - Instantaneous batch size per device = {self.config.train_batch_size}"
        initial_msg += f"\n  - Gradient Accumulation steps = {self.config.gradient_accumulation_steps}"
        initial_msg += f"\n-  Total optimization steps = {self.config.max_train_steps}"
        if self.state["global_step"] > 1:
            initial_msg += f"\n  - Steps completed: {self.state['global_step']}"
        steps_remaining_at_start = max(
            0, getattr(self.config, "total_steps_remaining_at_start", self.config.max_train_steps)
        )
        initial_msg += f"\n-  Total optimization steps remaining = {steps_remaining_at_start}"
        self.state["steps_remaining_at_start"] = steps_remaining_at_start
        self.state["total_num_steps"] = self.config.max_train_steps
        logger.info(initial_msg)
        # Send to Discord webhook but don't emit as event (to avoid cluttering event dock)
        if self.webhook_handler is not None:
            self.webhook_handler.send(message=initial_msg, message_level="info")
        # Cap global_step to max_train_steps in case of resume on already-completed environment
        capped_global_step = min(self.state["global_step"], self.config.max_train_steps)
        progress_percent = 0.0
        if self.config.max_train_steps:
            progress_percent = (capped_global_step / self.config.max_train_steps) * 100
        progress_payload = {
            "label": "Training initialisation",
            "current": capped_global_step,
            "total": self.config.max_train_steps,
            "percent": progress_percent,
            "metrics": {
                "epoch": self.state["first_epoch"],
                "total_epochs": self.config.num_train_epochs,
            },
        }
        status_event = training_status_event(
            status="running",  # Changed from "starting" - all initialization is complete
            message=initial_msg,
            job_id=self.job_id,
            severity="info",
            progress=progress_payload,
            extra={
                "total_num_batches": self.config.total_num_batches,
                "total_num_epochs": self.config.num_train_epochs,
                "total_num_steps": self.config.max_train_steps,
                "current_epoch": self.state["first_epoch"],
                "total_batch_size": self.config.total_batch_size,
                "micro_batch_size": self.config.train_batch_size,
                "global_step": self.state["global_step"],
                "remaining_num_steps": steps_remaining_at_start,
            },
        )
        self._emit_event(status_event)

    def _epoch_rollover(self, epoch):
        if self.state["first_epoch"] == epoch:
            return
        logger.debug(
            f"Just completed epoch {self.state['current_epoch']}. Beginning epoch {epoch}. Starting epoch was {self.state['first_epoch']}. Final epoch will be {self.config.num_train_epochs}"
        )
        for backend_id, backend in StateTracker.get_data_backends().items():
            backend_config = StateTracker.get_data_backend_config(backend_id)
            if (
                backend_config.get("crop")
                and backend_config.get("crop_aspect") == "random"
                and "metadata_backend" in backend
                and not self.config.aspect_bucket_disable_rebuild
            ) or (backend_config.get("vae_cache_clear_each_epoch") and "vaecache" in backend):
                # when the aspect ratio is random, we need to shuffle the dataset on each epoch.
                if self.accelerator.is_main_process:
                    # we only compute the aspect ratio indices on the main process.
                    # we have to set read_only to False since we're generating a new, un-split list.
                    # otherwise, we can't actually save the new cache to disk.
                    backend["metadata_backend"].read_only = False
                    # this will generate+save the new cache to the storage backend.
                    backend["metadata_backend"].compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
                self.accelerator.wait_for_everyone()
                logger.info(f"Reloading cache for backend {backend_id}")
                backend["metadata_backend"].reload_cache(set_config=False)
                logger.info("Waiting for other threads to finish..")
                self.accelerator.wait_for_everyone()
                # we'll have to split the buckets between GPUs again now, so that the VAE cache distributes properly.
                logger.info("Splitting buckets across GPUs")
                backend["metadata_backend"].split_buckets_between_processes(
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps
                )
                # we have to rebuild the VAE cache if it exists.
                if "vaecache" in backend:
                    logger.info("Rebuilding VAE cache..")
                    backend["vaecache"].rebuild_cache()
                # no need to manually call metadata_backend.save_cache() here.
        self.state["current_epoch"] = epoch
        StateTracker.set_epoch(epoch)
        if self.config.lr_scheduler == "cosine_with_restarts":
            self.extra_lr_scheduler_kwargs["epoch"] = epoch

    def _exit_on_signal(self):
        external_abort = False
        if callable(getattr(self, "_external_abort_checker", None)):
            try:
                external_abort = bool(self._external_abort_checker())
            except Exception:
                external_abort = False

        if external_abort and not self.should_abort:
            self.abort()

        if self.should_abort:
            event = lifecycle_stage_event(
                key="training_abort",
                label="Training Aborted",
                status="completed",
                message="Aborting training run.",
                job_id=self.job_id,
            )
            self._emit_event(event)
            raise StopIteration("Training run received abort signal.")

    def abort(self):
        logger.info("Aborting training run.")
        if self.bf is not None:
            self.bf.stop_fetching()
        # we should set should_abort = True on each data backend's vae cache, metadata, and text backend
        for _, backend in StateTracker.get_data_backends().items():
            if "vaecache" in backend:
                logger.debug(f"Aborting VAE cache")
                backend["vaecache"].should_abort = True
            if "metadata_backend" in backend:
                logger.debug(f"Aborting metadata backend")
                backend["metadata_backend"].should_abort = True
            if "text_backend" in backend:
                logger.debug(f"Aborting text backend")
                backend["text_backend"].should_abort = True
            if "sampler" in backend:
                logger.debug(f"Aborting sampler")
                backend["sampler"].should_abort = True
        self.should_abort = True

    def model_predict(
        self,
        prepared_batch,
        custom_timesteps: list = None,
    ):
        if custom_timesteps is not None:
            timesteps = custom_timesteps
        if not self.config.disable_accelerator:
            if self.config.controlnet:
                model_pred = self.model.controlnet_predict(
                    prepared_batch=prepared_batch,
                )
            else:
                model_pred = self.model.model_predict(
                    prepared_batch=prepared_batch,
                )

        else:
            # Dummy model prediction for debugging.
            model_pred = torch.randn_like(prepared_batch["noisy_latents"])

        # x-prediction requires that we now subtract the noise residual from the prediction to get the target sample.
        if (
            hasattr(self.noise_scheduler, "config")
            and hasattr(self.noise_scheduler.config, "prediction_type")
            and self.noise_scheduler.config.prediction_type == "sample"
        ):
            model_pred = model_pred - prepared_batch["noise"]

        return model_pred

    def _max_grad_value(self):
        max_grad_value = float("-inf")  # Start with a very small number
        for param in self._get_trainable_parameters():
            if param.grad is not None:
                max_grad_value = max(max_grad_value, param.grad.abs().max().item())

        return max_grad_value

    def prepare_batch(self, batch: dict):
        """
        Prepare a batch for the model prediction.

        Args:
            batch (dict): Batch from iterator_fn.

        Returns:
            batch (dict): Prepared batch.
        """
        if not batch:
            training_logger.debug(f"No batch was returned by the iterator_fn, returning {batch}")
            return batch

        prepared_batch = self.model.prepare_batch(batch, state=self.state)

        if getattr(self, "distiller", None) is not None:
            prepared_batch = self.distiller.prepare_batch(prepared_batch, self.model, self.state)

        return prepared_batch

    def get_prediction_target(self, prepared_batch: dict):
        return self.model.get_prediction_target(prepared_batch)

    def checkpoint_state_remove(self, output_dir, checkpoint):
        if self.checkpoint_manager:
            self.checkpoint_manager.remove_checkpoint(checkpoint)
        else:
            # Fallback to original implementation
            removing_checkpoint = os.path.join(output_dir, checkpoint)
            try:
                logger.debug(f"Removing {removing_checkpoint}")
                shutil.rmtree(removing_checkpoint, ignore_errors=True)
            except Exception as e:
                logger.error(f"Failed to remove directory: {removing_checkpoint}")
                print(e)

    def checkpoint_state_filter(self, output_dir, suffix=None):
        if self.checkpoint_manager:
            return self.checkpoint_manager._filter_checkpoints(suffix)
        else:
            # Fallback to original implementation
            checkpoints_keep = []
            checkpoints = os.listdir(output_dir)
            for checkpoint in checkpoints:
                cs = checkpoint.split("-")
                base = cs[0]
                sfx = None
                if len(cs) < 2:
                    continue
                elif len(cs) > 2:
                    sfx = cs[2]

                if base != "checkpoint":
                    continue
                if suffix and sfx and suffix != sfx:
                    continue
                if (suffix and not sfx) or (sfx and not suffix):
                    continue

                checkpoints_keep.append(checkpoint)

            return checkpoints_keep

    def checkpoint_state_cleanup(self, output_dir, limit, suffix=None):
        if self.checkpoint_manager:
            self.checkpoint_manager.cleanup_checkpoints(limit, suffix)
        else:
            # Fallback to original implementation
            # remove any left over temp checkpoints (partially written, etc)
            checkpoints = self.checkpoint_state_filter(output_dir, "tmp")
            for removing_checkpoint in checkpoints:
                self.checkpoint_state_remove(output_dir, removing_checkpoint)

            # now remove normal checkpoints past the limit
            checkpoints = self.checkpoint_state_filter(output_dir, suffix)
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            # before we save the new checkpoint, we need to have at _most_ `limit - 1` checkpoints
            if len(checkpoints) < limit:
                return

            num_to_remove = len(checkpoints) - limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.debug(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            logger.debug(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                self.checkpoint_state_remove(output_dir, removing_checkpoint)

    def checkpoint_state_save(self, output_dir, suffix=None):
        print("\n")

        save_path = os.path.join(
            output_dir,
            f"checkpoint-{self.state['global_step']}",
        )
        if suffix:
            save_path = f"{save_path}-{suffix}"

        # A temporary directory should be used so that saving state is an atomic operation.
        save_path_tmp = f"{save_path}-tmp" if self.config.checkpointing_use_tempdir else save_path

        # schedulefree optim needs the optimizer to be in eval mode to save the state (and then back to train after)
        event = lifecycle_stage_event(
            key="checkpoint_save",
            label="Saving Checkpoint",
            status="running",
            message=f"Saving checkpoint to {save_path}",
            job_id=self.job_id,
        )
        self._emit_event(event)
        self.mark_optimizer_eval()
        self.accelerator.save_state(save_path_tmp)
        event = lifecycle_stage_event(
            key="checkpoint_save",
            label="Saving Checkpoint",
            status="completed",
            message=f"Saved checkpoint to {save_path}",
            job_id=self.job_id,
        )
        self._emit_event(event)
        if getattr(self, "distiller", None) is not None:
            event = lifecycle_stage_event(
                key="checkpoint_save_distiller",
                label="Saving Distillation States",
                status="running",
                message=f"Saving distillation states to {save_path_tmp}",
                job_id=self.job_id,
            )
            self._emit_event(event)
            self.distiller.on_save_checkpoint(self.state["global_step"], save_path_tmp)
            event = lifecycle_stage_event(
                key="checkpoint_save_distiller",
                label="Saving Distillation States",
                status="completed",
                message=f"Saved distillation states to {save_path_tmp}",
                job_id=self.job_id,
            )
            self._emit_event(event)
        self.mark_optimizer_train()
        for _, backend in StateTracker.get_data_backends().items():
            if "sampler" in backend:
                logger.debug(f"Backend: {backend}")
                backend["sampler"].save_state(
                    state_path=os.path.join(
                        save_path_tmp,
                        self.model_hooks.training_state_path,
                    ),
                )
        if save_path != save_path_tmp:
            os.rename(save_path_tmp, save_path)
        event = checkpoint_event(
            path=save_path,
            label=f"Checkpoint saved to {save_path}",
            job_id=self.job_id,
        )
        self._emit_event(event)

    def checkpoint_state_latest(self, output_dir):
        if self.checkpoint_manager:
            return self.checkpoint_manager.get_latest_checkpoint()
        else:
            # Fallback to original implementation
            # both checkpoint-[0-9]+ and checkpoint-[0-9]-rolling are candidates
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint") and not d.endswith("tmp")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            return dirs[-1] if len(dirs) > 0 else None

    def train(self):
        self.init_trackers()
        self._train_initial_msg()
        self.mark_optimizer_train()

        # Only show the progress bar once on each machine.
        show_progress_bar = True
        if not self.accelerator.is_local_main_process:
            show_progress_bar = False
        progress_bar = tqdm(
            range(0, self.config.max_train_steps),
            disable=not show_progress_bar,
            initial=self.state["global_step"],
            desc=f"Epoch {self.state['first_epoch']}/{self.config.num_train_epochs} Steps",
            ncols=125,
        )
        self.accelerator.wait_for_everyone()

        # Some values that are required to be initialised later.
        step = self.state["global_step"]
        training_luminance_values = []
        current_epoch_step = None
        self.bf, fetch_thread = None, None
        iterator_fn = random_dataloader_iterator
        num_epochs_to_track = self.config.num_train_epochs + 1
        if self.config.ignore_final_epochs:
            num_epochs_to_track += 1000000
        for epoch in range(self.state["first_epoch"], num_epochs_to_track):
            if self.state["current_epoch"] > self.config.num_train_epochs + 1 and not self.config.ignore_final_epochs:
                # This might immediately end training, but that's useful for simply exporting the model.
                logger.info(
                    f"Training run is complete ({self.config.num_train_epochs}/{self.config.num_train_epochs} epochs, {self.state['global_step']}/{self.config.max_train_steps} steps)."
                )
                break
            self._epoch_rollover(epoch)
            self.model.get_trained_component(unwrap_model=False).train()
            training_models = [self.model.get_trained_component(unwrap_model=False)]
            if (
                "lora" in self.config.model_type
                and self.config.train_text_encoder
                and "standard" in self.config.lora_type.lower()
            ):
                for text_encoder in self.text_encoders:
                    if "t5" in str(text_encoder.__class__):
                        continue
                    text_encoder.train()
                    training_models.append(text_encoder)

            if current_epoch_step is not None:
                # We are resetting to the next epoch, if it is not none.
                current_epoch_step = 0
            else:
                # If it's None, we need to calculate the current epoch step based on the current global step.
                current_epoch_step = self.state["global_step"] % self.config.num_update_steps_per_epoch
            train_backends = {}
            for backend_id, backend in StateTracker.get_data_backends().items():
                if StateTracker.backend_status(backend_id) or "train_dataloader" not in backend:
                    # Exclude exhausted backends.
                    logger.debug(
                        f"Excluding backend: {backend_id}, as it is exhausted? {StateTracker.backend_status(backend_id)} or not found {('train_dataloader' not in backend)}"
                    )
                    continue
                if self.config.eval_dataset_id is not None:
                    # skip eval splits.
                    if isinstance(self.config.eval_dataset_id, str) and backend_id == self.config.eval_dataset_id:
                        continue
                    elif isinstance(self.config.eval_dataset_id, list) and backend_id in self.config.eval_dataset_id:
                        continue
                train_backends[backend_id] = backend["train_dataloader"]
            # Begin dataloader prefetch, if enabled.
            iterator_args = [train_backends]
            if self.config.dataloader_prefetch:
                iterator_args = []
                if self.bf is not None:
                    self.bf.stop_fetching()
                self.bf = BatchFetcher(
                    datasets=train_backends,
                    max_size=self.config.dataloader_prefetch_qlen,
                    step=step,
                )
                if fetch_thread is not None:
                    fetch_thread.join()
                fetch_thread = self.bf.start_fetching()
                iterator_fn = self.bf.next_response

            while True:
                self._exit_on_signal()
                step += 1
                prepared_batch = self.prepare_batch(iterator_fn(step, *iterator_args))
                training_logger.debug(f"Iterator: {iterator_fn}")
                if self.config.lr_scheduler == "cosine_with_restarts":
                    self.extra_lr_scheduler_kwargs["step"] = self.state["global_step"]

                if self.accelerator.is_main_process:
                    progress_bar.set_description(
                        f"Epoch {self.state['current_epoch']}/{self.config.num_train_epochs}, Steps"
                    )

                # If we receive a False from the enumerator, we know we reached the next epoch.
                if prepared_batch is False:
                    logger.debug(f"Reached the end of epoch {epoch}")
                    break

                if prepared_batch is None:
                    import traceback

                    raise ValueError(
                        f"Received a None batch, which is not a good thing. Traceback: {traceback.format_exc()}"
                    )

                # Add the current batch of training data's avg luminance to a list.
                if "batch_luminance" in prepared_batch:
                    training_luminance_values.append(prepared_batch["batch_luminance"])

                if getattr(self, "distiller", None) is not None:
                    self.distiller.pre_training_step(self.model, step)
                with self.accelerator.accumulate(training_models):
                    bsz = prepared_batch["latents"].shape[0]
                    training_logger.debug("Sending latent batch to GPU.")

                    if int(bsz) != int(self.config.train_batch_size):
                        logger.error(
                            f"Received {bsz} latents, but expected {self.config.train_batch_size}. Processing short batch."
                        )
                    training_logger.debug(f"Working on batch size: {bsz}")
                    # Prepare the data for the scatter plot
                    for timestep in prepared_batch["timesteps"].tolist():
                        self.timesteps_buffer.append((self.state["global_step"], timestep))

                    if "encoder_hidden_states" in prepared_batch:
                        encoder_hidden_states = prepared_batch["encoder_hidden_states"]
                        training_logger.debug(f"Encoder hidden states: {encoder_hidden_states.shape}")

                    if "add_text_embeds" in prepared_batch:
                        add_text_embeds = prepared_batch["add_text_embeds"]
                        training_logger.debug(
                            f"Pooled embeds: {add_text_embeds.shape if add_text_embeds is not None else None}"
                        )

                    # Predict the noise residual and compute loss
                    is_regularisation_data = prepared_batch.get("is_regularisation_data", False)
                    if is_regularisation_data and self.config.model_type == "lora":
                        training_logger.debug("Predicting parent model residual.")
                        with torch.no_grad():
                            if self.config.lora_type.lower() == "lycoris":
                                training_logger.debug("Detaching LyCORIS adapter for parent prediction.")
                                self.accelerator._lycoris_wrapped_network.set_multiplier(0.0)
                            else:
                                self.model.get_trained_component().disable_lora()
                            prepared_batch["target"] = self.model_predict(
                                prepared_batch=prepared_batch,
                            )["model_prediction"]
                            if self.config.lora_type.lower() == "lycoris":
                                training_logger.debug("Attaching LyCORIS adapter for student prediction.")
                                self.accelerator._lycoris_wrapped_network.set_multiplier(1.0)
                            else:
                                self.model.get_trained_component().enable_lora()

                    training_logger.debug("Predicting.")
                    model_pred = self.model_predict(
                        prepared_batch=prepared_batch,
                    )
                    loss = self.model.loss(
                        prepared_batch=prepared_batch,
                        model_output=model_pred,
                        apply_conditioning_mask=True,
                    )
                    loss, aux_loss_logs = self.model.auxiliary_loss(
                        prepared_batch=prepared_batch,
                        model_output=model_pred,
                        loss=loss,
                    )
                    distill_logs = {}
                    if self.config.distillation_method is not None:
                        loss, distill_logs = self.distiller.compute_distill_loss(prepared_batch, model_pred, loss)
                        loss, gen_logs = self.distiller.generator_loss_step(prepared_batch, model_pred, loss)
                        distill_logs.update(gen_logs)
                    parent_loss = None
                    if is_regularisation_data:
                        parent_loss = loss

                    # Gather the losses across all processes for logging (if using distributed training)
                    avg_loss = self.accelerator.gather(loss.repeat(self.config.train_batch_size)).mean()
                    self.train_loss += avg_loss.item() / self.config.gradient_accumulation_steps
                    # Backpropagate
                    self.grad_norm = None
                    if not self.config.disable_accelerator:
                        training_logger.debug("Backwards pass.")
                        self.accelerator.backward(loss)

                        if self.config.optimizer != "adam_bfloat16" and self.config.gradient_precision == "fp32":
                            # After backward, convert gradients to fp32 for stable accumulation
                            for param in self.params_to_optimize:
                                if param.grad is not None:
                                    param.grad.data = param.grad.data.to(torch.float32)

                        self.grad_norm = self._max_grad_value()
                        if (
                            self.accelerator.sync_gradients
                            and self.config.optimizer not in ["optimi-stableadamw", "prodigy"]
                            and self.config.max_grad_norm > 0
                        ):
                            # StableAdamW/Prodigy do not need clipping, similar to Adafactor.
                            if self.config.grad_clip_method == "norm":
                                self.grad_norm = self.accelerator.clip_grad_norm_(
                                    self._get_trainable_parameters(),
                                    self.config.max_grad_norm,
                                )
                            elif self.config.use_deepspeed_optimizer:
                                # deepspeed can only do norm clipping (internally)
                                pass
                            elif self.config.grad_clip_method == "value":
                                self.accelerator.clip_grad_value_(
                                    self._get_trainable_parameters(),
                                    self.config.max_grad_norm,
                                )
                            else:
                                raise ValueError(
                                    f"Unknown grad clip method: {self.config.grad_clip_method}. Supported methods: value, norm"
                                )
                        training_logger.debug("Stepping components forward.")
                        if self.config.optimizer_release_gradients:
                            step_offset = 0  # simpletuner indexes steps from 1.
                            should_not_release_gradients = (
                                step + step_offset
                            ) % self.config.gradient_accumulation_steps != 0
                            training_logger.debug(
                                f"step: {step}, should_not_release_gradients: {should_not_release_gradients}, self.config.optimizer_release_gradients: {self.config.optimizer_release_gradients}"
                            )
                            self.optimizer.optimizer_accumulation = should_not_release_gradients
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=self.config.set_grads_to_none)
                        if (
                            getattr(self, "distiller", None) is not None
                            and self.accelerator.sync_gradients  # run once per global step
                        ):
                            self.distiller.discriminator_step(prepared_batch=prepared_batch)
                            self.distiller.post_training_step(self.model, step)

                # Checks if the accelerator has performed an optimization step behind the scenes
                wandb_logs = {}
                if self.accelerator.sync_gradients:
                    try:
                        if self.config.peft_lora_mode == "singlora":
                            update_singlora_global_step(
                                model=self.model.get_trained_component(unwrap_model=True),
                                global_step=self.state["global_step"],
                            )
                        if "prodigy" in self.config.optimizer:
                            self.lr_scheduler.step(**self.extra_lr_scheduler_kwargs)
                            self.lr = self.optimizer.param_groups[0]["d"]
                        elif self.config.is_lr_scheduler_disabled:
                            # Alternative method for retrieving LR from accelerated optimizers
                            self.lr = StateTracker.get_last_lr()
                        else:
                            self.lr_scheduler.step(**self.extra_lr_scheduler_kwargs)
                            self.lr = self.lr_scheduler.get_last_lr()[0]
                    except Exception as e:
                        logger.error(f"Failed to get the last learning rate from the scheduler. Error: {e}")
                    wandb_logs.update(
                        {
                            "train_loss": self.train_loss,
                            "optimization_loss": loss,
                            "learning_rate": self.lr,
                            "epoch": epoch,
                        }
                    )
                    if distill_logs is not None:
                        wandb_logs.update(distill_logs)
                    if parent_loss is not None:
                        wandb_logs["regularisation_loss"] = parent_loss
                    if aux_loss_logs is not None:
                        for key, value in aux_loss_logs.items():
                            wandb_logs[f"aux_loss/{key}"] = value
                    self._update_grad_metrics(wandb_logs)
                    if self.validation is not None and hasattr(self.validation, "evaluation_result"):
                        eval_result = self.validation.get_eval_result()
                        if eval_result is not None and type(eval_result) == dict:
                            # add the dict to wandb_logs
                            self.validation.clear_eval_result()
                            wandb_logs.update(eval_result)

                    progress_bar.update(1)
                    self.state["global_step"] += 1
                    current_epoch_step += 1
                    StateTracker.set_global_step(self.state["global_step"])

                    ema_decay_value = "None (EMA not in use)"
                    if self.config.use_ema:
                        if self.ema_model is not None:
                            self.ema_model.step(
                                parameters=self._get_trainable_parameters(),
                                global_step=self.state["global_step"],
                            )
                            wandb_logs["ema_decay_value"] = self.ema_model.get_decay()
                            ema_decay_value = wandb_logs["ema_decay_value"]
                        self.accelerator.wait_for_everyone()

                    # Log scatter plot to wandb
                    if self.config.report_to == "wandb" and self.accelerator.is_main_process:
                        # Prepare the data for the scatter plot
                        data = [[iteration, timestep] for iteration, timestep in self.timesteps_buffer]
                        table = wandb.Table(data=data, columns=["global_step", "timestep"])
                        wandb_logs["timesteps_scatter"] = wandb.plot.scatter(
                            table,
                            "global_step",
                            "timestep",
                            title="Timestep distribution by step",
                        )

                    # Clear buffers
                    self.timesteps_buffer = []

                    # Average out the luminance values of each batch, so that we can store that in this step.
                    avg_training_data_luminance = sum(training_luminance_values) / len(training_luminance_values)
                    wandb_logs["train_luminance"] = avg_training_data_luminance

                    logger.debug(
                        f"Step {self.state['global_step']} of {self.config.max_train_steps}: loss {loss.item()}, lr {self.lr}, epoch {epoch}/{self.config.num_train_epochs}, ema_decay_value {ema_decay_value}, train_loss {self.train_loss}"
                    )
                    webhook_pending_msg = f"Step {self.state['global_step']} of {self.config.max_train_steps}: loss {round(loss.item(), 4)}, lr {self.lr}, epoch {epoch}/{self.config.num_train_epochs}, ema_decay_value {ema_decay_value}, train_loss {round(self.train_loss, 4)}"

                    if self.webhook_handler is not None:
                        current_state = self.state.copy()
                        current_state.pop("args")  # we don't need to send the config every time.
                        # Cap global_step to prevent exceeding total when resuming completed runs
                        capped_step = min(
                            current_state.get("global_step", 0),
                            current_state.get("total_num_steps", float("inf")),
                        )
                        event = training_status_event(
                            status="running",
                            job_id=self.job_id,
                            progress={
                                "current": capped_step,
                                "total": current_state.get("total_num_steps"),
                                "metrics": {
                                    "loss": self.train_loss,
                                    "parent_loss": parent_loss,
                                    "learning_rate": float(self.lr),
                                    "epoch": epoch,
                                },
                            },
                            extra={
                                "final_epoch": self.config.num_train_epochs,
                                **current_state,
                            },
                        )
                        self._emit_event(event)

                    if self.config.checkpointing_steps and self.state["global_step"] % self.config.checkpointing_steps == 0:
                        self._send_webhook_msg(
                            message=f"Checkpoint: `{webhook_pending_msg}`",
                            message_level="info",
                        )
                        # Also send structured progress update at checkpoint time
                        current_state = self.state.copy()
                        current_state.pop("args", None)
                        # Cap global_step to prevent exceeding total when resuming completed runs
                        capped_step = min(
                            current_state.get("global_step", 0),
                            current_state.get("total_num_steps", float("inf")),
                        )
                        event = training_status_event(
                            status="running",
                            job_id=self.job_id,
                            progress={
                                "current": capped_step,
                                "total": current_state.get("total_num_steps"),
                                "metrics": {
                                    "loss": self.train_loss,
                                    "parent_loss": parent_loss,
                                    "learning_rate": float(self.lr),
                                    "epoch": epoch,
                                },
                            },
                            extra={
                                "final_epoch": self.config.num_train_epochs,
                                **current_state,
                            },
                        )
                        self._emit_event(event)
                        if self.accelerator.is_main_process and self.config.checkpoints_total_limit is not None:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            self.checkpoint_state_cleanup(
                                self.config.output_dir,
                                self.config.checkpoints_total_limit,
                            )

                        if self.accelerator.is_main_process or self.config.use_deepspeed_optimizer:
                            self.checkpoint_state_save(self.config.output_dir)
                    elif (
                        self.config.checkpointing_rolling_steps
                        and self.state["global_step"] % self.config.checkpointing_rolling_steps == 0
                    ):
                        self._send_webhook_msg(
                            message=f"Checkpoint: `{webhook_pending_msg}`",
                            message_level="info",
                        )
                        # Also send structured progress update at checkpoint time
                        current_state = self.state.copy()
                        current_state.pop("args", None)
                        # Cap global_step to prevent exceeding total when resuming completed runs
                        capped_step = min(
                            current_state.get("global_step", 0),
                            current_state.get("total_num_steps", float("inf")),
                        )
                        event = training_status_event(
                            status="running",
                            job_id=self.job_id,
                            progress={
                                "current": capped_step,
                                "total": current_state.get("total_num_steps"),
                                "metrics": {
                                    "loss": self.train_loss,
                                    "parent_loss": parent_loss,
                                    "learning_rate": float(self.lr),
                                    "epoch": epoch,
                                },
                            },
                            extra={
                                "final_epoch": self.config.num_train_epochs,
                                **current_state,
                            },
                        )
                        self._emit_event(event)
                        if self.accelerator.is_main_process and self.config.checkpoints_rolling_total_limit is not None:
                            # _before_ saving state, check if this save would set us over the `checkpoints_rolling_total_limit`
                            self.checkpoint_state_cleanup(
                                self.config.output_dir,
                                self.config.checkpoints_rolling_total_limit,
                                "rolling",
                            )

                        if self.accelerator.is_main_process or self.config.use_deepspeed_optimizer:
                            self.checkpoint_state_save(self.config.output_dir, "rolling")

                    if (
                        self.config.accelerator_cache_clear_interval is not None
                        and self.config.accelerator.cache_clear_interval > 0
                        and self.state["global_step"] % self.config.accelerator_cache_clear_interval == 0
                    ):
                        reclaim_memory()

                    # here we might run eval loss calculations.
                    if self.evaluation is not None and self.evaluation.would_evaluate(self.state):
                        self.mark_optimizer_eval()
                        all_accumulated_losses = self.evaluation.execute_eval(
                            prepare_batch=self.prepare_batch,
                            model_predict=self.model_predict,
                            calculate_loss=self.model.loss,
                            get_prediction_target=self.get_prediction_target,
                            noise_scheduler=self._get_noise_schedule(),
                        )
                        if all_accumulated_losses:
                            tracker_table = self.evaluation.generate_tracker_table(
                                all_accumulated_losses=all_accumulated_losses
                            )
                            logger.debug(f"Tracking information: {tracker_table}")
                            wandb_logs.update(tracker_table)
                        self.mark_optimizer_train()

                    self.accelerator.log(
                        wandb_logs,
                        step=self.state["global_step"],
                    )

                    # Reset some values for the next go.
                    training_luminance_values = []
                    self.train_loss = 0.0

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": float(self.lr),
                }
                if aux_loss_logs is not None:
                    logs_to_print = {}
                    for key, value in aux_loss_logs.items():
                        logs_to_print[f"aux_loss/{key}"] = value
                    training_logger.debug(f"Aux loss: {logs_to_print}")
                self._update_grad_metrics(
                    logs,
                    require_value_method=True,
                    clone_norm_value=True,
                )

                progress_bar.set_postfix(**logs)

                if self.validation is not None:
                    if self.validation.would_validate():
                        self.mark_optimizer_eval()
                        self.enable_sageattention_inference()
                        self.disable_gradient_checkpointing()
                    self.validation.run_validations(validation_type="intermediary", step=step)
                    if self.validation.would_validate():
                        self.disable_sageattention_inference()
                        self.enable_gradient_checkpointing()
                        self.mark_optimizer_train()
                if (
                    self.hub_manager is not None
                    and step % self.config.gradient_accumulation_steps == 0
                    and self.state["global_step"] % self.config.checkpointing_steps == 0
                    and self.state["global_step"] > self.state["global_resume_step"]
                ):
                    if self.accelerator.is_main_process:
                        try:
                            self.hub_manager.upload_latest_checkpoint(
                                validation_images=(
                                    getattr(self.validation, "validation_images") if self.validation is not None else None
                                ),
                                webhook_handler=self.webhook_handler,
                            )
                        except Exception as e:
                            logger.error(f"Error uploading to hub: {e}, continuing training.")
                self.accelerator.wait_for_everyone()

                if self.state["global_step"] >= self.config.max_train_steps or (
                    epoch > self.config.num_train_epochs and not self.config.ignore_final_epochs
                ):
                    logger.info(
                        f"Training has completed."
                        f"\n -> global_step = {self.state['global_step']}, max_train_steps = {self.config.max_train_steps}, epoch = {epoch}, num_train_epochs = {self.config.num_train_epochs}",
                    )
                    # Note: training_complete event is emitted after final validation and model save
                    break
            if self.state["global_step"] >= self.config.max_train_steps or (
                epoch > self.config.num_train_epochs and not self.config.ignore_final_epochs
            ):
                logger.info(
                    f"Exiting training loop. Beginning model unwind at epoch {epoch}, step {self.state['global_step']}"
                )
                # Note: training_complete event is emitted after final validation and model save
                break

        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        validation_images = None
        if self.accelerator.is_main_process:
            event = lifecycle_stage_event(
                key="model_save",
                label="Saving Final Model",
                status="running",
                message=f"Finalizing model and saving to {self.config.output_dir}",
                job_id=self.job_id,
            )
            self._emit_event(event)
            self.mark_optimizer_eval()
            if self.validation is not None:
                self.enable_sageattention_inference()
                self.disable_gradient_checkpointing()
                # Emit validation start lifecycle event for final validations
                validation_start_event = lifecycle_stage_event(
                    key="final_validation",
                    label="Running Final Validations",
                    status="running",
                    message="Generating final validation images...",
                    job_id=self.job_id,
                )
                self._emit_event(validation_start_event)
                validation_images = self.validation.run_validations(
                    validation_type="final",
                    step=self.state["global_step"],
                    force_evaluation=True,
                    skip_execution=True,
                ).validation_images
                # Emit validation completed lifecycle event
                validation_completed_event = lifecycle_stage_event(
                    key="final_validation",
                    label="Running Final Validations",
                    status="completed",
                    message="Final validation images completed",
                    job_id=self.job_id,
                )
                self._emit_event(validation_completed_event)
                # we don't have to do this but we will anyway.
                self.disable_sageattention_inference()
            if self.model.get_trained_component() is not None:
                self.model.model = unwrap_model(self.accelerator, self.model.model)
            if "lora" in self.config.model_type and "standard" == self.config.lora_type.lower():
                lora_save_kwargs = {
                    "save_directory": self.config.output_dir,
                    f"{self.model.MODEL_TYPE.value}_lora_layers": get_peft_model_state_dict(
                        self.model.get_trained_component()
                    ),
                }

                if self.config.train_text_encoder:
                    if self.model.get_text_encoder(0) is not None:
                        self.text_encoder_1 = self.accelerator.unwrap_model(self.model.get_text_encoder(0))
                        lora_save_kwargs["text_encoder_lora_layers"] = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(self.text_encoder_1)
                        )
                    if self.model.get_text_encoder(1) is not None:
                        self.text_encoder_2 = self.accelerator.unwrap_model(self.model.get_text_encoder(1))
                        lora_save_kwargs["text_encoder_2_lora_layers"] = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(self.text_encoder_2)
                        )
                        if self.model.get_text_encoder(2) is not None:
                            self.text_encoder_3 = self.accelerator.unwrap_model(self.model.get_text_encoder(2))
                else:
                    text_encoder_lora_layers = None
                    text_encoder_2_lora_layers = None

                from simpletuner.helpers.models.common import PipelineTypes

                self.model.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG].save_lora_weights(
                    **lora_save_kwargs,
                )
                del text_encoder_lora_layers
                del text_encoder_2_lora_layers
                reclaim_memory()
            elif "lora" in self.config.model_type and "lycoris" == self.config.lora_type.lower():
                if self.accelerator.is_main_process or self.config.use_deepspeed_optimizer:
                    logger.info(f"Saving final LyCORIS checkpoint to {self.config.output_dir}")
                    # Save final LyCORIS checkpoint.
                    if getattr(self.accelerator, "_lycoris_wrapped_network", None) is not None:
                        from simpletuner.helpers.publishing.huggingface import LORA_SAFETENSORS_FILENAME

                        self.accelerator._lycoris_wrapped_network.save_weights(
                            os.path.join(self.config.output_dir, LORA_SAFETENSORS_FILENAME),
                            list(self.accelerator._lycoris_wrapped_network.parameters())[0].dtype,
                            {"lycoris_config": json.dumps(self.lycoris_config)},  # metadata
                        )
                        shutil.copy2(
                            self.config.lycoris_config,
                            os.path.join(self.config.output_dir, "lycoris_config.json"),
                        )

            elif self.config.use_ema:
                if self.model.get_trained_component() is not None:
                    self.ema_model.copy_to(self.model.get_trained_component().parameters())

            if self.config.model_type == "full":
                if self.config.save_text_encoder:
                    self.model.load_text_encoder()
                self.model.load_vae()
                pipeline = self.model.get_pipeline()
                pipeline.save_pretrained(
                    os.path.join(self.config.output_dir, "pipeline"),
                    safe_serialization=True,
                )
                logger.info(f"Wrote pipeline to disk: {self.config.output_dir}/pipeline")

            if self.hub_manager is not None and self.accelerator.is_main_process:
                self.hub_manager.upload_model(validation_images, self.webhook_handler)
        self.accelerator.end_training()
        # Emit training_complete event after all model saving and validation is complete
        event = lifecycle_stage_event(
            key="training_complete",
            label="Training Complete",
            status="completed",
            message="Training run complete.",
            job_id=self.job_id,
            percent=100,
            metrics={
                "loss": round(self.train_loss, 4),
                "learning_rate": self.lr,
                "epoch": self.state.get("current_epoch", 0),
            },
            extra={
                "state": self.state,
                "final_epoch": self.config.num_train_epochs,
            },
        )
        self._emit_event(event)


def run_trainer_job(config):
    """Create a Trainer from the provided config and execute the full run loop."""

    job_id = None
    should_abort_callable = None

    if hasattr(config, "__dict__"):
        attrs = vars(config).copy()
        should_abort_callable = attrs.pop("should_abort", None)
        job_id = attrs.pop("__job_id__", None) or attrs.pop("job_id", None)
        config_payload = attrs
    elif isinstance(config, dict):
        config_payload = dict(config)
        job_id = config_payload.pop("__job_id__", None) or config_payload.pop("job_id", None)
        should_abort_callable = config_payload.pop("should_abort", None)
    else:
        config_payload = config

    def _extract_value(mapping: Dict[str, object], *keys: str, default=None):
        for key in keys:
            if key in mapping:
                value = mapping[key]
                if isinstance(value, str) and value.strip() == "":
                    continue
                if value not in (None, "==SUPPRESS=="):
                    return value
        return default

    def _safe_int(value: object, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    accelerate_config_path = None
    accelerate_extra_args = None
    requested_processes = None
    requested_machines = None
    requested_dynamo_backend = None
    mixed_precision_value = "bf16"
    main_process_ip_value = "127.0.0.1"
    main_process_port_value = 29500
    machine_rank_value = 0
    same_network_value = True

    selected_device_ids: Optional[List[str]] = None

    if isinstance(config_payload, dict):
        accelerate_config_path = _extract_value(config_payload, "accelerate_config", "--accelerate_config")
        accelerate_extra_args = _extract_value(
            config_payload,
            "accelerate_extra_args",
            "--accelerate_extra_args",
        )
        requested_processes = _extract_value(config_payload, "num_processes", "--num_processes")
        requested_machines = _extract_value(config_payload, "num_machines", "--num_machines")
        requested_dynamo_backend = _extract_value(
            config_payload,
            "dynamo_backend",
            "--dynamo_backend",
            "training_dynamo_backend",
            "--training_dynamo_backend",
        )
        mixed_precision_value = _extract_value(
            config_payload, "mixed_precision", "--mixed_precision", default=mixed_precision_value
        )
        main_process_ip_value = _extract_value(
            config_payload,
            "main_process_ip",
            "--main_process_ip",
            default=main_process_ip_value,
        )
        main_process_port_value = _extract_value(
            config_payload,
            "main_process_port",
            "--main_process_port",
            default=main_process_port_value,
        )
        machine_rank_value = _extract_value(
            config_payload,
            "machine_rank",
            "--machine_rank",
            default=machine_rank_value,
        )
        same_network_value = _extract_value(
            config_payload,
            "same_network",
            "--same_network",
            default=same_network_value,
        )
        if isinstance(same_network_value, str):
            same_network_value = same_network_value.strip().lower() in {"1", "true", "yes", "on"}

        accelerate_visible_devices = _extract_value(
            config_payload,
            "accelerate_visible_devices",
            "--accelerate_visible_devices",
        )
        if isinstance(accelerate_visible_devices, str):
            tokens = [token.strip() for token in accelerate_visible_devices.split(",") if token.strip()]
            if tokens:
                selected_device_ids = tokens
        elif isinstance(accelerate_visible_devices, (list, tuple, set)):
            tokens: List[str] = []
            for item in accelerate_visible_devices:
                if isinstance(item, str):
                    stripped = item.strip()
                    if stripped:
                        tokens.append(stripped)
                else:
                    try:
                        tokens.append(str(int(item)))
                    except (TypeError, ValueError):
                        continue
            if tokens:
                selected_device_ids = tokens

    def _resolve_hf_token() -> Optional[str]:
        possible_env_vars = (
            "HUGGINGFACEHUB_API_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "HF_TOKEN",
            "HF_API_TOKEN",
        )
        for env_var in possible_env_vars:
            token = os.environ.get(env_var)
            if token:
                return token
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token()
            if token:
                return token
        except Exception:
            pass
        return None

    hf_token = _resolve_hf_token()

    def _launch_with_accelerate() -> Optional[int]:
        use_accelerate = False

        nonlocal accelerate_config_path, main_process_port_value, machine_rank_value

        config_candidate = accelerate_config_path
        if isinstance(config_candidate, str) and config_candidate.strip():
            use_accelerate = True

        extra_candidate = accelerate_extra_args
        if isinstance(extra_candidate, str) and extra_candidate.strip():
            use_accelerate = True

        proc_count = _safe_int(requested_processes, 1)
        machine_count = _safe_int(requested_machines, 1)
        if proc_count > 1 or machine_count > 1:
            use_accelerate = True

        dyn_backend_normalized = (requested_dynamo_backend or "").strip().lower()
        if dyn_backend_normalized and dyn_backend_normalized not in {"no", "none", ""}:
            use_accelerate = True

        if not use_accelerate:
            return None

        launch_env = os.environ.copy()
        if job_id:
            launch_env["SIMPLETUNER_JOB_ID"] = job_id
        if hf_token:
            for var in ("HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HF_API_TOKEN"):
                launch_env.setdefault(var, hf_token)

        if selected_device_ids:
            launch_env["CUDA_VISIBLE_DEVICES"] = ",".join(selected_device_ids)

        # Normalise accelerate config path if provided
        config_file_arg = None
        if isinstance(accelerate_config_path, str) and accelerate_config_path.strip():
            expanded_path = os.path.expanduser(accelerate_config_path.strip())
            accelerate_config_path = expanded_path
            if os.path.isfile(expanded_path):
                config_file_arg = expanded_path
            else:
                logger.warning("Accelerate config file not found at %s; falling back to CLI parameters", expanded_path)

        if mixed_precision_value:
            launch_env["MIXED_PRECISION"] = str(mixed_precision_value)
        launch_env.setdefault("MIXED_PRECISION", "bf16")

        if proc_count:
            launch_env["TRAINING_NUM_PROCESSES"] = str(proc_count)
        if machine_count:
            launch_env["TRAINING_NUM_MACHINES"] = str(machine_count)
        if dyn_backend_normalized:
            launch_env["TRAINING_DYNAMO_BACKEND"] = dyn_backend_normalized

        if accelerate_extra_args:
            launch_env["ACCELERATE_EXTRA_ARGS"] = str(accelerate_extra_args)

        if machine_count > 1:
            if not main_process_ip_value:
                raise ValueError(
                    "Main process IP is required when launching across multiple machines. "
                    "Provide it under Hardware > Accelerate settings or supply an accelerate config file."
                )
            if not main_process_port_value:
                raise ValueError(
                    "Main process port is required when launching across multiple machines. "
                    "Provide it under Hardware > Accelerate settings or supply an accelerate config file."
                )
            if machine_rank_value is None:
                raise ValueError(
                    "Machine rank must be specified for multi-node launches. "
                    "Set it under Hardware > Accelerate settings for each node."
                )

            main_process_port_int = _safe_int(main_process_port_value, 29500)
            if main_process_port_int <= 0:
                raise ValueError("Main process port must be a positive integer.")
            machine_rank_int = _safe_int(machine_rank_value, 0)
            if machine_rank_int < 0:
                raise ValueError("Machine rank must be zero or greater.")
            main_process_port_value = main_process_port_int
            machine_rank_value = machine_rank_int

        cmd = ["accelerate", "launch"]

        if config_file_arg:
            cmd.append(f"--config_file={config_file_arg}")
        else:
            cmd.extend(
                [
                    f"--mixed_precision={launch_env.get('MIXED_PRECISION', 'bf16')}",
                    f"--num_processes={launch_env.get('TRAINING_NUM_PROCESSES', '1')}",
                    f"--num_machines={launch_env.get('TRAINING_NUM_MACHINES', '1')}",
                    f"--dynamo_backend={launch_env.get('TRAINING_DYNAMO_BACKEND', 'no')}",
                ]
            )
            if machine_count > 1:
                cmd.append(f"--main_process_ip={main_process_ip_value}")
                cmd.append(f"--main_process_port={main_process_port_value}")
                cmd.append(f"--machine_rank={machine_rank_value}")
                if same_network_value:
                    cmd.append("--same_network")

        extra_args = []
        if accelerate_extra_args:
            try:
                extra_args = shlex.split(str(accelerate_extra_args))
            except ValueError:
                logger.warning("Failed to parse accelerate extra args; using raw string")
                extra_args = [str(accelerate_extra_args)]
        if extra_args:
            cmd.extend(extra_args)

        import simpletuner

        train_py = Path(simpletuner.__file__).parent / "train.py"
        cmd.append(str(train_py))

        cli_args: list[str] = []
        if isinstance(config_payload, dict):
            train_cli_payload = dict(config_payload)
            metadata_keys = [key for key in list(train_cli_payload.keys()) if isinstance(key, str) and key.startswith("__")]
            for key in metadata_keys:
                train_cli_payload.pop(key, None)
            for accel_key in {
                "accelerate_config",
                "--accelerate_config",
                "accelerate_extra_args",
                "--accelerate_extra_args",
                "main_process_ip",
                "--main_process_ip",
                "main_process_port",
                "--main_process_port",
                "machine_rank",
                "--machine_rank",
                "same_network",
                "--same_network",
            }:
                train_cli_payload.pop(accel_key, None)
            for webhook_key in ("--webhook_config", "webhook_config"):
                webhook_value = train_cli_payload.get(webhook_key)
                if isinstance(webhook_value, (dict, list)):
                    try:
                        train_cli_payload[webhook_key] = json.dumps(webhook_value)
                    except Exception:
                        train_cli_payload.pop(webhook_key, None)
            try:
                cli_args = mapping_to_cli_args(train_cli_payload)
            except Exception as exc:
                logger.warning("Failed to convert config payload to CLI args: %s", exc)
                cli_args = []
        if cli_args:
            launch_env.setdefault("CONFIG_BACKEND", "cmd")
            cmd.extend(cli_args)

        logging.info("Launching training via accelerate: %s", " ".join(cmd))

        popen_kwargs = {
            "env": launch_env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        if os.name != "nt":
            popen_kwargs["preexec_fn"] = os.setsid
        process = subprocess.Popen(cmd, **popen_kwargs)

        output_lock = threading.Lock()

        def _forward_output():
            if not process.stdout:
                return
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                with output_lock:
                    try:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    except Exception:
                        logger.info(line.rstrip())

        reader_thread = threading.Thread(target=_forward_output, daemon=True)
        reader_thread.start()

        try:
            while process.poll() is None:
                if callable(should_abort_callable) and should_abort_callable():
                    logger.info("Abort requested; terminating accelerate launcher")
                    _terminate_accelerate_process(process)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        logger.warning("Accelerate process unresponsive; forcing kill")
                        _kill_accelerate_process(process)
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received; terminating accelerate launcher")
            _terminate_accelerate_process(process)
            raise
        finally:
            if process.stdout:
                process.stdout.close()
            reader_thread.join(timeout=2)
            if process.poll() is None:
                _terminate_accelerate_process(process)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Accelerate process still alive; forcing kill")
                    _kill_accelerate_process(process)

        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"Accelerate launch exited with status {returncode}")
        return returncode

    try:
        accelerate_result = _launch_with_accelerate()
    except Exception as exc:
        webhook_handler = StateTracker.get_webhook_handler()
        if webhook_handler is not None:
            from simpletuner.simpletuner_sdk.api_state import APIState

            webhook_handler.send(
                message=f"Training job failed to start: {exc}",
                message_level="error",
            )
            payload = {"status": "training_failed", "error": str(exc)}
            try:
                import traceback

                payload["traceback"] = traceback.format_exc()
            except Exception:
                pass
            webhook_handler.send_raw(
                structured_data=payload,
                message_type="training.status",
                message_level="error",
                job_id=StateTracker.get_job_id(),
            )
            try:
                APIState.set_state("training_status", "failed")
                APIState.set_state("current_job_id", None)
            except Exception:
                pass
        APIState.set_state("training_status", "failed")
        raise

    if accelerate_result is not None:
        return accelerate_result

    if hf_token:
        for var in ("HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HF_API_TOKEN"):
            os.environ.setdefault(var, hf_token)

    try:
        trainer = Trainer(config=config_payload, job_id=job_id)

        if hf_token:
            trainer.hf_access_token = hf_token

    except Exception as e:
        webhook_handler = StateTracker.get_webhook_handler()
        if webhook_handler is not None:
            from simpletuner.simpletuner_sdk.api_state import APIState

            webhook_handler.send(
                message=f"Training job failed to start: {e}",
                message_level="error",
            )
            payload = {"status": "training_failed", "error": str(e)}
            try:
                import traceback

                payload["traceback"] = traceback.format_exc()
            except Exception:
                pass
            webhook_handler.send_raw(
                structured_data=payload,
                message_type="training.status",
                message_level="error",
                job_id=StateTracker.get_job_id(),
            )
            try:
                APIState.set_state("training_status", "failed")
                APIState.set_state("current_job_id", None)
            except Exception:
                pass
        raise e

    def _abort_monitor():
        if not callable(should_abort_callable):
            return
        while not trainer.should_abort:
            try:
                if should_abort_callable():
                    trainer.abort()
                    return
            except Exception:
                return
            time.sleep(0.5)

    if callable(should_abort_callable):
        trainer._external_abort_checker = should_abort_callable
        threading.Thread(target=_abort_monitor, daemon=True).start()

    trainer.run()
    return {"status": "completed"}


def _terminate_accelerate_process(process: subprocess.Popen) -> None:
    try:
        if process.poll() is None:
            if os.name != "nt" and getattr(process, "pid", None):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    return
                except Exception:
                    pass
            process.terminate()
    except Exception as exc:
        logger.warning("Failed to terminate accelerate process cleanly: %s", exc)


def _kill_accelerate_process(process: subprocess.Popen) -> None:
    try:
        if process.poll() is None:
            if os.name != "nt" and getattr(process, "pid", None):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    return
                except Exception:
                    pass
            process.kill()
    except Exception as exc:
        logger.warning("Failed to kill accelerate process: %s", exc)
