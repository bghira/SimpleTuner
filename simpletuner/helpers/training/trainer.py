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
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import huggingface_hub
from torch.distributed.fsdp.api import ShardedOptimStateDictConfig, ShardedStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

import wandb
from simpletuner.helpers import log_format  # noqa
from simpletuner.helpers.caching.memory import reclaim_memory
from simpletuner.helpers.configuration.cli_utils import mapping_to_cli_args
from simpletuner.helpers.configuration.loader import load_config
from simpletuner.helpers.data_backend.dataset_types import DatasetType, ensure_dataset_type
from simpletuner.helpers.data_backend.factory import (
    BatchFetcher,
    configure_multi_databackend,
    run_distillation_cache_generation,
)
from simpletuner.helpers.data_backend.runtime import random_dataloader_iterator
from simpletuner.helpers.distillation.registry import DistillationRegistry
from simpletuner.helpers.distillation.requirements import EMPTY_PROFILE, DistillerRequirementProfile
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.publishing import PublishingManager
from simpletuner.helpers.publishing.huggingface import HubManager
from simpletuner.helpers.training import _flatten_parameters, trainable_parameter_count
from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase
from simpletuner.helpers.training.custom_schedule import get_lr_scheduler
from simpletuner.helpers.training.deepspeed import prepare_model_for_deepspeed
from simpletuner.helpers.training.deepspeed_optimizers import DEFAULT_OPTIMIZER as DS_DEFAULT_OPTIMIZER
from simpletuner.helpers.training.deepspeed_optimizers import sanitize_optimizer_block
from simpletuner.helpers.training.default_settings.safety_check import safety_check
from simpletuner.helpers.training.evaluation import ModelEvaluator
from simpletuner.helpers.training.min_snr_gamma import compute_snr
from simpletuner.helpers.training.multi_process import _get_rank as get_rank
from simpletuner.helpers.training.multi_process import broadcast_object_from_main
from simpletuner.helpers.training.optimizer_param import (
    cpu_offload_optimizer,
    create_optimizer_with_param_groups,
    determine_optimizer_class_with_config,
    determine_params_to_optimize,
    is_bitsandbytes_available,
    is_lr_schedulefree,
    is_lr_scheduler_disabled,
)
from simpletuner.helpers.training.optimizers.adamw_bfloat16 import AdamWBF16
from simpletuner.helpers.training.peft_init import init_lokr_network_with_perturbed_normal
from simpletuner.helpers.training.script_runner import run_hook_script
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


def _summarize_accelerate_failure(exit_code: int, lines: Sequence[str]) -> tuple[str, Optional[str]]:
    """Derive a concise failure summary and optional log excerpt from accelerate output."""

    cleaned: List[str] = [line.rstrip("\n") for line in lines]
    non_empty = [line.strip() for line in cleaned if line and line.strip()]

    best_index: Optional[int] = None
    best_line: Optional[str] = None

    for idx in range(len(cleaned) - 1, -1, -1):
        candidate_raw = cleaned[idx].strip()
        if not candidate_raw:
            continue
        lowered = candidate_raw.lower()
        if "cuda out of memory" in lowered:
            best_index = idx
            best_line = candidate_raw
            break
        if "runtimeerror" in lowered and "cuda" in lowered:
            if best_line is None:
                best_index = idx
                best_line = candidate_raw
        if "childfailederror" in lowered:
            if best_line is None:
                best_index = idx
                best_line = candidate_raw
        if "error" in lowered and ("[error]" in lowered or lowered.startswith("error")):
            if best_line is None:
                best_index = idx
                best_line = candidate_raw

    if best_line is None:
        for idx in range(len(cleaned) - 1, -1, -1):
            candidate = cleaned[idx].strip()
            if candidate:
                best_index = idx
                best_line = candidate
                break

    summary = f"Accelerate launch exited with status {exit_code}"
    if best_line:
        summary = f"{summary}: {best_line.strip()}"
    summary = summary[:512]

    excerpt: Optional[str] = None
    if best_index is not None:
        start = max(0, best_index - 5)
        end = min(len(cleaned), best_index + 6)
        snippet_lines = [line.strip() for line in cleaned[start:end] if line.strip()]
        if snippet_lines:
            excerpt = "\n".join(snippet_lines)
    elif non_empty:
        excerpt = "\n".join(non_empty[-10:])

    if excerpt and len(excerpt) > 4000:
        excerpt = excerpt[-4000:]

    return summary, excerpt


def _setup_logger(name: str, *, env_var: str | None = None, default_level: str = "INFO") -> logging.Logger:
    if hasattr(log_format, "ensure_custom_handlers"):
        log_format.ensure_custom_handlers()

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
    # Don't add handlers - let the logger propagate to the root logger
    # which has the properly configured handlers from log_format.py
    logger_instance.propagate = True
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

from simpletuner.configure import model_classes, model_labels

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
from diffusers.utils import logging as diffusers_logging
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
if hasattr(diffusers_logging, "set_verbosity_warning"):
    diffusers_logging.set_verbosity_warning()

# Configure third-party loggers after imports
if hasattr(log_format, "configure_third_party_loggers"):
    log_format.configure_third_party_loggers()


class Trainer:
    # Provide safe defaults so tests that bypass __init__ via object.__new__ still have attributes resolved.
    sidecar_optimizer = None
    sidecar_lr_scheduler = None
    sidecar_lr = None
    sidecar_is_schedulefree = False
    sidecar_scheduler_disabled = False
    publishing_manager = None

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
        self.publishing_manager = None
        self.should_abort = False
        self._external_abort_checker = None
        self._manual_validation_consumer: Optional[Callable[[], bool]] = None
        self._manual_checkpoint_consumer: Optional[Callable[[], bool]] = None
        self.ema_model = None
        self.job_id = job_id
        self.sidecar_optimizer = None
        self.sidecar_lr_scheduler = None
        self.sidecar_lr = None
        self.sidecar_is_schedulefree = False
        self.sidecar_scheduler_disabled = False
        self._hub_upload_executor: ThreadPoolExecutor | None = None
        self._hub_upload_futures: deque[tuple[str, Future]] = deque()
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

        if getattr(self, "config", None) is not None and self.config.model_family in ModelRegistry.model_families().keys():
            self.model = ModelRegistry.model_families()[self.config.model_family](self.config, self.accelerator)
            self.model.check_user_config()
            StateTracker.set_model(self.model)
        if self.webhook_handler and isinstance(self.model, VideoModelFoundation) and not self.webhook_handler.send_video:
            self.configure_webhook(send_startup_message=False)
        self._misc_init()
        self.validation = None
        self.distiller_requirement_profile: DistillerRequirementProfile = EMPTY_PROFILE
        # this updates self.config further, so we will run it here.
        self.init_noise_schedule()

    def _config_to_obj(self, config):
        if not config:
            return None
        return type("Config", (object,), config)

    def register_manual_validation_trigger(self, consumer: Callable[[], bool]) -> None:
        """Register a callable that returns True once per manual validation request."""
        self._manual_validation_consumer = consumer

    def _consume_manual_validation_request(self) -> bool:
        if self._manual_validation_consumer is None:
            return False
        return bool(self._manual_validation_consumer())

    def register_manual_checkpoint_trigger(self, consumer: Callable[[], bool]) -> None:
        """Register a callable that returns True once per manual checkpoint request."""
        self._manual_checkpoint_consumer = consumer

    def _consume_manual_checkpoint_request(self) -> bool:
        if self._manual_checkpoint_consumer is None:
            return False
        return bool(self._manual_checkpoint_consumer())

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

    def _config_uses_bitsandbytes(self) -> bool:
        if not getattr(self, "config", None):
            return False

        def _contains_bnb(value: object) -> bool:
            if isinstance(value, str):
                return "bnb" in value.lower()
            if isinstance(value, dict):
                return any(_contains_bnb(item) for item in value.values())
            if isinstance(value, (list, tuple, set)):
                return any(_contains_bnb(item) for item in value)
            return False

        for attr_value in vars(self.config).values():
            try:
                if _contains_bnb(attr_value):
                    return True
            except Exception:
                continue
        return False

    def _init_publishing_manager(self):
        publishing_config = getattr(self.config, "publishing_config", None)
        if publishing_config in (None, "", [], {}, "None"):
            return

        is_main_process = True
        if getattr(self, "accelerator", None) is not None:
            is_main_process = getattr(self.accelerator, "is_main_process", True)
        if not is_main_process:
            logger.debug("Skipping publishing manager initialisation on non-main process.")
            return

        try:
            self.publishing_manager = PublishingManager(publishing_config)
            logger.info("Publishing manager initialised with %s provider(s).", len(self.publishing_manager.providers))
        except Exception as exc:
            logger.error("Failed to initialise publishing providers: %s", exc)
            self.publishing_manager = None

    def _enable_dynamo_dynamic_output_capture(self) -> None:
        try:
            import torch._dynamo as torch_dynamo
        except Exception as exc:
            logger.warning("Unable to configure Torch Dynamo dynamic output capture: %s", exc)
            return

        config_obj = getattr(torch_dynamo, "config", None)
        if config_obj is None:
            logger.debug("Torch Dynamo config unavailable; skipping dynamic output capture configuration.")
            return
        if not hasattr(config_obj, "capture_dynamic_output_shape_ops"):
            logger.debug(
                "Torch Dynamo config lacks capture_dynamic_output_shape_ops; skipping dynamic output capture configuration."
            )
            return
        if getattr(config_obj, "capture_dynamic_output_shape_ops", False):
            return

        config_obj.capture_dynamic_output_shape_ops = True
        logger.info("Torch Dynamo capture_dynamic_output_shape_ops enabled for bitsandbytes models.")

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

        checkpoint_step_interval_value = getattr(self.config, "checkpoint_step_interval", None)
        if checkpoint_step_interval_value in (None, "", "None", 0):
            legacy_checkpoint_steps = getattr(self.config, "checkpointing_steps", None)
            if legacy_checkpoint_steps not in (None, "", "None", 0):
                setattr(self.config, "checkpoint_step_interval", legacy_checkpoint_steps)
                checkpoint_step_interval_value = legacy_checkpoint_steps
        else:
            legacy_checkpoint_steps = getattr(self.config, "checkpointing_steps", None)
            if legacy_checkpoint_steps in (None, "", "None", 0):
                setattr(self.config, "checkpointing_steps", checkpoint_step_interval_value)

        checkpoint_epoch_interval_value = getattr(self.config, "checkpoint_epoch_interval", None)
        if checkpoint_epoch_interval_value in ("", "None", 0):
            setattr(self.config, "checkpoint_epoch_interval", None)

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

            will_create_dynamo_plugin = resolved_dynamo_backend and resolved_dynamo_backend != DynamoBackend.NO

            accelerator_kwargs = dict(
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                mixed_precision=self.config.mixed_precision,
                log_with=report_to,
                project_config=self.config.accelerator_project_config,
                kwargs_handlers=accelerator_custom_config,
            )

            if not will_create_dynamo_plugin and dynamo_backend_env:
                accelerator_kwargs["dynamo_backend"] = dynamo_backend_env

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
            if will_create_dynamo_plugin:
                if is_bitsandbytes_available and self._config_uses_bitsandbytes():
                    self._enable_dynamo_dynamic_output_capture()

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
                if "dynamo_plugin and dynamo_backend" in str(err):
                    has_backend = "dynamo_backend" in accelerator_kwargs
                    has_plugin = "dynamo_plugin" in accelerator_kwargs
                    backend_val = accelerator_kwargs.get("dynamo_backend", "not set")
                    plugin_val = "set" if has_plugin else "not set"
                    raise ValueError(
                        f"Conflicting Torch Dynamo configuration detected. "
                        f"Cannot pass both dynamo_plugin and dynamo_backend to Accelerator. "
                        f"Current state: dynamo_backend={backend_val} (present={has_backend}), "
                        f"dynamo_plugin={plugin_val} (present={has_plugin}). "
                        f"Original error: {err}"
                    ) from err
                elif not should_override_bf16 and self._should_force_bf16_override(err):
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
            # Clean up any handlers that Accelerate or DeepSpeed may have added
            if hasattr(log_format, "ensure_custom_handlers"):
                log_format.ensure_custom_handlers()
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
        limit_all_gathers = getattr(self.config, "fsdp_limit_all_gathers", True)
        cpu_offload = getattr(self.config, "fsdp_cpu_offload", False)
        activation_checkpointing = getattr(self.config, "fsdp_activation_checkpointing", False)
        state_dict_type = getattr(self.config, "fsdp_state_dict_type", None)
        auto_wrap_policy = getattr(self.config, "fsdp_auto_wrap_policy", None)
        transformer_cls = getattr(self.config, "fsdp_transformer_layer_cls_to_wrap", None)
        if isinstance(transformer_cls, str) and transformer_cls.strip() == "":
            transformer_cls = None

        if activation_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            logger.info(
                "FSDP activation checkpointing enabled; disabling model-level gradient checkpointing to avoid conflicts."
            )
            setattr(self.config, "gradient_checkpointing", False)

        plugin_kwargs = {
            "fsdp_version": fsdp_version,
            "reshard_after_forward": reshard_after_forward,
            "cpu_ram_efficient_loading": cpu_ram_efficient_loading,
            "limit_all_gathers": bool(limit_all_gathers),
        }

        if cpu_offload is not None:
            plugin_kwargs["cpu_offload"] = cpu_offload
        if activation_checkpointing:
            plugin_kwargs["activation_checkpointing"] = True

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

        parsed_transformer_cls = None
        if transformer_cls:
            if isinstance(transformer_cls, str):
                parsed_transformer_cls = [name.strip() for name in transformer_cls.split(",") if name.strip()]
            elif isinstance(transformer_cls, (list, tuple, set)):
                parsed_transformer_cls = [str(name).strip() for name in transformer_cls if str(name).strip()]
            else:
                parsed_transformer_cls = [str(transformer_cls).strip()]
            if parsed_transformer_cls:
                plugin_kwargs["transformer_cls_names_to_wrap"] = parsed_transformer_cls

        logger.info(
            "FSDP v%s configuration detected (%s); enabling FullyShardedDataParallelPlugin%s.",
            fsdp_version,
            ", ".join(f"{k}={v}" for k, v in plugin_kwargs.items()),
            f" (reshard_after_forward={reshard_after_forward})" if fsdp_version >= 2 else "",
        )
        plugin = FullyShardedDataParallelPlugin(**plugin_kwargs)

        def _normalize_candidate_names(raw_names: Any) -> Optional[List[str]]:
            if not raw_names:
                return None
            if isinstance(raw_names, str):
                raw_names = [raw_names]
            try:
                iterator = list(raw_names)
            except TypeError:
                iterator = [str(raw_names)]
            normalized: List[str] = []
            for name in iterator:
                candidate = str(name).strip()
                if candidate and candidate not in normalized:
                    normalized.append(candidate)
            return normalized or None

        original_plugin_names = _normalize_candidate_names(getattr(plugin, "transformer_cls_names_to_wrap", None))
        plugin_names = original_plugin_names
        plugin_names_origin = "config" if parsed_transformer_cls else None

        model_instance = getattr(self, "model", None)
        model_family = getattr(self.config, "model_family", None)
        model_family_cls = ModelRegistry.get(model_family) if model_family else None

        component = None
        candidate_names: Optional[List[str]] = None
        candidate_source = None

        if model_instance is not None:
            component = model_instance.get_trained_component(unwrap_model=False)
            candidate_names = _normalize_candidate_names(getattr(component, "_no_split_modules", None))
            if candidate_names:
                candidate_source = "instance"
            else:
                candidate_names = _normalize_candidate_names(getattr(model_instance, "_no_split_modules", None))
                if candidate_names:
                    candidate_source = "instance"

        if candidate_names is None and model_family_cls is not None:
            class_level_candidates = _normalize_candidate_names(getattr(model_family_cls, "_no_split_modules", None))
            if not class_level_candidates:
                transformer_cls = getattr(model_family_cls, "MODEL_CLASS", None)
                if transformer_cls is not None:
                    class_level_candidates = _normalize_candidate_names(getattr(transformer_cls, "_no_split_modules", None))
                    if class_level_candidates is None:
                        try:
                            transformer_instance = transformer_cls()  # type: ignore[call-arg]
                        except Exception:
                            transformer_instance = None
                        else:
                            class_level_candidates = _normalize_candidate_names(
                                getattr(transformer_instance, "_no_split_modules", None)
                            )
                        finally:
                            transformer_instance = None
            if class_level_candidates:
                candidate_names = class_level_candidates
                candidate_source = "class"

        def _collect_excluded_candidates() -> List[str]:
            excluded: List[str] = []
            sources = [
                component,
                model_instance,
                model_family_cls,
                getattr(model_family_cls, "MODEL_CLASS", None) if model_family_cls else None,
            ]
            for source in sources:
                if source is None:
                    continue
                source_exclusions = _normalize_candidate_names(getattr(source, "_fsdp_exclude_auto_wrap_modules", None))
                if not source_exclusions:
                    continue
                for name in source_exclusions:
                    if name not in excluded:
                        excluded.append(name)
            return excluded

        excluded_candidates = _collect_excluded_candidates()

        if excluded_candidates and getattr(plugin, "cpu_ram_efficient_loading", False):
            logger.warning(
                "Disabling FSDP cpu_ram_efficient_loading because some modules must remain unsharded: %s",
                ", ".join(excluded_candidates),
            )
            plugin.cpu_ram_efficient_loading = False
            setattr(self.config, "fsdp_cpu_ram_efficient_loading", False)

        def _apply_exclusions(names: Optional[List[str]], origin: str) -> Optional[List[str]]:
            if not names or not excluded_candidates:
                return names
            filtered = [name for name in names if name not in excluded_candidates]
            removed = [name for name in names if name in excluded_candidates]
            if removed:
                logger.info(
                    "Removing FSDP auto-wrap candidates excluded by %s: %s",
                    origin,
                    ", ".join(removed),
                )
            return filtered or None

        candidate_names = _apply_exclusions(candidate_names, "model configuration")
        plugin_names = _apply_exclusions(plugin_names, "model configuration")
        plugin.transformer_cls_names_to_wrap = plugin_names
        if plugin_names:
            setattr(self.config, "fsdp_transformer_layer_cls_to_wrap", ",".join(plugin_names))
        else:
            setattr(self.config, "fsdp_transformer_layer_cls_to_wrap", None)

        base_component_name = "<uninitialized>"
        if component is not None:
            base_component_name = component.__class__.__name__
        elif model_family_cls is not None:
            transformer_cls = getattr(model_family_cls, "MODEL_CLASS", None)
            if transformer_cls is not None:
                base_component_name = transformer_cls.__name__

        logger.info("FSDP base component: %s", base_component_name)

        if getattr(plugin, "auto_wrap_policy", None) is not None:
            if candidate_names:
                source_label = f"model family '{model_family}'" if candidate_source == "class" and model_family else "model"
                logger.info(
                    "FSDP auto-wrap candidates from %s: %s",
                    source_label,
                    ", ".join(candidate_names),
                )
            else:
                logger.info("FSDP auto-wrap candidates from model: <none>")

            if not candidate_names and component is not None:
                # As a fallback, scan module types to surface potential block classes.
                # The default limit of 8 is chosen to avoid excessive auto-detection in very large models.
                # You can override this limit by setting 'fsdp_max_auto_detect_candidates' in self.config.
                max_auto_candidates = getattr(self.config, "fsdp_max_auto_detect_candidates", 8)
                auto_candidates: List[str] = []
                for module in component.children():
                    cls_name = module.__class__.__name__
                    if cls_name not in auto_candidates and sum(p.numel() for p in module.parameters(recurse=False)) > 0:
                        auto_candidates.append(cls_name)
                        if len(auto_candidates) >= max_auto_candidates:
                            break
                if auto_candidates:
                    logger.info(
                        "FSDP auto-detected candidate classes: %s",
                        ", ".join(auto_candidates),
                    )
                    candidate_names = _apply_exclusions(auto_candidates, "auto-detected modules")
                    candidate_source = "auto"
                    setattr(component, "_no_split_modules", tuple(auto_candidates))

            if candidate_names and plugin_names:
                missing_from_plugin = [name for name in candidate_names if name and name not in plugin_names]
                if missing_from_plugin:
                    logger.info(
                        "Extending FSDP transformer classes to wrap with model hints: %s",
                        ", ".join(missing_from_plugin),
                    )
                    plugin_names = plugin_names + missing_from_plugin
                    plugin.transformer_cls_names_to_wrap = plugin_names
                    setattr(self.config, "fsdp_transformer_layer_cls_to_wrap", ",".join(plugin_names))

            if component is not None and plugin_names:
                try:
                    from accelerate.utils.dataclasses import get_module_class_from_name
                except ImportError:
                    get_module_class_from_name = None  # type: ignore
                if get_module_class_from_name and plugin_names_origin == "config":
                    valid_names: List[str] = []
                    invalid_names: List[str] = []
                    for name in plugin_names:
                        if get_module_class_from_name(component, name) is None:
                            invalid_names.append(name)
                        else:
                            valid_names.append(name)
                    if invalid_names:
                        logger.warning(
                            "Removing unknown FSDP transformer classes that are not present in the current model: %s",
                            ", ".join(invalid_names),
                        )
                        plugin_names = valid_names or None
                        plugin.transformer_cls_names_to_wrap = plugin_names
                        if plugin_names:
                            setattr(self.config, "fsdp_transformer_layer_cls_to_wrap", ",".join(plugin_names))
                        else:
                            setattr(self.config, "fsdp_transformer_layer_cls_to_wrap", None)

        if getattr(plugin, "transformer_cls_names_to_wrap", None) is None and candidate_names:
            names_list = [name for name in candidate_names if name]
            if names_list:
                logger.info(
                    "Using model-provided _no_split_modules for FSDP wrapping: %s",
                    ", ".join(names_list),
                )
                plugin.transformer_cls_names_to_wrap = names_list
                plugin_names = names_list
                plugin_names_origin = "model"
                current_cls_setting = getattr(self.config, "fsdp_transformer_layer_cls_to_wrap", None)
                if current_cls_setting is None:
                    setattr(self.config, "fsdp_transformer_layer_cls_to_wrap", ",".join(names_list))

        logger.info(
            "Resolved FSDP transformer classes to wrap: %s",
            (
                ", ".join(plugin.transformer_cls_names_to_wrap or [])
                if getattr(plugin, "transformer_cls_names_to_wrap", None)
                else "<none>"
            ),
        )

        if getattr(plugin, "cpu_ram_efficient_loading", False) and not getattr(
            plugin, "transformer_cls_names_to_wrap", None
        ):
            logger.warning(
                "Disabling FSDP cpu_ram_efficient_loading because no transformer blocks were wrapped. "
                "CPU RAM efficient loading requires per-layer sharding."
            )
            plugin.cpu_ram_efficient_loading = False
            setattr(self.config, "fsdp_cpu_ram_efficient_loading", False)
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

        state_dict_type = getattr(fsdp_plugin, "state_dict_type", None)
        if state_dict_type in (StateDictType.SHARDED_STATE_DICT, "sharded_state_dict"):
            fsdp_plugin.state_dict_config = ShardedStateDictConfig(
                offload_to_cpu=True,
                _use_dtensor=False,
            )
            fsdp_plugin.optim_state_dict_config = ShardedOptimStateDictConfig(
                offload_to_cpu=True,
                _use_dtensor=False,
            )

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
                    self.init_lyrics_embedder_training,
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
            AttentionBackendController.apply(self.config, AttentionPhase.EVAL)
            self.init_benchmark_base_model()
            AttentionBackendController.apply(self.config, AttentionPhase.TRAIN)
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
        finally:
            self._finish_hub_uploads()

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
            logging.debug(f"Creating webhook: {webhook_config}")
        else:
            webhook_config = getattr(getattr(self, "config", None), "webhook_config", None)
        if webhook_config is None:
            return

        # Handle string webhook_config (file path or JSON string)
        if isinstance(webhook_config, str):
            import json
            import os

            if webhook_config.startswith("{") or webhook_config.startswith("["):
                # Parse as JSON string
                try:
                    parsed_config = json.loads(webhook_config)
                    # Normalize single dict to list for consistency
                    if isinstance(parsed_config, dict):
                        webhook_config = [parsed_config]
                    elif isinstance(parsed_config, list):
                        webhook_config = parsed_config
                    else:
                        logging.error(f"webhook_config must be dict or list, got {type(parsed_config)}")
                        raise ValueError(f"Invalid webhook_config type: {type(parsed_config)}")
                    logging.info("Parsed webhook config from JSON string")
                except json.JSONDecodeError as e:
                    logging.error(f"Could not load webhook_config (invalid JSON): {e}")
                    raise
            else:
                # Try to load from file
                if os.path.isfile(webhook_config):
                    try:
                        with open(webhook_config, "r") as f:
                            loaded_config = json.load(f)
                            # Normalize single dict to list for consistency
                            if isinstance(loaded_config, dict):
                                webhook_config = [loaded_config]
                            elif isinstance(loaded_config, list):
                                webhook_config = loaded_config
                            else:
                                logging.error(f"webhook_config must be dict or list, got {type(loaded_config)}")
                                raise ValueError(f"Invalid webhook_config type: {type(loaded_config)}")
                        logging.info(f"Loaded webhook config from file: {webhook_config}")
                    except Exception as e:
                        logging.error(f"Could not load webhook_config from file: {e}")
                        raise
                else:
                    logging.error(f"Could not find webhook_config file: {webhook_config}")
                    raise ValueError(f"webhook_config file not found: {webhook_config}")
        elif isinstance(webhook_config, dict):
            # Normalize single dict to list for consistency
            webhook_config = [webhook_config]
        # list is already in the correct format

        from simpletuner.helpers.webhooks.handler import WebhookHandler

        send_video_flag = self._infer_send_video_flag(raw_config)
        video_framerate = self._infer_video_framerate(raw_config)
        if send_video_flag and video_framerate is None:
            candidate = getattr(self, "config", None)
            if candidate is not None:
                fallback = getattr(candidate, "framerate", None)
                if fallback not in (None, "", "None"):
                    try:
                        video_framerate = int(float(fallback))
                    except (ValueError, TypeError):
                        video_framerate = None
        if send_video_flag and video_framerate is None:
            video_framerate = 30

        self.webhook_handler = WebhookHandler(
            self.accelerator,
            (
                f"{getattr(self.config, 'tracker_project_name', 'unknown')} {getattr(self.config, 'tracker_run_name', 'unknown')}"
                if hasattr(self, "config")
                else "unknown"
            ),
            send_video=send_video_flag,
            video_framerate=video_framerate,
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

    def _extract_config_value(self, source, *keys):
        if source is None:
            return None
        if hasattr(source, "get"):
            for key in keys:
                if key in source and source[key] not in (None, "", "None"):
                    return source[key]
        for key in keys:
            if hasattr(source, key):
                value = getattr(source, key)
                if value not in (None, "", "None"):
                    return value
            if hasattr(source, "__dict__"):
                raw_dict = getattr(source, "__dict__", {})
                if key in raw_dict and raw_dict[key] not in (None, "", "None"):
                    return raw_dict[key]
        return None

    def _infer_send_video_flag(self, raw_config):
        if isinstance(self.model, VideoModelFoundation):
            return True
        candidate_family = None
        if getattr(self, "config", None) is not None:
            candidate_family = getattr(self.config, "model_family", None)
        if not candidate_family and raw_config is not None:
            candidate_family = self._extract_config_value(raw_config, "model_family", "--model_family")
        if candidate_family:
            candidate_family = str(candidate_family).strip()
            model_cls = ModelRegistry.model_families().get(candidate_family)
            if model_cls and issubclass(model_cls, VideoModelFoundation):
                return True
        return False

    def _infer_video_framerate(self, raw_config):
        framerate = None
        if getattr(self, "config", None) is not None:
            framerate = getattr(self.config, "framerate", None)
        if framerate in (None, "", "None") and raw_config is not None:
            framerate = self._extract_config_value(raw_config, "framerate", "--framerate")
        if framerate in (None, "", "None"):
            return None
        try:
            return int(float(framerate))
        except (ValueError, TypeError):
            return None

    def _push_to_hub_background_enabled(self) -> bool:
        return (
            bool(getattr(self.config, "push_to_hub_background", False))
            and bool(getattr(self.config, "push_to_hub", False))
            and (not self.accelerator or self.accelerator.is_main_process)
        )

    def _get_hub_executor(self) -> ThreadPoolExecutor | None:
        if not self._push_to_hub_background_enabled():
            return None
        if self._hub_upload_executor is None:
            self._hub_upload_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf_push_")
        return self._hub_upload_executor

    def _run_post_upload_script(self, *, local_path: str | None = None, remote_path: str | None = None) -> None:
        script_template = getattr(self.config, "post_upload_script", None)
        if script_template in (None, "", "None"):
            return
        current_step = None
        if isinstance(getattr(self, "state", None), dict):
            current_step = self.state.get("global_step")
        run_hook_script(
            script_template,
            config=self.config,
            local_path=local_path,
            remote_path=remote_path,
            global_step=current_step,
        )

    def _drain_hub_upload_futures(self, wait: bool = False) -> None:
        if not self._hub_upload_futures:
            return
        remaining: deque[tuple[str, Future]] = deque()
        while self._hub_upload_futures:
            description, future = self._hub_upload_futures.popleft()
            if wait or future.done():
                try:
                    future.result()
                except Exception as exc:
                    logger.error("Hugging Face upload task '%s' failed: %s", description, exc, exc_info=True)
                    if self.webhook_handler:
                        self.webhook_handler.send(
                            message=f"Hugging Face upload failed for {description}: {exc}",
                            message_level="error",
                        )
            else:
                remaining.append((description, future))
        self._hub_upload_futures = remaining

    def _schedule_hub_upload(self, description: str, upload_fn: Callable[[], object]) -> None:
        executor = self._get_hub_executor()

        def _wrapped_upload():
            result = upload_fn()
            remote_path = None
            local_path = None
            if isinstance(result, tuple):
                if len(result) >= 1:
                    remote_path = result[0]
                if len(result) >= 2:
                    local_path = result[1]
                # If third value looks like a repo URL and remote path is missing, use it
                if remote_path in (None, "") and len(result) >= 3:
                    remote_path = result[2]
            self._run_post_upload_script(local_path=local_path, remote_path=remote_path)
            return result

        if executor is None:
            _wrapped_upload()
            return
        self._drain_hub_upload_futures(wait=False)
        future = executor.submit(_wrapped_upload)
        self._hub_upload_futures.append((description, future))
        logger.info("Scheduled background Hugging Face upload: %s", description)

    def _finish_hub_uploads(self) -> None:
        self._drain_hub_upload_futures(wait=True)
        if self._hub_upload_executor is not None:
            self._hub_upload_executor.shutdown(wait=True)
            self._hub_upload_executor = None

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
        self._init_publishing_manager()
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

        model_implementation = ModelRegistry.model_families().get(model_family)
        StateTracker.set_model_family(model_family)
        label = getattr(model_implementation, "NAME", None)
        if not label:
            label = model_labels.get(model_family, model_family.replace("_", " ").title())
        self.config.model_type_label = label
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
        # Suppress verbose transformers/diffusers logging before loading models
        if hasattr(log_format, "configure_third_party_loggers"):
            log_format.configure_third_party_loggers()

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
        # Suppress verbose transformers/diffusers logging before loading models
        if hasattr(log_format, "configure_third_party_loggers"):
            log_format.configure_third_party_loggers()

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
            distiller_profile = self._resolve_distiller_profile()
            configure_multi_databackend(
                self.config,
                accelerator=self.accelerator,
                text_encoders=self.model.text_encoders,
                tokenizers=self.model.tokenizers,
                model=self.model,
                distiller_profile=distiller_profile,
                distillation_method=getattr(self.config, "distillation_method", None),
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

        if self.config.validation_disable:
            self.validation_prompt_metadata = None
            self.validation_shortnames = None
            self.validation_negative_prompt_embeds = None
            self.validation_negative_pooled_embeds = None
            StateTracker.set_validation_sample_images([])
            self.accelerator.wait_for_everyone()
            return

        validation_metadata = None
        if self.accelerator.is_main_process:
            validation_metadata = prepare_validation_prompt_list(
                args=self.config,
                embed_cache=StateTracker.get_default_text_embed_cache(),
                model=self.model,
            )
        validation_metadata = broadcast_object_from_main(validation_metadata)
        if validation_metadata is None:
            StateTracker.set_validation_sample_images([])
        else:
            StateTracker.set_validation_sample_images(validation_metadata.get("validation_sample_images") or [])
        self.validation_prompt_metadata = validation_metadata
        if validation_metadata is None:
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

    def init_lyrics_embedder_training(self):
        """
        Enable training for the ACE-Step lyrics embedder when requested.
        """
        if not getattr(self.config, "lyrics_embedder_train", False):
            return
        modules_fn = getattr(self.model, "get_lyrics_embedder_modules", None)
        if not callable(modules_fn):
            logger.warning("lyrics_embedder_train enabled, but model does not expose lyrics embedder modules.")
            return

        modules = modules_fn(unwrap=False) or []
        if not modules:
            logger.warning("lyrics_embedder_train enabled, but no lyrics embedder modules were found to unfreeze.")
            return

        enabled = []
        enable_fn = getattr(self.model, "enable_lyrics_embedder_training", None)
        if callable(enable_fn):
            enabled = enable_fn() or []
        else:
            for name, module in modules:
                if module is None:
                    continue
                module.requires_grad_(True)
                module.train()
                enabled.append(name)

        if enabled:
            logger.info("Enabled training for lyrics embedder modules: %s", ", ".join(enabled))

    def init_post_load_freeze(self):
        if self.config.layer_freeze_strategy == "bitfit":
            from simpletuner.helpers.training.model_freeze import apply_bitfit_freezing

            if self.model.get_trained_component() is not None:
                logger.info(f"Applying BitFit freezing strategy to the {self.model.MODEL_TYPE.value}.")
                self.model.model = apply_bitfit_freezing(unwrap_model(self.accelerator, self.model.model), self.config)
        self.enable_gradient_checkpointing()

    def _resolve_distiller_profile(self) -> DistillerRequirementProfile:
        """Return and cache the active distiller requirement profile."""
        method = getattr(self.config, "distillation_method", None)
        if not method:
            StateTracker.set_distiller_profile(None, EMPTY_PROFILE)
            self.distiller_requirement_profile = EMPTY_PROFILE
            return EMPTY_PROFILE

        profile = DistillationRegistry.get_requirement_profile(method)
        self.distiller_requirement_profile = profile
        StateTracker.set_distiller_profile(method, profile)
        return profile

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
                run_distillation_cache_generation(self.distiller)
        except Exception as e:
            logger.error(f"Failed to initialize distillation: {e}")
            raise

    def enable_gradient_checkpointing(self):
        if self.config.gradient_checkpointing:
            fsdp_plugin = getattr(getattr(self.accelerator, "state", None), "fsdp_plugin", None)
            if fsdp_plugin and getattr(self.config, "fsdp_activation_checkpointing", False):
                logger.info("Skipping model-level gradient checkpointing because FSDP activation checkpointing is active.")
                return
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

    def _ensure_parameter_dtype(self, parameters, target_dtype: torch.dtype, optimizer_name: str | None = None):
        converted = 0
        for param_or_group in parameters:
            if isinstance(param_or_group, dict):
                cand_params = param_or_group.get("params", [])
            else:
                cand_params = [param_or_group]

            for param in cand_params:
                if not hasattr(param, "data") or getattr(param, "is_meta", False):
                    continue
                tensor = param.data
                if tensor is None or tensor.dtype == target_dtype:
                    continue
                param.data = tensor.to(dtype=target_dtype)
                converted += 1

        if converted > 0:
            name_fragment = optimizer_name or getattr(self.config, "optimizer", "optimizer")
            logger.info(
                "Converted %s parameter%s to %s for %s compatibility.",
                converted,
                "" if converted == 1 else "s",
                target_dtype,
                name_fragment,
            )

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
        steps_per_epoch = max(self.config.num_update_steps_per_epoch, 1)
        target_epochs = float(self.config.num_train_epochs or 0)
        if getattr(self.config, "overrode_max_train_steps", False):
            self.config.max_train_steps = int(math.ceil(target_epochs * steps_per_epoch))
            # Afterwards we recalculate our number of training epochs
            self.config.num_train_epochs = math.ceil(self.config.max_train_steps / steps_per_epoch)
            logger.info(
                "After removing any undesired samples and updating cache entries, we have settled on"
                f" {self.config.num_train_epochs} epochs and {self.config.num_update_steps_per_epoch} steps per epoch."
            )
        if self.config.max_train_steps is None or self.config.max_train_steps == 0:
            if target_epochs == 0:
                raise ValueError("You must specify either --max_train_steps or --num_train_epochs with a value > 0")
            self.config.max_train_steps = int(math.ceil(target_epochs * steps_per_epoch))
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
        self.sidecar_optimizer = None
        self.sidecar_lr = None
        self.sidecar_is_schedulefree = False
        self.sidecar_scheduler_disabled = False

        def _normalize_lr(value):
            if value in (None, "", "None"):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                logger.warning("Could not parse learning rate override %r; ignoring.", value)
                return None

        extra_optimizer_args = {"lr": self.config.learning_rate}
        embedder_params: list[torch.nn.Parameter] = []
        embedder_param_ids: set[int] = set()
        embedder_lr_value = _normalize_lr(getattr(self.config, "lyrics_embedder_lr", None))
        text_encoder_lr_value = _normalize_lr(getattr(self.config, "text_encoder_lr", None))
        if getattr(self.config, "lyrics_embedder_train", False):
            embedder_param_fn = getattr(self.model, "get_lyrics_embedder_parameters", None)
            if callable(embedder_param_fn):
                embedder_params = embedder_param_fn(require_grad_only=True, unwrap=False)
                embedder_param_ids = {id(p) for p in embedder_params}
            else:
                logger.warning("lyrics_embedder_train enabled but model does not expose embedder parameters.")

        lyrics_optimizer_override = getattr(self.config, "lyrics_embedder_optimizer", None) not in (
            None,
            "",
            "None",
        )
        lyrics_scheduler_override = getattr(self.config, "lyrics_embedder_lr_scheduler", None) not in (
            None,
            "",
            "None",
        )
        if lyrics_scheduler_override and not lyrics_optimizer_override:
            logger.info(
                "A lyrics embedder LR scheduler was provided without a custom optimizer; a separate embedder optimizer "
                "will reuse the primary optimizer type."
            )
        use_separate_lyrics_opt = lyrics_optimizer_override or lyrics_scheduler_override
        use_separate_lyrics_opt = use_separate_lyrics_opt and bool(embedder_params)

        if use_separate_lyrics_opt and self.config.use_deepspeed_optimizer:
            logger.warning(
                "Separate lyrics embedder optimizers are not supported with DeepSpeed optimizers; falling back to the "
                "primary optimizer."
            )
            use_separate_lyrics_opt = False
        if self.config.use_deepspeed_optimizer and embedder_lr_value is not None:
            logger.warning("lyrics_embedder_lr will be ignored when using DeepSpeed optimizers.")
            embedder_lr_value = None
        if self.config.use_deepspeed_optimizer and text_encoder_lr_value is not None:
            logger.warning("text_encoder_lr will be ignored when using DeepSpeed optimizers.")
            text_encoder_lr_value = None
        text_encoder_params: list[torch.nn.Parameter] = []
        text_encoder_param_ids: set[int] = set()
        if getattr(self.config, "train_text_encoder", False):
            for text_encoder in self.model.text_encoders:
                if text_encoder is None:
                    continue
                if "t5" in str(text_encoder.__class__).lower():
                    logger.warning(f"{text_encoder.__class__} does not support finetuning, skipping model.")
                    continue
                text_encoder_params.extend([p for p in text_encoder.parameters() if p.requires_grad])
            text_encoder_param_ids = {id(p) for p in text_encoder_params}

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

        def _filter_params(exclude_param_ids: set[int], params):
            filtered = []
            for entry in params:
                if isinstance(entry, dict):
                    group_params = entry.get("params", [])
                    if not isinstance(group_params, (list, tuple, set)):
                        group_params = [group_params]
                    retained = [p for p in group_params if id(p) not in exclude_param_ids]
                    if retained:
                        new_group = dict(entry)
                        new_group["params"] = retained
                        filtered.append(new_group)
                else:
                    if id(entry) not in exclude_param_ids:
                        filtered.append(entry)
            return filtered

        if use_separate_lyrics_opt and embedder_param_ids:
            self.params_to_optimize = _filter_params(embedder_param_ids, self.params_to_optimize)

        if text_encoder_lr_value is not None and text_encoder_params:
            self.params_to_optimize = _filter_params(text_encoder_param_ids, self.params_to_optimize)
            if self.params_to_optimize and not isinstance(self.params_to_optimize[0], dict):
                self.params_to_optimize = [{"params": self.params_to_optimize}]
            self.params_to_optimize.append({"params": text_encoder_params, "lr": text_encoder_lr_value})
        elif text_encoder_lr_value is not None and not text_encoder_params:
            logger.warning("text_encoder_lr provided but no trainable text encoder parameters were found.")

        if embedder_lr_value is not None and embedder_params and not use_separate_lyrics_opt:
            self.params_to_optimize = _filter_params(embedder_param_ids, self.params_to_optimize)
            if self.params_to_optimize and not isinstance(self.params_to_optimize[0], dict):
                self.params_to_optimize = [{"params": self.params_to_optimize}]
            self.params_to_optimize.append({"params": embedder_params, "lr": embedder_lr_value})
        # Ensure embedder parameters are still optimized when training is enabled without custom LR/optimizer.
        if embedder_params and not use_separate_lyrics_opt and embedder_lr_value is None:
            current_param_ids = {id(p) for p in _flatten_parameters(self.params_to_optimize)}
            missing = [p for p in embedder_params if id(p) not in current_param_ids]
            if missing:
                if self.params_to_optimize and isinstance(self.params_to_optimize[0], dict):
                    self.params_to_optimize.append({"params": missing})
                else:
                    self.params_to_optimize.extend(missing)

        AttentionBackendController.attach_parameter_sink(self.params_to_optimize)
        logger.info(f"Connecting optimizer to {trainable_parameter_count(self.params_to_optimize)} trainable parameters")

        if optimizer_class is AdamWBF16:
            if getattr(self.config, "weight_dtype", None) != torch.bfloat16:
                logger.warning(
                    "AdamW_BF16 requires bf16 weights. Adjusting weight dtype from %s to torch.bfloat16.",
                    getattr(self.config, "weight_dtype", None),
                )
                self.config.weight_dtype = torch.bfloat16
            self._ensure_parameter_dtype(self.params_to_optimize, torch.bfloat16, optimizer_name="adamw_bf16")

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
            self.optimizer = cpu_offload_optimizer(
                params_to_optimize=self.params_to_optimize,
                optimizer_cls=optimizer_class,
                optimizer_parameters=extra_optimizer_args,
                fused=self.config.fuse_optimizer,
                offload_gradients=self.config.optimizer_offload_gradients,
                offload_mechanism=self.config.optimizer_cpu_offload_method,
            )

        AttentionBackendController.bind_optimizer(self.optimizer)
        if is_optimi_available and self.config.optimizer_release_gradients and "optimi" in self.config.optimizer:
            logger.warning(
                "Marking model for gradient release. This feature is experimental, and may use more VRAM or not work."
            )
            prepare_for_gradient_release(self.model.get_trained_component(), self.optimizer)

        if use_separate_lyrics_opt and embedder_params:
            lyrics_optimizer_name = getattr(self.config, "lyrics_embedder_optimizer", None) or self.config.optimizer
            lyrics_lr = embedder_lr_value if embedder_lr_value is not None else self.config.learning_rate
            lyrics_config = copy.copy(self.config)
            lyrics_config.optimizer = lyrics_optimizer_name
            lyrics_config.learning_rate = lyrics_lr
            lyrics_config.is_schedulefree = is_lr_schedulefree(lyrics_optimizer_name)
            lyrics_optimizer_args, lyrics_optimizer_class = determine_optimizer_class_with_config(
                args=lyrics_config,
                use_deepspeed_optimizer=False,
                is_quantized=self.config.is_quantized,
                enable_adamw_bf16=self.config.enable_adamw_bf16,
            )
            lyrics_extra_args = {"lr": lyrics_lr}
            lyrics_extra_args.update(lyrics_optimizer_args)
            if lyrics_optimizer_class is AdamWBF16 and getattr(self.config, "weight_dtype", None) != torch.bfloat16:
                logger.warning("Casting lyrics embedder parameters to bf16 for AdamWBF16 compatibility.")
                self._ensure_parameter_dtype(embedder_params, torch.bfloat16, optimizer_name="lyrics_embedder_adamw_bf16")
            self.sidecar_optimizer = cpu_offload_optimizer(
                params_to_optimize=embedder_params,
                optimizer_cls=lyrics_optimizer_class,
                optimizer_parameters=lyrics_extra_args,
                fused=self.config.fuse_optimizer,
                offload_gradients=self.config.optimizer_offload_gradients,
                offload_mechanism=self.config.optimizer_cpu_offload_method,
            )
            self.sidecar_is_schedulefree = is_lr_schedulefree(lyrics_optimizer_name)
            logger.info(
                "Configured lyrics embedder optimizer with %s parameters",
                trainable_parameter_count(embedder_params),
            )

    def init_lr_scheduler(self):
        self.sidecar_lr_scheduler = None
        self.sidecar_scheduler_disabled = False
        self.config.is_schedulefree = is_lr_schedulefree(self.config.optimizer)
        self.config.is_lr_scheduler_disabled = (
            is_lr_scheduler_disabled(self.config.optimizer) or self.config.use_deepspeed_scheduler
        )
        if self.config.is_schedulefree:
            logger.info("Using experimental ScheduleFree optimiser..")

        def _init_lyrics_scheduler():
            if self.sidecar_optimizer is None:
                return
            lyrics_scheduler_name = getattr(self.config, "lyrics_embedder_lr_scheduler", None) or getattr(
                self.config, "lr_scheduler", None
            )
            lyrics_optimizer_name = getattr(self.config, "lyrics_embedder_optimizer", None) or getattr(
                self.config, "optimizer", None
            )
            self.sidecar_is_schedulefree = is_lr_schedulefree(lyrics_optimizer_name)
            lyrics_config = copy.copy(self.config)
            lyrics_config.optimizer = lyrics_optimizer_name
            lyrics_config.lr_scheduler = lyrics_scheduler_name
            lyrics_config.is_schedulefree = self.sidecar_is_schedulefree
            self.sidecar_scheduler_disabled = (
                is_lr_scheduler_disabled(lyrics_optimizer_name) or self.config.use_deepspeed_scheduler
            )
            if self.sidecar_is_schedulefree:
                logger.info("Using experimental ScheduleFree optimiser for lyrics embedder..")
            if self.sidecar_scheduler_disabled:
                logger.info("Disabling lyrics embedder LR scheduler.")
                if torch.backends.mps.is_available():
                    self.sidecar_lr_scheduler = None
                else:
                    self.sidecar_lr_scheduler = accelerate.utils.DummyScheduler(
                        self.sidecar_optimizer,
                        total_num_steps=self.config.max_train_steps,
                        warmup_num_steps=self.config.lr_warmup_steps,
                    )
                return
            self.sidecar_lr_scheduler = get_lr_scheduler(
                lyrics_config,
                self.sidecar_optimizer,
                self.accelerator,
                logger,
                global_step=self.state["global_step"],
                use_deepspeed_scheduler=False,
            )
            if hasattr(self.sidecar_lr_scheduler, "num_update_steps_per_epoch"):
                self.sidecar_lr_scheduler.num_update_steps_per_epoch = self.config.num_update_steps_per_epoch
            if hasattr(self.sidecar_lr_scheduler, "last_step"):
                self.sidecar_lr_scheduler.last_step = self.state.get("global_resume_step", 0)

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
            _init_lyrics_scheduler()
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

        _init_lyrics_scheduler()
        return lr_scheduler

    def init_ema_model(self):
        # Create EMA for the model.
        self.ema_model = None
        if not self.config.use_ema:
            return
        # this runs on all processes to ensure shapes are aligned.
        self.model.pre_ema_creation()
        fsdp_version = getattr(self.config, "fsdp_version", 1)
        fsdp_enabled = getattr(self.config, "fsdp_enable", False)
        is_fsdp2_run = fsdp_enabled and fsdp_version == 2 and self.accelerator.distributed_type == DistributedType.FSDP

        should_log = self.accelerator.is_main_process
        instantiate_on_rank = should_log or is_fsdp2_run
        if should_log:
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

        if instantiate_on_rank:
            self.ema_model = EMAModel(
                self.config,
                self.accelerator,
                parameters=self._get_trainable_parameters(),
                model_cls=ema_model_cls,
                model_config=ema_model_config,
                decay=self.config.ema_decay,
                foreach=not self.config.ema_foreach_disable,
            )
            if should_log:
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
        if getattr(self.config, "use_fsdp", False):
            moved_param_count = 0
            for param in primary_model.parameters():
                if param.device.type != "cpu":
                    param.data = param.data.to("cpu")
                    if param.grad is not None:
                        param.grad = None
                    moved_param_count += 1
            if moved_param_count:
                logger.info(
                    "Moved %s parameters back to CPU before Accelerator.prepare for FSDP sharding.", moved_param_count
                )
        try:
            first_param = next(primary_model.parameters())
            logger.info("Primary model param device before Accelerator.prepare: %s", first_param.device)
        except StopIteration:
            logger.warning("Primary model has no parameters when preparing accelerator.")
        self.lr_scheduler = lr_scheduler
        self._finalize_deepspeed_config_auto_values(primary_model)
        prepare_targets = [primary_model]
        prepared_labels = ["primary_model"]
        if lr_scheduler is not None:
            prepare_targets.append(lr_scheduler)
            prepared_labels.append("lr_scheduler")
        prepare_targets.append(self.optimizer)
        prepared_labels.append("optimizer")
        if self.sidecar_optimizer is not None:
            prepare_targets.append(self.sidecar_optimizer)
            prepared_labels.append("sidecar_optimizer")
        if self.sidecar_lr_scheduler is not None:
            prepare_targets.append(self.sidecar_lr_scheduler)
            prepared_labels.append("sidecar_lr_scheduler")
        prepare_targets.append(self.train_dataloaders[0])
        prepared_labels.append("train_dataloader")

        results = self.accelerator.prepare(*prepare_targets)
        for label, prepared in zip(prepared_labels, results):
            if label == "primary_model":
                self.model.set_prepared_model(prepared)
            elif label == "lr_scheduler":
                self.lr_scheduler = prepared
            elif label == "optimizer":
                self.optimizer = prepared
            elif label == "sidecar_optimizer":
                self.sidecar_optimizer = prepared
            elif label == "sidecar_lr_scheduler":
                self.sidecar_lr_scheduler = prepared
            elif label == "train_dataloader":
                self.train_dataloaders = [prepared]
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
        if self.config.keep_vae_loaded or self.config.vae_cache_ondemand or getattr(self.config, "vae_cache_disable", False):
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
        # FSDP validation is now supported - we pass FSDP-wrapped models directly to pipelines
        # which transparently handle all-gather/reshard on each forward pass
        self.evaluation = None
        if self.config.validation_disable:
            return
        if self.config.eval_steps_interval is not None and self.config.eval_steps_interval > 0:
            from simpletuner.helpers.training.validation import Evaluation

            self.evaluation = Evaluation(accelerator=self.accelerator)
        model_evaluator = ModelEvaluator.from_config(args=self.config)
        weight_dtype = getattr(self.config, "weight_dtype", torch.float32)
        use_deepspeed_optimizer = getattr(self.config, "use_deepspeed_optimizer", False)
        self.validation = Validation(
            trainable_parameters=self._get_trainable_parameters,
            accelerator=self.accelerator,
            model=self.model,
            distiller=self.distiller,
            args=self.config,
            validation_prompt_metadata=self.validation_prompt_metadata,
            vae_path=getattr(self.config, "vae_path", None),
            weight_dtype=weight_dtype,
            embed_cache=StateTracker.get_default_text_embed_cache(),
            ema_model=getattr(self, "ema_model", None),
            model_evaluator=model_evaluator,
            is_deepspeed=use_deepspeed_optimizer,
            is_fsdp=self.config.use_fsdp,
            publishing_manager=self.publishing_manager,
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
        if self.config.validation_multigpu != "batch-parallel" and not self.accelerator.is_main_process:
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
        AttentionBackendController.on_load_checkpoint(checkpoint_dir)
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
            # use getattr with default False to handle optimizers without split_groups attribute
            split_groups = getattr(self.optimizer.optimizer, "split_groups", False)
            for group in self.optimizer.param_groups if split_groups else self.optimizer.param_groups[:1]:
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
            if hasattr(public_args, "publishing_config"):
                delattr(public_args, "publishing_config")
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
        AttentionBackendController.apply(self.config, AttentionPhase.EVAL)

    def disable_sageattention_inference(self):
        AttentionBackendController.apply(self.config, AttentionPhase.TRAIN)

    def enable_sageattention(self):
        AttentionBackendController.apply(self.config, AttentionPhase.TRAIN)

    def disable_sageattention(self):
        AttentionBackendController.restore_default()

    def move_models(self, destination: str = "accelerator"):
        is_accelerator_target = destination == "accelerator"
        fsdp_active = bool(getattr(self.config, "use_fsdp", False))
        fsdp_version = None
        if self.accelerator and hasattr(self.accelerator, "state"):
            fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
            if fsdp_plugin is not None:
                fsdp_active = True
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", None)

        if is_accelerator_target and fsdp_active:
            version_note = f" (v{fsdp_version})" if fsdp_version is not None else ""
            logger.info(
                "FSDP%s detected; skipping manual model move so Accelerator.prepare can shard from the host device.",
                version_note,
            )
            return

        if is_accelerator_target:
            if not self.accelerator:
                raise RuntimeError(
                    "Accelerator is not initialised but move_models(destination='accelerator') was requested."
                )
            target_device = self.accelerator.device
        else:
            target_device = destination

        group_offload_requested = bool(getattr(self.config, "enable_group_offload", False))
        group_offload_configured = getattr(self.model, "group_offload_configured", False)
        logger.info(
            f"Moving the {str(self.model.get_trained_component().__class__)} to {target_device} in {self.config.weight_dtype if not self.config.is_quantized else self.config.base_model_precision} precision."
        )
        if self.model.get_trained_component() is not None:
            should_move_trained_component = not (
                fsdp_active and group_offload_requested and group_offload_configured and is_accelerator_target
            )
            if should_move_trained_component:
                if self.config.is_quantized:
                    self.model.get_trained_component(unwrap_model=False).to(target_device)
                else:
                    self.model.get_trained_component(unwrap_model=False).to(target_device, dtype=self.config.weight_dtype)
        if getattr(self.accelerator, "_lycoris_wrapped_network", None) is not None:
            self.accelerator._lycoris_wrapped_network = self.accelerator._lycoris_wrapped_network.to(
                target_device, dtype=self.config.weight_dtype
            )

        if group_offload_requested and is_accelerator_target:
            self.model.configure_group_offload()

        AttentionBackendController.apply(self.config, AttentionPhase.TRAIN)
        if self.config.attention_mechanism == "xformers":
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
        if getattr(self, "sidecar_is_schedulefree", False) and hasattr(getattr(self, "sidecar_optimizer", None), "train"):
            logger.debug("Setting sidecar optimiser into train() mode.")
            self.sidecar_optimizer.train()

    def mark_optimizer_eval(self):
        if is_lr_schedulefree(self.config.optimizer) and hasattr(self.optimizer, "eval"):
            # we typically have to call eval() on the optim for schedulefree before saving or running validations.
            logger.debug("Setting optimiser into eval() mode.")
            self.optimizer.eval()
        if getattr(self, "sidecar_is_schedulefree", False) and hasattr(getattr(self, "sidecar_optimizer", None), "eval"):
            logger.debug("Setting sidecar optimiser into eval() mode.")
            self.sidecar_optimizer.eval()

    def _checkpoint_step_interval(self) -> int | None:
        raw_value = getattr(self.config, "checkpoint_step_interval", None)
        if raw_value in (None, "", "None", 0):
            raw_value = getattr(self.config, "checkpointing_steps", None)
        try:
            interval = int(raw_value)
        except (TypeError, ValueError):
            return None
        return interval if interval > 0 else None

    def _checkpoint_epoch_interval(self) -> int | None:
        raw_value = getattr(self.config, "checkpoint_epoch_interval", None)
        if raw_value in (None, "", "None"):
            return None
        try:
            interval = int(raw_value)
        except (TypeError, ValueError):
            return None
        return interval if interval > 0 else None

    def _should_checkpoint_epoch(self, epoch: int) -> bool:
        interval = self._checkpoint_epoch_interval()
        if interval is None or epoch is None:
            return False
        return epoch > 0 and epoch % interval == 0

    def _run_standard_checkpoint(self, webhook_message: str | None, parent_loss, epoch: int, *, upload_to_hub: bool = False):
        if webhook_message:
            self._send_webhook_msg(
                message=f"Checkpoint: `{webhook_message}`",
                message_level="info",
            )
        # Also send structured progress update at checkpoint time
        current_state = self.state.copy()
        current_state.pop("args", None)
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
            self.checkpoint_state_cleanup(
                self.config.output_dir,
                self.config.checkpoints_total_limit,
            )

        save_path = None
        if self.accelerator.is_main_process or self.config.use_deepspeed_optimizer or self.config.fsdp_enable:
            save_path = self.checkpoint_state_save(self.config.output_dir)

        hub_upload_planned = upload_to_hub and self.hub_manager is not None
        if hub_upload_planned:
            if self.accelerator.is_main_process:
                validation_images = getattr(self.validation, "validation_images") if self.validation is not None else None

                def _upload_latest_checkpoint():
                    remote_path, local_path, repo_url = self.hub_manager.upload_latest_checkpoint(
                        validation_images=validation_images,
                        webhook_handler=self.webhook_handler,
                    )
                    return remote_path, local_path, repo_url

                description = f"checkpoint step {self.state.get('global_step')}"
                try:
                    self._schedule_hub_upload(description, _upload_latest_checkpoint)
                except Exception as e:
                    logger.error(f"Error uploading to hub: {e}, continuing training.")
        else:
            if save_path:
                self._run_post_upload_script(local_path=save_path, remote_path=None)

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

        dataset_type = self._detect_batch_dataset_type(batch)
        if dataset_type is DatasetType.CAPTION:
            return self._prepare_caption_generated_batch(batch)

        prepared_batch = self.model.prepare_batch(batch, state=self.state)

        if getattr(self, "distiller", None) is not None:
            prepared_batch = self.distiller.prepare_batch(prepared_batch, self.model, self.state)

        return prepared_batch

    def _detect_batch_dataset_type(self, batch: dict) -> Optional[DatasetType]:
        if not isinstance(batch, dict):
            return None
        raw_type = batch.get("dataset_type")
        if raw_type is None:
            return None
        try:
            return ensure_dataset_type(raw_type)
        except ValueError:
            return None

    def _prepare_caption_generated_batch(self, batch: dict) -> Dict[str, Any]:
        backend_id = batch.get("data_backend_id", "<caption>")
        distiller = getattr(self, "distiller", None)
        if distiller is None:
            raise ValueError(
                f"Caption dataset '{backend_id}' cannot be used without a distiller that generates training batches."
            )
        if not getattr(distiller, "consumes_caption_batches", lambda: False)():
            raise ValueError(
                f"Distiller {distiller.__class__.__name__} does not support caption-only batches "
                f"(required for backend '{backend_id}')."
            )
        prepared_batch = distiller.prepare_caption_batch(batch, self.model, self.state)
        if not isinstance(prepared_batch, dict):
            raise ValueError(
                f"Distiller {distiller.__class__.__name__} returned an invalid caption batch payload "
                f"for backend '{backend_id}'. Expected a dict."
            )
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

        try:
            model_descriptions = [f"#{idx}:{type(model).__name__}" for idx, model in enumerate(self.accelerator._models)]
            logger.debug(
                "[Rank %s] Accelerator models registered for save_state: %s",
                getattr(self.accelerator, "process_index", "?"),
                ", ".join(model_descriptions) or "<none>",
            )
        except Exception as describe_err:  # pragma: no cover - defensive logging
            logger.warning("Unable to describe registered models before checkpoint: %s", describe_err)

        plugin = getattr(getattr(self.accelerator, "state", None), "fsdp_plugin", None)
        fsdp_v2_run = (
            getattr(self.accelerator, "distributed_type", DistributedType.NO) == DistributedType.FSDP
            and getattr(plugin, "fsdp_version", 1) == 2
        )
        if fsdp_v2_run:
            logger.info("FSDP v2 detected; saving with sharded state dict (_use_dtensor disabled for NCCL compatibility).")
        self.accelerator.save_state(save_path_tmp)
        AttentionBackendController.on_save_checkpoint(
            save_path_tmp,
            is_main_process=getattr(self.accelerator, "is_main_process", True),
        )
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
        return save_path

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
            range(0, int(self.config.max_train_steps)),
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
        num_epochs_to_track = int(math.ceil(self.config.num_train_epochs)) + 1
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

            epoch_checkpoint_pending = False
            last_step_saved_checkpoint = False

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
                checkpoint_saved_this_step = False
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
                    if (
                        self._should_checkpoint_epoch(epoch)
                        and not last_step_saved_checkpoint
                        and self.state["global_step"] > self.state["global_resume_step"]
                    ):
                        epoch_checkpoint_pending = True
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
                with self.accelerator.accumulate(*training_models):
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
                            for param in _flatten_parameters(
                                [self.params_to_optimize, getattr(self.sidecar_optimizer, "param_groups", None)]
                            ):
                                if param.grad is not None:
                                    param.grad.data = param.grad.data.to(torch.float32)

                        self.grad_norm = self._max_grad_value()
                        if (
                            self.accelerator.sync_gradients
                            and self.config.optimizer not in ["optimi-stableadamw", "prodigy"]
                            and self.config.max_grad_norm > 0
                        ):
                            # StableAdamW/Prodigy do not need clipping, similar to Adafactor.
                            if self.config.fsdp_enable:
                                # For FSDP, use the native clip_grad_norm_ method of the FSDP module
                                if self.config.grad_clip_method == "norm":
                                    fsdp_model = self.model.get_trained_component(unwrap_model=False)
                                    self.grad_norm = fsdp_model.clip_grad_norm_(self.config.max_grad_norm)
                                elif self.config.grad_clip_method == "value":
                                    logger.warning(
                                        "FSDP does not support grad_clip_method='value'. Skipping gradient clipping."
                                    )
                                else:
                                    raise ValueError(
                                        f"Unknown grad clip method: {self.config.grad_clip_method}. Supported methods: value, norm"
                                    )
                            elif self.config.grad_clip_method == "norm":
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
                            if self.sidecar_optimizer is not None:
                                self.sidecar_optimizer.optimizer_accumulation = should_not_release_gradients
                        else:
                            self.optimizer.step()
                            if self.sidecar_optimizer is not None:
                                self.sidecar_optimizer.step()
                        self.optimizer.zero_grad(set_to_none=self.config.set_grads_to_none)
                        if self.sidecar_optimizer is not None:
                            self.sidecar_optimizer.zero_grad(set_to_none=self.config.set_grads_to_none)
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
                    try:
                        if self.sidecar_optimizer is not None:
                            if self.sidecar_lr_scheduler is not None and not self.sidecar_scheduler_disabled:
                                sidecar_step_kwargs = {}
                                sidecar_sched_name = getattr(self.config, "lyrics_embedder_lr_scheduler", None) or getattr(
                                    self.config, "lr_scheduler", None
                                )
                                if sidecar_sched_name == getattr(self.config, "lr_scheduler", None):
                                    sidecar_step_kwargs = dict(self.extra_lr_scheduler_kwargs)
                                self.sidecar_lr_scheduler.step(**sidecar_step_kwargs)
                                self.sidecar_lr = self.sidecar_lr_scheduler.get_last_lr()[0]
                            else:
                                self.sidecar_lr = self.sidecar_optimizer.param_groups[0].get("lr")
                    except Exception as e:
                        logger.error(f"Failed to update lyrics embedder learning rate: {e}")
                    wandb_logs.update(
                        {
                            "train_loss": self.train_loss,
                            "optimization_loss": loss,
                            "learning_rate": self.lr,
                            "epoch": epoch,
                        }
                    )
                    if self.sidecar_lr is not None:
                        wandb_logs["lyrics_embedder_learning_rate"] = self.sidecar_lr
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
                    if training_luminance_values:
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

                    checkpoint_step_interval = self._checkpoint_step_interval()
                    upload_to_hub = (
                        self.hub_manager is not None
                        and step % self.config.gradient_accumulation_steps == 0
                        and self.state["global_step"] > self.state["global_resume_step"]
                    )
                    manual_checkpoint_requested = self._consume_manual_checkpoint_request()
                    scheduled_checkpoint_due = (
                        checkpoint_step_interval and self.state["global_step"] % checkpoint_step_interval == 0
                    )
                    if manual_checkpoint_requested or scheduled_checkpoint_due:
                        checkpoint_message = (
                            f"Manual checkpoint requested. {webhook_pending_msg}"
                            if manual_checkpoint_requested
                            else webhook_pending_msg
                        )
                        self._run_standard_checkpoint(
                            webhook_message=checkpoint_message,
                            parent_loss=parent_loss,
                            epoch=epoch,
                            upload_to_hub=upload_to_hub,
                        )
                        checkpoint_saved_this_step = True
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
                        and self.config.accelerator_cache_clear_interval > 0
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

                    try:
                        self.accelerator.log(
                            wandb_logs,
                            step=self.state["global_step"],
                        )
                    except Exception as e:
                        logger.error(f"Failed to log to accelerator; ignoring error: {e}")

                    # Reset some values for the next go.
                    training_luminance_values = []
                    self.train_loss = 0.0
                    last_step_saved_checkpoint = checkpoint_saved_this_step

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
                    manual_validation_requested = self._consume_manual_validation_request()
                    should_validate = manual_validation_requested or self.validation.would_validate()
                    if should_validate:
                        self.mark_optimizer_eval()
                        AttentionBackendController.apply(self.config, AttentionPhase.EVAL)
                        self.disable_gradient_checkpointing()

                    try:
                        self.validation.run_validations(
                            validation_type="intermediary",
                            step=step,
                            force_evaluation=manual_validation_requested,
                        )
                    except Exception as error:
                        # let's not crash training because of a validation error.
                        logger.error(f"Validation run failed at step {step}: {error}", exc_info=True)

                    if should_validate:
                        AttentionBackendController.apply(self.config, AttentionPhase.TRAIN)
                        self.enable_gradient_checkpointing()
                        self.mark_optimizer_train()
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
            if epoch_checkpoint_pending:
                epoch_message = f"Epoch {epoch} completed at step {self.state['global_step']}"
                epoch_upload_to_hub = self.hub_manager is not None and (
                    self.state["global_step"] > self.state["global_resume_step"]
                )
                self._run_standard_checkpoint(
                    webhook_message=epoch_message,
                    parent_loss=parent_loss,
                    epoch=epoch,
                    upload_to_hub=epoch_upload_to_hub,
                )

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
                AttentionBackendController.apply(self.config, AttentionPhase.EVAL)
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
                AttentionBackendController.apply(self.config, AttentionPhase.TRAIN)
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

                def _upload_final_model():
                    repo_url = self.hub_manager.upload_model(validation_images, self.webhook_handler)
                    return repo_url, self.config.output_dir, repo_url

                try:
                    self._schedule_hub_upload("final model upload", _upload_final_model)
                except Exception as e:
                    logger.error(f"Error uploading final model to hub: {e}")
                self._finish_hub_uploads()
            else:
                self._run_post_upload_script(local_path=self.config.output_dir, remote_path=None)
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
    manual_validation_consumer = None
    manual_checkpoint_consumer = None

    if hasattr(config, "__dict__"):
        attrs = vars(config).copy()
        should_abort_callable = attrs.pop("should_abort", None)
        manual_validation_consumer = attrs.pop("consume_manual_validation_request", None)
        manual_checkpoint_consumer = attrs.pop("consume_manual_checkpoint_request", None)
        job_id = attrs.pop("__job_id__", None) or attrs.pop("job_id", None)
        config_payload = attrs
    elif isinstance(config, dict):
        config_payload = dict(config)
        job_id = config_payload.pop("__job_id__", None) or config_payload.pop("job_id", None)
        should_abort_callable = config_payload.pop("should_abort", None)
        manual_validation_consumer = config_payload.pop("consume_manual_validation_request", None)
        manual_checkpoint_consumer = config_payload.pop("consume_manual_checkpoint_request", None)
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
        import json
        import logging
        import os
        import shlex
        import subprocess
        import threading
        import time
        from pathlib import Path

        import simpletuner.helpers.log_format  # noqa: F401

        launch_logger = logging.getLogger("SimpleTuner")
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

        # Force colors to be enabled in subprocess (stdout is piped so TTY detection fails)
        # Remove SIMPLETUNER_WEB_MODE so subprocess can use colors even when launched from web UI
        launch_env.pop("SIMPLETUNER_WEB_MODE", None)
        launch_env.pop("SIMPLETUNER_DISABLE_COLORS", None)
        launch_env["FORCE_COLOR"] = "1"
        launch_env["CLICOLOR_FORCE"] = "1"

        if job_id:
            launch_env["SIMPLETUNER_JOB_ID"] = job_id
        if hf_token:
            for var in ("HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HF_API_TOKEN"):
                launch_env.setdefault(var, hf_token)

        if selected_device_ids:
            launch_env["CUDA_VISIBLE_DEVICES"] = ",".join(selected_device_ids)

        config_backend_value: Optional[str] = None
        config_env_value: Optional[str] = None
        if isinstance(config_payload, dict):
            config_backend_value = _extract_value(
                config_payload,
                "config_backend",
                "--config_backend",
                "CONFIG_BACKEND",
                "SIMPLETUNER_CONFIG_BACKEND",
            )
            config_env_value = _extract_value(
                config_payload,
                "config_environment",
                "--config_environment",
                "config_env",
                "--config_env",
                "environment",
                "--environment",
                "env",
                "--env",
            )
        if config_backend_value:
            backend_text = str(config_backend_value).strip()
            if backend_text:
                launch_env["SIMPLETUNER_CONFIG_BACKEND"] = backend_text
                launch_env["CONFIG_BACKEND"] = backend_text
        if config_env_value:
            env_text = str(config_env_value).strip()
            if env_text:
                launch_env.setdefault("SIMPLETUNER_ENVIRONMENT", env_text)
                launch_env.setdefault("SIMPLETUNER_ENV", env_text)
                launch_env.setdefault("ENV", env_text)

        # Normalise accelerate config path if provided
        config_file_arg = None
        if isinstance(accelerate_config_path, str) and accelerate_config_path.strip():
            expanded_path = os.path.expanduser(accelerate_config_path.strip())
            accelerate_config_path = expanded_path
            if os.path.isfile(expanded_path):
                config_file_arg = expanded_path
            else:
                launch_logger.warning(
                    "Accelerate config file not found at %s; falling back to CLI parameters", expanded_path
                )

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
                launch_logger.warning("Failed to parse accelerate extra args; using raw string")
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
            for config_key in (
                "--webhook_config",
                "webhook_config",
                "--publishing_config",
                "publishing_config",
            ):
                webhook_value = train_cli_payload.get(config_key)
                if isinstance(webhook_value, (dict, list)):
                    try:
                        train_cli_payload[config_key] = json.dumps(webhook_value)
                    except Exception:
                        train_cli_payload.pop(config_key, None)
            from simpletuner.helpers.configuration.cli_utils import mapping_to_cli_args

            cli_args = mapping_to_cli_args(train_cli_payload)
        if cli_args:
            launch_env.setdefault("CONFIG_BACKEND", "cmd")
            cmd.extend(cli_args)

        launch_logger.info("Launching training via accelerate: %s", " ".join(cmd))

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
        from collections import deque as _deque

        recent_lines = _deque(maxlen=400)

        def _forward_output():
            if not process.stdout:
                return
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                recent_lines.append(line.rstrip("\n"))
                with output_lock:
                    try:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    except Exception:
                        launch_logger.info(line.rstrip())

        reader_thread = threading.Thread(target=_forward_output, daemon=True)
        reader_thread.start()

        try:
            while process.poll() is None:
                if callable(should_abort_callable) and should_abort_callable():
                    launch_logger.info("Abort requested; terminating accelerate launcher")
                    _terminate_accelerate_process(process)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        launch_logger.warning("Accelerate process unresponsive; forcing kill")
                        _kill_accelerate_process(process)
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            launch_logger.info("Keyboard interrupt received; terminating accelerate launcher")
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
                    launch_logger.warning("Accelerate process still alive; forcing kill")
                    _kill_accelerate_process(process)

        returncode = process.wait()
        if returncode != 0:
            helper = globals().get("_summarize_accelerate_failure")
            if helper is None:

                def helper(exit_code, lines):
                    cleaned = [line.rstrip("\n") for line in lines]
                    excerpt_lines = [line.strip() for line in cleaned[-10:] if line.strip()]
                    excerpt_text = "\n".join(excerpt_lines) if excerpt_lines else None
                    summary_text = f"Accelerate launch exited with status {exit_code}"
                    if excerpt_lines:
                        summary_text = f"{summary_text}: {excerpt_lines[-1]}"
                    return summary_text[:512], excerpt_text

            summary, excerpt = helper(returncode, list(recent_lines))
            launch_error = RuntimeError(summary)
            if excerpt:
                setattr(launch_error, "_simpletuner_log_excerpt", excerpt)
            if recent_lines:
                setattr(launch_error, "_simpletuner_recent_log_lines", list(recent_lines))
            raise launch_error
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
            excerpt = getattr(exc, "_simpletuner_log_excerpt", None)
            if excerpt:
                payload["log_excerpt"] = excerpt
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
        if callable(manual_validation_consumer):
            trainer.register_manual_validation_trigger(manual_validation_consumer)
        if callable(manual_checkpoint_consumer):
            trainer.register_manual_checkpoint_trigger(manual_checkpoint_consumer)

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
        import time

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
        logging.getLogger("SimpleTuner").warning("Failed to terminate accelerate process cleanly: %s", exc)


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
        logging.getLogger("SimpleTuner").warning("Failed to kill accelerate process: %s", exc)
