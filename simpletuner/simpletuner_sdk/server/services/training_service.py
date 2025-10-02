"""Core training service utilities extracted from the HTMX routes."""

from __future__ import annotations

import copy
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from simpletuner.helpers.training.trainer import run_trainer_job
from simpletuner.simpletuner_sdk import process_keeper
from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.simpletuner_sdk.server.dependencies.common import _load_active_config_cached
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService
from simpletuner.simpletuner_sdk.server.services.field_registry_wrapper import lazy_field_registry
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIDefaults, WebUIStateStore

logger = logging.getLogger(__name__)


DEFAULT_CALLBACK_URL = os.environ.get("SIMPLETUNER_WEBHOOK_CALLBACK_URL", "http://localhost:8001/callback")
DEFAULT_WEBHOOK_CONFIG = {
    "webhook_type": "raw",
    "callback_url": DEFAULT_CALLBACK_URL,
}


@dataclass
class TrainingConfigBundle:
    """Container describing the resolved configuration artefacts for a form post."""

    store: ConfigStore
    state_store: Optional[WebUIStateStore]
    webui_defaults: WebUIDefaults
    defaults_changed: bool
    save_options: Dict[str, bool]
    active_config: str
    complete_config: Dict[str, Any]
    config_dict: Dict[str, Any]
    save_config: Dict[str, Any]
    merge_environment_defaults: bool = False


@dataclass
class TrainingValidationResult:
    """Structured validation output for training configuration checks."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    raw_validation: Any = None

    @property
    def is_valid(self) -> bool:
        return not self.errors and getattr(self.raw_validation, "is_valid", True)


def get_config_store() -> ConfigStore:
    """Return a ConfigStore initialised with the user's configured directory where possible."""

    try:
        state_store = WebUIStateStore()
        defaults = state_store.load_defaults()
        if defaults.configs_dir:
            return ConfigStore(config_dir=defaults.configs_dir)
    except Exception:
        logger.debug("Falling back to default ConfigStore", exc_info=True)
    return ConfigStore()


def get_webui_state() -> Tuple[Optional[WebUIStateStore], WebUIDefaults]:
    """Load WebUI state store and defaults with safe fallbacks."""

    try:
        store = WebUIStateStore()
        defaults = store.load_defaults()
        return store, defaults
    except Exception:
        logger.debug("Unable to load WebUI defaults", exc_info=True)
        return None, WebUIDefaults()


def get_all_field_defaults() -> Dict[str, Any]:
    """Return mapping of CLI arg names to default values from the field registry."""

    defaults: Dict[str, Any] = {}
    for registry_field in lazy_field_registry.get_all_fields():
        defaults[registry_field.arg_name] = ConfigsService.convert_value_by_type(
            registry_field.default_value,
            registry_field.field_type,
        )
    return defaults


def build_config_bundle(form_data: Dict[str, Any]) -> TrainingConfigBundle:
    """Process raw form data into a complete training configuration bundle."""

    state_store, webui_defaults = get_webui_state()
    defaults_changed = False
    save_options: Dict[str, bool] = {}

    if hasattr(form_data, "getlist"):
        # Preserve multiple values (e.g. checkbox fallbacks) when provided
        form_dict: Dict[str, Any] = {}
        for key in form_data.keys():
            values = form_data.getlist(key)
            if not values:
                form_dict[key] = None
            elif len(values) == 1:
                form_dict[key] = values[0]
            else:
                form_dict[key] = values
    else:
        form_dict = dict(form_data)

    def _coerce_single(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return value[-1] if value else None
        return value

    merge_defaults_raw = _coerce_single(form_dict.pop("merge_environment_config", None))
    merge_environment_defaults = False
    if merge_defaults_raw is not None:
        merge_environment_defaults = str(merge_defaults_raw).strip().lower() not in {
            "false",
            "0",
            "no",
            "off",
        }

    resolved_defaults = None
    if state_store:
        try:
            resolved_defaults = state_store.resolve_defaults(webui_defaults)
        except Exception:
            resolved_defaults = None
    if not resolved_defaults:
        try:
            temp_store = state_store or WebUIStateStore()
            resolved_defaults = temp_store.resolve_defaults(webui_defaults)
        except Exception:
            resolved_defaults = None

    resolved_configs_dir = webui_defaults.configs_dir
    resolved_output_dir = webui_defaults.output_dir
    if resolved_defaults and isinstance(resolved_defaults, dict):
        resolved_map = resolved_defaults.get("resolved") or {}
        resolved_configs_dir = resolved_map.get("configs_dir", resolved_configs_dir)
        resolved_output_dir = resolved_map.get("output_dir", resolved_output_dir)

    if "configs_dir" in form_dict:
        value = _coerce_single(form_dict.pop("configs_dir"))
        normalized_configs_dir = os.path.abspath(os.path.expanduser(value)) if value else value
        if webui_defaults.configs_dir != normalized_configs_dir:
            webui_defaults.configs_dir = normalized_configs_dir
            defaults_changed = True
        resolved_configs_dir = normalized_configs_dir or resolved_configs_dir

    form_dict.pop("__active_tab__", None)

    if "preserve_defaults" in form_dict:
        save_options["preserve_defaults"] = _coerce_single(form_dict.pop("preserve_defaults")) == "true"
    if "create_backup" in form_dict:
        save_options["create_backup"] = _coerce_single(form_dict.pop("create_backup")) == "true"
    save_options["merge_environment_defaults"] = merge_environment_defaults

    directory_fields = ["--output_dir", "--instance_data_dir"]

    config_dict = ConfigsService.normalize_form_to_config(
        form_dict,
        directory_fields,
        output_root=resolved_output_dir,
        configs_dir=resolved_configs_dir,
    )

    logger.debug("Prepared config_dict: %s", config_dict)

    store = get_config_store()
    active_config = store.get_active_config() or "default"

    existing_config_cli: Dict[str, Any] = {}
    if active_config:
        try:
            existing_config_data, _ = store.load_config(active_config)
            if isinstance(existing_config_data, dict):
                for key, value in existing_config_data.items():
                    cli_key = key if key.startswith("--") else f"--{key}"
                    existing_config_cli[cli_key] = value
        except (FileNotFoundError, ValueError):
            logger.debug("Active config %s could not be loaded", active_config, exc_info=True)

    existing_config_cli = ConfigsService._migrate_legacy_keys(existing_config_cli)

    all_defaults = get_all_field_defaults()
    if merge_environment_defaults:
        base_config = {**all_defaults, **existing_config_cli}
    else:
        logger.debug("Skipping merge of active environment defaults at user request")
        base_config = dict(all_defaults)

    complete_config = {**base_config, **config_dict}
    complete_config = ConfigsService.coerce_config_values_by_field(complete_config)

    for ui_key in ("configs_dir", "--configs_dir", "__active_tab__", "--__active_tab__"):
        complete_config.pop(ui_key, None)

    # Ensure WebUI jobs always use the raw callback webhook configuration
    complete_config["--webhook_config"] = copy.deepcopy(DEFAULT_WEBHOOK_CONFIG)
    config_dict["--webhook_config"] = copy.deepcopy(DEFAULT_WEBHOOK_CONFIG)

    save_config: Dict[str, Any] = {}
    for key, value in complete_config.items():
        clean_key = key[2:] if key.startswith("--") else key
        if save_options.get("preserve_defaults", False):
            default_value = all_defaults.get(key)
            if value != default_value:
                save_config[clean_key] = value
        else:
            save_config[clean_key] = value

    return TrainingConfigBundle(
        store=store,
        state_store=state_store,
        webui_defaults=webui_defaults,
        defaults_changed=defaults_changed,
        save_options=save_options,
        active_config=active_config,
        complete_config=complete_config,
        config_dict=config_dict,
        save_config=save_config,
        merge_environment_defaults=merge_environment_defaults,
    )


def persist_config_bundle(bundle: TrainingConfigBundle) -> None:
    """Persist trainer configuration and related defaults to disk."""

    store = bundle.store
    state_store = bundle.state_store
    webui_defaults = bundle.webui_defaults

    if bundle.defaults_changed and state_store:
        state_store.save_defaults(webui_defaults)

    if bundle.save_options.get("create_backup", False):
        config_path = store._get_config_path(bundle.active_config)
        if config_path.exists():
            import shutil

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = config_path.with_suffix(f".json.backup-{timestamp}")
            shutil.copy2(config_path, backup_path)
            logger.info("Created backup at %s", backup_path)

    store.save_trainer_config(bundle.active_config, bundle.save_config, overwrite=True)

    try:
        _load_active_config_cached.clear_cache()
    except AttributeError:
        pass

    if not store.get_active_config():
        store.set_active_config(bundle.active_config)

    APIState.set_state("training_config", bundle.complete_config)


def _coerce_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError from exc


def validate_training_config(
    store: ConfigStore,
    complete_config: Dict[str, Any],
    config_dict: Optional[Dict[str, Any]] = None,
    *,
    strict_epoch_exclusivity: bool = False,
) -> TrainingValidationResult:
    """Run validation against the supplied configuration."""

    validation = store.validate_config(complete_config)

    errors = list(validation.errors) if validation.errors else []
    warnings = list(validation.warnings) if validation.warnings else []
    suggestions = list(validation.suggestions) if validation.suggestions else []

    source = config_dict or complete_config

    num_epochs = source.get("--num_train_epochs", complete_config.get("--num_train_epochs", 0))
    max_steps = source.get("--max_train_steps", complete_config.get("--max_train_steps", 0))

    try:
        epochs_val = _coerce_int(num_epochs)
        steps_val = _coerce_int(max_steps)

        if epochs_val == 0 and steps_val == 0:
            errors.append("Either num_train_epochs or max_train_steps must be greater than 0. You cannot set both to 0.")

        if strict_epoch_exclusivity and epochs_val > 0 and steps_val > 0:
            errors.append("num_train_epochs and max_train_steps cannot both be set. Set one of them to 0.")
    except ValueError:
        errors.append("Invalid value for num_train_epochs or max_train_steps. Must be numeric.")

    lr_scheduler = source.get("--lr_scheduler", complete_config.get("--lr_scheduler", ""))
    warmup_raw = source.get("--lr_warmup_steps", complete_config.get("--lr_warmup_steps", 0))
    try:
        warmup_val = _coerce_int(warmup_raw)
        if lr_scheduler == "constant" and warmup_val > 0:
            errors.append(
                "Warmup steps are not supported with the 'constant' learning rate scheduler. "
                "Use 'constant_with_warmup' or set warmup steps to 0."
            )
    except ValueError:
        errors.append("Warmup steps must be a whole number.")

    return TrainingValidationResult(
        errors=errors,
        warnings=warnings,
        suggestions=suggestions,
        raw_validation=validation,
    )


def start_training_job(runtime_config: Dict[str, Any]) -> str:
    """Submit a training job via the process keeper and return the job identifier."""

    runtime_payload = dict(runtime_config)
    runtime_payload.setdefault("--webhook_config", copy.deepcopy(DEFAULT_WEBHOOK_CONFIG))

    APIState.set_state("training_config", runtime_payload)
    APIState.set_state("training_status", "starting")

    job_id = str(uuid.uuid4())[:8]

    job_config = dict(runtime_payload)
    job_config["__job_id__"] = job_id

    process_keeper.submit_job(job_id, run_trainer_job, job_config)

    APIState.set_state("current_job_id", job_id)
    return job_id


def terminate_training_job(job_id: Optional[str], *, status: str, clear_job_id: bool) -> bool:
    """Terminate an active training job and update API state."""

    if not job_id:
        return False

    process_keeper.terminate_process(job_id)
    APIState.set_state("training_status", status)
    if clear_job_id:
        APIState.set_state("current_job_id", None)
    return True
