"""Core training service utilities extracted from the HTMX routes."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from simpletuner.helpers.training.deepspeed_optimizers import DEFAULT_OPTIMIZER as DS_DEFAULT_OPTIMIZER
from simpletuner.helpers.training.deepspeed_optimizers import sanitize_optimizer_mapping
from simpletuner.helpers.training.trainer import run_trainer_job
from simpletuner.simpletuner_sdk import process_keeper
from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.simpletuner_sdk.server.dependencies.common import _load_active_config_cached
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService
from simpletuner.simpletuner_sdk.server.services.field_registry.types import FieldType, ValidationRuleType
from simpletuner.simpletuner_sdk.server.services.field_registry_wrapper import lazy_field_registry
from simpletuner.simpletuner_sdk.server.services.hardware_service import detect_gpu_inventory
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIDefaults, WebUIStateStore
from simpletuner.simpletuner_sdk.server.utils.paths import resolve_config_path

from .webhook_defaults import DEFAULT_CALLBACK_URL, DEFAULT_WEBHOOK_CONFIG

logger = logging.getLogger(__name__)


def _get_job_store():
    """Lazily import and return the AsyncJobStore singleton to avoid circular imports."""
    from .cloud.container import get_job_store

    return get_job_store()


def _get_unified_job_class():
    """Lazily import UnifiedJob to avoid circular imports."""
    from .cloud.base import UnifiedJob

    return UnifiedJob


def _get_cloud_job_status():
    """Lazily import CloudJobStatus to avoid circular imports."""
    from .cloud.base import CloudJobStatus

    return CloudJobStatus


def _detect_local_hardware() -> str:
    """Detect local GPU hardware for job metadata."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Silicon (MPS)"
    except Exception:
        pass
    return "CPU"


_PROMPT_LIBRARY_RUNTIME_ROOT = Path(tempfile.gettempdir()) / "simpletuner_prompt_libraries"


def _ensure_prompt_library_runtime_dir(job_id: str) -> Path:
    """Return a clean runtime directory for prompt libraries for a given job."""

    _PROMPT_LIBRARY_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    job_dir = _PROMPT_LIBRARY_RUNTIME_ROOT / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def _normalise_prompt_library_path(value: Any) -> Optional[str]:
    """Return a cleaned string path for the prompt library CLI argument."""

    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed and trimmed.lower() not in {"none", "null", "false"}:
            return trimmed
    return None


def _prepare_user_prompt_library(
    runtime_payload: Dict[str, Any],
    *,
    job_id: str,
    configs_dir: Optional[str],
) -> None:
    """Copy or materialise the user prompt library into a job-scoped location."""

    inline_library = runtime_payload.get("user_prompt_library")
    library_is_inline = isinstance(inline_library, dict)

    cli_path = _normalise_prompt_library_path(runtime_payload.get("--user_prompt_library"))
    alias_path = None
    if isinstance(inline_library, str):
        alias_path = _normalise_prompt_library_path(inline_library)

    if not library_is_inline and not cli_path and not alias_path:
        # No prompt library configured
        return

    if library_is_inline:
        source_path = None
    else:
        candidate = cli_path or alias_path
        resolved = resolve_config_path(candidate, config_dir=configs_dir) if candidate else None
        if resolved is None or not resolved.exists():
            raise FileNotFoundError(f"User prompt library not found at '{candidate}'. Provide a valid JSON file.")
        source_path = resolved

    job_dir = _ensure_prompt_library_runtime_dir(job_id)
    if library_is_inline:
        target_path = job_dir / "user_prompt_library.json"
        with target_path.open("w", encoding="utf-8") as handle:
            json.dump(inline_library, handle, indent=4)
    else:
        target_path = job_dir / source_path.name
        shutil.copy2(source_path, target_path)

    runtime_payload["--user_prompt_library"] = str(target_path)
    runtime_payload["user_prompt_library"] = str(target_path)


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

    raw_interval_input = form_dict.get("accelerator_cache_clear_interval")
    if raw_interval_input is None:
        raw_interval_input = form_dict.get("--accelerator_cache_clear_interval")

    disabled_fields_raw = form_dict.pop("__disabled_fields__", None)
    disabled_arg_names = set()
    if disabled_fields_raw:
        if isinstance(disabled_fields_raw, (list, tuple)):
            raw_items = disabled_fields_raw
        else:
            raw_items = str(disabled_fields_raw).split(",")
        for item in raw_items:
            stripped = str(item).strip()
            if stripped:
                disabled_arg_names.add(stripped if stripped.startswith("--") else f"--{stripped}")

    def _is_required_field(arg_name: str) -> bool:
        lookup = arg_name.lstrip("-")
        field = lazy_field_registry.get_field(lookup)
        if not field:
            return False
        return any(rule.rule_type == ValidationRuleType.REQUIRED for rule in field.validation_rules)

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

    replace_config_raw = _coerce_single(form_dict.pop("__replace_active_config__", None))
    replace_config = False
    if replace_config_raw is not None:
        replace_config = str(replace_config_raw).strip().lower() == "true"

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

    save_options["preserve_defaults"] = getattr(webui_defaults, "auto_preserve_defaults", True)

    if "preserve_defaults" in form_dict:
        raw_preserve = _coerce_single(form_dict.pop("preserve_defaults"))
        save_options["preserve_defaults"] = ConfigsService._is_truthy(raw_preserve)
    if "create_backup" in form_dict:
        raw_backup = _coerce_single(form_dict.pop("create_backup"))
        save_options["create_backup"] = ConfigsService._is_truthy(raw_backup)
    save_options["merge_environment_defaults"] = merge_environment_defaults

    directory_fields = ["--output_dir", "--instance_data_dir"]

    config_dict = ConfigsService.normalize_form_to_config(
        form_dict,
        directory_fields,
        output_root=resolved_output_dir,
        configs_dir=resolved_configs_dir,
    )

    # Filter out webui_only fields from config_dict to prevent them from being merged into trainer configs
    # This is the primary defense against WebUI-only fields entering the trainer configuration
    webui_only_keys_to_remove = []
    for key in list(config_dict.keys()):
        # Check both with and without "--" prefix
        lookup_key = key.lstrip("-")
        field = lazy_field_registry.get_field(lookup_key)
        if field and getattr(field, "webui_only", False):
            webui_only_keys_to_remove.append(key)
            logger.debug(f"Filtering webui_only field from config_dict: {key}")

    # Also remove non-registry WebUI-only fields
    non_registry_webui_fields = {
        "datasets_dir",
        "--datasets_dir",
        "allow_dataset_paths_outside_dir",
        "--allow_dataset_paths_outside_dir",
        "uploadMode",
        "--uploadMode",
        "ui-accelerate-mode",
        "--ui-accelerate-mode",
    }
    for key in config_dict.keys():
        if key in non_registry_webui_fields:
            webui_only_keys_to_remove.append(key)
            logger.debug(f"Filtering non-registry webui_only field from config_dict: {key}")

    for key in webui_only_keys_to_remove:
        config_dict.pop(key, None)

    # Normalize legacy form keys that may still be emitted by older UI states
    legacy_key_map = {
        "maximum_caption_length": "--tokenizer_max_length",
        "--maximum_caption_length": "--tokenizer_max_length",
        "project_name": "--tracker_project_name",
        "--project_name": "--tracker_project_name",
        "__active_tab__": None,
        "--__active_tab__": None,
    }

    for legacy_key, target_key in legacy_key_map.items():
        if legacy_key not in config_dict:
            continue

        legacy_value = config_dict.pop(legacy_key)
        if target_key is None:
            continue

        # Normalize sentinel strings like "None" back to actual None
        if isinstance(legacy_value, str) and legacy_value.strip().lower() in {"", "none", "null"}:
            legacy_value = None

        if legacy_value is None:
            continue

        if target_key not in config_dict:
            config_dict[target_key] = legacy_value

    logger.debug("Prepared config_dict: %s", config_dict)

    interval_cleared = False
    if raw_interval_input is not None:
        interval_cleared = str(raw_interval_input).strip() == ""

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

    # Normalize existing config values using the same type coercion applied to complete_config.
    # This ensures comparisons for preserve_defaults work correctly (comparing apples-to-apples).
    existing_config_cli = ConfigsService.coerce_config_values_by_field(existing_config_cli)

    fallback_existing = sanitize_optimizer_mapping(existing_config_cli)
    if fallback_existing:
        logger.warning(
            "Unsupported DeepSpeed optimizer '%s' detected in active config; replacing with '%s'.",
            fallback_existing,
            DS_DEFAULT_OPTIMIZER,
        )
    fallback_form = sanitize_optimizer_mapping(config_dict)
    if fallback_form:
        logger.warning(
            "Unsupported DeepSpeed optimizer '%s' detected in submitted config; replacing with '%s'.",
            fallback_form,
            DS_DEFAULT_OPTIMIZER,
        )

    def _extract_form_value(arg_name: str) -> Any:
        """Return the raw value submitted for a given argument name (with or without --)."""
        candidates = [arg_name]
        trimmed = arg_name.lstrip("-")
        if arg_name.startswith("--"):
            candidates.append(trimmed)
        else:
            candidates.append(f"--{trimmed}")
        for candidate in candidates:
            if candidate in form_dict:
                raw_value = form_dict[candidate]
                if isinstance(raw_value, (list, tuple)):
                    non_empty = [item for item in raw_value if item not in (None, "")]
                    if not non_empty:
                        return ""
                    return non_empty[-1]
                return raw_value
        return None

    def _field_was_cleared(arg_name: str) -> bool:
        """Determine whether the user explicitly cleared a field in the submitted form."""
        raw_value = _extract_form_value(arg_name)
        if raw_value is None:
            return False
        if isinstance(raw_value, str):
            return raw_value.strip() == ""
        return raw_value in (None, [], {})

    cleared_text_fields: Set[str] = set()

    text_field_types = {FieldType.TEXT, FieldType.TEXTAREA, FieldType.FILE, FieldType.PASSWORD}
    blankable_field_args: Set[str] = set()
    allow_empty_field_args: Set[str] = set()
    for registry_field in lazy_field_registry.get_all_fields():
        field_type = getattr(registry_field, "field_type", None)
        if field_type not in text_field_types:
            continue
        default_value = getattr(registry_field, "default_value", None)
        allow_empty = getattr(registry_field, "allow_empty", False)
        arg_name = getattr(registry_field, "arg_name", None) or getattr(registry_field, "name", None)
        if not arg_name:
            continue
        canonical_arg = arg_name if arg_name.startswith("--") else f"--{arg_name}"

        # Fields with allow_empty should preserve empty strings, not be removed
        if allow_empty:
            allow_empty_field_args.add(canonical_arg)
        # Only blankable if no default value OR if default is empty string
        elif default_value is None or (isinstance(default_value, str) and default_value.strip() == ""):
            blankable_field_args.add(canonical_arg)

    # These fields are removed completely when cleared (legacy behavior)
    manual_blankable_fields = {
        # validation_negative_prompt removed from here - now uses allow_empty=True
    }
    blankable_field_args.update(manual_blankable_fields)

    def _register_cleared_field(arg_name: str) -> None:
        canonical = arg_name if arg_name.startswith("--") else f"--{arg_name.lstrip('-')}"
        alias = canonical.lstrip("-")
        cleared_text_fields.add(canonical)
        variants = {canonical, alias}
        for key in variants:
            existing_config_cli.pop(key, None)
            config_dict.pop(key, None)

    if _field_was_cleared("--deepspeed_config"):
        logger.debug("User cleared --deepspeed_config; removing from existing config merge.")
        for variant in ("--deepspeed_config", "deepspeed_config"):
            existing_config_cli.pop(variant, None)
            config_dict.pop(variant, None)
            alias = variant.lstrip("-")
            if alias != variant:
                existing_config_cli.pop(alias, None)
                config_dict.pop(alias, None)

    if _field_was_cleared("--user_prompt_library"):
        logger.debug("User cleared --user_prompt_library; removing from existing config merge.")
        cleared_keys = {"--user_prompt_library", "user_prompt_library"}
        cleared_keys.update({key.lstrip("-") for key in cleared_keys})
        for key in cleared_keys:
            existing_config_cli.pop(key, None)
            config_dict.pop(key, None)

    # Handle fields with allow_empty - preserve as empty string, don't remove
    for canonical_field in allow_empty_field_args:
        if _field_was_cleared(canonical_field):
            logger.debug("User cleared %s (allow_empty field); preserving as empty string.", canonical_field)
            # Remove from existing config to avoid merge, but keep empty string in config_dict
            canonical = canonical_field if canonical_field.startswith("--") else f"--{canonical_field.lstrip('-')}"
            alias = canonical.lstrip("-")
            variants = {canonical, alias}
            for key in variants:
                existing_config_cli.pop(key, None)
            # Ensure empty string is in config_dict
            if canonical not in config_dict:
                config_dict[canonical] = ""

    # Handle regular blankable fields - remove completely
    for canonical_field in blankable_field_args:
        if _field_was_cleared(canonical_field):
            logger.debug("User cleared %s; removing from existing config merge.", canonical_field)
            _register_cleared_field(canonical_field)

    all_defaults = get_all_field_defaults()

    def _has_accelerate_config(*sources: Dict[str, Any]) -> bool:
        for source in sources:
            if not isinstance(source, dict):
                continue
            for candidate in ("--accelerate_config", "accelerate_config"):
                value = source.get(candidate)
                if isinstance(value, str) and value.strip():
                    return True
        return False

    accelerate_config_present = _has_accelerate_config(existing_config_cli, config_dict)

    onboarding_accelerate_raw = getattr(webui_defaults, "accelerate_overrides", None)
    onboarding_accelerate = onboarding_accelerate_raw if isinstance(onboarding_accelerate_raw, dict) else {}
    accelerate_mode = onboarding_accelerate.get("mode") if isinstance(onboarding_accelerate, dict) else None
    manual_count_value = onboarding_accelerate.get("manual_count") if isinstance(onboarding_accelerate, dict) else None

    def _coerce_device_ids(value: Any) -> List[int]:
        if not isinstance(value, (list, tuple, set)):
            return []
        result: List[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result

    accelerate_device_ids = (
        _coerce_device_ids(onboarding_accelerate.get("device_ids")) if isinstance(onboarding_accelerate, dict) else []
    )
    cleaned_onboarding: Dict[str, Any] = {}
    meta_keys = {"mode", "device_ids", "manual_count"}
    for override_key, override_value in onboarding_accelerate.items():
        if override_value in (None, ""):
            continue
        if isinstance(override_key, str) and override_key.strip() in meta_keys:
            continue
        cli_key = override_key if str(override_key).startswith("--") else f"--{str(override_key).lstrip('-')}"
        cleaned_onboarding[cli_key] = override_value

    accelerate_visible_devices: Optional[List[int]] = None

    if onboarding_accelerate and not accelerate_config_present:
        process_count: Optional[int] = cleaned_onboarding.get("--num_processes")

        if not isinstance(process_count, int) or process_count <= 0:
            process_count = None

        normalized_mode = str(accelerate_mode).strip().lower() if accelerate_mode else None
        if normalized_mode not in {"auto", "manual", "disabled", "hardware"}:
            normalized_mode = None

        write_process_count = True
        if normalized_mode == "auto":
            gpu_inventory = detect_gpu_inventory()
            process_count = gpu_inventory.get("optimal_processes") or gpu_inventory.get("count") or 1
            accelerate_visible_devices = None
        elif normalized_mode == "manual":
            if accelerate_device_ids:
                accelerate_visible_devices = accelerate_device_ids
                process_count = max(len(accelerate_device_ids), 1)
            elif isinstance(manual_count_value, int) and manual_count_value > 0:
                process_count = manual_count_value
            elif isinstance(process_count, int) and process_count > 0:
                process_count = process_count
            else:
                process_count = 1
        elif normalized_mode == "disabled":
            process_count = 1
            accelerate_visible_devices = []
        elif normalized_mode == "hardware":
            cleaned_onboarding.pop("--num_processes", None)
            write_process_count = False
        else:
            # Fallback to previously stored value or default to auto-detect count if available
            if not isinstance(process_count, int) or process_count <= 0:
                gpu_inventory = detect_gpu_inventory()
                process_count = gpu_inventory.get("optimal_processes") or 1
            if process_count <= 1:
                normalized_mode = "disabled"
            else:
                normalized_mode = "manual"

        if normalized_mode in {"auto", "manual", "disabled"}:
            for key in ("--num_processes", "num_processes"):
                existing_config_cli.pop(key, None)
                config_dict.pop(key, None)

        if write_process_count:
            cleaned_onboarding["--num_processes"] = max(int(process_count or 1), 1)
        else:
            cleaned_onboarding.pop("--num_processes", None)
        accelerate_mode = normalized_mode

        all_defaults.update({k: v for k, v in cleaned_onboarding.items() if k.startswith("--")})

    if merge_environment_defaults and not replace_config:
        base_config = {**all_defaults, **existing_config_cli}
    else:
        if not replace_config:
            logger.debug("Skipping merge of active environment defaults at user request")
        else:
            logger.debug("Replacing active config - ignoring existing file content")
        base_config = dict(all_defaults)

    if cleared_text_fields:
        for canonical in cleared_text_fields:
            alias = canonical.lstrip("-")
            base_config.pop(canonical, None)
            base_config.pop(alias, None)

    selected_family = (
        config_dict.get("--model_family")
        or config_dict.get("model_family")
        or (existing_config_cli.get("--model_family") if not replace_config else None)
        or (existing_config_cli.get("model_family") if not replace_config else None)
        or base_config.get("--model_family")
        or base_config.get("model_family")
    )

    if selected_family:
        selected_family = str(selected_family)
        for field_obj in lazy_field_registry.get_all_fields():
            model_specific = getattr(field_obj, "model_specific", None)
            if not model_specific or selected_family in model_specific:
                continue

            for key in filter(None, {field_obj.arg_name, field_obj.name}):
                alias = key.lstrip("-")
                base_config.pop(key, None)
                base_config.pop(alias, None)
                if not replace_config:
                    existing_config_cli.pop(key, None)
                    existing_config_cli.pop(alias, None)
                config_dict.pop(key, None)
                config_dict.pop(alias, None)

    if disabled_arg_names:
        for arg_name in list(disabled_arg_names):
            if _is_required_field(arg_name):
                continue
            alias = arg_name.lstrip("-")
            # Only drop values coming from the current form submission; keep
            # existing/base values so required fields remain populated.
            config_dict.pop(arg_name, None)
            config_dict.pop(alias, None)

    if interval_cleared or config_dict.get("--accelerator_cache_clear_interval") is None:
        config_dict.pop("--accelerator_cache_clear_interval", None)
        if not replace_config:
            existing_config_cli.pop("--accelerator_cache_clear_interval", None)
            existing_config_cli.pop("accelerator_cache_clear_interval", None)
        base_config.pop("--accelerator_cache_clear_interval", None)
        base_config.pop("accelerator_cache_clear_interval", None)

    if replace_config:
        complete_config = {**base_config, **config_dict}
    else:
        complete_config = {**base_config, **existing_config_cli, **config_dict}

    fallback_complete = sanitize_optimizer_mapping(complete_config)
    if fallback_complete:
        logger.warning(
            "Unsupported DeepSpeed optimizer '%s' detected during config merge; replacing with '%s'.",
            fallback_complete,
            DS_DEFAULT_OPTIMIZER,
        )

    if not accelerate_config_present:
        if accelerate_visible_devices:
            device_list = [int(device_id) for device_id in accelerate_visible_devices]
            complete_config["accelerate_visible_devices"] = device_list
            config_dict["accelerate_visible_devices"] = device_list
        if accelerate_mode:
            complete_config["accelerate_strategy"] = accelerate_mode
            config_dict["accelerate_strategy"] = accelerate_mode

    # Ensure required core fields survive modal toggles even when not explicitly submitted
    def _get_with_alias(source: Dict[str, Any], arg: str) -> Any:
        if not isinstance(source, dict):
            return None
        return source.get(arg) or source.get(arg.lstrip("-"))

    required_args = ["--model_family", "--optimizer", "--data_backend_config", "--output_dir", "--model_type"]
    for arg_name in required_args:
        alias = arg_name.lstrip("-")
        explicit_present = isinstance(config_dict, dict) and (arg_name in config_dict or alias in config_dict)
        value = complete_config.get(arg_name)
        if value in (None, ""):
            if explicit_present:
                explicit_value = config_dict.get(arg_name)
                if explicit_value in (None, ""):
                    explicit_value = config_dict.get(alias)
                complete_config[arg_name] = explicit_value
                continue
            fallback = (
                _get_with_alias(config_dict, arg_name)
                or _get_with_alias(existing_config_cli, arg_name)
                or _get_with_alias(base_config, arg_name)
            )
            if fallback not in (None, ""):
                complete_config[arg_name] = fallback
                config_dict.setdefault(arg_name, fallback)

    complete_config = ConfigsService.coerce_config_values_by_field(complete_config)

    # Drop optional string fields that were explicitly cleared
    for optional_key in ("--cache_dir_vae", "cache_dir_vae"):
        if complete_config.get(optional_key) in ("", None):
            complete_config.pop(optional_key, None)
        config_dict.pop(optional_key, None)

    if interval_cleared or complete_config.get("--accelerator_cache_clear_interval") is None:
        complete_config.pop("--accelerator_cache_clear_interval", None)

    if disabled_arg_names:
        for arg_name in disabled_arg_names:
            if _is_required_field(arg_name):
                continue
            alias = arg_name.lstrip("-")
            complete_config.pop(arg_name, None)
            complete_config.pop(alias, None)

    # Remove WebUI-only fields from config
    from simpletuner.simpletuner_sdk.server.services.field_service import FieldService

    for ui_field in FieldService._WEBUI_ONLY_FIELDS:
        complete_config.pop(ui_field, None)
        complete_config.pop(f"--{ui_field}", None)

    # Parse user webhook_config for both runtime merge and persistence
    user_webhooks_raw = config_dict.get("--webhook_config") or config_dict.get("webhook_config") or []
    if isinstance(user_webhooks_raw, str):
        try:
            user_webhooks_raw = json.loads(user_webhooks_raw)
        except (json.JSONDecodeError, TypeError):
            user_webhooks_raw = []
    if not isinstance(user_webhooks_raw, list):
        user_webhooks_raw = [user_webhooks_raw] if user_webhooks_raw else []
    # Keep original user webhooks for persistence (without WebUI callback)
    user_webhooks_for_save = copy.deepcopy(user_webhooks_raw)

    # Merge with WebUI callback for runtime only
    merged_webhooks = copy.deepcopy(DEFAULT_WEBHOOK_CONFIG) + user_webhooks_raw
    complete_config["--webhook_config"] = merged_webhooks
    config_dict["--webhook_reporting_interval"] = 1

    save_config: Dict[str, Any] = {}
    non_persistent_keys = {
        "accelerate_visible_devices",
        "accelerate_strategy",
        "--accelerate_visible_devices",
        "--accelerate_strategy",
        "webhook_config",  # Handled separately to avoid saving merged WebUI callback
        "--webhook_config",
    }

    # Build set of valid field names from registry for validation
    valid_field_names = set()
    for registry_field in lazy_field_registry.get_all_fields():
        valid_field_names.add(registry_field.name)
        if registry_field.arg_name:
            valid_field_names.add(registry_field.arg_name)
            valid_field_names.add(registry_field.arg_name.lstrip("-"))

    for key, value in complete_config.items():
        if key in non_persistent_keys:
            continue
        if value is None:
            continue
        clean_key = key[2:] if key.startswith("--") else key

        # Only save fields that are actually registered in the field registry
        # This prevents UI-only fields from being saved to config files
        if clean_key not in valid_field_names:
            logger.debug(f"Skipping unregistered field '{clean_key}' from config save")
            continue

        arg_lookup = key if key.startswith("--") else f"--{clean_key}"
        is_required_field = _is_required_field(arg_lookup)

        # explicit_override means the user explicitly CHANGED this field in the current form submission.
        # The frontend sends all fields (via appendConfigValuesToFormData), including unchanged ones,
        # so we need to compare the normalized value with the saved config value to detect actual changes.
        # Only fields that were actually modified should be saved even if they match defaults.
        #
        # IMPORTANT: Use `value` from complete_config (which is normalized) for comparisons, not raw form_dict,
        # to ensure we're comparing apples-to-apples after type coercion.
        explicit_override = False
        if clean_key in form_dict or arg_lookup in form_dict:
            # Field is in form submission - check if it was actually changed
            # Use normalized value from complete_config for comparison
            saved_value = None
            if not replace_config:
                saved_value = existing_config_cli.get(arg_lookup) or existing_config_cli.get(key)

            if saved_value is None:
                # Field wasn't in saved config - check if it differs from default
                # Only treat as explicit if the user set it to a non-default value
                default_value = all_defaults.get(arg_lookup, all_defaults.get(key))
                if value != default_value:
                    explicit_override = True
            elif value != saved_value:
                # Field was in saved config and user changed it
                explicit_override = True

        if save_options.get("preserve_defaults", False) and not is_required_field:
            default_value = all_defaults.get(arg_lookup, all_defaults.get(key))
            if value != default_value or explicit_override:
                save_config[clean_key] = value
        else:
            save_config[clean_key] = value

    scientific_pattern = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE]-?\d+)$")
    for arg_key, clean_key in (("--learning_rate", "learning_rate"), ("--lr_end", "lr_end")):
        raw_value = form_dict.get(arg_key)
        if raw_value is None:
            raw_value = form_dict.get(arg_key.lstrip("-"))
        if isinstance(raw_value, str) and scientific_pattern.fullmatch(raw_value.strip()):
            if clean_key in save_config and isinstance(save_config[clean_key], (int, float)):
                save_config[clean_key] = raw_value.strip()

    # Add user's original webhook_config (without WebUI callback) if any were configured
    if user_webhooks_for_save:
        save_config["webhook_config"] = user_webhooks_for_save

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

    literal_scientific_keys = {
        key: value
        for key, value in bundle.save_config.items()
        if key in {"learning_rate", "lr_end"}
        and isinstance(value, str)
        and re.fullmatch(r"^-?\d+(?:\.\d+)?(?:[eE]-?\d+)$", value)
    }

    if literal_scientific_keys:
        config_path = store._get_config_path(bundle.active_config)
        try:
            contents = config_path.read_text(encoding="utf-8")
        except Exception:
            contents = ""

        if contents:
            for key, literal in literal_scientific_keys.items():
                pattern = re.compile(rf'("{re.escape(key)}"\s*:\s*)"{re.escape(literal)}"')

                def _replace(match, value=literal):
                    return f"{match.group(1)}{value}"

                contents = pattern.sub(_replace, contents)

            config_path.write_text(contents, encoding="utf-8")

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

    def _read_required(key: str) -> Any:
        alias = key.lstrip("-")
        for candidate in (source, complete_config):
            if not isinstance(candidate, dict):
                continue
            if key in candidate:
                return candidate.get(key)
            if alias in candidate:
                return candidate.get(alias)
        return None

    required_text_fields = {
        "--output_dir": "Output directory is required.",
        "--tracker_project_name": "Project name is required.",
    }
    for _field, message in required_text_fields.items():
        value = _read_required(_field)
        if isinstance(value, str):
            value = value.strip()
        if not value:
            errors.append(message)

    inline_prompt_library = _read_required("user_prompt_library")
    if not isinstance(inline_prompt_library, dict):
        prompt_library_path = _read_required("--user_prompt_library")
        if isinstance(prompt_library_path, str):
            candidate = prompt_library_path.strip()
            if candidate and candidate.lower() not in {"none", "null", "false"}:
                resolved = resolve_config_path(
                    candidate,
                    config_dir=getattr(store, "config_dir", None),
                    check_cwd_first=True,
                )
                if resolved is None or not resolved.exists():
                    errors.append(
                        f"User prompt library not found at '{candidate}'. " "Please provide a valid JSON file path."
                    )

    return TrainingValidationResult(
        errors=errors,
        warnings=warnings,
        suggestions=suggestions,
        raw_validation=validation,
    )


def start_training_job(runtime_config: Dict[str, Any]) -> str:
    """Submit a training job via the process keeper and return the job identifier."""

    job_id = str(uuid.uuid4())[:8]

    runtime_payload = dict(runtime_config)

    # Merge user webhook_config with WebUI callback
    user_webhooks = runtime_payload.get("--webhook_config") or runtime_payload.get("webhook_config") or []
    if isinstance(user_webhooks, str):
        try:
            user_webhooks = json.loads(user_webhooks)
        except (json.JSONDecodeError, TypeError):
            user_webhooks = []
    if not isinstance(user_webhooks, list):
        user_webhooks = [user_webhooks] if user_webhooks else []

    merged_webhooks = copy.deepcopy(DEFAULT_WEBHOOK_CONFIG) + user_webhooks
    runtime_payload["--webhook_config"] = merged_webhooks

    # Resolve the prompt library into a job-scoped path if one was configured.
    try:
        _, defaults = get_webui_state()
    except Exception:
        defaults = WebUIDefaults()
    configs_dir = getattr(defaults, "configs_dir", None)
    try:
        _prepare_user_prompt_library(runtime_payload, job_id=job_id, configs_dir=configs_dir)
    except FileNotFoundError:
        # Ensure API state does not retain a stale job identifier when the
        # prompt library is missing. This mirrors the expectations in the
        # training service unit tests and prevents state leakage between tests.
        APIState.set_state("training_config", None)
        APIState.set_state("training_status", None)
        APIState.set_state("current_job_id", None)
        APIState.set_state("training_progress", None)
        APIState.set_state("training_startup_stages", {})
        raise

    APIState.set_state("training_config", runtime_payload)
    APIState.set_state("training_status", "starting")
    APIState.set_state("training_progress", None)
    APIState.set_state("training_startup_stages", {})

    job_config = dict(runtime_payload)
    job_config["__job_id__"] = job_id
    # Ensure the trainer surfaces configuration parsing errors instead of silently
    # falling back to config/config.json when launched from the WebUI.
    job_config["__skip_config_fallback__"] = True

    # Remove non-CLI arguments that shouldn't be passed to the trainer
    non_cli_keys = {
        "accelerate_strategy",
        "--accelerate_strategy",
        "ui-accelerate-mode",
        "--ui-accelerate-mode",
    }
    for key in non_cli_keys:
        job_config.pop(key, None)

    process_keeper.submit_job(job_id, run_trainer_job, job_config)

    # Track local job in unified JobStore for Job Queue visibility
    try:
        UnifiedJob = _get_unified_job_class()
        CloudJobStatus = _get_cloud_job_status()
        job_store = _get_job_store()

        config_name = (
            runtime_config.get("--model_alias")
            or runtime_config.get("model_alias")
            or runtime_config.get("--tracker_run_name")
            or runtime_config.get("tracker_run_name")
            or "local"
        )
        output_url = runtime_config.get("--output_dir") or runtime_config.get("output_dir")

        local_job = UnifiedJob.create_local(
            job_id=job_id,
            config_name=config_name,
            hardware_type=_detect_local_hardware(),
        )
        local_job.output_url = output_url
        local_job.status = CloudJobStatus.RUNNING.value
        local_job.started_at = datetime.now(timezone.utc).isoformat()

        # Store job asynchronously (fire-and-forget)
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(job_store.add_job(local_job))
        except RuntimeError:
            # No running event loop, use sync approach via thread
            import threading

            def _add_job():
                import asyncio as aio

                aio.run(job_store.add_job(local_job))

            threading.Thread(target=_add_job, daemon=True).start()
    except Exception as exc:
        logger.warning("Failed to track local job in JobStore: %s", exc)

    APIState.set_state("current_job_id", job_id)
    return job_id


def terminate_training_job(job_id: Optional[str], *, status: str, clear_job_id: bool) -> bool:
    """Terminate an active training job and update API state."""

    if not job_id:
        return False

    terminated = process_keeper.terminate_process(job_id)
    if terminated:
        APIState.set_state("training_status", status)
        if clear_job_id:
            APIState.set_state("current_job_id", None)

        # Update job status in JobStore
        try:
            CloudJobStatus = _get_cloud_job_status()
            job_store = _get_job_store()

            # Map termination status to CloudJobStatus
            if status in {"stopped", "cancelled"}:
                new_status = CloudJobStatus.CANCELLED.value
            elif status in {"failed", "error"}:
                new_status = CloudJobStatus.FAILED.value
            else:
                new_status = CloudJobStatus.CANCELLED.value

            updates = {
                "status": new_status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(job_store.update_job(job_id, updates))
            except RuntimeError:
                import threading

                def _update_job():
                    import asyncio as aio

                    aio.run(job_store.update_job(job_id, updates))

                threading.Thread(target=_update_job, daemon=True).start()
        except Exception as exc:
            logger.warning("Failed to update job status in JobStore: %s", exc)

    return terminated


def request_manual_validation(job_id: Optional[str] = None) -> str:
    """Signal the running trainer to execute validation after the next gradient sync."""

    active_job = job_id or APIState.get_state("current_job_id")
    if not active_job:
        raise RuntimeError("No active training job to validate.")
    process_keeper.send_process_command(active_job, "trigger_validation", None)
    return active_job


def request_manual_checkpoint(job_id: Optional[str] = None) -> str:
    """Signal the running trainer to persist a checkpoint after the next gradient sync."""

    active_job = job_id or APIState.get_state("current_job_id")
    if not active_job:
        raise RuntimeError("No active training job to checkpoint.")
    process_keeper.send_process_command(active_job, "trigger_checkpoint", None)
    return active_job
