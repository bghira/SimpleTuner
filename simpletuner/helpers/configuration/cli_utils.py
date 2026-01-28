"""Helpers for turning config mappings into CLI argument lists."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Callable, Optional

TransformFunc = Optional[Callable[[str, object], object]]


def _ensure_prefixed(key: str) -> str:
    key = key.strip()
    if key.startswith("--"):
        return key
    return f"--{key.lstrip('-')}"


def _format_key_value(key: str, value: object) -> str:
    prefixed = _ensure_prefixed(key)
    return f"{prefixed}={value}" if value is not None else prefixed


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "0", "false", "no", "off", "none"}:
            return False
        if lowered in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _is_falsey(value: object) -> bool:
    return not _is_truthy(value)


_LEGACY_ARG_HANDLERS: dict[str, Callable[[object, list[str], dict[str, object]], bool]] = {}


def normalize_lr_scheduler_value(value: object, warmup_steps: object | None = None) -> object:
    if not isinstance(value, str):
        return value

    normalized = value.strip().lower()
    alias_map = {
        "cosine_with_warmup": "cosine",
    }
    if normalized in alias_map:
        return alias_map[normalized]

    if warmup_steps is not None:
        try:
            warmup_value = int(warmup_steps)
        except (TypeError, ValueError):
            warmup_value = 0
        if normalized == "constant" and warmup_value > 0:
            return "constant_with_warmup"

    return value


def _legacy_handler(name: str):
    def _decorator(func: Callable[[object, list[str], dict[str, object]], bool]):
        _LEGACY_ARG_HANDLERS[name] = func
        return func

    return _decorator


@_legacy_handler("vae_cache_preprocess")
def _handle_legacy_vae_cache_preprocess(value: object, cli_args: list[str], extras: dict[str, object]) -> bool:
    """Translate legacy pre-process flag to current on-demand flag."""

    if _is_falsey(value):
        cli_args.append(_ensure_prefixed("vae_cache_ondemand"))
    return True


@_legacy_handler("save_total_limit")
def _handle_legacy_save_total_limit(value: object, cli_args: list[str], extras: dict[str, object]) -> bool:
    if value in (None, ""):
        return True
    cli_args.append(_format_key_value("checkpoints_total_limit", value))
    return True


@_legacy_handler("aspect_ratio_bucketing")
def _handle_dataset_only_flag(value: object, cli_args: list[str], extras: dict[str, object]) -> bool:
    return True


@_legacy_handler("aspect_ratio_bucket_min")
def _handle_dataset_only_min(value: object, cli_args: list[str], extras: dict[str, object]) -> bool:
    return True


@_legacy_handler("aspect_ratio_bucket_max")
def _handle_dataset_only_max(value: object, cli_args: list[str], extras: dict[str, object]) -> bool:
    return True


@_legacy_handler("repeats")
def _handle_dataset_only_repeats(value: object, cli_args: list[str], extras: dict[str, object]) -> bool:
    return True


@_legacy_handler("ez_model_type")
def _handle_ui_only_ez_model_type(value: object, cli_args: list[str], extras: dict[str, object]) -> bool:
    """Ignore UI-only EZ Mode wizard sentinel field."""
    return True


@_legacy_handler("__disabled_fields__")
def _handle_ui_only_disabled_fields(value: object, cli_args: list[str], extras: dict[str, object]) -> bool:
    """Ignore UI-only disabled fields tracker."""
    return True


@_legacy_handler("__active_tab__")
def _handle_ui_only_active_tab(value: object, cli_args: list[str], extras: dict[str, object]) -> bool:
    """Ignore UI-only active tab tracker."""
    return True


@_legacy_handler("validation_steps")
def _handle_legacy_validation_steps(value: object, cli_args: list[str], extras: dict[str, object]) -> bool:
    """Support legacy validation_steps key by forwarding to validation_step_interval."""

    if value in (None, ""):
        return True

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        value = stripped

    cli_args.append(_format_key_value("validation_step_interval", value))
    return True


@_legacy_handler("checkpointing_steps")
def _handle_legacy_checkpointing_steps(value: object, cli_args: list[str], extras: dict[str, object]) -> bool:
    """Translate legacy checkpointing_steps to checkpoint_step_interval."""

    if value in (None, ""):
        return True

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        value = stripped

    cli_args.append(_format_key_value("checkpoint_step_interval", value))
    return True


def mapping_to_cli_args(
    mapping: Mapping[str, object], *, transform: TransformFunc = None, extras: Optional[dict[str, object]] = None
) -> list[str]:
    """Convert a mapping into CLI style arguments.

    Args:
        mapping: Source configuration values.
        transform: Optional callable to post-process values before formatting.

    Returns:
        List of CLI arguments ready for argparse parsing.
    """

    cli_args: list[str] = []

    extras_dict: dict[str, object] = extras if extras is not None else {}

    # Lazy import to avoid circular dependency
    try:
        from simpletuner.simpletuner_sdk.server.services.field_registry import field_registry
    except ImportError:
        field_registry = None

    for key, raw_value in mapping.items():
        if raw_value is None:
            continue

        # Skip fields that are registered with arg_name=None (dataset-level configs, not CLI args)
        # Also detect TEXT_JSON fields that need JSON serialization
        field = None
        if field_registry is not None:
            canonical_key = key.lstrip("-") if isinstance(key, str) else key
            field = field_registry.get_field(canonical_key)
            if field is not None and getattr(field, "arg_name", "NOT_NONE") is None:
                continue

        value = transform(key, raw_value) if transform else raw_value

        canonical_key = key.lstrip("-") if isinstance(key, str) else key
        handler = _LEGACY_ARG_HANDLERS.get(canonical_key)
        if handler and handler(value, cli_args, extras_dict):
            continue

        if callable(value):
            extras_dict[key] = value
            continue

        if isinstance(value, bool):
            prefixed = _ensure_prefixed(key)
            if value:
                cli_args.append(prefixed)
            else:
                cli_args.append(f"{prefixed}=false")
            continue

        if isinstance(value, str):
            value_lower = value.strip().lower()  # Do once
            if value_lower == "false":
                continue
            if value_lower == "true":
                cli_args.append(_ensure_prefixed(key))
                continue

        # Special handling for webhook_config-like options: always JSON-serialize dicts/lists
        # Also JSON-serialize TEXT_JSON fields from the registry (e.g., modelspec_comment)
        is_text_json_field = False
        if field is not None:
            from simpletuner.simpletuner_sdk.server.services.field_registry.types import FieldType

            is_text_json_field = getattr(field, "field_type", None) == FieldType.TEXT_JSON

        if (
            canonical_key in {"webhook_config", "deepspeed_config", "publishing_config", "peft_lora_target_modules"}
            or is_text_json_field
        ):
            if isinstance(value, (Mapping, list)):
                import json

                try:
                    value = json.dumps(value)
                except Exception:
                    extras_dict[key] = value
                    continue

        if isinstance(value, Mapping):
            extras_dict[key] = value
            continue

        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for item in value:
                item_str = str(item).strip()
                if item_str:
                    cli_args.append(_format_key_value(key, item_str))
            continue

        if not isinstance(value, (str, bytes, int, float)):
            extras_dict[key] = value
            continue

        value_str = str(value).strip()
        if not value_str:
            continue

        cli_args.append(_format_key_value(key, value_str))

    return cli_args
