"""Sanitizers for public config exports and log messages."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PUBLIC_CONFIG_EXCLUDED_KEYS = {
    "publishing_config",
    "webhook_config",
}
SENSITIVE_CONFIG_KEY_PARTS = (
    "secret",
    "password",
    "credential",
    "access_key",
    "api_key",
    "private_key",
)
SENSITIVE_CONFIG_KEYS = {
    "token",
    "access_token",
    "api_token",
    "auth_token",
    "bearer_token",
    "client_secret",
    "hf_token",
    "hub_token",
    "session_token",
    "upload_token",
}


def normalise_config_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").lstrip("_")


def should_skip_public_config_key(key: Any) -> bool:
    normalised_key = normalise_config_key(key)
    if normalised_key in PUBLIC_CONFIG_EXCLUDED_KEYS:
        return True
    if normalised_key in SENSITIVE_CONFIG_KEYS:
        return True
    return any(part in normalised_key for part in SENSITIVE_CONFIG_KEY_PARTS)


def make_public_config_serializable(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return make_public_config_serializable(obj.value)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {
            str(key): make_public_config_serializable(value)
            for key, value in obj.items()
            if not should_skip_public_config_key(key)
        }
    if isinstance(obj, (list, tuple, set)):
        return [make_public_config_serializable(value) for value in obj]
    if hasattr(obj, "__dict__"):
        return make_public_config_serializable(vars(obj))
    return str(obj)


def _sanitize_cli_value_for_logging(value: str) -> str:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return value
    return json.dumps(make_public_config_serializable(payload), sort_keys=True)


def sanitize_cli_args_for_public_logging(args: Sequence[str]) -> list[str]:
    sanitized: list[str] = []
    index = 0
    while index < len(args):
        arg = str(args[index])
        if not arg.startswith("--") or arg == "--":
            sanitized.append(arg)
            index += 1
            continue

        option, separator, value = arg.partition("=")
        key = option.lstrip("-")
        if should_skip_public_config_key(key):
            index += 1
            if not separator and index < len(args) and not str(args[index]).startswith("--"):
                index += 1
            continue

        if separator:
            sanitized.append(f"{option}={_sanitize_cli_value_for_logging(value)}")
        else:
            sanitized.append(arg)
        index += 1
    return sanitized
