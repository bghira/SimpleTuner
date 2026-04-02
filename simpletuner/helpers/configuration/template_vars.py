import json
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from simpletuner.helpers.training.state_tracker import StateTracker

_PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")


def resolve_string_placeholders(value: str, *, variables: Mapping[str, object] | None = None) -> str:
    """Resolve supported placeholders in a string value."""

    if not isinstance(value, str):
        return value

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token.startswith("env:"):
            return os.environ.get(token[4:], "")
        if variables is not None and token in variables:
            resolved = variables[token]
            return "" if resolved is None else str(resolved)
        return match.group(0)

    return _PLACEHOLDER_PATTERN.sub(_replace, value)


def resolve_value_placeholders(value: Any, *, variables: Mapping[str, object] | None = None) -> Any:
    """Recursively resolve placeholders in strings nested inside lists and dicts."""

    if isinstance(value, str):
        return resolve_string_placeholders(value, variables=variables)
    if isinstance(value, list):
        return [resolve_value_placeholders(item, variables=variables) for item in value]
    if isinstance(value, tuple):
        return tuple(resolve_value_placeholders(item, variables=variables) for item in value)
    if isinstance(value, dict):
        return {key: resolve_value_placeholders(item, variables=variables) for key, item in value.items()}
    return value


def render_modelspec_comment(value: Any, *, variables: Mapping[str, object] | None = None) -> str | None:
    """Normalize and render a modelspec comment value."""

    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    value = parsed
            except (json.JSONDecodeError, ValueError):
                pass

    if isinstance(value, list):
        value = "\n".join(str(item) for item in value)
    elif not isinstance(value, str):
        value = str(value)

    value = value.strip()
    if not value:
        return None

    value = resolve_string_placeholders(value, variables=variables)
    return value or None


def build_modelspec_template_values(now: datetime | None = None) -> dict[str, str]:
    """Build dynamic template values for modelspec metadata expansion."""

    timestamp = now or datetime.now(tz=timezone.utc)
    return {
        "current_step": str(StateTracker.get_global_step()),
        "current_epoch": str(StateTracker.get_epoch()),
        "timestamp": timestamp.isoformat(),
    }
