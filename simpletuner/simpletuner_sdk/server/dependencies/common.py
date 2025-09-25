"""Common dependencies for FastAPI routes.

This module provides reusable dependency functions that can be injected
into route handlers using FastAPI's Depends mechanism.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from fastapi.requests import Request

from ..services.config_store import ConfigStore
from ..services.webui_state import WebUIStateStore
from ..services.field_registry_wrapper import lazy_field_registry
from ..services.cache_service import cache_response

logger = logging.getLogger(__name__)


# Cache for config data and webui defaults
@cache_response(ttl_seconds=60)
def _load_active_config_cached() -> Dict[str, Any]:
    """Load active configuration with caching."""
    config_store = ConfigStore()
    active_config = config_store.get_active_config()

    if active_config is None:
        return {}

    # Load config file
    config_path = Path(config_store.configs_dir) / active_config / "config.json"
    if not config_path.exists():
        logger.warning(f"Active config file not found: {config_path}")
        return {}

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


async def get_config_data() -> Dict[str, Any]:
    """FastAPI dependency to get active configuration data.

    Returns:
        Dict containing the active configuration data
    """
    return _load_active_config_cached()


async def get_webui_defaults() -> Dict[str, Any]:
    """FastAPI dependency to get WebUI default settings.

    Returns:
        Dict with configs_dir and output_dir defaults
    """
    webui_defaults = {}
    try:
        state_store = WebUIStateStore()
        defaults = state_store.load_defaults()
        webui_defaults = {
            "configs_dir": defaults.configs_dir or "Not configured",
            "output_dir": defaults.output_dir or "Not configured",
        }
    except Exception as e:
        logger.error(f"Error loading WebUI defaults: {e}")
        webui_defaults = {"configs_dir": "Not configured", "output_dir": "Not configured"}

    return webui_defaults


async def get_field_registry():
    """FastAPI dependency to get the field registry.

    Returns:
        The lazy field registry instance

    Raises:
        HTTPException: If field registry is not available
    """
    if lazy_field_registry._registry is None:
        lazy_field_registry._get_registry()

    if lazy_field_registry._registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Field registry is not available. Please check server logs."
        )

    return lazy_field_registry


def get_config_value(config_data: Dict[str, Any], key: str, default: Any = "") -> Any:
    """Get a configuration value with fallback to default.

    Args:
        config_data: Configuration dictionary
        key: Key to look up
        default: Default value if key not found

    Returns:
        The configuration value or default
    """
    if not config_data:
        return default

    # Support nested keys with dot notation
    keys = key.split(".")
    value = config_data

    try:
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default
    except Exception:
        return default


def convert_field_to_template_format(field: Any, config_values: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a ConfigField to the format expected by templates.

    Args:
        field: ConfigField instance
        config_values: Current configuration values

    Returns:
        Dict with field data formatted for templates
    """
    # Get the field value
    field_value = config_values.get(field.name, field.default_value)

    # Special handling for num_train_epochs and max_train_steps
    if field.name == "num_train_epochs":
        logger.debug(f"num_train_epochs raw value: {field_value}, type: {type(field_value)}, in config: {field.name in config_values}")
        # Convert string "0" to integer 0
        if str(field_value) == "0" or field_value == 0:
            field_value = 0
        elif field_value == 1 and field.name not in config_values:
            # If using default value of 1, start empty to allow spinner to go to 0
            field_value = ""
        logger.debug(f"num_train_epochs final value: {field_value}")
    elif field.name == "max_train_steps":
        logger.debug(f"max_train_steps raw value: {field_value}, type: {type(field_value)}, in config: {field.name in config_values}")
        # Convert string "0" to integer 0
        if str(field_value) == "0" or field_value == 0:
            field_value = 0
        logger.debug(f"max_train_steps final value: {field_value}")

    field_dict = {
        "id": field.name,
        "name": field.arg_name,
        "label": field.ui_label,
        "type": field.field_type.value.lower(),
        "value": field_value,
        "description": field.help_text,
    }

    field_dict["extra_classes"] = ""

    # Add cmd_args help for detailed tooltip
    if hasattr(field, 'cmd_args_help') and field.cmd_args_help:
        field_dict["cmd_args_help"] = field.cmd_args_help

    # Add section ID if present
    if hasattr(field, 'section') and field.section:
        field_dict["section_id"] = field.section

    # Handle conditional display
    if hasattr(field, 'conditional_on'):
        field_dict["conditional_on"] = field.conditional_on
        field_dict["extra_classes"] += " conditional-field"

    # Add min/max for number fields
    if field.field_type.value == "NUMBER":
        if hasattr(field, 'min_value') and field.min_value is not None:
            field_dict["min"] = field.min_value
        if hasattr(field, 'max_value') and field.max_value is not None:
            field_dict["max"] = field.max_value
        if hasattr(field, 'step') and field.step is not None:
            field_dict["step"] = field.step

    # Add options for select/multi-select fields
    if field.field_type.value in ["SELECT", "MULTI_SELECT"]:
        if hasattr(field, 'choices') and field.choices:
            field_dict["options"] = [{"value": choice, "label": choice} for choice in field.choices]

    # Add placeholder for text fields
    if field.field_type.value in ["TEXT", "TEXTAREA"]:
        if hasattr(field, 'placeholder') and field.placeholder:
            field_dict["placeholder"] = field.placeholder

    # Add required flag
    if hasattr(field, 'required'):
        field_dict["required"] = field.required

    # Add disabled flag
    if hasattr(field, 'disabled'):
        field_dict["disabled"] = field.disabled

    return field_dict


class TabFieldsDependency:
    """Dependency class for getting tab fields with template formatting."""

    def __init__(self, tab_name: str):
        self.tab_name = tab_name

    async def __call__(
        self,
        field_registry = Depends(get_field_registry),
        config_data: Dict[str, Any] = Depends(get_config_data),
        webui_defaults: Dict[str, Any] = Depends(get_webui_defaults)
    ) -> List[Dict[str, Any]]:
        """Get fields for a specific tab formatted for templates.

        Returns:
            List of field dictionaries formatted for templates
        """
        # Get fields from registry
        try:
            tab_fields = field_registry.get_fields_for_tab(self.tab_name)
            logger.debug(f"Field registry returned {len(tab_fields)} fields for '{self.tab_name}' tab")
        except Exception as e:
            logger.error(f"Error getting fields for tab '{self.tab_name}': {e}")
            tab_fields = []

        # Build config values dictionary
        config_values = {}
        for field in tab_fields:
            # Special handling for output_dir to use webui defaults
            if field.name == "output_dir":
                config_values[field.name] = get_config_value(
                    config_data,
                    field.name,
                    webui_defaults.get("output_dir", "")
                )
            else:
                config_values[field.name] = get_config_value(
                    config_data,
                    field.name,
                    field.default_value
                )

        # Also add WebUI-specific values
        if self.tab_name == "basic":
            config_values["configs_dir"] = webui_defaults.get("configs_dir", "")
            config_values["job_id"] = get_config_value(config_data, "job_id", "")

        # Convert fields to template format
        return [convert_field_to_template_format(field, config_values) for field in tab_fields]


def get_tab_fields(tab_name: str) -> TabFieldsDependency:
    """Factory function to create tab fields dependency.

    Args:
        tab_name: Name of the tab

    Returns:
        TabFieldsDependency instance
    """
    return TabFieldsDependency(tab_name)


# Request context dependencies
async def get_request_id(request: Request) -> str:
    """Get or generate a request ID for correlation.

    Args:
        request: FastAPI request object

    Returns:
        Request ID string
    """
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        import uuid
        request_id = str(uuid.uuid4())
    return request_id


async def get_htmx_request(request: Request) -> bool:
    """Check if the request is from HTMX.

    Args:
        request: FastAPI request object

    Returns:
        True if HTMX request, False otherwise
    """
    return request.headers.get("HX-Request") == "true"