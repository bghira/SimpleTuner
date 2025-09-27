"""Common dependencies for FastAPI routes.

This module provides reusable dependency functions that can be injected
into route handlers using FastAPI's Depends mechanism.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.requests import Request

from ..services.cache_service import cache_response
from ..services.config_store import ConfigStore
from ..services.field_registry_wrapper import lazy_field_registry
from ..services.field_service import FieldFormat, FieldService
from ..services.webui_state import WebUIStateStore

logger = logging.getLogger(__name__)


# Shared service instances
_field_service = FieldService()


# Cache for config data and webui defaults
@cache_response(ttl_seconds=60)
def _load_active_config_cached() -> Dict[str, Any]:
    """Load active configuration with caching."""
    config_store = ConfigStore()
    active_config = config_store.get_active_config()

    if active_config is None:
        return {}

    # Ensure active_config is a string
    if not isinstance(active_config, str):
        logger.error(f"Invalid active_config type: {type(active_config)}, expected string")
        return {}

    # Validate config name doesn't contain path separators
    if "/" in active_config or "\\" in active_config:
        logger.error(f"Invalid config name contains path separator: {active_config}")
        return {}

    # Load config file - active_config is a string (config name)
    config_path = Path(config_store.config_dir) / active_config / "config.json"
    if not config_path.exists():
        logger.warning(f"Active config file not found: {config_path}")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            config_section = data.get("config")
            if isinstance(config_section, dict):
                merged = config_section.copy()
                for key, value in data.items():
                    if key == "config":
                        continue
                    merged.setdefault(key, value)
                return merged

        return data
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
    webui_defaults: Dict[str, str]
    try:
        state_store = WebUIStateStore()
        bundle = state_store.get_defaults_bundle()
        resolved = bundle["resolved"]

        webui_defaults = {
            "configs_dir": resolved.get("configs_dir", "Not configured"),
            "output_dir": resolved.get("output_dir", "Not configured"),
            "theme": resolved.get("theme", "dark"),
            "event_polling_interval": resolved.get("event_polling_interval", 5),
            "event_stream_enabled": resolved.get("event_stream_enabled", True),
        }
    except Exception as e:
        logger.error(f"Error loading WebUI defaults: {e}")
        webui_defaults = {
            "configs_dir": "Not configured",
            "output_dir": "Not configured",
            "theme": "dark",
            "event_polling_interval": 5,
            "event_stream_enabled": True,
        }

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
            detail="Field registry is not available. Please check server logs.",
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


@dataclass
class TabRenderData:
    """Prepared data required to render a trainer tab."""

    fields: List[Dict[str, Any]]
    config_values: Dict[str, Any]
    sections: Optional[List[Dict[str, Any]]]
    raw_config: Dict[str, Any]
    webui_defaults: Dict[str, Any]


async def get_tab_render_data(
    tab_name: str,
    field_registry=Depends(get_field_registry),
    config_data: Dict[str, Any] = Depends(get_config_data),
    webui_defaults: Dict[str, Any] = Depends(get_webui_defaults),
) -> TabRenderData:
    """Prepare template-ready data for a trainer tab."""

    try:
        tab_fields = field_registry.get_fields_for_tab(tab_name)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("Failed to pull fields for tab %s: %s", tab_name, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unable to load fields for tab '{tab_name}'"
        ) from exc

    config_values = _field_service.prepare_tab_field_values(
        tab_name=tab_name,
        config_data=config_data,
        webui_defaults=webui_defaults,
    )

    formatted_fields = _field_service.convert_fields(
        tab_fields,
        FieldFormat.TEMPLATE,
        config_values,
    )

    sections = field_registry.get_sections_for_tab(tab_name)

    return TabRenderData(
        fields=formatted_fields,
        config_values=config_values,
        sections=sections or None,
        raw_config=config_data,
        webui_defaults=webui_defaults,
    )


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


async def get_config_store() -> ConfigStore:
    """FastAPI dependency to get cached ConfigStore instance.

    Returns:
        ConfigStore instance (singleton with caching)
    """
    return ConfigStore()  # Returns singleton instance with caching
