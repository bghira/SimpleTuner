"""Service for handling tab rendering logic.

This service centralizes all tab-related business logic, including
field organization, template rendering, and tab-specific customizations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..services.webui_state import WebUIStateStore
from .custom_section_service import CUSTOM_SECTION_SERVICE
from .dataset_service import build_data_backend_choices

logger = logging.getLogger(__name__)


class TabType(str, Enum):
    """Available tab types in the trainer interface."""

    BASIC = "basic"
    MODEL = "model"
    TRAINING = "training"
    DATASETS = "datasets"
    ENVIRONMENTS = "environments"
    VALIDATION = "validation"
    PUBLISHING = "publishing"
    CHECKPOINTS = "checkpoints"
    UI_SETTINGS = "ui_settings"


@dataclass
class TabConfig:
    """Configuration for a single tab."""

    id: str
    title: str
    icon: str
    template: str
    description: Optional[str] = None
    extra_context_handler: Optional[callable] = None


class TabService:
    """Service for managing tab rendering and field organization."""

    def __init__(self, templates: Jinja2Templates):
        """Initialize tab service.

        Args:
            templates: Jinja2 templates instance
        """
        self.templates = templates
        self._tab_configs = self._initialize_tab_configs()

    def _initialize_tab_configs(self) -> Dict[str, TabConfig]:
        """Initialize tab configurations."""
        return {
            TabType.BASIC: TabConfig(
                id="basic-config",
                title="Basic",
                icon="fas fa-cog",
                template="form_tab.html",
                description="Essential settings to get started",
                extra_context_handler=None,
            ),
            TabType.MODEL: TabConfig(
                id="model-config",
                title="Model",
                icon="fas fa-brain",
                template="form_tab.html",
                description="Model architecture and settings",
                extra_context_handler=None,
            ),
            TabType.TRAINING: TabConfig(
                id="training-config",
                title="Training",
                icon="fas fa-graduation-cap",
                template="form_tab.html",
                description="Training parameters and optimization",
                extra_context_handler=None,
            ),
            TabType.DATASETS: TabConfig(
                id="datasets-config",
                title="Dataset",
                icon="fas fa-database",
                template="datasets_tab.html",
                description="Dataset loading and preprocessing",
                extra_context_handler=self._datasets_tab_context,
            ),
            TabType.ENVIRONMENTS: TabConfig(
                id="environments-config",
                title="Environment",
                icon="fas fa-server",
                template="environments_tab.html",
                description="Environment and compute settings",
                extra_context_handler=self._environments_tab_context,
            ),
            TabType.VALIDATION: TabConfig(
                id="validation-config",
                title="Validation & Output",
                icon="fas fa-check-circle",
                template="form_tab.html",
                description="Configure visual validation jobs and output targets",
                extra_context_handler=self._validation_tab_context,
            ),
            TabType.PUBLISHING: TabConfig(
                id="publishing",
                title="Publishing",
                icon="fas fa-cloud-upload-alt",
                template="form_tab.html",
                description="Configure HuggingFace Hub publishing",
                extra_context_handler=self._publishing_tab_context,
            ),
            TabType.CHECKPOINTS: TabConfig(
                id="checkpoints",
                title="Checkpoints",
                icon="fas fa-save",
                template="checkpoints_tab.html",
                description="Browse and manage training checkpoints",
                extra_context_handler=self._checkpoints_tab_context,
            ),
            TabType.UI_SETTINGS: TabConfig(
                id="ui-settings",
                title="UI Settings",
                icon="fas fa-sliders",
                template="ui_settings_tab.html",
                description="Adjust WebUI preferences and behaviour",
                extra_context_handler=self._ui_settings_tab_context,
            ),
        }

    def get_tab_config(self, tab_name: str) -> TabConfig:
        """Get configuration for a specific tab.

        Args:
            tab_name: Name of the tab

        Returns:
            TabConfig instance

        Raises:
            HTTPException: If tab not found
        """
        try:
            tab_type = TabType(tab_name)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tab '{tab_name}' not found")

        return self._tab_configs.get(tab_type)

    async def render_tab(
        self,
        request: Request,
        tab_name: str,
        fields: List[Dict[str, Any]],
        config_values: Dict[str, Any],
        sections: Optional[List[Dict[str, Any]]] = None,
        raw_config: Optional[Dict[str, Any]] = None,
        webui_defaults: Optional[Dict[str, Any]] = None,
    ) -> HTMLResponse:
        """Render a tab with the provided fields and configuration."""

        tab_config = self.get_tab_config(tab_name)

        def _coerce_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            if isinstance(value, (int, float)):
                return value != 0
            return False

        raw_config = raw_config or {}
        danger_mode_enabled = (
            _coerce_bool(config_values.get("i_know_what_i_am_doing"))
            or _coerce_bool(config_values.get("--i_know_what_i_am_doing"))
            or _coerce_bool(raw_config.get("i_know_what_i_am_doing"))
            or _coerce_bool(raw_config.get("--i_know_what_i_am_doing"))
        )

        context = {
            "request": request,
            "tab_name": tab_name,
            "tab_config": {
                "id": tab_config.id,
                "title": tab_config.title,
                "icon": tab_config.icon,
                "description": tab_config.description,
            },
            "section": {
                "id": tab_config.id,
                "title": tab_config.title,
                "icon": tab_config.icon,
                "description": tab_config.description,
            },
            "fields": fields,
            "config_values": config_values,
            "raw_config": raw_config,
            "webui_defaults": webui_defaults or {},
            "danger_mode_enabled": danger_mode_enabled,
        }

        # Add sections if provided
        if sections:
            context["sections"] = sections

        # Apply tab-specific context modifications
        if tab_config.extra_context_handler:
            context = tab_config.extra_context_handler(context, fields, config_values)

        # Render template
        return self.templates.TemplateResponse(request=context["request"], name=tab_config.template, context=context)

    # Tab-specific context handlers
    def _validation_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ensure validation tab renders its configuration fields."""
        if not context.get("sections"):
            context["sections"] = [
                {
                    "id": context["tab_config"]["id"],
                    "title": context["tab_config"]["title"],
                    "icon": context["tab_config"].get("icon"),
                    "description": context["tab_config"].get("description"),
                }
            ]
        return context

    def _publishing_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for publishing tab using standard form template."""
        field_sections = context.get("sections", [])
        context["sections"] = CUSTOM_SECTION_SERVICE.merge_custom_sections_with_field_sections(
            tab="publishing", field_sections=field_sections
        )
        return context

    def _checkpoints_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide context data for checkpoints tab."""
        context["checkpoints_total_limit"] = config_values.get("checkpoints_total_limit", 10)
        context["output_dir"] = config_values.get("output_dir", "output")
        return context

    def _ui_settings_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide context data for UI settings tab."""
        store = WebUIStateStore()
        bundle = store.get_defaults_bundle()

        context["ui_settings"] = {
            "defaults": bundle["resolved"],
            "raw_defaults": bundle["raw"],
            "fallbacks": bundle["fallbacks"],
            "themes": [
                {"value": "dark", "label": "Dark", "description": "Classic SimpleTuner palette"},
                {"value": "tron", "label": "Tron Prototype", "description": "Experimental neon styling"},
            ],
            "event_interval_options": [3, 5, 10, 15, 30, 60],
        }
        return context

    def _datasets_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for datasets tab."""
        # This tab uses a different template structure
        # Add data backend choices if available
        try:
            context["data_backend_choices"] = build_data_backend_choices()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Could not build data backend choices: %s", exc)

        return context

    def _environments_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for environments tab."""
        # Add environment-specific context
        context["available_accelerators"] = ["cuda", "mps", "cpu"]
        return context

    def get_all_tabs(self) -> List[Dict[str, str]]:
        """Get information about all available tabs.

        Returns:
            List of tab info dictionaries
        """
        return [
            {
                "id": config.id,
                "name": tab_type.value,
                "title": config.title,
                "icon": config.icon,
                "description": config.description,
            }
            for tab_type, config in self._tab_configs.items()
        ]
