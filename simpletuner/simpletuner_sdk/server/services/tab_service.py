"""Service for handling tab rendering logic.

This service centralizes all tab-related business logic, including
field organization, template rendering, and tab-specific customizations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from fastapi import HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)


class TabType(str, Enum):
    """Available tab types in the trainer interface."""
    BASIC = "basic"
    MODEL = "model"
    TRAINING = "training"
    ADVANCED = "advanced"
    DATASETS = "datasets"
    ENVIRONMENTS = "environments"
    VALIDATION = "validation"


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
                title="Basic Configuration",
                icon="fas fa-cog",
                template="web/tabs/form_tab.html",
                description="Essential settings to get started",
                extra_context_handler=self._basic_tab_context
            ),
            TabType.MODEL: TabConfig(
                id="model-config",
                title="Model Configuration",
                icon="fas fa-brain",
                template="web/tabs/form_tab.html",
                description="Model architecture and settings",
                extra_context_handler=self._model_tab_context
            ),
            TabType.TRAINING: TabConfig(
                id="training-config",
                title="Training Configuration",
                icon="fas fa-graduation-cap",
                template="web/tabs/form_tab.html",
                description="Training parameters and optimization"
            ),
            TabType.ADVANCED: TabConfig(
                id="advanced-config",
                title="Advanced Configuration",
                icon="fas fa-tools",
                template="web/tabs/form_tab.html",
                description="Advanced training options",
                extra_context_handler=self._advanced_tab_context
            ),
            TabType.DATASETS: TabConfig(
                id="datasets-config",
                title="Dataset Configuration",
                icon="fas fa-database",
                template="web/tabs/datasets_tab.html",
                description="Dataset loading and preprocessing",
                extra_context_handler=self._datasets_tab_context
            ),
            TabType.ENVIRONMENTS: TabConfig(
                id="environments-config",
                title="Environment Configuration",
                icon="fas fa-server",
                template="web/tabs/environments_tab.html",
                description="Environment and compute settings",
                extra_context_handler=self._environments_tab_context
            ),
            TabType.VALIDATION: TabConfig(
                id="validation-status",
                title="Validation Status",
                icon="fas fa-check-circle",
                template="web/tabs/validation_tab.html",
                description="Configuration validation status"
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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tab '{tab_name}' not found"
            )

        return self._tab_configs.get(tab_type)

    async def render_tab(
        self,
        request: Request,
        tab_name: str,
        fields: List[Dict[str, Any]],
        config_values: Dict[str, Any],
        sections: Optional[List[Dict[str, Any]]] = None
    ) -> HTMLResponse:
        """Render a tab with the provided fields and configuration.

        Args:
            request: FastAPI request object
            tab_name: Name of the tab to render
            fields: List of field dictionaries
            config_values: Current configuration values
            sections: Optional list of sections for grouping fields

        Returns:
            HTMLResponse with rendered tab content
        """
        tab_config = self.get_tab_config(tab_name)

        # Base context for all tabs
        context = {
            "request": request,
            "section": {
                "id": tab_config.id,
                "title": tab_config.title,
                "icon": tab_config.icon,
                "description": tab_config.description,
            },
            "fields": fields,
            "config_values": config_values,
        }

        # Add sections if provided
        if sections:
            context["sections"] = sections
            context["grouped_fields"] = self._group_fields_by_section(fields, sections)

        # Apply tab-specific context modifications
        if tab_config.extra_context_handler:
            context = tab_config.extra_context_handler(context, fields, config_values)

        # Render template
        return self.templates.TemplateResponse(tab_config.template, context)

    def _group_fields_by_section(
        self,
        fields: List[Dict[str, Any]],
        sections: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group fields by their section.

        Args:
            fields: List of field dictionaries
            sections: List of section dictionaries

        Returns:
            Dict mapping section IDs to field lists
        """
        grouped = {section["id"]: [] for section in sections}
        grouped["uncategorized"] = []

        for field in fields:
            section_id = field.get("section_id", "uncategorized")
            if section_id in grouped:
                grouped[section_id].append(field)
            else:
                grouped["uncategorized"].append(field)

        # Remove empty sections
        return {k: v for k, v in grouped.items() if v}

    # Tab-specific context handlers
    def _basic_tab_context(
        self,
        context: Dict[str, Any],
        fields: List[Dict[str, Any]],
        config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for basic tab."""
        # Group fields into sections for basic tab
        sections = [
            {"id": "webui", "title": "WebUI Settings", "icon": "fas fa-desktop"},
            {"id": "project", "title": "Project Settings", "icon": "fas fa-project-diagram"},
            {"id": "model", "title": "Model Selection", "icon": "fas fa-cube"},
        ]

        context["sections"] = sections
        context["grouped_fields"] = self._group_basic_fields(fields)

        return context

    def _model_tab_context(
        self,
        context: Dict[str, Any],
        fields: List[Dict[str, Any]],
        config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for model tab."""
        # Handle model_family label formatting
        for field_dict in fields:
            if field_dict["id"] == "model_family" and "options" in field_dict:
                field_dict["options"] = [
                    {"value": opt["value"], "label": self._get_model_family_label(opt["value"])}
                    for opt in field_dict["options"]
                ]
            elif field_dict["id"] == "lora_alpha":
                # Always set lora_alpha to match lora_rank value
                field_dict["value"] = config_values.get("lora_rank", "16")
                field_dict["disabled"] = True

        return context

    def _advanced_tab_context(
        self,
        context: Dict[str, Any],
        fields: List[Dict[str, Any]],
        config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for advanced tab."""
        # Add a warning about advanced settings
        context["warning_message"] = (
            "These are advanced settings. Modifying them incorrectly may impact training performance or stability."
        )
        return context

    def _datasets_tab_context(
        self,
        context: Dict[str, Any],
        fields: List[Dict[str, Any]],
        config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for datasets tab."""
        # This tab uses a different template structure
        # Import dataset-specific logic here
        try:
            from ..routes.fields import _build_data_backend_choices
            # Add data backend choices if needed
            context["data_backend_choices"] = _build_data_backend_choices()
        except ImportError:
            logger.warning("Could not import data backend choices builder")

        return context

    def _environments_tab_context(
        self,
        context: Dict[str, Any],
        fields: List[Dict[str, Any]],
        config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for environments tab."""
        # Add environment-specific context
        context["available_accelerators"] = ["cuda", "mps", "cpu"]
        return context

    def _group_basic_fields(self, fields: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group basic tab fields into sections."""
        grouped = {
            "webui": [],
            "project": [],
            "model": [],
            "other": []
        }

        for field in fields:
            field_id = field.get("id", "")
            if field_id in ["configs_dir", "output_dir"]:
                grouped["webui"].append(field)
            elif field_id in ["job_id", "project_name", "tracker_project_name", "tracker_run_name"]:
                grouped["project"].append(field)
            elif field_id in ["model_type", "pretrained_model_name_or_path", "model_family", "base_model_precision"]:
                grouped["model"].append(field)
            else:
                grouped["other"].append(field)

        # Remove empty groups
        return {k: v for k, v in grouped.items() if v}

    def _get_model_family_label(self, model_key: str) -> str:
        """Generate a human-readable label for a model family key."""
        labels = {
            "sd1x": "Stable Diffusion 1.x",
            "sd2x": "Stable Diffusion 2.x",
            "sd3": "Stable Diffusion 3",
            "deepfloyd": "DeepFloyd IF",
            "sana": "Sana",
            "sdxl": "Stable Diffusion XL",
            "kolors": "Kolors",
            "flux": "Flux",
            "wan": "Wan",
            "ltxvideo": "LTX Video",
            "pixart_sigma": "PixArt-Σ",
            "omnigen": "OmniGen",
            "hidream": "HiDream",
            "auraflow": "AuraFlow",
            "lumina2": "Lumina 2",
            "cosmos2image": "Cosmos2Image",
            "qwen_image": "Qwen Image"
        }
        return labels.get(model_key, model_key.upper())

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
                "description": config.description
            }
            for tab_type, config in self._tab_configs.items()
        ]