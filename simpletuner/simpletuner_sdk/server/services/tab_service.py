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

from .dataset_service import build_data_backend_choices
from ..services.webui_state import WebUIStateStore

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
                title="Basic Configuration",
                icon="fas fa-cog",
                template="form_tab.html",
                description="Essential settings to get started",
                extra_context_handler=self._basic_tab_context
            ),
            TabType.MODEL: TabConfig(
                id="model-config",
                title="Model Configuration",
                icon="fas fa-brain",
                template="form_tab.html",
                description="Model architecture and settings",
                extra_context_handler=self._model_tab_context
            ),
            TabType.TRAINING: TabConfig(
                id="training-config",
                title="Training Configuration",
                icon="fas fa-graduation-cap",
                template="form_tab.html",
                description="Training parameters and optimization"
            ),
            TabType.ADVANCED: TabConfig(
                id="advanced-config",
                title="Advanced Configuration",
                icon="fas fa-tools",
                template="form_tab.html",
                description="Advanced training options",
                extra_context_handler=self._advanced_tab_context
            ),
            TabType.DATASETS: TabConfig(
                id="datasets-config",
                title="Dataset Configuration",
                icon="fas fa-database",
                template="datasets_tab.html",
                description="Dataset loading and preprocessing",
                extra_context_handler=self._datasets_tab_context
            ),
            TabType.ENVIRONMENTS: TabConfig(
                id="environments-config",
                title="Environment Configuration",
                icon="fas fa-server",
                template="environments_tab.html",
                description="Environment and compute settings",
                extra_context_handler=self._environments_tab_context
            ),
            TabType.VALIDATION: TabConfig(
                id="validation-config",
                title="Validation & Output",
                icon="fas fa-check-circle",
                template="form_tab.html",
                description="Configure visual validation jobs and output targets",
                extra_context_handler=self._validation_tab_context
            ),
            TabType.UI_SETTINGS: TabConfig(
                id="ui-settings",
                title="UI Settings",
                icon="fas fa-sliders",
                template="ui_settings_tab.html",
                description="Adjust WebUI preferences and behaviour",
                extra_context_handler=self._ui_settings_tab_context
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
            {"id": "project", "title": "Project Settings", "icon": "fas fa-project-diagram"},
            {"id": "training_data", "title": "Training Data", "icon": "fas fa-database"},
            {"id": "logging", "title": "Logging & Checkpoints", "icon": "fas fa-stream"},
            {"id": "other", "title": "Other Settings", "icon": "fas fa-sliders-h"},
        ]

        # Group fields and assign section_id to each field
        grouped_fields = self._group_basic_fields(fields)
        for section_id, section_fields in grouped_fields.items():
            for field in section_fields:
                field["section_id"] = section_id

        sections_with_fields = [section for section in sections if grouped_fields.get(section["id"])]

        context["sections"] = sections_with_fields
        context["grouped_fields"] = grouped_fields

        return context

    def _model_tab_context(
        self,
        context: Dict[str, Any],
        fields: List[Dict[str, Any]],
        config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for model tab."""
        danger_mode_enabled = context.get("danger_mode_enabled", False)
        model_type_value = str(config_values.get("model_type") or "full")
        is_lora_type = model_type_value == "lora"

        # Handle model_family label formatting
        for field_dict in fields:
            if field_dict["id"] == "model_family" and "options" in field_dict:
                field_dict["options"] = [
                    {"value": opt["value"], "label": self._get_model_family_label(opt["value"])}
                    for opt in field_dict["options"]
                ]
            elif field_dict["id"] == "lora_alpha":
                if not is_lora_type:
                    field_dict["disabled"] = True
                elif danger_mode_enabled:
                    field_dict.pop("disabled", None)
                else:
                    field_dict["value"] = config_values.get("lora_rank", "16")
                    field_dict["disabled"] = True
            elif field_dict["id"] == "prediction_type":
                field_dict["disabled"] = not danger_mode_enabled
                extra_classes = field_dict.get("extra_classes", "")
                field_dict["extra_classes"] = f"{extra_classes} danger-mode-target".strip()
            elif field_dict["id"] in {"base_model_precision", "text_encoder_1_precision", "quantize_via"}:
                if is_lora_type:
                    field_dict.pop("disabled", None)
                    continue

                field_dict["disabled"] = True
                extra_classes = field_dict.get("extra_classes", "")
                flag = "field-disabled"
                field_dict["extra_classes"] = f"{extra_classes} {flag}".strip()

        desired_order = {
            "model_family": 0,
            "model_flavour": 1,
            "pretrained_model_name_or_path": 2,
            "model_type": 3,
            "base_model_precision": 4,
            "gradient_accumulation_steps": 5,
        }
        fields.sort(key=lambda item: desired_order.get(item.get("id", ""), len(desired_order)))

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

    def _validation_tab_context(
        self,
        context: Dict[str, Any],
        fields: List[Dict[str, Any]],
        config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ensure validation tab renders its configuration fields."""
        sections = context.get("sections") or []

        if not sections:
            sections = [
                {
                    "id": context["tab_config"]["id"],
                    "title": context["tab_config"]["title"],
                    "icon": context["tab_config"].get("icon"),
                    "description": context["tab_config"].get("description"),
                }
            ]
            context["sections"] = sections
        # Group fields under sections to reuse form_tab rendering
        context["grouped_fields"] = self._group_fields_by_section(fields, sections)

        return context

    def _ui_settings_tab_context(
        self,
        context: Dict[str, Any],
        fields: List[Dict[str, Any]],
        config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide context data for UI settings tab."""
        store = WebUIStateStore()
        bundle = store.get_defaults_bundle()

        context["ui_settings"] = {
            "defaults": bundle["resolved"],
            "raw_defaults": bundle["raw"],
            "fallbacks": bundle["fallbacks"],
            "themes": [
                {
                    "value": "dark",
                    "label": "Dark",
                    "description": "Classic SimpleTuner palette"
                },
                {
                    "value": "tron",
                    "label": "Tron Prototype",
                    "description": "Experimental neon styling"
                },
            ],
            "event_interval_options": [3, 5, 10, 15, 30, 60],
        }
        return context

    def _datasets_tab_context(
        self,
        context: Dict[str, Any],
        fields: List[Dict[str, Any]],
        config_values: Dict[str, Any]
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
            "project": [],
            "training_data": [],
            "logging": [],
            "other": []
        }

        project_order = [
            "tracker_project_name",
            "tracker_run_name",
            "configs_dir",
            "resume_from_checkpoint",
            "output_dir",
        ]

        training_data_order = [
            "data_backend_config",
            "train_batch_size",
            "resolution",
        ]

        logging_order = [
            "checkpointing_steps",
            "checkpoints_total_limit",
            "report_to",
            "logging_dir",
            "tracker_image_layout",
        ]

        for field in fields:
            field_id = field.get("id", "")

            if field_id in project_order:
                grouped["project"].append(field)
            elif field_id in training_data_order:
                grouped["training_data"].append(field)
            elif field_id in logging_order:
                grouped["logging"].append(field)
            else:
                grouped["other"].append(field)

        # Enforce ordering within groups
        def _sort_group(items, order):
            order_map = {value: idx for idx, value in enumerate(order)}
            return sorted(items, key=lambda item: order_map.get(item.get("id", ""), len(order_map)))

        grouped["project"] = _sort_group(grouped["project"], project_order)
        grouped["training_data"] = _sort_group(grouped["training_data"], training_data_order)
        grouped["logging"] = _sort_group(grouped["logging"], logging_order)

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
