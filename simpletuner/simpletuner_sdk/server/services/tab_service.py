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
    ADVANCED = "advanced"
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
                extra_context_handler=self._basic_tab_context,
            ),
            TabType.MODEL: TabConfig(
                id="model-config",
                title="Model",
                icon="fas fa-brain",
                template="form_tab.html",
                description="Model architecture and settings",
                extra_context_handler=self._model_tab_context,
            ),
            TabType.TRAINING: TabConfig(
                id="training-config",
                title="Training",
                icon="fas fa-graduation-cap",
                template="form_tab.html",
                description="Training parameters and optimization",
            ),
            TabType.ADVANCED: TabConfig(
                id="advanced-config",
                title="Advanced",
                icon="fas fa-tools",
                template="form_tab.html",
                description="Advanced training options",
                extra_context_handler=self._advanced_tab_context,
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
            grouped_fields = self._group_fields_by_section(fields, sections)
            filtered_sections = []
            for section in sections:
                section_id = section["id"]
                section_fields = grouped_fields.get(section_id, [])
                if section_fields or section.get("empty_message") or section.get("template"):
                    filtered_sections.append(section)
            context["sections"] = filtered_sections
            context["grouped_fields"] = grouped_fields

        # Apply tab-specific context modifications
        if tab_config.extra_context_handler:
            context = tab_config.extra_context_handler(context, fields, config_values)

        # Render template
        return self.templates.TemplateResponse(request=context["request"], name=tab_config.template, context=context)

    def _group_fields_by_section(
        self, fields: List[Dict[str, Any]], sections: List[Dict[str, Any]]
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

        return grouped

    # Tab-specific context handlers
    def _basic_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for basic tab."""
        # Group fields into sections for basic tab
        sections = [
            {"id": "project", "title": "Project Settings", "icon": "fas fa-project-diagram"},
            {"id": "project_advanced", "title": "Advanced Project Settings", "icon": "fas fa-cogs", "advanced": True},
            {"id": "training_data", "title": "Training Data", "icon": "fas fa-database"},
            {"id": "logging", "title": "Logging & Checkpoints", "icon": "fas fa-stream"},
            {"id": "logging_advanced", "title": "Advanced Logging Settings", "icon": "fas fa-tools", "advanced": True},
            {"id": "other", "title": "Other Settings", "icon": "fas fa-sliders-h"},
        ]

        # Group fields and assign section_id to each field
        grouped_fields = self._group_basic_fields(fields)
        for section_id, section_fields in grouped_fields.items():
            for field in section_fields:
                field["section_id"] = section_id
                # Set parent_section relationships for advanced sections
                if section_id == "project_advanced":
                    field["subsection"] = "advanced_project"
                    field["parent_section"] = "project"
                elif section_id == "logging_advanced":
                    field["subsection"] = "advanced_logging"
                    field["parent_section"] = "logging"

        sections_with_fields = [section for section in sections if grouped_fields.get(section["id"])]

        context["sections"] = sections_with_fields
        context["grouped_fields"] = grouped_fields

        return context

    def _model_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for model tab."""
        danger_mode_enabled = context.get("danger_mode_enabled", False)
        model_type_value = str(config_values.get("model_type") or "full")
        is_lora_type = model_type_value == "lora"

        # Group fields into sections for model tab
        sections = [
            {"id": "model_config", "title": "Model Configuration", "icon": "fas fa-brain"},
            {
                "id": "model_config_advanced_paths",
                "title": "",
                "icon": "fas",
                "advanced": True,
            },
            {"id": "architecture", "title": "Architecture", "icon": "fas fa-cogs"},
            {"id": "architecture_advanced", "title": "", "icon": "fas", "advanced": True},
            {"id": "lora_config", "title": "LoRA Configuration", "icon": "fas fa-layer-group"},
            {
                "id": "lora_config_model_specific",
                "title": "Model-Specific LoRA Settings",
                "icon": "fas fa-cube",
                "advanced": True,
            },
            {"id": "lora_config_advanced", "title": "", "icon": "fas", "advanced": True},
            {"id": "vae_config", "title": "VAE Configuration", "icon": "fas fa-image"},
            {"id": "quantization", "title": "Quantization", "icon": "fas fa-microchip"},
            {"id": "memory_optimization", "title": "Memory Optimization", "icon": "fas fa-memory"},
        ]

        # Group fields and assign section_id to each field
        grouped_fields = self._group_model_fields(fields)
        for section_id, section_fields in grouped_fields.items():
            for field in section_fields:
                # Keep advanced sections as separate sections for collapse functionality
                # but mark them as subsections for proper nesting
                if section_id == "model_config_advanced_paths":
                    field["section_id"] = "model_config_advanced_paths"
                    field["subsection"] = "advanced_paths"
                    field["parent_section"] = "model_config"
                elif section_id == "architecture_controlnet":
                    field["section_id"] = "architecture_controlnet"
                    field["subsection"] = "controlnet"
                    field["parent_section"] = "architecture"
                elif section_id == "architecture_advanced":
                    field["section_id"] = "architecture_advanced"
                    field["subsection"] = "advanced"
                    field["parent_section"] = "architecture"
                elif section_id == "lora_config_model_specific":
                    field["section_id"] = "lora_config_model_specific"
                    field["subsection"] = "model_specific"
                    field["parent_section"] = "lora_config"
                elif section_id == "lora_config_advanced":
                    field["section_id"] = "lora_config_advanced"
                    field["subsection"] = "advanced"
                    field["parent_section"] = "lora_config"
                else:
                    field["section_id"] = section_id

        # Handle model_family label formatting and field-specific logic
        for field_dict in fields:
            field_id = field_dict.get("id", "")

            if field_id == "model_family" and "options" in field_dict:
                field_dict["options"] = [
                    {"value": opt["value"], "label": self._get_model_family_label(opt["value"])}
                    for opt in field_dict["options"]
                ]
            elif field_id == "lora_alpha":
                if not is_lora_type:
                    field_dict["disabled"] = True
                elif danger_mode_enabled:
                    field_dict.pop("disabled", None)
                else:
                    field_dict["value"] = config_values.get("lora_rank", "16")
                    field_dict["disabled"] = True
            elif field_id == "prediction_type":
                field_dict["disabled"] = not danger_mode_enabled
                extra_classes = field_dict.get("extra_classes", "")
                field_dict["extra_classes"] = f"{extra_classes} danger-mode-target".strip()
            elif field_id in {"base_model_precision", "text_encoder_1_precision", "quantize_via"}:
                if is_lora_type:
                    field_dict.pop("disabled", None)
                    continue

                field_dict["disabled"] = True
                extra_classes = field_dict.get("extra_classes", "")
                flag = "field-disabled"
                field_dict["extra_classes"] = f"{extra_classes} {flag}".strip()

        sections_with_fields = [section for section in sections if grouped_fields.get(section["id"])]

        context["sections"] = sections_with_fields
        context["grouped_fields"] = grouped_fields

        return context

    def _advanced_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for advanced tab."""
        # Add a warning about advanced settings
        context["warning_message"] = (
            "These are advanced settings. Modifying them incorrectly may impact training performance or stability."
        )
        return context

    def _validation_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
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

    def _publishing_tab_context(
        self, context: Dict[str, Any], fields: List[Dict[str, Any]], config_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize context for publishing tab using standard form template."""
        # Group fields and assign section_id to each field
        grouped_fields = self._group_publishing_fields(fields)
        for section_id, section_fields in grouped_fields.items():
            for field in section_fields:
                field["section_id"] = section_id

        # Get field-based sections from context (what came from field registry)
        field_sections = context.get("sections", [])

        # Merge with custom sections using the custom section service
        all_sections = CUSTOM_SECTION_SERVICE.merge_custom_sections_with_field_sections(
            tab="publishing", field_sections=field_sections
        )

        # Update context with merged sections and grouped fields
        context["sections"] = all_sections
        context["grouped_fields"] = grouped_fields

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

    def _group_basic_fields(self, fields: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group basic tab fields into sections."""
        grouped = {
            "project": [],
            "project_advanced": [],
            "training_data": [],
            "logging": [],
            "logging_advanced": [],
            "other": [],
        }

        project_order = [
            "tracker_project_name",
            "tracker_run_name",
            "configs_dir",
            "resume_from_checkpoint",
            "output_dir",
        ]

        project_advanced_order = [
            "seed",
            "merge_environment_config",
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
        ]

        logging_advanced_order = [
            "tracker_image_layout",
            "checkpointing_rolling_steps",
            "checkpointing_use_tempdir",
            "checkpoints_rolling_total_limit",
        ]

        # Fields to remove from Other Settings (move to publishing)
        publishing_fields = {
            "push_to_hub",
            "push_checkpoints_to_hub",
            "hub_model_id",
            "model_card_safe_for_work",
        }

        for field in fields:
            field_id = field.get("id", "")

            if field_id in project_order:
                grouped["project"].append(field)
            elif field_id in project_advanced_order:
                grouped["project_advanced"].append(field)
            elif field_id in training_data_order:
                grouped["training_data"].append(field)
            elif field_id in logging_order:
                grouped["logging"].append(field)
            elif field_id in logging_advanced_order:
                grouped["logging_advanced"].append(field)
            elif field_id in publishing_fields:
                # Skip these fields - they'll be handled in publishing tab
                continue
            else:
                grouped["other"].append(field)

        # Enforce ordering within groups
        def _sort_group(items, order):
            order_map = {value: idx for idx, value in enumerate(order)}
            return sorted(items, key=lambda item: order_map.get(item.get("id", ""), len(order_map)))

        grouped["project"] = _sort_group(grouped["project"], project_order)
        grouped["project_advanced"] = _sort_group(grouped["project_advanced"], project_advanced_order)
        grouped["training_data"] = _sort_group(grouped["training_data"], training_data_order)
        grouped["logging"] = _sort_group(grouped["logging"], logging_order)
        grouped["logging_advanced"] = _sort_group(grouped["logging_advanced"], logging_advanced_order)

        # Remove empty groups
        return {k: v for k, v in grouped.items() if v}

    def _group_publishing_fields(self, fields: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group publishing tab fields into sections."""
        grouped = {
            "publishing_controls": [],
            "repository": [],
            "model_card": [],
        }

        publishing_controls_order = [
            "push_to_hub",
            "push_checkpoints_to_hub",
        ]

        repository_order = [
            "hub_model_id",
            "model_card_private",
        ]

        model_card_order = [
            "model_card_safe_for_work",
            "model_card_note",
        ]

        for field in fields:
            field_id = field.get("id", "")

            if field_id in publishing_controls_order:
                grouped["publishing_controls"].append(field)
            elif field_id in repository_order:
                grouped["repository"].append(field)
            elif field_id in model_card_order:
                grouped["model_card"].append(field)

        # Enforce ordering within groups
        def _sort_group(items, order):
            order_map = {value: idx for idx, value in enumerate(order)}
            return sorted(items, key=lambda item: order_map.get(item.get("id", ""), len(order_map)))

        grouped["publishing_controls"] = _sort_group(grouped["publishing_controls"], publishing_controls_order)
        grouped["repository"] = _sort_group(grouped["repository"], repository_order)
        grouped["model_card"] = _sort_group(grouped["model_card"], model_card_order)

        # Remove empty groups
        return {k: v for k, v in grouped.items() if v}

    def _group_model_fields(self, fields: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group model tab fields into sections."""
        grouped = {
            "model_config": [],
            "model_config_advanced_paths": [],
            "architecture": [],
            "architecture_advanced": [],
            "lora_config": [],
            "lora_config_model_specific": [],
            "lora_config_advanced": [],
            "vae_config": [],
            "quantization": [],
            "memory_optimization": [],
        }

        # Model Configuration (basic)
        model_config_order = [
            "model_type",
            "model_family",
            "model_flavour",
        ]

        # Advanced Model Paths
        model_config_advanced_paths_order = [
            "pretrained_model_name_or_path",
            "pretrained_transformer_model_name_or_path",
            "pretrained_transformer_subfolder",
            "pretrained_unet_model_name_or_path",
            "pretrained_unet_subfolder",
            "pretrained_vae_model_name_or_path",
            "pretrained_t5_model_name_or_path",
            "controlnet_model_name_or_path",
            "revision",
            "variant",
        ]

        # Architecture (basic)
        architecture_order = [
            "fused_qkv_projections",
            "control",
        ]

        # Advanced Architecture
        architecture_advanced_order = [
            "controlnet",
            "controlnet_custom_config",
            "prediction_type",
            "tread_config",
        ]

        # LoRA Configuration (basic)
        lora_config_order = [
            "lora_rank",
            "lora_type",
            "peft_lora_mode",
            "singlora_ramp_up_steps",
            "lycoris_config",
            "init_lokr_norm",
        ]

        # Model-specific LoRA Settings
        lora_model_specific_order = [
            "flux_lora_target",
        ]

        # Advanced LoRA Settings
        lora_config_advanced_order = [
            "lora_alpha",
            "lora_dropout",
            "lora_init_type",
            "init_lora",
            "use_dora",
        ]

        # VAE Configuration
        vae_config_order = [
            "vae_dtype",
            "vae_cache_ondemand",
        ]

        # Quantization
        quantization_order = [
            "base_model_precision",
            "base_model_default_dtype",
            "text_encoder_1_precision",
            "text_encoder_2_precision",
            "text_encoder_3_precision",
            "text_encoder_4_precision",
            "quantize_via",
        ]

        # Memory Optimization
        memory_optimization_order = [
            "gradient_checkpointing_interval",
            "offload_during_startup",
            "unet_attention_slice",
        ]

        for field in fields:
            field_id = field.get("id", "")
            section = field.get("section", "")
            subsection = field.get("subsection", "")

            # Determine section based on field metadata
            if section == "model_config" and subsection == "architecture":
                if field_id in model_config_order:
                    grouped["model_config"].append(field)
            elif section == "model_config" and subsection == "advanced_paths":
                if field_id in model_config_advanced_paths_order:
                    grouped["model_config_advanced_paths"].append(field)
            elif section == "architecture" and subsection in ("", None):
                if field_id in architecture_order:
                    grouped["architecture"].append(field)
            elif section == "architecture" and subsection == "advanced":
                if field_id in architecture_advanced_order:
                    grouped["architecture_advanced"].append(field)
            elif section == "lora_config" and subsection in ("", None):
                if field_id in lora_config_order:
                    grouped["lora_config"].append(field)
            elif section == "lora_config" and subsection == "model_specific":
                if field_id in lora_model_specific_order:
                    grouped["lora_config_model_specific"].append(field)
            elif section == "lora_config" and subsection == "advanced":
                if field_id in lora_config_advanced_order:
                    grouped["lora_config_advanced"].append(field)
            elif section == "vae_config":
                if field_id in vae_config_order:
                    grouped["vae_config"].append(field)
            elif section == "quantization":
                if field_id in quantization_order:
                    grouped["quantization"].append(field)
            elif section == "memory_optimization":
                if field_id in memory_optimization_order:
                    grouped["memory_optimization"].append(field)
            else:
                # Fallback: try to match by field ID patterns
                if field_id in model_config_order:
                    grouped["model_config"].append(field)
                elif field_id in model_config_advanced_paths_order:
                    grouped["model_config_advanced_paths"].append(field)
                elif field_id in architecture_order:
                    grouped["architecture"].append(field)
                elif field_id in architecture_advanced_order:
                    grouped["architecture_advanced"].append(field)
                elif field_id in lora_config_order:
                    grouped["lora_config"].append(field)
                elif field_id in lora_config_advanced_order:
                    grouped["lora_config_advanced"].append(field)
                elif field_id in vae_config_order:
                    grouped["vae_config"].append(field)
                elif field_id in quantization_order:
                    grouped["quantization"].append(field)
                elif field_id in memory_optimization_order:
                    grouped["memory_optimization"].append(field)

        # Enforce ordering within groups
        def _sort_group(items, order):
            order_map = {value: idx for idx, value in enumerate(order)}
            return sorted(items, key=lambda item: order_map.get(item.get("id", ""), len(order_map)))

        grouped["model_config"] = _sort_group(grouped["model_config"], model_config_order)
        grouped["model_config_advanced_paths"] = _sort_group(
            grouped["model_config_advanced_paths"], model_config_advanced_paths_order
        )
        grouped["architecture"] = _sort_group(grouped["architecture"], architecture_order)
        grouped["architecture_advanced"] = _sort_group(grouped["architecture_advanced"], architecture_advanced_order)
        grouped["lora_config"] = _sort_group(grouped["lora_config"], lora_config_order)
        grouped["lora_config_model_specific"] = _sort_group(grouped["lora_config_model_specific"], lora_model_specific_order)
        grouped["lora_config_advanced"] = _sort_group(grouped["lora_config_advanced"], lora_config_advanced_order)
        grouped["vae_config"] = _sort_group(grouped["vae_config"], vae_config_order)
        grouped["quantization"] = _sort_group(grouped["quantization"], quantization_order)
        grouped["memory_optimization"] = _sort_group(grouped["memory_optimization"], memory_optimization_order)

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
            "qwen_image": "Qwen Image",
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
                "description": config.description,
            }
            for tab_type, config in self._tab_configs.items()
        ]
