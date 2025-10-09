"""Service for field conversion and manipulation.

This service handles converting fields between different formats,
applying transformations, and managing field metadata.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from simpletuner.helpers.models.common import PredictionTypes, VideoModelFoundation
from simpletuner.helpers.models.registry import ModelRegistry

from ..services.field_registry_wrapper import lazy_field_registry
from ..utils.paths import resolve_config_path
from .dataset_plan import DatasetPlanStore
from .dataset_service import normalize_dataset_config_value
from .field_registry import FieldType
from .webhook_defaults import DEFAULT_WEBHOOK_CONFIG

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SectionLayout:
    """Declarative description of how fields map to UI sections."""

    id: str
    title: str
    icon: str = ""
    description: Optional[str] = None
    advanced: bool = False
    parent: Optional[str] = None
    match_section: Optional[str] = None
    match_sections: Optional[Tuple[str, ...]] = None
    match_subsections: Optional[Tuple[Optional[str], ...]] = None
    subsection_override: Optional[str] = None
    template: Optional[str] = None
    empty_message: Optional[str] = None
    order: int = 0
    include_if_empty: bool = False


class FieldFormat(str, Enum):
    """Supported field formats."""

    TEMPLATE = "template"  # Format for HTML templates
    API = "api"  # Format for API responses
    CONFIG = "config"  # Format for configuration files
    COMMAND = "command"  # Format for command-line arguments


class FieldService:
    """Service for field conversion and manipulation."""

    _TEXT_ENCODER_TRAINING_FIELDS = {
        "train_text_encoder",
        "text_encoder_lr",
    }

    _TEXT_ENCODER_PRECISION_FIELDS = {f"text_encoder_{idx}_precision" for idx in range(1, 5)}
    _NOISE_OFFSET_FIELDS = {"offset_noise", "noise_offset", "noise_offset_probability"}
    _FLOW_SCHEDULE_FIELDS = {
        "flow_sigmoid_scale",
        "flow_use_uniform_schedule",
        "flow_use_beta_schedule",
        "flow_beta_schedule_alpha",
        "flow_beta_schedule_beta",
        "flow_schedule_shift",
        "flow_schedule_auto_shift",
    }
    _SOFT_MIN_SNR_FIELDS = {
        "use_soft_min_snr",
        "soft_min_snr_sigma_data",
    }
    _SNR_GAMMA_FIELDS = {"snr_gamma"}

    _WEBUI_ONLY_FIELDS = {"configs_dir", "num_validation_images", "__active_tab__", "project_name", "uploadMode"}
    # Fields that the WebUI manages internally so users cannot override them
    _WEBUI_FORCED_VALUES = {
        "webhook_config": DEFAULT_WEBHOOK_CONFIG,
        "webhook_reporting_interval": 1,
    }
    _WEBUI_FIELD_HINTS = {
        "webhook_config": "Managed by the WebUI so training callbacks can reach the server. Use the CLI to supply custom webhooks.",
    }
    _VIDEO_ONLY_FIELDS = {
        "framerate",
        "validation_num_video_frames",
    }
    _EVALUATION_FIELDS = {
        "evaluation_type",
        "eval_dataset_id",
        "num_eval_images",
        "eval_steps_interval",
        "eval_timesteps",
        "eval_dataset_pooling",
        "pretrained_evaluation_model_name_or_path",
    }
    _EVALUATION_DEPENDENT_FIELDS = {
        "eval_dataset_id",
        "num_eval_images",
        "eval_steps_interval",
        "eval_timesteps",
        "eval_dataset_pooling",
        "pretrained_evaluation_model_name_or_path",
    }
    _EVAL_DATASET_REQUIRED_HINT = "Configure at least one evaluation dataset on the Datasets tab to enable these settings."
    _EVAL_TYPE_REQUIRED_HINT = "Select an evaluation type to adjust these settings."

    _TAB_SECTION_LAYOUTS: Dict[str, Tuple[SectionLayout, ...]] = {
        "basic": (
            SectionLayout(
                id="project",
                title="Project Settings",
                icon="fas fa-project-diagram",
                match_sections=("project", "essential_settings", "project_settings"),
                match_subsections=(None, "paths"),
                subsection_override="",
                order=10,
            ),
            SectionLayout(
                id="project_advanced",
                title="",
                icon="",
                advanced=True,
                parent="project",
                match_sections=("project", "project_settings"),
                match_subsections=("advanced",),
                subsection_override="advanced_project",
                order=11,
            ),
            SectionLayout(
                id="training_data",
                title="Training Data",
                icon="fas fa-database",
                match_sections=("training_data", "training_essentials", "data_config"),
                match_subsections=(None,),
                subsection_override="",
                order=20,
            ),
            SectionLayout(
                id="training_data_advanced",
                title="",
                icon="",
                advanced=True,
                parent="training_data",
                match_sections=("training_data", "training_essentials", "data_config"),
                match_subsections=("advanced",),
                subsection_override="advanced_training_data",
                order=21,
            ),
            SectionLayout(
                id="image_processing",
                title="Dataset Defaults",
                icon="fas fa-database",
                match_sections=("image_processing", "dataset_defaults"),
                match_subsections=(None,),
                subsection_override="",
                order=22,
            ),
            SectionLayout(
                id="image_processing_advanced",
                title="",
                icon="",
                advanced=True,
                parent="image_processing",
                match_sections=("image_processing", "dataset_defaults"),
                match_subsections=("advanced",),
                subsection_override="advanced_image_processing",
                order=23,
            ),
            SectionLayout(
                id="caching",
                title="Caching",
                icon="fas fa-hdd",
                match_sections=("caching",),
                match_subsections=(None,),
                subsection_override="",
                order=25,
            ),
            SectionLayout(
                id="caching_advanced",
                title="",
                icon="",
                advanced=True,
                parent="caching",
                match_sections=("caching",),
                match_subsections=("advanced",),
                subsection_override="advanced_caching",
                order=26,
            ),
            SectionLayout(
                id="logging",
                title="Logging",
                icon="fas fa-stream",
                match_sections=("logging",),
                match_subsections=(None,),
                subsection_override="",
                order=30,
            ),
            SectionLayout(
                id="logging_advanced",
                title="",
                icon="",
                advanced=True,
                parent="logging",
                match_section="logging",
                match_subsections=("monitoring",),
                subsection_override="advanced_logging",
                order=31,
            ),
            SectionLayout(
                id="checkpointing",
                title="Checkpointing",
                icon="fas fa-save",
                match_sections=("checkpointing", "training_config"),
                match_subsections=(None,),
                subsection_override="",
                order=12,
            ),
            SectionLayout(
                id="checkpointing_advanced",
                title="",
                icon="",
                advanced=True,
                parent="checkpointing",
                match_section="checkpointing",
                match_subsections=("advanced",),
                subsection_override="advanced_checkpointing",
                order=12,
            ),
            SectionLayout(
                id="other",
                title="Other Settings",
                icon="fas fa-sliders-h",
                match_section="other",
                match_subsections=None,
                subsection_override="",
                order=40,
            ),
        ),
        "hardware": (
            SectionLayout(
                id="accelerate",
                title="Accelerate & Distributed",
                icon="fas fa-project-diagram",
                match_sections=("accelerate",),
                match_subsections=(None,),
                subsection_override="",
                order=10,
            ),
            SectionLayout(
                id="accelerate_advanced",
                title="",
                icon="",
                advanced=True,
                parent="accelerate",
                match_sections=("accelerate",),
                match_subsections=("advanced",),
                subsection_override="advanced_accelerate",
                order=11,
            ),
            SectionLayout(
                id="hardware",
                title="Worker Settings",
                icon="fas fa-microchip",
                match_sections=("hardware",),
                match_subsections=(None,),
                subsection_override="",
                order=20,
            ),
            SectionLayout(
                id="hardware_advanced",
                title="",
                icon="",
                advanced=True,
                parent="hardware",
                match_sections=("hardware",),
                match_subsections=("advanced",),
                subsection_override="advanced_hardware",
                order=21,
            ),
        ),
        "model": (
            SectionLayout(
                id="model_config",
                title="Model Configuration",
                icon="fas fa-brain",
                match_section="model_config",
                match_subsections=(None, "architecture"),
                subsection_override="",
                order=10,
            ),
            SectionLayout(
                id="model_config_advanced_paths",
                title="",
                icon="",
                advanced=True,
                parent="model_config",
                match_section="model_config",
                match_subsections=("advanced_paths",),
                subsection_override="advanced_paths",
                order=11,
            ),
            SectionLayout(
                id="architecture",
                title="Architecture",
                icon="fas fa-cogs",
                match_section="architecture",
                match_subsections=(None,),
                subsection_override="",
                order=20,
            ),
            SectionLayout(
                id="architecture_advanced",
                title="",
                icon="",
                advanced=True,
                parent="architecture",
                match_section="architecture",
                match_subsections=("advanced",),
                subsection_override="advanced",
                order=21,
            ),
            SectionLayout(
                id="lora_config",
                title="LoRA Configuration",
                icon="fas fa-layer-group",
                match_section="lora_config",
                match_subsections=(None, "basic", "model_specific"),
                subsection_override="",
                order=30,
            ),
            SectionLayout(
                id="lora_config_advanced",
                title="",
                icon="",
                advanced=True,
                parent="lora_config",
                match_section="lora_config",
                match_subsections=("advanced",),
                subsection_override="advanced",
                order=32,
            ),
            SectionLayout(
                id="vae_config",
                title="VAE Configuration",
                icon="fas fa-image",
                match_section="vae_config",
                match_subsections=None,
                subsection_override="",
                order=40,
            ),
            SectionLayout(
                id="quantization",
                title="Quantization",
                icon="fas fa-microchip",
                match_section="quantization",
                match_subsections=None,
                subsection_override="",
                order=50,
            ),
            SectionLayout(
                id="memory_optimization",
                title="Memory Optimisation",
                icon="fas fa-memory",
                match_section="memory_optimization",
                match_subsections=(None, "memory_optimization"),
                subsection_override="",
                order=60,
            ),
            SectionLayout(
                id="memory_optimization_advanced",
                title="",
                icon="",
                advanced=True,
                parent="memory_optimization",
                match_section="memory_optimization",
                match_subsections=("advanced",),
                subsection_override="advanced",
                order=61,
            ),
            SectionLayout(
                id="ema_config",
                title="EMA Configuration",
                icon="fas fa-wave-square",
                match_section="ema_config",
                match_subsections=None,
                subsection_override="",
                order=70,
            ),
            SectionLayout(
                id="distillation",
                title="Distillation",
                icon="fas fa-copy",
                match_section="distillation",
                match_subsections=None,
                subsection_override="",
                order=80,
            ),
            SectionLayout(
                id="model_specific",
                title="Model-Specific Settings",
                icon="fas fa-cube",
                match_section="model_specific",
                match_subsections=None,
                subsection_override="",
                order=90,
            ),
        ),
        "training": (
            SectionLayout(
                id="training_schedule",
                title="Training Schedule",
                icon="fas fa-calendar-alt",
                match_section="training_schedule",
                match_subsections=(None,),
                subsection_override="",
                order=10,
            ),
            SectionLayout(
                id="training_schedule_advanced",
                title="",
                icon="",
                advanced=True,
                parent="training_schedule",
                match_section="training_schedule",
                match_subsections=("advanced",),
                subsection_override="advanced",
                order=11,
            ),
            SectionLayout(
                id="learning_rate",
                title="Learning Rate",
                icon="fas fa-chart-line",
                match_section="learning_rate",
                match_subsections=(None,),
                subsection_override="",
                order=20,
            ),
            SectionLayout(
                id="learning_rate_advanced",
                title="",
                icon="",
                advanced=True,
                parent="learning_rate",
                match_section="learning_rate",
                match_subsections=("advanced",),
                subsection_override="advanced",
                order=21,
            ),
            SectionLayout(
                id="optimizer_config",
                title="Optimizer Configuration",
                icon="fas fa-cogs",
                match_section="optimizer_config",
                match_subsections=(None,),
                subsection_override="",
                order=30,
            ),
            SectionLayout(
                id="optimizer_config_advanced",
                title="",
                icon="",
                advanced=True,
                parent="optimizer_config",
                match_section="optimizer_config",
                match_subsections=("advanced",),
                subsection_override="advanced",
                order=31,
            ),
            SectionLayout(
                id="text_encoder",
                title="Text Encoder",
                icon="fas fa-font",
                match_section="text_encoder",
                match_subsections=(None,),
                subsection_override="",
                order=70,
            ),
            SectionLayout(
                id="text_encoder_advanced",
                title="",
                icon="",
                advanced=True,
                parent="text_encoder",
                match_section="text_encoder",
                match_subsections=("advanced",),
                subsection_override="advanced",
                order=71,
            ),
            SectionLayout(
                id="memory_optimization",
                title="Memory Optimisation",
                icon="fas fa-memory",
                match_section="memory_optimization",
                match_subsections=(None, "memory_optimization"),
                subsection_override="",
                order=60,
            ),
            SectionLayout(
                id="memory_optimization_advanced",
                title="",
                icon="",
                advanced=True,
                parent="memory_optimization",
                match_section="memory_optimization",
                match_subsections=("advanced",),
                subsection_override="advanced",
                order=61,
            ),
            SectionLayout(
                id="noise_settings",
                title="Noise Settings",
                icon="fas fa-broadcast-tower",
                match_section="noise_settings",
                match_subsections=None,
                subsection_override="",
                order=50,
            ),
            SectionLayout(
                id="loss_functions",
                title="Loss Functions",
                icon="fas fa-calculator",
                match_section="loss_functions",
                match_subsections=(None,),
                subsection_override="",
                order=40,
            ),
            SectionLayout(
                id="loss_functions_advanced",
                title="",
                icon="",
                advanced=True,
                parent="loss_functions",
                match_section="loss_functions",
                match_subsections=("advanced",),
                subsection_override="advanced",
                order=41,
            ),
        ),
        "validation": (
            SectionLayout(
                id="validation_schedule",
                title="Validation Schedule",
                icon="fas fa-calendar-check",
                match_section="validation_schedule",
                match_subsections=(None,),
                subsection_override="",
                order=10,
            ),
            SectionLayout(
                id="validation_schedule_advanced",
                title="",
                icon="",
                advanced=True,
                parent="validation_schedule",
                match_section="validation_schedule",
                match_subsections=("advanced",),
                subsection_override="advanced",
                order=11,
            ),
            SectionLayout(
                id="prompt_management",
                title="Prompt Management",
                icon="fas fa-comment-dots",
                match_section="prompt_management",
                match_subsections=(None,),
                subsection_override="",
                order=20,
            ),
            SectionLayout(
                id="validation_guidance",
                title="Validation Guidance",
                icon="fas fa-sliders-h",
                match_section="validation_guidance",
                match_subsections=(None,),
                subsection_override="",
                order=30,
            ),
            SectionLayout(
                id="validation_guidance_advanced",
                title="",
                icon="",
                advanced=True,
                parent="validation_guidance",
                match_section="validation_guidance",
                match_subsections=("advanced",),
                subsection_override="advanced",
                order=31,
            ),
            SectionLayout(
                id="validation_options",
                title="Validation Options",
                icon="fas fa-wrench",
                match_section="validation_options",
                match_subsections=(None,),
                subsection_override="",
                order=40,
            ),
            SectionLayout(
                id="validation_options_advanced",
                title="",
                icon="",
                advanced=True,
                parent="validation_options",
                match_section="validation_options",
                match_subsections=("advanced",),
                subsection_override="advanced",
                order=41,
            ),
            SectionLayout(
                id="evaluation",
                title="Evaluation Metrics",
                icon="fas fa-chart-area",
                match_section="evaluation",
                match_subsections=(None,),
                subsection_override="",
                order=50,
            ),
            SectionLayout(
                id="evaluation_advanced",
                title="",
                icon="",
                advanced=True,
                parent="evaluation",
                match_section="evaluation",
                match_subsections=("advanced",),
                subsection_override="advanced",
                order=51,
            ),
        ),
    }

    def __init__(self):
        """Initialize field service."""
        self.field_registry = lazy_field_registry
        self._format_converters = {
            FieldFormat.TEMPLATE: self._convert_to_template_format,
            FieldFormat.API: self._convert_to_api_format,
            FieldFormat.CONFIG: self._convert_to_config_format,
            FieldFormat.COMMAND: self._convert_to_command_format,
        }

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        """Convert assorted representations into a boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return False

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        """Convert value to int, falling back to default on failure."""
        if value in (None, ""):
            return default
        try:
            if isinstance(value, bool):  # avoid True -> 1
                return default
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                return int(value.strip())
        except (ValueError, TypeError):
            return default
        return default

    @staticmethod
    def _extract_config_payload(raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise raw config to the actual parameter payload."""

        if not isinstance(raw_config, dict):
            return {}

        if "config" in raw_config and isinstance(raw_config["config"], dict):
            other_keys = {key for key in raw_config.keys() if key != "config"}
            if not other_keys or other_keys == {"_metadata"}:
                return dict(raw_config["config"])

        return raw_config

    @staticmethod
    def _resolve_dataset_config_path(value: Any, configs_dir: Optional[str]) -> Optional[Path]:
        if not value:
            return None

        try:
            resolved = resolve_config_path(value, config_dir=configs_dir, check_cwd_first=True)
        except Exception:
            resolved = None

        if resolved:
            return resolved

        try:
            candidate = Path(str(value)).expanduser()
            if candidate.exists():
                return candidate.resolve()
        except Exception:
            return None
        return None

    @staticmethod
    def _normalise_evaluation_type(value: Any) -> str:
        if value is None:
            return ""
        try:
            return str(value).strip().lower()
        except Exception:
            return ""

    def _collect_eval_dataset_options(
        self,
        config_data: Dict[str, Any],
        webui_defaults: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        configs_dir = webui_defaults.get("configs_dir") if isinstance(webui_defaults, dict) else None
        candidate_paths: List[Path] = []
        seen_paths: Set[Path] = set()

        def _maybe_add_path(path: Optional[Path]) -> None:
            if not path:
                return
            try:
                candidate = path.expanduser().resolve(strict=False)
            except Exception:
                candidate = path
            if candidate in seen_paths or not candidate.exists():
                return
            seen_paths.add(candidate)
            candidate_paths.append(candidate)

        for key in ("--data_backend_config", "data_backend_config"):
            candidate_value = config_data.get(key)
            resolved = self._resolve_dataset_config_path(candidate_value, configs_dir)
            if resolved:
                _maybe_add_path(resolved)

        if configs_dir:
            try:
                default_candidate = Path(str(configs_dir)).expanduser() / "multidatabackend.json"
                _maybe_add_path(default_candidate)
            except Exception:
                pass

        try:
            store_path = DatasetPlanStore().path
            _maybe_add_path(store_path)
        except Exception:
            pass

        options: Dict[str, Dict[str, str]] = {}

        for path in candidate_paths:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception:
                continue

            if not isinstance(data, list):
                continue

            for entry in data:
                if not isinstance(entry, dict):
                    continue

                dataset_type = str(entry.get("dataset_type", "")).strip().lower()
                if dataset_type != "eval":
                    continue

                if entry.get("disabled") or entry.get("disable"):
                    continue

                dataset_id = entry.get("id")
                if not isinstance(dataset_id, str) or not dataset_id.strip():
                    continue

                dataset_id = dataset_id.strip()
                label_candidates = [
                    entry.get("name"),
                    entry.get("label"),
                ]
                metadata = entry.get("_metadata")
                if isinstance(metadata, dict):
                    label_candidates.append(metadata.get("name"))

                label = next(
                    (
                        str(candidate).strip()
                        for candidate in label_candidates
                        if isinstance(candidate, str) and candidate.strip()
                    ),
                    dataset_id,
                )

                if dataset_id not in options:
                    options[dataset_id] = {"value": dataset_id, "label": label}

        return sorted(options.values(), key=lambda item: item["label"].lower())

    @staticmethod
    def _config_has_field(raw_config: Dict[str, Any], field_name: str) -> bool:
        """Check if a raw config explicitly sets a given field."""

        if not raw_config:
            return False

        base_name = field_name[2:] if field_name.startswith("--") else field_name
        variants = {
            field_name,
            f"--{base_name}",
            base_name,
            base_name.lower(),
            base_name.replace("_", "").lower(),
        }

        for key in variants:
            if key in raw_config:
                value = raw_config[key]
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                return True

        return False

    def _is_danger_mode_enabled(self, config_data: Dict[str, Any]) -> bool:
        """Determine whether dangerous overrides are enabled."""
        for key in ("i_know_what_i_am_doing", "--i_know_what_i_am_doing"):
            if key in config_data and self._coerce_bool(config_data[key]):
                return True
        return False

    def _get_model_class(self, config_values: Dict[str, Any]):
        model_family = config_values.get("model_family") or config_values.get("--model_family")
        if not model_family:
            return None

        try:
            return ModelRegistry.get(model_family)
        except Exception:
            return None

    def _get_text_encoder_configuration(self, config_values: Dict[str, Any]) -> Dict[str, Any]:
        model_class = self._get_model_class(config_values)
        if not model_class:
            return {}

        configuration = getattr(model_class, "TEXT_ENCODER_CONFIGURATION", None) or {}
        if isinstance(configuration, dict):
            return configuration
        return {}

    def _supports_text_encoder_training(self, config_values: Dict[str, Any]) -> bool:
        """Check whether the selected model supports text encoder training."""
        model_class = self._get_model_class(config_values)
        if not model_class:
            logger.warning(f"Could not get model class for config: {config_values}")
            return False

        return bool(getattr(model_class, "SUPPORTS_TEXT_ENCODER_TRAINING", False))

    def _supports_noise_offset(self, config_values: Dict[str, Any]) -> bool:
        """Determine if noise offset settings are compatible with the model."""

        prediction_type = self._resolve_prediction_type(config_values)
        if not prediction_type:
            return True  # default to visible when prediction type unknown

        prediction_type = prediction_type.lower()
        return prediction_type in {"epsilon", "v_prediction"}

    def _is_video_model(self, config_values: Dict[str, Any]) -> bool:
        """Check whether the selected model inherits from the video foundation base."""

        model_class = self._get_model_class(config_values)
        if not model_class:
            return False

        try:
            return issubclass(model_class, VideoModelFoundation)
        except TypeError:
            return False

    def _resolve_prediction_type(self, config_values: Dict[str, Any]) -> Optional[str]:
        """Resolve the effective prediction type from config or model defaults."""

        for key in ("prediction_type", "--prediction_type"):
            if key in config_values and config_values[key]:
                return self._normalise_prediction_type(config_values[key])

        model_class = self._get_model_class(config_values)
        if not model_class:
            return None

        attr = getattr(model_class, "PREDICTION_TYPE", None)
        return self._normalise_prediction_type(attr)

    @staticmethod
    def _normalise_prediction_type(value: Any) -> Optional[str]:
        if value is None:
            return None

        if PredictionTypes and isinstance(value, PredictionTypes):
            return value.value

        if hasattr(value, "value"):
            try:
                return str(value.value)
            except Exception:
                pass

        return str(value).strip().lower() if str(value).strip() else None

    def convert_field(
        self, field: Any, format: FieldFormat, config_values: Dict[str, Any], options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Convert a field to the specified format.

        Args:
            field: Field object to convert
            format: Target format
            config_values: Current configuration values
            options: Optional conversion options

        Returns:
            Converted field dictionary
        """
        converter = self._format_converters.get(format)
        if not converter:
            raise ValueError(f"Unsupported format: {format}")

        return converter(field, config_values, options or {})

    def convert_fields(
        self, fields: List[Any], format: FieldFormat, config_values: Dict[str, Any], options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Convert multiple fields to the specified format.

        Args:
            fields: List of field objects
            format: Target format
            config_values: Current configuration values
            options: Optional conversion options

        Returns:
            List of converted field dictionaries
        """
        options = options or {}
        combined_config: Dict[str, Any] = {}

        raw_config = options.get("raw_config")
        raw_config_data: Dict[str, Any] = {}
        if isinstance(raw_config, dict):
            combined_config.update(raw_config)
            raw_config_data = self._extract_config_payload(raw_config)

        if isinstance(config_values, dict):
            combined_config.update(config_values)

        supports_text_encoder = self._supports_text_encoder_training(combined_config)
        supports_noise_offset = self._supports_noise_offset(combined_config)
        text_encoder_config = self._get_text_encoder_configuration(combined_config)
        available_text_encoders = list(text_encoder_config.keys()) if text_encoder_config else []
        encoder_count = len(available_text_encoders)
        is_video_model = self._is_video_model(combined_config)

        filtered_fields: List[Any] = []

        for field in fields:
            model_specific = getattr(field, "model_specific", None)
            if model_specific:
                selected_family = self._get_config_value(combined_config, "model_family")
                if not selected_family or selected_family not in model_specific:
                    continue
            name = field.name

            if name in self._WEBUI_ONLY_FIELDS:
                continue

            if name in self._VIDEO_ONLY_FIELDS and not is_video_model:
                if not self._config_has_field(raw_config_data, name):
                    continue

            if name in self._TEXT_ENCODER_TRAINING_FIELDS and not supports_text_encoder:
                continue

            if name in self._TEXT_ENCODER_PRECISION_FIELDS:
                index = int(name.split("_")[2])
                has_config = index <= encoder_count
                has_existing_value = self._config_has_field(raw_config_data, name)

                if not (has_config or has_existing_value):
                    continue

            if name in self._NOISE_OFFSET_FIELDS and not supports_noise_offset:
                if not self._config_has_field(raw_config_data, name):
                    continue

            if name in self._FLOW_SCHEDULE_FIELDS:
                prediction_type = self._resolve_prediction_type(combined_config)
                if prediction_type != "flow_matching" and not self._config_has_field(raw_config_data, name):
                    continue

            if name in self._SOFT_MIN_SNR_FIELDS:
                prediction_type = self._resolve_prediction_type(combined_config)
                if prediction_type not in {"v_prediction", "epsilon"}:
                    continue

            if name in self._SNR_GAMMA_FIELDS:
                prediction_type = self._resolve_prediction_type(combined_config)
                if prediction_type not in {"v_prediction", "epsilon"}:
                    continue

            if name in {"hidream_use_load_balancing_loss", "hidream_load_balancing_loss_weight"}:
                model_family = self._get_config_value(combined_config, "model_family")
                if model_family != "hidream":
                    continue

            filtered_fields.append(field)

        return [self.convert_field(field, format, combined_config, options) for field in filtered_fields]

    @staticmethod
    def _get_config_value(config_values: Dict[str, Any], key: str) -> Any:
        if not isinstance(config_values, dict) or not key:
            return None

        variants = {key}
        base = key[2:] if key.startswith("--") else key
        variants.add(base)
        variants.add(base.lower())
        variants.add(base.replace("-", "_"))
        variants.add(base.replace("-", "").lower())
        variants.add(base.replace("_", "").lower())
        variants.add(f"--{base}")

        for candidate in variants:
            if candidate in config_values:
                return config_values[candidate]
        return None

    def get_fields_for_section(
        self, tab_name: str, section_name: str, format: FieldFormat, config_values: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get fields for a specific section in the desired format.

        Args:
            tab_name: Tab name
            section_name: Section name
            format: Desired format
            config_values: Current configuration values

        Returns:
            List of formatted field dictionaries
        """
        # Get sections for the tab
        sections = self.field_registry.get_sections_for_tab(tab_name)
        section_dict = {s["id"]: s for s in sections}

        if section_name not in section_dict:
            logger.warning(f"Section '{section_name}' not found in tab '{tab_name}'")
            return []

        # Get fields for the section
        section_fields = []
        all_fields = self.field_registry.get_fields_for_tab(tab_name)

        for field in all_fields:
            if hasattr(field, "section") and field.section == section_name:
                section_fields.append(field)

        return self.convert_fields(section_fields, format, config_values)

    def get_dependent_fields(self, field_name: str) -> Set[str]:
        """Get fields that depend on the given field.

        Args:
            field_name: Name of the field

        Returns:
            Set of dependent field names
        """
        return set(self.field_registry.get_dependent_fields(field_name))

    def get_field_dependencies(self, field_name: str) -> Set[str]:
        """Get fields that the given field depends on.

        Args:
            field_name: Name of the field

        Returns:
            Set of dependency field names
        """
        field = self.field_registry.get_field(field_name)
        if not field:
            return set()

        dependencies = set()
        if hasattr(field, "conditional_on"):
            dependencies.add(field.conditional_on)

        if hasattr(field, "depends_on") and isinstance(field.depends_on, list):
            dependencies.update(field.depends_on)

        return dependencies

    def apply_field_transformations(self, field_name: str, value: Any, config_values: Dict[str, Any]) -> Any:
        """Apply field-specific transformations to a value.

        Args:
            field_name: Name of the field
            value: Raw value
            config_values: Current configuration values

        Returns:
            Transformed value
        """
        # Special handling for specific fields
        if field_name == "num_train_epochs":
            # Convert string "0" to integer 0
            if str(value) == "0" or value == 0:
                return 0
            elif value == 1 and field_name not in config_values:
                # If using default value of 1, start empty for UI
                return ""
            return value

        elif field_name == "max_train_steps":
            # Convert string "0" to integer 0
            if str(value) == "0" or value == 0:
                return 0
            return value

        elif field_name == "checkpoints_total_limit":
            if value in (None, "") and "save_total_limit" in config_values:
                return config_values.get("save_total_limit")
            return value

        elif field_name == "vae_cache_ondemand":
            if field_name in config_values:
                return self._coerce_bool(config_values[field_name])
            legacy_value = config_values.get("vae_cache_preprocess")
            if legacy_value is not None:
                return not self._coerce_bool(legacy_value)
            return value

        elif field_name == "lora_alpha":
            # Always match lora_rank value
            return config_values.get("lora_rank", 16)

        return value

    # Format converters
    def _convert_to_template_format(
        self, field: Any, config_values: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert field to template format for HTML rendering."""
        # Get the field value with transformations
        field_value = config_values.get(field.name, field.default_value)
        field_value = self.apply_field_transformations(field.name, field_value, config_values)

        display_key = f"{field.name}__display"
        if display_key in config_values:
            field_value = config_values.get(display_key, field_value)

        resolved_value = config_values.get(f"{field.name}__resolved")
        additional_hint = config_values.get(f"{field.name}__hint")

        if field.name == "eval_dataset_id":
            if isinstance(field_value, (list, tuple)):
                field_value = next((item for item in field_value if isinstance(item, str) and item.strip()), None)

        eval_dataset_options = config_values.get("__eval_dataset_options__") or []
        has_eval_datasets = self._coerce_bool(config_values.get("__has_eval_datasets__")) or bool(eval_dataset_options)
        evaluation_type_active = self._coerce_bool(config_values.get("__evaluation_type_active__"))

        initial_value = field_value
        if field_value is None and field.field_type in {FieldType.TEXT, FieldType.TEXTAREA, FieldType.NUMBER}:
            initial_value = ""
        if field_value is None and field.field_type == FieldType.SELECT:
            initial_value = ""

        field_dict = {
            "id": field.name,
            "name": field.name,
            "label": field.ui_label,
            "type": field.field_type.value.lower(),
            "value": initial_value,
            "description": field.help_text,
            "order": getattr(field, "order", 0),
        }
        # Preserve CLI arg name for downstream consumers that still need it
        field_dict["arg_name"] = getattr(field, "arg_name", field.name)
        # Include location metadata for downstream grouping
        if hasattr(field, "tab") and field.tab:
            field_dict["tab"] = field.tab
        if hasattr(field, "section") and field.section:
            field_dict["section_id"] = field.section
            field_dict["_raw_section"] = field.section
        if hasattr(field, "subsection") and field.subsection:
            field_dict["subsection"] = field.subsection
            field_dict["_raw_subsection"] = field.subsection
        else:
            field_dict["_raw_subsection"] = None

        custom_component = getattr(field, "custom_component", None)
        if custom_component:
            field_dict["custom_component"] = custom_component

        checkbox_label = getattr(field, "checkbox_label", None)
        if checkbox_label:
            field_dict["checkbox_label"] = checkbox_label

        if resolved_value:
            field_dict["resolved_value"] = resolved_value
        if additional_hint:
            field_dict["additional_hint"] = additional_hint

        if field.name in self._TEXT_ENCODER_PRECISION_FIELDS:
            encoder_config = self._get_text_encoder_configuration(config_values)
            if encoder_config:
                encoder_keys = list(encoder_config.keys())
                try:
                    index = int(field.name.split("_")[2]) - 1
                except (ValueError, IndexError):
                    index = -1
                if 0 <= index < len(encoder_keys):
                    encoder_entry = encoder_config.get(encoder_keys[index], {})
                    encoder_name = encoder_entry.get("name") or encoder_keys[index].replace("_", " ").title()
                    field_dict["label"] = f"{encoder_name} Precision"
                    if encoder_entry.get("name"):
                        field_dict.setdefault(
                            "description",
                            f"Set quantisation precision for {encoder_entry['name']} text encoder.",
                        )
                    field_dict.setdefault(
                        "tooltip",
                        f"Quantisation precision applied to the {encoder_name} text encoder.",
                    )

        # Extra CSS classes
        extra_classes = []

        raw_config = options.get("raw_config") if isinstance(options, dict) else None

        def _append_hint(message: str) -> None:
            if not message:
                return
            existing_description = (field_dict.get("description") or "").strip()
            if existing_description:
                if message not in existing_description:
                    field_dict["description"] = f"{existing_description} {message}".strip()
            else:
                field_dict["description"] = message
            field_dict.setdefault("tooltip", message)

        # Add tooltip helpers
        if hasattr(field, "cmd_args_help") and field.cmd_args_help:
            field_dict["cmd_args_help"] = field.cmd_args_help
            field_dict["tooltip"] = field.cmd_args_help
        elif getattr(field, "tooltip", None):
            field_dict["tooltip"] = field.tooltip

        # Handle conditional display
        if hasattr(field, "conditional_on"):
            field_dict["conditional_on"] = field.conditional_on
            extra_classes.append("conditional-field")

        # Add min/max for number fields
        if field.field_type.value.upper() == "NUMBER":
            if hasattr(field, "min_value") and field.min_value is not None:
                field_dict["min"] = field.min_value
            if hasattr(field, "max_value") and field.max_value is not None:
                field_dict["max"] = field.max_value
            if hasattr(field, "step") and field.step is not None:
                field_dict["step"] = field.step

        # Add options for select fields
        field_type_upper = field.field_type.value.upper()

        if field_type_upper in ["SELECT", "MULTI_SELECT"]:
            choices = getattr(field, "choices", None) or []

            if getattr(field, "dynamic_choices", False):
                if field.name == "data_backend_config":
                    try:
                        from .dataset_service import build_data_backend_choices  # lazy import

                        dataset_choices = build_data_backend_choices()
                    except Exception as exc:  # pragma: no cover - defensive guard
                        logger.warning("Failed to build dataset choices for %s: %s", field.name, exc)
                        dataset_choices = []

                    field_dict["custom_component"] = "dataset_config_select"

                    selected_option = next(
                        (opt for opt in dataset_choices if opt.get("value") == field_value),
                        None,
                    )

                    external_selection_note: Optional[str] = None
                    if field_value and not selected_option:
                        display_value = str(field_value)
                        try:
                            value_path = Path(str(field_value)).expanduser()
                            display_value = value_path.as_posix()
                        except Exception:
                            pass

                        external_choice = {
                            "value": field_value,
                            "environment": "External",
                            "path": display_value,
                            "label": f"External | {display_value}",
                        }
                        dataset_choices = [external_choice] + dataset_choices
                        selected_option = external_choice
                        external_selection_note = (
                            "Selected dataset config is outside the configured config directories. "
                            "This is supported, but it will not appear in the managed list."
                        )

                    field_dict["options"] = dataset_choices

                    field_dict["selected_environment"] = (
                        selected_option.get("environment") if selected_option else "Select dataset"
                    )
                    field_dict["selected_path"] = selected_option.get("path") if selected_option else ""
                    field_dict["button_label"] = (
                        f"{field_dict['selected_environment']} | {field_dict['selected_path']}"
                        if selected_option
                        else "Select dataset configuration"
                    )

                    if external_selection_note:
                        existing_description = (field_dict.get("description") or "").strip()
                        field_dict["description"] = (
                            f"{existing_description} {external_selection_note}".strip()
                            if existing_description
                            else external_selection_note
                        )
                        if "field-external-selection" not in extra_classes:
                            extra_classes.append("field-external-selection")
                elif field.name == "eval_dataset_id":
                    normalized_options: List[Dict[str, str]] = []
                    seen_values: Set[str] = set()

                    for option in eval_dataset_options:
                        if not isinstance(option, dict):
                            continue
                        value = option.get("value")
                        label = option.get("label")
                        if not isinstance(value, str):
                            continue
                        value = value.strip()
                        if not value or value in seen_values:
                            continue
                        if not isinstance(label, str) or not label.strip():
                            label = value
                        normalized_options.append({"value": value, "label": label.strip()})
                        seen_values.add(value)

                    current_value = ""
                    if isinstance(field_value, str):
                        current_value = field_value.strip()
                    elif field_value is not None:
                        current_value = str(field_value).strip()

                    if current_value and current_value not in seen_values:
                        normalized_options.append({"value": current_value, "label": current_value})

                    field_dict["options"] = normalized_options
            elif choices:
                normalized_options = []
                for choice in choices:
                    if isinstance(choice, dict):
                        normalized_options.append(choice)
                    elif isinstance(choice, (tuple, list)) and len(choice) >= 2:
                        normalized_options.append({"value": choice[0], "label": choice[1]})
                    else:
                        normalized_options.append({"value": choice, "label": str(choice)})

                field_dict["options"] = normalized_options

        # Add placeholder
        if field_type_upper in ["TEXT", "TEXTAREA", "NUMBER"]:
            if hasattr(field, "placeholder") and field.placeholder:
                field_dict["placeholder"] = field.placeholder

        # Add flags
        if hasattr(field, "required"):
            field_dict["required"] = field.required
        if hasattr(field, "disabled"):
            field_dict["disabled"] = field.disabled

        if field.name in self._EVALUATION_FIELDS:
            disable_reason: Optional[str] = None

            if not has_eval_datasets:
                disable_reason = self._EVAL_DATASET_REQUIRED_HINT
            elif field.name in self._EVALUATION_DEPENDENT_FIELDS and not evaluation_type_active:
                disable_reason = self._EVAL_TYPE_REQUIRED_HINT

            if disable_reason:
                field_dict["disabled"] = True
                if "field-disabled" not in extra_classes:
                    extra_classes.append("field-disabled")

                existing_description = (field_dict.get("description") or "").strip()
                if disable_reason not in existing_description:
                    field_dict["description"] = (
                        f"{existing_description} {disable_reason}".strip() if existing_description else disable_reason
                    )

        if field.name in self._WEBUI_FORCED_VALUES:
            field_dict["disabled"] = True
            extra_classes.append("field-disabled")

            hint = self._WEBUI_FIELD_HINTS.get(field.name)
            if hint:
                existing_description = (field_dict.get("description") or "").strip()
                field_dict["description"] = f"{existing_description} {hint}".strip() if existing_description else hint
                field_dict.setdefault("tooltip", hint)

        if field.name == "data_backend_config":
            field_dict["col_class"] = "col-md-6"

        if field.name == "pretrained_model_name_or_path":
            if field_value is None or str(field_value).lower() == "none":
                field_dict["value"] = ""
                extra_classes.append("field-optional")
            _append_hint("Defaults to the selected model flavour at runtime.")

        if field.name in {"controlnet_model_name_or_path", "controlnet_custom_config"}:
            controlnet_flag = config_values.get("controlnet")
            if controlnet_flag is None and isinstance(raw_config, dict):
                controlnet_flag = raw_config.get("controlnet") or raw_config.get("--controlnet")
            controlnet_enabled = self._coerce_bool(controlnet_flag)
            if not controlnet_enabled:
                field_dict["disabled"] = True
                extra_classes.append("field-disabled")
                _append_hint("Enable ControlNet training to configure this field.")

        skip_guidance_fields = {
            "validation_guidance_skip_layers",
            "validation_guidance_skip_layers_start",
            "validation_guidance_skip_layers_stop",
            "validation_guidance_skip_scale",
        }
        if field.name in skip_guidance_fields:
            model_family = config_values.get("model_family")
            if not model_family and isinstance(raw_config, dict):
                model_family = raw_config.get("model_family") or raw_config.get("--model_family")
            allowed_families = {"auraflow", "sd3", "wan"}
            if not model_family or model_family not in allowed_families:
                field_dict["disabled"] = True
                extra_classes.append("field-disabled")
                _append_hint("Available when using Auraflow, SD3, or Wan model families.")

        field_dict["extra_classes"] = " ".join(extra_classes)

        return field_dict

    def build_template_tab(
        self,
        tab_name: str,
        config_values: Dict[str, Any],
        raw_config: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return template-ready fields and sections for a tab."""

        tab_fields = self.field_registry.get_fields_for_tab(tab_name)
        conversion_options = dict(options or {})
        conversion_options.setdefault("raw_config", raw_config or {})

        template_fields = self.convert_fields(
            tab_fields,
            FieldFormat.TEMPLATE,
            config_values,
            conversion_options,
        )

        arranged_fields, sections = self._apply_template_layout(tab_name, template_fields)
        self._apply_tab_specific_overrides(
            tab_name,
            arranged_fields,
            config_values,
            raw_config=conversion_options.get("raw_config") or {},
        )
        return arranged_fields, sections

    def _apply_template_layout(
        self, tab_name: str, fields: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Map fields into declarative sections for template rendering."""

        layout = self._TAB_SECTION_LAYOUTS.get(tab_name, ())
        layout_map = {entry.id: entry for entry in layout}
        used_layout_ids: Set[str] = set()
        fallback_sections: Dict[str, Dict[str, Any]] = {}

        for field in fields:
            raw_section = field.pop("_raw_section", None)
            raw_subsection = field.pop("_raw_subsection", None)

            matched_entry: Optional[SectionLayout] = None
            for entry in layout:
                targets = set()
                if entry.match_sections:
                    targets.update(entry.match_sections)
                if entry.match_section:
                    targets.add(entry.match_section)
                if not targets:
                    targets.add(entry.id)
                if raw_section not in targets:
                    continue
                if entry.match_subsections is None or raw_subsection in entry.match_subsections:
                    matched_entry = entry
                    break

            if matched_entry:
                field["section_id"] = matched_entry.id
                if matched_entry.parent:
                    field["parent_section"] = matched_entry.parent
                if matched_entry.subsection_override is not None:
                    if matched_entry.subsection_override == "":
                        field.pop("subsection", None)
                    else:
                        field["subsection"] = matched_entry.subsection_override
                field["_section_order"] = matched_entry.order
                used_layout_ids.add(matched_entry.id)
                continue

            fallback_id_parts = [raw_section or "uncategorized"]
            if raw_subsection not in (None, "", "general"):
                fallback_id_parts.append(raw_subsection)
            fallback_id = "__".join(fallback_id_parts)

            field["section_id"] = fallback_id
            field["_section_order"] = 10_000

            if raw_subsection not in (None, ""):
                field["subsection"] = raw_subsection

            if fallback_id not in fallback_sections:
                fallback_sections[fallback_id] = {
                    "id": fallback_id,
                    "title": fallback_id.replace("__", " ").replace("_", " ").title(),
                    "icon": "fas fa-sliders-h",
                    "description": None,
                    "advanced": False,
                    "template": None,
                    "empty_message": None,
                    "order": 50_000,
                }

        # Build ordered sections list
        sections: List[Dict[str, Any]] = []

        for entry in layout:
            if entry.id not in used_layout_ids and not entry.include_if_empty:
                continue
            sections.append(
                {
                    "id": entry.id,
                    "title": entry.title,
                    "icon": entry.icon,
                    "description": entry.description,
                    "advanced": entry.advanced,
                    "template": entry.template,
                    "empty_message": entry.empty_message,
                    "order": entry.order,
                }
            )

        if fallback_sections:
            sections.extend(fallback_sections.values())

        sections.sort(key=lambda item: item.get("order", 0))

        # Attach ids for quick lookup when pruning empty sections later
        section_ids = {section["id"] for section in sections}

        # Sort fields to keep section ordering consistent
        fields.sort(key=lambda item: (item.get("_section_order", 0), item.get("order", 0), item.get("label", "")))

        for field in fields:
            field.pop("_section_order", None)

        # Remove empty sections without templates
        populated_section_ids = {field.get("section_id") for field in fields}
        filtered_sections: List[Dict[str, Any]] = []
        for section in sections:
            layout_entry = layout_map.get(section["id"])
            include_when_empty = layout_entry.include_if_empty if layout_entry else False
            if section["id"] in populated_section_ids or include_when_empty:
                filtered_sections.append({key: value for key, value in section.items() if key != "order"})

        return fields, filtered_sections

    def _apply_tab_specific_overrides(
        self,
        tab_name: str,
        fields: List[Dict[str, Any]],
        config_values: Dict[str, Any],
        raw_config: Dict[str, Any],
    ) -> None:
        """Adjust template fields with tab-specific runtime rules."""

        if tab_name != "model":
            return

        model_type = str(config_values.get("model_type") or "full")
        is_lora_type = model_type == "lora"
        combined_config = dict(raw_config)
        combined_config.update(config_values)
        danger_mode_enabled = self._is_danger_mode_enabled(combined_config)
        lora_rank_value = config_values.get("lora_rank", "16")

        for field in fields:
            field_id = field.get("id")
            if field_id == "lora_alpha":
                if not is_lora_type:
                    field["disabled"] = True
                elif danger_mode_enabled:
                    field.pop("disabled", None)
                else:
                    field["value"] = lora_rank_value
                    field["disabled"] = True
            elif field_id == "prediction_type":
                field["disabled"] = not danger_mode_enabled
                extra_classes = field.get("extra_classes", "")
                flag = "danger-mode-target"
                field["extra_classes"] = f"{extra_classes} {flag}".strip()
            elif field_id in {"base_model_precision", "text_encoder_1_precision", "quantize_via"}:
                if is_lora_type:
                    field.pop("disabled", None)
                    continue

                field["disabled"] = True
                extra_classes = field.get("extra_classes", "")
                flag = "field-disabled"
                field["extra_classes"] = f"{extra_classes} {flag}".strip()

    def _get_default_model_path(self, config_values: Dict[str, Any]) -> Optional[str]:
        """Resolve the default model path for the selected family/flavour."""

        if not ModelRegistry:
            return None

        model_family = config_values.get("model_family")
        if not model_family:
            return None

        try:  # pragma: no cover - defensive import usage
            model_class = ModelRegistry.get(model_family)
        except Exception:
            model_class = None

        if not model_class:
            return None

        flavour = config_values.get("model_flavour") or getattr(model_class, "DEFAULT_MODEL_FLAVOUR", None)
        huggingface_paths = getattr(model_class, "HUGGINGFACE_PATHS", {})

        if not flavour or not huggingface_paths:
            return None

        return huggingface_paths.get(flavour)

    @staticmethod
    def _resolve_model_default_flavour(model_family: Optional[str]) -> Optional[str]:
        """Return the default flavour for a model family when available."""

        if not model_family:
            return None

        try:
            model_class = ModelRegistry.get(model_family)
        except Exception:
            return None

        default_flavour = getattr(model_class, "DEFAULT_MODEL_FLAVOUR", None)
        if default_flavour:
            return default_flavour

        huggingface_paths = getattr(model_class, "HUGGINGFACE_PATHS", None)
        if isinstance(huggingface_paths, dict) and huggingface_paths:
            try:
                return next(iter(huggingface_paths.keys()))
            except StopIteration:  # pragma: no cover - defensive guard
                return None

        return None

    def _convert_to_api_format(self, field: Any, config_values: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """Convert field to API response format."""
        field_value = config_values.get(field.name, field.default_value)

        return {
            "name": field.name,
            "value": field_value,
            "type": field.field_type.value.lower(),
            "label": field.ui_label,
            "description": field.help_text,
            "required": getattr(field, "required", False),
            "default": field.default_value,
            "validation_rules": getattr(field, "validation_rules", []),
        }

    def _convert_to_config_format(
        self, field: Any, config_values: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert field to configuration file format."""
        field_value = config_values.get(field.name, field.default_value)

        # Only include non-default values
        if options.get("include_defaults", False) or field_value != field.default_value:
            return {field.name: field_value}
        return {}

    def _convert_to_command_format(
        self, field: Any, config_values: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert field to command-line argument format."""
        field_value = config_values.get(field.name, field.default_value)

        # Skip if using default value
        if not options.get("include_defaults", False) and field_value == field.default_value:
            return {}

        # Format as command-line argument
        arg_name = field.arg_name
        if field.field_type.value == "BOOLEAN":
            if field_value:
                return {"arg": arg_name, "value": None}  # Flag argument
            else:
                return {}  # Don't include false boolean flags
        else:
            return {"arg": arg_name, "value": str(field_value)}

    def merge_field_values(
        self, base_config: Dict[str, Any], overrides: Dict[str, Any], validate: bool = True
    ) -> Dict[str, Any]:
        """Merge field values with proper type conversion.

        Args:
            base_config: Base configuration values
            overrides: Override values
            validate: Whether to validate merged values

        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()

        for field_name, override_value in overrides.items():
            field = self.field_registry.get_field(field_name)
            if field:
                field_type = (field.field_type.value or "").lower()

                try:
                    if field_type == "number":
                        if override_value is None:
                            converted_value = None
                        elif isinstance(override_value, (int, float)):
                            converted_value = override_value
                        elif isinstance(override_value, str):
                            stripped = override_value.strip()
                            if not stripped or stripped.lower() == "none":
                                converted_value = None
                            else:
                                if any(ch in stripped.lower() for ch in [".", "e"]):
                                    converted_value = float(stripped)
                                    if converted_value.is_integer():
                                        converted_value = int(converted_value)
                                else:
                                    converted_value = int(stripped)
                        else:
                            converted_value = None

                        override_value = converted_value

                    elif field_type == "checkbox":
                        if isinstance(override_value, str):
                            override_value = override_value.strip().lower() in {"true", "1", "yes", "on"}
                        else:
                            override_value = bool(override_value)

                    elif field_type == "multi_select" and isinstance(override_value, str):
                        override_value = [v.strip() for v in override_value.split(",") if v.strip()]

                except (ValueError, TypeError):
                    logger.warning(f"Failed to convert %s value: %s", field_name, override_value)
                    continue

            arg_name = getattr(field, "arg_name", None)
            legacy_key = f"--{field_name}" if not (field_name or "").startswith("--") else field_name

            if override_value is None:
                merged.pop(field_name, None)
                if arg_name:
                    merged.pop(arg_name, None)
                merged.pop(legacy_key, None)
            else:
                merged[field_name] = override_value
                if arg_name:
                    merged[arg_name] = override_value
                merged[legacy_key] = override_value

        return merged

    def prepare_tab_field_values(
        self, tab_name: str, config_data: Dict[str, Any], webui_defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare field values for a tab with webui defaults and special handling.

        This consolidates the logic previously duplicated across render_tab,
        TabFieldsDependency, and TabService.

        Args:
            tab_name: Name of the tab
            config_data: Current configuration data
            webui_defaults: WebUI default settings

        Returns:
            Dictionary of field values with appropriate defaults
        """
        tab_fields = self.field_registry.get_fields_for_tab(tab_name)
        config_values = {}

        danger_mode_enabled = self._is_danger_mode_enabled(config_data)
        lora_rank_value = self._coerce_int(
            config_data.get("lora_rank", config_data.get("--lora_rank", 16)),
            16,
        )

        selected_model_family = (
            config_data.get("model_family") or config_data.get("--model_family") or webui_defaults.get("model_family")
        )

        for field in tab_fields:
            # Special handling for output_dir in basic tab
            if field.name == "output_dir" and tab_name == "basic":
                if field.name in config_data:
                    config_values[field.name] = config_data[field.name]
                elif f"--{field.name}" in config_data:
                    config_values[field.name] = config_data[f"--{field.name}"]
                else:
                    config_values[field.name] = webui_defaults.get("output_dir", "")
            else:
                value = None

                # Prefer explicit config key, but fall back to legacy "--" prefix
                candidate_keys = [field.name, f"--{field.name}"]
                arg_name = getattr(field, "arg_name", "")
                if arg_name and arg_name not in candidate_keys:
                    candidate_keys.append(arg_name)

                for key in candidate_keys:
                    if key not in config_data:
                        continue

                    candidate_value = config_data[key]

                    if isinstance(candidate_value, str):
                        normalized = candidate_value.strip().lower()
                        if normalized == "not configured":
                            continue
                        if normalized == "none":
                            has_explicit_none_choice = any(
                                isinstance(choice, dict) and str(choice.get("value", "")).strip().lower() == "none"
                                for choice in (field.choices or [])
                            )
                            if not has_explicit_none_choice:
                                continue

                    if candidate_value not in (None, ""):
                        value = candidate_value
                        break

                if value is None:
                    value = field.default_value

                if field.name == "model_family" and value:
                    selected_model_family = value

                if field.name == "model_flavour" and value in (None, "", "default"):
                    default_flavour = self._resolve_model_default_flavour(selected_model_family)
                    if default_flavour:
                        config_values[f"{field.name}__resolved"] = default_flavour
                        config_values[f"{field.name}__hint"] = (
                            f"Using default flavour '{default_flavour}' for {selected_model_family or 'the selected family'}."
                        )

                if field.name == "lora_rank":
                    lora_rank_value = self._coerce_int(value, lora_rank_value)

                if isinstance(value, str) and value and value.lower().endswith(".json"):
                    value = self._normalize_json_field_value(value, webui_defaults.get("configs_dir"))

                if field.name in self._WEBUI_FORCED_VALUES:
                    forced_value = deepcopy(self._WEBUI_FORCED_VALUES[field.name])
                    value = forced_value

                    display_value = json.dumps(forced_value, sort_keys=True)
                    config_values[f"{field.name}__display"] = display_value

                    hint = self._WEBUI_FIELD_HINTS.get(field.name)
                    if hint:
                        config_values[f"{field.name}__hint"] = hint

                if field.name == "eval_dataset_id":
                    if isinstance(value, (list, tuple)):
                        value = next((item for item in value if isinstance(item, str) and item.strip()), None)
                    if isinstance(value, str):
                        trimmed = value.strip()
                        value = trimmed or None

                if field.name == "i_know_what_i_am_doing":
                    value = self._coerce_bool(value)
                elif field.name == "lora_alpha":
                    if not danger_mode_enabled:
                        value = lora_rank_value
                    elif value in (None, ""):
                        value = lora_rank_value

                config_values[field.name] = value
                arg_name = getattr(field, "arg_name", "")
                if arg_name and arg_name != field.name:
                    config_values[arg_name] = value
                legacy_key = f"--{field.name}"
                if legacy_key not in (field.name, arg_name):
                    config_values[legacy_key] = value

        # Add webui-specific values for basic tab
        if tab_name == "basic":
            configs_dir_value = (webui_defaults.get("configs_dir") if isinstance(webui_defaults, dict) else None) or ""
            config_values["configs_dir"] = configs_dir_value
            if configs_dir_value:
                config_values["--configs_dir"] = configs_dir_value
            config_values["job_id"] = config_data.get("job_id", "")

        if tab_name == "validation":
            eval_dataset_options = self._collect_eval_dataset_options(config_data, webui_defaults)
            config_values["__eval_dataset_options__"] = eval_dataset_options
            config_values["__has_eval_datasets__"] = bool(eval_dataset_options)

            evaluation_type_value = (
                config_values.get("evaluation_type")
                or config_values.get("--evaluation_type")
                or config_data.get("--evaluation_type")
            )
            evaluation_type_normalized = self._normalise_evaluation_type(evaluation_type_value)
            config_values["__evaluation_type_active__"] = bool(eval_dataset_options) and (
                evaluation_type_normalized not in {"", "none"}
            )

        return config_values

    @staticmethod
    def _normalize_json_field_value(value: str, configs_dir: Optional[str]) -> str:
        """Normalize JSON-backed config paths so selectors align with available choices."""

        if not value:
            return value

        try:
            normalized = normalize_dataset_config_value(value, configs_dir)
            if normalized:
                return normalized
        except Exception:
            pass

        if configs_dir:
            try:
                config_basename = Path(configs_dir).expanduser().name
                parts = Path(value).parts
                if parts and parts[0] == config_basename and len(parts) > 1:
                    return Path(*parts[1:]).as_posix()
            except Exception:
                return value

        return value
