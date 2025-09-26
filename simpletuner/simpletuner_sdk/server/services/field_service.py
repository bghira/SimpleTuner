"""Service for field conversion and manipulation.

This service handles converting fields between different formats,
applying transformations, and managing field metadata.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from ..services.field_registry_wrapper import lazy_field_registry

try:  # pragma: no cover - optional import for UI hints
    from simpletuner.helpers.models.registry import ModelRegistry
except Exception:  # pragma: no cover - fall back when models unavailable
    ModelRegistry = None

logger = logging.getLogger(__name__)


class FieldFormat(str, Enum):
    """Supported field formats."""
    TEMPLATE = "template"  # Format for HTML templates
    API = "api"            # Format for API responses
    CONFIG = "config"      # Format for configuration files
    COMMAND = "command"    # Format for command-line arguments


class FieldService:
    """Service for field conversion and manipulation."""

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

    def _is_danger_mode_enabled(self, config_data: Dict[str, Any]) -> bool:
        """Determine whether dangerous overrides are enabled."""
        for key in ("i_know_what_i_am_doing", "--i_know_what_i_am_doing"):
            if key in config_data and self._coerce_bool(config_data[key]):
                return True
        return False

    def convert_field(
        self,
        field: Any,
        format: FieldFormat,
        config_values: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
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
        self,
        fields: List[Any],
        format: FieldFormat,
        config_values: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
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
        return [
            self.convert_field(field, format, config_values, options)
            for field in fields
        ]

    def get_fields_for_section(
        self,
        tab_name: str,
        section_name: str,
        format: FieldFormat,
        config_values: Dict[str, Any]
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
            if hasattr(field, 'section') and field.section == section_name:
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
        if hasattr(field, 'conditional_on'):
            dependencies.add(field.conditional_on)

        if hasattr(field, 'depends_on') and isinstance(field.depends_on, list):
            dependencies.update(field.depends_on)

        return dependencies

    def apply_field_transformations(
        self,
        field_name: str,
        value: Any,
        config_values: Dict[str, Any]
    ) -> Any:
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

        elif field_name == "lora_alpha":
            # Always match lora_rank value
            return config_values.get("lora_rank", 16)

        return value

    # Format converters
    def _convert_to_template_format(
        self,
        field: Any,
        config_values: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert field to template format for HTML rendering."""
        # Get the field value with transformations
        field_value = config_values.get(field.name, field.default_value)
        field_value = self.apply_field_transformations(field.name, field_value, config_values)

        field_dict = {
            "id": field.name,
            "name": field.arg_name,
            "label": field.ui_label,
            "type": field.field_type.value.lower(),
            "value": field_value,
            "description": field.help_text,
        }

        # Extra CSS classes
        extra_classes = []

        # Add tooltip helpers
        if hasattr(field, 'cmd_args_help') and field.cmd_args_help:
            field_dict["cmd_args_help"] = field.cmd_args_help
            field_dict["tooltip"] = field.cmd_args_help
        elif getattr(field, "tooltip", None):
            field_dict["tooltip"] = field.tooltip

        # Add section ID
        if hasattr(field, 'section') and field.section:
            field_dict["section_id"] = field.section

        # Handle conditional display
        if hasattr(field, 'conditional_on'):
            field_dict["conditional_on"] = field.conditional_on
            extra_classes.append("conditional-field")

        # Add min/max for number fields
        if field.field_type.value == "NUMBER":
            if hasattr(field, 'min_value') and field.min_value is not None:
                field_dict["min"] = field.min_value
            if hasattr(field, 'max_value') and field.max_value is not None:
                field_dict["max"] = field.max_value
            if hasattr(field, 'step') and field.step is not None:
                field_dict["step"] = field.step

        # Add options for select fields
        field_type_upper = field.field_type.value.upper()

        if field_type_upper in ["SELECT", "MULTI_SELECT"]:
            choices = getattr(field, "choices", None) or []

            if getattr(field, "dynamic_choices", False) and field.name == "data_backend_config":
                try:
                    from .dataset_service import build_data_backend_choices  # lazy import

                    dataset_choices = build_data_backend_choices()
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning("Failed to build dataset choices for %s: %s", field.name, exc)
                    dataset_choices = []

                field_dict["custom_component"] = "dataset_config_select"
                field_dict["options"] = dataset_choices

                selected_option = next(
                    (opt for opt in dataset_choices if opt.get("value") == field_value),
                    None,
                )

                field_dict["selected_environment"] = (
                    selected_option.get("environment") if selected_option else "Select dataset"
                )
                field_dict["selected_path"] = selected_option.get("path") if selected_option else ""
                field_dict["button_label"] = (
                    f"{field_dict['selected_environment']} | {field_dict['selected_path']}"
                    if selected_option
                    else "Select dataset configuration"
                )
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
        if field_type_upper in ["TEXT", "TEXTAREA"]:
            if hasattr(field, 'placeholder') and field.placeholder:
                field_dict["placeholder"] = field.placeholder

        # Add flags
        if hasattr(field, 'required'):
            field_dict["required"] = field.required
        if hasattr(field, 'disabled'):
            field_dict["disabled"] = field.disabled

        if field.name == "data_backend_config":
            field_dict["col_class"] = "col-md-6"

        if field.name == "pretrained_model_name_or_path":
            default_path = self._get_default_model_path(config_values)
            if field_value is None or str(field_value).lower() == "none":
                field_dict["value"] = ""
                extra_classes.append("field-optional")
            if default_path and default_path not in str(field_dict.get("placeholder", "")):
                field_dict["placeholder"] = field_dict.get("placeholder") or default_path
            if default_path:
                hint = f"Defaults to {default_path} based on the selected model flavour."
                if field_dict.get("description"):
                    if default_path not in field_dict["description"]:
                        field_dict["description"] = f"{field_dict['description']} {hint}"
                else:
                    field_dict["description"] = hint

        field_dict["extra_classes"] = " ".join(extra_classes)

        return field_dict

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

    def _convert_to_api_format(
        self,
        field: Any,
        config_values: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert field to API response format."""
        field_value = config_values.get(field.name, field.default_value)

        return {
            "name": field.name,
            "value": field_value,
            "type": field.field_type.value.lower(),
            "label": field.ui_label,
            "description": field.help_text,
            "required": getattr(field, 'required', False),
            "default": field.default_value,
            "validation_rules": getattr(field, 'validation_rules', [])
        }

    def _convert_to_config_format(
        self,
        field: Any,
        config_values: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert field to configuration file format."""
        field_value = config_values.get(field.name, field.default_value)

        # Only include non-default values
        if options.get("include_defaults", False) or field_value != field.default_value:
            return {field.name: field_value}
        return {}

    def _convert_to_command_format(
        self,
        field: Any,
        config_values: Dict[str, Any],
        options: Dict[str, Any]
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
        self,
        base_config: Dict[str, Any],
        overrides: Dict[str, Any],
        validate: bool = True
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
                # Apply type conversion based on field type
                try:
                    if field.field_type.value == "NUMBER":
                        override_value = float(override_value) if "." in str(override_value) else int(override_value)
                    elif field.field_type.value == "BOOLEAN":
                        override_value = str(override_value).lower() in ("true", "1", "yes", "on")
                    elif field.field_type.value == "MULTI_SELECT" and isinstance(override_value, str):
                        override_value = [v.strip() for v in override_value.split(",")]
                except (ValueError, TypeError):
                    logger.warning(f"Failed to convert {field_name} value: {override_value}")
                    continue

            merged[field_name] = override_value

        return merged

    def prepare_tab_field_values(
        self,
        tab_name: str,
        config_data: Dict[str, Any],
        webui_defaults: Dict[str, Any]
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
                arg_name = getattr(field, 'arg_name', '')
                if arg_name and arg_name not in candidate_keys:
                    candidate_keys.append(arg_name)

                for key in candidate_keys:
                    if key not in config_data:
                        continue

                    candidate_value = config_data[key]

                    if isinstance(candidate_value, str) and candidate_value.strip().lower() in {"none", "not configured"}:
                        continue

                    if candidate_value not in (None, ""):
                        value = candidate_value
                        break

                if value is None:
                    value = field.default_value

                if field.name == "lora_rank":
                    lora_rank_value = self._coerce_int(value, lora_rank_value)

                if field.name == "i_know_what_i_am_doing":
                    value = self._coerce_bool(value)
                elif field.name == "lora_alpha":
                    if not danger_mode_enabled:
                        value = lora_rank_value
                    elif value in (None, ""):
                        value = lora_rank_value

                config_values[field.name] = value
                arg_name = getattr(field, 'arg_name', '')
                if arg_name and arg_name != field.name:
                    config_values[arg_name] = value
                legacy_key = f"--{field.name}"
                if legacy_key not in (field.name, arg_name):
                    config_values[legacy_key] = value

        # Add webui-specific values for basic tab
        if tab_name == "basic":
            config_values["configs_dir"] = webui_defaults.get("configs_dir", "")
            config_values["job_id"] = config_data.get("job_id", "")

        return config_values
