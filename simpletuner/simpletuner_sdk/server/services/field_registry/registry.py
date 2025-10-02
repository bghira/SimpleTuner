"""Field Registry for SimpleTuner configuration parameters."""

import logging
import math
from typing import Any, Dict, List, Optional

from .sections import register_all_sections
from .types import ConfigField, FieldDependency, FieldType, ImportanceLevel, ValidationRule, ValidationRuleType

logger = logging.getLogger(__name__)

try:
    from ..arg_parser_integration import arg_parser_integration

    logger.debug("Successfully imported arg_parser_integration")
except ImportError as exc:  # pragma: no cover - defensive fallback
    logger.error("Failed to import arg_parser_integration: %s", exc)
    arg_parser_integration = None


class FieldRegistry:
    """Central registry for all configuration fields."""

    def __init__(self):
        self._fields: Dict[str, ConfigField] = {}
        self._dependencies_map: Dict[str, List[str]] = {}  # field -> dependent fields
        logger.debug("FieldRegistry.__init__ called")
        self._initialize_fields()
        logger.debug(f"FieldRegistry initialized with {len(self._fields)} fields")

    def _initialize_fields(self):
        """Initialize all configuration fields."""
        logger.debug("FieldRegistry._initialize_fields called")
        register_all_sections(self)

    def _add_field(self, field: ConfigField):
        """Add a field to the registry and update dependency maps."""
        # Auto-populate help text from cmd_args.py if not provided
        if field.arg_name and arg_parser_integration:
            arg_help = arg_parser_integration.get_argument_help(field.arg_name)
            if arg_help:
                # Store cmd_args help separately for detailed tooltip
                field.cmd_args_help = arg_parser_integration.format_help_for_ui(arg_help)

                # Use cmd_args help as primary help text if not set
                if not field.help_text:
                    field.help_text = arg_help

        if field.field_type == FieldType.NUMBER and field.step is None:
            auto_step = self._compute_default_step(field.default_value)
            if auto_step is not None:
                field.step = auto_step

        self._fields[field.name] = field

        # Update dependency map
        for dep in field.dependencies:
            if dep.field not in self._dependencies_map:
                self._dependencies_map[dep.field] = []
            self._dependencies_map[dep.field].append(field.name)

    @staticmethod
    def _compute_default_step(default_value: Any) -> Optional[float]:
        """Derive a sensible numeric step based on the provided default value."""

        if default_value in (None, ""):
            return None

        if isinstance(default_value, bool):
            return None

        if isinstance(default_value, int):
            return 1.0

        if isinstance(default_value, float):
            if default_value == 0:
                return 1.0
            magnitude = math.pow(10, math.floor(math.log10(abs(default_value))))
            return float(magnitude)

        return None

    def get_field(self, field_name: str) -> Optional[ConfigField]:
        """Get a specific field by name."""
        return self._fields.get(field_name)

    def get_fields_for_tab(self, tab: str, context: Optional[Dict[str, Any]] = None) -> List[ConfigField]:
        """Get all fields for a specific tab, filtered by context."""
        fields = [f for f in self._fields.values() if f.tab == tab]

        if context:
            # Filter by dependencies
            fields = [f for f in fields if self._check_dependencies(f, context)]

            # Filter by model-specific
            model_family = context.get("model_family")
            if model_family:
                fields = [f for f in fields if not f.model_specific or model_family in f.model_specific]

            # Filter by platform-specific
            platform = context.get("platform", "cuda")
            fields = [f for f in fields if not f.platform_specific or platform in f.platform_specific]

        # Sort by section, subsection, and order
        fields.sort(key=lambda f: (f.section, f.subsection or "", f.order))
        return fields

    def get_fields_by_section(self, tab: str, section: str, context: Optional[Dict[str, Any]] = None) -> List[ConfigField]:
        """Get fields for a specific section within a tab."""
        fields = self.get_fields_for_tab(tab, context)
        return [f for f in fields if f.section == section]

    def _check_dependencies(self, field: ConfigField, context: Dict[str, Any]) -> bool:
        """Check if field dependencies are satisfied."""
        for dep in field.dependencies:
            dep_value = context.get(dep.field)

            if dep.operator == "equals":
                if dep_value != dep.value:
                    return False
            elif dep.operator == "not_equals":
                if dep_value == dep.value:
                    return False
            elif dep.operator == "in":
                if dep_value not in (dep.values or []):
                    return False
            elif dep.operator == "not_in":
                if dep_value in (dep.values or []):
                    return False
            elif dep.operator == "greater_than":
                if not dep_value or dep_value <= dep.value:
                    return False
            elif dep.operator == "less_than":
                if not dep_value or dep_value >= dep.value:
                    return False

        return True

    def get_dependent_fields(self, field_name: str) -> List[str]:
        """Get fields that depend on the given field."""
        return self._dependencies_map.get(field_name, [])

    def validate_field_value(self, field_name: str, value: Any, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Validate a field value and return error messages."""
        field = self.get_field(field_name)
        if not field:
            return [f"Unknown field: {field_name}"]

        errors = []

        for rule in field.validation_rules:
            # Check if rule applies based on condition
            if rule.condition and context:
                if not self._check_rule_condition(rule.condition, context):
                    continue

            # Apply validation rule
            if rule.rule_type == ValidationRuleType.REQUIRED:
                if value is None or (isinstance(value, str) and not value.strip()):
                    errors.append(rule.message or f"{field.ui_label} is required")

            elif rule.rule_type == ValidationRuleType.MIN:
                if value is not None and value < rule.value:
                    errors.append(rule.message or f"{field.ui_label} must be at least {rule.value}")

            elif rule.rule_type == ValidationRuleType.MAX:
                if value is not None and value > rule.value:
                    errors.append(rule.message or f"{field.ui_label} must be at most {rule.value}")

            elif rule.rule_type == ValidationRuleType.CHOICES:
                if value is not None and value not in rule.value:
                    errors.append(rule.message or f"{field.ui_label} must be one of: {', '.join(map(str, rule.value))}")

            elif rule.rule_type == ValidationRuleType.DIVISIBLE_BY:
                if value is not None and value % rule.value != 0:
                    errors.append(rule.message or f"{field.ui_label} must be divisible by {rule.value}")

            elif rule.rule_type == ValidationRuleType.PATTERN:
                import re

                if value is not None and not re.match(rule.value, str(value)):
                    errors.append(rule.message or f"{field.ui_label} has invalid format")

        return errors

    def _check_rule_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if a validation rule condition is met."""
        for dependency_field, expected in condition.items():
            if context.get(dependency_field) != expected:
                return False
        return True

    def get_sections_for_tab(self, tab: str) -> List[Dict[str, Any]]:
        """Get unique sections for a tab with metadata."""
        sections = {}
        fields = [f for f in self._fields.values() if f.tab == tab]

        for cfg_field in fields:
            if cfg_field.section not in sections:
                sections[cfg_field.section] = {
                    "id": cfg_field.section,
                    "title": cfg_field.section.replace("_", " ").title(),
                    "subsections": set(),
                    "empty_message": None,
                }
            if cfg_field.subsection:
                sections[cfg_field.section]["subsections"].add(cfg_field.subsection)

        # Convert sets to lists
        for section_id, section in sections.items():
            section["subsections"] = sorted(list(section["subsections"]))
            if section_id == "text_encoder_training":
                section["empty_message"] = "This model does not support text encoder training."

        return list(sections.values())

    def export_field_metadata(self) -> Dict[str, Any]:
        """Export all field metadata for frontend consumption."""
        return {
            "fields": {
                name: {
                    "name": f.name,
                    "arg_name": f.arg_name,
                    "ui_label": f.ui_label,
                    "field_type": f.field_type.value,
                    "tab": f.tab,
                    "section": f.section,
                    "subsection": f.subsection,
                    "default_value": f.default_value,
                    "choices": f.choices,
                    "dependencies": [
                        {"field": d.field, "value": d.value, "values": d.values, "operator": d.operator}
                        for d in f.dependencies
                    ],
                    "help_text": f.help_text,
                    "tooltip": f.tooltip,
                    "placeholder": f.placeholder,
                    "importance": f.importance.value,
                    "model_specific": f.model_specific,
                    "platform_specific": f.platform_specific,
                    "warning": f.warning,
                    "group": f.group,
                    "order": f.order,
                }
                for name, f in self._fields.items()
            },
            "dependencies_map": self._dependencies_map,
            "tabs": self._get_tab_structure(),
        }

    def _get_tab_structure(self) -> Dict[str, Any]:
        """Get the structure of all tabs and sections."""
        tabs = {}

        for cfg_field in self._fields.values():
            if cfg_field.tab not in tabs:
                tabs[cfg_field.tab] = {"sections": {}}

            if cfg_field.section not in tabs[cfg_field.tab]["sections"]:
                tabs[cfg_field.tab]["sections"][cfg_field.section] = {
                    "title": cfg_field.section.replace("_", " ").title(),
                    "subsections": set(),
                    "field_count": 0,
                }

            tabs[cfg_field.tab]["sections"][cfg_field.section]["field_count"] += 1
            if cfg_field.subsection:
                tabs[cfg_field.tab]["sections"][cfg_field.section]["subsections"].add(cfg_field.subsection)

        # Convert sets to lists
        for tab in tabs.values():
            for section in tab["sections"].values():
                section["subsections"] = sorted(list(section["subsections"]))

        return tabs

    def get_webui_onboarding_fields(self) -> List[ConfigField]:
        """Return fields that should be treated as WebUI onboarding state only."""

        return [field for field in self._fields.values() if getattr(field, "webui_onboarding", False)]


# Create a singleton instance
field_registry = FieldRegistry()
