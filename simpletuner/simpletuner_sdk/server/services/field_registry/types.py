"""Shared dataclasses and enums for the field registry."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FieldType(Enum):
    """UI field types exposed to the WebUI."""

    TEXT = "text"
    NUMBER = "number"
    SELECT = "select"
    CHECKBOX = "checkbox"
    TEXTAREA = "textarea"
    PASSWORD = "password"
    FILE = "file"
    MULTI_SELECT = "multi_select"


class ImportanceLevel(Enum):
    """Importance levels for progressive disclosure in the UI."""

    ESSENTIAL = "essential"
    IMPORTANT = "important"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"


class ValidationRuleType(Enum):
    """Supported validation rule types."""

    REQUIRED = "required"
    MIN = "min"
    MAX = "max"
    PATTERN = "pattern"
    CHOICES = "choices"
    CUSTOM = "custom"
    PATH_EXISTS = "path_exists"
    DIVISIBLE_BY = "divisible_by"


@dataclass
class ValidationRule:
    """Validation rule metadata."""

    rule_type: ValidationRuleType
    value: Any = None
    message: str = ""
    pattern: Optional[str] = None  # Used with PATTERN rules
    condition: Optional[Dict[str, Any]] = None  # Only validate when condition matches

    @property
    def type(self) -> ValidationRuleType:
        """Backward-compatible alias for legacy callers."""

        return self.rule_type


@dataclass
class FieldDependency:
    """Dependency rule describing when a field should change state."""

    field: str
    value: Any = None
    values: Optional[List[Any]] = None
    operator: str = "equals"  # equals, not_equals, in, not_in, greater_than, less_than
    action: str = "show"  # show, hide, enable, disable, set_value
    condition_met_value: Any = None
    condition_not_met_value: Any = None
    target_value: Any = None


@dataclass
class ConfigField:
    """Primary configuration field representation."""

    name: str
    arg_name: str
    ui_label: str
    field_type: FieldType
    tab: str
    section: str
    subsection: Optional[str] = None
    default_value: Any = None
    choices: Optional[List[Dict[str, Any]]] = None
    validation_rules: List[ValidationRule] = field(default_factory=list)
    dependencies: List[FieldDependency] = field(default_factory=list)
    help_text: str = ""
    tooltip: str = ""
    placeholder: str = ""
    importance: ImportanceLevel = ImportanceLevel.IMPORTANT
    model_specific: Optional[List[str]] = None
    platform_specific: Optional[List[str]] = None
    warning: Optional[str] = None
    group: Optional[str] = None
    order: int = 0
    dynamic_choices: bool = False
    cmd_args_help: Optional[str] = None
    step: Optional[float] = None
    custom_component: Optional[str] = None
    checkbox_label: Optional[str] = None
    webui_onboarding: bool = False
    webui_only: bool = False  # True if this field is WebUI-specific and should not be passed to the trainer
    disabled: bool = False
    aliases: Optional[List[str]] = None
