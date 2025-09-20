import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class RuleType(Enum):
    DEFAULT = "default"
    REQUIRED = "required"
    MIN = "min"
    MAX = "max"
    CHOICES = "choices"
    OVERRIDE = "override"
    CUSTOM = "custom"
    COMBINATION = "combination"
    INCOMPATIBLE = "incompatible"


@dataclass
class ConfigRule:
    """Represents a configuration validation rule."""

    field_name: str
    rule_type: RuleType
    value: Any
    message: str
    example: Optional[str] = None
    error_level: str = "error"  # "error" or "warning"
    condition: Optional[Callable[[dict], bool]] = None  # For conditional rules
    suggestion: Optional[str] = None
    category: Optional[str] = None  # For grouping in docs (e.g., "performance", "memory")


@dataclass
class ValidationResult:
    """Result of validating a configuration rule."""

    passed: bool
    field: str
    message: str
    level: str = "error"  # "error" or "warning"
    suggestion: Optional[str] = None
    rule: Optional[ConfigRule] = None


@dataclass
class ConfigValidator:
    """Encapsulates a custom validation function with documentation."""

    func: Callable[[dict], List[ValidationResult]]
    doc: str
    examples: List[str] = field(default_factory=list)


class ConfigRegistry:
    """Central registry for configuration rules and validators."""

    _rules: Dict[str, List[ConfigRule]] = defaultdict(list)
    _validators: Dict[str, List[ConfigValidator]] = defaultdict(list)
    _documentation: Dict[str, str] = {}

    @classmethod
    def register_rule(cls, category: str, rule: ConfigRule) -> None:
        """Register a single configuration rule."""
        cls._rules[category].append(rule)
        logger.debug(f"Registered rule for {category}: {rule.field_name} ({rule.rule_type.value})")

    @classmethod
    def register_rules(cls, category: str, rules: List[ConfigRule]) -> None:
        """Register multiple configuration rules at once."""
        for rule in rules:
            cls.register_rule(category, rule)

    @classmethod
    def register_validator(
        cls, category: str, validator_func: Callable, doc: str, examples: Optional[List[str]] = None
    ) -> None:
        """Register a custom validation function with documentation."""
        validator = ConfigValidator(func=validator_func, doc=doc, examples=examples or [])
        cls._validators[category].append(validator)
        logger.debug(f"Registered validator for {category}: {doc[:50]}...")

    @classmethod
    def register_documentation(cls, category: str, doc: str) -> None:
        """Register category-level documentation."""
        cls._documentation[category] = doc

    @classmethod
    def get_rules(cls, category: str) -> List[ConfigRule]:
        """Get all rules for a specific category."""
        return cls._rules.get(category, [])

    @classmethod
    def get_validators(cls, category: str) -> List[ConfigValidator]:
        """Get all validators for a specific category."""
        return cls._validators.get(category, [])

    @classmethod
    def get_all_categories(cls) -> List[str]:
        """Get all registered categories."""
        categories = set(cls._rules.keys())
        categories.update(cls._validators.keys())
        return sorted(list(categories))

    @classmethod
    def get_documentation(cls, category: str) -> Optional[str]:
        """Get documentation for a category."""
        return cls._documentation.get(category)

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered rules and validators (mainly for testing)."""
        cls._rules.clear()
        cls._validators.clear()
        cls._documentation.clear()


# Convenience functions for common rule patterns
def make_default_rule(field_name: str, default_value: Any, message: str, example: Optional[str] = None) -> ConfigRule:
    """Create a default value rule."""
    return ConfigRule(
        field_name=field_name,
        rule_type=RuleType.DEFAULT,
        value=default_value,
        message=message,
        example=example,
        error_level="info",
    )


def make_required_rule(
    field_name: str, message: str, example: Optional[str] = None, suggestion: Optional[str] = None
) -> ConfigRule:
    """Create a required field rule."""
    return ConfigRule(
        field_name=field_name,
        rule_type=RuleType.REQUIRED,
        value=True,
        message=message,
        example=example,
        suggestion=suggestion,
    )


def make_choice_rule(field_name: str, choices: List[Any], message: str, example: Optional[str] = None) -> ConfigRule:
    """Create a choice validation rule."""
    return ConfigRule(field_name=field_name, rule_type=RuleType.CHOICES, value=choices, message=message, example=example)


def make_range_rule(
    field_name: str, rule_type: RuleType, value: Union[int, float], message: str, example: Optional[str] = None
) -> ConfigRule:
    """Create a min/max range rule."""
    if rule_type not in [RuleType.MIN, RuleType.MAX]:
        raise ValueError("rule_type must be RuleType.MIN or RuleType.MAX")

    return ConfigRule(field_name=field_name, rule_type=rule_type, value=value, message=message, example=example)


def make_override_rule(field_name: str, value: Any, message: str, example: Optional[str] = None) -> ConfigRule:
    """Create an override rule that forces a specific value."""
    return ConfigRule(
        field_name=field_name,
        rule_type=RuleType.OVERRIDE,
        value=value,
        message=message,
        example=example,
        error_level="warning",
    )
