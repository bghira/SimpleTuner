import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from simpletuner.helpers.configuration.registry import ConfigRegistry, ConfigRule, RuleType, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ConfigValidationEngine:
    """Engine for validating configurations against registered rules."""

    def validate_config(self, config: dict, model_family: Optional[str] = None) -> List[ValidationResult]:
        """
        Validate a configuration against all applicable rules.

        Args:
            config: Configuration dictionary to validate
            model_family: Optional model family for model-specific validation

        Returns:
            List of ValidationResult objects
        """
        results = []

        # Always validate base/common rules
        results.extend(self._validate_category(config, "base"))

        # Validate model-specific rules if model_family is provided
        if model_family:
            results.extend(self._validate_category(config, model_family))

        # Validate dataloader configuration if present
        if config.get("data_backend_config") or config.get("datasets"):
            results.extend(self._validate_category(config, "dataloader"))

        # Run custom validators
        results.extend(self._run_custom_validators(config, model_family))

        # Sort results by severity (errors first, then warnings)
        results.sort(key=lambda r: (0 if r.level == "error" else 1, r.field))

        return results

    def _validate_category(self, config: dict, category: str) -> List[ValidationResult]:
        """Validate config against rules for a specific category."""
        results = []
        rules = ConfigRegistry.get_rules(category)

        for rule in rules:
            # Check if rule applies (conditional rules)
            if rule.condition and not rule.condition(config):
                continue

            result = self._validate_rule(config, rule)
            if result and not result.passed:
                results.append(result)

        return results

    def _validate_rule(self, config: dict, rule: ConfigRule) -> Optional[ValidationResult]:
        """Validate a single rule against the configuration."""
        field_value = self._get_nested_value(config, rule.field_name)

        if rule.rule_type == RuleType.REQUIRED:
            if field_value is None:
                return ValidationResult(
                    passed=False,
                    field=rule.field_name,
                    message=rule.message,
                    level=rule.error_level,
                    suggestion=rule.suggestion,
                    rule=rule,
                )

        elif rule.rule_type == RuleType.DEFAULT:
            # Defaults don't fail validation, they're informational
            if field_value is None:
                logger.debug(f"Using default value for {rule.field_name}: {rule.value}")
            return None

        elif rule.rule_type == RuleType.MIN:
            if field_value is not None and field_value < rule.value:
                return ValidationResult(
                    passed=False,
                    field=rule.field_name,
                    message=rule.message,
                    level=rule.error_level,
                    suggestion=f"Set {rule.field_name} to at least {rule.value}",
                    rule=rule,
                )

        elif rule.rule_type == RuleType.MAX:
            if field_value is not None and field_value > rule.value:
                return ValidationResult(
                    passed=False,
                    field=rule.field_name,
                    message=rule.message,
                    level=rule.error_level,
                    suggestion=f"Set {rule.field_name} to at most {rule.value}",
                    rule=rule,
                )

        elif rule.rule_type == RuleType.CHOICES:
            if field_value is not None and field_value not in rule.value:
                return ValidationResult(
                    passed=False,
                    field=rule.field_name,
                    message=rule.message,
                    level=rule.error_level,
                    suggestion=f"Choose one of: {', '.join(map(str, rule.value))}",
                    rule=rule,
                )

        elif rule.rule_type == RuleType.OVERRIDE:
            # Override rules are informational - they indicate a value will be changed
            if field_value != rule.value:
                logger.info(f"Overriding {rule.field_name} from {field_value} to {rule.value}: {rule.message}")
            return None

        elif rule.rule_type == RuleType.COMBINATION:
            # Special handling for combination rules (e.g., dataloader requirements)
            return self._validate_combination_rule(config, rule)

        elif rule.rule_type == RuleType.INCOMPATIBLE:
            # Check if incompatible fields are present together
            return self._validate_incompatible_rule(config, rule)

        return None

    def _validate_combination_rule(self, config: dict, rule: ConfigRule) -> Optional[ValidationResult]:
        """Validate combination rules (e.g., must have both X and Y)."""
        if rule.field_name == "datasets" and isinstance(rule.value, dict):
            # Special handling for dataset requirements
            datasets = config.get("datasets", [])
            if isinstance(datasets, list):
                dataset_types = {}
                for dataset in datasets:
                    dtype = dataset.get("dataset_type", "image")  # Default to 'image'
                    dataset_types[dtype] = dataset_types.get(dtype, 0) + 1

                for required_type, min_count in rule.value.items():
                    if dataset_types.get(required_type, 0) < min_count:
                        return ValidationResult(
                            passed=False,
                            field=rule.field_name,
                            message=rule.message,
                            level=rule.error_level,
                            suggestion=rule.suggestion,
                            rule=rule,
                        )

        return None

    def _validate_incompatible_rule(self, config: dict, rule: ConfigRule) -> Optional[ValidationResult]:
        """Validate that incompatible options aren't used together."""
        if isinstance(rule.value, dict):
            field1, field2 = list(rule.value.items())[0]
            value1 = self._get_nested_value(config, field1[0])
            value2 = self._get_nested_value(config, field2[0])

            if value1 == field1[1] and value2 == field2[1]:
                return ValidationResult(
                    passed=False,
                    field=rule.field_name,
                    message=rule.message,
                    level=rule.error_level,
                    suggestion=rule.suggestion,
                    rule=rule,
                )

        return None

    def _run_custom_validators(self, config: dict, model_family: Optional[str]) -> List[ValidationResult]:
        """Run custom validation functions."""
        results = []

        # Run base custom validators
        for validator in ConfigRegistry.get_validators("base"):
            try:
                results.extend(validator.func(config))
            except Exception as e:
                logger.error(f"Error running base validator: {e}")

        # Run model-specific custom validators
        if model_family:
            for validator in ConfigRegistry.get_validators(model_family):
                try:
                    results.extend(validator.func(config))
                except Exception as e:
                    logger.error(f"Error running {model_family} validator: {e}")

        return results

    def _get_nested_value(self, config: dict, field_path: str) -> Any:
        """Get a value from config, supporting nested paths like 'a.b.c'."""
        parts = field_path.split(".")
        value = config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value

    def get_errors(self, results: List[ValidationResult]) -> List[ValidationResult]:
        """Filter validation results to get only errors."""
        return [r for r in results if r.level == "error" and not r.passed]

    def get_warnings(self, results: List[ValidationResult]) -> List[ValidationResult]:
        """Filter validation results to get only warnings."""
        return [r for r in results if r.level == "warning" and not r.passed]

    def format_results(self, results: List[ValidationResult]) -> str:
        """Format validation results as a human-readable string."""
        if not results:
            return "Configuration validation passed!"

        errors = self.get_errors(results)
        warnings = self.get_warnings(results)

        output = []

        if errors:
            output.append(f"Configuration Errors ({len(errors)}):")
            for error in errors:
                output.append(f"  ✗ {error.field}: {error.message}")
                if error.suggestion:
                    output.append(f"    → {error.suggestion}")
                if error.rule and error.rule.example:
                    output.append(f"    Example: {error.rule.example}")

        if warnings:
            if output:
                output.append("")
            output.append(f"Configuration Warnings ({len(warnings)}):")
            for warning in warnings:
                output.append(f"  ⚠ {warning.field}: {warning.message}")
                if warning.suggestion:
                    output.append(f"    → {warning.suggestion}")

        return "\n".join(output)
