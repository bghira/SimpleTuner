"""Service for centralizing field validation logic.

This service provides validation functionality using the field registry's
validation rules, eliminating the need for hardcoded validation in routes.
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from ..services.field_registry_wrapper import lazy_field_registry

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation messages."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


class ValidationMessage:
    """Represents a validation message."""

    def __init__(
        self,
        field: str,
        message: str,
        severity: ValidationSeverity,
        suggestion: Optional[str] = None
    ):
        self.field = field
        self.message = message
        self.severity = severity
        self.suggestion = suggestion

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value
        }
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result


class ValidationResult:
    """Result of validation operation."""

    def __init__(self):
        self.messages: List[ValidationMessage] = []
        self.vram_estimate: Optional[Dict[str, Any]] = None

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return not any(msg.severity == ValidationSeverity.ERROR for msg in self.messages)

    @property
    def error_count(self) -> int:
        """Count of error messages."""
        return sum(1 for msg in self.messages if msg.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warning messages."""
        return sum(1 for msg in self.messages if msg.severity == ValidationSeverity.WARNING)

    @property
    def info_count(self) -> int:
        """Count of info messages."""
        return sum(1 for msg in self.messages if msg.severity == ValidationSeverity.INFO)

    def add_error(self, field: str, message: str, suggestion: Optional[str] = None):
        """Add an error message."""
        self.messages.append(ValidationMessage(field, message, ValidationSeverity.ERROR, suggestion))

    def add_warning(self, field: str, message: str, suggestion: Optional[str] = None):
        """Add a warning message."""
        self.messages.append(ValidationMessage(field, message, ValidationSeverity.WARNING, suggestion))

    def add_info(self, field: str, message: str, suggestion: Optional[str] = None):
        """Add an info message."""
        self.messages.append(ValidationMessage(field, message, ValidationSeverity.INFO, suggestion))

    def add_success(self, field: str, message: str):
        """Add a success message."""
        self.messages.append(ValidationMessage(field, message, ValidationSeverity.SUCCESS))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "valid": self.is_valid,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "info": self.info_count,
            "messages": [msg.to_dict() for msg in self.messages]
        }
        if self.vram_estimate:
            result["vram_estimate"] = self.vram_estimate
        return result


class ValidationService:
    """Service for validating configuration fields using registry rules."""

    def __init__(self):
        """Initialize validation service."""
        self.field_registry = lazy_field_registry

    def validate_field(self, field_name: str, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a single field value.

        Args:
            field_name: Name of the field to validate
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        field = self.field_registry.get_field(field_name)
        if not field:
            # If field not in registry, do basic validation
            return self._validate_unknown_field(field_name, value)

        # Use field registry validation rules
        if hasattr(field, 'validation_rules') and field.validation_rules:
            for rule in field.validation_rules:
                is_valid, error_msg = self._apply_validation_rule(field, rule, value)
                if not is_valid:
                    return False, error_msg

        return True, None

    def validate_configuration(
        self,
        config: Dict[str, Any],
        validate_paths: bool = True,
        estimate_vram: bool = True
    ) -> ValidationResult:
        """Validate an entire configuration.

        Args:
            config: Configuration dictionary to validate
            validate_paths: Whether to validate file/directory paths
            estimate_vram: Whether to estimate VRAM usage

        Returns:
            ValidationResult containing all validation messages
        """
        result = ValidationResult()

        # Validate each field
        for field_name, value in config.items():
            is_valid, error_msg = self.validate_field(field_name, value)
            if not is_valid:
                result.add_error(field_name, error_msg)

        # Additional cross-field validation
        self._validate_cross_fields(config, result)

        # Path validation
        if validate_paths:
            self._validate_paths(config, result)

        # VRAM estimation
        if estimate_vram:
            result.vram_estimate = self._estimate_vram(config)

        # Add success message if all valid
        if result.is_valid and result.warning_count == 0:
            result.add_success("configuration", "All configuration values are valid")

        return result

    def get_field_validation_html(self, field_name: str, value: Any) -> str:
        """Get HTML error fragment for field validation (HTMX compatibility).

        Args:
            field_name: Name of the field
            value: Value to validate

        Returns:
            HTML error string (empty if valid)
        """
        is_valid, error_msg = self.validate_field(field_name, value)
        return "" if is_valid else error_msg

    def _validate_unknown_field(self, field_name: str, value: Any) -> Tuple[bool, Optional[str]]:
        """Basic validation for fields not in registry."""
        # Apply some common validations based on field name patterns
        if field_name.endswith("_dir") or field_name.endswith("_path"):
            if not value or not str(value).strip():
                return False, f"{field_name.replace('_', ' ').title()} is required"

        elif field_name.startswith("num_") or field_name.endswith("_steps"):
            try:
                num_val = int(value) if value else 0
                if num_val < 0:
                    return False, f"{field_name.replace('_', ' ').title()} must be non-negative"
            except (ValueError, TypeError):
                return False, f"{field_name.replace('_', ' ').title()} must be a valid number"

        elif "learning_rate" in field_name or field_name.startswith("lr_"):
            try:
                lr_val = float(value) if value else 0
                if lr_val <= 0:
                    return False, f"{field_name.replace('_', ' ').title()} must be greater than 0"
                elif lr_val > 1:
                    return False, f"{field_name.replace('_', ' ').title()} should typically be less than 1"
            except (ValueError, TypeError):
                return False, f"{field_name.replace('_', ' ').title()} must be a valid number"

        return True, None

    def _apply_validation_rule(self, field: Any, rule: Any, value: Any) -> Tuple[bool, Optional[str]]:
        """Apply a single validation rule to a value."""
        rule_type = rule.get('type') if isinstance(rule, dict) else getattr(rule, 'type', None)

        if rule_type == 'required':
            if not value or (isinstance(value, str) and not value.strip()):
                return False, f"{field.ui_label} is required"

        elif rule_type == 'min':
            min_val = rule.get('value') if isinstance(rule, dict) else getattr(rule, 'value', None)
            if min_val is not None:
                try:
                    if float(value) < float(min_val):
                        return False, f"{field.ui_label} must be at least {min_val}"
                except (ValueError, TypeError):
                    return False, f"{field.ui_label} must be a valid number"

        elif rule_type == 'max':
            max_val = rule.get('value') if isinstance(rule, dict) else getattr(rule, 'value', None)
            if max_val is not None:
                try:
                    if float(value) > float(max_val):
                        return False, f"{field.ui_label} must be at most {max_val}"
                except (ValueError, TypeError):
                    return False, f"{field.ui_label} must be a valid number"

        elif rule_type == 'pattern':
            pattern = rule.get('value') if isinstance(rule, dict) else getattr(rule, 'value', None)
            if pattern and not re.match(pattern, str(value)):
                return False, f"{field.ui_label} format is invalid"

        elif rule_type == 'choices':
            choices = rule.get('value') if isinstance(rule, dict) else getattr(rule, 'value', None)
            if choices and value not in choices:
                return False, f"{field.ui_label} must be one of: {', '.join(str(c) for c in choices)}"

        elif rule_type == 'path_exists':
            if validate_paths and value:
                path = Path(value)
                if not path.exists():
                    return False, f"{field.ui_label} path does not exist: {value}"

        elif rule_type == 'divisible_by':
            divisor = rule.get('value') if isinstance(rule, dict) else getattr(rule, 'value', None)
            if divisor:
                try:
                    if int(value) % int(divisor) != 0:
                        return False, f"{field.ui_label} must be divisible by {divisor}"
                except (ValueError, TypeError):
                    return False, f"{field.ui_label} must be a valid integer"

        return True, None

    def _validate_cross_fields(self, config: Dict[str, Any], result: ValidationResult):
        """Validate relationships between fields."""
        # Example: num_train_epochs and max_train_steps
        num_epochs = config.get("num_train_epochs", 0)
        max_steps = config.get("max_train_steps", 0)

        if num_epochs and max_steps:
            result.add_warning(
                "training",
                "Both num_train_epochs and max_train_steps are set. max_train_steps will take precedence.",
                "Consider using only one of these parameters"
            )

        # Example: Model-specific validations
        model_type = config.get("model_type", "").lower()
        if model_type == "lora" and not config.get("lora_rank"):
            result.add_error("lora_rank", "LoRA rank is required when using LoRA model type")

        # Learning rate scheduler validations
        scheduler = config.get("lr_scheduler", "")
        if scheduler == "polynomial" and not config.get("lr_scheduler_polynomial_power"):
            result.add_warning(
                "lr_scheduler_polynomial_power",
                "Polynomial power not specified, defaulting to 1.0"
            )

    def _validate_paths(self, config: Dict[str, Any], result: ValidationResult):
        """Validate file and directory paths in configuration."""
        path_fields = [
            ("pretrained_model_name_or_path", "Model path"),
            ("output_dir", "Output directory"),
            ("logging_dir", "Logging directory"),
            ("resume_from_checkpoint", "Checkpoint path")
        ]

        for field_name, display_name in path_fields:
            path_value = config.get(field_name)
            if path_value and not path_value.startswith("http"):
                # Skip validation for URLs
                path = Path(path_value)
                if field_name.endswith("_dir"):
                    # Directory should exist or be creatable
                    if not path.exists():
                        try:
                            path.mkdir(parents=True, exist_ok=True)
                            result.add_info(field_name, f"{display_name} will be created: {path_value}")
                        except Exception as e:
                            result.add_error(field_name, f"Cannot create {display_name}: {str(e)}")
                else:
                    # File should exist
                    if not path.exists():
                        result.add_warning(field_name, f"{display_name} not found: {path_value}")

    def _estimate_vram(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate VRAM usage based on configuration."""
        # Simple estimation logic - can be enhanced
        model_type = config.get("model_type", "full")
        batch_size = int(config.get("train_batch_size", 1))
        gradient_accumulation = int(config.get("gradient_accumulation_steps", 1))
        resolution = int(config.get("resolution", 512))

        # Base estimates (in GB)
        base_vram = {
            "full": 24,
            "lora": 12,
            "dora": 16
        }.get(model_type, 24)

        # Adjust for batch size and resolution
        effective_batch = batch_size * gradient_accumulation
        resolution_factor = (resolution / 512) ** 2

        estimated_vram = base_vram * effective_batch * resolution_factor

        return {
            "estimated_gb": round(estimated_vram, 1),
            "model_type": model_type,
            "batch_size": batch_size,
            "resolution": resolution,
            "notes": [
                f"Base requirement for {model_type}: {base_vram}GB",
                f"Effective batch size: {effective_batch}",
                f"Resolution scaling: {resolution_factor:.2f}x"
            ]
        }