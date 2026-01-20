"""Service for centralizing field validation logic.

This service provides validation functionality using the field registry's
validation rules, eliminating the need for hardcoded validation in routes.
"""

from __future__ import annotations

import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from simpletuner.helpers.configuration.cli_utils import normalize_lr_scheduler_value
from simpletuner.helpers.training.attention_backend import is_sageattention_available, xformers_compute_capability_error

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

    def __init__(self, field: str, message: str, severity: ValidationSeverity, suggestion: Optional[str] = None):
        self.field = field
        self.message = message
        self.severity = severity
        self.suggestion = suggestion

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {"field": self.field, "message": self.message, "severity": self.severity.value}
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
            "messages": [msg.to_dict() for msg in self.messages],
        }
        if self.vram_estimate:
            result["vram_estimate"] = self.vram_estimate
        return result


class ValidationService:
    """Service for validating configuration fields using registry rules."""

    def __init__(self):
        """Initialize validation service."""
        self.field_registry = lazy_field_registry

    def validate_field(
        self,
        field_name: str,
        value: Any,
        *,
        config: Optional[Dict[str, Any]] = None,
        validate_paths: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Validate a single field value.

        Args:
            field_name: Name of the field to validate
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        field = self.field_registry.get_field(field_name)
        if not field:
            normalized_field = self._normalize_field_name(field_name)
            if normalized_field != field_name:
                field = self.field_registry.get_field(normalized_field)
        if not field:
            # If field not in registry, do basic validation
            return self._validate_unknown_field(field_name, value)

        # Use field registry validation rules
        if hasattr(field, "validation_rules") and field.validation_rules:
            for rule in field.validation_rules:
                is_valid, error_msg = self._apply_validation_rule(field, rule, value, validate_paths)
                if not is_valid:
                    return False, error_msg

        # If additional configuration context is provided, run cross-field rules and
        # surface any errors that target this specific field.
        if config is not None:
            working_config = dict(config)
            normalized_field = self._normalize_field_name(field_name)
            working_config[normalized_field] = value
            working_config[field_name] = value
            prefixed_name = field_name if field_name.startswith("--") else f"--{field_name}"
            working_config[prefixed_name] = value
            cross_result = ValidationResult()
            self._validate_cross_fields(working_config, cross_result)

            for message in cross_result.messages:
                message_field_normalized = self._normalize_field_name(message.field)
                if message.severity == ValidationSeverity.ERROR and message_field_normalized == normalized_field:
                    return False, message.message

        return True, None

    def validate_configuration(
        self, config: Dict[str, Any], validate_paths: bool = True, estimate_vram: bool = True
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
            is_valid, error_msg = self.validate_field(
                field_name,
                value,
                config=None,
                validate_paths=validate_paths,
            )
            if not is_valid:
                result.add_error(self._normalize_field_name(field_name), error_msg)

        # Additional cross-field validation
        self._validate_cross_fields(config, result)

        # Path validation
        if validate_paths:
            self._validate_paths(config, result)

        # VRAM estimation
        if estimate_vram:
            result.vram_estimate = self._estimate_vram(config)

        # Add success message if all valid
        if result.is_valid:
            result.add_success("configuration", "Configuration is valid and ready for training")

        return result

    def get_field_validation_html(
        self,
        field_name: str,
        value: Any,
        *,
        config: Optional[Dict[str, Any]] = None,
        validate_paths: bool = False,
    ) -> str:
        """Get HTML error fragment for field validation (HTMX compatibility).

        Args:
            field_name: Name of the field
            value: Value to validate

        Returns:
            HTML error string (empty if valid)
        """
        is_valid, error_msg = self.validate_field(
            field_name,
            value,
            config=config,
            validate_paths=validate_paths,
        )
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

    def _apply_validation_rule(
        self,
        field: Any,
        rule: Any,
        value: Any,
        validate_paths: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Apply a single validation rule to a value."""
        rule_type = rule.get("type") if isinstance(rule, dict) else getattr(rule, "type", None)

        if rule_type == "required":
            if not value or (isinstance(value, str) and not value.strip()):
                return False, f"{field.ui_label} is required"

        elif rule_type == "min":
            min_val = rule.get("value") if isinstance(rule, dict) else getattr(rule, "value", None)
            if min_val is not None:
                try:
                    if float(value) < float(min_val):
                        return False, f"{field.ui_label} must be at least {min_val}"
                except (ValueError, TypeError):
                    return False, f"{field.ui_label} must be a valid number"

        elif rule_type == "max":
            max_val = rule.get("value") if isinstance(rule, dict) else getattr(rule, "value", None)
            if max_val is not None:
                try:
                    if float(value) > float(max_val):
                        return False, f"{field.ui_label} must be at most {max_val}"
                except (ValueError, TypeError):
                    return False, f"{field.ui_label} must be a valid number"

        elif rule_type == "pattern":
            pattern = rule.get("value") if isinstance(rule, dict) else getattr(rule, "value", None)
            if pattern and not re.match(pattern, str(value)):
                return False, f"{field.ui_label} format is invalid"

        elif rule_type == "choices":
            choices = rule.get("value") if isinstance(rule, dict) else getattr(rule, "value", None)
            if choices and value not in choices:
                return False, f"{field.ui_label} must be one of: {', '.join(str(c) for c in choices)}"

        elif rule_type == "path_exists":
            if validate_paths and value:
                path = Path(value)
                if not path.exists():
                    return False, f"{field.ui_label} path does not exist: {value}"

        elif rule_type == "divisible_by":
            divisor = rule.get("value") if isinstance(rule, dict) else getattr(rule, "value", None)
            if divisor:
                try:
                    if int(value) % int(divisor) != 0:
                        return False, f"{field.ui_label} must be divisible by {divisor}"
                except (ValueError, TypeError):
                    return False, f"{field.ui_label} must be a valid integer"

        return True, None

    def _validate_cross_fields(self, config: Dict[str, Any], result: ValidationResult):
        """Validate relationships between fields."""
        num_epochs_raw = self._get_config_value(config, "num_train_epochs") or 0
        max_steps_raw = self._get_config_value(config, "max_train_steps") or 0

        def _to_int(value: Any) -> Tuple[Optional[int], Optional[str]]:
            if value is None or value == "":
                return 0, None
            try:
                return int(value), None
            except (ValueError, TypeError):
                return None, "must be a whole number"

        num_epochs, epochs_error = _to_int(num_epochs_raw)
        max_steps, steps_error = _to_int(max_steps_raw)

        if epochs_error:
            result.add_error("num_train_epochs", f"num_train_epochs {epochs_error}.")
        if steps_error:
            result.add_error("max_train_steps", f"max_train_steps {steps_error}.")

        if epochs_error is None and steps_error is None:
            if (num_epochs or 0) <= 0 and (max_steps or 0) <= 0:
                result.add_error(
                    "num_train_epochs",
                    "Either num_train_epochs or max_train_steps must be greater than 0",
                )
            if (num_epochs or 0) > 0 and (max_steps or 0) > 0:
                result.add_error(
                    "max_train_steps",
                    "num_train_epochs and max_train_steps cannot both be set. Set one of them to 0.",
                )

        # Example: Model-specific validations
        model_type_raw = self._get_config_value(config, "model_type") or ""
        model_type = str(model_type_raw).lower()
        if model_type == "lora" and not self._get_config_value(config, "lora_rank"):
            result.add_error("lora_rank", "LoRA rank is required when using LoRA model type")

        base_precision_raw = self._get_config_value(config, "base_model_precision")
        base_precision = ""
        if base_precision_raw not in (None, ""):
            base_precision = str(base_precision_raw).strip().lower()
            if base_precision in {"none", "null", "false"}:
                base_precision = ""

        if model_type == "full" and base_precision and base_precision != "no_change":
            result.add_error(
                "base_model_precision",
                "Full model training is incompatible with base model quantisation. "
                "Set base_model_precision to 'no_change' or switch to LoRA.",
            )
        quantization_config_raw = self._get_config_value(config, "quantization_config")
        if model_type == "full" and quantization_config_raw not in (None, "", "None"):
            result.add_error(
                "quantization_config",
                "Full model training is incompatible with pipeline quantization configs. Clear quantization_config or switch to LoRA.",
            )

        deepspeed_raw = self._get_config_value(config, "deepspeed_config")
        if model_type == "lora" and deepspeed_raw not in (None, "", "None", False):
            result.add_error(
                "deepspeed_config",
                "LoRA training cannot be combined with DeepSpeed. " "Clear deepspeed_config or use full model training.",
            )

        # Learning rate scheduler validations
        scheduler_raw = self._get_config_value(config, "lr_scheduler")
        warmup_raw = self._get_config_value(config, "lr_warmup_steps") or 0
        warmup_value, warmup_error = _to_int(warmup_raw)
        scheduler = normalize_lr_scheduler_value(
            scheduler_raw,
            None if warmup_error else warmup_value,
        )

        if scheduler != scheduler_raw and scheduler is not None:
            for key in ("lr_scheduler", "--lr_scheduler"):
                if key in config:
                    config[key] = scheduler

        scheduler_str = str(scheduler or "")
        if scheduler_str == "polynomial" and not self._get_config_value(config, "lr_scheduler_polynomial_power"):
            result.add_warning(
                "lr_scheduler_polynomial_power",
                "Polynomial power not specified, defaulting to 1.0",
            )

        if warmup_error:
            result.add_error("lr_warmup_steps", "Warmup steps must be a whole number.")

        # Attention mechanism availability checks
        attention_mech = str(self._get_config_value(config, "attention_mechanism") or "diffusers")
        if attention_mech == "xformers":
            xformers_error = xformers_compute_capability_error()
            if xformers_error:
                result.add_error("attention_mechanism", xformers_error)

        if attention_mech.startswith("sage") and not is_sageattention_available():
            result.add_error(
                "attention_mechanism",
                f"SageAttention is not installed but '{attention_mech}' was selected. "
                "Install it with: pip install sageattention",
            )

        # Disk low space detection validation
        disk_threshold = self._get_config_value(config, "disk_low_threshold")
        if disk_threshold not in (None, "", "None"):
            disk_action = self._get_config_value(config, "disk_low_action")
            disk_script = self._get_config_value(config, "disk_low_script")

            if disk_action == "script" and disk_script in (None, "", "None"):
                result.add_error(
                    "disk_low_script",
                    "Cleanup script path is required when disk_low_action is 'script'.",
                )
            elif disk_action == "script" and disk_script:
                from pathlib import Path as PathLib

                script_path = PathLib(disk_script).expanduser()
                if not script_path.exists():
                    result.add_error(
                        "disk_low_script",
                        f"Cleanup script does not exist: {disk_script}",
                    )

    @staticmethod
    def _get_config_value(config: Dict[str, Any], field_name: str) -> Any:
        """Fetch a field value considering CLI-prefixed variants."""
        if field_name in config:
            return config[field_name]

        prefixed = field_name if field_name.startswith("--") else f"--{field_name}"
        if prefixed in config:
            return config[prefixed]

        if field_name.startswith("--"):
            stripped = field_name.lstrip("-")
            if stripped in config:
                return config[stripped]

        return None

    @staticmethod
    def _normalize_field_name(field_name: str) -> str:
        """Return a canonical representation for comparing field identifiers."""
        return field_name.lstrip("-")

    def _validate_paths(self, config: Dict[str, Any], result: ValidationResult):
        """Validate file and directory paths in configuration."""
        path_fields = [
            ("pretrained_model_name_or_path", "Model path"),
            ("output_dir", "Output directory"),
            ("logging_dir", "Logging directory"),
            ("resume_from_checkpoint", "Checkpoint path"),
        ]

        for field_name, display_name in path_fields:
            path_value = self._get_config_value(config, field_name)
            if not path_value:
                continue

            if isinstance(path_value, str) and path_value.startswith("http"):
                continue

            path = Path(str(path_value))
            normalized_field = self._normalize_field_name(field_name)

            if normalized_field.endswith("_dir"):
                # Directory should exist or be creatable
                if not path.exists():
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                        result.add_info(normalized_field, f"{display_name} will be created: {path_value}")
                    except Exception as exc:
                        result.add_error(normalized_field, f"Cannot create {display_name}: {exc}")
            else:
                # File should exist
                if not path.exists():
                    result.add_warning(normalized_field, f"{display_name} not found: {path_value}")

    def _estimate_vram(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate VRAM usage based on configuration."""

        # Simple estimation logic - can be enhanced
        def _int_or_default(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        model_type = str(self._get_config_value(config, "model_type") or "full").lower()
        batch_size = _int_or_default(self._get_config_value(config, "train_batch_size"), 1)
        gradient_accumulation = _int_or_default(
            self._get_config_value(config, "gradient_accumulation_steps"),
            1,
        )
        resolution = _int_or_default(self._get_config_value(config, "resolution"), 512)

        # Base estimates (in GB)
        base_vram = {"full": 24, "lora": 12, "dora": 16}.get(model_type, 24)

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
                f"Resolution scaling: {resolution_factor:.2f}x",
            ],
        }
