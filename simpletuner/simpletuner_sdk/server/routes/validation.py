"""Field validation routes for HTMX forms and comprehensive configuration validation."""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Dict, List, Optional, Any
from enum import Enum

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..services.field_registry import field_registry, ValidationRuleType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/validate", tags=["validation"])


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


class ValidationMessage(BaseModel):
    field: str
    message: str
    severity: ValidationSeverity
    suggestion: Optional[str] = None


class ValidationStatus(BaseModel):
    valid: bool
    errors: int = 0
    warnings: int = 0
    info: int = 0
    messages: List[ValidationMessage]
    vram_estimate: Optional[Dict[str, Any]] = None


class ConfigValidationRequest(BaseModel):
    config: Dict[str, Any] = Field(..., description="Configuration to validate")
    validate_paths: bool = Field(True, description="Whether to validate file/directory paths")
    estimate_vram: bool = Field(True, description="Whether to estimate VRAM usage")


@router.post("/{field_name}", response_class=HTMLResponse)
async def validate_field(field_name: str, value: str = Form("")) -> str:
    """Validate a single field and return HTML error fragment."""
    error_html = ""

    # General field validations
    if field_name in ["model_name", "output_dir", "pretrained_model_name_or_path"]:
        if not value or not value.strip():
            error_html = f"{field_name.replace('_', ' ').title()} is required"
        elif len(value) > 200:
            error_html = f"{field_name.replace('_', ' ').title()} must be 200 characters or less"

    elif field_name in ["learning_rate", "lr_scheduler_eta_min"]:
        try:
            lr = float(value) if value else 0
            if lr <= 0:
                error_html = "Learning rate must be greater than 0"
            elif lr > 1:
                error_html = "Learning rate should typically be less than 1"
        except ValueError:
            error_html = "Learning rate must be a valid number"

    elif field_name in ["num_train_epochs", "max_train_steps"]:
        try:
            epochs = int(value) if value else 0
            if epochs <= 0:
                error_html = f"{field_name.replace('_', ' ').title()} must be greater than 0"
        except ValueError:
            error_html = f"{field_name.replace('_', ' ').title()} must be a valid number"

    elif field_name in ["train_batch_size", "gradient_accumulation_steps"]:
        try:
            batch_size = int(value) if value else 0
            if batch_size <= 0:
                error_html = f"{field_name.replace('_', ' ').title()} must be greater than 0"
            elif batch_size > 128:
                error_html = f"{field_name.replace('_', ' ').title()} should typically be 128 or less"
        except ValueError:
            error_html = f"{field_name.replace('_', ' ').title()} must be a valid number"

    elif field_name == "resolution":
        try:
            res = int(value) if value else 0
            if res < 256:
                error_html = "Resolution must be at least 256"
            elif res > 4096:
                error_html = "Resolution must not exceed 4096"
            elif res % 64 != 0:
                error_html = "Resolution must be divisible by 64"
        except ValueError:
            error_html = "Resolution must be a valid number"

    elif field_name in ["output_dir", "logging_dir", "instance_data_dir"]:
        if not value or not value.strip():
            error_html = f"{field_name.replace('_', ' ').title()} is required"
        elif not os.path.isabs(value) and not ":" in value:
            error_html = "Path must be absolute (start with /) or include drive letter on Windows"

    elif field_name == "webhook_url":
        if value and not re.match(r"^https?://", value):
            error_html = "Webhook URL must start with http:// or https://"

    elif field_name in ["lora_rank", "lora_alpha"]:
        try:
            rank = int(value) if value else 0
            if rank <= 0:
                error_html = f"{field_name.replace('_', ' ').title()} must be greater than 0"
            elif rank > 512:
                error_html = f"{field_name.replace('_', ' ').title()} should typically be 512 or less"
        except ValueError:
            error_html = f"{field_name.replace('_', ' ').title()} must be a valid number"

    elif field_name == "mixed_precision":
        if value not in ["no", "fp16", "bf16"]:
            error_html = "Mixed precision must be 'no', 'fp16', or 'bf16'"

    elif field_name == "lr_scheduler":
        valid_schedulers = [
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
        ]
        if value and value not in valid_schedulers:
            error_html = f"Learning rate scheduler must be one of: {', '.join(valid_schedulers)}"

    elif field_name == "optimizer":
        valid_optimizers = ["adamw", "adam", "sgd", "adafactor", "adamw_bnb_8bit"]
        if value and value not in valid_optimizers:
            error_html = f"Optimizer must be one of: {', '.join(valid_optimizers)}"

    return error_html


def validate_field_value(field_name: str, value: Any, config: Dict[str, Any]) -> List[ValidationMessage]:
    """Validate a single field value against its rules."""
    messages = []

    field = field_registry.get_field(field_name)
    if not field:
        return messages

    for rule in field.validation_rules:
        if rule.type == ValidationRuleType.REQUIRED and not value:
            messages.append(ValidationMessage(
                field=field_name,
                message=rule.message or f"{field.ui_label} is required",
                severity=ValidationSeverity.ERROR
            ))

        elif rule.type == ValidationRuleType.MIN and value is not None:
            try:
                if float(value) < rule.value:
                    messages.append(ValidationMessage(
                        field=field_name,
                        message=rule.message or f"{field.ui_label} must be at least {rule.value}",
                        severity=ValidationSeverity.ERROR
                    ))
            except (TypeError, ValueError):
                pass

        elif rule.type == ValidationRuleType.MAX and value is not None:
            try:
                if float(value) > rule.value:
                    messages.append(ValidationMessage(
                        field=field_name,
                        message=rule.message or f"{field.ui_label} must be at most {rule.value}",
                        severity=ValidationSeverity.WARNING,
                        suggestion=f"Consider using a value <= {rule.value}"
                    ))
            except (TypeError, ValueError):
                pass

        elif rule.type == ValidationRuleType.PATTERN and value:
            import re
            if not re.match(rule.pattern, str(value)):
                messages.append(ValidationMessage(
                    field=field_name,
                    message=rule.message or f"{field.ui_label} format is invalid",
                    severity=ValidationSeverity.ERROR
                ))

        elif rule.type == ValidationRuleType.DIVISIBLE_BY and value is not None:
            try:
                if float(value) % rule.value != 0:
                    messages.append(ValidationMessage(
                        field=field_name,
                        message=rule.message or f"{field.ui_label} must be divisible by {rule.value}",
                        severity=ValidationSeverity.ERROR
                    ))
            except (TypeError, ValueError):
                pass

    return messages


def validate_paths(config: Dict[str, Any]) -> List[ValidationMessage]:
    """Validate file and directory paths in the configuration."""
    messages = []

    # Check output directory
    output_dir = config.get('output_dir')
    if output_dir:
        if not os.path.exists(output_dir):
            messages.append(ValidationMessage(
                field='output_dir',
                message=f"Output directory '{output_dir}' does not exist",
                severity=ValidationSeverity.WARNING,
                suggestion="Directory will be created automatically"
            ))

    # Check model path
    model_path = config.get('pretrained_model_name_or_path')
    if model_path and not model_path.startswith('http'):
        if '/' not in model_path and not os.path.exists(model_path):
            # Likely a HuggingFace model ID
            messages.append(ValidationMessage(
                field='pretrained_model_name_or_path',
                message=f"Model '{model_path}' will be downloaded from HuggingFace",
                severity=ValidationSeverity.INFO
            ))
        elif not os.path.exists(model_path):
            messages.append(ValidationMessage(
                field='pretrained_model_name_or_path',
                message=f"Model path '{model_path}' does not exist",
                severity=ValidationSeverity.ERROR
            ))

    # Check data backend config
    data_config = config.get('data_backend_config')
    if data_config and os.path.isfile(data_config):
        try:
            with open(data_config, 'r') as f:
                json.load(f)
        except json.JSONDecodeError:
            messages.append(ValidationMessage(
                field='data_backend_config',
                message=f"Data backend config '{data_config}' is not valid JSON",
                severity=ValidationSeverity.ERROR
            ))

    return messages


def estimate_vram_usage(config: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate VRAM usage based on configuration."""
    model_family = config.get('model_family', 'sdxl')
    model_type = config.get('model_type', 'full')
    resolution = int(config.get('resolution', 1024))
    batch_size = int(config.get('train_batch_size', 1))
    mixed_precision = config.get('mixed_precision', 'bf16')
    gradient_accumulation = int(config.get('gradient_accumulation_steps', 1))

    # Base model VRAM estimates (in GB)
    model_vram = {
        'sd15': 2.5,
        'sd2x': 3.5,
        'sdxl': 7.0,
        'sd3': 12.0,
        'flux': 24.0,
        'sana': 6.0,
        'kolors': 8.0,
        'pixart_sigma': 5.0
    }.get(model_family, 7.0)

    # Adjust for model type
    if model_type == 'lora':
        model_vram *= 0.3  # LoRA uses much less VRAM

    # Adjust for precision
    precision_multiplier = {
        'fp32': 1.0,
        'fp16': 0.5,
        'bf16': 0.5,
        'fp8-e4m3fn': 0.25,
        'fp8-e5m2': 0.25
    }.get(mixed_precision, 0.5)

    model_vram *= precision_multiplier

    # Calculate activation memory
    pixels_per_image = resolution * resolution
    channels = 4  # Latent space channels
    bytes_per_pixel = 4 if mixed_precision == 'fp32' else 2

    activation_vram = (pixels_per_image * channels * bytes_per_pixel * batch_size) / (1024**3)

    # Optimizer states (Adam uses 2x model parameters)
    optimizer_vram = model_vram * 2 if model_type == 'full' else model_vram * 0.5

    # Gradient accumulation reduces peak memory
    effective_batch_vram = activation_vram / gradient_accumulation

    # Total estimate
    total_vram = model_vram + effective_batch_vram + optimizer_vram

    return {
        'model': round(model_vram, 2),
        'activations': round(effective_batch_vram, 2),
        'optimizer': round(optimizer_vram, 2),
        'total': round(total_vram, 2),
        'recommended_gpu': get_recommended_gpu(total_vram)
    }


def get_recommended_gpu(vram_gb: float) -> str:
    """Get recommended GPU based on VRAM requirements."""
    if vram_gb <= 8:
        return "RTX 3060 12GB, RTX 4060 Ti 16GB"
    elif vram_gb <= 16:
        return "RTX 3080 10GB, RTX 4070 Ti 12GB"
    elif vram_gb <= 24:
        return "RTX 3090 24GB, RTX 4090 24GB"
    elif vram_gb <= 40:
        return "A100 40GB, A6000 48GB"
    else:
        return "A100 80GB or multi-GPU setup"


@router.post("/config", response_model=ValidationStatus)
async def validate_configuration(request: ConfigValidationRequest):
    """Validate a training configuration."""
    config = request.config
    messages = []

    # Validate each field
    for field_name, value in config.items():
        field_messages = validate_field_value(field_name, value, config)
        messages.extend(field_messages)

    # Validate paths if requested
    if request.validate_paths:
        path_messages = validate_paths(config)
        messages.extend(path_messages)

    # Check for conflicting settings
    if config.get('model_type') == 'lora' and config.get('train_text_encoder'):
        messages.append(ValidationMessage(
            field='train_text_encoder',
            message="Training text encoder with LoRA is not recommended",
            severity=ValidationSeverity.WARNING,
            suggestion="Consider disabling text encoder training for LoRA"
        ))

    # Count message types
    error_count = sum(1 for m in messages if m.severity == ValidationSeverity.ERROR)
    warning_count = sum(1 for m in messages if m.severity == ValidationSeverity.WARNING)
    info_count = sum(1 for m in messages if m.severity == ValidationSeverity.INFO)

    # Estimate VRAM if requested
    vram_estimate = None
    if request.estimate_vram:
        try:
            vram_estimate = estimate_vram_usage(config)
        except Exception as e:
            logger.error(f"Error estimating VRAM: {e}")

    # Add success message if no errors
    if error_count == 0:
        messages.append(ValidationMessage(
            field="_overall",
            message="Configuration is valid and ready for training",
            severity=ValidationSeverity.SUCCESS
        ))

    return ValidationStatus(
        valid=error_count == 0,
        errors=error_count,
        warnings=warning_count,
        info=info_count,
        messages=messages,
        vram_estimate=vram_estimate
    )


@router.post("/field")
async def validate_single_field(
    field_name: str,
    value: Any,
    config: Dict[str, Any] = {}
):
    """Validate a single field value."""
    messages = validate_field_value(field_name, value, config)

    return {
        "field": field_name,
        "valid": not any(m.severity == ValidationSeverity.ERROR for m in messages),
        "messages": messages
    }


@router.get("/presets")
async def get_validation_presets():
    """Get common validation presets for different scenarios."""
    return {
        "presets": [
            {
                "name": "Quick Test",
                "description": "Minimal settings for testing",
                "overrides": {
                    "num_train_epochs": 1,
                    "max_train_steps": 100,
                    "save_every_n_steps": 50,
                    "validation_every_n_steps": 50
                }
            },
            {
                "name": "Low VRAM",
                "description": "Optimized for GPUs with <12GB VRAM",
                "overrides": {
                    "train_batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "resolution": 512,
                    "mixed_precision": "fp16",
                    "gradient_checkpointing": True
                }
            },
            {
                "name": "High Quality",
                "description": "Settings for best quality output",
                "overrides": {
                    "num_train_epochs": 10,
                    "learning_rate": 1e-5,
                    "lr_scheduler": "cosine",
                    "lr_warmup_steps": 500,
                    "save_every_n_steps": 500
                }
            }
        ]
    }
