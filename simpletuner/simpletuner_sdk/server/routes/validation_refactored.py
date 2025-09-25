"""Refactored validation routes using ValidationService."""

from __future__ import annotations

import logging
from typing import Dict, Any

from fastapi import APIRouter, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..services.validation_service import ValidationService
from ..dependencies.common import get_config_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/validate", tags=["validation"])

# Initialize validation service
validation_service = ValidationService()


class ConfigValidationRequest(BaseModel):
    """Request model for configuration validation."""
    config: Dict[str, Any] = Field(..., description="Configuration to validate")
    validate_paths: bool = Field(True, description="Whether to validate file/directory paths")
    estimate_vram: bool = Field(True, description="Whether to estimate VRAM usage")


@router.post("/{field_name}", response_class=HTMLResponse)
async def validate_field(field_name: str, value: str = Form("")) -> str:
    """Validate a single field and return HTML error fragment for HTMX.

    This endpoint uses the ValidationService to validate fields
    based on registry rules instead of hardcoded validation.
    """
    logger.debug(f"Validating field '{field_name}' with value: {value}")

    # Use validation service to get HTML error fragment
    error_html = validation_service.get_field_validation_html(field_name, value)

    logger.debug(f"Validation result for '{field_name}': {'valid' if not error_html else 'invalid'}")

    return error_html


@router.post("/config/full")
async def validate_full_config(request: ConfigValidationRequest):
    """Validate an entire configuration.

    This endpoint provides comprehensive validation including:
    - Individual field validation
    - Cross-field validation
    - Path existence checks
    - VRAM estimation
    """
    logger.info("Starting full configuration validation")

    # Use validation service for comprehensive validation
    result = validation_service.validate_configuration(
        config=request.config,
        validate_paths=request.validate_paths,
        estimate_vram=request.estimate_vram
    )

    logger.info(
        f"Validation complete - Valid: {result.is_valid}, "
        f"Errors: {result.error_count}, Warnings: {result.warning_count}"
    )

    return result.to_dict()


@router.get("/config/current")
async def validate_current_config(
    config_data: Dict[str, Any] = Depends(get_config_data),
    validate_paths: bool = True,
    estimate_vram: bool = True
):
    """Validate the currently active configuration.

    Query parameters:
    - validate_paths: Whether to validate file/directory paths (default: true)
    - estimate_vram: Whether to estimate VRAM usage (default: true)
    """
    logger.info("Validating current active configuration")

    # Use validation service
    result = validation_service.validate_configuration(
        config=config_data,
        validate_paths=validate_paths,
        estimate_vram=estimate_vram
    )

    return result.to_dict()


@router.get("/field/{field_name}/rules")
async def get_field_validation_rules(field_name: str):
    """Get validation rules for a specific field.

    This endpoint exposes the validation rules from the field registry
    for client-side validation or documentation purposes.
    """
    from ..services.field_registry_wrapper import lazy_field_registry

    field = lazy_field_registry.get_field(field_name)
    if not field:
        raise HTTPException(
            status_code=404,
            detail=f"Field '{field_name}' not found in registry"
        )

    # Extract validation rules
    rules = []
    if hasattr(field, 'validation_rules') and field.validation_rules:
        for rule in field.validation_rules:
            if isinstance(rule, dict):
                rules.append(rule)
            else:
                # Convert rule object to dict
                rules.append({
                    "type": getattr(rule, 'type', 'unknown'),
                    "value": getattr(rule, 'value', None),
                    "message": getattr(rule, 'message', None)
                })

    return {
        "field": field_name,
        "label": field.ui_label,
        "type": field.field_type.value,
        "required": getattr(field, 'required', False),
        "validation_rules": rules
    }


@router.post("/batch", response_class=HTMLResponse)
async def validate_fields_batch(fields: Dict[str, Any]):
    """Validate multiple fields at once and return combined HTML results.

    This is useful for validating related fields together.
    """
    error_messages = []

    for field_name, value in fields.items():
        error_html = validation_service.get_field_validation_html(field_name, value)
        if error_html:
            error_messages.append(f'<div class="field-error" data-field="{field_name}">{error_html}</div>')

    if error_messages:
        return f'<div class="validation-errors">{"".join(error_messages)}</div>'
    else:
        return '<div class="validation-success">All fields are valid</div>'


# Utility endpoints for HTMX integration
@router.get("/status", response_class=HTMLResponse)
async def validation_status(config_data: Dict[str, Any] = Depends(get_config_data)):
    """Get validation status summary as HTML fragment.

    Returns a status badge/indicator for the current configuration.
    """
    result = validation_service.validate_configuration(
        config=config_data,
        validate_paths=False,  # Quick check without path validation
        estimate_vram=False
    )

    if result.is_valid and result.warning_count == 0:
        status_class = "success"
        status_text = "Valid"
        status_icon = "check-circle"
    elif result.is_valid:
        status_class = "warning"
        status_text = f"Valid with {result.warning_count} warning(s)"
        status_icon = "exclamation-triangle"
    else:
        status_class = "danger"
        status_text = f"{result.error_count} error(s)"
        status_icon = "times-circle"

    return f'''
    <div class="validation-status text-{status_class}">
        <i class="fas fa-{status_icon}"></i>
        <span>{status_text}</span>
    </div>
    '''