"""Field validation endpoints that delegate to the shared validation service."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import get_current_user
from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User

from ..services.validation_service import ValidationService

router = APIRouter(prefix="/api/validate", tags=["validation"])
_validation_service = ValidationService()


class ValidationSeverity(str, Enum):
    """HTTP response severity mirror for validation messages."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


class ValidationMessage(BaseModel):
    """API representation of a validation message."""

    field: str
    message: str
    severity: ValidationSeverity
    suggestion: Optional[str] = None


class ValidationStatus(BaseModel):
    """Top-level validation payload return type."""

    valid: bool
    errors: int = 0
    warnings: int = 0
    info: int = 0
    messages: List[ValidationMessage]
    vram_estimate: Optional[Dict[str, Any]] = None


class ConfigValidationRequest(BaseModel):
    """Expected structure for configuration validation requests."""

    config: Dict[str, Any] = Field(..., description="Configuration to validate")
    validate_paths: bool = Field(True, description="Whether to validate file/directory paths")
    estimate_vram: bool = Field(True, description="Whether to estimate VRAM usage")


def _form_to_context(form: Any, ignore_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convert a Starlette FormData into a simple dictionary context."""
    if ignore_keys is None:
        ignore_keys = []

    context: Dict[str, Any] = {}
    try:
        items = form.multi_items()  # type: ignore[attr-defined]
    except AttributeError:
        items = form.items()

    for key, value in items:
        if key in ignore_keys:
            continue
        context[key] = value
    return context


@router.post("/{field_name}", response_class=HTMLResponse)
async def validate_field(field_name: str, request: Request, _user: User = Depends(get_current_user)) -> str:
    """Validate a single field and return the rendered error fragment."""
    form = await request.form()

    value = form.get("value")
    if value is None:
        value = form.get(field_name)
    if value is None:
        value = form.get(f"--{field_name}")
    if value is None:
        value = ""

    context = _form_to_context(form, ignore_keys=["value", field_name, f"--{field_name}"])

    return _validation_service.get_field_validation_html(
        field_name,
        value,
        config=context if context else None,
        validate_paths=False,
    )


@router.post("/config", response_model=ValidationStatus)
async def validate_configuration(
    request: ConfigValidationRequest, _user: User = Depends(get_current_user)
) -> ValidationStatus:
    """Validate an entire configuration payload."""
    result = _validation_service.validate_configuration(
        request.config,
        validate_paths=request.validate_paths,
        estimate_vram=request.estimate_vram,
    )

    result_dict = result.to_dict()
    # Pydantic model coercion handles Enum conversion for us.
    return ValidationStatus(**result_dict)


@router.post("/field")
async def validate_single_field(
    field_name: str,
    value: Any,
    config: Dict[str, Any] = {},
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Validate a single field with optional configuration context."""
    working_config = dict(config)
    normalized_field = field_name.lstrip("-")
    working_config[field_name] = value
    working_config[normalized_field] = value
    working_config[f"--{normalized_field}"] = value

    result = _validation_service.validate_configuration(
        working_config,
        validate_paths=False,
        estimate_vram=False,
    )

    result_payload = result.to_dict()
    filtered_messages = [
        ValidationMessage(**message)
        for message in result_payload.get("messages", [])
        if message.get("field") == normalized_field and message.get("severity") != ValidationSeverity.SUCCESS.value
    ]

    is_valid = not any(msg.severity == ValidationSeverity.ERROR for msg in filtered_messages)

    return {
        "field": normalized_field,
        "valid": is_valid,
        "messages": [message.model_dump() for message in filtered_messages],
    }
