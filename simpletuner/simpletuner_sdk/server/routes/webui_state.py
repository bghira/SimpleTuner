"""API routes for persisting Web UI state across sessions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from simpletuner.simpletuner_sdk.server.services.webui_state import (
    OnboardingStepState,
    WebUIDefaults,
    WebUIOnboardingState,
    WebUIState,
    WebUIStateStore,
    _normalise_accelerate_overrides,
)

router = APIRouter(prefix="/api/webui", tags=["webui"])


@dataclass(frozen=True)
class OnboardingStepDefinition:
    """Definition of a single onboarding step."""

    id: str
    title: str
    prompt: str
    input_type: str
    version: int
    required: bool = True
    applies_to_default: Optional[str] = None


_STEP_DEFINITIONS: List[OnboardingStepDefinition] = [
    OnboardingStepDefinition(
        id="default_configs_dir",
        title="Default configurations directory",
        prompt="Where do you want to store your training configurations?",
        input_type="directory",
        version=2,
        required=True,
        applies_to_default="configs_dir",
    ),
    OnboardingStepDefinition(
        id="default_output_dir",
        title="Default output directory",
        prompt="Where do you want to store outputs?",
        input_type="directory",
        version=1,
        required=True,
        applies_to_default="output_dir",
    ),
    OnboardingStepDefinition(
        id="default_datasets_dir",
        title="Default datasets directory",
        prompt="Where do you want to store your datasets? (Leave blank to allow datasets anywhere)",
        input_type="directory",
        version=2,
        required=True,
        applies_to_default="datasets_dir",
    ),
    OnboardingStepDefinition(
        id="accelerate_defaults",
        title="Accelerate GPU Defaults",
        prompt="Review the detected hardware and choose how many processes Accelerate should launch by default.",
        input_type="accelerate_auto",
        version=1,
        required=True,
        applies_to_default="accelerate_overrides",
    ),
]
_STEP_INDEX: Dict[str, OnboardingStepDefinition] = {step.id: step for step in _STEP_DEFINITIONS}


class OnboardingStepUpdate(BaseModel):
    """Payload for onboarding step updates."""

    value: Optional[Any] = None


def _build_state_response(state: WebUIState, steps: List[OnboardingStepDefinition]) -> Dict[str, object]:
    store = WebUIStateStore()
    bundle = store.resolve_defaults(state.defaults)
    defaults_dict = bundle["raw"]
    resolved_defaults = bundle["resolved"]
    fallbacks = bundle["fallbacks"]

    overlay_required = False
    onboarding_steps = []

    for step in steps:
        stored: Optional[OnboardingStepState] = state.onboarding.steps.get(step.id)
        completed_version = stored.completed_version if stored else 0
        is_complete = completed_version >= step.version
        if not is_complete and step.required:
            overlay_required = True

        onboarding_steps.append(
            {
                "id": step.id,
                "title": step.title,
                "prompt": step.prompt,
                "input_type": step.input_type,
                "version": step.version,
                "required": step.required,
                "applies_to_default": step.applies_to_default,
                "value": stored.value if stored else None,
                "completed_version": completed_version,
                "completed_at": stored.completed_at if stored else None,
                "is_complete": is_complete,
            }
        )

    return {
        "defaults": defaults_dict,
        "resolved_defaults": resolved_defaults,
        "fallbacks": fallbacks,
        "onboarding": {
            "steps": onboarding_steps,
            "overlay_required": overlay_required,
        },
    }


@router.get("/state")
async def get_webui_state() -> Dict[str, object]:
    """Get the persisted Web UI state."""
    store = WebUIStateStore()
    try:
        state = store.load_state()
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error
    return _build_state_response(state, _STEP_DEFINITIONS)


def _normalise_value(step: OnboardingStepDefinition, value: object) -> Optional[object]:
    if step.input_type == "accelerate_auto":
        if value is None:
            return None
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError as error:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid accelerator selection payload: {error}",
                ) from error
            return payload
        if isinstance(value, dict):
            return value
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Accelerator selection must be an object.",
        )

    if value is None:
        return None
    normalised = value.strip()
    if not normalised:
        return None
    if step.input_type == "directory":
        return os.path.abspath(os.path.expanduser(normalised))
    return normalised


def _apply_step_to_defaults(
    defaults: WebUIDefaults,
    step: OnboardingStepDefinition,
    value: Optional[str],
) -> None:
    if step.applies_to_default == "output_dir" and value is not None:
        defaults.output_dir = value
    elif step.applies_to_default == "configs_dir" and value is not None:
        defaults.configs_dir = value
    elif step.applies_to_default == "datasets_dir" and value is not None:
        defaults.datasets_dir = value
    elif step.applies_to_default == "accelerate_overrides":
        defaults.accelerate_overrides = _normalise_accelerate_overrides(value)


@router.post("/onboarding/steps/{step_id}")
async def update_onboarding_step(step_id: str, payload: OnboardingStepUpdate) -> Dict[str, object]:
    """Update onboarding progress for a specific step."""
    definition = _STEP_INDEX.get(step_id)
    if definition is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown onboarding step '{step_id}'")

    value = _normalise_value(definition, payload.value)
    # Special case: datasets_dir can be empty (means allow datasets anywhere)
    if definition.required and not value and definition.id != "default_datasets_dir":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="A value is required to complete this step.",
        )

    # Auto-create directory if it's a directory type and doesn't exist
    if definition.input_type == "directory" and value:
        from pathlib import Path
        dir_path = Path(value)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as error:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create directory '{value}': {error}",
                ) from error

    store = WebUIStateStore()
    try:
        defaults = store.load_defaults()
        store.record_onboarding_step(step_id, definition.version, value=value)
        _apply_step_to_defaults(defaults, definition, value)
        store.save_defaults(defaults)
        state = store.load_state()
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error

    return _build_state_response(state, _STEP_DEFINITIONS)


@router.post("/onboarding/reset")
async def reset_onboarding() -> Dict[str, object]:
    """Reset onboarding data to allow starting fresh."""
    store = WebUIStateStore()
    try:
        # Clear both onboarding and defaults
        store.save_onboarding(WebUIOnboardingState())
        store.save_defaults(WebUIDefaults())

        # Return fresh state
        state = store.load_state()
        return _build_state_response(state, _STEP_DEFINITIONS)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error


class DefaultsUpdate(BaseModel):
    """Payload for updating WebUI defaults."""

    configs_dir: Optional[str] = None
    output_dir: Optional[str] = None
    datasets_dir: Optional[str] = None
    active_config: Optional[str] = None
    theme: Optional[str] = None
    event_polling_interval: Optional[int] = None
    event_stream_enabled: Optional[bool] = None
    auto_preserve_defaults: Optional[bool] = None
    allow_dataset_paths_outside_dir: Optional[bool] = None
    accelerate_overrides: Optional[Dict[str, object]] = None


@router.post("/defaults/update")
async def update_defaults(payload: DefaultsUpdate) -> Dict[str, object]:
    """Update WebUI default settings."""
    store = WebUIStateStore()
    try:
        # Load current defaults
        defaults = store.load_defaults()

        # Update fields if provided
        if payload.configs_dir is not None:
            normalized_path = os.path.abspath(os.path.expanduser(payload.configs_dir.strip()))
            defaults.configs_dir = normalized_path
        if payload.output_dir is not None:
            normalized_path = os.path.abspath(os.path.expanduser(payload.output_dir.strip()))
            defaults.output_dir = normalized_path
        if payload.datasets_dir is not None:
            if payload.datasets_dir.strip():
                normalized_path = os.path.abspath(os.path.expanduser(payload.datasets_dir.strip()))
                defaults.datasets_dir = normalized_path
            else:
                defaults.datasets_dir = None
        if payload.active_config is not None:
            defaults.active_config = payload.active_config
        if payload.theme is not None:
            theme = payload.theme.strip().lower()
            defaults.theme = theme if theme in {"dark", "tron"} else "dark"
        if payload.event_polling_interval is not None:
            try:
                interval = int(payload.event_polling_interval)
            except (TypeError, ValueError):
                interval = defaults.event_polling_interval
            defaults.event_polling_interval = max(1, interval)
        if payload.event_stream_enabled is not None:
            defaults.event_stream_enabled = bool(payload.event_stream_enabled)
        if payload.auto_preserve_defaults is not None:
            defaults.auto_preserve_defaults = bool(payload.auto_preserve_defaults)
        if payload.allow_dataset_paths_outside_dir is not None:
            defaults.allow_dataset_paths_outside_dir = bool(payload.allow_dataset_paths_outside_dir)
        if payload.accelerate_overrides is not None:
            defaults.accelerate_overrides = _normalise_accelerate_overrides(payload.accelerate_overrides)

        # Save updated defaults
        store.save_defaults(defaults)

        # Return updated state
        state = store.load_state()
        return _build_state_response(state, _STEP_DEFINITIONS)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error
