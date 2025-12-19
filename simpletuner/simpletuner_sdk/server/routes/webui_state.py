"""API routes for persisting Web UI state across sessions."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from simpletuner.simpletuner_sdk.server.services.configs_service import CONFIGS_SERVICE, ConfigServiceError
from simpletuner.simpletuner_sdk.server.services.git_config_service import GIT_CONFIG_SERVICE
from simpletuner.simpletuner_sdk.server.services.webui_state import (
    OnboardingStepState,
    WebUIDefaults,
    WebUIOnboardingState,
    WebUIState,
    WebUIStateStore,
    _normalise_accelerate_overrides,
)

router = APIRouter(prefix="/api/webui", tags=["webui"])
logger = logging.getLogger(__name__)


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


_CORE_STEP_DEFINITIONS: List[OnboardingStepDefinition] = [
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

_OPTIONAL_STEPS: Dict[str, OnboardingStepDefinition] = {
    "create_initial_environment": OnboardingStepDefinition(
        id="create_initial_environment",
        title="Create your training environment",
        prompt=(
            "The Default environment is a safety net. Create the environment you'll use for training "
            "so SimpleTuner can keep your changes separate."
        ),
        input_type="environment",
        version=1,
        required=True,
    )
}

_ALL_STEPS: Dict[str, OnboardingStepDefinition] = {
    **{step.id: step for step in _CORE_STEP_DEFINITIONS},
    **_OPTIONAL_STEPS,
}


def _sanitize_environment_name(candidate: str) -> str:
    """Normalise environment names similar to the frontend sanitiser."""
    value = (candidate or "").strip()
    if value.lower().endswith(".json"):
        value = value[:-5].strip()
    return value


def _should_include_environment_step() -> bool:
    """Determine if the environment creation onboarding step should be shown."""
    try:
        payload = CONFIGS_SERVICE.list_configs("model")
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to inspect environments for onboarding step: %s", exc, exc_info=True)
        return False

    configs = payload.get("configs") if isinstance(payload, dict) else None
    if not isinstance(configs, list):
        return False

    non_default_found = False
    for entry in configs:
        if not isinstance(entry, dict):
            continue
        name = _sanitize_environment_name(str(entry.get("name") or ""))
        if not name:
            continue
        if name.lower() != "default":
            non_default_found = True
            break

    return not non_default_found


def _resolve_step_definitions() -> List[OnboardingStepDefinition]:
    """Return the ordered onboarding steps, including optional ones when required."""
    steps = list(_CORE_STEP_DEFINITIONS)
    if _should_include_environment_step():
        steps.append(_OPTIONAL_STEPS["create_initial_environment"])
    return steps


class OnboardingStepUpdate(BaseModel):
    """Payload for onboarding step updates."""

    value: Optional[Any] = None


def _build_state_response(state: WebUIState, steps: List[OnboardingStepDefinition]) -> Dict[str, object]:
    store = WebUIStateStore()
    synced_defaults = _sync_defaults_from_onboarding(store, state.defaults)
    bundle = store.resolve_defaults(synced_defaults)
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
    return _build_state_response(state, _resolve_step_definitions())


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

    if step.input_type == "environment":
        if value is None:
            return None
        if not isinstance(value, str):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Environment name must be a string.",
            )
        candidate = _sanitize_environment_name(value)
        if not candidate:
            return None
        try:
            CONFIGS_SERVICE.get_config(candidate, config_type="model")
        except ConfigServiceError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Environment '{candidate}' was not found. Please create it before continuing.",
            ) from exc
        except Exception:
            # If we cannot verify, fall back to accepting the candidate.
            pass
        return candidate

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


def _sync_defaults_from_onboarding(store: WebUIStateStore, defaults: WebUIDefaults) -> WebUIDefaults:
    """Apply onboarding answers to defaults when sync is enabled."""

    if not getattr(defaults, "sync_onboarding_defaults", False):
        return defaults

    try:
        onboarding_state = store.load_onboarding()
    except Exception as exc:
        logger.debug("Failed to load onboarding state during defaults sync: %s", exc, exc_info=True)
        return defaults

    opt_out = set(getattr(defaults, "onboarding_sync_opt_out", []) or [])
    changed = False

    for step in _resolve_step_definitions():
        if not step.applies_to_default or step.id in opt_out or step.applies_to_default in opt_out:
            continue
        stored_step = onboarding_state.steps.get(step.id)
        if stored_step is None:
            continue
        if stored_step.value is None:
            continue
        before = defaults.__dict__.get(step.applies_to_default)
        _apply_step_to_defaults(defaults, step, stored_step.value)
        after = defaults.__dict__.get(step.applies_to_default)
        if before != after:
            changed = True

    if changed:
        store.save_defaults(defaults)
    return defaults


@router.post("/onboarding/steps/{step_id}")
async def update_onboarding_step(step_id: str, payload: OnboardingStepUpdate) -> Dict[str, object]:
    """Update onboarding progress for a specific step."""
    definition = _ALL_STEPS.get(step_id)
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

    return _build_state_response(state, _resolve_step_definitions())


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
        return _build_state_response(state, _resolve_step_definitions())
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
    show_documentation_links: Optional[bool] = None
    accelerate_overrides: Optional[Dict[str, object]] = None
    git_mirror_enabled: Optional[bool] = None
    git_remote: Optional[str] = None
    git_branch: Optional[str] = None
    git_auto_commit: Optional[bool] = None
    git_require_clean: Optional[bool] = None
    git_push_on_snapshot: Optional[bool] = None
    git_include_untracked: Optional[bool] = None
    sync_onboarding_defaults: Optional[bool] = None
    onboarding_sync_opt_out: Optional[List[str]] = None


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
        if payload.show_documentation_links is not None:
            defaults.show_documentation_links = bool(payload.show_documentation_links)
        if payload.accelerate_overrides is not None:
            defaults.accelerate_overrides = _normalise_accelerate_overrides(payload.accelerate_overrides)
        if payload.git_mirror_enabled is not None:
            defaults.git_mirror_enabled = bool(payload.git_mirror_enabled)
        if payload.git_auto_commit is not None:
            defaults.git_auto_commit = bool(payload.git_auto_commit)
        if payload.git_require_clean is not None:
            defaults.git_require_clean = bool(payload.git_require_clean)
        if payload.git_push_on_snapshot is not None:
            defaults.git_push_on_snapshot = bool(payload.git_push_on_snapshot)
        if payload.git_include_untracked is not None:
            defaults.git_include_untracked = bool(payload.git_include_untracked)
        if payload.git_remote is not None:
            normalized = payload.git_remote.strip() if isinstance(payload.git_remote, str) else ""
            defaults.git_remote = normalized or None
        if payload.git_branch is not None:
            normalized = payload.git_branch.strip() if isinstance(payload.git_branch, str) else ""
            defaults.git_branch = normalized or None
        if payload.sync_onboarding_defaults is not None:
            defaults.sync_onboarding_defaults = bool(payload.sync_onboarding_defaults)
        if payload.onboarding_sync_opt_out is not None:
            opt_out = payload.onboarding_sync_opt_out
            if isinstance(opt_out, str):
                opt_out = [opt_out]
            if isinstance(opt_out, (list, tuple, set)):
                cleaned = []
                for item in opt_out:
                    if not isinstance(item, str):
                        continue
                    candidate = item.strip()
                    if candidate and candidate not in cleaned:
                        cleaned.append(candidate)
                defaults.onboarding_sync_opt_out = cleaned

        # Save updated defaults
        store.save_defaults(defaults)
        try:
            GIT_CONFIG_SERVICE.reset_cache()
        except Exception as exc:  # pragma: no cover - best-effort cache bust
            logger.debug("Failed to reset git config cache after defaults update: %s", exc, exc_info=True)

        # Return updated state
        state = store.load_state()
        return _build_state_response(state, _resolve_step_definitions())
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error


class CollapsedSectionsPayload(BaseModel):
    """Payload for updating collapsed sections state."""

    sections: Dict[str, bool]


@router.get("/ui-state/collapsed-sections/{tab_name}")
async def get_collapsed_sections(tab_name: str) -> Dict[str, bool]:
    """Get collapsed state for sections in a specific tab."""
    store = WebUIStateStore()
    try:
        return store.get_collapsed_sections(tab_name)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error


@router.post("/ui-state/collapsed-sections/{tab_name}")
async def save_collapsed_sections(tab_name: str, payload: CollapsedSectionsPayload) -> Dict[str, str]:
    """Save collapsed state for sections in a specific tab."""
    store = WebUIStateStore()
    try:
        store.save_collapsed_sections(tab_name, payload.sections)
        return {"status": "success", "message": f"Saved collapsed sections for tab '{tab_name}'"}
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error
