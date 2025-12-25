"""Configuration management routes for SimpleTuner WebUI."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from simpletuner.simpletuner_sdk.server.services.configs_service import CONFIGS_SERVICE, ConfigServiceError
from simpletuner.simpletuner_sdk.server.services.git_config_service import (
    GIT_CONFIG_SERVICE,
    GitConfigError,
    SnapshotPreferences,
)
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore

router = APIRouter(prefix="/api/configs", tags=["configurations"])
logger = logging.getLogger(__name__)


class ConfigRequest(BaseModel):
    """Request model for configuration operations."""

    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    tags: List[str] = Field(default_factory=list)


class ConfigRenameRequest(BaseModel):
    """Request model for renaming configurations."""

    new_name: str


class ConfigCopyRequest(BaseModel):
    """Request model for copying configurations."""

    target_name: str


class ConfigImportRequest(BaseModel):
    """Request model for importing configurations."""

    data: Dict[str, Any]
    name: Optional[str] = None
    overwrite: bool = False


class ConfigFromTemplateRequest(BaseModel):
    """Request model for creating from template."""

    template_name: str
    config_name: str


class EnvironmentCreateRequest(BaseModel):
    """Request parameters for creating a new environment."""

    name: str
    model_family: str
    model_flavour: Optional[str] = None
    model_type: Optional[str] = None
    lora_type: Optional[str] = None
    description: Optional[str] = None
    example: Optional[str] = None
    dataloader_path: Optional[str] = None
    create_dataloader: bool = True


class EnvironmentDataloaderRequest(BaseModel):
    """Request parameters for creating a dataloader config for an environment."""

    path: Optional[str] = None
    include_defaults: bool = True


class LycorisConfigRequest(BaseModel):
    """Request model for Lycoris configuration operations."""

    config: Dict[str, Any]


def _call_service(func, *args, **kwargs):
    """Execute a service call and translate domain errors to HTTP errors."""
    try:
        return func(*args, **kwargs)
    except ConfigServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


def _normalize_dataset(dataset: Any) -> Any:
    if not isinstance(dataset, dict):
        return dataset
    clone = dict(dataset)
    conditioning = clone.get("conditioning")
    if isinstance(conditioning, dict):
        clone["conditioning"] = [conditioning]
    elif conditioning is None:
        clone["conditioning"] = []
    return clone


def _normalize_conditioning_payload(payload: Any) -> Any:
    if isinstance(payload, list):
        return [_normalize_dataset(entry) for entry in payload]
    if isinstance(payload, dict):
        return _normalize_dataset(payload)
    return payload


def _git_preferences() -> tuple[bool, SnapshotPreferences]:
    """Load git-related WebUI defaults."""
    try:
        defaults = WebUIStateStore().load_defaults()
    except Exception:
        return False, SnapshotPreferences()

    enabled = bool(getattr(defaults, "git_mirror_enabled", False))
    prefs = SnapshotPreferences(
        auto_commit=bool(getattr(defaults, "git_auto_commit", False)),
        require_clean=bool(getattr(defaults, "git_require_clean", False)),
        include_untracked=bool(getattr(defaults, "git_include_untracked", False)),
        push_on_snapshot=bool(getattr(defaults, "git_push_on_snapshot", False)),
        default_message=None,
    )
    return enabled, prefs


def _enforce_git_requirements(config_type: str, enabled: bool, prefs: SnapshotPreferences) -> None:
    if not enabled or not prefs.require_clean:
        return
    status = GIT_CONFIG_SERVICE.is_git_ready(config_type)
    if not status.repo_present:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Git repository is not initialized for the configs directory.",
        )
    if status.dirty_paths:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Working tree has uncommitted changes. Commit or stash before saving.",
        )


def _maybe_snapshot_on_save(
    name: str,
    config_type: str,
    enabled: bool,
    prefs: SnapshotPreferences,
    response: Dict[str, Any],
    action: str = "update",
) -> Dict[str, Any]:
    if not enabled or not prefs.auto_commit:
        return response
    message = f"env:{name} {action}"
    try:
        snapshot = GIT_CONFIG_SERVICE.snapshot_on_save(name, config_type, prefs, message=message)
        if snapshot:
            response["git_snapshot"] = snapshot
            if "push_error" in snapshot:
                response["git_push_error"] = snapshot["push_error"]
    except GitConfigError as exc:
        response["git_error"] = exc.message
        logger.warning("Git snapshot on save failed for '%s' (%s): %s", name, config_type, exc.message)
    return response


@router.get("/")
async def list_configs(config_type: str = "model") -> Dict[str, Any]:
    """List all available configurations."""
    return _call_service(CONFIGS_SERVICE.list_configs, config_type)


@router.get("/data-backend-file")
async def get_data_backend_file(path: str) -> Any:
    """Get the contents of a data backend configuration file."""
    data = _call_service(CONFIGS_SERVICE.read_data_backend_file, path)
    return _normalize_conditioning_payload(data)


@router.get("/templates")
async def list_templates() -> Dict[str, Any]:
    """List all available configuration templates."""
    return _call_service(CONFIGS_SERVICE.list_templates)


@router.get("/examples")
async def list_examples() -> Dict[str, Any]:
    """List available example environments."""
    return _call_service(CONFIGS_SERVICE.list_examples)


@router.get("/project-name")
async def generate_project_name() -> Dict[str, str]:
    """Generate a random project name slug."""
    return _call_service(CONFIGS_SERVICE.generate_project_name)


@router.get("/active")
async def get_active_config() -> Dict[str, Any]:
    """Get the currently active configuration."""
    return _call_service(CONFIGS_SERVICE.get_active_config)


# Webhook Configuration endpoints
# Note: These routes must come before /{name} to avoid path parameter conflicts
class WebhookConfigRequest(BaseModel):
    """Request model for Webhook configuration operations."""

    name: str
    config: Dict[str, Any]


@router.post("/webhooks/validate")
async def validate_webhook_config(request: LycorisConfigRequest) -> Dict[str, Any]:
    """Validate webhook configuration without saving."""
    return _call_service(CONFIGS_SERVICE.validate_webhook_config, request.config)


@router.post("/webhooks")
async def create_webhook_config(request: WebhookConfigRequest) -> Dict[str, Any]:
    """Create a new webhook configuration."""
    # Validate first
    validation = _call_service(CONFIGS_SERVICE.validate_webhook_config, request.config)
    if not validation["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Invalid webhook configuration", "errors": validation["errors"]},
        )
    result = _call_service(CONFIGS_SERVICE.save_webhook_config, request.name, request.config)
    return {
        "message": f"Webhook configuration '{request.name}' created",
        **result,
    }


@router.get("/webhooks/{name}")
async def get_webhook_config(name: str) -> Dict[str, Any]:
    """Get a webhook configuration by name."""
    webhook_config = _call_service(CONFIGS_SERVICE.get_webhook_config, name)
    if webhook_config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Webhook configuration '{name}' not found",
        )
    return {"name": name, "config": webhook_config}


@router.put("/webhooks/{name}")
async def update_webhook_config(name: str, request: LycorisConfigRequest) -> Dict[str, Any]:
    """Update an existing webhook configuration."""
    # Validate first
    validation = _call_service(CONFIGS_SERVICE.validate_webhook_config, request.config)
    if not validation["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Invalid webhook configuration", "errors": validation["errors"]},
        )
    result = _call_service(CONFIGS_SERVICE.save_webhook_config, name, request.config)
    return {
        "message": f"Webhook configuration '{name}' updated",
        **result,
    }


@router.delete("/webhooks/{name}")
async def delete_webhook_config(name: str) -> Dict[str, Any]:
    """Delete a webhook configuration."""
    result = _call_service(CONFIGS_SERVICE.delete_webhook_config, name)
    return {
        "message": f"Webhook configuration '{name}' deleted",
        **result,
    }


@router.post("/webhooks/{name}/test")
async def test_webhook_config(name: str) -> Dict[str, Any]:
    """Test a webhook configuration by sending a test message."""
    result = _call_service(CONFIGS_SERVICE.test_webhook_config, name)
    return result


@router.get("/{name}")
async def get_config(name: str, config_type: str = "model") -> Dict[str, Any]:
    """Get a specific configuration by name."""
    return _call_service(CONFIGS_SERVICE.get_config, name, config_type)


@router.post("/")
async def create_config(request: ConfigRequest, config_type: str = "model") -> Dict[str, Any]:
    """Create a new configuration."""
    git_enabled, prefs = _git_preferences()
    _enforce_git_requirements(config_type, git_enabled, prefs)
    result = _call_service(
        CONFIGS_SERVICE.create_config,
        name=request.name,
        config=request.config,
        description=request.description,
        tags=request.tags,
        config_type=config_type,
    )
    return _maybe_snapshot_on_save(request.name, config_type, git_enabled, prefs, result, action="create")


@router.post("/environments")
async def create_environment(request: EnvironmentCreateRequest) -> Dict[str, Any]:
    """Create a new training environment."""
    git_enabled, prefs = _git_preferences()
    _enforce_git_requirements("model", git_enabled, prefs)
    result = _call_service(CONFIGS_SERVICE.create_environment, request)
    env_name = getattr(request, "name", None) or ""
    return _maybe_snapshot_on_save(env_name, "model", git_enabled, prefs, result, action="create")


@router.put("/{name}")
async def update_config(name: str, request: ConfigRequest, config_type: str = "model") -> Dict[str, Any]:
    """Update an existing configuration."""
    git_enabled, prefs = _git_preferences()
    _enforce_git_requirements(config_type, git_enabled, prefs)
    result = _call_service(
        CONFIGS_SERVICE.update_config,
        name=name,
        config=request.config,
        description=request.description,
        tags=request.tags,
        config_type=config_type,
    )
    return _maybe_snapshot_on_save(name, config_type, git_enabled, prefs, result, action="update")


@router.delete("/{name}")
async def delete_config(name: str, config_type: str = "model") -> Dict[str, Any]:
    """Delete a configuration."""
    return _call_service(CONFIGS_SERVICE.delete_config, name, config_type)


@router.post("/{name}/rename")
async def rename_config(name: str, request: ConfigRenameRequest, config_type: str = "model") -> Dict[str, Any]:
    """Rename a configuration."""
    return _call_service(CONFIGS_SERVICE.rename_config, name, request.new_name, config_type)


@router.post("/{name}/copy")
async def copy_config(name: str, request: ConfigCopyRequest, config_type: str = "model") -> Dict[str, Any]:
    """Copy a configuration."""
    return _call_service(CONFIGS_SERVICE.copy_config, name, request.target_name, config_type)


@router.post("/{name}/dataloader")
async def create_environment_dataloader(name: str, request: EnvironmentDataloaderRequest) -> Dict[str, Any]:
    """Create a dataloader configuration for an environment."""
    git_enabled, prefs = _git_preferences()
    _enforce_git_requirements("dataloader", git_enabled, prefs)
    result = _call_service(CONFIGS_SERVICE.create_environment_dataloader, name, request.path, request.include_defaults)
    return _maybe_snapshot_on_save(name, "dataloader", git_enabled, prefs, result, action="create-dataloader")


@router.get("/dataloader/content")
async def get_dataloader_content(path: str) -> Any:
    """Get the JSON contents of a dataloader configuration file."""
    data = _call_service(CONFIGS_SERVICE.read_data_backend_file, path)
    return _normalize_conditioning_payload(data)


@router.delete("/dataloader")
async def delete_dataloader_config(path: str) -> Dict[str, Any]:
    """Delete a dataloader configuration file."""
    return _call_service(CONFIGS_SERVICE.delete_dataloader_config, path)


@router.post("/{name}/activate")
async def activate_config(name: str) -> Dict[str, Any]:
    """Set a configuration as active."""
    return _call_service(CONFIGS_SERVICE.activate_config, name)


@router.get("/{name}/export")
async def export_config(name: str, include_metadata: bool = True, config_type: str = "model") -> Dict[str, Any]:
    """Export a configuration for sharing."""
    return _call_service(CONFIGS_SERVICE.export_config, name, include_metadata, config_type)


@router.post("/import")
async def import_config(request: ConfigImportRequest, config_type: str = "model") -> Dict[str, Any]:
    """Import a configuration."""
    return _call_service(
        CONFIGS_SERVICE.import_config,
        data=request.data,
        name=request.name,
        overwrite=request.overwrite,
        config_type=config_type,
    )


@router.post("/from-template")
async def create_from_template(request: ConfigFromTemplateRequest, config_type: str = "model") -> Dict[str, Any]:
    """Create a configuration from a template."""
    return _call_service(
        CONFIGS_SERVICE.create_from_template,
        request.template_name,
        request.config_name,
        config_type,
    )


@router.post("/{name}/validate")
async def validate_config(name: str, config_type: str = "model") -> Dict[str, Any]:
    """Validate a configuration."""
    return _call_service(CONFIGS_SERVICE.validate_config, name, config_type)


@router.post("/validate")
async def validate_config_data(config: Dict[str, Any], config_type: str = "model") -> Dict[str, Any]:
    """Validate configuration data without saving."""
    return _call_service(CONFIGS_SERVICE.validate_config_data, config, config_type)


@router.get("/environments/{environment_id}/lycoris")
async def get_lycoris_config(environment_id: str) -> Dict[str, Any]:
    """Get Lycoris configuration for an environment."""
    lycoris_config = _call_service(CONFIGS_SERVICE.get_lycoris_config, environment_id)
    if lycoris_config is None:
        raise HTTPException(
            status_code=404,
            detail=f"No Lycoris configuration found for environment '{environment_id}'",
        )
    return {"environment_id": environment_id, "config": lycoris_config}


@router.put("/environments/{environment_id}/lycoris")
async def save_lycoris_config(environment_id: str, request: LycorisConfigRequest) -> Dict[str, Any]:
    """Save Lycoris configuration for an environment."""
    result = _call_service(CONFIGS_SERVICE.save_lycoris_config, environment_id, request.config)
    return {
        "message": f"Lycoris configuration saved for environment '{environment_id}'",
        "environment_id": environment_id,
        **result,
    }


@router.post("/lycoris/validate")
async def validate_lycoris_config(request: LycorisConfigRequest) -> Dict[str, Any]:
    """Validate Lycoris configuration without saving."""
    return _call_service(CONFIGS_SERVICE.validate_lycoris_config, request.config)
