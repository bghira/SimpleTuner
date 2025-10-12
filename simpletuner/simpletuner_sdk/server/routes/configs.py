"""Configuration management routes for SimpleTuner WebUI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from simpletuner.simpletuner_sdk.server.services.configs_service import CONFIGS_SERVICE, ConfigServiceError

router = APIRouter(prefix="/api/configs", tags=["configurations"])


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


@router.get("/")
async def list_configs(config_type: str = "model") -> Dict[str, Any]:
    """List all available configurations."""
    return _call_service(CONFIGS_SERVICE.list_configs, config_type)


@router.get("/data-backend-file")
async def get_data_backend_file(path: str) -> Any:
    """Get the contents of a data backend configuration file."""
    return _call_service(CONFIGS_SERVICE.read_data_backend_file, path)


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


@router.get("/{name}")
async def get_config(name: str, config_type: str = "model") -> Dict[str, Any]:
    """Get a specific configuration by name."""
    return _call_service(CONFIGS_SERVICE.get_config, name, config_type)


@router.post("/")
async def create_config(request: ConfigRequest, config_type: str = "model") -> Dict[str, Any]:
    """Create a new configuration."""
    return _call_service(
        CONFIGS_SERVICE.create_config,
        name=request.name,
        config=request.config,
        description=request.description,
        tags=request.tags,
        config_type=config_type,
    )


@router.post("/environments")
async def create_environment(request: EnvironmentCreateRequest) -> Dict[str, Any]:
    """Create a new training environment."""
    return _call_service(CONFIGS_SERVICE.create_environment, request)


@router.put("/{name}")
async def update_config(name: str, request: ConfigRequest, config_type: str = "model") -> Dict[str, Any]:
    """Update an existing configuration."""
    return _call_service(
        CONFIGS_SERVICE.update_config,
        name=name,
        config=request.config,
        description=request.description,
        tags=request.tags,
        config_type=config_type,
    )


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
    return _call_service(CONFIGS_SERVICE.create_environment_dataloader, name, request.path, request.include_defaults)


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
