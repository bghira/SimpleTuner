"""Configuration management routes for SimpleTuner WebUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore, ConfigMetadata
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore

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


def _get_store(config_type: str = "model") -> ConfigStore:
    """Get the configuration store instance with user defaults if available.

    Args:
        config_type: Type of configuration ('model', 'dataloader', 'webhook', or 'lycoris'). Defaults to 'model'.
    """
    # Try to load user's preferred configs directory
    try:
        state_store = WebUIStateStore()
        defaults = state_store.load_defaults()
        if defaults.configs_dir:
            # Expand the path before passing to ConfigStore (though ConfigStore also does this now)
            import os
            expanded_dir = os.path.expanduser(defaults.configs_dir)
            return ConfigStore(config_dir=expanded_dir, config_type=config_type)
    except Exception:
        # Fall back to default behavior if loading defaults fails
        pass

    return ConfigStore(config_type=config_type)


@router.get("/")
async def list_configs(config_type: str = "model") -> Dict[str, Any]:
    """List all available configurations.

    Args:
        config_type: Type of configuration to list ('model' or 'dataloader'). Defaults to 'model'.

    Returns:
        List of configuration metadata.
    """
    store = _get_store(config_type)
    configs = store.list_configs()
    active = store.get_active_config() if config_type == "model" else None

    return {"configs": configs, "active": active, "count": len(configs), "config_type": config_type}


@router.get("/data-backend-file")
async def get_data_backend_file(path: str) -> Any:
    """Get the contents of a data backend configuration file.

    Args:
        path: Path to the data backend config file (can be relative or absolute)

    Returns:
        JSON contents of the data backend config file
    """
    import json
    from simpletuner.simpletuner_sdk.server.utils.paths import resolve_config_path

    def _is_relative_to(candidate: Path, base: Path) -> bool:
        try:
            candidate.relative_to(base)
            return True
        except ValueError:
            return False

    try:
        store = _get_store()

        user_config_dir = store.config_dir

        resolved_path = resolve_config_path(
            path,
            config_dir=user_config_dir,
            check_cwd_first=True
        )

        if not resolved_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data backend config file not found: {path}"
            )

        allowed_dirs = [Path.cwd()]
        try:
            allowed_dirs.append(Path(user_config_dir))
        except Exception:
            pass

        resolved_real = resolved_path.resolve()
        if not any(_is_relative_to(resolved_real, directory.resolve()) for directory in allowed_dirs):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resolved path is outside allowed directories",
            )

        with open(resolved_real, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON in data backend config file: {str(e)}"
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading data backend config file: {str(e)}"
        )


@router.get("/templates")
async def list_templates() -> Dict[str, Any]:
    """List all available configuration templates.

    Returns:
        List of template metadata.
    """
    store = _get_store()
    templates = store.list_templates()

    return {"templates": templates, "count": len(templates)}


@router.get("/active")
async def get_active_config() -> Dict[str, Any]:
    """Get the currently active configuration.

    Returns:
        Active configuration name and data.
    """
    store = _get_store()
    active_name = store.get_active_config()

    if not active_name:
        return {"name": None, "config": {}, "metadata": None}

    try:
        config, metadata = store.load_config(active_name)
        return {"name": active_name, "config": config, "metadata": metadata.model_dump()}
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Active configuration '{active_name}' not found")


@router.get("/{name}")
async def get_config(name: str, config_type: str = "model") -> Dict[str, Any]:
    """Get a specific configuration by name.

    Args:
        name: Configuration name.
        config_type: Type of configuration ('model', 'dataloader', 'webhook', or 'lycoris'). Defaults to 'model'.

    Returns:
        Configuration data and metadata.
    """
    store = _get_store(config_type)

    try:
        config, metadata = store.load_config(name)
        return {"name": name, "config": config, "metadata": metadata.model_dump(), "config_type": config_type}
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Configuration '{name}' not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@router.post("/")
async def create_config(request: ConfigRequest, config_type: str = "model") -> Dict[str, Any]:
    """Create a new configuration.

    Args:
        request: Configuration data.
        config_type: Type of configuration ('model', 'dataloader', 'webhook', or 'lycoris'). Defaults to 'model'.

    Returns:
        Created configuration metadata.
    """
    store = _get_store(config_type)

    # Validate the configuration
    validation = store.validate_config(request.config)
    if not validation.is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Configuration validation failed",
                "errors": validation.errors,
                "warnings": validation.warnings,
            },
        )

    # Create metadata
    metadata = ConfigMetadata(
        name=request.name,
        description=request.description,
        tags=request.tags,
        created_at="",  # Will be set by store
        modified_at="",  # Will be set by store
    )

    try:
        saved_metadata = store.save_config(request.name, request.config, metadata, overwrite=False)
        return {
            "message": f"Configuration '{request.name}' created successfully",
            "metadata": saved_metadata.model_dump(),
            "validation": {"warnings": validation.warnings, "suggestions": validation.suggestions},
        }
    except FileExistsError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Configuration '{request.name}' already exists")


@router.put("/{name}")
async def update_config(name: str, request: ConfigRequest, config_type: str = "model") -> Dict[str, Any]:
    """Update an existing configuration.

    Args:
        name: Configuration name.
        request: Updated configuration data.
        config_type: Type of configuration ('model', 'dataloader', 'webhook', or 'lycoris'). Defaults to 'model'.

    Returns:
        Updated configuration metadata.
    """
    store = _get_store(config_type)

    # Validate the configuration
    validation = store.validate_config(request.config)
    if not validation.is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Configuration validation failed",
                "errors": validation.errors,
                "warnings": validation.warnings,
            },
        )

    try:
        # Load existing metadata
        _, existing_metadata = store.load_config(name)

        # Update metadata
        existing_metadata.description = request.description or existing_metadata.description
        existing_metadata.tags = request.tags if request.tags else existing_metadata.tags

        # Save updated config
        saved_metadata = store.save_config(name, request.config, existing_metadata, overwrite=True)

        return {
            "message": f"Configuration '{name}' updated successfully",
            "metadata": saved_metadata.model_dump(),
            "validation": {"warnings": validation.warnings, "suggestions": validation.suggestions},
        }
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Configuration '{name}' not found")


@router.delete("/{name}")
async def delete_config(name: str, config_type: str = "model") -> Dict[str, Any]:
    """Delete a configuration.

    Args:
        name: Configuration name.
        config_type: Type of configuration ('model', 'dataloader', 'webhook', or 'lycoris'). Defaults to 'model'.

    Returns:
        Deletion confirmation.
    """
    store = _get_store(config_type)

    # Don't allow deleting the active config
    if store.get_active_config() == name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete the active configuration")

    if store.delete_config(name):
        return {"message": f"Configuration '{name}' deleted successfully"}
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Configuration '{name}' not found")


@router.post("/{name}/rename")
async def rename_config(name: str, request: ConfigRenameRequest, config_type: str = "model") -> Dict[str, Any]:
    """Rename a configuration.

    Args:
        name: Current configuration name.
        request: New name.
        config_type: Type of configuration ('model', 'dataloader', 'webhook', or 'lycoris'). Defaults to 'model'.

    Returns:
        Updated metadata.
    """
    store = _get_store(config_type)

    try:
        metadata = store.rename_config(name, request.new_name)
        return {"message": f"Configuration renamed from '{name}' to '{request.new_name}'", "metadata": metadata.model_dump()}
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Configuration '{name}' not found")
    except FileExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=f"Configuration '{request.new_name}' already exists"
        )


@router.post("/{name}/copy")
async def copy_config(name: str, request: ConfigCopyRequest, config_type: str = "model") -> Dict[str, Any]:
    """Copy a configuration.

    Args:
        name: Source configuration name.
        request: Target name.
        config_type: Type of configuration ('model', 'dataloader', 'webhook', or 'lycoris'). Defaults to 'model'.

    Returns:
        New configuration metadata.
    """
    store = _get_store(config_type)

    try:
        metadata = store.copy_config(name, request.target_name)
        return {"message": f"Configuration '{name}' copied to '{request.target_name}'", "metadata": metadata.model_dump()}
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Configuration '{name}' not found")
    except FileExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=f"Configuration '{request.target_name}' already exists"
        )


@router.post("/{name}/activate")
async def activate_config(name: str) -> Dict[str, Any]:
    """Set a configuration as active.

    Args:
        name: Configuration name.

    Returns:
        Activation confirmation.
    """
    store = _get_store()

    try:
        store.set_active_config(name)
        return {"message": f"Configuration '{name}' is now active", "active": name}
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Configuration '{name}' not found")


@router.get("/{name}/export")
async def export_config(name: str, include_metadata: bool = True, config_type: str = "model") -> Dict[str, Any]:
    """Export a configuration for sharing.

    Args:
        name: Configuration name.
        include_metadata: Whether to include metadata.
        config_type: Type of configuration ('model', 'dataloader', 'webhook', or 'lycoris'). Defaults to 'model'.

    Returns:
        Configuration data.
    """
    store = _get_store(config_type)

    try:
        data = store.export_config(name, include_metadata)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Configuration '{name}' not found")


@router.post("/import")
async def import_config(request: ConfigImportRequest, config_type: str = "model") -> Dict[str, Any]:
    """Import a configuration.

    Args:
        request: Configuration data to import.
        config_type: Type of configuration ('model', 'dataloader', 'webhook', or 'lycoris'). Defaults to 'model'.

    Returns:
        Imported configuration metadata.
    """
    store = _get_store(config_type)

    try:
        metadata = store.import_config(request.data, request.name, request.overwrite)
        return {"message": f"Configuration imported as '{metadata.name}'", "metadata": metadata.model_dump()}
    except FileExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Failed to import configuration: {str(e)}"
        )


@router.post("/from-template")
async def create_from_template(request: ConfigFromTemplateRequest, config_type: str = "model") -> Dict[str, Any]:
    """Create a configuration from a template.

    Args:
        request: Template and new config names.
        config_type: Type of configuration ('model', 'dataloader', 'webhook', or 'lycoris'). Defaults to 'model'.

    Returns:
        Created configuration metadata.
    """
    store = _get_store(config_type)

    try:
        metadata = store.create_from_template(request.template_name, request.config_name)
        return {
            "message": f"Configuration '{request.config_name}' created from template '{request.template_name}'",
            "metadata": metadata.model_dump(),
        }
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template '{request.template_name}' not found")
    except FileExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=f"Configuration '{request.config_name}' already exists"
        )


@router.post("/{name}/validate")
async def validate_config(name: str, config_type: str = "model") -> Dict[str, Any]:
    """Validate a configuration.

    Args:
        name: Configuration name.
        config_type: Type of configuration ('model', 'dataloader', 'webhook', or 'lycoris'). Defaults to 'model'.

    Returns:
        Validation results.
    """
    store = _get_store(config_type)

    try:
        config, _ = store.load_config(name)
        validation = store.validate_config(config)

        return {
            "name": name,
            "is_valid": validation.is_valid,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "suggestions": validation.suggestions,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Configuration '{name}' not found")


@router.post("/validate")
async def validate_config_data(config: Dict[str, Any], config_type: str = "model") -> Dict[str, Any]:
    """Validate configuration data without saving.

    Args:
        config: Configuration data to validate.
        config_type: Type of configuration ('model', 'dataloader', 'webhook', or 'lycoris'). Defaults to 'model'.

    Returns:
        Validation results.
    """
    store = _get_store(config_type)
    validation = store.validate_config(config)

    return {
        "is_valid": validation.is_valid,
        "errors": validation.errors,
        "warnings": validation.warnings,
        "suggestions": validation.suggestions,
    }
