"""Checkpoint management routes for SimpleTuner WebUI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from simpletuner.simpletuner_sdk.server.services.checkpoints_service import CHECKPOINTS_SERVICE, CheckpointsServiceError
from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import get_current_user
from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User
from simpletuner.simpletuner_sdk.server.services.huggingface_service import HUGGINGFACE_SERVICE, HuggingfaceServiceError

router = APIRouter(prefix="/api/checkpoints", tags=["checkpoints"])


class ValidateCheckpointRequest(BaseModel):
    """Request model for checkpoint validation."""

    environment: str = Field(..., description="Environment ID (config name)")


class CleanupRequest(BaseModel):
    """Request model for checkpoint cleanup operations."""

    environment: str = Field(..., description="Environment ID (config name)")
    limit: int = Field(..., description="Maximum number of checkpoints to keep", ge=1)


class RetentionConfigUpdate(BaseModel):
    """Request model for updating retention configuration."""

    environment: str = Field(..., description="Environment ID (config name)")
    retention_limit: int = Field(..., description="Maximum number of checkpoints to keep", ge=1)


class UploadCheckpointRequest(BaseModel):
    """Request model for uploading a single checkpoint."""

    environment: str = Field(..., description="Environment ID (config name)")
    repo_id: Optional[str] = Field(None, description="Target repository ID (overrides config)")
    branch: Optional[str] = Field(None, description="Target branch (None for main)")
    subfolder: Optional[str] = Field(None, description="Subfolder path in repo")
    callback_url: Optional[str] = Field(None, description="Webhook URL for progress callbacks")


class UploadCheckpointsRequest(BaseModel):
    """Request model for uploading multiple checkpoints."""

    environment: str = Field(..., description="Environment ID (config name)")
    checkpoint_names: List[str] = Field(..., description="List of checkpoint names to upload")
    repo_id: Optional[str] = Field(None, description="Target repository ID (overrides config)")
    upload_mode: str = Field("single_commit", description="Upload mode: 'single_commit' or 'separate_branches'")
    callback_url: Optional[str] = Field(None, description="Webhook URL for progress callbacks")


def _call_service(func, *args, **kwargs):
    """Execute a service call and translate domain errors to HTTP errors."""
    try:
        return func(*args, **kwargs)
    except CheckpointsServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HuggingfaceServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


@router.get("")
async def list_checkpoints(
    environment: str = Query(..., description="Environment ID (config name)"),
    sort_by: str = Query("step-desc", description="Sort order: step-desc, step-asc, size-desc"),
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    List all checkpoints for an environment.

    Args:
        environment: Environment ID (config name).
        sort_by: Sort order - one of "step-desc", "step-asc", "size-desc".

    Returns:
        Dictionary with checkpoint list and metadata.
    """
    valid_sort_options = ["step-desc", "step-asc", "size-desc"]
    if sort_by not in valid_sort_options:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sort_by value. Must be one of: {', '.join(valid_sort_options)}",
        )

    return _call_service(CHECKPOINTS_SERVICE.list_checkpoints, environment, sort_by)


@router.post("/{checkpoint_name}/validate")
async def validate_checkpoint(
    checkpoint_name: str,
    request: ValidateCheckpointRequest,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Validate a checkpoint for resuming training.

    Args:
        checkpoint_name: Name of the checkpoint (e.g., "checkpoint-1000").
        request: Validation request with environment ID.

    Returns:
        Dictionary with validation results.
    """
    return _call_service(
        CHECKPOINTS_SERVICE.validate_checkpoint,
        request.environment,
        checkpoint_name,
    )


@router.post("/cleanup/preview")
async def preview_cleanup(
    request: CleanupRequest,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Preview what checkpoints would be deleted by cleanup operation.

    Args:
        request: Cleanup request with environment ID and limit.

    Returns:
        Dictionary with checkpoints that would be removed.
    """
    return _call_service(
        CHECKPOINTS_SERVICE.preview_cleanup,
        request.environment,
        request.limit,
    )


@router.post("/cleanup/execute")
async def execute_cleanup(
    request: CleanupRequest,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Execute cleanup operation to remove old checkpoints.

    Args:
        request: Cleanup request with environment ID and limit.

    Returns:
        Dictionary with cleanup results.
    """
    return _call_service(
        CHECKPOINTS_SERVICE.execute_cleanup,
        request.environment,
        request.limit,
    )


@router.delete("/{checkpoint_name}")
async def delete_checkpoint(
    checkpoint_name: str,
    environment: str = Query(..., description="Environment ID (config name)"),
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Remove a checkpoint directory for an environment."""
    return _call_service(
        CHECKPOINTS_SERVICE.delete_checkpoint,
        environment,
        checkpoint_name,
    )


@router.get("/for-resume")
async def get_checkpoints_for_resume(
    environment: str = Query(..., description="Environment ID (config name)"),
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get checkpoints formatted for resume dropdown.

    Returns simplified list with just name and step for UI dropdown.

    Args:
        environment: Environment ID (config name).

    Returns:
        Dictionary with simplified checkpoint list.
    """
    return _call_service(CHECKPOINTS_SERVICE.get_checkpoints_for_resume, environment)


@router.get("/retention")
async def get_retention_config(
    environment: str = Query(..., description="Environment ID (config name)"),
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get checkpoint retention configuration for an environment.

    Args:
        environment: Environment ID (config name).

    Returns:
        Dictionary with retention configuration.
    """
    return _call_service(CHECKPOINTS_SERVICE.get_retention_config, environment)


@router.put("/retention")
async def update_retention_config(
    request: RetentionConfigUpdate,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Update checkpoint retention configuration for an environment.

    Args:
        request: Retention configuration update request.

    Returns:
        Dictionary with updated retention configuration.
    """
    return _call_service(
        CHECKPOINTS_SERVICE.update_retention_config,
        request.environment,
        request.retention_limit,
    )


@router.post("/{checkpoint_name}/upload")
async def upload_checkpoint(
    checkpoint_name: str,
    request: UploadCheckpointRequest,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Upload a single checkpoint to HuggingFace Hub.

    Args:
        checkpoint_name: Name of the checkpoint to upload.
        request: Upload request with configuration.

    Returns:
        Dictionary with upload task information.
    """
    return _call_service(
        HUGGINGFACE_SERVICE.upload_checkpoint,
        request.environment,
        checkpoint_name,
        request.repo_id,
        request.branch,
        request.subfolder,
        request.callback_url,
    )


@router.post("/upload")
async def upload_checkpoints(
    request: UploadCheckpointsRequest,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Upload multiple checkpoints to HuggingFace Hub.

    Args:
        request: Upload request with list of checkpoints and configuration.

    Returns:
        Dictionary with upload task information.
    """
    return _call_service(
        HUGGINGFACE_SERVICE.upload_checkpoints,
        request.environment,
        request.checkpoint_names,
        request.repo_id,
        request.upload_mode,
        request.callback_url,
    )


@router.get("/upload/{task_id}/status")
async def get_upload_status(
    task_id: str,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get the status of an upload task.

    Args:
        task_id: The upload task ID.

    Returns:
        Dictionary with task status information.
    """
    return _call_service(HUGGINGFACE_SERVICE.get_task_status, task_id)


@router.post("/upload/{task_id}/cancel")
async def cancel_upload(
    task_id: str,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Cancel an upload task.

    Args:
        task_id: The upload task ID.

    Returns:
        Dictionary with cancellation status.
    """
    return _call_service(HUGGINGFACE_SERVICE.cancel_task, task_id)
