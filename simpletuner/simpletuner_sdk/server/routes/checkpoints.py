"""Checkpoint management routes for SimpleTuner WebUI."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from simpletuner.simpletuner_sdk.server.services.checkpoints_service import CHECKPOINTS_SERVICE, CheckpointsServiceError

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


def _call_service(func, *args, **kwargs):
    """Execute a service call and translate domain errors to HTTP errors."""
    try:
        return func(*args, **kwargs)
    except CheckpointsServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


@router.get("")
async def list_checkpoints(
    environment: str = Query(..., description="Environment ID (config name)"),
    sort_by: str = Query("step-desc", description="Sort order: step-desc, step-asc, size-desc"),
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
async def preview_cleanup(request: CleanupRequest) -> Dict[str, Any]:
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
async def execute_cleanup(request: CleanupRequest) -> Dict[str, Any]:
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


@router.get("/for-resume")
async def get_checkpoints_for_resume(
    environment: str = Query(..., description="Environment ID (config name)"),
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
async def update_retention_config(request: RetentionConfigUpdate) -> Dict[str, Any]:
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
