"""Checkpoint management routes for SimpleTuner WebUI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from simpletuner.simpletuner_sdk.server.services.checkpoint_inference_service import (
    CHECKPOINT_INFERENCE_SERVICE,
    CheckpointInferenceServiceError,
)
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


class InferenceSettings(BaseModel):
    seed: Optional[int] = None
    num_inference_steps: Optional[int] = Field(None, ge=1, le=1000)
    guidance_scale: Optional[float] = Field(None, ge=0, le=100)
    validation_resolution: Optional[str] = Field(None, min_length=1, max_length=256)


class StartInferenceRequest(BaseModel):
    environment: str
    checkpoint_names: List[str] = Field(..., min_length=1)
    use_configured_prompt: bool = True
    use_builtin_library: bool = False
    user_library_filename: Optional[str] = None
    custom_prompts: List[str] = Field(default_factory=list)
    filename_style: str = "descriptive"
    keep_loaded: bool = False
    streaming_preview: bool = False
    idle_timeout_minutes: int = Field(15, ge=1, le=1440)
    settings: InferenceSettings = Field(default_factory=InferenceSettings)


class GenerateInferenceRequest(BaseModel):
    environment: str
    custom_prompts: List[str] = Field(..., min_length=1)
    filename_style: Optional[str] = None
    settings: InferenceSettings = Field(default_factory=InferenceSettings)


class InferenceSessionRequest(BaseModel):
    environment: str


class DeleteInferenceHistoryRequest(BaseModel):
    environment: str
    media_paths: List[str] = Field(..., min_length=1, max_length=100)


def _call_service(func, *args, **kwargs):
    """Execute a service call and translate domain errors to HTTP errors."""
    try:
        return func(*args, **kwargs)
    except CheckpointsServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HuggingfaceServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except CheckpointInferenceServiceError as exc:
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


@router.get("/inference/prompt-sources")
async def inference_prompt_sources(
    environment: str = Query(...),
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    return _call_service(CHECKPOINT_INFERENCE_SERVICE.prompt_sources, environment)


@router.get("/inference/active")
async def active_inference_session(
    environment: str = Query(...),
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    return {"session": _call_service(CHECKPOINT_INFERENCE_SERVICE.active_environment_session, environment)}


@router.post("/inference/start")
async def start_inference(
    request: StartInferenceRequest,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    return _call_service(
        CHECKPOINT_INFERENCE_SERVICE.start,
        environment=request.environment,
        checkpoint_names=request.checkpoint_names,
        use_configured_prompt=request.use_configured_prompt,
        use_builtin_library=request.use_builtin_library,
        user_library_filename=request.user_library_filename,
        custom_prompts=request.custom_prompts,
        filename_style=request.filename_style,
        keep_loaded=request.keep_loaded,
        streaming_preview=request.streaming_preview,
        idle_timeout_minutes=request.idle_timeout_minutes,
        settings=request.settings.model_dump(exclude_none=True),
    )


@router.get("/inference/{session_id}/status")
async def inference_status(
    session_id: str,
    environment: str = Query(...),
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    return _call_service(CHECKPOINT_INFERENCE_SERVICE.status, environment, session_id)


@router.post("/inference/{session_id}/generate")
async def generate_inference(
    session_id: str,
    request: GenerateInferenceRequest,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    return _call_service(
        CHECKPOINT_INFERENCE_SERVICE.generate,
        environment=request.environment,
        session_id=session_id,
        custom_prompts=request.custom_prompts,
        filename_style=request.filename_style,
        settings=request.settings.model_dump(exclude_none=True),
    )


@router.post("/inference/{session_id}/unload")
async def unload_inference(
    session_id: str,
    request: InferenceSessionRequest,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    return _call_service(CHECKPOINT_INFERENCE_SERVICE.stop, request.environment, session_id, cancel=False)


@router.post("/inference/{session_id}/cancel")
async def cancel_inference(
    session_id: str,
    request: InferenceSessionRequest,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    return _call_service(CHECKPOINT_INFERENCE_SERVICE.stop, request.environment, session_id, cancel=True)


@router.get("/inference/history")
async def inference_history(
    environment: str = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    return _call_service(CHECKPOINT_INFERENCE_SERVICE.history, environment, page=page, page_size=page_size)


@router.delete("/inference/history")
async def delete_inference_history(
    request: DeleteInferenceHistoryRequest,
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    return _call_service(CHECKPOINT_INFERENCE_SERVICE.delete_history, request.environment, request.media_paths)


@router.get("/inference/media/{media_path:path}")
async def inference_media(
    media_path: str,
    environment: str = Query(...),
    _user: User = Depends(get_current_user),
):
    path = _call_service(CHECKPOINT_INFERENCE_SERVICE.media_path, environment, media_path)
    return FileResponse(path)


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
