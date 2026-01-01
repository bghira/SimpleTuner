"""Queue management endpoints for training jobs.

NOTE: This module was moved from routes/cloud/queue.py to become a top-level
global route, as job queuing is a global concept in SimpleTuner.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..services.cloud.auth import get_current_user, get_optional_user, require_permission
from ..services.cloud.auth.models import User
from ..services.cloud.background_tasks import get_queue_scheduler
from ..services.cloud.queue import QueueEntry, QueuePriority, QueueScheduler, QueueStatus, QueueStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/queue", tags=["queue"])


def get_scheduler() -> QueueScheduler:
    """Get the queue scheduler from the background task manager.

    Raises:
        HTTPException: If the scheduler is not running.
    """
    scheduler = get_queue_scheduler()
    if scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue scheduler not running. Server may still be starting.",
        )
    return scheduler


# --- Request/Response Models ---


class QueueEntryResponse(BaseModel):
    """Response for a single queue entry."""

    id: int
    job_id: str
    user_id: Optional[int] = None
    provider: str
    config_name: Optional[str] = None
    priority: int
    priority_name: str
    priority_override: Optional[int] = None
    effective_priority: int
    status: str
    position: int
    queued_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_cost: float = 0.0
    requires_approval: bool = False
    attempt: int = 1
    error_message: Optional[str] = None


class QueueListResponse(BaseModel):
    """Response for queue listing."""

    entries: List[QueueEntryResponse]
    total_pending: int = 0
    total_running: int = 0


class LocalGPUStats(BaseModel):
    """Local GPU allocation statistics."""

    running_jobs: int = 0
    pending_jobs: int = 0
    allocated_gpus: List[int] = Field(default_factory=list)
    available_gpus: List[int] = Field(default_factory=list)
    total_gpus: int = 0
    max_concurrent_gpus: Optional[int] = None
    max_concurrent_jobs: int = 1


class QueueStatsResponse(BaseModel):
    """Response for queue statistics."""

    by_status: Dict[str, int] = Field(default_factory=dict)
    by_user: Dict[str, int] = Field(default_factory=dict)
    queue_depth: int = 0
    running: int = 0
    avg_wait_seconds: Optional[float] = None
    max_concurrent: int = 5
    user_max_concurrent: int = 2
    team_max_concurrent: int = 10
    enable_fair_share: bool = False
    # Local GPU concurrency
    local_gpu_max_concurrent: Optional[int] = None
    local_job_max_concurrent: int = 1
    local: Optional[LocalGPUStats] = None


class UserQueueResponse(BaseModel):
    """Response for user's queue info."""

    pending_count: int = 0
    running_count: int = 0
    blocked_count: int = 0
    best_position: Optional[int] = None
    pending_jobs: List[QueueEntryResponse] = Field(default_factory=list)
    running_jobs: List[QueueEntryResponse] = Field(default_factory=list)


class ConcurrencyUpdateRequest(BaseModel):
    """Request to update concurrency limits."""

    max_concurrent: Optional[int] = Field(None, ge=1, le=100)
    user_max_concurrent: Optional[int] = Field(None, ge=1, le=20)
    team_max_concurrent: Optional[int] = Field(None, ge=1, le=50)
    enable_fair_share: Optional[bool] = None
    # Local GPU concurrency limits
    local_gpu_max_concurrent: Optional[int] = Field(None, ge=1, description="Max GPUs for local jobs (null = unlimited)")
    local_job_max_concurrent: Optional[int] = Field(None, ge=1, le=10, description="Max local jobs simultaneously")


class PriorityUpdateRequest(BaseModel):
    """Request to update job priority."""

    priority: Optional[int] = Field(None, ge=0, le=50, description="Priority override (0-50, or null to clear)")


class PriorityUpdateResponse(BaseModel):
    """Response for priority update."""

    success: bool
    job_id: str
    previous_effective_priority: int
    new_effective_priority: int
    priority_override: Optional[int] = None


# --- Endpoints ---


@router.get("", response_model=QueueListResponse)
async def list_queue(
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    include_completed: bool = Query(False),
    user: Optional[User] = Depends(get_optional_user),
) -> QueueListResponse:
    """List queue entries.

    Users see their own entries unless they have queue.view.all permission.
    """
    queue_store = QueueStore()

    # Parse status filter
    queue_status = None
    if status_filter:
        try:
            queue_status = QueueStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}",
            )

    # Filter by user unless they can view all
    filter_user_id = None
    if user and not user.has_permission("queue.view.all"):
        filter_user_id = user.id

    entries = await queue_store.list_entries(
        limit=limit,
        offset=offset,
        status=queue_status,
        user_id=filter_user_id,
        include_completed=include_completed,
    )

    # Get counts
    stats = await queue_store.get_queue_stats()

    return QueueListResponse(
        entries=[_entry_to_response(e) for e in entries],
        total_pending=stats.get("by_status", {}).get("pending", 0),
        total_running=stats.get("by_status", {}).get("running", 0),
    )


@router.get("/stats", response_model=QueueStatsResponse)
async def get_queue_stats(
    user: User = Depends(require_permission("queue.view")),
) -> QueueStatsResponse:
    """Get queue statistics."""
    scheduler = get_scheduler()
    overview = await scheduler.get_queue_overview()

    # Running count comes from the queue table (local jobs are now tracked there too)
    running_count = overview.get("running", 0)

    # Get local GPU allocation info
    local_stats = None
    local_gpu_max = None
    local_job_max = 1
    try:
        from ..services.local_gpu_allocator import get_gpu_allocator
        from ..services.webui_state import WebUIStateStore

        allocator = get_gpu_allocator()
        gpu_status = await allocator.get_gpu_status()

        defaults = WebUIStateStore().load_defaults()
        local_gpu_max = defaults.local_gpu_max_concurrent
        local_job_max = defaults.local_job_max_concurrent

        local_stats = LocalGPUStats(
            running_jobs=gpu_status["running_local_jobs"],
            pending_jobs=len(await allocator._queue_store.get_pending_local_jobs()),
            allocated_gpus=gpu_status["allocated_gpus"],
            available_gpus=gpu_status["available_gpus"],
            total_gpus=gpu_status["total_gpus"],
            max_concurrent_gpus=local_gpu_max,
            max_concurrent_jobs=local_job_max,
        )
    except Exception:
        pass

    return QueueStatsResponse(
        by_status=overview.get("by_status", {}),
        by_user={str(k): v for k, v in overview.get("by_user", {}).items()},
        queue_depth=overview.get("queue_depth", 0),
        running=running_count,
        avg_wait_seconds=overview.get("avg_wait_seconds"),
        max_concurrent=overview.get("max_concurrent", 5),
        user_max_concurrent=overview.get("user_max_concurrent", 2),
        team_max_concurrent=overview.get("team_max_concurrent", 10),
        enable_fair_share=overview.get("enable_fair_share", False),
        local_gpu_max_concurrent=local_gpu_max,
        local_job_max_concurrent=local_job_max,
        local=local_stats,
    )


@router.get("/me", response_model=UserQueueResponse)
async def get_my_queue(
    user: User = Depends(get_current_user),
) -> UserQueueResponse:
    """Get the current user's queue status."""
    scheduler = get_scheduler()
    info = await scheduler.get_user_queue_info(user.id)

    return UserQueueResponse(
        pending_count=info.get("pending_count", 0),
        running_count=info.get("running_count", 0),
        blocked_count=info.get("blocked_count", 0),
        best_position=info.get("best_position"),
        pending_jobs=[QueueEntryResponse(**j) for j in info.get("pending_jobs", [])],
        running_jobs=[QueueEntryResponse(**j) for j in info.get("running_jobs", [])],
    )


@router.get("/user/{user_id}", response_model=UserQueueResponse)
async def get_user_queue(
    user_id: int,
    user: User = Depends(require_permission("queue.view.all")),
) -> UserQueueResponse:
    """Get a specific user's queue status (admin only)."""
    scheduler = get_scheduler()
    info = await scheduler.get_user_queue_info(user_id)

    return UserQueueResponse(
        pending_count=info.get("pending_count", 0),
        running_count=info.get("running_count", 0),
        blocked_count=info.get("blocked_count", 0),
        best_position=info.get("best_position"),
        pending_jobs=[QueueEntryResponse(**j) for j in info.get("pending_jobs", [])],
        running_jobs=[QueueEntryResponse(**j) for j in info.get("running_jobs", [])],
    )


@router.get("/position/{job_id}")
async def get_queue_position(
    job_id: str,
    user: Optional[User] = Depends(get_optional_user),
) -> Dict[str, Any]:
    """Get a job's position in the queue with estimated wait time."""
    scheduler = get_scheduler()
    details = await scheduler.get_queue_position_details(job_id)

    if not details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found in queue: {job_id}",
        )

    # Check access
    queue_store = QueueStore()
    entry = await queue_store.get_entry_by_job_id(job_id)
    if user and entry and entry.user_id != user.id and not user.has_permission("queue.view.all"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied",
        )

    return details


@router.post("/{job_id}/cancel")
async def cancel_queued_job(
    job_id: str,
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Cancel a job in the queue."""
    queue_store = QueueStore()
    entry = await queue_store.get_entry_by_job_id(job_id)

    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found in queue: {job_id}",
        )

    # Check permission
    is_own = entry.user_id == user.id
    can_cancel_all = user.has_permission("queue.cancel.all")
    can_cancel_own = user.has_permission("queue.cancel.own")

    if is_own and not can_cancel_own and not can_cancel_all:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: queue.cancel.own",
        )
    if not is_own and not can_cancel_all:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: queue.cancel.all",
        )

    scheduler = get_scheduler()
    success = await scheduler.cancel_job(job_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot cancel job (may be already running)",
        )

    return {"success": True, "job_id": job_id}


@router.post("/{job_id}/approve")
async def approve_queued_job(
    job_id: str,
    user: User = Depends(require_permission("queue.approve")),
) -> Dict[str, Any]:
    """Approve a blocked job for execution (admin only)."""
    scheduler = get_scheduler()

    # Create a simple approval record (approval_id = user.id for now)
    success = await scheduler.approve_job(job_id, user.id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job not found or not blocked",
        )

    return {"success": True, "job_id": job_id, "approved_by": user.id}


@router.post("/{job_id}/reject")
async def reject_queued_job(
    job_id: str,
    reason: str = Query(..., min_length=1),
    user: User = Depends(require_permission("queue.approve")),
) -> Dict[str, Any]:
    """Reject a blocked job (admin only)."""
    scheduler = get_scheduler()

    success = await scheduler.reject_job(job_id, reason)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job not found or not blocked",
        )

    return {"success": True, "job_id": job_id, "rejected_by": user.id, "reason": reason}


@router.post("/{job_id}/priority", response_model=PriorityUpdateResponse)
async def set_job_priority(
    job_id: str,
    request: PriorityUpdateRequest,
    user: User = Depends(require_permission("queue.manage")),
) -> PriorityUpdateResponse:
    """Set priority override for a queued job.

    Allows leads/admins to adjust the scheduling priority of pending jobs.
    Higher priority jobs are scheduled before lower priority ones.

    Priority values:
    - 0-9: Low (background tasks)
    - 10-19: Normal (standard researcher)
    - 20-29: High (lead researcher)
    - 30-39: Urgent (admin)
    - 40-50: Critical (system/emergency)

    Set priority to null to clear the override and revert to level-based priority.
    """
    queue_store = QueueStore()
    entry = await queue_store.get_entry_by_job_id(job_id)

    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found in queue: {job_id}",
        )

    previous_priority = entry.effective_priority

    scheduler = get_scheduler()
    updated = await scheduler.set_priority(job_id, request.priority)

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update priority: job status is {entry.status.value} (must be pending/blocked)",
        )

    return PriorityUpdateResponse(
        success=True,
        job_id=job_id,
        previous_effective_priority=previous_priority,
        new_effective_priority=updated.effective_priority,
        priority_override=updated.priority_override,
    )


@router.post("/concurrency")
async def update_concurrency_limits(
    request: ConcurrencyUpdateRequest,
    user: User = Depends(require_permission("queue.manage")),
) -> Dict[str, Any]:
    """Update queue concurrency limits (admin only).

    Supports global, per-user, per-team, and local GPU limits with optional fair-share scheduling.
    """
    scheduler = get_scheduler()

    await scheduler.set_concurrency_limits(
        max_concurrent=request.max_concurrent,
        user_max_concurrent=request.user_max_concurrent,
        team_max_concurrent=request.team_max_concurrent,
        enable_fair_share=request.enable_fair_share,
    )

    # Update local GPU limits if provided
    local_gpu_max = None
    local_job_max = 1
    if request.local_gpu_max_concurrent is not None or request.local_job_max_concurrent is not None:
        from ..services.webui_state import WebUIStateStore

        store = WebUIStateStore()
        defaults = store.load_defaults()

        if request.local_gpu_max_concurrent is not None:
            defaults.local_gpu_max_concurrent = request.local_gpu_max_concurrent
        if request.local_job_max_concurrent is not None:
            defaults.local_job_max_concurrent = request.local_job_max_concurrent

        store.save_defaults(defaults)
        local_gpu_max = defaults.local_gpu_max_concurrent
        local_job_max = defaults.local_job_max_concurrent
    else:
        # Read current values
        from ..services.webui_state import WebUIStateStore

        defaults = WebUIStateStore().load_defaults()
        local_gpu_max = defaults.local_gpu_max_concurrent
        local_job_max = defaults.local_job_max_concurrent

    return {
        "success": True,
        "max_concurrent": scheduler._max_concurrent,
        "user_max_concurrent": scheduler._user_max_concurrent,
        "team_max_concurrent": scheduler._policy.config.team_max_concurrent,
        "enable_fair_share": scheduler._policy.config.enable_fair_share,
        "local_gpu_max_concurrent": local_gpu_max,
        "local_job_max_concurrent": local_job_max,
    }


@router.post("/process")
async def trigger_queue_processing(
    user: User = Depends(require_permission("queue.manage")),
) -> Dict[str, Any]:
    """Manually trigger queue processing (admin only)."""
    scheduler = get_scheduler()

    count, job_ids = await scheduler.process_queue()

    return {
        "success": True,
        "dispatched": count,
        "job_ids": job_ids,
    }


@router.post("/cleanup")
async def cleanup_old_entries(
    days: int = Query(30, ge=1, le=365),
    user: User = Depends(require_permission("queue.manage")),
) -> Dict[str, Any]:
    """Clean up old completed queue entries (admin only)."""
    queue_store = QueueStore()

    deleted = await queue_store.cleanup_old_entries(days=days)

    return {
        "success": True,
        "deleted": deleted,
        "days": days,
    }


# --- Local job submission ---


class LocalJobSubmitRequest(BaseModel):
    """Request to submit a local training job."""

    config_name: str = Field(..., description="Name of config to load from disk")
    no_wait: bool = Field(False, description="Reject immediately if GPUs unavailable")
    any_gpu: bool = Field(False, description="Use any available GPUs instead of configured device IDs")
    for_approval: bool = Field(False, description="Request approval to exceed org GPU quota")


class LocalJobSubmitResponse(BaseModel):
    """Response from local job submission."""

    success: bool
    job_id: Optional[str] = None
    status: Optional[str] = None  # "running", "queued", "rejected", "blocked"
    allocated_gpus: Optional[List[int]] = None
    queue_position: Optional[int] = None
    requires_approval: bool = False
    error: Optional[str] = None
    reason: Optional[str] = None


@router.post("/submit", response_model=LocalJobSubmitResponse)
async def submit_local_job(
    request: LocalJobSubmitRequest,
    user: Optional[User] = Depends(get_optional_user),
) -> LocalJobSubmitResponse:
    """Submit a local training job.

    Loads the config from disk and starts the training process locally.
    Respects GPU allocation and queuing when resources are unavailable.
    """
    from pathlib import Path

    from ..services.config_store import ConfigStore
    from ..services.training_service import start_training_job
    from ..services.webui_state import WebUIStateStore

    try:
        # Load config from disk
        defaults = WebUIStateStore().load_defaults()
        if not defaults.configs_dir:
            return LocalJobSubmitResponse(
                success=False,
                error="No configs directory configured",
            )

        config_store = ConfigStore(
            config_dir=Path(defaults.configs_dir).expanduser(),
            config_type="model",
        )

        try:
            config, _ = config_store.load_config(request.config_name)
        except FileNotFoundError:
            return LocalJobSubmitResponse(
                success=False,
                error=f"Config '{request.config_name}' not found",
            )

        # Get user's org_id for quota checks
        user_org_id = user.org_id if user else None

        # Start the training job (path resolution happens inside start_training_job)
        result = start_training_job(
            config,
            env_name=request.config_name,
            no_wait=request.no_wait,
            any_gpu=request.any_gpu,
            for_approval=request.for_approval,
            org_id=user_org_id,
            user_id=user.id if user else None,
        )

        # Handle rejected jobs
        if result.status == "rejected":
            return LocalJobSubmitResponse(
                success=False,
                status="rejected",
                error=result.reason or "Required GPUs unavailable",
                reason=result.reason,
            )

        # Handle jobs requiring approval
        if result.status == "blocked":
            return LocalJobSubmitResponse(
                success=True,
                job_id=result.job_id,
                status="blocked",
                queue_position=result.queue_position,
                requires_approval=True,
                reason=result.reason,
            )

        # Broadcast SSE event so UI updates in real-time
        if result.status == "running":
            try:
                from ..services.sse_manager import get_sse_manager

                sse_manager = get_sse_manager()
                await sse_manager.broadcast(
                    data={
                        "type": "training.status",
                        "status": "starting",
                        "job_id": result.job_id,
                        "config_name": request.config_name,
                        "allocated_gpus": result.allocated_gpus,
                        "message": f"Training job {result.job_id} starting",
                    },
                    event_type="training.status",
                )
            except Exception as exc:
                logger.warning("Failed to broadcast SSE event: %s", exc)

        return LocalJobSubmitResponse(
            success=True,
            job_id=result.job_id,
            status=result.status,
            allocated_gpus=result.allocated_gpus,
            queue_position=result.queue_position,
            reason=result.reason,
        )

    except Exception as exc:
        logger.error("Failed to submit local job: %s", exc, exc_info=True)
        return LocalJobSubmitResponse(
            success=False,
            error=str(exc),
        )


# --- Polling preference endpoints ---


@router.put("/polling/setting")
async def set_polling_preference(
    enabled: bool = Query(..., description="Whether to enable automatic queue polling"),
    user: User = Depends(require_permission("queue.manage")),
) -> Dict[str, Any]:
    """Set the polling preference for the queue scheduler."""
    from ..services.webui_state import WebUIStateStore

    store = WebUIStateStore()
    defaults = store.load_defaults()
    defaults.queue_polling_enabled = enabled
    store.save_defaults(defaults)

    return {"success": True, "polling_enabled": enabled}


@router.get("/polling/status")
async def get_polling_status(
    user: Optional[User] = Depends(get_optional_user),
) -> Dict[str, Any]:
    """Get the current polling status."""
    from ..services.webui_state import WebUIStateStore

    store = WebUIStateStore()
    defaults = store.load_defaults()

    return {
        "polling_enabled": getattr(defaults, "queue_polling_enabled", True),
    }


def _entry_to_response(entry: QueueEntry) -> QueueEntryResponse:
    """Convert a QueueEntry to response model."""
    return QueueEntryResponse(
        id=entry.id,
        job_id=entry.job_id,
        user_id=entry.user_id,
        provider=entry.provider,
        config_name=entry.config_name,
        priority=entry.priority.value,
        priority_name=entry.priority.name.lower(),
        priority_override=entry.priority_override,
        effective_priority=entry.effective_priority,
        status=entry.status.value,
        position=entry.position,
        queued_at=entry.queued_at,
        started_at=entry.started_at,
        completed_at=entry.completed_at,
        estimated_cost=entry.estimated_cost,
        requires_approval=entry.requires_approval,
        attempt=entry.attempt,
        error_message=entry.error_message,
    )
