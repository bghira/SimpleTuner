"""Worker API routes for GPU worker management.

Two sets of endpoints:
1. Worker-facing (token auth via X-Worker-Token header)
2. Admin-facing (session auth, existing patterns)
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, FiniteFloat

from simpletuner.worker_protocol import WORKER_LOG_TAIL_SIZE

from ..models.worker import WorkerStatus, WorkerType
from ..services.callback_events import StageStatus
from ..services.callback_service import get_default_callback_service
from ..services.cloud.auth import User, get_optional_user
from ..services.worker_credentials import generate_worker_token, hash_token
from ..services.worker_run_metrics import ACTIVE_STATUSES, WORKER_RUN_METRICS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workers", tags=["workers"])
admin_router = APIRouter(prefix="/api/admin/workers", tags=["admin-workers"])


# Global dict of worker_id -> asyncio.Queue for SSE management
worker_streams: Dict[str, asyncio.Queue] = {}
manual_control_workers: set[str] = set()


# Request/Response Models


class WorkerRegistrationRequest(BaseModel):
    """Worker registration request."""

    name: str
    gpu_info: Dict[str, Any] = Field(default_factory=dict)
    persistent: bool = False
    provider: Optional[str] = None
    labels: Dict[str, str] = Field(default_factory=dict)
    current_job_id: Optional[str] = None


class WorkerRegistrationResponse(BaseModel):
    """Worker registration response."""

    worker_id: str
    sse_url: str
    resume_job: Optional[Dict[str, Any]] = None
    abandon_job: Optional[str] = None


class HeartbeatRequest(BaseModel):
    """Worker heartbeat request."""

    worker_id: str
    status: str
    current_job_id: Optional[str] = None
    job_progress: Optional[Dict[str, Any]] = None
    log_tail: str | None = Field(default=None, max_length=WORKER_LOG_TAIL_SIZE)
    gpu_utilization: Optional[float] = None
    vram_used_gb: Optional[float] = None
    system_metrics: dict[Annotated[str, Field(max_length=80)], Annotated[FiniteFloat, Field(ge=-1e12, le=1e12)]] | None = (
        Field(default=None, max_length=512)
    )


class WorkerValidationUpdate(BaseModel):
    status: StageStatus
    progress: Optional[Dict[str, Any]] = None
    timestamp: datetime


class JobStatusUpdate(BaseModel):
    """Job status update from worker."""

    status: str | None = None
    progress: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    log_tail: str | None = Field(default=None, max_length=WORKER_LOG_TAIL_SIZE)
    validation: WorkerValidationUpdate | None = None


class CreateWorkerRequest(BaseModel):
    """Create worker request."""

    name: str
    worker_type: Literal["persistent", "ephemeral", "peer"] = "persistent"
    job_id: str | None = None
    labels: Dict[str, str] = Field(default_factory=dict)


class CreateWorkerResponse(BaseModel):
    """Create worker response."""

    worker_id: str
    token: str
    connection_command: str


class WorkerListItem(BaseModel):
    """Worker list item."""

    worker_id: str
    name: str
    worker_type: str
    status: str
    gpu_info: Dict[str, Any]
    provider: Optional[str]
    labels: Dict[str, str]
    current_job_id: Optional[str]
    last_heartbeat: Optional[str]
    connected_at: Optional[str]
    created_at: str
    connected: bool
    uptime_seconds: Optional[float] = None


class WorkerListResponse(BaseModel):
    """Worker list response."""

    workers: list[WorkerListItem]
    total: int


class TokenRotationResponse(BaseModel):
    """Token rotation response."""

    worker_id: str
    token: str
    connection_command: str


# Helper Functions


async def validate_worker_token(token: str) -> Any:
    """Validate X-Worker-Token header and return worker.

    Args:
        token: The worker token from the header

    Returns:
        Worker object if valid

    Raises:
        HTTPException: 401 if token is invalid
    """
    from ..services.worker_repository import get_worker_repository

    worker_repo = get_worker_repository()
    token_hash = hash_token(token)
    worker = await worker_repo.get_worker_by_token_hash(token_hash)

    if not worker:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid worker token",
        )

    return worker


async def push_to_worker(worker_id: str, event: Dict[str, Any]) -> bool:
    """Push event to worker's SSE stream.

    Args:
        worker_id: The worker ID
        event: The event data to send

    Returns:
        True if event was pushed, False if worker not connected
    """
    if worker_id in worker_streams:
        try:
            await worker_streams[worker_id].put(event)
            return True
        except Exception as exc:
            logger.warning("Failed to push event to worker %s: %s", worker_id, exc)
            return False
    return False


def push_group_commands(commands: dict[str, dict]) -> bool:
    """Check and enqueue the complete group without yielding between workers."""
    if any(worker_id not in worker_streams or worker_streams[worker_id].full() for worker_id in commands):
        return False
    for worker_id, command in commands.items():
        worker_streams[worker_id].put_nowait(command)
    return True


def is_worker_connected(worker_id: str) -> bool:
    """Check if worker is connected to SSE stream.

    Args:
        worker_id: The worker ID

    Returns:
        True if worker has an active SSE connection
    """
    return worker_id in worker_streams


def get_client_ip(request: Request) -> str:
    """Extract client IP from request.

    Args:
        request: The FastAPI request

    Returns:
        The client IP address
    """
    if request.client:
        return request.client.host
    return "unknown"


# Worker-facing endpoints


async def _reported_worker_job(job_repo, job_id, worker):
    job = await job_repo.get(job_id)
    if (
        job
        and job.status == "pending"
        and job.provider == "worker"
        and not worker.is_job_bound
        and worker.current_job_id in (None, job_id)
        and (job.metadata or {}).get("worker_id") == worker.worker_id
        and job.metadata.get("retry_count", 0) > 0
    ):
        # The reporting worker kept training through a disconnect. A redispatch
        # or cancellation may have won since we read the pending record.
        await job_repo.update(
            job_id,
            {
                "status": "running",
                "completed_at": None,
                "error_message": None,
                "metadata": {"needs_reconciliation": None, "retry_count": job.metadata["retry_count"] - 1},
            },
            expected_status="pending",
            worker_id=worker.worker_id,
        )
        job = await job_repo.get(job_id)
    return job


async def _check_peer_binding(worker, reported_job_id: str | None) -> None:
    """Keep a peer attached only to the job selected by its administrator."""
    from ..services.cloud.storage.job_repository import get_job_repository
    from ..services.worker_repository import get_worker_repository

    if reported_job_id is not None and reported_job_id != worker.current_job_id:
        raise HTTPException(status_code=409, detail="Peer is bound to another job")
    job = await get_job_repository().get(worker.current_job_id) if worker.current_job_id else None
    if job is None or job.is_terminal:
        await get_worker_repository().update_worker(
            worker.worker_id, {"status": WorkerStatus.OFFLINE, "current_job_id": None}, expected_job_id=worker.current_job_id
        )
        await push_to_worker(worker.worker_id, {"type": "shutdown", "reason": "Peer job ended"})
        raise HTTPException(status_code=410, detail="Peer job ended")


@router.post("/register", response_model=WorkerRegistrationResponse)
async def register_worker(
    request: WorkerRegistrationRequest,
    http_request: Request,
    x_worker_token: str = Header(..., alias="X-Worker-Token"),
) -> WorkerRegistrationResponse:
    """Worker self-registers on startup or reconnection.

    Args:
        request: The registration request
        http_request: The HTTP request for IP extraction
        x_worker_token: Worker authentication token

    Returns:
        Registration response with SSE URL and job instructions

    Raises:
        HTTPException: 401 if token is invalid
    """
    from ..services.worker_repository import get_worker_repository

    worker = await validate_worker_token(x_worker_token)
    worker_repo = get_worker_repository()
    client_ip = get_client_ip(http_request)

    logger.info(
        "Worker %s registering from IP %s (name=%s, persistent=%s)",
        worker.worker_id,
        client_ip,
        request.name,
        request.persistent,
    )

    now = datetime.now(timezone.utc)

    if worker.worker_type == WorkerType.PEER:
        await _check_peer_binding(worker, request.current_job_id)
        await worker_repo.update_worker(
            worker.worker_id,
            {"gpu_info": request.gpu_info, "status": WorkerStatus.BUSY, "connected_at": now, "last_heartbeat": now},
            expected_job_id=worker.current_job_id,
        )
        return WorkerRegistrationResponse(
            worker_id=worker.worker_id, sse_url=f"/api/workers/stream?worker_id={worker.worker_id}"
        )

    if worker.is_job_bound and request.current_job_id is not None and request.current_job_id != worker.current_job_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(f"Worker {worker.worker_id} is bound to job " f"{worker.current_job_id}, not {request.current_job_id}"),
        )

    worker_type = (
        WorkerType.EPHEMERAL
        if worker.is_job_bound
        else WorkerType.PERSISTENT if request.persistent else WorkerType.EPHEMERAL
    )

    # Update worker with registration info
    updates = {
        "name": request.name,
        "gpu_info": request.gpu_info,
        "status": WorkerStatus.CONNECTING if worker.is_job_bound else WorkerStatus.IDLE,
        "worker_type": worker_type,
        "labels": request.labels,
        "last_heartbeat": now,
        "connected_at": now,
    }

    updates["current_job_id"] = worker.current_job_id if worker.is_job_bound else None
    if not worker.is_job_bound and request.provider:
        updates["provider"] = request.provider

    # Handle reconnection reconciliation
    resume_job = None
    abandon_job = None

    if not request.current_job_id and worker.current_job_id and not worker.is_job_bound:
        from ..services.cloud.storage.job_repository import get_job_repository
        from ..services.worker_manager import get_worker_manager

        assigned = await get_job_repository().get(worker.current_job_id)
        if assigned and worker.worker_id in (assigned.metadata or {}).get("worker_ids", []):
            manager = get_worker_manager()
            if manager:
                await manager._handle_orphaned_job(worker.current_job_id, worker)

    if request.current_job_id:
        # Worker thinks it's running a job - verify if it should continue
        from ..services.cloud.storage.job_repository import get_job_repository

        job_repo = get_job_repository()
        job = await _reported_worker_job(job_repo, request.current_job_id, worker)

        if worker.is_job_bound and job and job.status in ["pending", "queued"]:
            # The SSE connection triggers the first and only dispatch.
            pass
        elif (
            job
            and job.status in {"running", "starting", "training"}
            and (
                worker.worker_id == (job.metadata or {}).get("worker_id")
                or worker.worker_id in (job.metadata or {}).get("worker_ids", [])
            )
        ):
            updates["status"] = WorkerStatus.BUSY
            updates["current_job_id"] = job.job_id
            # Job is still active - worker should resume
            resume_job = {
                "job_id": job.job_id,
                "config_name": job.config_name,
                "status": job.status,
            }
            logger.info("Worker %s reconnected, resuming job %s", worker.worker_id, request.current_job_id)
        else:
            # Job was cancelled or completed while worker was offline
            abandon_job = request.current_job_id
            logger.info("Worker %s reconnected, abandoning stale job %s", worker.worker_id, request.current_job_id)

    await worker_repo.update_worker(worker.worker_id, updates)

    # Build SSE URL
    sse_url = f"/api/workers/stream?worker_id={worker.worker_id}"

    return WorkerRegistrationResponse(
        worker_id=worker.worker_id,
        sse_url=sse_url,
        resume_job=resume_job,
        abandon_job=abandon_job,
    )


@router.get("/stream")
async def worker_stream(
    worker_id: str,
    x_worker_token: str = Header(..., alias="X-Worker-Token"),
    manual_controls: bool = False,
) -> StreamingResponse:
    """SSE stream for commands to worker.

    Args:
        worker_id: The worker ID
        x_worker_token: Worker authentication token

    Returns:
        SSE streaming response

    Raises:
        HTTPException: 401 if token is invalid, 400 if worker_id mismatch
    """
    worker = await validate_worker_token(x_worker_token)

    if worker.worker_id != worker_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Worker ID mismatch",
        )

    logger.info("Worker %s connected to SSE stream", worker_id)

    # Create queue for this worker
    queue = asyncio.Queue()
    worker_streams[worker_id] = queue
    if manual_controls and worker.worker_type != WorkerType.PEER:
        manual_control_workers.add(worker_id)
    else:
        manual_control_workers.discard(worker_id)

    from ..services.worker_manager import get_worker_manager

    worker_manager = get_worker_manager()
    if worker_manager:
        if worker.is_job_bound:
            await worker_manager.dispatch_bound_job(worker_id)
        elif worker.status == WorkerStatus.IDLE:
            await worker_manager.dispatch_pending_jobs()

    async def event_generator():
        """Generate SSE events for the worker."""
        try:
            while True:
                # Wait for events with timeout to allow cleanup check
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)

                    # Format as SSE - send full event payload as JSON
                    event_type = event.get("type", "message")
                    data = json.dumps(event)

                    yield f"event: {event_type}\ndata: {data}\n\n"

                except asyncio.TimeoutError:
                    # Send keepalive ping
                    yield f"event: ping\ndata: {{}}\n\n"

        except asyncio.CancelledError:
            logger.info("SSE stream cancelled for worker %s", worker_id)
        except Exception as exc:
            logger.error("SSE stream error for worker %s: %s", worker_id, exc)
        finally:
            # Clean up queue
            if worker_streams.get(worker_id) is queue:
                del worker_streams[worker_id]
                manual_control_workers.discard(worker_id)
            logger.info("Worker %s disconnected from SSE stream", worker_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _assigned_job(job_repo, job_id: str, worker, *, reports_progress: bool):
    worker_id = worker.worker_id
    job = await _reported_worker_job(job_repo, job_id, worker) if reports_progress else await job_repo.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    # Verify worker is assigned to this job
    metadata = job.metadata or {}
    if metadata.get("worker_id") != worker_id and worker_id not in metadata.get("worker_ids", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Worker not assigned to this job",
        )

    return job


def _record_worker_stats(job, worker, status: str, progress: dict | None, validation: WorkerValidationUpdate | None = None):
    if progress and status in ACTIVE_STATUSES:
        WORKER_RUN_METRICS.record(job, worker.name, progress)
    event = {
        "type": "training.status",
        "job_id": job.job_id,
        "data": {"status": "running" if status == "training" else status, "worker": worker.name},
        "progress": progress,
    }
    if validation is not None:
        event.update(
            type="lifecycle.stage",
            timestamp=validation.timestamp.isoformat(),
            stage={"key": "validation", "status": validation.status, "progress": validation.progress},
        )
    get_default_callback_service().handle_incoming(
        event,
        assigned_worker={
            "assigned_worker": True,
            "worker": (job.metadata or {}).get("worker_names", [worker.name])[0],
            "worker_names": (job.metadata or {}).get("worker_names"),
            "gpu": (worker.gpu_info or {}).get("name") or job.hardware_type,
            "config_name": job.config_name,
            "user_id": job.user_id,
        },
    )


@router.post("/heartbeat")
async def worker_heartbeat(
    request: HeartbeatRequest,
    x_worker_token: str = Header(..., alias="X-Worker-Token"),
) -> Dict[str, Any]:
    """Periodic keepalive from worker.

    Args:
        request: The heartbeat request
        x_worker_token: Worker authentication token

    Returns:
        Success response

    Raises:
        HTTPException: 401 if token is invalid, 400 if worker_id mismatch
    """
    from ..services.cloud.storage.job_repository import get_job_repository
    from ..services.worker_repository import get_worker_repository

    worker = await validate_worker_token(x_worker_token)

    if worker.worker_id != request.worker_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Worker ID mismatch",
        )

    worker_repo = get_worker_repository()

    now = datetime.now(timezone.utc)
    updates = {"last_heartbeat": now}
    if request.system_metrics:
        updates["system_sample"] = {"sampled_at": now.isoformat(), "metrics": request.system_metrics}

    if worker.worker_type == WorkerType.PEER:
        await _check_peer_binding(worker, request.current_job_id)
        if request.job_progress is not None or request.log_tail is not None:
            raise HTTPException(status_code=403, detail="Only the assigned worker can report job progress or logs")
        updates["status"] = WorkerStatus.BUSY
        await worker_repo.update_worker(worker.worker_id, updates, expected_job_id=worker.current_job_id)
        return {"success": True}

    # Update worker with heartbeat info
    if worker.is_job_bound and request.current_job_id is not None and request.current_job_id != worker.current_job_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(f"Worker {worker.worker_id} is bound to job " f"{worker.current_job_id}, not {request.current_job_id}"),
        )

    if request.job_progress or request.log_tail is not None:
        if not request.current_job_id:
            raise HTTPException(status_code=422, detail="Job progress and logs require a job ID")
        job = await _assigned_job(
            get_job_repository(), request.current_job_id, worker, reports_progress=request.status == WorkerStatus.BUSY
        )
        if request.job_progress and not job.is_terminal and (job.metadata or {}).get("worker_id") == worker.worker_id:
            _record_worker_stats(job, worker, "running", request.job_progress)

    if request.status == WorkerStatus.BUSY and request.current_job_id and not worker.is_job_bound:
        await get_job_repository().record_worker_heartbeat(request.current_job_id, worker.worker_id, request.log_tail)

    updates["status"] = request.status
    if worker.is_job_bound:
        updates["status"] = WorkerStatus.BUSY if request.status == WorkerStatus.BUSY.value else WorkerStatus.CONNECTING
        updates["current_job_id"] = worker.current_job_id
    if worker.connected_at is None:
        updates["connected_at"] = now

    if request.current_job_id and not worker.is_job_bound:
        updates["current_job_id"] = request.current_job_id

    ready = request.status == WorkerStatus.IDLE and not worker.is_job_bound and not request.current_job_id
    if ready and worker.current_job_id:
        # An idle heartbeat can cross a newly dispatched command on the wire.
        assigned = await get_job_repository().get(worker.current_job_id)
        ready = assigned is None or assigned.is_terminal
        if ready:
            updates["current_job_id"] = None
        else:
            updates.pop("status")
    await worker_repo.update_worker(worker.worker_id, updates, expected_job_id=worker.current_job_id)

    if ready:
        from ..services.worker_manager import get_worker_manager

        manager = get_worker_manager()
        if manager:
            await manager.dispatch_pending_jobs()

    logger.debug(
        "Heartbeat from worker %s: status=%s, job=%s",
        worker.worker_id,
        request.status,
        request.current_job_id,
    )

    return {"success": True}


@router.get("/queue")
async def worker_has_pending_jobs(worker_id: str, x_worker_token: str = Header(..., alias="X-Worker-Token")):
    """Let an ephemeral worker stay online for jobs waiting for it."""
    from ..services.cloud.container import get_job_store

    worker = await validate_worker_token(x_worker_token)
    if worker.worker_id != worker_id:
        raise HTTPException(status_code=403, detail="Worker ID mismatch")
    if worker.worker_type == WorkerType.PEER:
        return {"has_jobs": False}
    jobs = await get_job_store().list_jobs(status="pending", provider="worker", limit=-1)
    return {
        "has_jobs": any(
            (job.metadata or {}).get("target_worker_id") in (None, worker_id)
            or worker_id in (job.metadata or {}).get("target_worker_ids", [])
            for job in jobs
        )
    }


@router.post("/job/{job_id}/status")
async def update_job_status(
    job_id: str,
    request: JobStatusUpdate,
    x_worker_token: str = Header(..., alias="X-Worker-Token"),
) -> Dict[str, Any]:
    """Worker reports job progress.

    Args:
        job_id: The job ID
        request: The status update
        x_worker_token: Worker authentication token

    Returns:
        Success response

    Raises:
        HTTPException: 401 if token is invalid, 404 if job not found
    """
    from ..services.cloud.storage.job_repository import get_job_repository

    worker = await validate_worker_token(x_worker_token)
    job_repo = get_job_repository()

    job = await _assigned_job(
        job_repo, job_id, worker, reports_progress=bool(request.progress) and request.status in (None, "training", "running")
    )
    group = (job.metadata or {}).get("worker_ids", [])
    group_finished = False
    if group:
        if request.status in {"completed", "failed", "cancelled"}:
            group_finished = await job_repo.finish_group_member(
                job_id, worker.worker_id, request.status, request.error, request.log_tail
            )
            if not group_finished:
                return {"success": True}
            if request.status != "completed":
                for member in group:
                    if member != worker.worker_id:
                        await push_to_worker(member, {"type": "job_cancel", "job_id": job_id})
        elif job.is_terminal or worker.worker_id != group[0]:
            return {"success": True}
    if (job.metadata or {}).get("group_check"):
        return {"success": True}
    if not request.status:
        if not request.progress and request.validation is None:
            raise HTTPException(status_code=422, detail="Expected worker status, progress, or validation")
        if not job.is_terminal:
            _record_worker_stats(job, worker, "running", request.progress, request.validation)
        return {"success": True}

    logger.info(
        "Job %s status update from worker %s: %s",
        job_id,
        worker.worker_id,
        request.status,
    )

    completion_metadata = None
    if request.status == "completed" and worker.is_job_bound:
        from ..services.kubeflow_job_service import get_kubeflow_job_service

        service = get_kubeflow_job_service()
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Kubeflow service is unavailable",
            )
        if not service.artifacts_received(job):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Central LoRA artifact upload has not completed",
            )

        completion_metadata = dict(job.metadata or {})
        upload_state = dict(completion_metadata.get("artifact_upload") or {})
        upload_state["status"] = "complete"
        completion_metadata["artifact_upload"] = upload_state

    # Update job status
    updates: Dict[str, Any] = {"status": request.status}
    if completion_metadata is not None:
        updates["metadata"] = completion_metadata

    if request.error:
        updates["error_message"] = request.error

    if request.status in ["completed", "failed", "cancelled"]:
        updates["completed_at"] = datetime.now(timezone.utc).isoformat()

    updated = group_finished or await job_repo.update(
        job_id,
        updates,
        worker_id=worker.worker_id,
        worker_log_tail=request.log_tail if job.provider == "worker" else None,
        **({"expected_status": job.status} if group else {}),
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job assignment changed")
    _record_worker_stats(job, worker, request.status, request.progress)

    if worker.is_job_bound and request.status in ["completed", "failed", "cancelled"]:
        from ..services.kubeflow_job_service import get_kubeflow_job_service
        from ..services.worker_repository import get_worker_repository

        await get_worker_repository().update_worker(
            worker.worker_id, {"status": WorkerStatus.DRAINING, "current_job_id": worker.current_job_id}
        )
        service = get_kubeflow_job_service()
        if service is None:
            logger.error("Kubeflow service unavailable while finalizing job %s", job_id)
        else:
            await service.finalize(job_id)

    # Emit SSE event for UI updates
    try:
        from ..services.sse_manager import get_sse_manager

        sse_manager = get_sse_manager()
        await sse_manager.broadcast(
            data={
                "type": "job.status",
                "job_id": job_id,
                "status": request.status,
                "progress": request.progress,
                "error": request.error,
            },
            event_type="job.status",
        )
    except Exception as exc:
        logger.warning("Failed to broadcast SSE event: %s", exc)

    return {"success": True}


# Admin-facing endpoints


@admin_router.get("", response_model=WorkerListResponse)
async def list_workers(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
    status_filter: Optional[str] = None,
) -> WorkerListResponse:
    """List all workers (admin only).

    Args:
        request: The HTTP request
        user: The authenticated user
        status_filter: Optional status filter

    Returns:
        List of workers

    Raises:
        HTTPException: 403 if user lacks permission
    """
    from ..services.worker_repository import get_worker_repository

    if not user or not user.has_permission("admin.workers"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: admin.workers",
        )

    worker_repo = get_worker_repository()

    status_enum = None
    if status_filter:
        try:
            status_enum = WorkerStatus(status_filter.upper())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}",
            )

    workers = await worker_repo.list_workers(status=status_enum)

    # Convert to response format
    worker_items = []
    now = datetime.now(timezone.utc)
    for worker in workers:
        connected = is_worker_connected(worker.worker_id)
        uptime_seconds = None
        if connected and worker.connected_at:
            uptime_seconds = round((now - worker.connected_at).total_seconds(), 1)
        worker_items.append(
            WorkerListItem(
                worker_id=worker.worker_id,
                name=worker.name,
                worker_type=worker.worker_type.value,
                status=worker.status.value,
                gpu_info=worker.gpu_info,
                provider=worker.provider,
                labels=worker.labels,
                current_job_id=worker.current_job_id,
                last_heartbeat=worker.last_heartbeat.isoformat() if worker.last_heartbeat else None,
                connected_at=worker.connected_at.isoformat() if worker.connected_at else None,
                created_at=worker.created_at.isoformat(),
                connected=connected,
                uptime_seconds=uptime_seconds,
            )
        )

    return WorkerListResponse(
        workers=worker_items,
        total=len(worker_items),
    )


@admin_router.post("", response_model=CreateWorkerResponse)
async def create_worker(
    http_request: Request,
    body: CreateWorkerRequest,
    user: Optional[User] = Depends(get_optional_user),
) -> CreateWorkerResponse:
    """Provision a worker token, optionally bound to a distributed job as a peer.

    Args:
        http_request: The HTTP request
        body: The create request
        user: The authenticated user

    Returns:
        Worker creation response with token

    Raises:
        HTTPException: 403 if user lacks permission
    """
    from ..services.worker_repository import get_worker_repository

    if not user or not user.has_permission("admin.workers"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: admin.workers",
        )

    client_ip = get_client_ip(http_request)
    worker_repo = get_worker_repository()

    if (body.worker_type == "peer") != (body.job_id is not None):
        raise HTTPException(status_code=422, detail="Supply job_id when creating a peer worker")
    if body.worker_type == "peer":
        from ..services.cloud.storage.job_repository import get_job_repository

        job = await get_job_repository().get(body.job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Peer job does not exist")
        if job.provider != "worker" or job.is_terminal or not (job.metadata or {}).get("worker_id"):
            raise HTTPException(status_code=409, detail="Peer requires a live job assigned to a worker")

    # Generate worker ID and token
    worker_id = f"worker-{secrets.token_urlsafe(8)}"
    token = generate_worker_token()
    token_hash = hash_token(token)

    # Create worker
    from ..models.worker import Worker, WorkerStatus

    worker = Worker(
        worker_id=worker_id,
        name=body.name,
        worker_type=WorkerType(body.worker_type),
        status=WorkerStatus.CONNECTING,
        token_hash=token_hash,
        user_id=user.id if user else 0,
        labels=body.labels,
        current_job_id=body.job_id,
        created_at=datetime.now(timezone.utc),
    )

    await worker_repo.create_worker(worker)

    logger.info(
        "Worker %s created by user %s from IP %s",
        worker_id,
        user.username if user else "system",
        client_ip,
    )

    # Emit audit event
    from ..services.cloud.audit import AuditEventType, audit_log

    await audit_log(
        AuditEventType.CONFIG_CHANGED,
        f"Worker '{worker_id}' created",
        actor_id=user.id if user else None,
        actor_username=user.username if user else None,
        actor_ip=client_ip,
        target_type="worker",
        target_id=worker_id,
        details={"name": body.name, "worker_type": body.worker_type},
    )

    # Build connection command with actual server URL
    server_url = str(http_request.base_url).rstrip("/")
    connection_command = f"simpletuner worker --orchestrator-url {server_url} --worker-token {token}"

    return CreateWorkerResponse(
        worker_id=worker_id,
        token=token,
        connection_command=connection_command,
    )


@admin_router.delete("/{worker_id}")
async def delete_worker(
    worker_id: str,
    http_request: Request,
    user: Optional[User] = Depends(get_optional_user),
    force: bool = False,
) -> Dict[str, Any]:
    """Remove a worker.

    Args:
        worker_id: The worker ID
        http_request: The HTTP request
        user: The authenticated user
        force: Force deletion even if worker has active job

    Returns:
        Success response

    Raises:
        HTTPException: 403 if user lacks permission, 400 if worker has active job
    """
    from ..services.worker_repository import get_worker_repository

    if not user or not user.has_permission("admin.workers"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: admin.workers",
        )

    client_ip = get_client_ip(http_request)
    worker_repo = get_worker_repository()

    # Get worker
    worker = await worker_repo.get_worker(worker_id)
    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worker not found: {worker_id}",
        )

    # Check if worker has active job
    if worker.current_job_id and worker.worker_type != WorkerType.PEER and not force:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Worker has active job {worker.current_job_id}. Use force=true to delete anyway.",
        )

    # Delete worker
    success = await worker_repo.delete_worker(worker_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete worker",
        )

    logger.info(
        "Worker %s deleted by user %s from IP %s",
        worker_id,
        user.username if user else "system",
        client_ip,
    )

    # Emit audit event
    from ..services.cloud.audit import AuditEventType, audit_log

    await audit_log(
        AuditEventType.CONFIG_CHANGED,
        f"Worker '{worker_id}' deleted",
        actor_id=user.id if user else None,
        actor_username=user.username if user else None,
        actor_ip=client_ip,
        target_type="worker",
        target_id=worker_id,
        details={"name": worker.name, "force": force},
    )

    # Disconnect SSE stream if connected
    if worker_id in worker_streams:
        await push_to_worker(worker_id, {"type": "shutdown", "data": {"reason": "worker_deleted"}})

    return {"success": True, "worker_id": worker_id}


@admin_router.post("/{worker_id}/drain")
async def drain_worker(
    worker_id: str,
    http_request: Request,
    user: Optional[User] = Depends(get_optional_user),
) -> Dict[str, Any]:
    """Mark worker as draining (finish job, then offline).

    Args:
        worker_id: The worker ID
        http_request: The HTTP request
        user: The authenticated user

    Returns:
        Success response

    Raises:
        HTTPException: 403 if user lacks permission, 404 if worker not found
    """
    from ..services.worker_repository import get_worker_repository

    if not user or not user.has_permission("admin.workers"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: admin.workers",
        )

    client_ip = get_client_ip(http_request)
    worker_repo = get_worker_repository()

    # Get worker
    worker = await worker_repo.get_worker(worker_id)
    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worker not found: {worker_id}",
        )

    if worker.worker_type == WorkerType.PEER:
        raise HTTPException(status_code=409, detail="Remove the peer to stop monitoring this node")

    # Set status to draining
    await worker_repo.update_worker(worker_id, {"status": WorkerStatus.DRAINING.value})

    logger.info(
        "Worker %s set to draining by user %s from IP %s",
        worker_id,
        user.username if user else "system",
        client_ip,
    )

    # Emit audit event
    from ..services.cloud.audit import AuditEventType, audit_log

    await audit_log(
        AuditEventType.CONFIG_CHANGED,
        f"Worker '{worker_id}' set to draining",
        actor_id=user.id if user else None,
        actor_username=user.username if user else None,
        actor_ip=client_ip,
        target_type="worker",
        target_id=worker_id,
        details={"name": worker.name},
    )

    # Push drain command to SSE stream if connected
    if is_worker_connected(worker_id):
        await push_to_worker(worker_id, {"type": "drain", "data": {}})

    return {"success": True, "worker_id": worker_id, "status": "draining"}


@admin_router.post("/{worker_id}/token", response_model=TokenRotationResponse)
async def rotate_worker_token(
    worker_id: str,
    http_request: Request,
    user: Optional[User] = Depends(get_optional_user),
) -> TokenRotationResponse:
    """Generate new token for worker.

    Args:
        worker_id: The worker ID
        http_request: The HTTP request
        user: The authenticated user

    Returns:
        New token and connection command

    Raises:
        HTTPException: 403 if user lacks permission, 404 if worker not found
    """
    from ..services.worker_repository import get_worker_repository

    if not user or not user.has_permission("admin.workers"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: admin.workers",
        )

    client_ip = get_client_ip(http_request)
    worker_repo = get_worker_repository()

    # Get worker
    worker = await worker_repo.get_worker(worker_id)
    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worker not found: {worker_id}",
        )

    # Generate new token
    token = generate_worker_token()
    token_hash = hash_token(token)

    # Update token
    await worker_repo.update_worker(worker_id, {"token_hash": token_hash})

    logger.info(
        "Worker %s token rotated by user %s from IP %s",
        worker_id,
        user.username if user else "system",
        client_ip,
    )

    # Emit audit event
    from ..services.cloud.audit import AuditEventType, audit_log

    await audit_log(
        AuditEventType.CONFIG_CHANGED,
        f"Worker '{worker_id}' token rotated",
        actor_id=user.id if user else None,
        actor_username=user.username if user else None,
        actor_ip=client_ip,
        target_type="worker",
        target_id=worker_id,
        details={"name": worker.name},
    )

    # Build connection command with actual server URL
    server_url = str(http_request.base_url).rstrip("/")
    connection_command = f"simpletuner worker --orchestrator-url {server_url} --worker-token {token}"

    return TokenRotationResponse(
        worker_id=worker_id,
        token=token,
        connection_command=connection_command,
    )
