"""Worker API routes for GPU worker management.

Two sets of endpoints:
1. Worker-facing (token auth via X-Worker-Token header)
2. Admin-facing (session auth, existing patterns)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..services.cloud.auth import User, get_optional_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workers", tags=["workers"])
admin_router = APIRouter(prefix="/api/admin/workers", tags=["admin-workers"])


# Global dict of worker_id -> asyncio.Queue for SSE management
worker_streams: Dict[str, asyncio.Queue] = {}


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
    gpu_utilization: Optional[float] = None
    vram_used_gb: Optional[float] = None


class JobStatusUpdate(BaseModel):
    """Job status update from worker."""

    status: str
    progress: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class CreateWorkerRequest(BaseModel):
    """Create worker request."""

    name: str
    worker_type: str = "persistent"
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
    created_at: str
    connected: bool


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


def hash_token(token: str) -> str:
    """Hash a worker token using SHA-256.

    Args:
        token: The plaintext token

    Returns:
        The hex digest of the token hash
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_worker_token() -> str:
    """Generate a secure worker token.

    Returns:
        A URL-safe token string
    """
    return secrets.token_urlsafe(32)


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

    # Determine worker type
    worker_type = "persistent" if request.persistent else "ephemeral"

    # Update worker with registration info
    updates = {
        "name": request.name,
        "gpu_info": request.gpu_info,
        "status": "idle",
        "worker_type": worker_type.upper(),
        "labels": request.labels,
        "last_heartbeat": datetime.now(timezone.utc),
    }

    if request.provider:
        updates["provider"] = request.provider

    await worker_repo.update_worker(worker.worker_id, updates)

    # Handle reconnection reconciliation
    resume_job = None
    abandon_job = None

    if request.current_job_id:
        # Worker thinks it's running a job - verify if it should continue
        from ..services.cloud.storage.job_repository import get_job_repository

        job_repo = get_job_repository()
        job = await job_repo.get(request.current_job_id)

        if job and job.status in ["running", "starting"]:
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
            if worker_id in worker_streams:
                del worker_streams[worker_id]
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
    from ..services.worker_repository import get_worker_repository

    worker = await validate_worker_token(x_worker_token)

    if worker.worker_id != request.worker_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Worker ID mismatch",
        )

    worker_repo = get_worker_repository()

    # Update worker with heartbeat info
    updates = {
        "status": request.status,
        "last_heartbeat": datetime.now(timezone.utc),
    }

    if request.current_job_id:
        updates["current_job_id"] = request.current_job_id

    await worker_repo.update_worker(worker.worker_id, updates)

    logger.debug(
        "Heartbeat from worker %s: status=%s, job=%s",
        worker.worker_id,
        request.status,
        request.current_job_id,
    )

    return {"success": True}


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

    # Get job and verify worker ownership
    job = await job_repo.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    # Verify worker is assigned to this job
    if job.metadata and job.metadata.get("worker_id") != worker.worker_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Worker not assigned to this job",
        )

    logger.info(
        "Job %s status update from worker %s: %s",
        job_id,
        worker.worker_id,
        request.status,
    )

    # Update job status
    updates = {"status": request.status}

    if request.error:
        updates["error"] = request.error

    if request.status in ["completed", "failed", "cancelled"]:
        updates["completed_at"] = datetime.now(timezone.utc)

    await job_repo.update_job(job_id, updates)

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

    # Get workers with optional status filter
    from ..models.worker import WorkerStatus

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
    for worker in workers:
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
                created_at=worker.created_at.isoformat(),
                connected=is_worker_connected(worker.worker_id),
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
    """Manually create a persistent worker.

    Args:
        http_request: The HTTP request
        body: The create request
        user: The authenticated user

    Returns:
        Worker creation response with token

    Raises:
        HTTPException: 403 if user lacks permission
    """
    from ..models.worker import WorkerType
    from ..services.worker_repository import get_worker_repository

    if not user or not user.has_permission("admin.workers"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: admin.workers",
        )

    client_ip = get_client_ip(http_request)
    worker_repo = get_worker_repository()

    # Generate worker ID and token
    worker_id = f"worker-{secrets.token_urlsafe(8)}"
    token = generate_worker_token()
    token_hash = hash_token(token)

    # Create worker
    from ..models.worker import Worker, WorkerStatus

    worker = Worker(
        worker_id=worker_id,
        name=body.name,
        worker_type=WorkerType(body.worker_type.upper()),
        status=WorkerStatus.CONNECTING,
        token_hash=token_hash,
        user_id=user.id if user else 0,
        labels=body.labels,
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

    # Build connection command
    connection_command = (
        f"simpletuner-worker --server-url <server_url> " f"--worker-token {token} " f"--worker-id {worker_id}"
    )

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
    if worker.current_job_id and not force:
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
    from ..models.worker import WorkerStatus
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

    # Build connection command
    connection_command = (
        f"simpletuner-worker --server-url <server_url> " f"--worker-token {token} " f"--worker-id {worker_id}"
    )

    return TokenRotationResponse(
        worker_id=worker_id,
        token=token,
        connection_command=connection_command,
    )
