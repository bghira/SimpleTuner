"""Job management endpoints for cloud training.

Thin route handlers that delegate to service modules.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from ...services.cloud import CloudJobStatus, JobType, UnifiedJob
from ...services.cloud.auth import get_optional_user
from ...services.cloud.auth.models import User
from ...services.cloud.commands import CommandContext, SubmitJobCommand
from ...services.cloud.factory import ProviderFactory
from ...services.cloud.job_logs import fetch_job_logs
from ...services.cloud.job_logs import get_inline_progress as get_job_inline_progress
from ...services.cloud.job_sync import sync_active_job_statuses, sync_replicate_jobs
from ._shared import (
    JobListResponse,
    JobResponse,
    SubmitJobRequest,
    SubmitJobResponse,
    emit_cloud_event,
    enrich_jobs_with_queue_info,
    get_client_ip,
    get_job_store,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/jobs/sync")
async def sync_jobs() -> Dict[str, Any]:
    """Sync jobs from cloud providers into local store."""
    store = get_job_store()
    new_count = await sync_replicate_jobs(store)
    return {"synced": new_count, "message": f"Discovered {new_count} new jobs from Replicate"}


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    job_type: Optional[str] = Query(None, description="Filter by job type (local/cloud)"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    provider: Optional[str] = Query(None, description="Filter by cloud provider (e.g., 'replicate')"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip for pagination"),
    sync_active: bool = Query(False, description="Sync status of active cloud jobs from provider"),
    user: Optional[User] = Depends(get_optional_user),
) -> JobListResponse:
    """List jobs with optional filtering."""
    store = get_job_store()

    if sync_active:
        await sync_active_job_statuses(store)

    jt = None
    if job_type:
        try:
            jt = JobType(job_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid job_type: {job_type}. Must be 'local' or 'cloud'.",
            )

    filter_user_id = None
    if user:
        if not user.has_permission("job.view.all"):
            filter_user_id = user.id

    jobs = await store.list_jobs(
        limit=limit,
        offset=offset,
        job_type=jt,
        status=status_filter,
        user_id=filter_user_id,
        provider=provider,
    )

    job_dicts = await enrich_jobs_with_queue_info(jobs)
    return JobListResponse(jobs=job_dicts, total=len(jobs))


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    user: Optional[User] = Depends(get_optional_user),
) -> JobResponse:
    """Get details for a specific job."""
    from ...services.cloud.exceptions import JobNotFoundError, PermissionDeniedError

    store = get_job_store()
    job = await store.get_job(job_id)
    if job is None:
        raise JobNotFoundError(job_id)

    if user:
        is_own_job = job.user_id == user.id
        can_view_all = user.has_permission("job.view.all")
        if not is_own_job and not can_view_all:
            raise PermissionDeniedError("job.view.all")

    enriched = await enrich_jobs_with_queue_info([job])
    return JobResponse(job=enriched[0])


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    http_request: Request,
    user: Optional[User] = Depends(get_optional_user),
) -> Dict[str, Any]:
    """Cancel a running job."""
    from ...services.cloud.exceptions import (
        InvalidConfigError,
        JobNotFoundError,
        JobStateError,
        PermissionDeniedError,
        ProviderError,
    )

    store = get_job_store()
    client_ip = get_client_ip(http_request)
    job = await store.get_job(job_id)

    if job is None:
        raise JobNotFoundError(job_id)

    if user:
        is_own_job = job.user_id == user.id
        can_cancel_all = user.has_permission("job.cancel.all")
        can_cancel_own = user.has_permission("job.cancel.own")

        if is_own_job and not can_cancel_own and not can_cancel_all:
            raise PermissionDeniedError("job.cancel.own")
        if not is_own_job and not can_cancel_all:
            raise PermissionDeniedError("job.cancel.all")

    cancellable_states = [
        CloudJobStatus.PENDING.value,
        CloudJobStatus.UPLOADING.value,
        CloudJobStatus.QUEUED.value,
        CloudJobStatus.RUNNING.value,
    ]
    if job.status not in cancellable_states:
        raise JobStateError(job_id, job.status, cancellable_states)

    if job.job_type == JobType.CLOUD and job.provider:
        try:
            client = ProviderFactory.get_provider(job.provider)
            success = await client.cancel_job(job_id)
            if not success:
                raise ProviderError(job.provider, f"Failed to cancel job {job_id}")
        except ValueError:
            raise InvalidConfigError(f"Unknown provider: {job.provider}")
        except (JobNotFoundError, JobStateError, PermissionDeniedError, ProviderError, InvalidConfigError):
            raise
        except Exception as exc:
            logger.error("Error cancelling cloud job %s: %s", job_id, exc)
            raise ProviderError(job.provider, f"Error cancelling job: {exc}", cause=exc)

    if job.job_type == JobType.LOCAL:
        try:
            from simpletuner.simpletuner_sdk import process_keeper

            success = process_keeper.terminate_process(job_id)
            if not success:
                logger.warning("Job %s not found in process_keeper registry", job_id)
        except Exception as exc:
            logger.warning("Could not terminate local process %s: %s", job_id, exc)

    from datetime import datetime, timezone

    await store.update_job(
        job_id,
        {
            "status": CloudJobStatus.CANCELLED.value,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Update queue entry and release GPUs for local jobs
    if job.job_type == JobType.LOCAL:
        try:
            from ...services.cloud.queue import QueueStore
            from ...services.local_gpu_allocator import get_gpu_allocator

            queue_store = QueueStore()
            entry = await queue_store.get_entry_by_job_id(job_id)
            if entry:
                await queue_store.mark_cancelled(entry.id)

            allocator = get_gpu_allocator()
            await allocator.release(job_id)
        except Exception as exc:
            logger.warning("Could not update queue/release GPUs for %s: %s", job_id, exc)

    store.log_audit_event(
        action="job.cancelled",
        job_id=job_id,
        provider=job.provider,
        config_name=job.config_name,
        user_ip=client_ip,
        user_id=str(user.id) if user else None,
    )

    job_type_label = "Local" if job.is_local else "Cloud"
    emit_cloud_event(
        "cloud.job.cancelled",
        job_id,
        f"{job_type_label} job cancelled: {job.config_name or job_id[:12]}",
        severity="warning",
        config_name=job.config_name,
        provider=job.provider,
    )

    # Broadcast SSE event so UI updates training status
    try:
        from ...services.sse_manager import get_sse_manager

        sse_manager = get_sse_manager()
        await sse_manager.broadcast(
            data={
                "type": "training.status",
                "status": "cancelled",
                "job_id": job_id,
                "config_name": job.config_name,
                "message": f"{job_type_label} job cancelled",
            },
            event_type="training.status",
        )
    except Exception as exc:
        logger.warning("Failed to broadcast SSE event: %s", exc)

    return {"success": True, "job_id": job_id, "status": CloudJobStatus.CANCELLED.value}


@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    http_request: Request,
    user: Optional[User] = Depends(get_optional_user),
) -> Dict[str, Any]:
    """Delete a job from the local job history."""
    store = get_job_store()
    client_ip = get_client_ip(http_request)
    job = await store.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}")

    if user:
        is_own_job = job.user_id == user.id
        can_delete_all = user.has_permission("job.delete.all")
        can_delete_own = user.has_permission("job.delete.own")

        if is_own_job and not can_delete_own and not can_delete_all:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied: job.delete.own")
        if not is_own_job and not can_delete_all:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied: job.delete.all")

    terminal_states = {
        CloudJobStatus.COMPLETED.value,
        CloudJobStatus.FAILED.value,
        CloudJobStatus.CANCELLED.value,
    }
    if job.status not in terminal_states:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete job in state '{job.status}'. Cancel it first.",
        )

    job_provider = job.provider
    job_config_name = job.config_name

    success = await store.delete_job(job_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete job",
        )

    store.log_audit_event(
        action="job.deleted",
        job_id=job_id,
        provider=job_provider,
        config_name=job_config_name,
        user_ip=client_ip,
        user_id=str(user.id) if user else None,
    )

    return {"success": True, "job_id": job_id}


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str) -> Dict[str, Any]:
    """Fetch logs for a job."""
    store = get_job_store()
    job = await store.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}")

    logs = await fetch_job_logs(job)
    return {"job_id": job_id, "logs": logs}


@router.get("/jobs/{job_id}/logs/stream")
async def stream_job_logs(job_id: str) -> StreamingResponse:
    """Stream logs for a job in real-time (SSE).

    Tails the log file and sends new lines as they appear.
    Ends when the job reaches a terminal state.
    """
    import asyncio

    from ...services.cloud.job_logs import stream_local_logs

    store = get_job_store()
    job = await store.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}")

    async def event_generator():
        terminal_states = {"completed", "failed", "cancelled"}

        async for line in stream_local_logs(job):
            # Send line as SSE event
            yield f"data: {line}\n\n"

            # Check if job finished
            updated_job = await store.get_job(job_id)
            if updated_job and updated_job.status in terminal_states:
                yield f"event: done\ndata: {updated_job.status}\n\n"
                break

        yield "event: done\ndata: stream_end\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs/{job_id}/inline-progress")
async def get_inline_progress(job_id: str) -> Dict[str, Any]:
    """Get compact inline progress for job list display."""
    store = get_job_store()
    job = await store.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job not found: {job_id}")

    progress = await get_job_inline_progress(job)
    return {
        "job_id": progress.job_id,
        "stage": progress.stage,
        "last_log": progress.last_log,
        "progress": progress.progress,
    }


async def _build_submit_command(
    request: SubmitJobRequest,
    provider: str,
    webhook_url: Optional[str],
    user: Optional[User],
) -> "SubmitJobCommand":
    """Build a SubmitJobCommand from the request.

    Args:
        request: The job submission request
        provider: Cloud provider name
        webhook_url: Resolved webhook URL
        user: Authenticated user (if any)

    Returns:
        Configured SubmitJobCommand

    Raises:
        ValueError: If request validation fails
        Exception: If config loading fails
    """
    from ...services.cloud.commands import SubmitJobCommand

    return await SubmitJobCommand.from_request(
        config=request.config,
        dataloader_config=request.dataloader_config,
        config_name_to_load=request.config_name_to_load,
        config_name=request.config_name,
        provider=provider,
        user=user,
        webhook_url=webhook_url,
        tracker_run_name=request.tracker_run_name,
        snapshot_name=request.snapshot_name,
        snapshot_message=request.snapshot_message,
        upload_id=request.upload_id,
        idempotency_key=request.idempotency_key,
    )


def _build_command_context(
    user: Optional[User],
    client_ip: str,
    idempotency_key: Optional[str],
    store: Any,
) -> "CommandContext":
    """Build the command execution context.

    Args:
        user: Authenticated user (if any)
        client_ip: Client IP address
        idempotency_key: Optional idempotency key
        store: Job store instance

    Returns:
        Configured CommandContext
    """
    from ...services.cloud.auth.user_store import UserStore
    from ...services.cloud.commands import CommandContext

    user_store = UserStore() if user else None
    return CommandContext(
        user_id=user.id if user else None,
        user_permissions=[p for p in user.permissions] if user else [],
        client_ip=client_ip,
        idempotency_key=idempotency_key,
        job_store=store,
        user_store=user_store,
        audit_callback=store.log_audit_event,
    )


def _emit_job_submitted_event(job_id: str, config_name: Optional[str], provider: str) -> None:
    """Emit a cloud event for successful job submission.

    Args:
        job_id: The submitted job ID
        config_name: Config name used for the job
        provider: Cloud provider name
    """
    display_name = config_name or "unnamed"
    emit_cloud_event(
        "cloud.job.submitted",
        job_id,
        f"Cloud job submitted ({provider}): {display_name}",
        severity="info",
        config_name=config_name,
        provider=provider,
    )


def _build_submit_response(result: Any) -> SubmitJobResponse:
    """Build the submission response from the command result.

    Args:
        result: Command execution result

    Returns:
        SubmitJobResponse with success or error details
    """
    if not result.success:
        return SubmitJobResponse(success=False, error=result.error)

    return SubmitJobResponse(
        success=True,
        job_id=result.data.job_id,
        status=result.data.status,
        data_uploaded=result.data.data_uploaded,
        idempotent_hit=result.data.idempotent_hit,
        cost_limit_warning=result.data.cost_limit_warning,
        quota_warnings=result.warnings,
    )


@router.post("/jobs/submit", response_model=SubmitJobResponse)
async def submit_job(
    request: SubmitJobRequest,
    http_request: Request,
    provider: str = Query("replicate", description="Cloud provider to submit to"),
    user: Optional[User] = Depends(get_optional_user),
) -> SubmitJobResponse:
    """Submit a training job to a cloud provider."""
    from ...services.cloud.commands import get_dispatcher

    store = get_job_store()
    client_ip = get_client_ip(http_request)

    # Resolve webhook URL from provider config if not specified
    provider_config = await store.get_provider_config(provider)
    webhook_url = request.webhook_url or provider_config.get("webhook_url")
    if webhook_url and isinstance(webhook_url, str):
        webhook_url = webhook_url.strip() or None

    # Build command from request
    try:
        command = await _build_submit_command(request, provider, webhook_url, user)
    except ValueError as exc:
        return SubmitJobResponse(success=False, error=str(exc))
    except Exception as exc:
        logger.error("Failed to create submit command: %s", exc)
        return SubmitJobResponse(success=False, error=f"Failed to load config: {exc}")

    # Build context and dispatch
    ctx = _build_command_context(user, client_ip, request.idempotency_key, store)
    dispatcher = get_dispatcher()
    result = await dispatcher.dispatch(command, ctx)

    # Emit event on success
    if result.success and result.data:
        _emit_job_submitted_event(result.data.job_id, result.data.config_name, provider)

    return _build_submit_response(result)
