"""System status API endpoints."""

import asyncio
import logging
import os
import signal
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import get_current_user
from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User
from simpletuner.simpletuner_sdk.server.services.maintenance_service import MAINTENANCE_SERVICE, MaintenanceServiceError
from simpletuner.simpletuner_sdk.server.services.system_status_service import SystemStatusService

router = APIRouter(prefix="/api/system", tags=["system"])
_service = SystemStatusService()
logger = logging.getLogger("SimpleTunerSystemRoutes")

_SHUTDOWN_IN_PROGRESS = False
_SHUTDOWN_DELAY_SECONDS = 0.75


def _coerce_pid(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        pid = int(str(value))
    except (TypeError, ValueError):
        return None
    if pid <= 1:
        return None
    return pid


async def _terminate_training_processes() -> int:
    """Attempt to terminate all tracked training processes."""
    try:
        from simpletuner.simpletuner_sdk import process_keeper  # Imported lazily to avoid circular deps
    except Exception as exc:  # pragma: no cover - defensive in case module import fails
        logger.debug("process_keeper import failed during shutdown: %s", exc)
        return 0

    try:
        processes = process_keeper.list_processes() or {}
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.warning("Unable to list training processes while shutting down: %s", exc)
        return 0

    terminated = 0
    for job_id in list(processes.keys()):
        try:
            await asyncio.to_thread(process_keeper.terminate_process, job_id)
            terminated += 1
        except Exception as exc:  # pragma: no cover - terminating should not crash shutdown
            logger.warning("Failed to terminate training job %s: %s", job_id, exc)
    return terminated


def _signal_process_exit() -> None:
    """Signal running processes (including reload supervisors) to exit, then hard-exit."""
    pid = os.getpid()
    target_pids = set()

    root_pid = _coerce_pid(os.environ.get("SIMPLETUNER_SERVER_ROOT_PID"))
    if root_pid:
        target_pids.add(root_pid)

    managed_parent_pid = _coerce_pid(os.environ.get("SIMPLETUNER_SERVER_PARENT_PID"))
    if managed_parent_pid:
        target_pids.add(managed_parent_pid)

    if root_pid and root_pid == os.getppid():
        target_pids.add(root_pid)

    try:
        import multiprocessing

        parent_proc = multiprocessing.parent_process()
    except Exception:  # pragma: no cover - best effort reflection
        parent_proc = None

    if parent_proc and parent_proc.pid and parent_proc.pid > 1:
        target_pids.add(parent_proc.pid)
    if parent_proc and parent_proc.pid and parent_proc.pid > 1:
        target_pids.add(parent_proc.pid)

    # Never send a signal to the current worker via os.kill; we'll exit via os._exit below.
    if pid in target_pids:
        target_pids.remove(pid)

    sent_signal = False
    for target_pid in list(target_pids):
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                os.kill(target_pid, sig)
                sent_signal = True
                break
            except Exception as exc:  # pragma: no cover - best effort logging
                logger.debug("Sending %s to pid %s failed: %s", sig_name, target_pid, exc)
                continue

    if target_pids and not sent_signal:
        logger.warning("Unable to deliver shutdown signals to parent processes; forcing exit")

    if parent_proc:
        try:
            parent_proc.terminate()
        except Exception:  # pragma: no cover - best effort logging
            logger.debug("Unable to terminate parent process via multiprocessing API", exc_info=True)

    # Finally exit the current worker immediately regardless of outstanding ASGI tasks.
    os._exit(0)


async def _initiate_shutdown_sequence(delay_seconds: float = _SHUTDOWN_DELAY_SECONDS) -> None:
    """Background task that handles graceful shutdown."""
    logger.info("Shutdown requested via API; beginning graceful termination.")
    try:
        terminated = await _terminate_training_processes()
        if terminated:
            logger.info("Requested termination for %s training process(es) before shutdown.", terminated)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Error while terminating training jobs during shutdown: %s", exc)

    await asyncio.sleep(max(delay_seconds, 0.1))
    _signal_process_exit()


@router.get("/status")
async def get_system_status(include_allocation: bool = False, _user: User = Depends(get_current_user)) -> dict:
    """Return current system load, memory usage, and GPU utilisation.

    Args:
        include_allocation: If True, include GPU allocation info for local training jobs.
    """
    status = _service.get_status()

    if include_allocation:
        try:
            from simpletuner.simpletuner_sdk.server.services.local_gpu_allocator import get_gpu_allocator

            allocator = get_gpu_allocator()
            gpu_status = await allocator.get_gpu_status()
            status["gpu_allocation"] = {
                "allocated_gpus": gpu_status.get("allocated_gpus", []),
                "available_gpus": gpu_status.get("available_gpus", []),
                "running_local_jobs": gpu_status.get("running_local_jobs", 0),
                "devices": gpu_status.get("devices", []),
            }
        except Exception as exc:
            logger.debug("Failed to get GPU allocation status: %s", exc)
            status["gpu_allocation"] = None

    return status


@router.post("/maintenance/clear-fsdp-block-cache")
async def clear_fsdp_block_cache(_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Clear cached FSDP detection metadata."""
    try:
        return MAINTENANCE_SERVICE.clear_fsdp_block_cache()
    except MaintenanceServiceError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=exc.message) from exc


@router.post("/maintenance/clear-deepspeed-offload")
async def clear_deepspeed_offload(
    payload: Optional[Dict[str, Any]] = None, _user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete DeepSpeed NVMe offload cache directories."""
    config_name = None
    if isinstance(payload, dict):
        config_name = payload.get("config") or payload.get("config_name")
    try:
        return MAINTENANCE_SERVICE.clear_deepspeed_offload_cache(config_name=config_name)
    except MaintenanceServiceError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc


@router.post("/shutdown")
async def shutdown_simpletuner(background_tasks: BackgroundTasks, _user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Schedule a graceful shutdown of the SimpleTuner process."""
    global _SHUTDOWN_IN_PROGRESS
    if _SHUTDOWN_IN_PROGRESS:
        return {"status": "shutting_down", "message": "Shutdown already in progress."}

    _SHUTDOWN_IN_PROGRESS = True
    background_tasks.add_task(_initiate_shutdown_sequence)

    return {
        "status": "shutting_down",
        "message": "SimpleTuner is shutting down. Active training processes will be stopped.",
    }
