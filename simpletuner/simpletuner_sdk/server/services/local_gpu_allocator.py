"""GPU allocation tracking for local training jobs."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from .hardware_service import detect_gpu_inventory

if TYPE_CHECKING:
    from .cloud.base import UnifiedJob

logger = logging.getLogger(__name__)

# Singleton instance
_allocator_instance: Optional["LocalGPUAllocator"] = None


def get_gpu_allocator() -> "LocalGPUAllocator":
    """Get the singleton LocalGPUAllocator instance."""
    global _allocator_instance
    if _allocator_instance is None:
        _allocator_instance = LocalGPUAllocator()
    return _allocator_instance


class LocalGPUAllocator:
    """Manages GPU allocation state for local training jobs.

    Thread-safe via JobRepository's SQLite WAL mode.
    GPU state is derived from running jobs with non-null allocated_gpus.
    """

    def __init__(self):
        self._job_repo = None
        self._reconciled = False

    async def reconcile_on_startup(self, max_entries: int = 20) -> Dict[str, int]:
        """Reconcile LOCAL jobs with actual process status.

        Called on startup to check running LOCAL jobs and handle orphaned ones:
        - Jobs with live PIDs are marked as "adopted" (process alive, IPC broken)
        - Jobs with dead PIDs are marked as failed
        - Jobs without PIDs (legacy) are marked as failed

        Args:
            max_entries: Maximum number of running entries to check (default 20).

        Returns:
            Dictionary with stats: {"orphaned": N, "adopted": N, "no_pid": N}
        """
        stats = {"orphaned": 0, "adopted": 0, "no_pid": 0}

        if self._reconciled:
            return stats

        job_repo = self._get_job_repo()
        # Only get LOCAL running jobs - this is bounded by the number of
        # GPUs on this machine, so should be small (typically 0-8)
        running_jobs = await job_repo.get_running_local_jobs()

        if not running_jobs:
            self._reconciled = True
            return stats

        # Limit entries to check
        jobs_to_check = running_jobs[:max_entries]
        if len(running_jobs) > max_entries:
            logger.warning(
                "Found %d running local jobs, only checking first %d",
                len(running_jobs),
                max_entries,
            )

        for job in jobs_to_check:
            pid = job.metadata.get("pid") if job.metadata else None

            if pid:
                if self._is_process_alive(pid):
                    # Process still alive - mark as "adopted" (IPC broken but killable)
                    metadata = job.metadata.copy() if job.metadata else {}
                    metadata["orphaned"] = True  # IPC is broken
                    await job_repo.update(job.job_id, {"metadata": metadata})
                    stats["adopted"] += 1
                    logger.warning(
                        "Job %s adopted (PID %d alive, IPC broken)",
                        job.job_id,
                        pid,
                    )
                else:
                    # Process died - mark as failed
                    await job_repo.mark_failed(
                        job.job_id,
                        "Process crashed or killed (orphaned after restart)",
                    )
                    await self.release(job.job_id)
                    stats["orphaned"] += 1
                    logger.info(
                        "Job %s marked as failed (PID %d dead)",
                        job.job_id,
                        pid,
                    )
            else:
                # No PID stored - legacy entry, mark as failed
                await job_repo.mark_failed(
                    job.job_id,
                    "Job orphaned (no PID tracked)",
                )
                await self.release(job.job_id)
                stats["no_pid"] += 1
                logger.info(
                    "Job %s marked as failed (no PID tracked)",
                    job.job_id,
                )

        self._reconciled = True
        return stats

    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process with given PID is still running.

        Args:
            pid: The process ID to check.

        Returns:
            True if the process exists and is running, False otherwise.
        """
        try:
            os.kill(pid, 0)  # Signal 0 = check if process exists
            return True
        except OSError:
            return False

    async def _reconcile_stale_entries(self, job_repo) -> int:
        """No-op: With unified JobRepository, there are no stale entries to reconcile.

        This method previously reconciled between separate QueueStore and JobStore.
        Now that they're merged, it's kept for API compatibility but does nothing.

        Returns:
            Always returns 0.
        """
        return 0

    def _get_job_repo(self):
        """Lazy-load job repository to avoid circular imports."""
        if self._job_repo is None:
            from .cloud.storage.job_repository import get_job_repository

            self._job_repo = get_job_repository()
        return self._job_repo

    async def get_allocated_gpus(self) -> Set[int]:
        """Get set of currently allocated GPU indices.

        Returns device indices from all running local jobs.
        """
        job_repo = self._get_job_repo()
        return await job_repo.get_allocated_gpus()

    async def get_available_gpus(self) -> List[int]:
        """Get list of available GPU indices.

        Returns GPUs that exist but are not currently allocated.
        """
        inventory = detect_gpu_inventory()
        all_gpus = [d["index"] for d in inventory.get("devices", [])]
        allocated = await self.get_allocated_gpus()
        return [gpu for gpu in all_gpus if gpu not in allocated]

    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get comprehensive GPU allocation status.

        Returns:
            Dictionary with total_gpus, allocated_gpus, available_gpus,
            running_local_jobs, and GPU details.
        """
        inventory = detect_gpu_inventory()
        all_devices = inventory.get("devices", [])
        all_gpu_indices = [d["index"] for d in all_devices]

        job_repo = self._get_job_repo()

        # No-op: with unified store, no stale entries to reconcile
        await self._reconcile_stale_entries(job_repo)

        allocated = await job_repo.get_allocated_gpus()
        running_jobs = await job_repo.get_running_local_jobs()
        available = [gpu for gpu in all_gpu_indices if gpu not in allocated]

        # Build per-GPU status
        gpu_details = []
        for device in all_devices:
            idx = device["index"]
            job_id = None
            for job in running_jobs:
                if job.allocated_gpus and idx in job.allocated_gpus:
                    job_id = job.job_id
                    break
            gpu_details.append(
                {
                    "index": idx,
                    "name": device.get("name", f"GPU {idx}"),
                    "memory_gb": device.get("memory_gb"),
                    "allocated": idx in allocated,
                    "job_id": job_id,
                }
            )

        return {
            "total_gpus": len(all_devices),
            "allocated_gpus": sorted(allocated),
            "available_gpus": available,
            "running_local_jobs": len(running_jobs),
            "backend": inventory.get("backend", "cpu"),
            "devices": gpu_details,
        }

    async def can_allocate(
        self,
        required_count: int,
        preferred_gpus: Optional[List[int]] = None,
        any_gpu: bool = False,
        org_id: Optional[int] = None,
        for_approval: bool = False,
    ) -> Tuple[bool, List[int], str]:
        """Check if required GPUs can be allocated.

        Args:
            required_count: Number of GPUs needed (num_processes)
            preferred_gpus: Preferred device IDs (from config/Hardware page)
            any_gpu: If True, use any available GPUs if preferred are unavailable
            org_id: Organization ID for org-level quota checks
            for_approval: If True, job will be submitted for approval if over quota

        Returns:
            Tuple of (can_allocate, gpu_list, reason)
            - If for_approval is True and org quota exceeded, reason starts with "APPROVAL_REQUIRED:"
        """
        inventory = detect_gpu_inventory()
        all_gpus = set(d["index"] for d in inventory.get("devices", []))
        allocated = await self.get_allocated_gpus()
        available = sorted(gpu for gpu in all_gpus if gpu not in allocated)

        # Check concurrency limits
        from .webui_state import WebUIStateStore

        try:
            defaults = WebUIStateStore().load_defaults()
            max_gpus = getattr(defaults, "local_gpu_max_concurrent", None)
            max_jobs = getattr(defaults, "local_job_max_concurrent", 1) or 1
        except Exception:
            max_gpus = None
            max_jobs = 1

        job_repo = self._get_job_repo()
        running_count = await job_repo.count_running_local_jobs()

        if running_count >= max_jobs:
            return (
                False,
                [],
                f"Maximum concurrent local jobs ({max_jobs}) reached",
            )

        if max_gpus is not None:
            gpus_in_use = len(allocated)
            if gpus_in_use + required_count > max_gpus:
                return (
                    False,
                    [],
                    f"Would exceed GPU limit ({gpus_in_use + required_count} > {max_gpus})",
                )

        # Check org-level GPU quota
        if org_id is not None:
            org_quota_result = await self._check_org_gpu_quota(org_id, required_count, for_approval)
            if org_quota_result is not None:
                return org_quota_result

        # Check if we have enough GPUs
        if len(available) < required_count:
            return (
                False,
                [],
                f"Insufficient GPUs available ({len(available)} available, {required_count} required)",
            )

        # If preferred GPUs are specified, try to use them
        if preferred_gpus and not any_gpu:
            preferred_available = [gpu for gpu in preferred_gpus if gpu in available]
            if len(preferred_available) >= required_count:
                return (True, preferred_available[:required_count], "")
            elif len(preferred_available) < required_count:
                return (
                    False,
                    [],
                    f"Preferred GPUs {preferred_gpus} not fully available "
                    f"(only {preferred_available} available, need {required_count})",
                )

        # Use any available GPUs
        return (True, available[:required_count], "")

    async def _check_org_gpu_quota(
        self,
        org_id: int,
        required_count: int,
        for_approval: bool = False,
    ) -> Optional[Tuple[bool, List[int], str]]:
        """Check if org GPU quota allows allocation.

        Args:
            org_id: Organization ID
            required_count: Number of GPUs requested
            for_approval: If True, return approval-required message instead of blocking

        Returns:
            None if quota allows allocation, otherwise a (False, [], reason) tuple.
            If for_approval is True and overflow is allowed, reason starts with
            "APPROVAL_REQUIRED:" to signal the job should be queued for approval.
        """
        try:
            from .cloud.container import get_user_store

            user_store = get_user_store()

            # Get org's LOCAL_GPUS quota
            org_quotas = await user_store.get_org_quotas(org_id)
            local_gpu_quota = None
            for quota in org_quotas:
                if quota.quota_type.value == "local_gpus":
                    local_gpu_quota = quota
                    break

            if local_gpu_quota is None:
                # No org GPU quota set, allow
                return None

            # Get current org GPU usage
            job_repo = self._get_job_repo()
            org_gpu_usage = await job_repo.get_org_gpu_usage(org_id)

            if org_gpu_usage + required_count > local_gpu_quota.limit_value:
                # Would exceed org quota
                if for_approval:
                    # Check if org allows overflow approval
                    org = await user_store.get_organization(org_id)
                    if org and org.settings.get("allow_overflow_approval", False):
                        return (
                            False,
                            [],
                            f"APPROVAL_REQUIRED:Would exceed org GPU limit "
                            f"({org_gpu_usage + required_count} > {int(local_gpu_quota.limit_value)})",
                        )

                return (
                    False,
                    [],
                    f"Would exceed organization GPU limit "
                    f"({org_gpu_usage + required_count} > {int(local_gpu_quota.limit_value)})",
                )

            return None  # Quota allows allocation

        except Exception as exc:
            logger.warning("Failed to check org GPU quota: %s", exc)
            # Don't block on quota check failure
            return None

    async def allocate(self, job_id: str, gpu_indices: List[int]) -> bool:
        """Record GPU allocation for a job.

        Updates the job's allocated_gpus field.
        """
        job_repo = self._get_job_repo()
        success = await job_repo.update_allocated_gpus(job_id, gpu_indices)
        if success:
            logger.info("Allocated GPUs %s to job %s", gpu_indices, job_id)
        return success

    async def release(self, job_id: str) -> bool:
        """Release GPUs when job completes/fails/cancels.

        Clears the allocated_gpus field in job.
        """
        job_repo = self._get_job_repo()
        success = await job_repo.update_allocated_gpus(job_id, None)
        if success:
            logger.info("Released GPUs for job %s", job_id)
        return success

    async def process_pending_jobs(self) -> List[str]:
        """Process pending local jobs and start those that can run.

        Returns list of job IDs that were started.
        """
        logger.info("process_pending_jobs called - checking for pending jobs")
        job_repo = self._get_job_repo()
        pending_jobs = await job_repo.get_pending_local_jobs()
        logger.info("Found %d pending local jobs", len(pending_jobs))
        started_jobs = []

        for job in pending_jobs:
            # Get any_gpu setting from metadata
            metadata = job.metadata or {}
            any_gpu = metadata.get("any_gpu", False)

            can_start, gpus, reason = await self.can_allocate(
                required_count=job.num_processes,
                preferred_gpus=job.allocated_gpus if not any_gpu else None,
                any_gpu=any_gpu,
            )

            if can_start:
                # Allocate GPUs first - must succeed before marking running
                if not await self.allocate(job.job_id, gpus):
                    logger.error(
                        "Failed to allocate GPUs %s to job %s",
                        gpus,
                        job.job_id,
                    )
                    await job_repo.mark_failed(job.job_id, f"Failed to allocate GPUs {gpus}")
                    continue

                # Mark as running only after successful allocation
                await job_repo.mark_running(job.job_id)

                # Actually start the training job
                try:
                    await self._start_queued_job(job, gpus)
                    started_jobs.append(job.job_id)
                    logger.info(
                        "Started queued job %s with GPUs %s",
                        job.job_id,
                        gpus,
                    )
                except Exception as exc:
                    logger.error("Failed to start queued job %s: %s", job.job_id, exc)
                    # Release GPUs and mark as failed - ensure both happen
                    try:
                        await self.release(job.job_id)
                    except Exception as release_exc:
                        logger.error(
                            "Failed to release GPUs for job %s: %s",
                            job.job_id,
                            release_exc,
                        )
                    try:
                        await job_repo.mark_failed(job.job_id, f"Failed to start: {exc}")
                    except Exception as mark_exc:
                        logger.error(
                            "Failed to mark job %s as failed: %s",
                            job.job_id,
                            mark_exc,
                        )
            else:
                # Stop processing - jobs are ordered by priority
                logger.debug(
                    "Job %s cannot start: %s",
                    job.job_id,
                    reason,
                )
                break

        return started_jobs

    async def _start_queued_job(self, job: "UnifiedJob", gpus: List[int]) -> None:
        """Actually start a queued training job.

        Args:
            job: The UnifiedJob with job metadata.
            gpus: The GPUs allocated to this job.
        """
        from . import training_service

        metadata = job.metadata or {}
        runtime_config = metadata.get("runtime_config", {})

        if not runtime_config:
            raise ValueError("No runtime_config in job metadata")

        # Update config with allocated GPUs and job ID
        runtime_config["accelerate_visible_devices"] = gpus
        runtime_config["--num_processes"] = len(gpus)
        runtime_config["__job_id__"] = job.job_id
        # Ensure the trainer surfaces configuration parsing errors instead of silently
        # falling back to config/config.json when launched from the WebUI.
        runtime_config["__skip_config_fallback__"] = True

        # Submit to process_keeper
        from simpletuner.simpletuner_sdk import process_keeper

        process_keeper.submit_job(
            job.job_id,
            training_service.run_trainer_job,
            runtime_config,
        )

        # Store PID in job metadata for orphan detection on restart
        pid = process_keeper.get_process_pid(job.job_id)
        if pid:
            job_repo = self._get_job_repo()
            updated_metadata = metadata.copy()
            updated_metadata["pid"] = pid
            await job_repo.update(job.job_id, {"metadata": updated_metadata})
            logger.debug("Stored PID %d for job %s", pid, job.job_id)

        # Update APIState
        training_service.APIState.set_state("current_job_id", job.job_id)
        training_service.APIState.set_state("training_status", "starting")
        training_service.APIState.set_state("training_config", runtime_config)

        # Broadcast SSE event so UI updates in real-time
        try:
            from .sse_manager import get_sse_manager

            sse_manager = get_sse_manager()
            num_connections = len(sse_manager.connections) if hasattr(sse_manager, "connections") else "unknown"
            logger.info(
                "Broadcasting SSE event for queued job %s (connections: %s)",
                job.job_id,
                num_connections,
            )
            await sse_manager.broadcast(
                data={
                    "type": "training.status",
                    "status": "starting",
                    "job_id": job.job_id,
                    "config_name": job.config_name,
                    "allocated_gpus": gpus,
                    "message": f"Queued training job {job.job_id} starting",
                },
                event_type="training.status",
            )
            logger.info("SSE broadcast completed for job %s", job.job_id)
        except Exception as exc:
            logger.warning("Failed to broadcast SSE event for queued job: %s", exc, exc_info=True)
