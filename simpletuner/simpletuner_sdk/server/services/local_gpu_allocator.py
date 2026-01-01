"""GPU allocation tracking for local training jobs."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .hardware_service import detect_gpu_inventory

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

    Thread-safe via QueueStore's SQLite WAL mode.
    GPU state is derived from running queue entries with non-null allocated_gpus.
    """

    def __init__(self):
        self._queue_store = None

    def _get_queue_store(self):
        """Lazy-load queue store to avoid circular imports."""
        if self._queue_store is None:
            from .cloud.queue import QueueStore

            self._queue_store = QueueStore()
        return self._queue_store

    async def get_allocated_gpus(self) -> Set[int]:
        """Get set of currently allocated GPU indices.

        Returns device indices from all running local jobs.
        """
        queue_store = self._get_queue_store()
        return await queue_store.get_allocated_gpus()

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

        queue_store = self._get_queue_store()
        allocated = await queue_store.get_allocated_gpus()
        running_jobs = await queue_store.get_running_local_jobs()
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
    ) -> Tuple[bool, List[int], str]:
        """Check if required GPUs can be allocated.

        Args:
            required_count: Number of GPUs needed (num_processes)
            preferred_gpus: Preferred device IDs (from config/Hardware page)
            any_gpu: If True, use any available GPUs if preferred are unavailable

        Returns:
            Tuple of (can_allocate, gpu_list, reason)
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

        queue_store = self._get_queue_store()
        running_count = await queue_store.count_running_local_jobs()

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

    async def allocate(self, job_id: str, gpu_indices: List[int]) -> bool:
        """Record GPU allocation for a job.

        Updates the queue entry's allocated_gpus field.
        """
        queue_store = self._get_queue_store()
        success = await queue_store.update_allocated_gpus(job_id, gpu_indices)
        if success:
            logger.info("Allocated GPUs %s to job %s", gpu_indices, job_id)
        return success

    async def release(self, job_id: str) -> bool:
        """Release GPUs when job completes/fails/cancels.

        Clears the allocated_gpus field in queue entry.
        """
        queue_store = self._get_queue_store()
        success = await queue_store.update_allocated_gpus(job_id, None)
        if success:
            logger.info("Released GPUs for job %s", job_id)
        return success

    async def process_pending_jobs(self) -> List[str]:
        """Process pending local jobs and start those that can run.

        Returns list of job IDs that were started.
        """
        queue_store = self._get_queue_store()
        pending_jobs = await queue_store.get_pending_local_jobs()
        started_jobs = []

        for job in pending_jobs:
            can_start, gpus, reason = await self.can_allocate(
                required_count=job.num_processes,
                preferred_gpus=job.allocated_gpus,  # May have preferred GPUs stored
                any_gpu=False,
            )

            if can_start:
                # Allocate GPUs and mark as running
                await self.allocate(job.job_id, gpus)
                await queue_store.mark_running(job.id)
                started_jobs.append(job.job_id)
                logger.info(
                    "Started queued job %s with GPUs %s",
                    job.job_id,
                    gpus,
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
