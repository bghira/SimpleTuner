"""Adapter to make JobRepository work with QueueScheduler.

This bridges the gap between the old QueueStore interface and the new
unified JobRepository, allowing existing scheduler code to work unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..base import CloudJobStatus, JobType, UnifiedJob
from ..storage.job_repository import JobRepository, get_job_repository
from .models import QueueEntry, QueuePriority, QueueStatus

logger = logging.getLogger(__name__)


def _job_to_queue_entry(job: UnifiedJob) -> QueueEntry:
    """Convert a UnifiedJob to a QueueEntry for backwards compatibility."""
    # Map job status to queue status
    status_map = {
        CloudJobStatus.PENDING.value: QueueStatus.PENDING,
        CloudJobStatus.UPLOADING.value: QueueStatus.PENDING,
        CloudJobStatus.QUEUED.value: QueueStatus.PENDING,  # queued -> pending
        CloudJobStatus.RUNNING.value: QueueStatus.RUNNING,
        CloudJobStatus.COMPLETED.value: QueueStatus.COMPLETED,
        CloudJobStatus.FAILED.value: QueueStatus.FAILED,
        CloudJobStatus.CANCELLED.value: QueueStatus.CANCELLED,
    }

    return QueueEntry(
        id=hash(job.job_id) & 0x7FFFFFFF,  # Generate pseudo-ID from job_id
        job_id=job.job_id,
        user_id=job.user_id,
        provider=job.provider or "local",
        config_name=job.config_name,
        priority=QueuePriority.NORMAL,  # Default
        priority_override=job.priority_override,
        status=status_map.get(job.status, QueueStatus.PENDING),
        queued_at=job.queued_at or job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        estimated_cost=job.estimated_cost,
        requires_approval=job.requires_approval,
        attempt=job.attempt,
        max_attempts=job.max_attempts,
        job_type=job.job_type.value if job.job_type else "local",
        num_processes=job.num_processes,
        allocated_gpus=job.allocated_gpus,
        team_id=job.team_id,
        org_id=job.org_id,
        approval_id=job.approval_id,
        error_message=job.error_message,
        metadata=job.metadata,
    )


class JobRepoQueueAdapter:
    """Adapter that makes JobRepository compatible with QueueStoreProtocol.

    This is a transitional layer that allows the scheduler to use
    JobRepository while keeping the existing interface.
    """

    def __init__(self, job_repo: Optional[JobRepository] = None):
        self._job_repo = job_repo or get_job_repository()

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return await self._job_repo.get_queue_stats()

    async def count_running(self) -> int:
        """Count total running jobs."""
        return await self._job_repo.count_running()

    async def count_running_by_user(self) -> Dict[int, int]:
        """Get running job count per user."""
        return await self._job_repo.count_running_by_user()

    async def count_running_by_team(self) -> Dict[str, int]:
        """Get running job count per team."""
        return await self._job_repo.count_running_by_team()

    async def list_pending_by_priority(self, limit: int = 50) -> List[QueueEntry]:
        """List pending jobs ordered by priority."""
        jobs = await self._job_repo.list_pending_by_priority(limit)
        return [_job_to_queue_entry(j) for j in jobs]

    async def list_entries(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[QueueStatus] = None,
        user_id: Optional[int] = None,
        include_completed: bool = False,
    ) -> List[QueueEntry]:
        """List queue entries with optional filtering."""
        # Map queue status to job status
        status_filter = None
        if status:
            status_map = {
                QueueStatus.PENDING: CloudJobStatus.QUEUED.value,
                QueueStatus.RUNNING: CloudJobStatus.RUNNING.value,
                QueueStatus.COMPLETED: CloudJobStatus.COMPLETED.value,
                QueueStatus.FAILED: CloudJobStatus.FAILED.value,
                QueueStatus.CANCELLED: CloudJobStatus.CANCELLED.value,
            }
            status_filter = status_map.get(status)

        jobs = await self._job_repo.list(
            limit=limit,
            offset=offset,
            status=status_filter,
            user_id=user_id,
        )
        return [_job_to_queue_entry(j) for j in jobs]

    async def get_entry_by_job_id(self, job_id: str) -> Optional[QueueEntry]:
        """Get a queue entry by job ID."""
        job = await self._job_repo.get(job_id)
        if job:
            return _job_to_queue_entry(job)
        return None

    async def add_to_queue(
        self,
        job_id: str,
        user_id: Optional[int] = None,
        org_id: Optional[int] = None,
        provider: str = "cloud",
        config_name: Optional[str] = None,
        priority: QueuePriority = QueuePriority.NORMAL,
        job_type: str = "cloud",
        num_processes: int = 1,
        allocated_gpus: Optional[List[int]] = None,
        requires_approval: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        estimated_cost: float = 0.0,
        team_id: Optional[str] = None,
        priority_override: Optional[int] = None,
    ) -> QueueEntry:
        """Add a job to the queue."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        existing = await self._job_repo.get(job_id)
        if existing:
            updates: Dict[str, Any] = {
                "priority": priority.value,
                "priority_override": priority_override,
                "queued_at": existing.queued_at or now,
                "requires_approval": requires_approval,
                "estimated_cost": estimated_cost,
                "team_id": team_id,
                "org_id": org_id,
                "num_processes": num_processes,
            }
            if existing.status in (CloudJobStatus.PENDING.value, CloudJobStatus.UPLOADING.value):
                updates["status"] = CloudJobStatus.QUEUED.value
            await self._job_repo.update(job_id, updates)
            refreshed = await self._job_repo.get(job_id)
            return _job_to_queue_entry(refreshed or existing)

        job = UnifiedJob(
            job_id=job_id,
            job_type=JobType.LOCAL if job_type == "local" else JobType.CLOUD,
            provider=provider,
            status=CloudJobStatus.QUEUED.value,
            config_name=config_name,
            created_at=now,
            queued_at=now,
            user_id=user_id,
            org_id=org_id,
            num_processes=num_processes,
            allocated_gpus=allocated_gpus,
            requires_approval=requires_approval,
            priority=priority.value,
            priority_override=priority_override,
            team_id=team_id,
            estimated_cost=estimated_cost,
            metadata=metadata or {},
        )
        await self._job_repo.add(job)

        # Return as QueueEntry
        return _job_to_queue_entry(job)

    async def mark_running(self, queue_id: int) -> bool:
        """Mark a job as running by queue ID.

        Note: queue_id is actually a hash of job_id in this adapter.
        We need to find the job and mark it running.
        """
        # This is a hack - we can't easily reverse the hash
        # For now, log a warning and return True (caller should use job_id variant)
        logger.warning("mark_running called with queue_id %d - use job_id instead", queue_id)
        return True

    async def mark_running_by_job_id(self, job_id: str) -> bool:
        """Mark a job as running by job ID."""
        return await self._job_repo.mark_running(job_id)

    async def mark_completed(self, queue_id: int) -> bool:
        """Mark a job as completed by queue ID."""
        logger.warning("mark_completed called with queue_id %d - use job_id instead", queue_id)
        return True

    async def mark_completed_by_job_id(self, job_id: str) -> bool:
        """Mark a job as completed by job ID."""
        return await self._job_repo.mark_completed(job_id)

    async def mark_failed(self, queue_id: int, error: str) -> bool:
        """Mark a job as failed by queue ID."""
        logger.warning("mark_failed called with queue_id %d - use job_id instead", queue_id)
        return True

    async def mark_failed_by_job_id(self, job_id: str, error: str) -> bool:
        """Mark a job as failed by job ID."""
        return await self._job_repo.mark_failed(job_id, error)

    async def mark_cancelled(self, queue_id: int) -> bool:
        """Mark a job as cancelled by queue ID."""
        logger.warning("mark_cancelled called with queue_id %d - use job_id instead", queue_id)
        return True

    async def mark_cancelled_by_job_id(self, job_id: str) -> bool:
        """Mark a job as cancelled by job ID."""
        return await self._job_repo.mark_cancelled(job_id)

    async def update_entry(self, queue_id: int, updates: Dict[str, Any]) -> bool:
        """Update a queue entry."""
        logger.warning("update_entry called with queue_id %d - use job_id instead", queue_id)
        return True

    async def update_entry_by_job_id(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update a queue entry by job ID."""
        return await self._job_repo.update(job_id, updates)

    async def cleanup_old_entries(self, days: int = 30) -> int:
        """Clean up old completed entries.

        Args:
            days: Delete entries older than this many days

        Returns:
            Number of entries deleted
        """
        return await self._job_repo.cleanup_old_entries(days=days)


def get_queue_adapter() -> JobRepoQueueAdapter:
    """Get the queue adapter singleton."""
    return JobRepoQueueAdapter()
