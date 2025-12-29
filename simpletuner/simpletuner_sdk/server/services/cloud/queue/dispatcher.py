"""Queue dispatcher for cloud training jobs.

Handles the actual submission of queued jobs to cloud providers.
This module bridges the queue system with the provider clients.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..base import CloudJobStatus, JobType, UnifiedJob
from ..factory import ProviderFactory
from .models import QueueEntry

if TYPE_CHECKING:
    from ..async_job_store import AsyncJobStore

logger = logging.getLogger(__name__)


class QueueDispatcher:
    """Dispatches queued jobs to cloud providers.

    This class is used as the dispatch callback for the QueueScheduler.
    When the scheduler determines a job is ready to run, it calls the
    dispatcher which handles the actual provider submission.
    """

    def __init__(self, job_store: "AsyncJobStore"):
        """Initialize the dispatcher.

        Args:
            job_store: The job store for retrieving job details and updating status.
        """
        self._job_store = job_store

    async def dispatch(self, entry: QueueEntry) -> bool:
        """Dispatch a queued job to its provider.

        This is called by the scheduler when a job is ready to run.
        The job's data should already be uploaded; this just triggers
        the provider to start training.

        Args:
            entry: The queue entry to dispatch.

        Returns:
            True if dispatch succeeded, False otherwise.

        Raises:
            Exception: If dispatch fails (scheduler will handle retry logic).
        """
        job = await self._job_store.get_job(entry.job_id)
        if job is None:
            logger.error("Job not found for queue entry: %s", entry.job_id)
            raise ValueError(f"Job not found: {entry.job_id}")

        # Job should already have data uploaded and be ready to run
        # The provider submission happens at upload time; we just need to
        # update the job status to indicate it's running via the queue

        # If the job is already submitted to the provider (has a status),
        # we just need to update our tracking
        if job.status in (
            CloudJobStatus.PENDING.value,
            CloudJobStatus.QUEUED.value,
            CloudJobStatus.RUNNING.value,
        ):
            logger.info(
                "Job %s already submitted to provider (status: %s), updating queue",
                entry.job_id,
                job.status,
            )
            return True

        # For new jobs that haven't been submitted yet, this would be where
        # we'd submit them. However, in the current architecture, jobs are
        # submitted at upload time. The queue is for managing concurrency
        # and scheduling, not the actual submission.
        #
        # Future enhancement: Move provider submission here for true queue-based
        # scheduling where jobs wait in queue before any provider interaction.

        logger.info(
            "Dispatched job %s (provider: %s, config: %s)",
            entry.job_id,
            entry.provider,
            entry.config_name,
        )

        return True

    async def on_job_completed(self, job_id: str, cost_usd: Optional[float] = None) -> None:
        """Handle job completion.

        Called when a job finishes (success or failure) to update the queue.

        Args:
            job_id: The completed job ID.
            cost_usd: Final job cost if available.
        """
        from .queue_store import QueueStore

        try:
            queue_store = QueueStore()
            entry = await queue_store.get_entry_by_job_id(job_id)
            if entry:
                await queue_store.mark_completed(entry.id)
                logger.debug("Marked queue entry as completed: %s", job_id)
        except Exception as exc:
            logger.warning("Failed to update queue on job completion: %s", exc)

    async def on_job_failed(self, job_id: str, error: str) -> None:
        """Handle job failure.

        Args:
            job_id: The failed job ID.
            error: Error message.
        """
        from .queue_store import QueueStore

        try:
            queue_store = QueueStore()
            entry = await queue_store.get_entry_by_job_id(job_id)
            if entry:
                await queue_store.mark_failed(entry.id, error)
                logger.debug("Marked queue entry as failed: %s", job_id)
        except Exception as exc:
            logger.warning("Failed to update queue on job failure: %s", exc)


# Singleton dispatcher instance
_dispatcher: Optional[QueueDispatcher] = None


def get_dispatcher(job_store: Optional["JobStore"] = None) -> QueueDispatcher:  # noqa: F821
    """Get the singleton queue dispatcher.

    Args:
        job_store: Job store instance (required on first call).

    Returns:
        The queue dispatcher instance.
    """
    global _dispatcher
    if _dispatcher is None:
        if job_store is None:
            from ..container import get_job_store

            job_store = get_job_store()
        _dispatcher = QueueDispatcher(job_store)
    return _dispatcher


def reset_dispatcher() -> None:
    """Reset the dispatcher singleton (for testing)."""
    global _dispatcher
    _dispatcher = None
