"""Job queue scheduler for cloud training.

Handles job dispatch, concurrency management, and fair scheduling.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .models import QueueEntry, QueuePriority, QueueStatus, get_priority_for_level
from .protocol import QueueStoreProtocol

logger = logging.getLogger(__name__)


@dataclass
class SchedulingConfig:
    """Configuration for the scheduling policy."""

    max_concurrent: int = 5
    """Global maximum running jobs."""

    user_max_concurrent: int = 2
    """Maximum running jobs per user."""

    team_max_concurrent: int = 10
    """Maximum running jobs per team (for fair-share)."""

    enable_fair_share: bool = False
    """Enable fair-share scheduling across teams."""

    starvation_threshold_minutes: int = 60
    """Boost priority for jobs waiting longer than this."""


@dataclass
class SchedulingDecision:
    """Result of a scheduling decision."""

    entry: Optional[QueueEntry]
    """The entry to schedule, or None if nothing is ready."""

    reason: str
    """Human-readable reason for the decision."""

    blocked_users: List[int] = field(default_factory=list)
    """Users who have jobs but are blocked by concurrency limits."""


class SchedulingPolicy:
    """Encapsulates the scheduling algorithm.

    Separates scheduling logic from data access, making it testable
    and allowing different policies to be swapped in.

    The default policy implements:
    - Priority-based ordering (higher priority first)
    - FIFO within same priority (older jobs first)
    - Global concurrency limit
    - Per-user concurrency limit
    """

    def __init__(self, config: Optional[SchedulingConfig] = None):
        """Initialize with configuration.

        Args:
            config: Scheduling configuration, or use defaults.
        """
        self.config = config or SchedulingConfig()

    def select_next(
        self,
        running_count: int,
        user_running: Dict[int, int],
        pending_entries: List[QueueEntry],
        team_running: Optional[Dict[str, int]] = None,
    ) -> SchedulingDecision:
        """Select the next entry to run.

        This is a pure function that makes scheduling decisions based on
        the current state, without any database access.

        Args:
            running_count: Total number of currently running jobs
            user_running: Dict mapping user_id -> running count
            pending_entries: List of pending entries, pre-sorted by priority desc, age asc
            team_running: Dict mapping team_id -> running count (optional, for fair-share)

        Returns:
            SchedulingDecision with the selected entry (or None) and reason
        """
        # Check global concurrency limit
        if running_count >= self.config.max_concurrent:
            return SchedulingDecision(
                entry=None,
                reason=f"Global limit reached ({running_count}/{self.config.max_concurrent})",
            )

        if not pending_entries:
            return SchedulingDecision(
                entry=None,
                reason="No pending jobs in queue",
            )

        blocked_users: List[int] = []
        team_running = team_running or {}

        # Find first candidate that passes concurrency limits
        for entry in pending_entries:
            # Check user concurrency limit
            if entry.user_id is not None:
                user_count = user_running.get(entry.user_id, 0)
                if user_count >= self.config.user_max_concurrent:
                    if entry.user_id not in blocked_users:
                        blocked_users.append(entry.user_id)
                    continue  # Skip, user at limit

            # Check team concurrency limit (when fair-share is enabled)
            if self.config.enable_fair_share and entry.team_id is not None:
                team_count = team_running.get(entry.team_id, 0)
                if team_count >= self.config.team_max_concurrent:
                    logger.debug(
                        "Job %s blocked: team %s at limit (%d/%d)",
                        entry.job_id,
                        entry.team_id,
                        team_count,
                        self.config.team_max_concurrent,
                    )
                    continue  # Skip, team at limit

            return SchedulingDecision(
                entry=entry,
                reason=f"Selected job {entry.job_id} (priority={entry.effective_priority})",
                blocked_users=blocked_users,
            )

        # All candidates blocked by limits
        if blocked_users:
            return SchedulingDecision(
                entry=None,
                reason="All pending jobs blocked by per-user concurrency limits",
                blocked_users=blocked_users,
            )

        return SchedulingDecision(
            entry=None,
            reason="All pending jobs blocked by team concurrency limits",
            blocked_users=blocked_users,
        )

    def can_accept_job(
        self,
        user_id: Optional[int],
        running_count: int,
        user_running: Dict[int, int],
    ) -> Tuple[bool, str]:
        """Check if a new job can be accepted.

        Args:
            user_id: User submitting the job
            running_count: Current running job count
            user_running: Dict mapping user_id -> running count

        Returns:
            Tuple of (can_accept, reason)
        """
        if running_count >= self.config.max_concurrent:
            return False, f"Global limit reached ({self.config.max_concurrent})"

        if user_id is not None:
            user_count = user_running.get(user_id, 0)
            if user_count >= self.config.user_max_concurrent:
                return False, f"User limit reached ({self.config.user_max_concurrent})"

        return True, "Job can be scheduled"


class QueueScheduler:
    """Manages job scheduling and dispatch.

    Responsibilities:
    - Pick next job to run based on priority and fairness
    - Enforce global and per-user concurrency limits
    - Handle job lifecycle transitions
    - Provide queue visibility and statistics

    Uses SchedulingPolicy for scheduling decisions and QueueStoreProtocol
    for data access, enabling clean separation of concerns.
    """

    def __init__(
        self,
        queue_store: Optional[QueueStoreProtocol] = None,
        policy: Optional[SchedulingPolicy] = None,
        max_concurrent: int = 5,
        user_max_concurrent: int = 2,
    ):
        """Initialize the scheduler.

        Args:
            queue_store: Queue store instance (uses default if None)
            policy: Scheduling policy (uses default if None)
            max_concurrent: Global maximum running jobs
            user_max_concurrent: Maximum running jobs per user
        """
        if queue_store is None:
            from .queue_store import QueueStore

            queue_store = QueueStore()

        self._queue_store: QueueStoreProtocol = queue_store

        # Create policy with provided limits
        config = SchedulingConfig(
            max_concurrent=max_concurrent,
            user_max_concurrent=user_max_concurrent,
        )
        self._policy = policy or SchedulingPolicy(config)

        # Keep for backwards compatibility
        self._max_concurrent = max_concurrent
        self._user_max_concurrent = user_max_concurrent

        # Callback for dispatching jobs to providers
        self._dispatch_callback: Optional[Callable[[QueueEntry], Any]] = None

        # Background task for processing queue
        self._processing_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    @property
    def policy(self) -> SchedulingPolicy:
        """Get the scheduling policy."""
        return self._policy

    def set_dispatch_callback(self, callback: Callable[[QueueEntry], Any]) -> None:
        """Set the callback for dispatching jobs.

        The callback receives a QueueEntry and should submit it to the provider.
        It should return True on success, or raise an exception on failure.
        """
        self._dispatch_callback = callback

    async def enqueue_job(
        self,
        job_id: str,
        user_id: Optional[int] = None,
        team_id: Optional[str] = None,
        provider: str = "replicate",
        config_name: Optional[str] = None,
        user_levels: Optional[List[str]] = None,
        priority_override: Optional[int] = None,
        estimated_cost: float = 0.0,
        requires_approval: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QueueEntry:
        """Add a job to the queue.

        Args:
            job_id: The job ID
            user_id: User who submitted
            team_id: Team/department for fair-share scheduling
            provider: Target cloud provider
            config_name: Training config name
            user_levels: User's level names (for priority determination)
            priority_override: Lead-set priority override (bypasses level-based priority)
            estimated_cost: Estimated cost in USD
            requires_approval: Whether the job needs approval
            metadata: Additional metadata

        Returns:
            The created queue entry
        """
        # Determine priority from user levels
        priority = QueuePriority.NORMAL
        if user_levels:
            for level in user_levels:
                level_priority = get_priority_for_level(level)
                if level_priority.value > priority.value:
                    priority = level_priority

        entry = await self._queue_store.add_to_queue(
            job_id=job_id,
            user_id=user_id,
            team_id=team_id,
            provider=provider,
            config_name=config_name,
            priority=priority,
            priority_override=priority_override,
            estimated_cost=estimated_cost,
            requires_approval=requires_approval,
            metadata=metadata,
        )

        logger.info(
            "Enqueued job %s for user %s at priority %s (position %d)",
            job_id,
            user_id,
            priority.name,
            entry.position,
        )

        return entry

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job.

        Returns True if cancelled, False if not found or already running.
        """
        entry = await self._queue_store.get_entry_by_job_id(job_id)
        if not entry:
            return False

        if entry.status == QueueStatus.RUNNING:
            logger.warning("Cannot cancel running job %s via queue", job_id)
            return False

        success = await self._queue_store.mark_cancelled(entry.id)
        if success:
            logger.info("Cancelled queued job %s", job_id)
        return success

    async def get_queue_position(self, job_id: str) -> Optional[int]:
        """Get a job's current position in the queue."""
        entry = await self._queue_store.get_entry_by_job_id(job_id)
        if not entry or entry.status not in (QueueStatus.PENDING, QueueStatus.READY):
            return None
        return entry.position

    async def get_queue_position_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed queue position including estimated wait time.

        Args:
            job_id: The job ID to look up

        Returns:
            Dict with position, status, priority, queued_at, and estimated_wait_seconds,
            or None if job not found in queue.
        """
        entry = await self._queue_store.get_entry_by_job_id(job_id)
        if not entry:
            return None

        # Calculate estimated wait time if job is pending/ready
        estimated_wait_seconds = None
        if entry.status in (QueueStatus.PENDING, QueueStatus.READY, QueueStatus.BLOCKED):
            estimated_wait_seconds = await self._calculate_eta(entry.position)

        return {
            "job_id": job_id,
            "position": entry.position,
            "status": entry.status.value,
            "priority": entry.priority.name.lower(),
            "queued_at": entry.queued_at,
            "estimated_wait_seconds": estimated_wait_seconds,
        }

    async def _calculate_eta(self, position: int) -> Optional[float]:
        """Calculate estimated wait time for a given queue position.

        Uses historical average wait time and current throughput to estimate.

        Args:
            position: The job's position in the queue

        Returns:
            Estimated wait time in seconds, or None if no historical data
        """
        if position <= 0:
            return 0.0

        try:
            stats = await self._queue_store.get_queue_stats()
            avg_wait = stats.get("avg_wait_seconds")
            running = stats.get("running", 0)

            # Only calculate if we have historical data
            if avg_wait is not None and avg_wait > 0:
                effective_concurrency = max(running, 1)
                return (position * avg_wait) / effective_concurrency

            # No historical data available
            return None

        except Exception as exc:
            logger.debug("Failed to calculate ETA: %s", exc)
            return None

    async def get_user_queue_info(self, user_id: int) -> Dict[str, Any]:
        """Get queue information for a user.

        Returns summary of their queued, running, and completed jobs.
        """
        entries = await self._queue_store.list_entries(
            user_id=user_id,
            include_completed=True,
            limit=100,
        )

        pending = [e for e in entries if e.status in (QueueStatus.PENDING, QueueStatus.READY)]
        running = [e for e in entries if e.status == QueueStatus.RUNNING]
        blocked = [e for e in entries if e.status == QueueStatus.BLOCKED]

        # Get user's best position
        best_position = None
        for e in pending:
            if best_position is None or e.position < best_position:
                best_position = e.position

        return {
            "pending_count": len(pending),
            "running_count": len(running),
            "blocked_count": len(blocked),
            "best_position": best_position,
            "pending_jobs": [e.to_dict() for e in pending[:10]],
            "running_jobs": [e.to_dict() for e in running],
        }

    async def get_queue_overview(self) -> Dict[str, Any]:
        """Get overall queue status."""
        stats = await self._queue_store.get_queue_stats()

        # Get next jobs in line
        next_entries = await self._queue_store.list_entries(
            status=QueueStatus.PENDING,
            limit=10,
        )

        return {
            **stats,
            "max_concurrent": self._max_concurrent,
            "user_max_concurrent": self._user_max_concurrent,
            "team_max_concurrent": self._policy.config.team_max_concurrent,
            "enable_fair_share": self._policy.config.enable_fair_share,
            "next_in_line": [e.to_dict() for e in next_entries],
        }

    async def _get_next_ready(self) -> Optional[QueueEntry]:
        """Get the next job ready to run using pure data methods and policy.

        This replaces direct calls to queue_store.get_next_ready().
        """
        # Fetch scheduling data using pure data access methods
        running_count = await self._queue_store.count_running()
        user_running = await self._queue_store.count_running_by_user()
        pending_entries = await self._queue_store.list_pending_by_priority()

        # Fetch team running counts if fair-share is enabled
        team_running: Optional[Dict[str, int]] = None
        if self._policy.config.enable_fair_share:
            team_running = await self._queue_store.count_running_by_team()

        # Make scheduling decision using policy
        decision = self._policy.select_next(
            running_count=running_count,
            user_running=user_running,
            pending_entries=pending_entries,
            team_running=team_running,
        )

        if decision.entry is None:
            logger.debug("No job ready: %s", decision.reason)

        return decision.entry

    async def process_queue(self) -> Tuple[int, List[str]]:
        """Process the queue and dispatch ready jobs concurrently.

        Uses pure data access methods and SchedulingPolicy for decisions.

        Returns:
            Tuple of (jobs_dispatched, list of job_ids dispatched)
        """
        if not self._dispatch_callback:
            logger.warning("No dispatch callback set, cannot process queue")
            return 0, []

        # Collect all ready entries first
        entries_to_dispatch: List[QueueEntry] = []

        while True:
            entry = await self._get_next_ready()
            if not entry:
                break
            # Mark as running immediately to prevent re-selection
            await self._queue_store.mark_running(entry.id)
            entries_to_dispatch.append(entry)

        if not entries_to_dispatch:
            return 0, []

        async def dispatch_one(entry: QueueEntry) -> Tuple[str, Optional[Exception]]:
            """Dispatch a single job, returning (job_id, error or None)."""
            try:
                logger.info("Dispatching job %s to %s", entry.job_id, entry.provider)
                await self._dispatch_callback(entry)
                return entry.job_id, None
            except Exception as exc:
                logger.error("Failed to dispatch job %s: %s", entry.job_id, exc)
                return entry.job_id, exc

        # Dispatch all jobs concurrently
        results = await asyncio.gather(
            *[dispatch_one(e) for e in entries_to_dispatch],
            return_exceptions=False,
        )

        dispatched = []
        for entry, (job_id, exc) in zip(entries_to_dispatch, results):
            if exc is None:
                dispatched.append(job_id)
            else:
                # Mark as failed if out of retries, otherwise requeue
                if entry.attempt >= entry.max_attempts:
                    await self._queue_store.mark_failed(entry.id, str(exc))
                else:
                    await self._queue_store.update_entry(
                        entry.id,
                        {
                            "status": QueueStatus.PENDING.value,
                            "attempt": entry.attempt + 1,
                            "error_message": str(exc),
                        },
                    )

        if dispatched:
            logger.info("Dispatched %d jobs: %s", len(dispatched), dispatched)

        return len(dispatched), dispatched

    async def job_completed(self, job_id: str) -> bool:
        """Mark a job as completed in the queue."""
        entry = await self._queue_store.get_entry_by_job_id(job_id)
        if not entry:
            return False

        return await self._queue_store.mark_completed(entry.id)

    async def job_failed(self, job_id: str, error: str) -> bool:
        """Mark a job as failed in the queue."""
        entry = await self._queue_store.get_entry_by_job_id(job_id)
        if not entry:
            return False

        return await self._queue_store.mark_failed(entry.id, error)

    async def start_background_processing(self, interval_seconds: float = 5.0) -> None:
        """Start background queue processing.

        Args:
            interval_seconds: How often to check the queue
        """
        if self._processing_task is not None:
            logger.warning("Background processing already running")
            return

        self._stop_event.clear()

        async def process_loop():
            logger.info("Queue background processing started (interval: %.1fs)", interval_seconds)
            while not self._stop_event.is_set():
                try:
                    await self.process_queue()
                except Exception as exc:
                    logger.error("Error in queue processing: %s", exc, exc_info=True)

                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue loop

            logger.info("Queue background processing stopped")

        self._processing_task = asyncio.create_task(process_loop())

    async def stop_background_processing(self) -> None:
        """Stop background queue processing."""
        if self._processing_task is None:
            return

        self._stop_event.set()
        await self._processing_task
        self._processing_task = None

    async def set_concurrency_limits(
        self,
        max_concurrent: Optional[int] = None,
        user_max_concurrent: Optional[int] = None,
        team_max_concurrent: Optional[int] = None,
        enable_fair_share: Optional[bool] = None,
    ) -> None:
        """Update concurrency limits.

        Args:
            max_concurrent: Global maximum running jobs
            user_max_concurrent: Maximum running jobs per user
            team_max_concurrent: Maximum running jobs per team (fair-share)
            enable_fair_share: Enable/disable fair-share scheduling
        """
        if max_concurrent is not None:
            self._max_concurrent = max_concurrent
            self._policy.config.max_concurrent = max_concurrent
        if user_max_concurrent is not None:
            self._user_max_concurrent = user_max_concurrent
            self._policy.config.user_max_concurrent = user_max_concurrent
        if team_max_concurrent is not None:
            self._policy.config.team_max_concurrent = team_max_concurrent
        if enable_fair_share is not None:
            self._policy.config.enable_fair_share = enable_fair_share

        logger.info(
            "Updated concurrency limits: global=%d, per_user=%d, per_team=%d, fair_share=%s",
            self._max_concurrent,
            self._user_max_concurrent,
            self._policy.config.team_max_concurrent,
            self._policy.config.enable_fair_share,
        )

    async def approve_job(self, job_id: str, approval_id: int) -> bool:
        """Approve a blocked job for execution.

        Args:
            job_id: The job to approve
            approval_id: ID of the approval record

        Returns:
            True if approved, False if not found or not blocked
        """
        entry = await self._queue_store.get_entry_by_job_id(job_id)
        if not entry:
            return False

        if entry.status != QueueStatus.BLOCKED:
            logger.warning("Job %s is not blocked (status: %s)", job_id, entry.status)
            return False

        success = await self._queue_store.update_entry(
            entry.id,
            {
                "status": QueueStatus.PENDING.value,
                "requires_approval": 0,
                "approval_id": approval_id,
            },
        )

        if success:
            logger.info("Approved job %s (approval_id: %d)", job_id, approval_id)

        return success

    async def reject_job(self, job_id: str, reason: str) -> bool:
        """Reject a blocked job.

        Args:
            job_id: The job to reject
            reason: Reason for rejection

        Returns:
            True if rejected, False if not found or not blocked
        """
        entry = await self._queue_store.get_entry_by_job_id(job_id)
        if not entry:
            return False

        if entry.status != QueueStatus.BLOCKED:
            logger.warning("Job %s is not blocked (status: %s)", job_id, entry.status)
            return False

        return await self._queue_store.mark_failed(entry.id, f"Rejected: {reason}")

    async def set_priority(
        self,
        job_id: str,
        priority_override: Optional[int] = None,
    ) -> Optional[QueueEntry]:
        """Set priority override for a queued job.

        This allows leads/admins to bump or lower the priority of pending jobs.
        Setting priority_override to None removes the override and reverts to
        the user's level-based priority.

        Args:
            job_id: The job to update
            priority_override: New priority (0-50), or None to clear override

        Returns:
            Updated QueueEntry, or None if job not found or not modifiable
        """
        entry = await self._queue_store.get_entry_by_job_id(job_id)
        if not entry:
            logger.warning("Cannot set priority: job %s not found", job_id)
            return None

        # Only allow priority changes on pending/blocked jobs
        if entry.status not in (QueueStatus.PENDING, QueueStatus.READY, QueueStatus.BLOCKED):
            logger.warning(
                "Cannot set priority on job %s: status is %s (must be pending/blocked)",
                job_id,
                entry.status.value,
            )
            return None

        # Validate priority range if setting an override
        if priority_override is not None and not (0 <= priority_override <= 50):
            logger.warning("Invalid priority_override %d (must be 0-50)", priority_override)
            return None

        success = await self._queue_store.update_entry(
            entry.id,
            {"priority_override": priority_override},
        )

        if not success:
            return None

        # Fetch updated entry
        updated = await self._queue_store.get_entry_by_job_id(job_id)
        if updated:
            logger.info(
                "Set priority override for job %s: %s (effective: %d)",
                job_id,
                priority_override,
                updated.effective_priority,
            )
        return updated
