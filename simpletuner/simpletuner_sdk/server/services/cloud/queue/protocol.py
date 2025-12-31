"""Protocol definitions for queue store abstraction.

Enables type-safe polymorphism between QueueStore and AsyncQueueStore.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .models import QueueEntry, QueuePriority, QueueStatus


@runtime_checkable
class QueueStoreProtocol(Protocol):
    """Protocol that both QueueStore and AsyncQueueStore must implement.

    Defines the interface for queue data access, enabling the scheduler
    to work with either store implementation.
    """

    # --- Core CRUD Operations ---

    async def add_to_queue(
        self,
        job_id: str,
        user_id: Optional[int] = None,
        team_id: Optional[str] = None,
        provider: str = "replicate",
        config_name: Optional[str] = None,
        priority: QueuePriority = QueuePriority.NORMAL,
        priority_override: Optional[int] = None,
        estimated_cost: float = 0.0,
        requires_approval: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QueueEntry:
        """Add a job to the queue."""
        ...

    async def get_entry(self, queue_id: int) -> Optional[QueueEntry]:
        """Get a queue entry by ID."""
        ...

    async def get_entry_by_job_id(self, job_id: str) -> Optional[QueueEntry]:
        """Get a queue entry by job ID."""
        ...

    async def update_entry(self, queue_id: int, updates: Dict[str, Any]) -> bool:
        """Update a queue entry."""
        ...

    async def remove_entry(self, queue_id: int) -> bool:
        """Remove an entry from the queue."""
        ...

    async def list_entries(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[QueueStatus] = None,
        user_id: Optional[int] = None,
        include_completed: bool = False,
    ) -> List[QueueEntry]:
        """List queue entries."""
        ...

    # --- Pure Data Access Methods (for scheduling) ---

    async def count_running(self) -> int:
        """Count total running jobs."""
        ...

    async def count_running_by_user(self) -> Dict[int, int]:
        """Get running job count per user."""
        ...

    async def list_pending_by_priority(self, limit: int = 50) -> List[QueueEntry]:
        """List pending entries ordered by priority (desc) and age (asc)."""
        ...

    async def get_positions_batch(self, job_ids: List[str]) -> Dict[str, int]:
        """Get queue positions for multiple jobs."""
        ...

    # --- Status Transitions ---

    async def mark_running(self, queue_id: int) -> bool:
        """Mark an entry as running."""
        ...

    async def mark_completed(self, queue_id: int) -> bool:
        """Mark an entry as completed."""
        ...

    async def mark_failed(self, queue_id: int, error: str) -> bool:
        """Mark an entry as failed."""
        ...

    async def mark_cancelled(self, queue_id: int) -> bool:
        """Mark an entry as cancelled."""
        ...

    async def count_running_by_team(self) -> Dict[str, int]:
        """Get running job count per team.

        Returns:
            Dict mapping team_id -> running count
        """
        ...

    # --- Statistics ---

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        ...

    async def get_user_position(self, user_id: int) -> Optional[int]:
        """Get a user's position in the queue."""
        ...

    async def cleanup_old_entries(self, days: int = 30) -> int:
        """Remove old completed/failed entries."""
        ...
