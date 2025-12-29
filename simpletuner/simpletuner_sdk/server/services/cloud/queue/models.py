"""Queue data models for cloud training."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class QueueStatus(str, Enum):
    """Status of a queue entry."""

    PENDING = "pending"  # Waiting to be scheduled
    READY = "ready"  # Ready to run (passed quota/approval checks)
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"  # Finished with error
    CANCELLED = "cancelled"  # Cancelled by user
    BLOCKED = "blocked"  # Blocked by quota/approval


class QueuePriority(int, Enum):
    """Priority levels for queue entries.

    Higher values = higher priority = processed first.
    """

    LOW = 0  # Background tasks
    NORMAL = 10  # Standard researcher priority
    HIGH = 20  # Lead researcher priority
    URGENT = 30  # Admin priority
    CRITICAL = 40  # System-initiated, bypass most limits


@dataclass
class QueueEntry:
    """A job in the queue.

    Represents a pending or active job with its scheduling metadata.
    """

    id: int
    job_id: str  # Links to UnifiedJob
    user_id: Optional[int]  # User who submitted
    team_id: Optional[str] = None  # Team/department for fair-share scheduling
    provider: str = "replicate"  # Target provider (replicate, simpletuner_io, etc.)
    config_name: Optional[str] = None  # Config being trained

    # Scheduling metadata
    priority: QueuePriority = QueuePriority.NORMAL
    priority_override: Optional[int] = None  # Lead-set priority override
    status: QueueStatus = QueueStatus.PENDING
    position: int = 0  # Current position in queue (0 = running or next)

    # Timestamps
    queued_at: str = ""  # When added to queue (ISO format)
    started_at: Optional[str] = None  # When execution began
    completed_at: Optional[str] = None  # When finished

    # Quota/approval tracking
    estimated_cost: float = 0.0  # Estimated USD cost
    requires_approval: bool = False
    approval_id: Optional[int] = None  # Link to approval request

    # Retry handling
    attempt: int = 1
    max_attempts: int = 3
    error_message: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_priority(self) -> int:
        """Get the effective priority (override if set, else level-based)."""
        if self.priority_override is not None:
            return self.priority_override
        return self.priority.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "user_id": self.user_id,
            "team_id": self.team_id,
            "provider": self.provider,
            "config_name": self.config_name,
            "priority": self.priority.value,
            "priority_name": self.priority.name.lower(),
            "priority_override": self.priority_override,
            "effective_priority": self.effective_priority,
            "status": self.status.value,
            "position": self.position,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "estimated_cost": self.estimated_cost,
            "requires_approval": self.requires_approval,
            "approval_id": self.approval_id,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueueEntry":
        """Create from dictionary."""
        priority = data.get("priority", QueuePriority.NORMAL.value)
        if isinstance(priority, int):
            priority = QueuePriority(priority)
        elif isinstance(priority, str):
            priority = QueuePriority[priority.upper()]

        status = data.get("status", QueueStatus.PENDING.value)
        if isinstance(status, str):
            status = QueueStatus(status)

        return cls(
            id=data.get("id", 0),
            job_id=data["job_id"],
            user_id=data.get("user_id"),
            team_id=data.get("team_id"),
            provider=data.get("provider", "replicate"),
            config_name=data.get("config_name"),
            priority=priority,
            priority_override=data.get("priority_override"),
            status=status,
            position=data.get("position", 0),
            queued_at=data.get("queued_at", ""),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            estimated_cost=data.get("estimated_cost", 0.0),
            requires_approval=data.get("requires_approval", False),
            approval_id=data.get("approval_id"),
            attempt=data.get("attempt", 1),
            max_attempts=data.get("max_attempts", 3),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


# Default priority mapping from user level
LEVEL_PRIORITY_MAP = {
    "admin": QueuePriority.URGENT,
    "lead": QueuePriority.HIGH,
    "researcher": QueuePriority.NORMAL,
    "viewer": QueuePriority.LOW,
}


def get_priority_for_level(level_name: str) -> QueuePriority:
    """Get the default priority for a user level."""
    return LEVEL_PRIORITY_MAP.get(level_name.lower(), QueuePriority.NORMAL)
