"""Job queue management for cloud training.

Provides queuing, priority, fair scheduling, and concurrency management.
"""

from .dispatcher import QueueDispatcher, get_dispatcher, reset_dispatcher
from .models import QueueEntry, QueuePriority, QueueStatus
from .protocol import QueueStoreProtocol
from .queue_store import QueueStore
from .scheduler import QueueScheduler, SchedulingConfig, SchedulingDecision, SchedulingPolicy

# Backwards compatibility alias - QueueStore already has async methods
AsyncQueueStore = QueueStore

__all__ = [
    # Models
    "QueueEntry",
    "QueueStatus",
    "QueuePriority",
    # Protocol
    "QueueStoreProtocol",
    # Store
    "QueueStore",
    "AsyncQueueStore",
    # Scheduler
    "QueueScheduler",
    "SchedulingPolicy",
    "SchedulingConfig",
    "SchedulingDecision",
    # Dispatcher
    "QueueDispatcher",
    "get_dispatcher",
    "reset_dispatcher",
]
