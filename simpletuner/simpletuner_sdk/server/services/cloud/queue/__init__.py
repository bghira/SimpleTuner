"""Job queue management for cloud training.

Provides queuing, priority, fair scheduling, and concurrency management.
"""

from .dispatcher import QueueDispatcher, get_dispatcher, reset_dispatcher
from .job_repo_adapter import JobRepoQueueAdapter, get_queue_adapter
from .models import QueueEntry, QueuePriority, QueueStatus
from .protocol import QueueStoreProtocol
from .scheduler import QueueScheduler, SchedulingConfig, SchedulingDecision, SchedulingPolicy

# Backwards compatibility aliases - use JobRepoQueueAdapter
QueueStore = JobRepoQueueAdapter
AsyncQueueStore = JobRepoQueueAdapter

__all__ = [
    # Models
    "QueueEntry",
    "QueueStatus",
    "QueuePriority",
    # Protocol
    "QueueStoreProtocol",
    # Store (backwards compat aliases)
    "QueueStore",
    "AsyncQueueStore",
    "JobRepoQueueAdapter",
    "get_queue_adapter",
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
