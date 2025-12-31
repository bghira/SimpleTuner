"""Command pattern implementation for job operations.

Commands are first-class objects that encapsulate all information needed
to perform an action. This enables:
- Audit trails (commands logged before/after execution)
- Undo operations (commands can implement rollback)
- Queuing (commands can be serialized)
- Retry logic (commands can be retried on failure)
- Testing (commands tested in isolation)
"""

from .base import Command, CommandContext, CommandDispatcher, CommandResult, CommandStatus, get_dispatcher
from .job_commands import CancelJobCommand, DeleteJobCommand, SubmitJobCommand, SyncJobsCommand

__all__ = [
    # Base
    "Command",
    "CommandContext",
    "CommandDispatcher",
    "CommandResult",
    "CommandStatus",
    "get_dispatcher",
    # Job commands
    "CancelJobCommand",
    "DeleteJobCommand",
    "SubmitJobCommand",
    "SyncJobsCommand",
]
