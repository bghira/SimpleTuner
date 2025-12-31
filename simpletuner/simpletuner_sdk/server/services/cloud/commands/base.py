"""Base command pattern infrastructure.

Provides the foundation for implementing commands as first-class objects.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CommandStatus(str, Enum):
    """Status of command execution."""

    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class CommandResult(Generic[T]):
    """Result of command execution."""

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Execution metadata
    command_id: Optional[str] = None
    executed_at: Optional[str] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "error_code": self.error_code,
            "warnings": self.warnings,
            "command_id": self.command_id,
            "executed_at": self.executed_at,
            "duration_ms": self.duration_ms,
        }


@dataclass
class CommandContext:
    """Context passed to command execution.

    Contains all dependencies and contextual information needed
    for command execution, avoiding tight coupling to global state.
    """

    # User context
    user_id: Optional[int] = None
    user_permissions: List[str] = field(default_factory=list)
    client_ip: Optional[str] = None

    # Request context
    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None

    # Service dependencies (injected)
    job_store: Optional[Any] = None
    user_store: Optional[Any] = None
    queue_store: Optional[Any] = None

    # Audit callback
    audit_callback: Optional[Callable[..., None]] = None

    def has_permission(self, permission: str) -> bool:
        """Check if context has a specific permission."""
        return permission in self.user_permissions

    def log_audit(
        self,
        action: str,
        job_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Log an audit event if callback is configured."""
        if self.audit_callback:
            self.audit_callback(
                action=action,
                job_id=job_id,
                user_id=str(self.user_id) if self.user_id else None,
                user_ip=self.client_ip,
                details=details,
                **kwargs,
            )


class Command(ABC, Generic[T]):
    """Abstract base class for all commands.

    Commands encapsulate all information needed to perform an action.
    They are immutable after creation and can be logged, queued, or retried.
    """

    def __init__(self):
        self._id = str(uuid.uuid4())
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._status = CommandStatus.PENDING

    @property
    def command_id(self) -> str:
        """Unique identifier for this command instance."""
        return self._id

    @property
    def command_type(self) -> str:
        """Type name of the command."""
        return self.__class__.__name__

    @property
    def created_at(self) -> str:
        """When the command was created."""
        return self._created_at

    @property
    def status(self) -> CommandStatus:
        """Current status of the command."""
        return self._status

    @abstractmethod
    async def execute(self, ctx: CommandContext) -> CommandResult[T]:
        """Execute the command.

        Args:
            ctx: Execution context with dependencies and user info

        Returns:
            CommandResult with success/failure and data
        """
        pass

    async def validate(self, ctx: CommandContext) -> Optional[str]:
        """Validate the command before execution.

        Override to add validation logic. Return error message if invalid.

        Args:
            ctx: Execution context

        Returns:
            Error message if invalid, None if valid
        """
        return None

    async def rollback(self, ctx: CommandContext) -> bool:
        """Rollback the command if supported.

        Override to implement undo functionality.

        Args:
            ctx: Execution context

        Returns:
            True if rollback succeeded
        """
        return False

    @property
    def is_reversible(self) -> bool:
        """Whether this command supports rollback."""
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize command to dictionary for logging/queuing."""
        return {
            "command_id": self._id,
            "command_type": self.command_type,
            "created_at": self._created_at,
            "status": self._status.value,
        }


class CommandDispatcher:
    """Dispatches and executes commands with cross-cutting concerns.

    Handles:
    - Command validation
    - Pre/post execution hooks
    - Audit logging
    - Error handling
    - Metrics collection
    """

    def __init__(self):
        self._pre_execute_hooks: List[Callable[[Command, CommandContext], None]] = []
        self._post_execute_hooks: List[Callable[[Command, CommandContext, CommandResult], None]] = []
        self._error_handlers: Dict[Type[Exception], Callable[[Exception, Command, CommandContext], CommandResult]] = {}

    def add_pre_execute_hook(self, hook: Callable[[Command, CommandContext], None]) -> None:
        """Add a hook to run before command execution."""
        self._pre_execute_hooks.append(hook)

    def add_post_execute_hook(self, hook: Callable[[Command, CommandContext, CommandResult], None]) -> None:
        """Add a hook to run after command execution."""
        self._post_execute_hooks.append(hook)

    def register_error_handler(
        self,
        exception_type: Type[Exception],
        handler: Callable[[Exception, Command, CommandContext], CommandResult],
    ) -> None:
        """Register a handler for a specific exception type."""
        self._error_handlers[exception_type] = handler

    async def dispatch(self, command: Command[T], ctx: CommandContext) -> CommandResult[T]:
        """Execute a command with all cross-cutting concerns.

        Args:
            command: The command to execute
            ctx: Execution context

        Returns:
            CommandResult from command execution
        """
        import time

        start_time = time.time()
        command._status = CommandStatus.EXECUTING

        # Validate
        validation_error = await command.validate(ctx)
        if validation_error:
            command._status = CommandStatus.FAILED
            return CommandResult(
                success=False,
                error=validation_error,
                error_code="VALIDATION_ERROR",
                command_id=command.command_id,
            )

        # Pre-execute hooks
        for hook in self._pre_execute_hooks:
            try:
                hook(command, ctx)
            except Exception as exc:
                logger.warning("Pre-execute hook failed: %s", exc)

        # Execute
        try:
            result = await command.execute(ctx)
            command._status = CommandStatus.SUCCESS if result.success else CommandStatus.FAILED

        except Exception as exc:
            command._status = CommandStatus.FAILED

            # Check for registered error handlers
            for exc_type, handler in self._error_handlers.items():
                if isinstance(exc, exc_type):
                    result = handler(exc, command, ctx)
                    break
            else:
                logger.error(
                    "Command %s failed: %s",
                    command.command_type,
                    exc,
                    exc_info=True,
                )
                result = CommandResult(
                    success=False,
                    error=str(exc),
                    error_code="EXECUTION_ERROR",
                    command_id=command.command_id,
                )

        # Add execution metadata
        duration_ms = (time.time() - start_time) * 1000
        result.command_id = command.command_id
        result.executed_at = datetime.now(timezone.utc).isoformat()
        result.duration_ms = round(duration_ms, 2)

        # Post-execute hooks
        for hook in self._post_execute_hooks:
            try:
                hook(command, ctx, result)
            except Exception as exc:
                logger.warning("Post-execute hook failed: %s", exc)

        return result


# Global dispatcher instance (can be replaced for testing)
_dispatcher: Optional[CommandDispatcher] = None


def get_dispatcher() -> CommandDispatcher:
    """Get the global command dispatcher."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = CommandDispatcher()
        _setup_default_hooks(_dispatcher)
    return _dispatcher


def set_dispatcher(dispatcher: CommandDispatcher) -> None:
    """Set the global command dispatcher (for testing)."""
    global _dispatcher
    _dispatcher = dispatcher


def _setup_default_hooks(dispatcher: CommandDispatcher) -> None:
    """Configure default hooks for the dispatcher."""

    def audit_pre_hook(command: Command, ctx: CommandContext) -> None:
        """Log command start."""
        logger.debug(
            "Executing command %s (id=%s, user=%s)",
            command.command_type,
            command.command_id,
            ctx.user_id,
        )

    def audit_post_hook(command: Command, ctx: CommandContext, result: CommandResult) -> None:
        """Log command completion."""
        if result.success:
            logger.info(
                "Command %s succeeded (id=%s, duration=%.2fms)",
                command.command_type,
                command.command_id,
                result.duration_ms or 0,
            )
        else:
            logger.warning(
                "Command %s failed (id=%s, error=%s)",
                command.command_type,
                command.command_id,
                result.error,
            )

    dispatcher.add_pre_execute_hook(audit_pre_hook)
    dispatcher.add_post_execute_hook(audit_post_hook)
