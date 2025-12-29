"""Protocol definitions for notification system.

This module defines the interfaces for notification channels and response handlers,
following the existing Protocol pattern in services/cloud/protocols.py.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .models import ChannelConfig, DeliveryResult, NotificationEvent, ResponseAction


class NotificationEventType(str, Enum):
    """All notification event types supported by the system."""

    # Approval workflow events
    APPROVAL_REQUIRED = "approval.required"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_REJECTED = "approval.rejected"
    APPROVAL_EXPIRED = "approval.expired"

    # Job lifecycle events
    JOB_SUBMITTED = "job.submitted"
    JOB_QUEUED = "job.queued"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"

    # Quota and billing events
    QUOTA_WARNING = "quota.warning"
    QUOTA_EXCEEDED = "quota.exceeded"
    COST_LIMIT_WARNING = "cost.warning"
    COST_LIMIT_EXCEEDED = "cost.exceeded"

    # System health events
    PROVIDER_ERROR = "system.provider_error"
    PROVIDER_DEGRADED = "system.provider_degraded"
    CONNECTION_RESTORED = "system.connection_restored"
    WEBHOOK_FAILURE = "system.webhook_failure"

    # Authentication events
    AUTH_LOGIN_FAILURE = "auth.login_failure"
    AUTH_SESSION_EXPIRY = "auth.session_expiry"
    AUTH_PASSWORD_RESET = "auth.password_reset"
    AUTH_NEW_DEVICE = "auth.new_device"

    # Administrative events
    ADMIN_USER_CREATED = "admin.user_created"
    ADMIN_ROLE_CHANGED = "admin.role_changed"
    ADMIN_CONFIG_CHANGED = "admin.config_changed"

    @classmethod
    def approval_events(cls) -> List["NotificationEventType"]:
        """Return all approval-related event types."""
        return [
            cls.APPROVAL_REQUIRED,
            cls.APPROVAL_GRANTED,
            cls.APPROVAL_REJECTED,
            cls.APPROVAL_EXPIRED,
        ]

    @classmethod
    def job_events(cls) -> List["NotificationEventType"]:
        """Return all job lifecycle event types."""
        return [
            cls.JOB_SUBMITTED,
            cls.JOB_QUEUED,
            cls.JOB_STARTED,
            cls.JOB_COMPLETED,
            cls.JOB_FAILED,
            cls.JOB_CANCELLED,
        ]

    @classmethod
    def quota_events(cls) -> List["NotificationEventType"]:
        """Return all quota-related event types."""
        return [
            cls.QUOTA_WARNING,
            cls.QUOTA_EXCEEDED,
            cls.COST_LIMIT_WARNING,
            cls.COST_LIMIT_EXCEEDED,
        ]

    @classmethod
    def system_events(cls) -> List["NotificationEventType"]:
        """Return all system health event types."""
        return [
            cls.PROVIDER_ERROR,
            cls.PROVIDER_DEGRADED,
            cls.CONNECTION_RESTORED,
            cls.WEBHOOK_FAILURE,
        ]

    @classmethod
    def auth_events(cls) -> List["NotificationEventType"]:
        """Return all authentication event types."""
        return [
            cls.AUTH_LOGIN_FAILURE,
            cls.AUTH_SESSION_EXPIRY,
            cls.AUTH_PASSWORD_RESET,
            cls.AUTH_NEW_DEVICE,
        ]

    @classmethod
    def admin_events(cls) -> List["NotificationEventType"]:
        """Return all administrative event types."""
        return [
            cls.ADMIN_USER_CREATED,
            cls.ADMIN_ROLE_CHANGED,
            cls.ADMIN_CONFIG_CHANGED,
        ]


class ChannelType(str, Enum):
    """Supported notification channel types."""

    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    # Future expansion
    # SMS = "sms"
    # DISCORD = "discord"
    # PAGERDUTY = "pagerduty"


class DeliveryStatus(str, Enum):
    """Notification delivery status."""

    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"

    @classmethod
    def terminal_statuses(cls) -> List["DeliveryStatus"]:
        """Return statuses that indicate delivery is complete (success or failure)."""
        return [cls.DELIVERED, cls.FAILED, cls.BOUNCED]


class Severity(str, Enum):
    """Notification severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @classmethod
    def severity_order(cls) -> Dict[str, int]:
        """Return severity ordering for threshold comparison."""
        return {
            cls.DEBUG.value: 0,
            cls.INFO.value: 1,
            cls.WARNING.value: 2,
            cls.ERROR.value: 3,
            cls.CRITICAL.value: 4,
        }

    @classmethod
    def meets_threshold(cls, severity: str, threshold: str) -> bool:
        """Check if severity meets or exceeds threshold."""
        order = cls.severity_order()
        return order.get(severity, 1) >= order.get(threshold, 1)


@runtime_checkable
class NotificationChannelProtocol(Protocol):
    """Protocol for notification delivery channels.

    All channel implementations must satisfy this interface.
    """

    channel_type: ChannelType

    async def send(
        self,
        event: "NotificationEvent",
        recipient: str,
        template_vars: Dict[str, Any],
    ) -> "DeliveryResult":
        """Send a notification to a recipient.

        Args:
            event: The notification event to send
            recipient: Channel-specific recipient (email address, webhook URL, etc.)
            template_vars: Variables for message templating

        Returns:
            DeliveryResult with success status and details
        """
        ...

    async def validate_config(self, config: "ChannelConfig") -> tuple[bool, Optional[str]]:
        """Validate channel configuration.

        Args:
            config: Channel configuration to validate

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        ...

    async def test_connection(self, config: "ChannelConfig") -> tuple[bool, Optional[str], Optional[float]]:
        """Test channel connectivity.

        Args:
            config: Channel configuration to test

        Returns:
            Tuple of (success, error_message, latency_ms)
        """
        ...


@runtime_checkable
class ResponseHandlerProtocol(Protocol):
    """Protocol for handling responses to notifications.

    Used primarily for email reply processing (IMAP IDLE) to enable
    mobile approval workflows.
    """

    async def start(self) -> None:
        """Start listening for responses.

        This may start a background task (e.g., IMAP IDLE loop).
        """
        ...

    async def stop(self) -> None:
        """Stop listening for responses.

        Clean up any background tasks and connections.
        """
        ...

    @property
    def is_running(self) -> bool:
        """Check if the handler is currently running."""
        ...

    async def process_response(
        self,
        raw_response: Any,
        context: Dict[str, Any],
    ) -> Optional["ResponseAction"]:
        """Parse a raw response and determine the action.

        Args:
            raw_response: Raw response data (e.g., email.message.Message)
            context: Additional context for processing

        Returns:
            ResponseAction if the response was understood, None otherwise
        """
        ...


@runtime_checkable
class NotificationServiceProtocol(Protocol):
    """Protocol for the main notification service.

    Provides the public API for sending notifications and managing channels.
    """

    async def notify(
        self,
        event_type: NotificationEventType,
        context: Dict[str, Any],
        recipients: Optional[List[str]] = None,
    ) -> List["DeliveryResult"]:
        """Send notifications for an event.

        Args:
            event_type: Type of notification event
            context: Event context (job_id, user_id, message, etc.)
            recipients: Optional explicit recipients (overrides preferences)

        Returns:
            List of delivery results for each channel/recipient
        """
        ...

    async def notify_approval_required(
        self,
        approval_request_id: int,
        job_id: str,
        approvers: List[str],
        reason: str,
    ) -> List["DeliveryResult"]:
        """Send approval request notifications with response tokens.

        This is a specialized method for approval workflow that generates
        response tokens for email-based approval/rejection.

        Args:
            approval_request_id: ID of the approval request
            job_id: Associated job ID
            approvers: List of approver email addresses
            reason: Reason approval is required

        Returns:
            List of delivery results
        """
        ...

    async def handle_response(self, action: "ResponseAction") -> bool:
        """Process an approval response.

        Args:
            action: Parsed response action from email/webhook

        Returns:
            True if response was processed successfully
        """
        ...

    async def get_channels(self, enabled_only: bool = True) -> List["ChannelConfig"]:
        """List configured notification channels.

        Args:
            enabled_only: If True, only return enabled channels

        Returns:
            List of channel configurations
        """
        ...

    async def configure_channel(
        self,
        channel_type: ChannelType,
        name: str,
        config: Dict[str, Any],
    ) -> int:
        """Configure a notification channel.

        Args:
            channel_type: Type of channel to configure
            name: Display name for the channel
            config: Channel-specific configuration

        Returns:
            ID of the created/updated channel
        """
        ...
