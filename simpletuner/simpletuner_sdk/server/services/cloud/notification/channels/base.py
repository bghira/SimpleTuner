"""Base class for notification channels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..models import ChannelConfig, DeliveryResult, NotificationEvent
from ..protocols import ChannelType


class BaseNotificationChannel(ABC):
    """Abstract base class for notification channels.

    All channel implementations should inherit from this class and implement
    the required methods.
    """

    channel_type: ChannelType

    def __init__(self, config: ChannelConfig):
        """Initialize the channel with configuration.

        Args:
            config: Channel configuration
        """
        self._config = config

    @property
    def config(self) -> ChannelConfig:
        """Get channel configuration."""
        return self._config

    @property
    def is_enabled(self) -> bool:
        """Check if channel is enabled."""
        return self._config.is_enabled

    @abstractmethod
    async def send(
        self,
        event: NotificationEvent,
        recipient: str,
        template_vars: Dict[str, Any],
    ) -> DeliveryResult:
        """Send a notification.

        Args:
            event: Notification event to send
            recipient: Recipient address (email, URL, etc.)
            template_vars: Variables for message templating

        Returns:
            DeliveryResult with success status
        """
        ...

    @abstractmethod
    async def validate_config(self, config: ChannelConfig) -> tuple[bool, Optional[str]]:
        """Validate channel configuration.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        ...

    @abstractmethod
    async def test_connection(self, config: ChannelConfig) -> tuple[bool, Optional[str], Optional[float]]:
        """Test channel connectivity.

        Args:
            config: Configuration to test

        Returns:
            Tuple of (success, error_message, latency_ms)
        """
        ...

    def _format_title(self, event: NotificationEvent) -> str:
        """Format notification title.

        Args:
            event: Notification event

        Returns:
            Formatted title
        """
        if event.title:
            return event.title

        # Generate title from event type
        event_labels = {
            "approval.required": "Approval Required",
            "approval.granted": "Approval Granted",
            "approval.rejected": "Approval Rejected",
            "approval.expired": "Approval Expired",
            "job.submitted": "Job Submitted",
            "job.started": "Job Started",
            "job.completed": "Job Completed",
            "job.failed": "Job Failed",
            "job.cancelled": "Job Cancelled",
            "quota.warning": "Quota Warning",
            "quota.exceeded": "Quota Exceeded",
            "system.provider_error": "Provider Error",
            "system.webhook_failure": "Webhook Failure",
            "auth.login_failure": "Login Failure",
        }
        return event_labels.get(event.event_type.value, "SimpleTuner Notification")

    def _format_message(self, event: NotificationEvent, template_vars: Dict[str, Any]) -> str:
        """Format notification message with variables.

        Args:
            event: Notification event
            template_vars: Template variables

        Returns:
            Formatted message
        """
        message = event.message

        # Simple variable substitution
        for key, value in template_vars.items():
            if value is not None:
                message = message.replace(f"{{{key}}}", str(value))

        return message
