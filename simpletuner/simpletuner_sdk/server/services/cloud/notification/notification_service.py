"""Core notification service orchestrating channels and delivery."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from .models import ChannelConfig, DeliveryResult, NotificationEvent, NotificationPreference, ResponseAction
from .notification_router import NotificationRouter
from .protocols import (
    ChannelType,
    DeliveryStatus,
    NotificationChannelProtocol,
    NotificationEventType,
    ResponseHandlerProtocol,
)

if TYPE_CHECKING:
    from .notification_store import NotificationStore

logger = logging.getLogger(__name__)


class NotificationService:
    """Main notification service registered in ServiceContainer.

    Responsibilities:
    - Route events to appropriate channels based on preferences
    - Manage channel lifecycle (create, test, destroy)
    - Track delivery status
    - Handle response callbacks from IMAP/webhooks

    Access patterns:
    - FastAPI routes: Depends(get_notification_service)
    - Non-FastAPI code: get_notifier() module function
    """

    def __init__(self, store: "NotificationStore"):
        """Initialize the notification service.

        Args:
            store: NotificationStore for persistence
        """
        self._store = store
        self._router = NotificationRouter(store)
        self._channels: Dict[int, NotificationChannelProtocol] = {}
        self._channel_classes: Dict[ChannelType, Type[NotificationChannelProtocol]] = {}
        self._response_handlers: List[ResponseHandlerProtocol] = []
        self._initialized = False
        self._lock = asyncio.Lock()
        # Register channel classes immediately (sync operation)
        self._register_channel_classes()

    async def initialize(self) -> None:
        """Load channels and start response handlers.

        Call this during application startup to load existing channels
        and start any IMAP listeners.
        """
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            # Load existing channels
            channels = await self._store.list_channels(enabled_only=True)
            for channel_config in channels:
                await self._instantiate_channel(channel_config)

            # Start response handlers for email channels with IMAP enabled
            for channel_config in channels:
                if channel_config.channel_type == ChannelType.EMAIL and channel_config.imap_enabled:
                    await self._start_response_handler(channel_config)

            self._initialized = True
            logger.info(
                "NotificationService initialized with %d channels, %d response handlers",
                len(self._channels),
                len(self._response_handlers),
            )

    def _register_channel_classes(self) -> None:
        """Register available channel implementations."""
        # Defer imports to avoid circular dependencies
        try:
            from .channels.email_channel import EmailChannel

            self._channel_classes[ChannelType.EMAIL] = EmailChannel
        except ImportError:
            logger.debug("EmailChannel not available")

        try:
            from .channels.webhook_channel import WebhookChannel

            self._channel_classes[ChannelType.WEBHOOK] = WebhookChannel
        except ImportError:
            logger.debug("WebhookChannel not available")

        try:
            from .channels.slack_channel import SlackChannel

            self._channel_classes[ChannelType.SLACK] = SlackChannel
        except ImportError:
            logger.debug("SlackChannel not available")

    async def _instantiate_channel(self, config: ChannelConfig) -> Optional[NotificationChannelProtocol]:
        """Create a channel instance from configuration.

        Args:
            config: Channel configuration

        Returns:
            Channel instance or None if not available
        """
        channel_class = self._channel_classes.get(config.channel_type)
        if not channel_class:
            logger.warning("No implementation for channel type: %s", config.channel_type)
            return None

        try:
            channel = channel_class(config)
            self._channels[config.id] = channel
            return channel
        except Exception as exc:
            logger.error(
                "Failed to instantiate channel %d (%s): %s",
                config.id,
                config.name,
                exc,
            )
            return None

    async def _start_response_handler(self, config: ChannelConfig) -> None:
        """Start a response handler for an email channel.

        Args:
            config: Email channel configuration with IMAP enabled
        """
        try:
            from .response_handlers.imap_handler import IMAPResponseHandler

            handler = IMAPResponseHandler(config, self)
            await handler.start()
            self._response_handlers.append(handler)
            logger.info("Started IMAP response handler for channel %d", config.id)
        except ImportError:
            logger.debug("IMAPResponseHandler not available")
        except Exception as exc:
            logger.error("Failed to start response handler for channel %d: %s", config.id, exc)

    async def shutdown(self) -> None:
        """Stop all response handlers and clean up."""
        for handler in self._response_handlers:
            try:
                await handler.stop()
            except Exception as exc:
                logger.warning("Error stopping response handler: %s", exc)
        self._response_handlers.clear()
        self._channels.clear()
        self._initialized = False

    # --- Public API ---

    async def notify(
        self,
        event_type: NotificationEventType,
        context: Dict[str, Any],
        recipients: Optional[List[str]] = None,
    ) -> List[DeliveryResult]:
        """Send notifications for an event.

        Args:
            event_type: Type of notification event
            context: Event context (job_id, user_id, message, etc.)
            recipients: Optional explicit recipients (overrides preferences)

        Returns:
            List of delivery results for each channel/recipient
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        # Create event from context
        event = NotificationEvent.from_context(event_type, context)

        # Get routes
        routes = await self._router.get_routes(event)
        if not routes:
            logger.debug("No routes found for event: %s", event_type.value)
            return []

        # Send to each channel
        results: List[DeliveryResult] = []
        tasks = []

        for channel_config, route_recipients in routes:
            # Override recipients if specified
            final_recipients = recipients if recipients else route_recipients

            channel = self._channels.get(channel_config.id)
            if not channel:
                channel = await self._instantiate_channel(channel_config)
                if not channel:
                    continue

            # Create send tasks for each recipient
            for recipient in final_recipients:
                task = asyncio.create_task(self._send_with_logging(channel, channel_config, event, recipient))
                tasks.append(task)

        # Wait for all sends to complete
        if tasks:
            send_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in send_results:
                if isinstance(result, DeliveryResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.error("Notification send failed: %s", result)

        return results

    async def _send_with_logging(
        self,
        channel: NotificationChannelProtocol,
        config: ChannelConfig,
        event: NotificationEvent,
        recipient: str,
    ) -> DeliveryResult:
        """Send a notification and log the result.

        Args:
            channel: Channel instance
            config: Channel configuration
            event: Notification event
            recipient: Recipient address

        Returns:
            Delivery result
        """
        start_time = time.monotonic()

        try:
            result = await channel.send(
                event=event,
                recipient=recipient,
                template_vars=self._build_template_vars(event),
            )
            result.latency_ms = (time.monotonic() - start_time) * 1000

            # Log delivery
            await self._store.log_delivery(
                channel_id=config.id,
                event_type=event.event_type,
                recipient=recipient,
                status=result.delivery_status,
                job_id=event.job_id,
                user_id=event.user_id,
                error_message=result.error_message,
                provider_message_id=result.provider_message_id,
            )

            return result

        except Exception as exc:
            latency = (time.monotonic() - start_time) * 1000
            error_msg = str(exc)

            # Log failure
            await self._store.log_delivery(
                channel_id=config.id,
                event_type=event.event_type,
                recipient=recipient,
                status=DeliveryStatus.FAILED,
                job_id=event.job_id,
                user_id=event.user_id,
                error_message=error_msg,
            )

            return DeliveryResult(
                success=False,
                channel_id=config.id,
                channel_type=config.channel_type,
                recipient=recipient,
                event_type=event.event_type,
                delivery_status=DeliveryStatus.FAILED,
                error_message=error_msg,
                latency_ms=latency,
            )

    def _build_template_vars(self, event: NotificationEvent) -> Dict[str, Any]:
        """Build template variables for message rendering.

        Args:
            event: Notification event

        Returns:
            Template variables dict
        """
        return {
            "event_type": event.event_type.value,
            "title": event.title,
            "message": event.message,
            "severity": event.severity,
            "job_id": event.job_id,
            "user_id": event.user_id,
            "timestamp": event.created_at.isoformat(),
            "approval_request_id": event.approval_request_id,
            "response_token": event.response_token,
            **event.data,
        }

    async def notify_approval_required(
        self,
        approval_request_id: int,
        job_id: str,
        approvers: List[str],
        reason: str,
        estimated_cost: Optional[float] = None,
        config_name: Optional[str] = None,
    ) -> List[DeliveryResult]:
        """Send approval request notifications with response tokens.

        This is a specialized method for approval workflow that generates
        response tokens for email-based approval/rejection.

        Args:
            approval_request_id: ID of the approval request
            job_id: Associated job ID
            approvers: List of approver email addresses
            reason: Reason approval is required
            estimated_cost: Estimated job cost
            config_name: Training configuration name

        Returns:
            List of delivery results
        """
        event = NotificationEvent(
            event_type=NotificationEventType.APPROVAL_REQUIRED,
            title="Approval Required",
            message=reason,
            severity="warning",
            job_id=job_id,
            approval_request_id=approval_request_id,
            data={
                "estimated_cost": estimated_cost,
                "config_name": config_name,
                "approvers": approvers,
            },
        )

        # Generate response token for email replies
        event = event.with_response_token()

        # Store pending response for correlation
        email_channels = await self._store.list_channels(enabled_only=True, channel_type=ChannelType.EMAIL)
        for channel in email_channels:
            if channel.imap_enabled:
                await self._store.create_pending_response(
                    response_token=event.response_token,
                    approval_request_id=approval_request_id,
                    channel_id=channel.id,
                    authorized_senders=approvers,
                )

        # Send notifications
        return await self.notify(
            event_type=NotificationEventType.APPROVAL_REQUIRED,
            context=event.to_dict(),
            recipients=approvers,
        )

    async def handle_response(self, action: ResponseAction) -> bool:
        """Process an approval response from email/webhook.

        Args:
            action: Parsed response action

        Returns:
            True if response was processed successfully
        """
        if action.approval_request_id is None:
            logger.warning("Response action missing approval_request_id")
            return False

        # Validate sender is authorized
        pending = await self._store.get_pending_by_token(action.raw_body)  # Token extracted from email subject/headers

        # Process the action
        try:
            if action.is_approval():
                return await self._process_approval(action)
            elif action.is_rejection():
                return await self._process_rejection(action)
            else:
                # Send unknown response message
                await self._send_unknown_response_reply(action)
                return False
        finally:
            # Clean up pending response
            if pending:
                await self._store.delete_pending_response(pending.response_token)

    async def _process_approval(self, action: ResponseAction) -> bool:
        """Process an approval action.

        Args:
            action: Approval action

        Returns:
            True if processed successfully
        """
        try:
            from ..approval.approval_store import ApprovalStore
            from ..approval.models import ApprovalStatus

            store = ApprovalStore()
            request = await store.get_request(action.approval_request_id)
            if not request:
                logger.warning("Approval request %d not found", action.approval_request_id)
                return False

            # Update approval status
            await store.update_request(
                action.approval_request_id,
                {
                    "status": ApprovalStatus.APPROVED.value,
                    "review_notes": f"Approved via email by {action.sender_email}",
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Notify the approval workflow
            from ..background_tasks import get_queue_scheduler

            scheduler = get_queue_scheduler()
            if scheduler:
                await scheduler.approve_job(request.job_id, action.approval_request_id)

            logger.info(
                "Approval request %d approved via email by %s",
                action.approval_request_id,
                action.sender_email,
            )

            # Send confirmation notification
            await self.notify(
                NotificationEventType.APPROVAL_GRANTED,
                {
                    "job_id": request.job_id,
                    "message": f"Job approved by {action.sender_email}",
                    "approval_request_id": action.approval_request_id,
                },
            )

            return True

        except Exception as exc:
            logger.error(
                "Failed to process approval for request %d: %s",
                action.approval_request_id,
                exc,
            )
            return False

    async def _process_rejection(self, action: ResponseAction) -> bool:
        """Process a rejection action.

        Args:
            action: Rejection action

        Returns:
            True if processed successfully
        """
        try:
            from ..approval.approval_store import ApprovalStore
            from ..approval.models import ApprovalStatus

            store = ApprovalStore()
            request = await store.get_request(action.approval_request_id)
            if not request:
                logger.warning("Approval request %d not found", action.approval_request_id)
                return False

            # Update approval status
            await store.update_request(
                action.approval_request_id,
                {
                    "status": ApprovalStatus.REJECTED.value,
                    "review_notes": f"Rejected via email by {action.sender_email}",
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Notify the approval workflow
            from ..background_tasks import get_queue_scheduler

            scheduler = get_queue_scheduler()
            if scheduler:
                await scheduler.reject_job(
                    request.job_id,
                    f"Rejected by {action.sender_email}",
                )

            logger.info(
                "Approval request %d rejected via email by %s",
                action.approval_request_id,
                action.sender_email,
            )

            # Send confirmation notification
            await self.notify(
                NotificationEventType.APPROVAL_REJECTED,
                {
                    "job_id": request.job_id,
                    "message": f"Job rejected by {action.sender_email}",
                    "approval_request_id": action.approval_request_id,
                },
            )

            return True

        except Exception as exc:
            logger.error(
                "Failed to process rejection for request %d: %s",
                action.approval_request_id,
                exc,
            )
            return False

    async def _send_unknown_response_reply(self, action: ResponseAction) -> None:
        """Send a reply explaining valid response options.

        Args:
            action: Unknown response action
        """
        from .response_handlers.parser import get_unknown_response_message

        help_message = get_unknown_response_message()

        # Try to send via email channel
        email_channels = await self._store.list_channels(enabled_only=True, channel_type=ChannelType.EMAIL)

        if email_channels:
            channel = self._channels.get(email_channels[0].id)
            if channel:
                try:
                    await channel.send(
                        event=NotificationEvent(
                            event_type=NotificationEventType.APPROVAL_REQUIRED,
                            title="Unknown Response",
                            message=help_message,
                        ),
                        recipient=action.sender_email,
                        template_vars={"is_help_reply": True},
                    )
                except Exception as exc:
                    logger.debug("Could not send unknown response help: %s", exc)

    # --- Channel Management ---

    async def get_channels(self, enabled_only: bool = True) -> List[ChannelConfig]:
        """List configured notification channels.

        Args:
            enabled_only: If True, only return enabled channels

        Returns:
            List of channel configurations
        """
        return await self._store.list_channels(enabled_only=enabled_only)

    async def get_channel(self, channel_id: int) -> Optional[ChannelConfig]:
        """Get a channel by ID.

        Args:
            channel_id: Channel ID

        Returns:
            Channel configuration or None
        """
        return await self._store.get_channel(channel_id)

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
        channel_config = ChannelConfig(
            channel_type=channel_type,
            name=name,
            is_enabled=config.get("is_enabled", True),
            smtp_host=config.get("smtp_host"),
            smtp_port=config.get("smtp_port", 587),
            smtp_username=config.get("smtp_username"),
            smtp_password_key=config.get("smtp_password_key"),
            smtp_use_tls=config.get("smtp_use_tls", True),
            smtp_from_address=config.get("smtp_from_address"),
            smtp_from_name=config.get("smtp_from_name"),
            webhook_url=config.get("webhook_url"),
            webhook_secret_key=config.get("webhook_secret_key"),
            imap_enabled=config.get("imap_enabled", False),
            imap_host=config.get("imap_host"),
            imap_port=config.get("imap_port", 993),
            imap_username=config.get("imap_username"),
            imap_password_key=config.get("imap_password_key"),
            imap_use_ssl=config.get("imap_use_ssl", True),
            imap_folder=config.get("imap_folder", "INBOX"),
        )

        channel_id = await self._store.create_channel(channel_config)

        # Instantiate the channel
        channel_config.id = channel_id
        await self._instantiate_channel(channel_config)

        # Start response handler if IMAP enabled
        if channel_type == ChannelType.EMAIL and config.get("imap_enabled"):
            await self._start_response_handler(channel_config)

        return channel_id

    async def update_channel(self, channel_id: int, updates: Dict[str, Any]) -> bool:
        """Update a channel configuration.

        Args:
            channel_id: Channel ID
            updates: Fields to update

        Returns:
            True if updated
        """
        success = await self._store.update_channel(channel_id, updates)

        if success:
            # Reload channel instance
            config = await self._store.get_channel(channel_id)
            if config:
                await self._instantiate_channel(config)

        return success

    async def delete_channel(self, channel_id: int) -> bool:
        """Delete a channel.

        Args:
            channel_id: Channel ID

        Returns:
            True if deleted
        """
        # Stop any response handlers for this channel
        self._response_handlers = [
            h
            for h in self._response_handlers
            if getattr(h, "_config", None) is None or getattr(h._config, "id", None) != channel_id
        ]

        # Remove from cache
        self._channels.pop(channel_id, None)

        return await self._store.delete_channel(channel_id)

    async def test_channel(self, channel_id: int) -> tuple[bool, Optional[str], Optional[float]]:
        """Test a channel's connectivity.

        Args:
            channel_id: Channel ID

        Returns:
            Tuple of (success, error_message, latency_ms)
        """
        config = await self._store.get_channel(channel_id)
        if not config:
            return False, "Channel not found", None

        channel_class = self._channel_classes.get(config.channel_type)
        if not channel_class:
            return False, f"No implementation for {config.channel_type}", None

        try:
            channel = channel_class(config)
            return await channel.test_connection(config)
        except Exception as exc:
            return False, str(exc), None

    # --- Preference Management ---

    async def get_preferences(self, user_id: Optional[int] = None) -> List[NotificationPreference]:
        """Get notification preferences.

        Args:
            user_id: Filter by user (None for global)

        Returns:
            List of preferences
        """
        return await self._store.get_preferences(user_id)

    async def set_preference(self, preference: NotificationPreference) -> int:
        """Set a notification preference.

        Args:
            preference: Preference to save

        Returns:
            Preference ID
        """
        return await self._store.set_preference(preference)

    async def delete_preference(self, preference_id: int) -> bool:
        """Delete a preference.

        Args:
            preference_id: Preference ID

        Returns:
            True if deleted
        """
        return await self._store.delete_preference(preference_id)

    # --- Statistics ---

    async def get_status(self) -> Dict[str, Any]:
        """Get notification system status.

        Returns:
            Status dict with channel count, handler status, etc.
        """
        channels = await self._store.list_channels()
        stats = await self._store.get_overall_stats()

        return {
            "initialized": self._initialized,
            "channels": {
                "total": len(channels),
                "enabled": len([c for c in channels if c.is_enabled]),
                "by_type": {ct.value: len([c for c in channels if c.channel_type == ct]) for ct in ChannelType},
            },
            "response_handlers": {
                "total": len(self._response_handlers),
                "running": len([h for h in self._response_handlers if h.is_running]),
            },
            "stats": stats,
        }

    async def get_delivery_history(
        self,
        limit: int = 50,
        offset: int = 0,
        job_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get delivery history.

        Args:
            limit: Maximum records
            offset: Number of records to skip for pagination
            job_id: Filter by job
            channel_id: Filter by channel
            start_date: Filter entries on or after this date (ISO format)
            end_date: Filter entries on or before this date (ISO format)

        Returns:
            List of log entries with channel_name included
        """
        return await self._store.get_delivery_history(
            limit=limit,
            offset=offset,
            job_id=job_id,
            channel_id=channel_id,
            start_date=start_date,
            end_date=end_date,
        )
