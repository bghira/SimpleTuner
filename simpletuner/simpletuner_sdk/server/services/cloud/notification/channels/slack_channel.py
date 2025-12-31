"""Slack notification channel via incoming webhook."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ..models import ChannelConfig, DeliveryResult, NotificationEvent
from ..protocols import ChannelType, DeliveryStatus
from .base import BaseNotificationChannel

logger = logging.getLogger(__name__)


class SlackChannel(BaseNotificationChannel):
    """Send notifications to Slack via incoming webhook.

    Uses Slack's Block Kit for rich formatting.
    """

    channel_type = ChannelType.SLACK

    async def send(
        self,
        event: NotificationEvent,
        recipient: str,
        template_vars: Dict[str, Any],
    ) -> DeliveryResult:
        """Send Slack notification.

        Args:
            event: Notification event
            recipient: Slack webhook URL (overrides config if provided)
            template_vars: Template variables

        Returns:
            DeliveryResult
        """
        start_time = time.monotonic()
        webhook_url = recipient or self._config.webhook_url

        if not webhook_url:
            return DeliveryResult(
                success=False,
                channel_id=self._config.id,
                channel_type=self.channel_type,
                recipient=webhook_url or "none",
                event_type=event.event_type,
                delivery_status=DeliveryStatus.FAILED,
                error_message="No Slack webhook URL configured",
            )

        try:
            # Build Slack payload with blocks
            payload = self._build_slack_payload(event, template_vars)

            # Send request
            from ...http_client import get_async_client

            async with get_async_client(timeout=30.0) as client:
                response = await client.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

            latency = (time.monotonic() - start_time) * 1000

            # Slack returns "ok" on success
            if response.is_success and response.text == "ok":
                return DeliveryResult(
                    success=True,
                    channel_id=self._config.id,
                    channel_type=self.channel_type,
                    recipient=webhook_url,
                    event_type=event.event_type,
                    delivery_status=DeliveryStatus.DELIVERED,
                    latency_ms=latency,
                )
            else:
                return DeliveryResult(
                    success=False,
                    channel_id=self._config.id,
                    channel_type=self.channel_type,
                    recipient=webhook_url,
                    event_type=event.event_type,
                    delivery_status=DeliveryStatus.FAILED,
                    error_message=f"Slack error: {response.text[:200]}",
                    latency_ms=latency,
                )

        except Exception as exc:
            latency = (time.monotonic() - start_time) * 1000
            logger.error("Slack send failed to %s: %s", webhook_url, exc)

            return DeliveryResult(
                success=False,
                channel_id=self._config.id,
                channel_type=self.channel_type,
                recipient=webhook_url or "none",
                event_type=event.event_type,
                delivery_status=DeliveryStatus.FAILED,
                error_message=str(exc),
                latency_ms=latency,
            )

    def _build_slack_payload(self, event: NotificationEvent, template_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Build Slack Block Kit payload.

        Args:
            event: Notification event
            template_vars: Template variables

        Returns:
            Slack payload dict
        """
        # Severity to emoji mapping
        severity_emoji = {
            "info": ":information_source:",
            "success": ":white_check_mark:",
            "warning": ":warning:",
            "error": ":x:",
            "critical": ":rotating_light:",
        }
        emoji = severity_emoji.get(event.severity, ":bell:")

        # Build blocks
        blocks: List[Dict[str, Any]] = []

        # Header block
        blocks.append(
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {self._format_title(event)}",
                    "emoji": True,
                },
            }
        )

        # Message section
        message = self._format_message(event, template_vars)
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message,
                },
            }
        )

        # Context fields
        context_elements = []

        if event.job_id:
            context_elements.append(
                {
                    "type": "mrkdwn",
                    "text": f"*Job:* `{event.job_id[:12]}`",
                }
            )

        if template_vars.get("estimated_cost"):
            context_elements.append(
                {
                    "type": "mrkdwn",
                    "text": f"*Est. Cost:* ${template_vars['estimated_cost']:.2f}",
                }
            )

        if template_vars.get("config_name"):
            context_elements.append(
                {
                    "type": "mrkdwn",
                    "text": f"*Config:* {template_vars['config_name']}",
                }
            )

        if context_elements:
            blocks.append(
                {
                    "type": "context",
                    "elements": context_elements,
                }
            )

        # Divider
        blocks.append({"type": "divider"})

        # Footer
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"SimpleTuner Cloud • {event.created_at.strftime('%Y-%m-%d %H:%M UTC')}",
                    },
                ],
            }
        )

        return {
            "blocks": blocks,
            "text": f"{self._format_title(event)}: {message[:100]}",  # Fallback
        }

    async def validate_config(self, config: ChannelConfig) -> tuple[bool, Optional[str]]:
        """Validate Slack channel configuration.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not config.webhook_url:
            return False, "Slack webhook URL is required"

        # Validate URL format (should be hooks.slack.com)
        from urllib.parse import urlparse

        try:
            parsed = urlparse(config.webhook_url)
            if parsed.scheme != "https":
                return False, "Slack webhook must use https"
            if "slack.com" not in parsed.netloc:
                return False, "URL doesn't appear to be a Slack webhook"
        except Exception:
            return False, "Invalid webhook URL format"

        return True, None

    async def test_connection(self, config: ChannelConfig) -> tuple[bool, Optional[str], Optional[float]]:
        """Test Slack webhook connectivity.

        Args:
            config: Configuration to test

        Returns:
            Tuple of (success, error_message, latency_ms)
        """
        if not config.webhook_url:
            return False, "No Slack webhook URL configured", None

        start_time = time.monotonic()

        try:
            from ...http_client import get_async_client

            # Send a minimal test message
            test_payload = {
                "text": ":white_check_mark: SimpleTuner webhook test successful",
            }

            async with get_async_client(timeout=10.0) as client:
                response = await client.post(
                    config.webhook_url,
                    json=test_payload,
                    headers={"Content-Type": "application/json"},
                )

            latency = (time.monotonic() - start_time) * 1000

            if response.is_success and response.text == "ok":
                return True, None, latency
            else:
                return False, f"Slack error: {response.text}", latency

        except Exception as exc:
            latency = (time.monotonic() - start_time) * 1000
            return False, str(exc), latency
