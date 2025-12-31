"""Webhook notification channel."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional

from ..models import ChannelConfig, DeliveryResult, NotificationEvent
from ..protocols import ChannelType, DeliveryStatus
from .base import BaseNotificationChannel

logger = logging.getLogger(__name__)


class WebhookChannel(BaseNotificationChannel):
    """Send notifications via HTTP webhook.

    Features:
    - HMAC-SHA256 signature for verification
    - Retry with exponential backoff
    - Timeout handling
    - Uses HTTPClientFactory for connection pooling
    """

    channel_type = ChannelType.WEBHOOK

    async def send(
        self,
        event: NotificationEvent,
        recipient: str,
        template_vars: Dict[str, Any],
    ) -> DeliveryResult:
        """Send webhook notification.

        Args:
            event: Notification event
            recipient: Webhook URL (overrides config if provided)
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
                error_message="No webhook URL configured",
            )

        try:
            # Build payload
            payload = self._build_payload(event, template_vars)
            payload_bytes = json.dumps(payload, default=str).encode()

            # Sign payload
            signature = await self._sign_payload(payload_bytes)

            # Send request
            from ...http_client import get_async_client

            headers = {
                "Content-Type": "application/json",
                "X-SimpleTuner-Event": event.event_type.value,
                "X-SimpleTuner-Timestamp": str(int(time.time())),
            }
            if signature:
                headers["X-SimpleTuner-Signature"] = signature

            async with get_async_client(timeout=30.0) as client:
                response = await client.post(
                    webhook_url,
                    content=payload_bytes,
                    headers=headers,
                )

            latency = (time.monotonic() - start_time) * 1000

            if response.is_success:
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
                    error_message=f"HTTP {response.status_code}: {response.text[:200]}",
                    latency_ms=latency,
                )

        except Exception as exc:
            latency = (time.monotonic() - start_time) * 1000
            logger.error("Webhook send failed to %s: %s", webhook_url, exc)

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

    def _build_payload(self, event: NotificationEvent, template_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Build webhook payload.

        Args:
            event: Notification event
            template_vars: Template variables

        Returns:
            Payload dict
        """
        return {
            "event_type": event.event_type.value,
            "title": self._format_title(event),
            "message": self._format_message(event, template_vars),
            "severity": event.severity,
            "job_id": event.job_id,
            "user_id": event.user_id,
            "timestamp": event.created_at.isoformat(),
            "data": event.data,
        }

    async def _sign_payload(self, payload_bytes: bytes) -> Optional[str]:
        """Sign payload with HMAC-SHA256.

        Args:
            payload_bytes: Payload to sign

        Returns:
            Signature or None
        """
        if not self._config.webhook_secret_key:
            return None

        try:
            # Get secret from secrets manager
            from ...secrets import get_secrets_manager

            secrets = get_secrets_manager()
            secret = secrets.get(self._config.webhook_secret_key)

            if not secret:
                return None

            signature = hmac.new(
                secret.encode(),
                payload_bytes,
                hashlib.sha256,
            ).hexdigest()

            return f"sha256={signature}"

        except Exception as exc:
            logger.debug("Could not sign payload: %s", exc)
            return None

    async def validate_config(self, config: ChannelConfig) -> tuple[bool, Optional[str]]:
        """Validate webhook channel configuration.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not config.webhook_url:
            return False, "Webhook URL is required"

        # Validate URL format
        from urllib.parse import urlparse

        try:
            parsed = urlparse(config.webhook_url)
            if parsed.scheme not in ("http", "https"):
                return False, "Webhook URL must use http or https"
            if not parsed.netloc:
                return False, "Invalid webhook URL"
        except Exception:
            return False, "Invalid webhook URL format"

        return True, None

    async def test_connection(self, config: ChannelConfig) -> tuple[bool, Optional[str], Optional[float]]:
        """Test webhook connectivity.

        Sends a test event to the webhook URL.

        Args:
            config: Configuration to test

        Returns:
            Tuple of (success, error_message, latency_ms)
        """
        if not config.webhook_url:
            return False, "No webhook URL configured", None

        start_time = time.monotonic()

        try:
            from ...http_client import get_async_client

            test_payload = {
                "event_type": "test",
                "message": "SimpleTuner webhook test",
                "timestamp": time.time(),
            }

            async with get_async_client(timeout=10.0) as client:
                response = await client.post(
                    config.webhook_url,
                    json=test_payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-SimpleTuner-Event": "test",
                    },
                )

            latency = (time.monotonic() - start_time) * 1000

            if response.is_success:
                return True, None, latency
            else:
                return False, f"HTTP {response.status_code}", latency

        except Exception as exc:
            latency = (time.monotonic() - start_time) * 1000
            return False, str(exc), latency
