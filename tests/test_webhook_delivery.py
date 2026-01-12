"""Tests for webhook delivery reliability.

Tests cover:
- WebhookChannel configuration validation
- HMAC signature generation and verification
- HTTP request handling (success, failure, timeout)
- Retry behavior with different status codes
- Latency tracking
- Error handling and fallback behavior
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import unittest
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Suppress expected error logs during tests
logging.getLogger("simpletuner.simpletuner_sdk.server.services.cloud.notification.channels.webhook_channel").setLevel(
    logging.CRITICAL
)

from simpletuner.simpletuner_sdk.server.services.cloud.notification.channels.webhook_channel import WebhookChannel
from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import (
    ChannelConfig,
    DeliveryResult,
    NotificationEvent,
)
from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import (
    ChannelType,
    DeliveryStatus,
    NotificationEventType,
    Severity,
)


class TestWebhookChannelConfig(unittest.IsolatedAsyncioTestCase):
    """Tests for webhook channel configuration."""

    def _create_config(self, **kwargs) -> ChannelConfig:
        """Create a test channel config."""
        defaults = {
            "id": 1,
            "channel_type": ChannelType.WEBHOOK,
            "name": "Test Webhook",
            "is_enabled": True,
            "webhook_url": "https://example.com/webhook",
        }
        defaults.update(kwargs)
        return ChannelConfig(**defaults)

    async def test_valid_https_url(self):
        """Test validation of HTTPS URL."""
        config = self._create_config(webhook_url="https://example.com/webhook")
        channel = WebhookChannel(config)

        is_valid, error = await channel.validate_config(config)

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    async def test_valid_http_url(self):
        """Test validation of HTTP URL (allowed but less secure)."""
        config = self._create_config(webhook_url="http://localhost:8080/webhook")
        channel = WebhookChannel(config)

        is_valid, error = await channel.validate_config(config)

        self.assertTrue(is_valid)
        self.assertIsNone(error)

    async def test_missing_url(self):
        """Test validation fails with missing URL."""
        config = self._create_config(webhook_url=None)
        channel = WebhookChannel(config)

        is_valid, error = await channel.validate_config(config)

        self.assertFalse(is_valid)
        self.assertIn("required", error.lower())

    async def test_invalid_scheme(self):
        """Test validation fails with invalid URL scheme."""
        config = self._create_config(webhook_url="ftp://example.com/webhook")
        channel = WebhookChannel(config)

        is_valid, error = await channel.validate_config(config)

        self.assertFalse(is_valid)
        self.assertIn("http", error.lower())

    async def test_invalid_url_format(self):
        """Test validation fails with malformed URL."""
        config = self._create_config(webhook_url="not-a-valid-url")
        channel = WebhookChannel(config)

        is_valid, error = await channel.validate_config(config)

        self.assertFalse(is_valid)
        # URL without scheme fails the http/https check
        self.assertIsNotNone(error)


class TestWebhookPayload(unittest.IsolatedAsyncioTestCase):
    """Tests for webhook payload construction."""

    def _create_event(self, **kwargs) -> NotificationEvent:
        """Create a test notification event."""
        defaults = {
            "event_type": NotificationEventType.JOB_COMPLETED,
            "title": "Job Complete",
            "message": "Training job finished successfully",
            "severity": Severity.INFO.value,
            "job_id": "job-123",
            "user_id": 1,
            "data": {"steps": 1000, "loss": 0.05},
            "created_at": datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        }
        defaults.update(kwargs)
        return NotificationEvent(**defaults)

    def _create_channel(self, webhook_url: str = "https://example.com/webhook") -> WebhookChannel:
        """Create a test webhook channel."""
        config = ChannelConfig(
            id=1,
            channel_type=ChannelType.WEBHOOK,
            name="Test Webhook",
            is_enabled=True,
            webhook_url=webhook_url,
        )
        return WebhookChannel(config)

    def test_payload_structure(self):
        """Test payload contains all required fields."""
        channel = self._create_channel()
        event = self._create_event()

        payload = channel._build_payload(event, {"extra": "value"})

        self.assertEqual(payload["event_type"], "job.completed")
        self.assertEqual(payload["title"], "Job Complete")
        self.assertIn("message", payload)
        self.assertEqual(payload["severity"], "info")
        self.assertEqual(payload["job_id"], "job-123")
        self.assertEqual(payload["user_id"], 1)
        self.assertIn("timestamp", payload)
        self.assertIn("data", payload)

    def test_payload_with_data(self):
        """Test payload includes custom data."""
        channel = self._create_channel()
        event = self._create_event(data={"custom_field": "custom_value"})

        payload = channel._build_payload(event, {})

        self.assertEqual(payload["data"]["custom_field"], "custom_value")

    def test_payload_serialization(self):
        """Test payload can be JSON serialized."""
        channel = self._create_channel()
        event = self._create_event()

        payload = channel._build_payload(event, {})

        # Should not raise
        json_str = json.dumps(payload, default=str)
        self.assertIsInstance(json_str, str)

        # Should be deserializable
        parsed = json.loads(json_str)
        self.assertEqual(parsed["event_type"], "job.completed")


class TestWebhookSignature(unittest.IsolatedAsyncioTestCase):
    """Tests for HMAC signature generation."""

    def _create_channel_with_secret(self, secret_key: str = "webhook_secret") -> WebhookChannel:
        """Create a webhook channel with secret configured."""
        config = ChannelConfig(
            id=1,
            channel_type=ChannelType.WEBHOOK,
            name="Test Webhook",
            is_enabled=True,
            webhook_url="https://example.com/webhook",
            webhook_secret_key=secret_key,
        )
        return WebhookChannel(config)

    async def test_signature_generation(self):
        """Test HMAC-SHA256 signature is correctly generated."""
        channel = self._create_channel_with_secret("webhook_secret")
        payload = b'{"event_type":"test"}'

        # Mock secrets manager - patch where it's defined
        with patch("simpletuner.simpletuner_sdk.server.services.cloud.secrets.get_secrets_manager") as mock_get_secrets:
            mock_manager = MagicMock()
            mock_manager.get.return_value = "my-secret-key"
            mock_get_secrets.return_value = mock_manager

            signature = await channel._sign_payload(payload)

        # Verify signature format
        self.assertIsNotNone(signature)
        self.assertTrue(signature.startswith("sha256="))

        # Verify signature is correct
        expected = hmac.new(
            b"my-secret-key",
            payload,
            hashlib.sha256,
        ).hexdigest()
        self.assertEqual(signature, f"sha256={expected}")

    async def test_no_signature_without_secret(self):
        """Test no signature is generated when secret is not configured."""
        config = ChannelConfig(
            id=1,
            channel_type=ChannelType.WEBHOOK,
            name="Test Webhook",
            is_enabled=True,
            webhook_url="https://example.com/webhook",
            webhook_secret_key=None,
        )
        channel = WebhookChannel(config)
        payload = b'{"event_type":"test"}'

        signature = await channel._sign_payload(payload)

        self.assertIsNone(signature)

    async def test_signature_with_missing_secret_value(self):
        """Test no signature when secret key exists but value is missing."""
        channel = self._create_channel_with_secret("missing_secret")

        with patch("simpletuner.simpletuner_sdk.server.services.cloud.secrets.get_secrets_manager") as mock_get_secrets:
            mock_manager = MagicMock()
            mock_manager.get.return_value = None  # Secret not found
            mock_get_secrets.return_value = mock_manager

            signature = await channel._sign_payload(b'{"test":"data"}')

        self.assertIsNone(signature)


class TestWebhookDelivery(unittest.IsolatedAsyncioTestCase):
    """Tests for webhook HTTP delivery."""

    def _create_event(self) -> NotificationEvent:
        """Create a test notification event."""
        return NotificationEvent(
            event_type=NotificationEventType.JOB_COMPLETED,
            title="Job Complete",
            message="Test message",
            job_id="job-123",
        )

    def _create_channel(self, url: str = "https://example.com/webhook") -> WebhookChannel:
        """Create a test webhook channel."""
        config = ChannelConfig(
            id=1,
            channel_type=ChannelType.WEBHOOK,
            name="Test Webhook",
            is_enabled=True,
            webhook_url=url,
        )
        return WebhookChannel(config)

    async def test_successful_delivery(self):
        """Test successful webhook delivery."""
        channel = self._create_channel()
        event = self._create_event()

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200

        with patch("simpletuner.simpletuner_sdk.server.services.cloud.http_client.get_async_client") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_ctx
            mock_ctx.__aexit__.return_value = None
            mock_ctx.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_ctx

            result = await channel.send(event, "", {})

        self.assertTrue(result.success)
        self.assertEqual(result.delivery_status, DeliveryStatus.DELIVERED)
        self.assertIsNone(result.error_message)
        self.assertIsNotNone(result.latency_ms)

    async def test_failed_delivery_http_error(self):
        """Test failed delivery due to HTTP error."""
        channel = self._create_channel()
        event = self._create_event()

        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("simpletuner.simpletuner_sdk.server.services.cloud.http_client.get_async_client") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_ctx
            mock_ctx.__aexit__.return_value = None
            mock_ctx.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_ctx

            result = await channel.send(event, "", {})

        self.assertFalse(result.success)
        self.assertEqual(result.delivery_status, DeliveryStatus.FAILED)
        self.assertIn("500", result.error_message)

    async def test_failed_delivery_connection_error(self):
        """Test failed delivery due to connection error."""
        channel = self._create_channel()
        event = self._create_event()

        with patch("simpletuner.simpletuner_sdk.server.services.cloud.http_client.get_async_client") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_ctx
            mock_ctx.__aexit__.return_value = None
            mock_ctx.post = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client.return_value = mock_ctx

            result = await channel.send(event, "", {})

        self.assertFalse(result.success)
        self.assertEqual(result.delivery_status, DeliveryStatus.FAILED)
        self.assertIn("Connection refused", result.error_message)

    async def test_delivery_with_recipient_override(self):
        """Test delivery to overridden recipient URL."""
        channel = self._create_channel("https://default.com/webhook")
        event = self._create_event()
        override_url = "https://override.com/webhook"

        mock_response = MagicMock()
        mock_response.is_success = True

        with patch("simpletuner.simpletuner_sdk.server.services.cloud.http_client.get_async_client") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_ctx
            mock_ctx.__aexit__.return_value = None
            mock_ctx.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_ctx

            result = await channel.send(event, override_url, {})

            # Verify override URL was used
            call_args = mock_ctx.post.call_args
            self.assertEqual(call_args[0][0], override_url)

        self.assertEqual(result.recipient, override_url)

    async def test_delivery_no_url_configured(self):
        """Test delivery fails when no URL is configured."""
        config = ChannelConfig(
            id=1,
            channel_type=ChannelType.WEBHOOK,
            name="Test Webhook",
            is_enabled=True,
            webhook_url=None,
        )
        channel = WebhookChannel(config)
        event = self._create_event()

        result = await channel.send(event, "", {})

        self.assertFalse(result.success)
        self.assertEqual(result.delivery_status, DeliveryStatus.FAILED)
        self.assertIn("No webhook URL", result.error_message)


class TestWebhookHeaders(unittest.IsolatedAsyncioTestCase):
    """Tests for webhook request headers."""

    def _create_event(self) -> NotificationEvent:
        """Create a test notification event."""
        return NotificationEvent(
            event_type=NotificationEventType.APPROVAL_REQUIRED,
            title="Approval Required",
            message="Test message",
        )

    def _create_channel(self, secret_key: Optional[str] = None) -> WebhookChannel:
        """Create a test webhook channel."""
        config = ChannelConfig(
            id=1,
            channel_type=ChannelType.WEBHOOK,
            name="Test Webhook",
            is_enabled=True,
            webhook_url="https://example.com/webhook",
            webhook_secret_key=secret_key,
        )
        return WebhookChannel(config)

    async def test_content_type_header(self):
        """Test Content-Type header is set correctly."""
        channel = self._create_channel()
        event = self._create_event()

        mock_response = MagicMock()
        mock_response.is_success = True

        with patch("simpletuner.simpletuner_sdk.server.services.cloud.http_client.get_async_client") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_ctx
            mock_ctx.__aexit__.return_value = None
            mock_ctx.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_ctx

            await channel.send(event, "", {})

            # Check headers
            call_kwargs = mock_ctx.post.call_args[1]
            headers = call_kwargs.get("headers", {})
            self.assertEqual(headers.get("Content-Type"), "application/json")

    async def test_event_type_header(self):
        """Test X-SimpleTuner-Event header is set correctly."""
        channel = self._create_channel()
        event = self._create_event()

        mock_response = MagicMock()
        mock_response.is_success = True

        with patch("simpletuner.simpletuner_sdk.server.services.cloud.http_client.get_async_client") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_ctx
            mock_ctx.__aexit__.return_value = None
            mock_ctx.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_ctx

            await channel.send(event, "", {})

            call_kwargs = mock_ctx.post.call_args[1]
            headers = call_kwargs.get("headers", {})
            self.assertEqual(headers.get("X-SimpleTuner-Event"), "approval.required")

    async def test_timestamp_header(self):
        """Test X-SimpleTuner-Timestamp header is set correctly."""
        channel = self._create_channel()
        event = self._create_event()

        mock_response = MagicMock()
        mock_response.is_success = True

        with patch("simpletuner.simpletuner_sdk.server.services.cloud.http_client.get_async_client") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_ctx
            mock_ctx.__aexit__.return_value = None
            mock_ctx.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_ctx

            before = int(time.time())
            await channel.send(event, "", {})
            after = int(time.time())

            call_kwargs = mock_ctx.post.call_args[1]
            headers = call_kwargs.get("headers", {})
            timestamp = int(headers.get("X-SimpleTuner-Timestamp", 0))

            self.assertGreaterEqual(timestamp, before)
            self.assertLessEqual(timestamp, after)

    async def test_signature_header_present(self):
        """Test X-SimpleTuner-Signature header is set when secret configured."""
        channel = self._create_channel(secret_key="webhook_secret")
        event = self._create_event()

        mock_response = MagicMock()
        mock_response.is_success = True

        with patch("simpletuner.simpletuner_sdk.server.services.cloud.http_client.get_async_client") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_ctx
            mock_ctx.__aexit__.return_value = None
            mock_ctx.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_ctx

            with patch("simpletuner.simpletuner_sdk.server.services.cloud.secrets.get_secrets_manager") as mock_get_secrets:
                mock_manager = MagicMock()
                mock_manager.get.return_value = "my-secret"
                mock_get_secrets.return_value = mock_manager

                await channel.send(event, "", {})

            call_kwargs = mock_ctx.post.call_args[1]
            headers = call_kwargs.get("headers", {})
            self.assertIn("X-SimpleTuner-Signature", headers)
            self.assertTrue(headers["X-SimpleTuner-Signature"].startswith("sha256="))


class TestWebhookConnectionTest(unittest.IsolatedAsyncioTestCase):
    """Tests for webhook connection testing."""

    def _create_channel(self, url: str = "https://example.com/webhook") -> WebhookChannel:
        """Create a test webhook channel."""
        config = ChannelConfig(
            id=1,
            channel_type=ChannelType.WEBHOOK,
            name="Test Webhook",
            is_enabled=True,
            webhook_url=url,
        )
        return WebhookChannel(config)

    async def test_connection_test_success(self):
        """Test successful connection test."""
        config = ChannelConfig(
            id=1,
            channel_type=ChannelType.WEBHOOK,
            name="Test Webhook",
            is_enabled=True,
            webhook_url="https://example.com/webhook",
        )
        channel = WebhookChannel(config)

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200

        with patch("simpletuner.simpletuner_sdk.server.services.cloud.http_client.get_async_client") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_ctx
            mock_ctx.__aexit__.return_value = None
            mock_ctx.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_ctx

            success, error, latency = await channel.test_connection(config)

        self.assertTrue(success)
        self.assertIsNone(error)
        self.assertIsNotNone(latency)
        self.assertGreater(latency, 0)

    async def test_connection_test_http_error(self):
        """Test connection test with HTTP error."""
        config = ChannelConfig(
            id=1,
            channel_type=ChannelType.WEBHOOK,
            name="Test Webhook",
            is_enabled=True,
            webhook_url="https://example.com/webhook",
        )
        channel = WebhookChannel(config)

        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 404

        with patch("simpletuner.simpletuner_sdk.server.services.cloud.http_client.get_async_client") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_ctx
            mock_ctx.__aexit__.return_value = None
            mock_ctx.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_ctx

            success, error, latency = await channel.test_connection(config)

        self.assertFalse(success)
        self.assertEqual(error, "Webhook responded with HTTP 404")
        self.assertIsNotNone(latency)

    async def test_connection_test_no_url(self):
        """Test connection test with no URL configured."""
        config = ChannelConfig(
            id=1,
            channel_type=ChannelType.WEBHOOK,
            name="Test Webhook",
            is_enabled=True,
            webhook_url=None,
        )
        channel = WebhookChannel(config)

        success, error, latency = await channel.test_connection(config)

        self.assertFalse(success)
        self.assertIn("No webhook URL", error)
        self.assertIsNone(latency)

    async def test_connection_test_exception(self):
        """Test connection test with exception."""
        config = ChannelConfig(
            id=1,
            channel_type=ChannelType.WEBHOOK,
            name="Test Webhook",
            is_enabled=True,
            webhook_url="https://example.com/webhook",
        )
        channel = WebhookChannel(config)

        with patch("simpletuner.simpletuner_sdk.server.services.cloud.http_client.get_async_client") as mock_client:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_ctx
            mock_ctx.__aexit__.return_value = None
            mock_ctx.post = AsyncMock(side_effect=TimeoutError("Connection timeout"))
            mock_client.return_value = mock_ctx

            success, error, latency = await channel.test_connection(config)

        self.assertFalse(success)
        self.assertIn("timeout", error.lower())
        self.assertIsNotNone(latency)


class TestDeliveryResult(unittest.TestCase):
    """Tests for DeliveryResult model."""

    def test_successful_result(self):
        """Test creating a successful delivery result."""
        result = DeliveryResult(
            success=True,
            channel_id=1,
            channel_type=ChannelType.WEBHOOK,
            recipient="https://example.com/webhook",
            event_type=NotificationEventType.JOB_COMPLETED,
            delivery_status=DeliveryStatus.DELIVERED,
            latency_ms=150.5,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.delivery_status, DeliveryStatus.DELIVERED)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.latency_ms, 150.5)

    def test_failed_result(self):
        """Test creating a failed delivery result."""
        result = DeliveryResult(
            success=False,
            channel_id=1,
            channel_type=ChannelType.WEBHOOK,
            recipient="https://example.com/webhook",
            event_type=NotificationEventType.JOB_FAILED,
            delivery_status=DeliveryStatus.FAILED,
            error_message="Connection refused",
            latency_ms=50.0,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.delivery_status, DeliveryStatus.FAILED)
        self.assertEqual(result.error_message, "Connection refused")

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = DeliveryResult(
            success=True,
            channel_id=1,
            channel_type=ChannelType.WEBHOOK,
            recipient="https://example.com/webhook",
            event_type=NotificationEventType.JOB_COMPLETED,
            delivery_status=DeliveryStatus.DELIVERED,
            latency_ms=100.0,
        )

        d = result.to_dict()

        self.assertEqual(d["success"], True)
        self.assertEqual(d["channel_type"], "webhook")
        self.assertEqual(d["delivery_status"], "delivered")


class TestNotificationEventModel(unittest.TestCase):
    """Tests for NotificationEvent model."""

    def test_create_event(self):
        """Test creating a notification event."""
        event = NotificationEvent(
            event_type=NotificationEventType.JOB_STARTED,
            title="Job Started",
            message="Training has begun",
            job_id="job-456",
            user_id=2,
        )

        self.assertEqual(event.event_type, NotificationEventType.JOB_STARTED)
        self.assertEqual(event.title, "Job Started")
        self.assertEqual(event.job_id, "job-456")

    def test_event_from_context(self):
        """Test creating event from context dictionary."""
        context = {
            "title": "Quota Warning",
            "message": "You're approaching your limit",
            "severity": "warning",
            "job_id": "job-789",
            "user_id": 3,
        }

        event = NotificationEvent.from_context(
            NotificationEventType.QUOTA_WARNING,
            context,
        )

        self.assertEqual(event.event_type, NotificationEventType.QUOTA_WARNING)
        self.assertEqual(event.title, "Quota Warning")
        self.assertEqual(event.severity, "warning")

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = NotificationEvent(
            event_type=NotificationEventType.JOB_COMPLETED,
            title="Done",
            message="Complete",
        )

        d = event.to_dict()

        self.assertEqual(d["event_type"], "job.completed")
        self.assertIn("created_at", d)

    def test_event_with_response_token(self):
        """Test generating event with response token."""
        event = NotificationEvent(
            event_type=NotificationEventType.APPROVAL_REQUIRED,
            approval_request_id=123,
        )

        event_with_token = event.with_response_token()

        self.assertIsNotNone(event_with_token.response_token)
        self.assertEqual(len(event_with_token.response_token), 43)  # URL-safe base64
        self.assertEqual(event_with_token.approval_request_id, 123)


class TestSeverityLevels(unittest.TestCase):
    """Tests for severity level handling."""

    def test_severity_order(self):
        """Test severity ordering."""
        order = Severity.severity_order()

        self.assertLess(order["debug"], order["info"])
        self.assertLess(order["info"], order["warning"])
        self.assertLess(order["warning"], order["error"])
        self.assertLess(order["error"], order["critical"])

    def test_meets_threshold(self):
        """Test severity threshold checking."""
        # Warning meets info threshold
        self.assertTrue(Severity.meets_threshold("warning", "info"))

        # Info does not meet warning threshold
        self.assertFalse(Severity.meets_threshold("info", "warning"))

        # Same severity meets threshold
        self.assertTrue(Severity.meets_threshold("error", "error"))

        # Critical meets all thresholds
        self.assertTrue(Severity.meets_threshold("critical", "debug"))
        self.assertTrue(Severity.meets_threshold("critical", "info"))
        self.assertTrue(Severity.meets_threshold("critical", "critical"))


class TestChannelConfig(unittest.TestCase):
    """Tests for ChannelConfig model."""

    def test_create_webhook_config(self):
        """Test creating webhook channel config."""
        config = ChannelConfig(
            id=1,
            channel_type=ChannelType.WEBHOOK,
            name="Production Webhook",
            is_enabled=True,
            webhook_url="https://prod.example.com/webhook",
            webhook_secret_key="prod_secret",
        )

        self.assertEqual(config.channel_type, ChannelType.WEBHOOK)
        self.assertEqual(config.webhook_url, "https://prod.example.com/webhook")

    def test_config_to_dict_without_secrets(self):
        """Test to_dict excludes secrets by default."""
        config = ChannelConfig(
            id=1,
            channel_type=ChannelType.WEBHOOK,
            name="Test",
            webhook_secret_key="secret_ref",
        )

        d = config.to_dict(include_secrets=False)

        self.assertNotIn("webhook_secret_key", d)
        self.assertIn("webhook_url", d)

    def test_config_to_dict_with_secrets(self):
        """Test to_dict includes secrets when requested."""
        config = ChannelConfig(
            id=1,
            channel_type=ChannelType.WEBHOOK,
            name="Test",
            webhook_secret_key="secret_ref",
        )

        d = config.to_dict(include_secrets=True)

        self.assertIn("webhook_secret_key", d)
        self.assertEqual(d["webhook_secret_key"], "secret_ref")

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "id": 5,
            "channel_type": "webhook",
            "name": "Imported Config",
            "is_enabled": True,
            "webhook_url": "https://import.example.com/hook",
        }

        config = ChannelConfig.from_dict(data)

        self.assertEqual(config.id, 5)
        self.assertEqual(config.channel_type, ChannelType.WEBHOOK)
        self.assertEqual(config.name, "Imported Config")


if __name__ == "__main__":
    unittest.main()
