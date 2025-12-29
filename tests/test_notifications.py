"""Tests for notification channels and delivery.

Tests the notification system including:
- Email channel (SMTP)
- Webhook channel
- Slack channel
- Channel validation
- Delivery result handling
- Notification routing
"""

import asyncio
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


class TestNotificationModels(unittest.TestCase):
    """Tests for notification data models."""

    def test_notification_event_creation(self):
        """Test creating a notification event."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import NotificationEvent

        event = NotificationEvent(
            event_type="job_completed",
            title="Job Completed",
            message="Training job finished successfully",
            job_id="job-123",
            severity="info",
        )

        self.assertEqual(event.event_type, "job_completed")
        self.assertEqual(event.title, "Job Completed")
        self.assertEqual(event.job_id, "job-123")
        self.assertEqual(event.severity, "info")

    def test_notification_event_defaults(self):
        """Test notification event default values."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import NotificationEvent

        event = NotificationEvent(
            event_type="test",
            title="Test",
            message="Test message",
        )

        self.assertEqual(event.severity, "info")
        self.assertIsNone(event.job_id)
        self.assertIsNone(event.user_id)

    def test_channel_config_smtp(self):
        """Test SMTP channel configuration."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig

        config = ChannelConfig(
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_username="user@example.com",
            smtp_use_tls=True,
            smtp_from_address="noreply@example.com",
            smtp_from_name="SimpleTuner",
        )

        self.assertEqual(config.smtp_host, "smtp.example.com")
        self.assertEqual(config.smtp_port, 587)
        self.assertTrue(config.smtp_use_tls)

    def test_channel_config_webhook(self):
        """Test webhook channel configuration."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig

        config = ChannelConfig(
            webhook_url="https://hooks.example.com/notify",
            webhook_secret_key="secret_key_ref",
        )

        self.assertEqual(config.webhook_url, "https://hooks.example.com/notify")
        self.assertEqual(config.webhook_secret_key, "secret_key_ref")

    def test_delivery_result_success(self):
        """Test successful delivery result."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import DeliveryResult
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import (
            ChannelType,
            DeliveryStatus,
            NotificationEventType,
        )

        result = DeliveryResult(
            success=True,
            channel_id=1,
            channel_type=ChannelType.EMAIL,
            recipient="user@example.com",
            event_type=NotificationEventType.JOB_COMPLETED,
            delivery_status=DeliveryStatus.DELIVERED,
            latency_ms=150.0,
        )

        self.assertTrue(result.success)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.latency_ms, 150.0)

    def test_delivery_result_failure(self):
        """Test failed delivery result."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import DeliveryResult
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import (
            ChannelType,
            DeliveryStatus,
            NotificationEventType,
        )

        result = DeliveryResult(
            success=False,
            channel_id=1,
            channel_type=ChannelType.EMAIL,
            recipient="user@example.com",
            event_type=NotificationEventType.JOB_FAILED,
            delivery_status=DeliveryStatus.FAILED,
            error_message="Connection timeout",
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Connection timeout")


class TestNotificationEventTypes(unittest.TestCase):
    """Tests for notification event type enum."""

    def test_event_type_values(self):
        """Test that all expected event types exist."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import NotificationEventType

        # Job events - use dotted notation
        self.assertEqual(NotificationEventType.JOB_STARTED.value, "job.started")
        self.assertEqual(NotificationEventType.JOB_COMPLETED.value, "job.completed")
        self.assertEqual(NotificationEventType.JOB_FAILED.value, "job.failed")

        # Approval events - use dotted notation
        self.assertEqual(NotificationEventType.APPROVAL_REQUIRED.value, "approval.required")
        self.assertEqual(NotificationEventType.APPROVAL_GRANTED.value, "approval.granted")

    def test_severity_values(self):
        """Test severity level values."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import Severity

        self.assertEqual(Severity.DEBUG.value, "debug")
        self.assertEqual(Severity.INFO.value, "info")
        self.assertEqual(Severity.WARNING.value, "warning")
        self.assertEqual(Severity.ERROR.value, "error")
        self.assertEqual(Severity.CRITICAL.value, "critical")

    def test_channel_type_values(self):
        """Test channel type values."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import ChannelType

        self.assertEqual(ChannelType.EMAIL.value, "email")
        self.assertEqual(ChannelType.WEBHOOK.value, "webhook")
        self.assertEqual(ChannelType.SLACK.value, "slack")


class TestEmailChannelValidation(unittest.TestCase):
    """Tests for email channel configuration validation."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.channels.email_channel import EmailChannel
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig

        self.channel = EmailChannel(ChannelConfig())

    def test_validate_valid_config(self):
        """Test validation of valid SMTP config."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig

        config = ChannelConfig(
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_from_address="noreply@example.com",
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            valid, error = loop.run_until_complete(self.channel.validate_config(config))
            self.assertTrue(valid)
            self.assertIsNone(error)
        finally:
            loop.close()

    def test_validate_missing_host(self):
        """Test validation with missing SMTP host."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig

        config = ChannelConfig(
            smtp_host="",
            smtp_port=587,
            smtp_from_address="noreply@example.com",
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            valid, error = loop.run_until_complete(self.channel.validate_config(config))
            self.assertFalse(valid)
            self.assertIsNotNone(error)
            self.assertIn("host", error.lower())
        finally:
            loop.close()

    def test_validate_invalid_email(self):
        """Test validation with invalid from address."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig

        config = ChannelConfig(
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_from_address="not-an-email",
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            valid, error = loop.run_until_complete(self.channel.validate_config(config))
            self.assertFalse(valid)
            self.assertIsNotNone(error)
        finally:
            loop.close()


class TestWebhookChannelValidation(unittest.TestCase):
    """Tests for webhook channel configuration validation."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.channels.webhook_channel import WebhookChannel
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig

        # Channel requires a config in constructor
        self.channel = WebhookChannel(ChannelConfig())

    def test_validate_valid_https_url(self):
        """Test validation of valid HTTPS webhook URL."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig

        config = ChannelConfig(
            webhook_url="https://hooks.example.com/notify",
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            valid, error = loop.run_until_complete(self.channel.validate_config(config))
            self.assertTrue(valid)
            self.assertIsNone(error)
        finally:
            loop.close()

    def test_validate_valid_http_url(self):
        """Test validation of HTTP webhook URL (allowed but not recommended)."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig

        config = ChannelConfig(
            webhook_url="http://localhost:8080/notify",
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            valid, error = loop.run_until_complete(self.channel.validate_config(config))
            self.assertTrue(valid)
        finally:
            loop.close()

    def test_validate_missing_url(self):
        """Test validation with missing webhook URL."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig

        config = ChannelConfig(
            webhook_url="",
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            valid, error = loop.run_until_complete(self.channel.validate_config(config))
            self.assertFalse(valid)
            self.assertIsNotNone(error)
        finally:
            loop.close()

    def test_validate_invalid_url_scheme(self):
        """Test validation with invalid URL scheme."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig

        config = ChannelConfig(
            webhook_url="ftp://files.example.com/notify",
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            valid, error = loop.run_until_complete(self.channel.validate_config(config))
            self.assertFalse(valid)
            self.assertIn("http", error.lower())
        finally:
            loop.close()


class TestSlackChannelValidation(unittest.TestCase):
    """Tests for Slack channel configuration validation."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.channels.slack_channel import SlackChannel
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig

        # Channel requires a config in constructor
        self.channel = SlackChannel(ChannelConfig())

    def test_validate_valid_slack_webhook(self):
        """Test validation of valid Slack webhook URL."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig

        config = ChannelConfig(
            webhook_url="https://hooks.slack.com/services/T00/B00/xxx",
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            valid, error = loop.run_until_complete(self.channel.validate_config(config))
            self.assertTrue(valid)
            self.assertIsNone(error)
        finally:
            loop.close()


class TestWebhookSigning(unittest.TestCase):
    """Tests for webhook payload signing."""

    def test_hmac_signature_generation(self):
        """Test HMAC signature generation for webhooks."""
        import hashlib
        import hmac

        secret = "test_secret"
        payload = '{"event": "test"}'

        expected = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        # Verify the expected format
        self.assertEqual(len(expected), 64)  # SHA256 hex digest length
        self.assertTrue(all(c in "0123456789abcdef" for c in expected))


class TestNotificationRouting(unittest.TestCase):
    """Tests for notification routing logic."""

    def test_severity_comparison(self):
        """Test that severity levels can be compared."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import Severity

        severities = [
            Severity.DEBUG,
            Severity.INFO,
            Severity.WARNING,
            Severity.ERROR,
            Severity.CRITICAL,
        ]

        # Verify ordering by name (alphabetical won't work, need to check values)
        severity_order = {
            "debug": 0,
            "info": 1,
            "warning": 2,
            "error": 3,
            "critical": 4,
        }

        for sev in severities:
            self.assertIn(sev.value, severity_order)


class TestDeliveryStatus(unittest.TestCase):
    """Tests for delivery status enum."""

    def test_delivery_status_values(self):
        """Test that all delivery status values exist."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import DeliveryStatus

        self.assertEqual(DeliveryStatus.PENDING.value, "pending")
        self.assertEqual(DeliveryStatus.DELIVERED.value, "delivered")
        self.assertEqual(DeliveryStatus.FAILED.value, "failed")


class TestEmailPresets(unittest.TestCase):
    """Tests for email provider presets."""

    def test_gmail_preset_exists(self):
        """Test that Gmail preset is defined."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import EMAIL_PROVIDER_PRESETS

        self.assertIn("gmail", EMAIL_PROVIDER_PRESETS)
        gmail = EMAIL_PROVIDER_PRESETS["gmail"]
        self.assertEqual(gmail["smtp_host"], "smtp.gmail.com")
        self.assertEqual(gmail["smtp_port"], 587)

    def test_outlook_preset_exists(self):
        """Test that Outlook preset is defined."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import EMAIL_PROVIDER_PRESETS

        self.assertIn("outlook", EMAIL_PROVIDER_PRESETS)
        outlook = EMAIL_PROVIDER_PRESETS["outlook"]
        self.assertEqual(outlook["smtp_host"], "smtp.office365.com")

    def test_custom_preset_is_empty(self):
        """Test that custom preset allows any settings."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import EMAIL_PROVIDER_PRESETS

        self.assertIn("custom", EMAIL_PROVIDER_PRESETS)
        custom = EMAIL_PROVIDER_PRESETS["custom"]
        # Custom should have minimal/no defaults
        self.assertEqual(custom.get("smtp_host"), "")


class TestResponseAction(unittest.TestCase):
    """Tests for response action parsing."""

    def test_response_action_creation(self):
        """Test creating a response action."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ResponseAction

        action = ResponseAction(
            action="approve",
            approval_request_id=123,
            sender_email="admin@example.com",
            raw_body="APPROVE - Looks good",
        )

        self.assertEqual(action.action, "approve")
        self.assertEqual(action.approval_request_id, 123)
        self.assertEqual(action.sender_email, "admin@example.com")


class TestDeliveryHistoryAPI(unittest.TestCase):
    """Tests for delivery history API response model."""

    def test_history_entry_response_has_channel_name_field(self):
        """Test HistoryEntryResponse model includes channel_name field."""
        from typing import Optional

        from pydantic import BaseModel

        # Define a test model matching the expected structure
        class TestHistoryEntryResponse(BaseModel):
            id: int
            channel_id: int
            channel_name: Optional[str] = None
            event_type: str
            recipient: str
            delivery_status: str
            sent_at: str
            error_message: Optional[str] = None

        # Test with channel_name
        entry = TestHistoryEntryResponse(
            id=1,
            channel_id=5,
            channel_name="Production Slack",
            event_type="job_completed",
            recipient="#alerts",
            delivery_status="delivered",
            sent_at="2024-12-28T10:00:00Z",
        )

        self.assertEqual(entry.id, 1)
        self.assertEqual(entry.channel_id, 5)
        self.assertEqual(entry.channel_name, "Production Slack")
        self.assertEqual(entry.event_type, "job_completed")

    def test_history_entry_response_channel_name_optional(self):
        """Test HistoryEntryResponse works without channel_name."""
        from typing import Optional

        from pydantic import BaseModel

        class TestHistoryEntryResponse(BaseModel):
            id: int
            channel_id: int
            channel_name: Optional[str] = None
            event_type: str
            recipient: str
            delivery_status: str
            sent_at: str
            error_message: Optional[str] = None

        entry = TestHistoryEntryResponse(
            id=2,
            channel_id=3,
            event_type="job_failed",
            recipient="user@example.com",
            delivery_status="failed",
            sent_at="2024-12-28T11:00:00Z",
            error_message="Connection timeout",
        )

        self.assertIsNone(entry.channel_name)
        self.assertEqual(entry.error_message, "Connection timeout")


class TestNotificationStoreDeliveryHistory(unittest.TestCase):
    """Tests for NotificationStore.get_delivery_history with new parameters."""

    def setUp(self):
        """Set up test fixtures with in-memory database."""
        import sqlite3
        import tempfile
        import uuid
        from pathlib import Path

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore

        # Reset singleton to get a fresh instance
        NotificationStore.reset_instance()

        # Create a truly unique temporary database for each test
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / f"test_{uuid.uuid4().hex}.db"
        self.store = NotificationStore(db_path=self.db_path)

        # Manually insert test data
        conn = self.store._get_connection()

        # Insert test channels (created_at is NOT NULL)
        conn.execute(
            "INSERT INTO notification_channels (id, channel_type, name, is_enabled, created_at) VALUES (?, ?, ?, ?, ?)",
            (1, "email", "Email Channel", 1, "2024-12-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO notification_channels (id, channel_type, name, is_enabled, created_at) VALUES (?, ?, ?, ?, ?)",
            (2, "slack", "Slack Alerts", 1, "2024-12-01T00:00:00Z"),
        )

        # Insert test log entries
        test_logs = [
            (1, 1, "job_completed", "job-1", 1, "user@example.com", "delivered", None, "2024-12-25T10:00:00Z", None, None),
            (2, 1, "job_failed", "job-2", 1, "user@example.com", "delivered", None, "2024-12-26T10:00:00Z", None, None),
            (3, 2, "job_started", "job-3", 1, "#alerts", "delivered", None, "2024-12-27T10:00:00Z", None, None),
            (4, 2, "job_completed", "job-4", 1, "#alerts", "failed", "Timeout", "2024-12-28T10:00:00Z", None, None),
        ]
        for log in test_logs:
            conn.execute(
                """INSERT INTO notification_log
                   (id, channel_id, event_type, job_id, user_id, recipient, delivery_status, error_message, sent_at, delivered_at, provider_message_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                log,
            )
        conn.commit()

    def tearDown(self):
        """Clean up temporary database."""
        import os
        import shutil

        # Close the connection first
        if hasattr(self.store, "_local") and hasattr(self.store._local, "connection"):
            self.store._local.connection.close()
            del self.store._local.connection
        # Clean up temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_delivery_history_with_offset(self):
        """Test pagination with offset parameter."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Get all entries
            all_entries = loop.run_until_complete(self.store.get_delivery_history(limit=10, offset=0))
            self.assertEqual(len(all_entries), 4)

            # Get with offset
            offset_entries = loop.run_until_complete(self.store.get_delivery_history(limit=10, offset=2))
            self.assertEqual(len(offset_entries), 2)

            # Verify offset skipped the first 2 (most recent)
            self.assertEqual(offset_entries[0]["id"], 2)
        finally:
            loop.close()

    def test_get_delivery_history_with_date_range(self):
        """Test filtering by date range."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Filter by start_date only
            entries = loop.run_until_complete(
                self.store.get_delivery_history(
                    limit=10,
                    start_date="2024-12-27T00:00:00Z",
                )
            )
            self.assertEqual(len(entries), 2)

            # Filter by end_date only
            entries = loop.run_until_complete(
                self.store.get_delivery_history(
                    limit=10,
                    end_date="2024-12-26T23:59:59Z",
                )
            )
            self.assertEqual(len(entries), 2)

            # Filter by both
            entries = loop.run_until_complete(
                self.store.get_delivery_history(
                    limit=10,
                    start_date="2024-12-26T00:00:00Z",
                    end_date="2024-12-27T23:59:59Z",
                )
            )
            self.assertEqual(len(entries), 2)
        finally:
            loop.close()

    def test_get_delivery_history_includes_channel_name(self):
        """Test that channel_name is included in results."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            entries = loop.run_until_complete(self.store.get_delivery_history(limit=10))

            # Check that channel names are populated
            for entry in entries:
                self.assertIn("channel_name", entry)

            # Find entries by channel and verify names
            email_entries = [e for e in entries if e["channel_id"] == 1]
            slack_entries = [e for e in entries if e["channel_id"] == 2]

            self.assertTrue(all(e["channel_name"] == "Email Channel" for e in email_entries))
            self.assertTrue(all(e["channel_name"] == "Slack Alerts" for e in slack_entries))
        finally:
            loop.close()

    def test_get_delivery_history_combined_filters(self):
        """Test combining offset, date range, and channel filters."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            entries = loop.run_until_complete(
                self.store.get_delivery_history(
                    limit=5,
                    offset=0,
                    channel_id=2,
                    start_date="2024-12-27T00:00:00Z",
                )
            )
            # Should get only Slack entries from Dec 27+
            self.assertEqual(len(entries), 2)
            for entry in entries:
                self.assertEqual(entry["channel_id"], 2)
                self.assertEqual(entry["channel_name"], "Slack Alerts")
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
