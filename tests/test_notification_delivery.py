"""End-to-end tests for notification delivery.

Tests the complete notification flow:
- NotificationService.notify() sends to configured channels
- Webhook channel delivers HTTP POST
- Delivery logging and history
- Preference-based routing
"""

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch


class TestNotificationDeliveryE2E(unittest.TestCase):
    """End-to-end tests for notification delivery."""

    def setUp(self):
        """Set up test fixtures with temp database."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore

        # Reset singleton before each test
        NotificationStore.reset_instance()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "notifications.db"

    def tearDown(self):
        """Clean up temp files."""
        import shutil

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore

        NotificationStore.reset_instance()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_initializes_with_schema(self):
        """Test notification store creates schema on init."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore

        store = NotificationStore(db_path=self.db_path)

        # Verify database file exists
        self.assertTrue(self.db_path.exists())

        # Verify tables exist by trying to query them
        conn = store._get_connection()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [t["name"] for t in tables]

        self.assertIn("notification_channels", table_names)
        self.assertIn("notification_preferences", table_names)
        self.assertIn("notification_log", table_names)

    def test_store_singleton_pattern(self):
        """Test notification store is a singleton per db path."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore

        store1 = NotificationStore(db_path=self.db_path)
        store2 = NotificationStore(db_path=self.db_path)

        self.assertIs(store1, store2)


class TestWebhookChannelDelivery(unittest.TestCase):
    """Test webhook channel delivery."""

    def test_webhook_sends_post_request(self):
        """Test webhook channel sends HTTP POST."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.channels.webhook_channel import WebhookChannel
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig, NotificationEvent
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import (
            ChannelType,
            NotificationEventType,
        )

        config = ChannelConfig(
            id=1,
            name="Test Webhook",
            channel_type=ChannelType.WEBHOOK,
            webhook_url="https://example.com/webhook",
            is_enabled=True,
        )

        channel = WebhookChannel(config)

        event = NotificationEvent(
            event_type=NotificationEventType.JOB_COMPLETED,
            title="Job Complete",
            message="Training complete",
            job_id="job-456",
        )

        # Mock the http client factory (patch at source location)
        with patch("simpletuner.simpletuner_sdk.server.services.cloud.http_client.get_async_client") as mock_get_client:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.status_code = 200

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_response)

            mock_get_client.return_value = mock_client

            result = asyncio.run(channel.send(event, "https://example.com/webhook", {}))

        # Verify success
        self.assertTrue(result.success)
        self.assertEqual(result.channel_id, 1)

        # Verify POST was called
        mock_client.post.assert_called_once()

    def test_webhook_handles_connection_error(self):
        """Test webhook channel handles connection errors gracefully."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.channels.webhook_channel import WebhookChannel
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig, NotificationEvent
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import (
            ChannelType,
            NotificationEventType,
        )

        config = ChannelConfig(
            id=2,
            name="Failing Webhook",
            channel_type=ChannelType.WEBHOOK,
            webhook_url="https://unreachable.example.com/webhook",
            is_enabled=True,
        )

        channel = WebhookChannel(config)

        event = NotificationEvent(
            event_type=NotificationEventType.JOB_FAILED,
            title="Job Failed",
            message="Out of memory",
            job_id="job-789",
        )

        # Mock to raise connection error (patch at source location)
        with patch("simpletuner.simpletuner_sdk.server.services.cloud.http_client.get_async_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))

            mock_get_client.return_value = mock_client

            result = asyncio.run(channel.send(event, "", {}))

        # Should return failure result, not raise exception
        self.assertFalse(result.success)
        self.assertIn("refused", result.error_message.lower())

    def test_webhook_returns_failure_without_url(self):
        """Test webhook fails gracefully without URL configured."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.channels.webhook_channel import WebhookChannel
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig, NotificationEvent
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import (
            ChannelType,
            NotificationEventType,
        )

        config = ChannelConfig(
            id=3,
            name="No URL Webhook",
            channel_type=ChannelType.WEBHOOK,
            # No webhook_url set
            is_enabled=True,
        )

        channel = WebhookChannel(config)

        event = NotificationEvent(
            event_type=NotificationEventType.JOB_COMPLETED,
            title="Test",
            message="Test",
        )

        result = asyncio.run(channel.send(event, "", {}))  # Empty recipient too

        self.assertFalse(result.success)
        self.assertIn("url", result.error_message.lower())


class TestDeliveryLogging(unittest.TestCase):
    """Test notification delivery is logged to history."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore

        NotificationStore.reset_instance()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "notifications.db"

    def tearDown(self):
        """Clean up."""
        import shutil

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore

        NotificationStore.reset_instance()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_delivery_is_logged_to_store(self):
        """Test successful delivery is logged to notification store."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import (
            ChannelType,
            DeliveryStatus,
            NotificationEventType,
        )

        store = NotificationStore(db_path=self.db_path)

        # Create a channel first (FK constraint)
        channel = ChannelConfig(
            name="Test Channel",
            channel_type=ChannelType.WEBHOOK,
            webhook_url="https://example.com/webhook",
        )
        channel_id = asyncio.run(store.create_channel(channel))

        # Log a delivery
        asyncio.run(
            store.log_delivery(
                channel_id=channel_id,
                event_type=NotificationEventType.JOB_COMPLETED,
                recipient="admin@example.com",
                status=DeliveryStatus.DELIVERED,
                job_id="job-001",
                provider_message_id="msg-123",
            )
        )

        # Verify it's in history
        history = asyncio.run(store.get_delivery_history(limit=10))

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["channel_id"], channel_id)
        self.assertEqual(history[0]["job_id"], "job-001")
        self.assertEqual(history[0]["provider_message_id"], "msg-123")

    def test_failed_delivery_is_logged(self):
        """Test failed delivery is logged with error message."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import (
            ChannelType,
            DeliveryStatus,
            NotificationEventType,
        )

        store = NotificationStore(db_path=self.db_path)

        # Create a channel first (FK constraint)
        channel = ChannelConfig(
            name="Test Channel",
            channel_type=ChannelType.WEBHOOK,
            webhook_url="https://example.com/webhook",
        )
        channel_id = asyncio.run(store.create_channel(channel))

        asyncio.run(
            store.log_delivery(
                channel_id=channel_id,
                event_type=NotificationEventType.JOB_FAILED,
                recipient="admin@example.com",
                status=DeliveryStatus.FAILED,
                job_id="job-002",
                error_message="Connection timeout",
            )
        )

        history = asyncio.run(store.get_delivery_history(limit=10, status=DeliveryStatus.FAILED))

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["error_message"], "Connection timeout")

    def test_get_channel_stats(self):
        """Test channel statistics are computed correctly."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import (
            ChannelType,
            DeliveryStatus,
            NotificationEventType,
        )

        store = NotificationStore(db_path=self.db_path)

        # Create a channel first (FK constraint)
        channel = ChannelConfig(
            name="Stats Channel",
            channel_type=ChannelType.WEBHOOK,
            webhook_url="https://example.com/webhook",
        )
        channel_id = asyncio.run(store.create_channel(channel))

        # Log multiple deliveries
        for i in range(5):
            asyncio.run(
                store.log_delivery(
                    channel_id=channel_id,
                    event_type=NotificationEventType.JOB_COMPLETED,
                    recipient=f"user{i}@example.com",
                    status=DeliveryStatus.DELIVERED,
                )
            )

        for i in range(2):
            asyncio.run(
                store.log_delivery(
                    channel_id=channel_id,
                    event_type=NotificationEventType.JOB_FAILED,
                    recipient=f"fail{i}@example.com",
                    status=DeliveryStatus.FAILED,
                    error_message="Test failure",
                )
            )

        stats = asyncio.run(store.get_channel_stats(channel_id=channel_id))

        self.assertEqual(stats["total"], 7)
        self.assertEqual(stats["delivered"], 5)
        self.assertEqual(stats["failed"], 2)


class TestPreferenceBasedRouting(unittest.TestCase):
    """Test notification routing based on user preferences."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore

        NotificationStore.reset_instance()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "notifications.db"

    def tearDown(self):
        """Clean up."""
        import shutil

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore

        NotificationStore.reset_instance()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_preference_can_be_saved_and_retrieved(self):
        """Test preferences can be saved and retrieved from store."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import (
            ChannelConfig,
            NotificationPreference,
        )
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import (
            ChannelType,
            NotificationEventType,
        )

        store = NotificationStore(db_path=self.db_path)

        # Create a channel first (FK constraint)
        channel = ChannelConfig(
            name="Pref Channel",
            channel_type=ChannelType.WEBHOOK,
            webhook_url="https://example.com/webhook",
        )
        channel_id = asyncio.run(store.create_channel(channel))

        # Save a preference
        pref = NotificationPreference(
            event_type=NotificationEventType.JOB_COMPLETED,
            channel_id=channel_id,
            is_enabled=True,
        )

        asyncio.run(store.set_preference(pref))

        # Retrieve preferences
        prefs = asyncio.run(store.get_preferences())

        self.assertEqual(len(prefs), 1)
        self.assertEqual(prefs[0].channel_id, channel_id)
        self.assertEqual(prefs[0].event_type, NotificationEventType.JOB_COMPLETED)
        self.assertTrue(prefs[0].is_enabled)

    def test_disabled_preference_not_enabled(self):
        """Test disabled preference shows as disabled."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import (
            ChannelConfig,
            NotificationPreference,
        )
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import (
            ChannelType,
            NotificationEventType,
        )

        store = NotificationStore(db_path=self.db_path)

        # Create a channel first (FK constraint)
        channel = ChannelConfig(
            name="Disabled Pref Channel",
            channel_type=ChannelType.WEBHOOK,
            webhook_url="https://example.com/webhook",
        )
        channel_id = asyncio.run(store.create_channel(channel))

        # Save a disabled preference
        pref = NotificationPreference(
            event_type=NotificationEventType.JOB_FAILED,
            channel_id=channel_id,
            is_enabled=False,
        )

        asyncio.run(store.set_preference(pref))

        prefs = asyncio.run(store.get_preferences())

        self.assertEqual(len(prefs), 1)
        self.assertFalse(prefs[0].is_enabled)

    def test_get_preferences_for_event_type(self):
        """Test filtering preferences by event type."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import (
            ChannelConfig,
            NotificationPreference,
        )
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import (
            ChannelType,
            NotificationEventType,
        )

        store = NotificationStore(db_path=self.db_path)

        # Create a channel first (required for preference)
        channel = ChannelConfig(
            name="Test Channel",
            channel_type=ChannelType.WEBHOOK,
            webhook_url="https://example.com/webhook",
            is_enabled=True,
        )
        channel_id = asyncio.run(store.create_channel(channel))

        # Create preferences for different event types
        pref1 = NotificationPreference(
            event_type=NotificationEventType.JOB_COMPLETED,
            channel_id=channel_id,
            is_enabled=True,
        )
        pref2 = NotificationPreference(
            event_type=NotificationEventType.JOB_FAILED,
            channel_id=channel_id,
            is_enabled=True,
        )

        asyncio.run(store.set_preference(pref1))
        asyncio.run(store.set_preference(pref2))

        # Get preferences for specific event
        completed_prefs = asyncio.run(store.get_preferences_for_event(NotificationEventType.JOB_COMPLETED))

        self.assertEqual(len(completed_prefs), 1)
        self.assertEqual(completed_prefs[0].event_type, NotificationEventType.JOB_COMPLETED)


class TestChannelManagement(unittest.TestCase):
    """Test channel creation and management."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore

        NotificationStore.reset_instance()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "notifications.db"

    def tearDown(self):
        """Clean up."""
        import shutil

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore

        NotificationStore.reset_instance()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_and_get_channel(self):
        """Test creating and retrieving a channel."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import ChannelType

        store = NotificationStore(db_path=self.db_path)

        channel = ChannelConfig(
            name="Slack Webhook",
            channel_type=ChannelType.SLACK,
            webhook_url="https://hooks.slack.com/test",
            is_enabled=True,
        )

        channel_id = asyncio.run(store.create_channel(channel))
        self.assertGreater(channel_id, 0)

        retrieved = asyncio.run(store.get_channel(channel_id))
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Slack Webhook")
        self.assertEqual(retrieved.webhook_url, "https://hooks.slack.com/test")

    def test_list_enabled_channels(self):
        """Test listing only enabled channels."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import ChannelType

        store = NotificationStore(db_path=self.db_path)

        # Create enabled and disabled channels
        enabled = ChannelConfig(
            name="Enabled",
            channel_type=ChannelType.WEBHOOK,
            webhook_url="https://example.com/enabled",
            is_enabled=True,
        )
        disabled = ChannelConfig(
            name="Disabled",
            channel_type=ChannelType.WEBHOOK,
            webhook_url="https://example.com/disabled",
            is_enabled=False,
        )

        asyncio.run(store.create_channel(enabled))
        asyncio.run(store.create_channel(disabled))

        # Get only enabled
        channels = asyncio.run(store.list_channels(enabled_only=True))
        self.assertEqual(len(channels), 1)
        self.assertEqual(channels[0].name, "Enabled")

        # Get all
        all_channels = asyncio.run(store.list_channels(enabled_only=False))
        self.assertEqual(len(all_channels), 2)

    def test_delete_channel(self):
        """Test deleting a channel."""
        import asyncio

        from simpletuner.simpletuner_sdk.server.services.cloud.notification.models import ChannelConfig
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.notification_store import NotificationStore
        from simpletuner.simpletuner_sdk.server.services.cloud.notification.protocols import ChannelType

        store = NotificationStore(db_path=self.db_path)

        channel = ChannelConfig(
            name="To Delete",
            channel_type=ChannelType.WEBHOOK,
            webhook_url="https://example.com/delete",
        )

        channel_id = asyncio.run(store.create_channel(channel))
        self.assertIsNotNone(asyncio.run(store.get_channel(channel_id)))

        # Delete
        deleted = asyncio.run(store.delete_channel(channel_id))
        self.assertTrue(deleted)

        # Verify gone
        self.assertIsNone(asyncio.run(store.get_channel(channel_id)))


if __name__ == "__main__":
    unittest.main()
