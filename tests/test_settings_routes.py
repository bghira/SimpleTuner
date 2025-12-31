"""Tests for cloud settings routes.

Tests cover:
- Hints management (status, dismiss, show)
- Local upload configuration
- Data consent settings
- Data upload preview
- Polling status and settings
- System status
- Publishing status
- Credential security settings
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from tests.unittest_support import APITestCase


class TestHintsRoutes(APITestCase, unittest.TestCase):
    """Tests for UI hints management."""

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def _get_client(self):
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        app = create_app(mode=ServerMode.UNIFIED)
        return TestClient(app)

    def test_get_hints_status(self):
        """GET /api/cloud/hints/status should return hint status."""
        with self._get_client() as client:
            response = client.get("/api/cloud/hints/status")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertIn("dataloader_hint_dismissed", data)
            self.assertIn("git_hint_dismissed", data)
            self.assertIsInstance(data["dataloader_hint_dismissed"], bool)
            self.assertIsInstance(data["git_hint_dismissed"], bool)

    def test_get_all_hints(self):
        """GET /api/cloud/hints should return all dismissed hints."""
        with self._get_client() as client:
            response = client.get("/api/cloud/hints")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertIn("dismissed_hints", data)
            self.assertIsInstance(data["dismissed_hints"], list)

    def test_dismiss_dataloader_hint(self):
        """POST /api/cloud/hints/dismiss/dataloader should dismiss hint."""
        with self._get_client() as client:
            response = client.post("/api/cloud/hints/dismiss/dataloader")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertTrue(data["success"])
            self.assertEqual(data["hint"], "dataloader")

            # Verify it's actually dismissed
            status = client.get("/api/cloud/hints/status").json()
            self.assertTrue(status["dataloader_hint_dismissed"])

    def test_dismiss_git_hint(self):
        """POST /api/cloud/hints/dismiss/git should dismiss git hint."""
        with self._get_client() as client:
            response = client.post("/api/cloud/hints/dismiss/git")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertTrue(data["success"])
            self.assertEqual(data["hint"], "git")

    def test_show_dataloader_hint(self):
        """POST /api/cloud/hints/show/dataloader should re-show hint."""
        with self._get_client() as client:
            # First dismiss
            client.post("/api/cloud/hints/dismiss/dataloader")

            # Then show again
            response = client.post("/api/cloud/hints/show/dataloader")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertTrue(data["success"])

            # Verify it's shown again
            status = client.get("/api/cloud/hints/status").json()
            self.assertFalse(status["dataloader_hint_dismissed"])

    def test_dismiss_admin_hint(self):
        """POST should support admin hints with admin_ prefix."""
        with self._get_client() as client:
            response = client.post("/api/cloud/hints/dismiss/admin_overview")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertTrue(data["success"])
            self.assertEqual(data["hint"], "admin_overview")

    def test_dismiss_unknown_hint_error(self):
        """POST with unknown hint should return 400."""
        with self._get_client() as client:
            response = client.post("/api/cloud/hints/dismiss/unknown_hint_type")
            self.assertEqual(response.status_code, 400)
            self.assertIn("Unknown hint", response.json()["detail"])


class TestDataConsentRoutes(APITestCase, unittest.TestCase):
    """Tests for data consent settings."""

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def _get_client(self):
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        app = create_app(mode=ServerMode.UNIFIED)
        return TestClient(app)

    def test_get_data_consent_setting(self):
        """GET /api/cloud/data-consent/setting should return consent mode."""
        with self._get_client() as client:
            response = client.get("/api/cloud/data-consent/setting")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertIn("consent", data)
            self.assertIn(data["consent"], ["ask", "allow", "deny"])

    def test_set_data_consent_allow(self):
        """PUT /api/cloud/data-consent/setting should update to allow."""
        with self._get_client() as client:
            response = client.put(
                "/api/cloud/data-consent/setting",
                json={"consent": "allow"},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertTrue(data["success"])
            self.assertEqual(data["consent"], "allow")

            # Verify setting persisted
            get_response = client.get("/api/cloud/data-consent/setting")
            self.assertEqual(get_response.json()["consent"], "allow")

    def test_set_data_consent_deny(self):
        """PUT should update to deny."""
        with self._get_client() as client:
            response = client.put(
                "/api/cloud/data-consent/setting",
                json={"consent": "deny"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["consent"], "deny")

    def test_set_data_consent_ask(self):
        """PUT should update to ask."""
        with self._get_client() as client:
            response = client.put(
                "/api/cloud/data-consent/setting",
                json={"consent": "ask"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["consent"], "ask")

    def test_set_data_consent_invalid_value(self):
        """PUT with invalid consent value should return 400."""
        with self._get_client() as client:
            response = client.put(
                "/api/cloud/data-consent/setting",
                json={"consent": "invalid"},
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("Invalid consent value", response.json()["detail"])


class TestDataUploadPreview(APITestCase, unittest.TestCase):
    """Tests for data upload preview."""

    def setUp(self):
        super().setUp()
        self._data_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self._data_dir, ignore_errors=True)
        super().tearDown()

    def _get_client(self):
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        app = create_app(mode=ServerMode.UNIFIED)
        return TestClient(app)

    def test_preview_empty_config(self):
        """POST with empty config should indicate no upload needed."""
        with self._get_client() as client:
            response = client.post("/api/cloud/data-consent/preview", json=[])
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertFalse(data["requires_upload"])
            self.assertEqual(len(data["datasets"]), 0)

    def test_preview_local_dataset(self):
        """POST with local dataset should calculate upload requirements."""
        # Create test files - use larger sizes so MB doesn't round to 0.00
        dataset_path = Path(self._data_dir) / "my_dataset"
        dataset_path.mkdir()
        (dataset_path / "image1.png").write_bytes(b"x" * (1024 * 1024))  # 1 MB
        (dataset_path / "image2.png").write_bytes(b"y" * (1024 * 1024))  # 1 MB

        with self._get_client() as client:
            # Ensure consent is set to "ask" (not "deny")
            client.put("/api/cloud/data-consent/setting", json={"consent": "ask"})

            response = client.post(
                "/api/cloud/data-consent/preview",
                json=[
                    {
                        "id": "test-dataset",
                        "type": "local",
                        "instance_data_dir": str(dataset_path),
                    }
                ],
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertTrue(data["requires_upload"])
            self.assertEqual(len(data["datasets"]), 1)
            self.assertEqual(data["datasets"][0]["id"], "test-dataset")
            self.assertEqual(data["datasets"][0]["file_count"], 2)
            self.assertGreater(data["total_size_mb"], 0)

    def test_preview_nonlocal_dataset_skipped(self):
        """POST with non-local datasets should skip them."""
        with self._get_client() as client:
            response = client.post(
                "/api/cloud/data-consent/preview",
                json=[
                    {"id": "aws-dataset", "type": "aws", "bucket": "my-bucket"},
                    {"id": "hf-dataset", "type": "huggingface", "dataset_name": "test"},
                ],
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            # Non-local datasets should not require upload
            self.assertFalse(data["requires_upload"])
            self.assertEqual(len(data["datasets"]), 0)

    def test_preview_when_consent_denied(self):
        """POST should indicate uploads disabled when consent is deny."""
        with self._get_client() as client:
            # First set consent to deny
            client.put("/api/cloud/data-consent/setting", json={"consent": "deny"})

            # Preview should show uploads disabled
            response = client.post(
                "/api/cloud/data-consent/preview",
                json=[{"id": "test", "type": "local", "instance_data_dir": "/tmp"}],
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertFalse(data["requires_upload"])
            self.assertIn("disabled", data["message"].lower())


class TestPollingRoutes(APITestCase, unittest.TestCase):
    """Tests for job polling settings."""

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def _get_client(self):
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        app = create_app(mode=ServerMode.UNIFIED)
        return TestClient(app)

    def test_get_polling_status(self):
        """GET /api/cloud/polling/status should return polling status."""
        with self._get_client() as client:
            response = client.get("/api/cloud/polling/status")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertIn("is_active", data)
            self.assertIn("preference", data)
            self.assertIsInstance(data["is_active"], bool)

    def test_update_polling_setting(self):
        """PUT /api/cloud/polling/setting should update preference."""
        with self._get_client() as client:
            # Enable polling
            response = client.put(
                "/api/cloud/polling/setting",
                json={"enabled": True},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertTrue(data["success"])
            self.assertTrue(data["enabled"])

    def test_disable_polling(self):
        """PUT should disable polling."""
        with self._get_client() as client:
            response = client.put(
                "/api/cloud/polling/setting",
                json={"enabled": False},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertTrue(data["success"])
            self.assertFalse(data["enabled"])


class TestLocalUploadConfig(APITestCase, unittest.TestCase):
    """Tests for local upload configuration."""

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def _get_client(self):
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        app = create_app(mode=ServerMode.UNIFIED)
        return TestClient(app)

    def test_get_local_upload_config_disabled(self):
        """GET should return disabled when no webhook configured."""
        with self._get_client() as client:
            response = client.get("/api/cloud/local-upload/config")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            # Without webhook URL, local upload should be disabled
            self.assertIn("enabled", data)
            self.assertIn("bucket", data)
            self.assertEqual(data["bucket"], "outputs")


class TestSystemStatus(APITestCase, unittest.TestCase):
    """Tests for system status endpoint."""

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def _get_client(self):
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        app = create_app(mode=ServerMode.UNIFIED)
        return TestClient(app)

    def test_get_system_status_structure(self):
        """GET /api/cloud/system-status should return expected structure."""
        with self._get_client() as client:
            response = client.get("/api/cloud/system-status")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            # Verify structure (may have error if network unavailable)
            self.assertIn("operational", data)
            self.assertIn("ongoing_incidents", data)
            self.assertIn("in_progress_maintenances", data)
            self.assertIn("scheduled_maintenances", data)


class TestPublishingStatus(APITestCase, unittest.TestCase):
    """Tests for publishing status endpoint."""

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def _get_client(self):
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        app = create_app(mode=ServerMode.UNIFIED)
        return TestClient(app)

    def test_get_publishing_status(self):
        """GET /api/cloud/publishing-status should return status."""
        with self._get_client() as client:
            response = client.get("/api/cloud/publishing-status")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertIn("push_to_hub", data)
            self.assertIn("hub_model_id", data)
            self.assertIn("s3_configured", data)
            self.assertIn("s3_bucket", data)
            self.assertIn("local_upload_configured", data)

            self.assertIsInstance(data["push_to_hub"], bool)
            self.assertIsInstance(data["s3_configured"], bool)
            self.assertIsInstance(data["local_upload_configured"], bool)


class TestCredentialSecuritySettings(APITestCase, unittest.TestCase):
    """Tests for credential security settings."""

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def _get_client(self):
        from simpletuner.simpletuner_sdk.server import ServerMode, create_app

        app = create_app(mode=ServerMode.UNIFIED)
        return TestClient(app)

    def test_get_credential_settings(self):
        """GET /api/cloud/settings/credentials should return settings."""
        with self._get_client() as client:
            response = client.get("/api/cloud/settings/credentials")
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertIn("stale_threshold_days", data)
            self.assertIn("early_warning_enabled", data)
            self.assertIn("early_warning_percent", data)

            self.assertIsInstance(data["stale_threshold_days"], int)
            self.assertIsInstance(data["early_warning_enabled"], bool)
            self.assertIsInstance(data["early_warning_percent"], int)

    def test_update_credential_settings(self):
        """PUT /api/cloud/settings/credentials should update settings."""
        with self._get_client() as client:
            response = client.put(
                "/api/cloud/settings/credentials",
                json={
                    "stale_threshold_days": 60,
                    "early_warning_enabled": True,
                    "early_warning_percent": 80,
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertEqual(data["stale_threshold_days"], 60)
            self.assertTrue(data["early_warning_enabled"])
            self.assertEqual(data["early_warning_percent"], 80)

    def test_update_credential_settings_partial(self):
        """PUT should allow partial updates."""
        with self._get_client() as client:
            # Update only threshold
            response = client.put(
                "/api/cloud/settings/credentials",
                json={"stale_threshold_days": 45},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertEqual(data["stale_threshold_days"], 45)

    def test_credential_settings_persist(self):
        """Settings should persist across requests."""
        with self._get_client() as client:
            # Set values
            client.put(
                "/api/cloud/settings/credentials",
                json={"stale_threshold_days": 30, "early_warning_enabled": True},
            )

            # Get and verify
            response = client.get("/api/cloud/settings/credentials")
            data = response.json()

            self.assertEqual(data["stale_threshold_days"], 30)
            self.assertTrue(data["early_warning_enabled"])


if __name__ == "__main__":
    unittest.main()
