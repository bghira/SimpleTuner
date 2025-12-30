"""
Unit tests for webhook configuration API routes.

Tests cover:
- GET /api/configs/webhooks/{name}
- POST /api/configs/webhooks
- PUT /api/configs/webhooks/{name}
- DELETE /api/configs/webhooks/{name}
- POST /api/configs/webhooks/validate
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_webui_api import _WebUIBaseTestCase

try:
    from simpletuner.simpletuner_sdk.server.routes import configs as configs_routes
    from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
except ModuleNotFoundError:  # pragma: no cover
    ConfigStore = None  # type: ignore[assignment]
    configs_routes = None  # type: ignore[assignment]


@unittest.skipIf(
    ConfigStore is None or configs_routes is None,
    "Dependencies unavailable",
)
class WebhookRoutesTestCase(_WebUIBaseTestCase, unittest.TestCase):
    """Test suite for webhook configuration API routes."""

    def setUp(self) -> None:
        # Clear ConfigStore singleton cache before parent setup
        self._instances_backup = ConfigStore._instances.copy()
        ConfigStore._instances = {}

        super().setUp()

        # Create config root and webhooks directory
        self.config_root = self.temp_dir / "config"
        self.webhooks_dir = self.config_root / "webhooks"
        self.webhooks_dir.mkdir(parents=True, exist_ok=True)

        # Create a test ConfigStore for the service to use
        self.test_store = ConfigStore(config_dir=self.config_root, config_type="model")

        # Patch _get_store on the CONFIGS_SERVICE instance used by the routes
        self._get_store_patch = patch.object(
            configs_routes.CONFIGS_SERVICE,
            "_get_store",
            return_value=self.test_store,
        )
        self._get_store_patch.start()

    def tearDown(self) -> None:
        self._get_store_patch.stop()
        super().tearDown()
        # Restore ConfigStore singleton instances
        ConfigStore._instances = self._instances_backup

    def _create_webhook_config(self, name: str, config: dict) -> Path:
        """Create a test webhook config file."""
        webhook_file = self.webhooks_dir / f"{name}.json"
        with webhook_file.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
        return webhook_file

    # ------------------------------------------------------------------
    # GET /api/configs/webhooks/{name} tests
    # ------------------------------------------------------------------

    def test_get_webhook_config_returns_config(self) -> None:
        """Test that GET returns existing webhook config."""
        webhook_config = {
            "webhook_type": "discord",
            "webhook_url": "https://discord.com/api/webhooks/123/abc",
            "message_prefix": "Test",
            "log_level": "info",
        }
        self._create_webhook_config("test-webhook", webhook_config)

        response = self.client.get("/api/configs/webhooks/test-webhook")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["name"], "test-webhook")
        self.assertEqual(payload["config"]["webhook_type"], "discord")
        self.assertEqual(payload["config"]["webhook_url"], "https://discord.com/api/webhooks/123/abc")

    def test_get_webhook_config_returns_404_when_not_found(self) -> None:
        """Test that GET returns 404 when webhook doesn't exist."""
        response = self.client.get("/api/configs/webhooks/nonexistent")

        self.assertEqual(response.status_code, 404)

    # ------------------------------------------------------------------
    # POST /api/configs/webhooks tests
    # ------------------------------------------------------------------

    def test_create_webhook_config_success(self) -> None:
        """Test that POST creates new webhook config."""
        payload = {
            "name": "new-webhook",
            "config": {
                "webhook_type": "discord",
                "webhook_url": "https://discord.com/api/webhooks/123/abc",
                "log_level": "info",
            },
        }

        response = self.client.post("/api/configs/webhooks", json=payload)

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["success"])
        self.assertEqual(result["name"], "new-webhook")

        # Verify file was created
        webhook_file = self.webhooks_dir / "new-webhook.json"
        self.assertTrue(webhook_file.exists())

    def test_create_webhook_config_validates_input(self) -> None:
        """Test that POST validates webhook config before creating."""
        payload = {
            "name": "invalid-webhook",
            "config": {
                "webhook_type": "invalid_type",
                "log_level": "info",
            },
        }

        response = self.client.post("/api/configs/webhooks", json=payload)

        self.assertEqual(response.status_code, 400)

    def test_create_discord_webhook_requires_url(self) -> None:
        """Test that POST requires webhook_url for Discord webhooks."""
        payload = {
            "name": "discord-no-url",
            "config": {
                "webhook_type": "discord",
                "log_level": "info",
            },
        }

        response = self.client.post("/api/configs/webhooks", json=payload)

        self.assertEqual(response.status_code, 400)

    def test_create_raw_webhook_requires_callback_url(self) -> None:
        """Test that POST requires callback_url for raw webhooks."""
        payload = {
            "name": "raw-no-url",
            "config": {
                "webhook_type": "raw",
                "log_level": "info",
            },
        }

        response = self.client.post("/api/configs/webhooks", json=payload)

        self.assertEqual(response.status_code, 400)

    # ------------------------------------------------------------------
    # PUT /api/configs/webhooks/{name} tests
    # ------------------------------------------------------------------

    def test_update_webhook_config_success(self) -> None:
        """Test that PUT updates existing webhook config."""
        # Create initial config
        self._create_webhook_config(
            "existing-webhook",
            {"webhook_type": "discord", "webhook_url": "old-url", "log_level": "info"},
        )

        payload = {
            "config": {
                "webhook_type": "raw",
                "callback_url": "https://example.com/new",
                "log_level": "error",
            },
        }

        response = self.client.put("/api/configs/webhooks/existing-webhook", json=payload)

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["success"])

        # Verify file was updated
        webhook_file = self.webhooks_dir / "existing-webhook.json"
        with webhook_file.open("r", encoding="utf-8") as handle:
            saved_config = json.load(handle)

        self.assertEqual(saved_config["webhook_type"], "raw")
        self.assertEqual(saved_config["callback_url"], "https://example.com/new")

    def test_update_webhook_config_validates_input(self) -> None:
        """Test that PUT validates webhook config before updating."""
        self._create_webhook_config(
            "existing-webhook",
            {"webhook_type": "discord", "webhook_url": "url", "log_level": "info"},
        )

        payload = {
            "config": {
                "webhook_type": "invalid",
                "log_level": "info",
            },
        }

        response = self.client.put("/api/configs/webhooks/existing-webhook", json=payload)

        self.assertEqual(response.status_code, 400)

    # ------------------------------------------------------------------
    # DELETE /api/configs/webhooks/{name} tests
    # ------------------------------------------------------------------

    def test_delete_webhook_config_success(self) -> None:
        """Test that DELETE removes webhook config."""
        self._create_webhook_config(
            "to-delete",
            {"webhook_type": "discord", "webhook_url": "url", "log_level": "info"},
        )

        webhook_file = self.webhooks_dir / "to-delete.json"
        self.assertTrue(webhook_file.exists())

        response = self.client.delete("/api/configs/webhooks/to-delete")

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["success"])
        self.assertFalse(webhook_file.exists())

    def test_delete_webhook_config_returns_404_when_not_found(self) -> None:
        """Test that DELETE returns 404 when webhook doesn't exist."""
        response = self.client.delete("/api/configs/webhooks/nonexistent")

        self.assertEqual(response.status_code, 404)

    # ------------------------------------------------------------------
    # POST /api/configs/webhooks/validate tests
    # ------------------------------------------------------------------

    def test_validate_webhook_config_valid(self) -> None:
        """Test that validate endpoint accepts valid config."""
        payload = {
            "config": {
                "webhook_type": "discord",
                "webhook_url": "https://discord.com/api/webhooks/123/abc",
                "log_level": "info",
            },
        }

        response = self.client.post("/api/configs/webhooks/validate", json=payload)

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_webhook_config_invalid(self) -> None:
        """Test that validate endpoint rejects invalid config."""
        payload = {
            "config": {
                "webhook_type": "invalid",
                "log_level": "info",
            },
        }

        response = self.client.post("/api/configs/webhooks/validate", json=payload)

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertFalse(result["valid"])
        self.assertTrue(len(result["errors"]) > 0)

    def test_validate_webhook_config_returns_warnings(self) -> None:
        """Test that validate endpoint returns warnings."""
        payload = {
            "config": {
                "webhook_type": "discord",
                "webhook_url": "https://example.com/not-discord",
                "log_level": "info",
            },
        }

        response = self.client.post("/api/configs/webhooks/validate", json=payload)

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["valid"])  # Warnings don't make it invalid
        self.assertTrue(len(result["warnings"]) > 0)


if __name__ == "__main__":
    unittest.main()
