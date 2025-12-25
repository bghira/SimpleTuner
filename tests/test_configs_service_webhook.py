"""
Unit tests for webhook configuration management in ConfigsService.

Tests cover:
- CRUD operations for webhook configs
- Validation of webhook config structure
- Path handling and security constraints
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
    from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigServiceError, ConfigsService
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    ConfigStore = None  # type: ignore[assignment]
    ConfigsService = None  # type: ignore[assignment]
    ConfigServiceError = None  # type: ignore[assignment]
    _SKIP_REASON = f"Dependencies unavailable: {exc}"
else:
    _SKIP_REASON = ""


@unittest.skipIf(
    ConfigStore is None or ConfigsService is None or ConfigServiceError is None,
    _SKIP_REASON,
)
class ConfigsServiceWebhookTests(unittest.TestCase):
    """Test suite for Webhook configuration management in ConfigsService."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self._instances_backup = ConfigStore._instances.copy()
        ConfigStore._instances = {}
        self.addCleanup(self._restore_config_store_instances)

        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.config_root = Path(self._tempdir.name).resolve()

        # Create isolated config store
        self.model_store = ConfigStore(config_dir=self.config_root, config_type="model")

        # Create webhooks directory
        self.webhooks_dir = self.config_root / "webhooks"
        self.webhooks_dir.mkdir(parents=True, exist_ok=True)

    def _restore_config_store_instances(self) -> None:
        """Restore ConfigStore singleton instances."""
        ConfigStore._instances = self._instances_backup

    def _create_webhook_config(self, name: str, config: dict) -> Path:
        """Create a test webhook config file."""
        webhook_file = self.webhooks_dir / f"{name}.json"
        with webhook_file.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
        return webhook_file

    # ------------------------------------------------------------------
    # get_webhook_config tests
    # ------------------------------------------------------------------

    def test_get_webhook_config_returns_none_when_not_found(self) -> None:
        """Test that get_webhook_config returns None when config doesn't exist."""
        service = ConfigsService()

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.get_webhook_config("nonexistent")

        self.assertIsNone(result)

    def test_get_webhook_config_loads_existing_config(self) -> None:
        """Test that get_webhook_config successfully loads an existing config."""
        webhook_config = {
            "webhook_type": "discord",
            "webhook_url": "https://discord.com/api/webhooks/123/abc",
            "message_prefix": "Test",
            "log_level": "info",
        }
        self._create_webhook_config("test-webhook", webhook_config)

        service = ConfigsService()
        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.get_webhook_config("test-webhook")

        self.assertIsNotNone(result)
        self.assertEqual(result["webhook_type"], "discord")
        self.assertEqual(result["webhook_url"], "https://discord.com/api/webhooks/123/abc")
        self.assertEqual(result["message_prefix"], "Test")
        self.assertEqual(result["log_level"], "info")

    def test_get_webhook_config_handles_raw_type(self) -> None:
        """Test that get_webhook_config handles raw webhook type."""
        webhook_config = {
            "webhook_type": "raw",
            "callback_url": "https://example.com/webhook",
            "log_level": "warning",
        }
        self._create_webhook_config("raw-webhook", webhook_config)

        service = ConfigsService()
        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.get_webhook_config("raw-webhook")

        self.assertIsNotNone(result)
        self.assertEqual(result["webhook_type"], "raw")
        self.assertEqual(result["callback_url"], "https://example.com/webhook")

    # ------------------------------------------------------------------
    # save_webhook_config tests
    # ------------------------------------------------------------------

    def test_save_webhook_config_creates_new_file(self) -> None:
        """Test that save_webhook_config creates a new config file."""
        service = ConfigsService()

        webhook_config = {
            "webhook_type": "discord",
            "webhook_url": "https://discord.com/api/webhooks/123/abc",
            "log_level": "info",
        }

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.save_webhook_config("new-webhook", webhook_config)

        self.assertTrue(result["success"])
        self.assertEqual(result["name"], "new-webhook")
        self.assertIn("webhooks/new-webhook.json", result["path"])

        # Verify file was created
        webhook_file = self.webhooks_dir / "new-webhook.json"
        self.assertTrue(webhook_file.exists())

        with webhook_file.open("r", encoding="utf-8") as handle:
            saved_config = json.load(handle)

        self.assertEqual(saved_config["webhook_type"], "discord")
        self.assertEqual(saved_config["webhook_url"], "https://discord.com/api/webhooks/123/abc")

    def test_save_webhook_config_overwrites_existing(self) -> None:
        """Test that save_webhook_config overwrites existing config."""
        # Create initial config
        self._create_webhook_config(
            "existing-webhook",
            {"webhook_type": "discord", "webhook_url": "old-url", "log_level": "info"},
        )

        service = ConfigsService()
        new_config = {
            "webhook_type": "raw",
            "callback_url": "https://example.com/new",
            "log_level": "error",
        }

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.save_webhook_config("existing-webhook", new_config)

        self.assertTrue(result["success"])

        # Verify file was overwritten
        webhook_file = self.webhooks_dir / "existing-webhook.json"
        with webhook_file.open("r", encoding="utf-8") as handle:
            saved_config = json.load(handle)

        self.assertEqual(saved_config["webhook_type"], "raw")
        self.assertEqual(saved_config["callback_url"], "https://example.com/new")

    def test_save_webhook_config_creates_webhooks_directory(self) -> None:
        """Test that save_webhook_config creates webhooks directory if it doesn't exist."""
        # Remove webhooks directory
        import shutil

        shutil.rmtree(self.webhooks_dir)
        self.assertFalse(self.webhooks_dir.exists())

        service = ConfigsService()
        webhook_config = {
            "webhook_type": "discord",
            "webhook_url": "https://discord.com/api/webhooks/123/abc",
            "log_level": "info",
        }

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.save_webhook_config("new-webhook", webhook_config)

        self.assertTrue(result["success"])
        self.assertTrue(self.webhooks_dir.exists())

    def test_save_webhook_config_rejects_invalid_name(self) -> None:
        """Test that save_webhook_config rejects invalid config names."""
        service = ConfigsService()
        webhook_config = {"webhook_type": "discord", "webhook_url": "url", "log_level": "info"}

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            with self.assertRaises(ConfigServiceError) as ctx:
                service.save_webhook_config("../escape", webhook_config)

        self.assertEqual(ctx.exception.status_code, 400)

    # ------------------------------------------------------------------
    # delete_webhook_config tests
    # ------------------------------------------------------------------

    def test_delete_webhook_config_removes_file(self) -> None:
        """Test that delete_webhook_config removes the config file."""
        self._create_webhook_config(
            "to-delete",
            {"webhook_type": "discord", "webhook_url": "url", "log_level": "info"},
        )

        webhook_file = self.webhooks_dir / "to-delete.json"
        self.assertTrue(webhook_file.exists())

        service = ConfigsService()
        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.delete_webhook_config("to-delete")

        self.assertTrue(result["success"])
        self.assertFalse(webhook_file.exists())

    def test_delete_webhook_config_raises_error_when_not_found(self) -> None:
        """Test that delete_webhook_config raises error when config doesn't exist."""
        service = ConfigsService()

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            with self.assertRaises(ConfigServiceError) as ctx:
                service.delete_webhook_config("nonexistent")

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("not found", ctx.exception.message)

    # ------------------------------------------------------------------
    # validate_webhook_config tests
    # ------------------------------------------------------------------

    def test_validate_webhook_config_accepts_valid_discord_config(self) -> None:
        """Test that validate_webhook_config accepts valid Discord config."""
        service = ConfigsService()

        webhook_config = {
            "webhook_type": "discord",
            "webhook_url": "https://discord.com/api/webhooks/123/abc",
            "message_prefix": "Training",
            "log_level": "info",
        }

        result = service.validate_webhook_config(webhook_config)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_webhook_config_accepts_valid_raw_config(self) -> None:
        """Test that validate_webhook_config accepts valid raw webhook config."""
        service = ConfigsService()

        webhook_config = {
            "webhook_type": "raw",
            "callback_url": "https://example.com/webhook",
            "log_level": "warning",
        }

        result = service.validate_webhook_config(webhook_config)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_webhook_config_rejects_missing_webhook_type(self) -> None:
        """Test that validate_webhook_config rejects config without webhook_type."""
        service = ConfigsService()

        webhook_config = {
            "webhook_url": "https://discord.com/api/webhooks/123/abc",
            "log_level": "info",
        }

        result = service.validate_webhook_config(webhook_config)

        self.assertFalse(result["valid"])
        self.assertIn("Missing required field: 'webhook_type'", result["errors"])

    def test_validate_webhook_config_rejects_invalid_webhook_type(self) -> None:
        """Test that validate_webhook_config rejects invalid webhook_type."""
        service = ConfigsService()

        webhook_config = {
            "webhook_type": "invalid",
            "webhook_url": "https://example.com",
            "log_level": "info",
        }

        result = service.validate_webhook_config(webhook_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("Invalid webhook_type" in e for e in result["errors"]))

    def test_validate_webhook_config_rejects_missing_discord_url(self) -> None:
        """Test that validate_webhook_config rejects Discord config without webhook_url."""
        service = ConfigsService()

        webhook_config = {
            "webhook_type": "discord",
            "log_level": "info",
        }

        result = service.validate_webhook_config(webhook_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("webhook_url" in e for e in result["errors"]))

    def test_validate_webhook_config_rejects_missing_raw_callback_url(self) -> None:
        """Test that validate_webhook_config rejects raw config without callback_url."""
        service = ConfigsService()

        webhook_config = {
            "webhook_type": "raw",
            "log_level": "info",
        }

        result = service.validate_webhook_config(webhook_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("callback_url" in e for e in result["errors"]))

    def test_validate_webhook_config_rejects_invalid_log_level(self) -> None:
        """Test that validate_webhook_config rejects invalid log_level."""
        service = ConfigsService()

        webhook_config = {
            "webhook_type": "discord",
            "webhook_url": "https://discord.com/api/webhooks/123/abc",
            "log_level": "invalid_level",
        }

        result = service.validate_webhook_config(webhook_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("Invalid log_level" in e for e in result["errors"]))

    def test_validate_webhook_config_warns_on_non_discord_url(self) -> None:
        """Test that validate_webhook_config warns when Discord URL doesn't match pattern."""
        service = ConfigsService()

        webhook_config = {
            "webhook_type": "discord",
            "webhook_url": "https://example.com/not-discord",
            "log_level": "info",
        }

        result = service.validate_webhook_config(webhook_config)

        self.assertTrue(result["valid"])  # Warning, not error
        self.assertTrue(any("discord.com/api/webhooks" in w for w in result["warnings"]))

    def test_validate_webhook_config_accepts_all_log_levels(self) -> None:
        """Test that validate_webhook_config accepts all valid log levels."""
        service = ConfigsService()

        valid_levels = ["debug", "info", "warning", "error", "critical"]

        for level in valid_levels:
            webhook_config = {
                "webhook_type": "discord",
                "webhook_url": "https://discord.com/api/webhooks/123/abc",
                "log_level": level,
            }

            result = service.validate_webhook_config(webhook_config)
            self.assertTrue(result["valid"], f"log_level '{level}' should be valid")


@unittest.skipIf(
    ConfigStore is None or ConfigsService is None or ConfigServiceError is None,
    _SKIP_REASON,
)
class ConfigStoreWebhookListTests(unittest.TestCase):
    """Test suite for webhook listing in ConfigStore."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self._instances_backup = ConfigStore._instances.copy()
        ConfigStore._instances = {}
        self.addCleanup(self._restore_config_store_instances)

        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.config_root = Path(self._tempdir.name).resolve()

        # Create webhooks directory
        self.webhooks_dir = self.config_root / "webhooks"
        self.webhooks_dir.mkdir(parents=True, exist_ok=True)

    def _restore_config_store_instances(self) -> None:
        """Restore ConfigStore singleton instances."""
        ConfigStore._instances = self._instances_backup

    def _create_webhook_config(self, name: str, config: dict) -> Path:
        """Create a test webhook config file."""
        webhook_file = self.webhooks_dir / f"{name}.json"
        with webhook_file.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
        return webhook_file

    def test_list_configs_returns_empty_when_no_webhooks(self) -> None:
        """Test that list_configs returns empty list when no webhooks exist."""
        store = ConfigStore(config_dir=self.config_root, config_type="webhook")
        configs = store.list_configs()

        self.assertEqual(len(configs), 0)

    def test_list_configs_finds_webhook_files(self) -> None:
        """Test that list_configs finds webhook config files."""
        self._create_webhook_config(
            "webhook1",
            {"webhook_type": "discord", "webhook_url": "url1", "log_level": "info"},
        )
        self._create_webhook_config(
            "webhook2",
            {"webhook_type": "raw", "callback_url": "url2", "log_level": "error"},
        )

        store = ConfigStore(config_dir=self.config_root, config_type="webhook")
        configs = store.list_configs()

        self.assertEqual(len(configs), 2)
        names = [c["name"] for c in configs]
        self.assertIn("webhook1", names)
        self.assertIn("webhook2", names)

    def test_list_configs_includes_metadata(self) -> None:
        """Test that list_configs includes webhook metadata."""
        self._create_webhook_config(
            "test-webhook",
            {
                "webhook_type": "discord",
                "webhook_url": "https://discord.com/api/webhooks/123/abc",
                "message_prefix": "Test",
                "log_level": "warning",
            },
        )

        store = ConfigStore(config_dir=self.config_root, config_type="webhook")
        configs = store.list_configs()

        self.assertEqual(len(configs), 1)
        config = configs[0]

        self.assertEqual(config["name"], "test-webhook")
        self.assertEqual(config["webhook_type"], "discord")
        self.assertEqual(config["log_level"], "warning")
        self.assertIn("created_at", config)
        self.assertIn("modified_at", config)

    def test_list_configs_skips_invalid_json(self) -> None:
        """Test that list_configs skips files with invalid JSON."""
        # Create valid webhook
        self._create_webhook_config(
            "valid-webhook",
            {"webhook_type": "discord", "webhook_url": "url", "log_level": "info"},
        )

        # Create invalid JSON file
        invalid_file = self.webhooks_dir / "invalid.json"
        with invalid_file.open("w", encoding="utf-8") as handle:
            handle.write("{ invalid json }")

        store = ConfigStore(config_dir=self.config_root, config_type="webhook")
        configs = store.list_configs()

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["name"], "valid-webhook")

    def test_list_configs_skips_files_without_webhook_type(self) -> None:
        """Test that list_configs skips files without webhook_type key."""
        # Create valid webhook
        self._create_webhook_config(
            "valid-webhook",
            {"webhook_type": "discord", "webhook_url": "url", "log_level": "info"},
        )

        # Create file without webhook_type
        other_file = self.webhooks_dir / "other.json"
        with other_file.open("w", encoding="utf-8") as handle:
            json.dump({"some_key": "some_value"}, handle)

        store = ConfigStore(config_dir=self.config_root, config_type="webhook")
        configs = store.list_configs()

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["name"], "valid-webhook")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
