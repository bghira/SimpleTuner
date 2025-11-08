#!/usr/bin/env python3
"""Test that invalid fields are not saved to config files."""

import unittest
from unittest.mock import MagicMock, patch

from simpletuner.simpletuner_sdk.server.services.training_service import build_config_bundle


class TestConfigFieldValidation(unittest.TestCase):
    """Test that only valid registry fields are saved to config."""

    @patch("simpletuner.simpletuner_sdk.server.services.training_service.get_config_store")
    @patch("simpletuner.simpletuner_sdk.server.services.training_service.get_webui_state")
    def test_invalid_fields_not_in_save_config(self, mock_state, mock_store):
        """Test that UI-only fields are excluded from save_config."""
        # Mock the store and state
        mock_store_instance = MagicMock()
        mock_store_instance.get_active_config.return_value = "test_config"
        mock_store_instance.load_config.return_value = ({}, {})
        mock_store.return_value = mock_store_instance

        mock_state_instance = MagicMock()
        mock_defaults = MagicMock()
        mock_defaults.configs_dir = "/tmp/configs"
        mock_defaults.output_dir = "/tmp/output"
        mock_state.return_value = (mock_state_instance, mock_defaults)

        # Create form data with both valid and invalid fields
        form_data = {
            "--model_family": "pixart_sigma",
            "--output_dir": "/tmp/output",
            "--model_type": "lora",
            "--optimizer": "adamw_bf16",
            "--data_backend_config": "/tmp/config.json",
            # Invalid fields that should be filtered
            "currentDataset.resolution_type": "pixel_area",
            "datasets_page_data_backend_config": "some_config.json",
            "--currentDataset.cache_dir": "cache",
        }

        bundle = build_config_bundle(form_data)

        # Check that invalid fields are NOT in save_config
        self.assertNotIn("currentDataset.resolution_type", bundle.save_config)
        self.assertNotIn("datasets_page_data_backend_config", bundle.save_config)
        self.assertNotIn("currentDataset.cache_dir", bundle.save_config)

        # Check that valid fields ARE in save_config
        self.assertIn("model_family", bundle.save_config)
        self.assertIn("output_dir", bundle.save_config)
        self.assertIn("model_type", bundle.save_config)
        self.assertIn("optimizer", bundle.save_config)

    @patch("simpletuner.simpletuner_sdk.server.services.training_service.get_config_store")
    @patch("simpletuner.simpletuner_sdk.server.services.training_service.get_webui_state")
    def test_webhook_config_preserved_in_save(self, mock_state, mock_store):
        """Test that webhook_config is properly preserved as a list."""
        # Mock the store and state
        mock_store_instance = MagicMock()
        mock_store_instance.get_active_config.return_value = "test_config"
        mock_store_instance.load_config.return_value = ({}, {})
        mock_store.return_value = mock_store_instance

        mock_state_instance = MagicMock()
        mock_defaults = MagicMock()
        mock_defaults.configs_dir = "/tmp/configs"
        mock_defaults.output_dir = "/tmp/output"
        mock_state.return_value = (mock_state_instance, mock_defaults)

        # Create form data
        form_data = {
            "--model_family": "pixart_sigma",
            "--output_dir": "/tmp/output",
            "--model_type": "lora",
            "--optimizer": "adamw_bf16",
            "--data_backend_config": "/tmp/config.json",
        }

        bundle = build_config_bundle(form_data)

        # webhook_config should be in save_config as a list
        self.assertIn("webhook_config", bundle.save_config)
        self.assertIsInstance(bundle.save_config["webhook_config"], list)
        self.assertEqual(len(bundle.save_config["webhook_config"]), 1)
        self.assertEqual(bundle.save_config["webhook_config"][0]["webhook_type"], "raw")

    @patch("simpletuner.simpletuner_sdk.server.services.training_service.get_config_store")
    @patch("simpletuner.simpletuner_sdk.server.services.training_service.get_webui_state")
    def test_ssl_no_verify_true_for_localhost_https(self, mock_state, mock_store):
        """Test that ssl_no_verify is True for localhost HTTPS webhooks."""
        # Mock the store and state
        mock_store_instance = MagicMock()
        mock_store_instance.get_active_config.return_value = "test_config"
        mock_store_instance.load_config.return_value = ({}, {})
        mock_store.return_value = mock_store_instance

        mock_state_instance = MagicMock()
        mock_defaults = MagicMock()
        mock_defaults.configs_dir = "/tmp/configs"
        mock_defaults.output_dir = "/tmp/output"
        mock_state.return_value = (mock_state_instance, mock_defaults)

        # Create form data
        form_data = {
            "--model_family": "pixart_sigma",
            "--output_dir": "/tmp/output",
            "--model_type": "lora",
            "--optimizer": "adamw_bf16",
            "--data_backend_config": "/tmp/config.json",
        }

        import os

        # Test with SSL enabled
        # Need to reload the module to pick up new env vars
        with patch.dict(os.environ, {"SIMPLETUNER_SSL_ENABLED": "true"}, clear=False):
            # Force re-evaluation of DEFAULT_WEBHOOK_CONFIG
            from importlib import reload

            from simpletuner.simpletuner_sdk.server.services import webhook_defaults

            reload(webhook_defaults)

            # Patch the DEFAULT_WEBHOOK_CONFIG in training_service
            with patch(
                "simpletuner.simpletuner_sdk.server.services.training_service.DEFAULT_WEBHOOK_CONFIG",
                webhook_defaults.get_default_webhook_config(),
            ):
                bundle = build_config_bundle(form_data)

                # webhook_config should have ssl_no_verify=True for localhost HTTPS
                webhook_config = bundle.save_config.get("webhook_config")
                self.assertIsNotNone(webhook_config)
                self.assertTrue(webhook_config[0].get("ssl_no_verify"))
                self.assertTrue(webhook_config[0]["callback_url"].startswith("https://"))


class TestValidationPreviewField(unittest.TestCase):
    """Ensure the validation_preview field is registered correctly."""

    def test_validation_preview_field_metadata(self):
        from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry
        from simpletuner.simpletuner_sdk.server.services.field_registry.types import FieldType

        registry = FieldRegistry()
        field = registry.get_field("validation_preview")
        self.assertIsNotNone(field, "validation_preview field should be registered")
        self.assertEqual(field.arg_name, "--validation_preview")
        self.assertEqual(field.field_type, FieldType.CHECKBOX)
        self.assertFalse(field.default_value, "validation_preview should default to False")
        interval_field = registry.get_field("validation_preview_steps")
        self.assertIsNotNone(interval_field, "validation_preview_steps field should be registered")
        self.assertEqual(interval_field.arg_name, "--validation_preview_steps")
        self.assertEqual(interval_field.field_type, FieldType.NUMBER)
        self.assertEqual(interval_field.default_value, 1)


if __name__ == "__main__":
    unittest.main()
