#!/usr/bin/env python3
"""Tests for TEXT_JSON field type."""

import json
import unittest
from unittest.mock import MagicMock, patch

from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService
from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry
from simpletuner.simpletuner_sdk.server.services.field_registry.types import FieldType


class TestTextJsonConversion(unittest.TestCase):
    """Test convert_value_by_type with TEXT_JSON field type."""

    def test_json_array_string_parses_to_list(self):
        """JSON array string should parse to Python list."""
        result = ConfigsService.convert_value_by_type(
            '["line1", "line2", "line3"]',
            FieldType.TEXT_JSON,
            default_value=None,
        )
        self.assertEqual(result, ["line1", "line2", "line3"])

    def test_json_string_parses_to_string(self):
        """JSON string should parse to Python string."""
        result = ConfigsService.convert_value_by_type(
            '"single string value"',
            FieldType.TEXT_JSON,
            default_value=None,
        )
        self.assertEqual(result, "single string value")

    def test_json_object_parses_to_dict(self):
        """JSON object should parse to Python dict."""
        result = ConfigsService.convert_value_by_type(
            '{"key": "value", "nested": {"a": 1}}',
            FieldType.TEXT_JSON,
            default_value=None,
        )
        self.assertEqual(result, {"key": "value", "nested": {"a": 1}})

    def test_non_string_value_passes_through(self):
        """Non-string values should pass through unchanged."""
        # Already a list
        result = ConfigsService.convert_value_by_type(
            ["already", "a", "list"],
            FieldType.TEXT_JSON,
            default_value=None,
        )
        self.assertEqual(result, ["already", "a", "list"])

        # Already a dict
        result = ConfigsService.convert_value_by_type(
            {"already": "a dict"},
            FieldType.TEXT_JSON,
            default_value=None,
        )
        self.assertEqual(result, {"already": "a dict"})

    def test_empty_string_returns_default(self):
        """Empty string should return default value."""
        result = ConfigsService.convert_value_by_type(
            "",
            FieldType.TEXT_JSON,
            default_value=["default"],
        )
        self.assertEqual(result, ["default"])

        result = ConfigsService.convert_value_by_type(
            "   ",  # whitespace only
            FieldType.TEXT_JSON,
            default_value=None,
        )
        self.assertIsNone(result)

    def test_none_returns_default(self):
        """None should return default value."""
        result = ConfigsService.convert_value_by_type(
            None,
            FieldType.TEXT_JSON,
            default_value=["default"],
        )
        self.assertEqual(result, ["default"])

    def test_invalid_json_returns_raw_string(self):
        """Invalid JSON should gracefully return the raw string."""
        result = ConfigsService.convert_value_by_type(
            "not valid json at all",
            FieldType.TEXT_JSON,
            default_value=None,
        )
        self.assertEqual(result, "not valid json at all")

    def test_json_with_env_placeholder(self):
        """JSON string with env placeholder should parse correctly."""
        # The env placeholder is in the string value, not the JSON structure
        result = ConfigsService.convert_value_by_type(
            '["line1", "{env:MY_VAR}", "line3"]',
            FieldType.TEXT_JSON,
            default_value=None,
        )
        self.assertEqual(result, ["line1", "{env:MY_VAR}", "line3"])


class TestModelspecCommentFieldRegistration(unittest.TestCase):
    """Test that modelspec_comment field is registered with TEXT_JSON type."""

    def test_modelspec_comment_is_text_json(self):
        """modelspec_comment should be registered as TEXT_JSON field type."""
        registry = FieldRegistry()
        field = registry.get_field("modelspec_comment")
        self.assertIsNotNone(field, "modelspec_comment field should be registered")
        self.assertEqual(field.field_type, FieldType.TEXT_JSON)
        self.assertEqual(field.arg_name, "--modelspec_comment")
        self.assertTrue(field.allow_empty)


class TestTextJsonRoundTrip(unittest.TestCase):
    """Test TEXT_JSON values survive the config bundle round-trip."""

    @patch("simpletuner.simpletuner_sdk.server.services.training_service.get_config_store")
    @patch("simpletuner.simpletuner_sdk.server.services.training_service.get_webui_state")
    def test_array_preserved_through_save(self, mock_state, mock_store):
        """Array value should be preserved when saving config."""
        from simpletuner.simpletuner_sdk.server.services.training_service import build_config_bundle

        mock_store_instance = MagicMock()
        mock_store_instance.get_active_config.return_value = "test_config"
        mock_store_instance.load_config.return_value = ({}, {})
        mock_store.return_value = mock_store_instance

        mock_state_instance = MagicMock()
        mock_defaults = MagicMock()
        mock_defaults.configs_dir = "/tmp/configs"
        mock_defaults.output_dir = "/tmp/output"
        mock_state.return_value = (mock_state_instance, mock_defaults)

        # Simulate form data with JSON array string (as it would come from the textarea)
        form_data = {
            "--model_family": "pixart_sigma",
            "--output_dir": "/tmp/output",
            "--model_type": "lora",
            "--optimizer": "adamw_bf16",
            "--data_backend_config": "/tmp/config.json",
            "--modelspec_comment": '["Trigger word: test", "", "Internal info"]',
        }

        bundle = build_config_bundle(form_data)

        # The value should be parsed back to a list in save_config
        self.assertIn("modelspec_comment", bundle.save_config)
        self.assertIsInstance(bundle.save_config["modelspec_comment"], list)
        self.assertEqual(
            bundle.save_config["modelspec_comment"],
            ["Trigger word: test", "", "Internal info"],
        )

    @patch("simpletuner.simpletuner_sdk.server.services.training_service.get_config_store")
    @patch("simpletuner.simpletuner_sdk.server.services.training_service.get_webui_state")
    def test_string_preserved_through_save(self, mock_state, mock_store):
        """String value should be preserved when saving config."""
        from simpletuner.simpletuner_sdk.server.services.training_service import build_config_bundle

        mock_store_instance = MagicMock()
        mock_store_instance.get_active_config.return_value = "test_config"
        mock_store_instance.load_config.return_value = ({}, {})
        mock_store.return_value = mock_store_instance

        mock_state_instance = MagicMock()
        mock_defaults = MagicMock()
        mock_defaults.configs_dir = "/tmp/configs"
        mock_defaults.output_dir = "/tmp/output"
        mock_state.return_value = (mock_state_instance, mock_defaults)

        # Simulate form data with JSON string (as it would come from the textarea)
        form_data = {
            "--model_family": "pixart_sigma",
            "--output_dir": "/tmp/output",
            "--model_type": "lora",
            "--optimizer": "adamw_bf16",
            "--data_backend_config": "/tmp/config.json",
            "--modelspec_comment": '"Single line comment"',
        }

        bundle = build_config_bundle(form_data)

        # The value should be parsed back to a string in save_config
        self.assertIn("modelspec_comment", bundle.save_config)
        self.assertIsInstance(bundle.save_config["modelspec_comment"], str)
        self.assertEqual(bundle.save_config["modelspec_comment"], "Single line comment")

    @patch("simpletuner.simpletuner_sdk.server.services.training_service.get_config_store")
    @patch("simpletuner.simpletuner_sdk.server.services.training_service.get_webui_state")
    def test_existing_array_not_corrupted(self, mock_state, mock_store):
        """Existing array in config should not be corrupted when re-saved unchanged."""
        from simpletuner.simpletuner_sdk.server.services.training_service import build_config_bundle

        original_array = [
            "Trigger word: murderboots",
            "",
            "Internal information:",
            "- CUDA arch list: {env:TORCH_CUDA_ARCH_LIST}",
        ]

        mock_store_instance = MagicMock()
        mock_store_instance.get_active_config.return_value = "test_config"
        # Simulate loading a config that already has modelspec_comment as an array
        mock_store_instance.load_config.return_value = (
            {"modelspec_comment": original_array},
            {},
        )
        mock_store.return_value = mock_store_instance

        mock_state_instance = MagicMock()
        mock_defaults = MagicMock()
        mock_defaults.configs_dir = "/tmp/configs"
        mock_defaults.output_dir = "/tmp/output"
        mock_state.return_value = (mock_state_instance, mock_defaults)

        # Simulate form data where modelspec_comment comes back as JSON string
        # (as it would after being displayed in the textarea and submitted)
        form_data = {
            "--model_family": "pixart_sigma",
            "--output_dir": "/tmp/output",
            "--model_type": "lora",
            "--optimizer": "adamw_bf16",
            "--data_backend_config": "/tmp/config.json",
            "--modelspec_comment": json.dumps(original_array),
        }

        bundle = build_config_bundle(form_data)

        # The array should be preserved exactly
        self.assertIn("modelspec_comment", bundle.save_config)
        self.assertIsInstance(bundle.save_config["modelspec_comment"], list)
        self.assertEqual(bundle.save_config["modelspec_comment"], original_array)


if __name__ == "__main__":
    unittest.main()
