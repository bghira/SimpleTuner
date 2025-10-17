#!/usr/bin/env python3
"""Test that WebUI-specific fields are properly filtered from training args."""

import json
import unittest

from simpletuner.helpers.configuration.cli_utils import mapping_to_cli_args
from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService


class TestWebUIArgsFiltering(unittest.TestCase):
    """Test filtering of WebUI-only fields and webhook_config handling."""

    def test_filters_alpine_ui_variables(self):
        """Test that Alpine.js UI state variables are filtered out."""
        form_data = {
            "--model_family": "pixart_sigma",
            "--output_dir": "/tmp/output",
            "currentDataset.resolution_type": "pixel_area",
            "--currentDataset.cache_dir": "cache",
            "datasets_page_data_backend_config": "some_config.json",
            "--datasets_page_active": "true",
        }

        config_dict = ConfigsService.normalize_form_to_config(form_data)

        # Alpine.js variables should be filtered out
        self.assertNotIn("--currentDataset.resolution_type", config_dict)
        self.assertNotIn("--currentDataset.cache_dir", config_dict)
        self.assertNotIn("--datasets_page_data_backend_config", config_dict)
        self.assertNotIn("--datasets_page_active", config_dict)

        # Real config fields should remain
        self.assertIn("--model_family", config_dict)
        self.assertIn("--output_dir", config_dict)

    def test_webhook_config_list_serialization(self):
        """Test that webhook_config list is properly JSON-serialized."""
        webhook_config = [
            {
                "webhook_type": "raw",
                "callback_url": "https://0.0.0.0:8001/callback",
                "log_level": "info",
                "ssl_no_verify": False,
            }
        ]

        config = {
            "--model_family": "pixart_sigma",
            "--webhook_config": webhook_config,
        }

        cli_args = mapping_to_cli_args(config)

        # Find the webhook_config arg
        webhook_arg = None
        for arg in cli_args:
            if arg.startswith("--webhook_config="):
                webhook_arg = arg
                break

        self.assertIsNotNone(webhook_arg, "webhook_config argument should be present")

        # Extract the value part
        webhook_value = webhook_arg.split("=", 1)[1]

        # Should be valid JSON (not Python dict representation)
        parsed = json.loads(webhook_value)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["webhook_type"], "raw")
        self.assertEqual(parsed[0]["callback_url"], "https://0.0.0.0:8001/callback")

    def test_webhook_config_dict_serialization(self):
        """Test that webhook_config dict is properly JSON-serialized."""
        webhook_config = {
            "webhook_type": "raw",
            "callback_url": "https://localhost:8001/callback",
            "log_level": "info",
        }

        config = {
            "--model_family": "pixart_sigma",
            "--webhook_config": webhook_config,
        }

        cli_args = mapping_to_cli_args(config)

        # Find the webhook_config arg
        webhook_arg = None
        for arg in cli_args:
            if arg.startswith("--webhook_config="):
                webhook_arg = arg
                break

        self.assertIsNotNone(webhook_arg, "webhook_config argument should be present")

        # Extract the value part
        webhook_value = webhook_arg.split("=", 1)[1]

        # Should be valid JSON (not Python dict representation)
        parsed = json.loads(webhook_value)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["webhook_type"], "raw")

    def test_error_scenario_from_logs(self):
        """Test the exact scenario that caused the error in the logs."""
        # Simulate the form data that caused the issue
        form_data = {
            "--model_family": "pixart_sigma",
            "--output_dir": "/tmp/output",
            "--model_type": "lora",
            "--optimizer": "adamw_bf16",
            "--data_backend_config": "/tmp/config.json",
            "currentDataset.resolution_type": "pixel_area",
            "datasets_page_data_backend_config": "krypton-harbor/multidatabackend.json",
            "--webhook_config": [
                {
                    "webhook_type": "raw",
                    "callback_url": "https://0.0.0.0:8001/callback",
                    "log_level": "info",
                    "ssl_no_verify": False,
                }
            ],
        }

        # First pass through normalize_form_to_config
        config_dict = ConfigsService.normalize_form_to_config(form_data)

        # Alpine variables should be gone
        self.assertNotIn("--currentDataset.resolution_type", config_dict)
        self.assertNotIn("--datasets_page_data_backend_config", config_dict)

        # Required fields should be present
        self.assertIn("--model_family", config_dict)
        self.assertIn("--output_dir", config_dict)
        self.assertIn("--model_type", config_dict)
        self.assertIn("--optimizer", config_dict)
        self.assertIn("--data_backend_config", config_dict)

        # Now convert to CLI args
        cli_args = mapping_to_cli_args(config_dict)

        # webhook_config should be properly serialized
        webhook_args = [arg for arg in cli_args if arg.startswith("--webhook_config=")]
        self.assertEqual(len(webhook_args), 1, "Should have exactly one webhook_config arg")

        webhook_value = webhook_args[0].split("=", 1)[1]

        # Should be valid JSON
        parsed = json.loads(webhook_value)
        self.assertIsInstance(parsed, list)
        self.assertEqual(parsed[0]["callback_url"], "https://0.0.0.0:8001/callback")

        # All required args should be present
        arg_keys = set()
        for arg in cli_args:
            if "=" in arg:
                key = arg.split("=", 1)[0]
                arg_keys.add(key)
            else:
                arg_keys.add(arg)

        self.assertIn("--model_family", arg_keys)
        self.assertIn("--output_dir", arg_keys)
        self.assertIn("--model_type", arg_keys)
        self.assertIn("--optimizer", arg_keys)
        self.assertIn("--data_backend_config", arg_keys)


if __name__ == "__main__":
    unittest.main()
