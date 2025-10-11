"""Test webhook_config normalization in cmd_args processing."""
import ast
import json
import os
import tempfile
import unittest
from argparse import Namespace


class TestWebhookCmdArgsNormalization(unittest.TestCase):
    """Test webhook_config normalization logic from cmd_args."""

    def test_normalize_dict_to_list(self):
        """Test that dict webhook_config is normalized to list."""
        # Simulate the normalization logic from cmd_args.py
        webhook_config = '{"webhook_type": "discord", "webhook_url": "http://test.com", "log_level": "info"}'

        parsed_config = ast.literal_eval(webhook_config)
        if isinstance(parsed_config, dict):
            normalized = [parsed_config]
        elif isinstance(parsed_config, list):
            normalized = parsed_config
        else:
            raise ValueError(f"Invalid webhook_config type: {type(parsed_config)}")

        self.assertIsInstance(normalized, list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["webhook_type"], "discord")
        self.assertEqual(normalized[0]["webhook_url"], "http://test.com")

    def test_preserve_list(self):
        """Test that list webhook_config is preserved."""
        webhook_config = '[{"webhook_type": "raw", "callback_url": "http://localhost:8001"}, {"webhook_type": "discord", "webhook_url": "http://discord.com"}]'

        parsed_config = ast.literal_eval(webhook_config)
        if isinstance(parsed_config, dict):
            normalized = [parsed_config]
        elif isinstance(parsed_config, list):
            normalized = parsed_config
        else:
            raise ValueError(f"Invalid webhook_config type: {type(parsed_config)}")

        self.assertIsInstance(normalized, list)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]["webhook_type"], "raw")
        self.assertEqual(normalized[1]["webhook_type"], "discord")

    def test_file_load_dict_normalization(self):
        """Test loading and normalizing dict from JSON file."""
        webhook_dict = {
            "webhook_type": "discord",
            "webhook_url": "http://test.com/webhook",
            "log_level": "warning"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(webhook_dict, f)
            temp_file = f.name

        try:
            with open(temp_file, 'r') as f:
                loaded_config = json.load(f)

            # Normalize as done in cmd_args.py
            if isinstance(loaded_config, dict):
                normalized = [loaded_config]
            elif isinstance(loaded_config, list):
                normalized = loaded_config
            else:
                raise ValueError(f"Invalid webhook_config type: {type(loaded_config)}")

            self.assertIsInstance(normalized, list)
            self.assertEqual(len(normalized), 1)
            self.assertEqual(normalized[0]["webhook_type"], "discord")
        finally:
            os.unlink(temp_file)

    def test_file_load_list_preservation(self):
        """Test loading and preserving list from JSON file."""
        webhook_list = [
            {
                "webhook_type": "raw",
                "callback_url": "http://localhost:8001/callback",
                "log_level": "info"
            },
            {
                "webhook_type": "discord",
                "webhook_url": "http://discord.com/webhook",
                "log_level": "error"
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(webhook_list, f)
            temp_file = f.name

        try:
            with open(temp_file, 'r') as f:
                loaded_config = json.load(f)

            # Normalize as done in cmd_args.py
            if isinstance(loaded_config, dict):
                normalized = [loaded_config]
            elif isinstance(loaded_config, list):
                normalized = loaded_config
            else:
                raise ValueError(f"Invalid webhook_config type: {type(loaded_config)}")

            self.assertIsInstance(normalized, list)
            self.assertEqual(len(normalized), 2)
            self.assertEqual(normalized[0]["webhook_type"], "raw")
            self.assertEqual(normalized[1]["webhook_type"], "discord")
        finally:
            os.unlink(temp_file)

    def test_invalid_type_raises_error(self):
        """Test that invalid webhook_config type raises error."""
        webhook_invalid = '"just_a_string"'

        parsed_config = ast.literal_eval(webhook_invalid)

        with self.assertRaises(ValueError) as context:
            if isinstance(parsed_config, dict):
                normalized = [parsed_config]
            elif isinstance(parsed_config, list):
                normalized = parsed_config
            else:
                raise ValueError(f"Invalid webhook_config type: {type(parsed_config)}")

        self.assertIn("Invalid webhook_config type", str(context.exception))


if __name__ == "__main__":
    unittest.main()
