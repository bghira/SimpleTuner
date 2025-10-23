#!/usr/bin/env python3
"""Test that webhook_config JSON parsing handles boolean values correctly."""

import json
import unittest
from unittest.mock import MagicMock, patch

from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args


class TestWebhookJSONParsing(unittest.TestCase):
    """Test webhook_config JSON parsing with various data types."""

    def test_json_booleans_parsed_correctly(self):
        """Test that JSON booleans (true/false) are parsed correctly."""
        webhook_config_json = json.dumps(
            [
                {
                    "webhook_type": "raw",
                    "callback_url": "https://0.0.0.0:8001/callback",
                    "log_level": "info",
                    "ssl_no_verify": True,  # Python bool -> JSON true
                }
            ]
        )

        # Create minimal args list with webhook_config
        args_list = [
            "--model_family=pixart_sigma",
            "--output_dir=/tmp/output",
            "--model_type=lora",
            "--optimizer=adamw_bf16",
            "--data_backend_config=/tmp/config.json",
            f"--webhook_config={webhook_config_json}",
        ]

        # Parse args
        args = parse_cmdline_args(input_args=args_list, exit_on_error=False)

        # Verify webhook_config was parsed correctly
        self.assertIsNotNone(args)
        self.assertIsInstance(args.webhook_config, list)
        self.assertEqual(len(args.webhook_config), 1)
        self.assertEqual(args.webhook_config[0]["webhook_type"], "raw")
        self.assertEqual(args.webhook_config[0]["ssl_no_verify"], True)
        self.assertIsInstance(args.webhook_config[0]["ssl_no_verify"], bool)

    def test_json_with_false_boolean(self):
        """Test that JSON false is parsed correctly."""
        webhook_config_json = json.dumps(
            [
                {
                    "webhook_type": "raw",
                    "callback_url": "https://example.com/callback",
                    "log_level": "info",
                    "ssl_no_verify": False,  # Python bool -> JSON false
                }
            ]
        )

        args_list = [
            "--model_family=pixart_sigma",
            "--output_dir=/tmp/output",
            "--model_type=lora",
            "--optimizer=adamw_bf16",
            "--data_backend_config=/tmp/config.json",
            f"--webhook_config={webhook_config_json}",
        ]

        args = parse_cmdline_args(input_args=args_list, exit_on_error=False)

        self.assertIsNotNone(args)
        self.assertEqual(args.webhook_config[0]["ssl_no_verify"], False)
        self.assertIsInstance(args.webhook_config[0]["ssl_no_verify"], bool)

    def test_json_dict_webhook_config(self):
        """Test that single dict webhook_config is normalized to list."""
        webhook_config_json = json.dumps(
            {
                "webhook_type": "discord",
                "webhook_url": "https://discord.com/webhook",
                "log_level": "debug",
                "ssl_no_verify": True,
            }
        )

        args_list = [
            "--model_family=pixart_sigma",
            "--output_dir=/tmp/output",
            "--model_type=lora",
            "--optimizer=adamw_bf16",
            "--data_backend_config=/tmp/config.json",
            f"--webhook_config={webhook_config_json}",
        ]

        args = parse_cmdline_args(input_args=args_list, exit_on_error=False)

        # Dict should be normalized to list
        self.assertIsInstance(args.webhook_config, list)
        self.assertEqual(len(args.webhook_config), 1)
        self.assertEqual(args.webhook_config[0]["webhook_type"], "discord")

    def test_json_with_multiple_webhooks(self):
        """Test that multiple webhooks in list are preserved."""
        webhook_config_json = json.dumps(
            [
                {
                    "webhook_type": "raw",
                    "callback_url": "https://localhost:8001/callback",
                    "log_level": "info",
                    "ssl_no_verify": True,
                },
                {
                    "webhook_type": "discord",
                    "webhook_url": "https://discord.com/webhook",
                    "log_level": "warning",
                    "ssl_no_verify": False,
                },
            ]
        )

        args_list = [
            "--model_family=pixart_sigma",
            "--output_dir=/tmp/output",
            "--model_type=lora",
            "--optimizer=adamw_bf16",
            "--data_backend_config=/tmp/config.json",
            f"--webhook_config={webhook_config_json}",
        ]

        args = parse_cmdline_args(input_args=args_list, exit_on_error=False)

        self.assertIsInstance(args.webhook_config, list)
        self.assertEqual(len(args.webhook_config), 2)
        self.assertEqual(args.webhook_config[0]["webhook_type"], "raw")
        self.assertEqual(args.webhook_config[1]["webhook_type"], "discord")
        self.assertTrue(args.webhook_config[0]["ssl_no_verify"])
        self.assertFalse(args.webhook_config[1]["ssl_no_verify"])

    def test_json_with_special_characters(self):
        """Test that JSON with special characters in strings is parsed correctly."""
        webhook_config_json = json.dumps(
            [
                {
                    "webhook_type": "raw",
                    "callback_url": "https://localhost:8001/callback?token=abc123&key=xyz",
                    "message_prefix": "Test: [RANK 0]",
                    "log_level": "info",
                    "ssl_no_verify": True,
                }
            ]
        )

        args_list = [
            "--model_family=pixart_sigma",
            "--output_dir=/tmp/output",
            "--model_type=lora",
            "--optimizer=adamw_bf16",
            "--data_backend_config=/tmp/config.json",
            f"--webhook_config={webhook_config_json}",
        ]

        args = parse_cmdline_args(input_args=args_list, exit_on_error=False)

        self.assertEqual(args.webhook_config[0]["callback_url"], "https://localhost:8001/callback?token=abc123&key=xyz")
        self.assertEqual(args.webhook_config[0]["message_prefix"], "Test: [RANK 0]")


if __name__ == "__main__":
    unittest.main()
