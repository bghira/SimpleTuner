"""Test webhook_config file loading in Trainer.configure_webhook()."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from simpletuner.helpers.training.trainer import Trainer


class TestTrainerWebhookFileLoading(unittest.TestCase):
    """Test that Trainer.configure_webhook() properly loads webhook_config from files."""

    def test_configure_webhook_with_file_path(self):
        """Test that configure_webhook loads webhook_config from a file path."""
        webhook_dict = {"webhook_type": "discord", "webhook_url": "http://test.com/webhook", "log_level": "info"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(webhook_dict, f)
            temp_file = f.name

        try:
            # Create a minimal Trainer instance
            with patch("simpletuner.helpers.training.trainer.load_config") as mock_load_config:
                # Mock the config to avoid full initialization
                mock_config = MagicMock()
                mock_config.webhook_config = None
                mock_load_config.return_value = mock_config

                trainer = Trainer.__new__(Trainer)
                trainer.accelerator = None
                trainer.webhook_handler = None
                trainer.config = None
                trainer.model = None
                trainer.job_id = "test_job"

                # Test configure_webhook with file path
                raw_config = {"webhook_config": temp_file}

                with (
                    patch("simpletuner.helpers.webhooks.handler.WebhookHandler") as mock_webhook_handler,
                    patch.object(trainer, "_infer_send_video_flag", return_value=False),
                    patch.object(trainer, "_infer_video_framerate", return_value=30),
                    patch.object(trainer, "_emit_event"),
                ):

                    trainer.configure_webhook(raw_config=raw_config, send_startup_message=False)

                    # Verify WebhookHandler was called with a list
                    self.assertTrue(mock_webhook_handler.called)
                    call_args = mock_webhook_handler.call_args
                    webhook_config_arg = call_args[1]["webhook_config"]

                    # Should be normalized to a list
                    self.assertIsInstance(webhook_config_arg, list)
                    self.assertEqual(len(webhook_config_arg), 1)
                    self.assertEqual(webhook_config_arg[0], webhook_dict)
        finally:
            os.unlink(temp_file)

    def test_configure_webhook_with_file_path_list(self):
        """Test that configure_webhook loads webhook_config list from a file path."""
        webhook_list = [
            {"webhook_type": "raw", "callback_url": "http://localhost:8001/callback", "log_level": "info"},
            {"webhook_type": "discord", "webhook_url": "http://discord.com/webhook", "log_level": "error"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(webhook_list, f)
            temp_file = f.name

        try:
            with patch("simpletuner.helpers.training.trainer.load_config") as mock_load_config:
                mock_config = MagicMock()
                mock_config.webhook_config = None
                mock_load_config.return_value = mock_config

                trainer = Trainer.__new__(Trainer)
                trainer.accelerator = None
                trainer.webhook_handler = None
                trainer.config = None
                trainer.model = None
                trainer.job_id = "test_job"

                raw_config = {"webhook_config": temp_file}

                with (
                    patch("simpletuner.helpers.webhooks.handler.WebhookHandler") as mock_webhook_handler,
                    patch.object(trainer, "_infer_send_video_flag", return_value=False),
                    patch.object(trainer, "_infer_video_framerate", return_value=30),
                    patch.object(trainer, "_emit_event"),
                ):

                    trainer.configure_webhook(raw_config=raw_config, send_startup_message=False)

                    self.assertTrue(mock_webhook_handler.called)
                    call_args = mock_webhook_handler.call_args
                    webhook_config_arg = call_args[1]["webhook_config"]

                    # Should preserve the list
                    self.assertIsInstance(webhook_config_arg, list)
                    self.assertEqual(len(webhook_config_arg), 2)
                    self.assertEqual(webhook_config_arg, webhook_list)
        finally:
            os.unlink(temp_file)

    def test_configure_webhook_with_json_string(self):
        """Test that configure_webhook parses JSON string."""
        webhook_dict = {"webhook_type": "discord", "webhook_url": "http://test.com/webhook", "log_level": "info"}
        json_string = json.dumps(webhook_dict)

        with patch("simpletuner.helpers.training.trainer.load_config") as mock_load_config:
            mock_config = MagicMock()
            mock_config.webhook_config = None
            mock_load_config.return_value = mock_config

            trainer = Trainer.__new__(Trainer)
            trainer.accelerator = None
            trainer.webhook_handler = None
            trainer.config = None
            trainer.model = None
            trainer.job_id = "test_job"

            raw_config = {"webhook_config": json_string}

            with (
                patch("simpletuner.helpers.webhooks.handler.WebhookHandler") as mock_webhook_handler,
                patch.object(trainer, "_infer_send_video_flag", return_value=False),
                patch.object(trainer, "_infer_video_framerate", return_value=30),
                patch.object(trainer, "_emit_event"),
            ):

                trainer.configure_webhook(raw_config=raw_config, send_startup_message=False)

                self.assertTrue(mock_webhook_handler.called)
                call_args = mock_webhook_handler.call_args
                webhook_config_arg = call_args[1]["webhook_config"]

                # Should be normalized to a list
                self.assertIsInstance(webhook_config_arg, list)
                self.assertEqual(len(webhook_config_arg), 1)
                self.assertEqual(webhook_config_arg[0], webhook_dict)

    def test_configure_webhook_with_dict(self):
        """Test that configure_webhook normalizes dict to list."""
        webhook_dict = {"webhook_type": "discord", "webhook_url": "http://test.com/webhook", "log_level": "info"}

        with patch("simpletuner.helpers.training.trainer.load_config") as mock_load_config:
            mock_config = MagicMock()
            mock_config.webhook_config = None
            mock_load_config.return_value = mock_config

            trainer = Trainer.__new__(Trainer)
            trainer.accelerator = None
            trainer.webhook_handler = None
            trainer.config = None
            trainer.model = None
            trainer.job_id = "test_job"

            raw_config = {"webhook_config": webhook_dict}

            with (
                patch("simpletuner.helpers.webhooks.handler.WebhookHandler") as mock_webhook_handler,
                patch.object(trainer, "_infer_send_video_flag", return_value=False),
                patch.object(trainer, "_infer_video_framerate", return_value=30),
                patch.object(trainer, "_emit_event"),
            ):

                trainer.configure_webhook(raw_config=raw_config, send_startup_message=False)

                self.assertTrue(mock_webhook_handler.called)
                call_args = mock_webhook_handler.call_args
                webhook_config_arg = call_args[1]["webhook_config"]

                # Should be normalized to a list
                self.assertIsInstance(webhook_config_arg, list)
                self.assertEqual(len(webhook_config_arg), 1)
                self.assertEqual(webhook_config_arg[0], webhook_dict)

    def test_configure_webhook_with_invalid_file_path(self):
        """Test that configure_webhook raises error for invalid file path."""
        with patch("simpletuner.helpers.training.trainer.load_config") as mock_load_config:
            mock_config = MagicMock()
            mock_config.webhook_config = None
            mock_load_config.return_value = mock_config

            trainer = Trainer.__new__(Trainer)
            trainer.accelerator = None
            trainer.webhook_handler = None
            trainer.config = None

            raw_config = {"webhook_config": "/nonexistent/path/to/webhook.json"}

            with self.assertRaises(ValueError) as context:
                trainer.configure_webhook(raw_config=raw_config, send_startup_message=False)

            self.assertIn("webhook_config file not found", str(context.exception))


if __name__ == "__main__":
    unittest.main()
