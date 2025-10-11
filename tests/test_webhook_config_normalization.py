"""Test webhook_config normalization in ConfigModel."""
import unittest

from simpletuner.simpletuner_sdk.configuration import ConfigModel


class TestWebhookConfigNormalization(unittest.TestCase):
    """Test that webhook_config is properly normalized in ConfigModel."""

    def test_none_webhook_config(self):
        """Test that None webhook_config is normalized to empty list."""
        config = ConfigModel(
            trainer_config={},
            dataloader_config=[],
            webhook_config=None,
            job_id="test_none"
        )
        self.assertEqual(config.webhook_config, [])
        self.assertIsInstance(config.webhook_config, list)

    def test_dict_webhook_config(self):
        """Test that dict webhook_config is normalized to list with one element."""
        webhook_dict = {
            "webhook_type": "discord",
            "webhook_url": "http://discord.com/webhook",
            "log_level": "info"
        }
        config = ConfigModel(
            trainer_config={},
            dataloader_config=[],
            webhook_config=webhook_dict,
            job_id="test_dict"
        )
        self.assertIsInstance(config.webhook_config, list)
        self.assertEqual(len(config.webhook_config), 1)
        self.assertEqual(config.webhook_config[0], webhook_dict)

    def test_list_webhook_config(self):
        """Test that list webhook_config is preserved as-is."""
        webhook_list = [
            {
                "webhook_type": "raw",
                "callback_url": "http://localhost:8001/callback",
                "log_level": "info"
            },
            {
                "webhook_type": "discord",
                "webhook_url": "http://discord.com/webhook",
                "log_level": "warning",
                "message_prefix": "MyProject"
            }
        ]
        config = ConfigModel(
            trainer_config={},
            dataloader_config=[],
            webhook_config=webhook_list,
            job_id="test_list"
        )
        self.assertIsInstance(config.webhook_config, list)
        self.assertEqual(len(config.webhook_config), 2)
        self.assertEqual(config.webhook_config, webhook_list)

    def test_empty_list_webhook_config(self):
        """Test that empty list webhook_config is preserved."""
        config = ConfigModel(
            trainer_config={},
            dataloader_config=[],
            webhook_config=[],
            job_id="test_empty_list"
        )
        self.assertEqual(config.webhook_config, [])
        self.assertIsInstance(config.webhook_config, list)

    def test_invalid_webhook_config_type(self):
        """Test that invalid webhook_config type raises ValueError."""
        with self.assertRaises(ValueError) as context:
            ConfigModel(
                trainer_config={},
                dataloader_config=[],
                webhook_config="invalid_string",
                job_id="test_invalid"
            )
        self.assertIn("webhook_config must be dict or list", str(context.exception))

    def test_backward_compatibility_with_empty_dict(self):
        """Test that empty dict is handled gracefully (backward compatibility)."""
        config = ConfigModel(
            trainer_config={},
            dataloader_config=[],
            webhook_config={},
            job_id="test_empty_dict"
        )
        self.assertIsInstance(config.webhook_config, list)
        self.assertEqual(len(config.webhook_config), 1)
        self.assertEqual(config.webhook_config[0], {})


if __name__ == "__main__":
    unittest.main()
