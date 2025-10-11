import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image

from simpletuner.helpers.webhooks.config import WebhookConfig
from simpletuner.helpers.webhooks.handler import WebhookHandler


class TestWebhookHandler(unittest.TestCase):
    def setUp(self):
        # Create a mock for the WebhookConfig
        self.mock_config_instance = MagicMock(spec=WebhookConfig)
        self.mock_config_instance.webhook_url = "http://example.com/webhook"
        self.mock_config_instance.webhook_type = "discord"
        self.mock_config_instance.log_level = "info"
        self.mock_config_instance.message_prefix = "TestPrefix"
        self.mock_config_instance.values = {
            "webhook_url": "http://example.com/webhook",
            "webhook_type": "discord",
            "log_level": "info",
            "message_prefix": "TestPrefix",
        }

        # Mock the accelerator object
        self.mock_accelerator = MagicMock()
        self.mock_accelerator.is_main_process = True

        # Instantiate the handler with the mocked config
        self.handler = WebhookHandler(
            accelerator=self.mock_accelerator,
            project_name="TestProject",
            mock_webhook_config=self.mock_config_instance,
            video_framerate=99,
        )

    @patch("requests.post")
    def test_send_message_info_level(self, mock_post):
        # Test sending a simple info level message
        message = "Test message"
        self.handler.send(message, message_level="info")
        mock_post.assert_called_once()
        # Capture the call arguments
        args, kwargs = mock_post.call_args
        # Assuming the message is sent in 'data' parameter
        self.assertIn("data", kwargs)
        self.assertIn(message, kwargs["data"].get("content"))

    @patch("requests.post")
    def test_debug_message_wont_send(self, mock_post):
        # Test that debug logs don't send when the log level is info
        self.handler.send("Test message", message_level="debug")
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_do_not_send_lower_than_configured_level(self, mock_post):
        # Create handler with error log level and test
        config = {
            "webhook_type": "discord",
            "webhook_url": "http://test.com/webhook",
            "log_level": "error",  # Higher log level
        }
        handler = WebhookHandler(accelerator=self.mock_accelerator, project_name="TestProject", webhook_config=config)
        handler.send("Test message", message_level="info")
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_send_with_images(self, mock_post):
        # Test sending messages with images
        image = Image.new("RGB", (60, 30), color="red")
        message = "Test message with image"
        self.handler.send(message, images=[image], message_level="info")
        args, kwargs = mock_post.call_args
        self.assertIn("files", kwargs)
        self.assertEqual(len(kwargs["files"]), 1)
        # Check that the message is in the 'data' parameter
        content = kwargs.get("data", {}).get("content", "")
        self.assertIn(self.mock_config_instance.values.get("message_prefix"), content)
        self.assertIn("data", kwargs, f"Check data for contents: {kwargs}")
        self.assertIn(message, content)

    @patch("requests.post")
    def test_response_storage(self, mock_post):
        # Mock response object
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_post.return_value = mock_response

        self.handler.send("Test message", message_level="info", store_response=True)
        self.assertEqual(self.handler.stored_response, mock_response.headers)
        # Also check that the message is sent
        args, kwargs = mock_post.call_args
        content = kwargs.get("data", {}).get("content", "")
        self.assertIn(self.mock_config_instance.values.get("message_prefix"), content)
        self.assertIn("Test message", content)

    def test_single_dict_config(self):
        """Test that a single dict config is properly converted to list format"""
        config = {"webhook_type": "discord", "webhook_url": "http://test.com/webhook", "log_level": "info"}
        handler = WebhookHandler(accelerator=self.mock_accelerator, project_name="TestProject", webhook_config=config)
        self.assertEqual(len(handler.backends), 1)
        self.assertEqual(handler.backends[0]["webhook_url"], "http://test.com/webhook")

    def test_list_config(self):
        """Test that a list config with multiple webhooks is properly handled"""
        config = [
            {"webhook_type": "raw", "callback_url": "http://localhost:8001/callback", "log_level": "info"},
            {"webhook_type": "discord", "webhook_url": "http://discord.com/webhook1", "log_level": "warning"},
        ]
        handler = WebhookHandler(accelerator=self.mock_accelerator, project_name="TestProject", webhook_config=config)
        self.assertEqual(len(handler.backends), 2)
        self.assertEqual(handler.backends[0]["webhook_type"], "raw")
        self.assertEqual(handler.backends[1]["webhook_type"], "discord")

    @patch("requests.post")
    def test_multiple_backends_send(self, mock_post):
        """Test that messages are sent to all configured backends"""
        config = [
            {"webhook_type": "discord", "webhook_url": "http://discord.com/webhook1", "log_level": "info"},
            {"webhook_type": "discord", "webhook_url": "http://discord.com/webhook2", "log_level": "info"},
        ]
        handler = WebhookHandler(accelerator=self.mock_accelerator, project_name="TestProject", webhook_config=config)
        handler.send("Test message", message_level="info")
        # Should be called twice, once for each backend
        self.assertEqual(mock_post.call_count, 2)

    @patch("requests.post")
    def test_log_level_filtering(self, mock_post):
        """Test that backends filter messages based on their log levels"""
        config = [
            {"webhook_type": "discord", "webhook_url": "http://discord.com/webhook1", "log_level": "info"},
            {"webhook_type": "discord", "webhook_url": "http://discord.com/webhook2", "log_level": "error"},
        ]
        handler = WebhookHandler(accelerator=self.mock_accelerator, project_name="TestProject", webhook_config=config)
        # Send info message - should only go to first backend
        handler.send("Test info", message_level="info")
        self.assertEqual(mock_post.call_count, 1)

        mock_post.reset_mock()

        # Send error message - should go to both backends
        handler.send("Test error", message_level="error")
        self.assertEqual(mock_post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
