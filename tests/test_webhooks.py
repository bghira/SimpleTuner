import unittest
from unittest.mock import patch, MagicMock
from simpletuner.helpers.webhooks.handler import WebhookHandler
from simpletuner.helpers.webhooks.config import WebhookConfig
from io import BytesIO
from PIL import Image


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
            config_path="dummy_path",
            accelerator=self.mock_accelerator,
            project_name="TestProject",
            mock_webhook_config=self.mock_config_instance,
            args=MagicMock(framerate=99),
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
        # Set a higher log level and test
        self.handler.log_level = 1  # Error level
        self.handler.send("Test message", message_level="info")
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


if __name__ == "__main__":
    unittest.main()
