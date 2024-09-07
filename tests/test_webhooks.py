import unittest
from unittest.mock import patch, MagicMock
from helpers.webhooks.handler import WebhookHandler
from helpers.webhooks.config import WebhookConfig
from io import BytesIO
from PIL import Image


class TestWebhookHandler(unittest.TestCase):
    def setUp(self):
        # Create a mock for the WebhookConfig
        mock_config_instance = MagicMock(spec=WebhookConfig)
        mock_config_instance.webhook_url = "http://example.com/webhook"
        mock_config_instance.webhook_type = "discord"
        mock_config_instance.log_level = "info"
        mock_config_instance.message_prefix = "TestPrefix"

        # Mock the accelerator object
        self.mock_accelerator = MagicMock()
        self.mock_accelerator.is_main_process = True

        # Instantiate the handler with the mocked config
        self.handler = WebhookHandler(
            config_path="dummy_path",
            accelerator=self.mock_accelerator,
            project_name="TestProject",
            mock_webhook_config=mock_config_instance,
        )

    @patch("requests.post")
    def test_send_message_info_level(self, mock_post):
        # Test sending a simple info level message
        self.handler.send("Test message", message_level="info")
        mock_post.assert_called_once()

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
        self.handler.send(
            "Test message with image", images=[image], message_level="info"
        )
        args, kwargs = mock_post.call_args
        self.assertIn("files", kwargs)
        self.assertEqual(len(kwargs["files"]), 1)

    @patch("requests.post")
    def test_response_storage(self, mock_post):
        # Mock response object
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_post.return_value = mock_response

        self.handler.send("Test message", message_level="info", store_response=True)
        self.assertEqual(self.handler.stored_response, mock_response.headers)


if __name__ == "__main__":
    unittest.main()
