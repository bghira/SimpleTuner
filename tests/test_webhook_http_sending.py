#!/usr/bin/env python3
"""Test that webhooks are actually sent via HTTP."""

import json
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from time import sleep
from unittest.mock import MagicMock

from simpletuner.helpers.webhooks.handler import WebhookHandler


class TestWebhookHTTPServer(BaseHTTPRequestHandler):
    """Simple HTTP server to receive webhook requests."""

    received_requests = []

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode("utf-8"))
            self.received_requests.append(
                {
                    "path": self.path,
                    "data": data,
                    "headers": dict(self.headers),
                }
            )
        except Exception as e:
            print(f"Error parsing request: {e}")

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def log_message(self, format, *args):
        # Suppress server logs
        pass


class TestWebhookHTTPSending(unittest.TestCase):
    """Test that webhooks are sent correctly via HTTP."""

    @classmethod
    def setUpClass(cls):
        """Start test HTTP server."""
        cls.server = HTTPServer(("127.0.0.1", 8999), TestWebhookHTTPServer)
        cls.server_thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        sleep(0.1)  # Give server time to start

    @classmethod
    def tearDownClass(cls):
        """Stop test HTTP server."""
        cls.server.shutdown()

    def setUp(self):
        """Clear received requests before each test."""
        TestWebhookHTTPServer.received_requests.clear()

    def test_simple_message_sent(self):
        """Test that a simple text message is sent via HTTP."""
        webhook_config = [
            {
                "webhook_type": "raw",
                "callback_url": "http://127.0.0.1:8999/callback",
                "log_level": "info",
            }
        ]

        mock_accelerator = MagicMock()
        mock_accelerator.is_main_process = True

        handler = WebhookHandler(
            mock_accelerator,
            "test_project",
            webhook_config=webhook_config,
        )

        # Send a test message
        handler.send(message="Test webhook message", message_level="info")

        # Give the request time to complete
        sleep(0.2)

        # Check that request was received
        self.assertEqual(len(TestWebhookHTTPServer.received_requests), 1)
        request = TestWebhookHTTPServer.received_requests[0]
        self.assertEqual(request["path"], "/callback")
        self.assertEqual(request["data"]["message"], "Test webhook message")

    def test_structured_data_sent(self):
        """Test that structured data is sent correctly."""
        webhook_config = [
            {
                "webhook_type": "raw",
                "callback_url": "http://127.0.0.1:8999/callback",
                "log_level": "info",
            }
        ]

        mock_accelerator = MagicMock()
        mock_accelerator.is_main_process = True

        handler = WebhookHandler(
            mock_accelerator,
            "test_project",
            webhook_config=webhook_config,
        )

        # Send structured data
        handler.send_raw(
            structured_data={"status": "training_started", "epoch": 1},
            message_type="training.status",
            message_level="info",
        )

        sleep(0.2)

        self.assertEqual(len(TestWebhookHTTPServer.received_requests), 1)
        request = TestWebhookHTTPServer.received_requests[0]
        self.assertEqual(request["data"]["status"], "training_started")
        self.assertEqual(request["data"]["type"], "training_status")
        self.assertEqual(request["data"]["epoch"], 1)

    def test_error_message_sent(self):
        """Test that error messages are sent with correct severity."""
        webhook_config = [
            {
                "webhook_type": "raw",
                "callback_url": "http://127.0.0.1:8999/callback",
                "log_level": "error",  # Only send errors
            }
        ]

        mock_accelerator = MagicMock()
        mock_accelerator.is_main_process = True

        handler = WebhookHandler(
            mock_accelerator,
            "test_project",
            webhook_config=webhook_config,
        )

        # Send an info message (should be filtered out)
        handler.send(message="Info message", message_level="info")
        sleep(0.1)

        # Send an error message (should be sent)
        handler.send(message="Error occurred", message_level="error")
        sleep(0.2)

        # Only the error should have been sent
        self.assertEqual(len(TestWebhookHTTPServer.received_requests), 1)
        request = TestWebhookHTTPServer.received_requests[0]
        self.assertEqual(request["data"]["message"], "Error occurred")

    def test_localhost_webhooks_use_http_fallback(self):
        """Test that localhost webhooks fall back to HTTP when callback service unavailable."""
        # Use 0.0.0.0 which triggers localhost detection
        webhook_config = [
            {
                "webhook_type": "raw",
                "callback_url": "http://127.0.0.1:8999/callback",
                "log_level": "info",
            }
        ]

        mock_accelerator = MagicMock()
        mock_accelerator.is_main_process = True

        handler = WebhookHandler(
            mock_accelerator,
            "test_project",
            webhook_config=webhook_config,
        )

        # This should trigger localhost detection and HTTP fallback
        handler.send(message="Localhost webhook test", message_level="info")
        sleep(0.2)

        # Request should have been received via HTTP fallback
        self.assertEqual(len(TestWebhookHTTPServer.received_requests), 1)
        request = TestWebhookHTTPServer.received_requests[0]
        self.assertIn("Localhost webhook test", request["data"]["message"])


if __name__ == "__main__":
    unittest.main()
