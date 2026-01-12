#!/usr/bin/env python3
"""Test webhook authentication functionality."""

import json
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from time import sleep
from unittest.mock import MagicMock, patch

from simpletuner.helpers.webhooks.handler import WebhookHandler


class AuthTestHTTPServer(BaseHTTPRequestHandler):
    """HTTP server that checks for X-API-Key header."""

    received_requests = []

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length else b""

        try:
            data = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            data = {}

        self.received_requests.append(
            {
                "path": self.path,
                "data": data,
                "headers": dict(self.headers),
                "api_key": self.headers.get("X-API-Key"),
            }
        )

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def log_message(self, format, *args):
        pass


class TestWebhookAuthHeader(unittest.TestCase):
    """Test that auth tokens are sent as X-API-Key header."""

    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 8998), AuthTestHTTPServer)
        cls.server_thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        AuthTestHTTPServer.received_requests.clear()

    def test_auth_token_sent_in_header(self):
        """Test that auth_token in config is sent as X-API-Key header."""
        webhook_config = [
            {
                "webhook_type": "raw",
                "callback_url": "http://127.0.0.1:8998/callback",
                "log_level": "info",
                "auth_token": "test_secret_token_12345",
            }
        ]

        mock_accelerator = MagicMock()
        mock_accelerator.is_main_process = True

        handler = WebhookHandler(
            mock_accelerator,
            "test_project",
            webhook_config=webhook_config,
        )

        handler.send_raw(
            structured_data={"message": "Auth test"},
            message_type="test",
            message_level="info",
        )

        sleep(0.2)

        self.assertEqual(len(AuthTestHTTPServer.received_requests), 1)
        request = AuthTestHTTPServer.received_requests[0]
        self.assertEqual(request["api_key"], "test_secret_token_12345")

    def test_no_auth_header_when_token_missing(self):
        """Test that no X-API-Key header is sent when auth_token is not provided."""
        webhook_config = [
            {
                "webhook_type": "raw",
                "callback_url": "http://127.0.0.1:8998/callback",
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

        handler.send_raw(
            structured_data={"message": "No auth test"},
            message_type="test",
            message_level="info",
        )

        sleep(0.2)

        self.assertEqual(len(AuthTestHTTPServer.received_requests), 1)
        request = AuthTestHTTPServer.received_requests[0]
        self.assertIsNone(request["api_key"])

    def test_auth_token_from_backend_config(self):
        """Test that auth_token is correctly extracted from WebhookConfig."""
        from simpletuner.helpers.webhooks.config import WebhookConfig

        config_dict = {
            "webhook_type": "raw",
            "callback_url": "http://127.0.0.1:8998/callback",
            "auth_token": "my_auth_token",
        }
        config = WebhookConfig(config_dict)

        # WebhookConfig uses __getattr__ to proxy attributes
        self.assertEqual(config.auth_token, "my_auth_token")


class TestWebhookConfigDuplicateFiltering(unittest.TestCase):
    """Test that duplicate webhook configs are filtered correctly."""

    def test_filter_removes_duplicate_callback_url(self):
        """Test that webhooks with matching callback_url are filtered."""
        default_url = "http://localhost:8001/callback"

        user_webhooks = [
            {
                "webhook_type": "raw",
                "callback_url": default_url,
                "log_level": "info",
            },
            {
                "webhook_type": "raw",
                "callback_url": "http://example.com/webhook",
                "log_level": "info",
            },
        ]

        # Filter out default callback URL
        filtered = [w for w in user_webhooks if w.get("callback_url") != default_url]

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["callback_url"], "http://example.com/webhook")

    def test_filter_preserves_user_webhooks(self):
        """Test that non-default webhooks are preserved."""
        default_url = "http://localhost:8001/callback"

        user_webhooks = [
            {
                "webhook_type": "discord",
                "webhook_url": "https://discord.com/api/webhooks/123",
            },
            {
                "webhook_type": "raw",
                "callback_url": "http://my-server.com/webhook",
                "log_level": "debug",
            },
        ]

        filtered = [w for w in user_webhooks if w.get("callback_url") != default_url]

        # Both should be preserved since neither matches default
        self.assertEqual(len(filtered), 2)


class TestAuthenticatedWebhookConfig(unittest.TestCase):
    """Test get_authenticated_webhook_config function."""

    @patch("simpletuner.simpletuner_sdk.server.services.webhook_defaults.get_or_create_callback_token")
    def test_authenticated_config_includes_token(self, mock_get_token):
        """Test that get_authenticated_webhook_config includes auth_token."""
        mock_get_token.return_value = "generated_token_xyz"

        from simpletuner.simpletuner_sdk.server.services.webhook_defaults import get_authenticated_webhook_config

        config = get_authenticated_webhook_config()

        self.assertEqual(len(config), 1)
        self.assertEqual(config[0]["webhook_type"], "raw")
        self.assertEqual(config[0]["auth_token"], "generated_token_xyz")

    @patch("simpletuner.simpletuner_sdk.server.services.webhook_defaults.get_or_create_callback_token")
    def test_authenticated_config_without_token(self, mock_get_token):
        """Test that config works even when token generation fails."""
        mock_get_token.return_value = None

        from simpletuner.simpletuner_sdk.server.services.webhook_defaults import get_authenticated_webhook_config

        config = get_authenticated_webhook_config()

        self.assertEqual(len(config), 1)
        self.assertEqual(config[0]["webhook_type"], "raw")
        self.assertNotIn("auth_token", config[0])


if __name__ == "__main__":
    unittest.main()
