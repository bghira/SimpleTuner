#!/usr/bin/env python3
"""Test that SSL verification is automatically disabled for localhost HTTPS webhooks."""

import os
import unittest
from unittest.mock import patch


class TestWebhookSSLLocalhostDefaults(unittest.TestCase):
    """Test automatic SSL verification disabling for localhost HTTPS URLs."""

    def test_localhost_https_disables_ssl_verification(self):
        """Test that localhost HTTPS URLs automatically set ssl_no_verify=True."""
        with patch.dict(os.environ, {"SIMPLETUNER_SSL_ENABLED": "true"}, clear=False):
            from simpletuner.simpletuner_sdk.server.services.webhook_defaults import get_default_webhook_config

            config = get_default_webhook_config()
            self.assertEqual(len(config), 1)
            self.assertTrue(config[0]["ssl_no_verify"])

    def test_localhost_http_does_not_disable_ssl_verification(self):
        """Test that localhost HTTP URLs don't set ssl_no_verify=True."""
        with patch.dict(os.environ, {"SIMPLETUNER_SSL_ENABLED": "false"}, clear=False):
            from simpletuner.simpletuner_sdk.server.services.webhook_defaults import get_default_webhook_config

            config = get_default_webhook_config()
            self.assertEqual(len(config), 1)
            # HTTP doesn't need SSL verification (there's no SSL)
            self.assertFalse(config[0]["ssl_no_verify"])

    def test_localhost_ip_https_disables_ssl_verification(self):
        """Test that 127.0.0.1 HTTPS URLs automatically set ssl_no_verify=True."""
        with patch.dict(
            os.environ,
            {
                "SIMPLETUNER_SSL_ENABLED": "true",
                "SIMPLETUNER_WEBHOOK_HOST": "127.0.0.1",
            },
            clear=False,
        ):
            from simpletuner.simpletuner_sdk.server.services.webhook_defaults import get_default_webhook_config

            config = get_default_webhook_config()
            self.assertTrue(config[0]["ssl_no_verify"])
            self.assertIn("127.0.0.1", config[0]["callback_url"])

    def test_zero_ip_https_disables_ssl_verification(self):
        """Test that 0.0.0.0 HTTPS URLs automatically set ssl_no_verify=True."""
        with patch.dict(
            os.environ,
            {
                "SIMPLETUNER_SSL_ENABLED": "true",
                "SIMPLETUNER_WEBHOOK_HOST": "0.0.0.0",
            },
            clear=False,
        ):
            from simpletuner.simpletuner_sdk.server.services.webhook_defaults import get_default_webhook_config

            config = get_default_webhook_config()
            self.assertTrue(config[0]["ssl_no_verify"])
            self.assertIn("0.0.0.0", config[0]["callback_url"])

    def test_external_domain_respects_env_var(self):
        """Test that external domains respect SIMPLETUNER_SSL_NO_VERIFY env var."""
        with patch.dict(
            os.environ,
            {
                "SIMPLETUNER_SSL_ENABLED": "true",
                "SIMPLETUNER_WEBHOOK_CALLBACK_URL": "https://example.com:8001/callback",
                "SIMPLETUNER_SSL_NO_VERIFY": "false",
            },
            clear=False,
        ):
            from simpletuner.simpletuner_sdk.server.services.webhook_defaults import get_default_webhook_config

            config = get_default_webhook_config()
            # External domain should respect the env var (false)
            self.assertFalse(config[0]["ssl_no_verify"])

    def test_external_domain_with_explicit_no_verify(self):
        """Test that external domains can explicitly disable SSL verification."""
        with patch.dict(
            os.environ,
            {
                "SIMPLETUNER_SSL_ENABLED": "true",
                "SIMPLETUNER_WEBHOOK_CALLBACK_URL": "https://example.com:8001/callback",
                "SIMPLETUNER_SSL_NO_VERIFY": "true",
            },
            clear=False,
        ):
            from simpletuner.simpletuner_sdk.server.services.webhook_defaults import get_default_webhook_config

            config = get_default_webhook_config()
            # Explicitly set to true via env var
            self.assertTrue(config[0]["ssl_no_verify"])

    def test_env_var_override_for_localhost(self):
        """Test that SIMPLETUNER_SSL_NO_VERIFY can override localhost detection."""
        # This test verifies that explicit env var always wins
        with patch.dict(
            os.environ,
            {
                "SIMPLETUNER_SSL_ENABLED": "true",
                "SIMPLETUNER_WEBHOOK_HOST": "localhost",
                "SIMPLETUNER_SSL_NO_VERIFY": "true",
            },
            clear=False,
        ):
            from simpletuner.simpletuner_sdk.server.services.webhook_defaults import get_default_webhook_config

            config = get_default_webhook_config()
            # Both auto-detection and env var say true
            self.assertTrue(config[0]["ssl_no_verify"])


if __name__ == "__main__":
    unittest.main()
