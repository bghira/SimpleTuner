"""Shared webhook defaults for trainer services."""

from __future__ import annotations

import os


def get_default_callback_url() -> str:
    """Get the default callback URL, supporting HTTPS when SSL is enabled."""
    # Check if SSL is enabled via environment variable
    ssl_enabled = os.environ.get("SIMPLETUNER_SSL_ENABLED", "false").lower() == "true"
    protocol = "https" if ssl_enabled else "http"

    # Get host and port from environment or use defaults
    host = os.environ.get("SIMPLETUNER_WEBHOOK_HOST", "localhost")
    port = os.environ.get("SIMPLETUNER_WEBHOOK_PORT", "8001")

    # Construct URL
    base_url = f"{protocol}://{host}:{port}"
    return os.environ.get("SIMPLETUNER_WEBHOOK_CALLBACK_URL", f"{base_url}/callback")


def get_default_webhook_config() -> list:
    """Get the default webhook configuration with SSL verification settings."""
    callback_url = get_default_callback_url()

    # Determine if SSL verification should be disabled
    # For localhost HTTPS URLs with self-signed certs, we should skip verification by default
    ssl_no_verify = os.environ.get("SIMPLETUNER_SSL_NO_VERIFY", "false").lower() == "true"

    # Auto-detect: if using HTTPS with localhost, disable verification (self-signed cert)
    if callback_url.startswith("https://"):
        try:
            from urllib.parse import urlparse

            parsed = urlparse(callback_url)
            hostname = parsed.hostname
            if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
                ssl_no_verify = True
        except Exception:
            pass

    return [
        {
            "webhook_type": "raw",
            "callback_url": callback_url,
            "log_level": "info",
            "ssl_no_verify": ssl_no_verify,
        }
    ]


# Legacy compatibility
DEFAULT_CALLBACK_URL = get_default_callback_url()
DEFAULT_WEBHOOK_CONFIG = get_default_webhook_config()

__all__ = [
    "DEFAULT_CALLBACK_URL",
    "DEFAULT_WEBHOOK_CONFIG",
    "get_default_callback_url",
    "get_default_webhook_config",
]
