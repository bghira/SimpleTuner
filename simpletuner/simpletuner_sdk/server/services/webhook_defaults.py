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
    ssl_no_verify = os.environ.get("SIMPLETUNER_SSL_NO_VERIFY", "false").lower() == "true"

    return [
        {
            "webhook_type": "raw",
            "callback_url": get_default_callback_url(),
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
