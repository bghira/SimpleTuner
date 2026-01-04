"""Shared webhook defaults for trainer services."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Cache for the callback auth token (generated once per server session)
_callback_auth_token: Optional[str] = None
_callback_token_lock = asyncio.Lock()


async def _get_or_create_callback_token_async() -> Optional[str]:
    """Get or create a callback API key for webhook authentication.

    This creates an API key associated with the local admin user that the
    trainer uses to authenticate callbacks to the server.

    Returns:
        The raw API key string, or None if token generation fails.
    """
    global _callback_auth_token

    if _callback_auth_token is not None:
        return _callback_auth_token

    async with _callback_token_lock:
        # Double-check after acquiring lock
        if _callback_auth_token is not None:
            return _callback_auth_token

        try:
            from .cloud.auth.user_store import UserStore

            store = UserStore()

            # Ensure local admin user exists (single-user mode)
            local_admin = await store._ensure_single_user_mode()
            if not local_admin:
                # Try to get an existing user
                users = await store.list_users(limit=1)
                if not users:
                    logger.warning("No users available for callback token generation")
                    return None
                local_admin = users[0]

            # Check if callback key already exists
            existing_keys = await store.list_api_keys(local_admin.id)
            for key in existing_keys:
                if key.name == "__callback__" and key.is_active:
                    # Key exists but we don't have the raw token
                    # We need to create a new one since raw keys are not stored
                    logger.debug("Existing callback key found but raw token unavailable, creating new one")
                    # Revoke the old key
                    await store.revoke_api_key(key.id)
                    break

            # Create new callback API key (no expiration for local callbacks)
            api_key, raw_key = await store.create_api_key(
                user_id=local_admin.id,
                name="__callback__",
                expires_in_days=None,  # No expiration for callback tokens
            )
            _callback_auth_token = raw_key
            logger.debug("Created callback auth token for user %s", local_admin.username)
            return raw_key

        except Exception as exc:
            logger.warning("Failed to generate callback auth token: %s", exc)
            return None


def get_or_create_callback_token() -> Optional[str]:
    """Synchronous wrapper for getting or creating callback auth token.

    Returns:
        The raw API key string, or None if token generation fails.
    """
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context, run in executor to avoid blocking
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(lambda: asyncio.run(_get_or_create_callback_token_async()))
            return future.result(timeout=10)
    except RuntimeError:
        # No running event loop, run directly
        return asyncio.run(_get_or_create_callback_token_async())
    except Exception as exc:
        logger.warning("Error getting callback token: %s", exc)
        return None


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


def get_authenticated_webhook_config() -> list:
    """Get webhook configuration with authentication token included.

    This should be called at job submission time to include a valid auth token
    that the trainer can use to authenticate callbacks to the server.

    Returns:
        List containing a single webhook config dict with auth_token included.
    """
    config = get_default_webhook_config()

    # Generate or retrieve the callback auth token
    auth_token = get_or_create_callback_token()
    if auth_token:
        # Add auth_token to the first (and typically only) webhook config
        for webhook_config in config:
            if webhook_config.get("webhook_type") == "raw":
                webhook_config["auth_token"] = auth_token

    return config


# Legacy compatibility
DEFAULT_CALLBACK_URL = get_default_callback_url()
DEFAULT_WEBHOOK_CONFIG = get_default_webhook_config()

__all__ = [
    "DEFAULT_CALLBACK_URL",
    "DEFAULT_WEBHOOK_CONFIG",
    "get_default_callback_url",
    "get_default_webhook_config",
    "get_authenticated_webhook_config",
    "get_or_create_callback_token",
]
