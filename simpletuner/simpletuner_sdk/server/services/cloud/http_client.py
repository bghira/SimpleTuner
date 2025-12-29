"""HTTP client factory with SSL, proxy, and observability support.

Provides consistent HTTP client configuration across the application,
including SSL verification, custom CA bundles, proxy settings, and
request tracing via correlation IDs.
"""

from __future__ import annotations

import logging
import os
import ssl
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# Thread-local storage for correlation IDs
_correlation_id = threading.local()


def get_correlation_id() -> str:
    """Get the current correlation ID, or generate a new one."""
    if not hasattr(_correlation_id, "value") or _correlation_id.value is None:
        _correlation_id.value = str(uuid.uuid4())[:12]
    return _correlation_id.value


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context."""
    _correlation_id.value = correlation_id


def clear_correlation_id() -> None:
    """Clear the correlation ID for the current context."""
    _correlation_id.value = None


@dataclass
class HTTPClientConfig:
    """Configuration for HTTP clients.

    Attributes:
        ssl_verify: Whether to verify SSL certificates. Defaults to True.
        ssl_ca_bundle: Path to custom CA bundle file for SSL verification.
        proxy_url: HTTP/HTTPS proxy URL (e.g., "http://proxy:8080").
        timeout: Default request timeout in seconds.
        max_retries: Maximum number of retry attempts for failed requests.
        user_agent: Custom User-Agent header.
    """

    ssl_verify: bool = True
    ssl_ca_bundle: Optional[str] = None
    proxy_url: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    user_agent: str = "SimpleTuner-Cloud/1.0"

    # Metrics tracking
    _request_count: int = field(default=0, repr=False)
    _error_count: int = field(default=0, repr=False)
    _total_latency_ms: float = field(default=0.0, repr=False)

    def get_verify(self) -> Union[bool, str]:
        """Get the SSL verify setting for httpx/requests.

        Returns:
            - False if ssl_verify is False
            - Path to CA bundle if ssl_ca_bundle is set and ssl_verify is True
            - True otherwise
        """
        if not self.ssl_verify:
            return False
        if self.ssl_ca_bundle:
            return self.ssl_ca_bundle
        return True

    def get_proxies(self) -> Optional[Dict[str, str]]:
        """Get proxy configuration for requests library."""
        if not self.proxy_url:
            return None
        return {
            "http": self.proxy_url,
            "https": self.proxy_url,
        }

    def get_mounts(self) -> Optional[Dict[str, Any]]:
        """Get proxy mounts for httpx library."""
        if not self.proxy_url:
            return None

        try:
            import httpx

            transport = httpx.HTTPTransport(proxy=self.proxy_url)
            return {
                "http://": transport,
                "https://": transport,
            }
        except ImportError:
            return None


class HTTPClientFactory:
    """Factory for creating HTTP clients with consistent configuration.

    Supports both sync (requests) and async (httpx) clients.
    Configuration is loaded from provider config or environment variables.

    Features connection pooling for HTTP clients to reuse connections
    and reduce latency.
    """

    _instance: Optional["HTTPClientFactory"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "HTTPClientFactory":
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self._config = self._load_config()
        self._async_client: Optional[Any] = None  # httpx.AsyncClient
        self._async_client_lock = threading.Lock()
        self._initialized = True

    def _load_config(self) -> HTTPClientConfig:
        """Load configuration from provider config or environment."""
        config = HTTPClientConfig()

        # Try to load from provider config first (use sync method to avoid
        # event loop issues during singleton initialization)
        try:
            from .storage.provider_config_store import ProviderConfigStore

            store = ProviderConfigStore()
            # Load from any configured provider - check replicate first
            provider_config = store.get_sync("replicate")

            if "ssl_verify" in provider_config:
                config.ssl_verify = provider_config["ssl_verify"]
            if "ssl_ca_bundle" in provider_config:
                config.ssl_ca_bundle = provider_config["ssl_ca_bundle"]
            if "proxy_url" in provider_config:
                config.proxy_url = provider_config["proxy_url"]
            if "http_timeout" in provider_config:
                config.timeout = provider_config["http_timeout"]

        except Exception as exc:
            logger.debug("Could not load HTTP config from provider config: %s", exc)

        # Environment variables override provider config
        if os.environ.get("SIMPLETUNER_SSL_VERIFY") is not None:
            config.ssl_verify = os.environ.get("SIMPLETUNER_SSL_VERIFY", "").lower() not in ("0", "false", "no")

        if os.environ.get("SIMPLETUNER_CA_BUNDLE"):
            config.ssl_ca_bundle = os.environ["SIMPLETUNER_CA_BUNDLE"]

        if os.environ.get("HTTPS_PROXY"):
            config.proxy_url = os.environ["HTTPS_PROXY"]
        elif os.environ.get("HTTP_PROXY"):
            config.proxy_url = os.environ["HTTP_PROXY"]

        if os.environ.get("SIMPLETUNER_HTTP_TIMEOUT"):
            try:
                config.timeout = float(os.environ["SIMPLETUNER_HTTP_TIMEOUT"])
            except ValueError:
                pass

        return config

    def reload_config(self) -> None:
        """Reload configuration from provider config and environment."""
        self._config = self._load_config()
        # Close pooled client so it picks up new config
        self._close_pooled_client()

    @property
    def config(self) -> HTTPClientConfig:
        """Get the current configuration."""
        return self._config

    def _close_pooled_client(self) -> None:
        """Close the pooled async client if it exists."""
        with self._async_client_lock:
            if self._async_client is not None:
                import asyncio

                try:
                    # Try to close gracefully
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self._async_client.aclose())
                    else:
                        loop.run_until_complete(self._async_client.aclose())
                except Exception as exc:
                    logger.debug("Error closing pooled HTTP client: %s", exc)
                self._async_client = None

    async def get_pooled_client(self):
        """Get a pooled async HTTP client.

        The client is created once and reused for connection pooling.
        This reduces latency by keeping connections alive.

        Note: The pooled client should be used for simple requests.
        For requests needing circuit breaker protection, use async_client() instead.
        """
        import httpx

        if self._async_client is None:
            with self._async_client_lock:
                if self._async_client is None:
                    verify = self._config.get_verify()
                    client_kwargs: Dict[str, Any] = {
                        "timeout": self._config.timeout,
                        "verify": verify,
                        "headers": self.get_default_headers(),
                        # Connection pool settings
                        "limits": httpx.Limits(
                            max_keepalive_connections=20,
                            max_connections=100,
                            keepalive_expiry=30.0,
                        ),
                    }
                    if self._config.proxy_url:
                        client_kwargs["proxy"] = self._config.proxy_url

                    self._async_client = httpx.AsyncClient(**client_kwargs)
                    logger.debug("Created pooled HTTP client with connection limits")

        return self._async_client

    def get_default_headers(self) -> Dict[str, str]:
        """Get default headers for all requests."""
        return {
            "User-Agent": self._config.user_agent,
            "X-Correlation-ID": get_correlation_id(),
        }

    @contextmanager
    def sync_client(self, timeout: Optional[float] = None):
        """Get a configured sync HTTP client (requests).

        Usage:
            with factory.sync_client() as session:
                response = session.get("https://api.example.com")
        """
        import requests

        session = requests.Session()

        # Apply configuration
        session.verify = self._config.get_verify()
        if self._config.get_proxies():
            session.proxies = self._config.get_proxies()
        session.headers.update(self.get_default_headers())

        # Set timeout on a per-request basis via adapter
        effective_timeout = timeout or self._config.timeout

        try:
            yield session
        finally:
            session.close()

    @asynccontextmanager
    async def async_client(
        self,
        timeout: Optional[float] = None,
        circuit_breaker_name: Optional[str] = None,
    ):
        """Get a configured async HTTP client (httpx).

        Args:
            timeout: Request timeout in seconds
            circuit_breaker_name: Name of circuit breaker to use (optional)

        Usage:
            async with factory.async_client() as client:
                response = await client.get("https://api.example.com")

            # With circuit breaker protection
            async with factory.async_client(circuit_breaker_name="replicate-api") as client:
                response = await client.get("https://api.replicate.com")
        """
        import httpx

        effective_timeout = timeout or self._config.timeout
        verify = self._config.get_verify()

        # Build client kwargs
        client_kwargs: Dict[str, Any] = {
            "timeout": effective_timeout,
            "verify": verify,
            "headers": self.get_default_headers(),
        }

        # Add proxy if configured
        if self._config.proxy_url:
            client_kwargs["proxy"] = self._config.proxy_url

        # Optionally wrap with circuit breaker
        if circuit_breaker_name:
            from .resilience import get_circuit_breaker

            breaker = get_circuit_breaker(circuit_breaker_name)
            async with breaker:
                async with httpx.AsyncClient(**client_kwargs) as client:
                    yield client
        else:
            async with httpx.AsyncClient(**client_kwargs) as client:
                yield client

    def create_ssl_context(self) -> ssl.SSLContext:
        """Create an SSL context with the configured settings.

        Useful for libraries that need a raw SSLContext.
        """
        if not self._config.ssl_verify:
            # Create context that doesn't verify
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx

        if self._config.ssl_ca_bundle:
            ctx = ssl.create_default_context(cafile=self._config.ssl_ca_bundle)
        else:
            ctx = ssl.create_default_context()

        return ctx

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._close_pooled_client()
            cls._instance = None


def get_http_client_factory() -> HTTPClientFactory:
    """Get the global HTTPClientFactory instance."""
    return HTTPClientFactory()


# Convenience functions for common use cases


@asynccontextmanager
async def get_async_client(
    timeout: Optional[float] = None,
    circuit_breaker_name: Optional[str] = None,
):
    """Convenience function to get an async HTTP client.

    Args:
        timeout: Request timeout in seconds
        circuit_breaker_name: Optional circuit breaker to protect calls

    Usage:
        async with get_async_client() as client:
            response = await client.get("https://api.example.com")

        # With circuit breaker
        async with get_async_client(circuit_breaker_name="replicate-api") as client:
            response = await client.get("https://api.replicate.com")
    """
    factory = get_http_client_factory()
    async with factory.async_client(timeout, circuit_breaker_name) as client:
        yield client


@contextmanager
def get_sync_client(timeout: Optional[float] = None):
    """Convenience function to get a sync HTTP client.

    Usage:
        with get_sync_client() as session:
            response = session.get("https://api.example.com")
    """
    factory = get_http_client_factory()
    with factory.sync_client(timeout) as session:
        yield session


async def get_pooled_async_client():
    """Get the pooled async HTTP client for efficient connection reuse.

    Unlike get_async_client(), this returns a persistent client with
    connection pooling enabled. Use this for high-frequency API calls.

    Usage:
        client = await get_pooled_async_client()
        response = await client.get("https://api.example.com")

    Note: Don't close the returned client - it's shared across the application.
    """
    factory = get_http_client_factory()
    return await factory.get_pooled_client()
