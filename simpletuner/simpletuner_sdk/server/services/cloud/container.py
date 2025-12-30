"""Dependency injection container for cloud services.

Provides FastAPI-compatible dependency providers that can be overridden for testing.

Usage in routes:
    from fastapi import Depends
    from ...services.cloud.container import get_job_store, get_secrets_manager

    @router.get("/jobs")
    async def list_jobs(store: AsyncJobStore = Depends(get_job_store)):
        return await store.list_jobs()

Testing:
    from fastapi.testclient import TestClient
    from ...services.cloud.container import container

    # Override before creating test client
    container.set_job_store(mock_store)
    client = TestClient(app)

    # Reset after tests
    container.reset()

Multi-Worker Deployment Notes:
    When running with multiple worker processes (e.g., uvicorn --workers 4):

    1. **Each worker has its own memory space** - singleton instances are
       per-process, not shared across workers.

    2. **asyncio.Lock is per-process** - The _state_backend_lock protects
       singleton creation within a single worker, not across workers.
       This is intentional and correct behavior.

    3. **Cross-worker coordination** - For features requiring coordination
       across workers (like distributed rate limiting), use:
       - Database transactions (SQLite WAL mode, PostgreSQL, etc.)
       - Redis-based state backends
       - External coordination services

    4. **State backends** - Each worker creates its own connection to the
       shared database. SQLite WAL mode allows concurrent reads and
       serializes writes correctly.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

if TYPE_CHECKING:
    from .async_job_store import AsyncJobStore
    from .secrets import SecretsManager
    from .state_backend import StateBackendProtocol

T = TypeVar("T")


class ServiceContainer:
    """Thread-safe container for service instances.

    Supports:
    - Lazy initialization of services
    - Singleton pattern with per-instance caching
    - Override capability for testing
    - Thread-safe access
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._instances: dict[str, object] = {}
        self._factories: dict[str, Callable[[], object]] = {}
        self._overrides: dict[str, object] = {}

    def register(self, name: str, factory: Callable[[], T]) -> None:
        """Register a factory function for a service.

        Args:
            name: Service identifier
            factory: Zero-argument callable that creates the service
        """
        with self._lock:
            self._factories[name] = factory

    def get(self, name: str) -> object:
        """Get or create a service instance.

        Returns override if set, otherwise creates/returns singleton.
        """
        with self._lock:
            # Check for test override first
            if name in self._overrides:
                return self._overrides[name]

            # Return existing instance if available
            if name in self._instances:
                return self._instances[name]

            # Create new instance from factory
            factory = self._factories.get(name)
            if factory is None:
                raise KeyError(f"No factory registered for service: {name}")

            instance = factory()
            self._instances[name] = instance
            return instance

    def override(self, name: str, instance: object) -> None:
        """Override a service with a specific instance (for testing).

        Args:
            name: Service identifier
            instance: Instance to use instead of factory-created one
        """
        with self._lock:
            self._overrides[name] = instance

    def clear_override(self, name: str) -> None:
        """Remove an override for a service."""
        with self._lock:
            self._overrides.pop(name, None)

    def reset(self) -> None:
        """Reset all overrides and cached instances.

        Call this between tests to ensure clean state.
        """
        with self._lock:
            self._overrides.clear()
            self._instances.clear()

    def reset_overrides(self) -> None:
        """Reset only overrides, keeping cached instances."""
        with self._lock:
            self._overrides.clear()


# Global container instance
container = ServiceContainer()


# --- Factory registrations ---


def _create_job_store():
    """Factory for AsyncJobStore (sync compatibility mode)."""
    from .async_job_store import AsyncJobStore

    return AsyncJobStore.get_instance_sync()


def _create_secrets_manager():
    """Factory for SecretsManager."""
    from .secrets import get_secrets_manager as _get_secrets

    return _get_secrets()


def _create_user_store():
    """Factory for UserStore."""
    from .auth import UserStore

    return UserStore()


def _create_quota_checker():
    """Factory for QuotaChecker."""
    from .auth import QuotaChecker

    job_store = container.get("job_store")
    user_store = container.get("user_store")
    return QuotaChecker(job_store, user_store)


def _create_queue_manager():
    """Factory for QueueStore."""
    from .queue import QueueStore

    return QueueStore()


def _create_notification_service():
    """Factory for NotificationService."""
    from .notification import NotificationService
    from .notification.notification_store import NotificationStore

    store = NotificationStore()
    return NotificationService(store)


# State backend singleton management
_state_backend: Optional["StateBackendProtocol"] = None
_state_backend_lock: Optional[asyncio.Lock] = None
_state_backend_lock_loop: Optional[asyncio.AbstractEventLoop] = None


async def _get_or_create_state_backend() -> "StateBackendProtocol":
    """Get or create the state backend singleton."""
    global _state_backend, _state_backend_lock, _state_backend_lock_loop

    if _state_backend is not None:
        return _state_backend

    # Recreate lock if bound to a different event loop
    current_loop = asyncio.get_running_loop()
    if _state_backend_lock is None or _state_backend_lock_loop is not current_loop:
        _state_backend_lock = asyncio.Lock()
        _state_backend_lock_loop = current_loop

    async with _state_backend_lock:
        if _state_backend is not None:
            return _state_backend

        from .state_backend import get_state_backend

        _state_backend = await get_state_backend()
        return _state_backend


def _reset_state_backend() -> None:
    """Reset the state backend singleton."""
    global _state_backend, _state_backend_lock, _state_backend_lock_loop
    _state_backend = None
    _state_backend_lock = None
    _state_backend_lock_loop = None


# Register default factories
container.register("job_store", _create_job_store)
container.register("secrets_manager", _create_secrets_manager)
container.register("user_store", _create_user_store)
container.register("quota_checker", _create_quota_checker)
container.register("queue_manager", _create_queue_manager)
container.register("notification_service", _create_notification_service)


# --- FastAPI dependency providers ---
# These can be used with Depends() in route handlers


def get_job_store() -> "AsyncJobStore":
    """FastAPI dependency that provides AsyncJobStore."""
    return container.get("job_store")


def get_secrets_manager() -> "SecretsManager":
    """FastAPI dependency that provides SecretsManager."""
    return container.get("secrets_manager")


def get_user_store():
    """FastAPI dependency that provides UserStore."""
    return container.get("user_store")


def get_quota_checker():
    """FastAPI dependency that provides QuotaChecker."""
    return container.get("quota_checker")


def get_queue_manager():
    """FastAPI dependency that provides QueueManager."""
    return container.get("queue_manager")


def get_notification_service():
    """FastAPI dependency that provides NotificationService."""
    return container.get("notification_service")


async def get_state_backend() -> "StateBackendProtocol":
    """FastAPI dependency that provides the StateBackend.

    Usage:
        @router.get("/endpoint")
        async def handler(backend: StateBackendProtocol = Depends(get_state_backend)):
            await backend.set("key", b"value", ttl=60)
    """
    return await _get_or_create_state_backend()


async def get_rate_limiter(
    max_requests: int = 100,
    window_seconds: int = 60,
    key_prefix: str = "ratelimit:",
):
    """Get an async rate limiter instance.

    Usage:
        @router.post("/endpoint")
        async def handler(request: Request):
            limiter = await get_rate_limiter(max_requests=100, window_seconds=60)
            client_ip = get_client_ip(request)
            if not await limiter.is_allowed(client_ip):
                raise HTTPException(429, "Too Many Requests")
    """
    from .state_backend.adapters import AsyncRateLimiter

    backend = await _get_or_create_state_backend()
    return AsyncRateLimiter(
        backend,
        max_requests=max_requests,
        window_seconds=window_seconds,
        key_prefix=key_prefix,
    )


async def get_circuit_breaker(name: str, **kwargs):
    """Get an async circuit breaker instance.

    Usage:
        breaker = await get_circuit_breaker("external-api")
        async with breaker:
            result = await call_external_api()
    """
    from .state_backend.adapters import AsyncCircuitBreaker

    backend = await _get_or_create_state_backend()
    return AsyncCircuitBreaker(backend, name, **kwargs)


# --- Convenience methods for common test scenarios ---


def set_job_store(instance: "AsyncJobStore") -> None:
    """Set a specific AsyncJobStore instance (for testing)."""
    container.override("job_store", instance)


def set_secrets_manager(instance: "SecretsManager") -> None:
    """Set a specific SecretsManager instance (for testing)."""
    container.override("secrets_manager", instance)


def set_user_store(instance) -> None:
    """Set a specific UserStore instance (for testing)."""
    container.override("user_store", instance)


def reset_container() -> None:
    """Reset the container to its initial state."""
    container.reset()
    _reset_state_backend()


async def close_state_backend() -> None:
    """Close the state backend on application shutdown.

    Should be called from app shutdown event.
    """
    global _state_backend
    if _state_backend is not None:
        await _state_backend.close()
        _state_backend = None
