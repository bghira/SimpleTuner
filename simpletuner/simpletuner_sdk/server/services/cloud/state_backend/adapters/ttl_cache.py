"""Async TTL Cache Adapter.

Generic TTL cache using the state backend.
Replaces the threading.RLock-based TTLCache with async operations.
"""

from __future__ import annotations

import pickle
from typing import Callable, Generic, Optional, TypeVar

from ..protocols import StateBackendProtocol

T = TypeVar("T")


class AsyncTTLCache(Generic[T]):
    """Async TTL cache using pluggable state backend.

    Generic cache with configurable TTL and automatic expiration.
    Values are serialized using pickle for storage.

    Example:
        backend = await get_state_backend()
        cache = AsyncTTLCache[dict](backend, ttl_seconds=300, key_prefix="config:")

        # Get or compute
        config = await cache.get_or_set(
            "provider:replicate",
            lambda: fetch_config_from_db(),
        )

        # Simple get/set
        await cache.set("key", {"foo": "bar"})
        value = await cache.get("key")
    """

    def __init__(
        self,
        backend: StateBackendProtocol,
        ttl_seconds: int = 300,
        key_prefix: str = "cache:",
        max_size: Optional[int] = None,
    ):
        """Initialize TTL cache.

        Args:
            backend: State backend instance.
            ttl_seconds: Default TTL in seconds.
            key_prefix: Prefix for cache keys.
            max_size: Optional max entries (not enforced, just advisory).
        """
        self._backend = backend
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix
        self._max_size = max_size

    def _get_key(self, key: str) -> str:
        """Get full key with prefix."""
        return f"{self._key_prefix}{key}"

    def _serialize(self, value: T) -> bytes:
        """Serialize value for storage."""
        return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> T:
        """Deserialize value from storage."""
        return pickle.loads(data)

    async def get(self, key: str) -> Optional[T]:
        """Get value by key.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        full_key = self._get_key(key)
        data = await self._backend.get(full_key)

        if data is None:
            return None

        try:
            return self._deserialize(data)
        except (pickle.PickleError, Exception):
            # Invalid data, delete it
            await self._backend.delete(full_key)
            return None

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[int] = None,
    ) -> None:
        """Set value with TTL.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Optional TTL override in seconds.
        """
        full_key = self._get_key(key)
        data = self._serialize(value)
        await self._backend.set(full_key, data, ttl or self._ttl_seconds)

    async def delete(self, key: str) -> bool:
        """Delete a cached value.

        Args:
            key: Cache key.

        Returns:
            True if key existed and was deleted.
        """
        full_key = self._get_key(key)
        return await self._backend.delete(full_key)

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key.

        Returns:
            True if key exists and not expired.
        """
        full_key = self._get_key(key)
        return await self._backend.exists(full_key)

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[int] = None,
    ) -> T:
        """Get value or compute and cache it.

        Args:
            key: Cache key.
            factory: Function to compute value if not cached.
            ttl: Optional TTL override.

        Returns:
            Cached or computed value.
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        value = factory()
        await self.set(key, value, ttl)
        return value

    async def get_or_set_async(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[int] = None,
    ) -> T:
        """Get value or compute asynchronously and cache it.

        Args:
            key: Cache key.
            factory: Async function to compute value if not cached.
            ttl: Optional TTL override.

        Returns:
            Cached or computed value.
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value asynchronously
        value = await factory()
        await self.set(key, value, ttl)
        return value

    async def invalidate(self, key: str) -> bool:
        """Invalidate a cached value.

        Alias for delete().

        Args:
            key: Cache key.

        Returns:
            True if key existed.
        """
        return await self.delete(key)

    async def invalidate_prefix(self, prefix: str) -> int:
        """Invalidate all keys with given prefix.

        Args:
            prefix: Key prefix (without cache prefix).

        Returns:
            Number of keys invalidated.
        """
        full_prefix = self._get_key(prefix)
        return await self._backend.delete_prefix(full_prefix)

    async def clear(self) -> int:
        """Clear all cached values.

        Returns:
            Number of keys cleared.
        """
        return await self._backend.delete_prefix(self._key_prefix)

    @property
    def ttl_seconds(self) -> int:
        """Get default TTL."""
        return self._ttl_seconds


class AsyncProviderConfigCache(AsyncTTLCache[dict]):
    """Cache for provider configurations.

    5-minute TTL, provider-specific prefix.
    """

    def __init__(
        self,
        backend: StateBackendProtocol,
        ttl_seconds: int = 300,
    ):
        """Initialize provider config cache."""
        super().__init__(
            backend=backend,
            ttl_seconds=ttl_seconds,
            key_prefix="cache:provider:",
            max_size=100,
        )

    async def get_provider_config(self, provider_name: str) -> Optional[dict]:
        """Get cached provider config.

        Args:
            provider_name: Provider name.

        Returns:
            Cached config or None.
        """
        return await self.get(provider_name)

    async def set_provider_config(
        self,
        provider_name: str,
        config: dict,
    ) -> None:
        """Cache provider config.

        Args:
            provider_name: Provider name.
            config: Configuration dict.
        """
        await self.set(provider_name, config)

    async def invalidate_provider(self, provider_name: str) -> bool:
        """Invalidate cached provider config.

        Args:
            provider_name: Provider name.

        Returns:
            True if was cached.
        """
        return await self.invalidate(provider_name)


class AsyncUserPermissionCache(AsyncTTLCache[set]):
    """Cache for user permissions.

    1-minute TTL for quick permission checks.
    """

    def __init__(
        self,
        backend: StateBackendProtocol,
        ttl_seconds: int = 60,
    ):
        """Initialize user permission cache."""
        super().__init__(
            backend=backend,
            ttl_seconds=ttl_seconds,
            key_prefix="cache:perm:",
            max_size=500,
        )

    async def get_permissions(self, user_id: int) -> Optional[set[str]]:
        """Get cached user permissions.

        Args:
            user_id: User ID.

        Returns:
            Set of permission names or None.
        """
        return await self.get(str(user_id))

    async def set_permissions(
        self,
        user_id: int,
        permissions: set[str],
    ) -> None:
        """Cache user permissions.

        Args:
            user_id: User ID.
            permissions: Set of permission names.
        """
        await self.set(str(user_id), permissions)

    async def invalidate_user(self, user_id: int) -> bool:
        """Invalidate cached user permissions.

        Args:
            user_id: User ID.

        Returns:
            True if was cached.
        """
        return await self.invalidate(str(user_id))

    async def invalidate_all_users(self) -> int:
        """Invalidate all cached permissions.

        Returns:
            Number of entries invalidated.
        """
        return await self.clear()
