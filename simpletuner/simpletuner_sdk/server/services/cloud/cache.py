"""In-memory caching with TTL support.

Provides a simple, thread-safe caching layer for frequently accessed data
like provider configurations, user permissions, and quota settings.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cached value with expiration time."""

    value: T
    expires_at: float
    created_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return time.time() >= self.expires_at

    @property
    def ttl_remaining(self) -> float:
        """Get remaining TTL in seconds."""
        return max(0, self.expires_at - time.time())


class TTLCache(Generic[T]):
    """Thread-safe in-memory cache with TTL support.

    Features:
    - Per-key TTL
    - Automatic cleanup of expired entries
    - Size limit with LRU eviction
    - Thread-safe operations

    Usage:
        cache = TTLCache[str](default_ttl=60)
        cache.set("key", "value")
        value = cache.get("key")  # Returns "value" or None if expired
    """

    def __init__(
        self,
        default_ttl: float = 300.0,  # 5 minutes
        max_size: int = 1000,
        cleanup_interval: float = 60.0,
    ):
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

    def get(self, key: str) -> Optional[T]:
        """Get a value from the cache.

        Returns None if key doesn't exist or has expired.
        """
        with self._lock:
            self._maybe_cleanup()

            entry = self._cache.get(key)
            if entry is None:
                return None

            if entry.is_expired:
                del self._cache[key]
                return None

            return entry.value

    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds (uses default if not specified)
        """
        with self._lock:
            self._maybe_cleanup()
            self._maybe_evict()

            effective_ttl = ttl if ttl is not None else self._default_ttl
            expires_at = time.time() + effective_ttl

            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> bool:
        """Delete a key from the cache.

        Returns True if key was deleted, False if it didn't exist.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all entries from the cache.

        Returns the number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def invalidate_prefix(self, prefix: str) -> int:
        """Invalidate all keys starting with a prefix.

        Returns the number of entries invalidated.
        """
        with self._lock:
            keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[float] = None,
    ) -> T:
        """Get a value or compute and cache it if missing.

        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl: Optional TTL for newly computed values

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        # Compute and cache
        value = factory()
        self.set(key, value, ttl)
        return value

    async def get_or_set_async(
        self,
        key: str,
        factory: Callable[[], Any],  # Actually Awaitable[T]
        ttl: Optional[float] = None,
    ) -> T:
        """Async version of get_or_set.

        Args:
            key: Cache key
            factory: Async function to compute value if not cached
            ttl: Optional TTL for newly computed values

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        # Compute and cache
        value = await factory()
        self.set(key, value, ttl)
        return value

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        expired_keys = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug("Cleaned up %d expired cache entries", len(expired_keys))

    def _maybe_evict(self) -> None:
        """Evict oldest entries if cache is at capacity."""
        if len(self._cache) < self._max_size:
            return

        # Sort by created_at and remove oldest 10%
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at,
        )
        evict_count = max(1, len(sorted_keys) // 10)

        for key in sorted_keys[:evict_count]:
            del self._cache[key]

        logger.debug("Evicted %d cache entries (at capacity)", evict_count)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            expired_count = sum(1 for v in self._cache.values() if v.is_expired)
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "expired_count": expired_count,
                "default_ttl": self._default_ttl,
            }


# Global cache instances
_provider_config_cache: Optional[TTLCache[Dict[str, Any]]] = None
_user_permission_cache: Optional[TTLCache[Any]] = None


def get_provider_config_cache() -> TTLCache[Dict[str, Any]]:
    """Get the global provider config cache."""
    global _provider_config_cache
    if _provider_config_cache is None:
        # Provider configs change rarely, use longer TTL
        _provider_config_cache = TTLCache(default_ttl=300.0, max_size=100)
    return _provider_config_cache


def get_user_permission_cache() -> TTLCache[Any]:
    """Get the global user permission cache."""
    global _user_permission_cache
    if _user_permission_cache is None:
        # Permissions can change, use shorter TTL
        _user_permission_cache = TTLCache(default_ttl=60.0, max_size=500)
    return _user_permission_cache


def cached(
    ttl: float = 60.0,
    key_prefix: str = "",
) -> Callable:
    """Decorator for caching function results.

    The cache key is generated from the function name and arguments.

    Usage:
        @cached(ttl=120)
        def get_user(user_id: int) -> User:
            ...

        @cached(ttl=60)
        async def fetch_data() -> dict:
            ...
    """
    cache: TTLCache[Any] = TTLCache(default_ttl=ttl, max_size=1000)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Build cache key from function name and args
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            result = await func(*args, **kwargs)
            cache.set(cache_key, result)
            return result

        if asyncio.iscoroutinefunction(func):
            async_wrapper._cache = cache  # Expose for testing/invalidation
            return async_wrapper
        else:
            sync_wrapper._cache = cache
            return sync_wrapper

    return decorator


def invalidate_user_cache(user_id: int) -> None:
    """Invalidate all cached data for a user."""
    permission_cache = get_user_permission_cache()
    permission_cache.invalidate_prefix(f"user:{user_id}:")


def invalidate_provider_cache(provider: str) -> None:
    """Invalidate cached config for a provider."""
    config_cache = get_provider_config_cache()
    config_cache.invalidate_prefix(f"provider:{provider}:")
