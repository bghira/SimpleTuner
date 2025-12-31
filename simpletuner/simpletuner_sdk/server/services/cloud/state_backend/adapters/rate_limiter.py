"""Async Rate Limiter Adapter.

Provides sliding window rate limiting using the state backend.
Replaces the threading.Lock-based RateLimiter with async operations.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

from ..protocols import StateBackendProtocol


class AsyncRateLimiter:
    """Sliding window rate limiter using pluggable state backend.

    Implements the sliding window algorithm for rate limiting.
    Each key tracks timestamps of requests within the window.

    Example:
        backend = await get_state_backend()
        limiter = AsyncRateLimiter(backend, max_requests=100, window_seconds=60)

        # Check if request is allowed
        if await limiter.is_allowed("user:123"):
            # Process request
            pass
        else:
            # Rate limited
            raise HTTPException(429, "Too Many Requests")
    """

    def __init__(
        self,
        backend: StateBackendProtocol,
        max_requests: int = 100,
        window_seconds: int = 60,
        key_prefix: str = "ratelimit:",
    ):
        """Initialize rate limiter.

        Args:
            backend: State backend instance.
            max_requests: Maximum requests allowed in window.
            window_seconds: Window size in seconds.
            key_prefix: Prefix for rate limit keys.
        """
        self._backend = backend
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._key_prefix = key_prefix

    def _get_key(self, identifier: str) -> str:
        """Get full key for identifier."""
        return f"{self._key_prefix}{identifier}"

    async def is_allowed(self, identifier: str) -> bool:
        """Check if a request is allowed under the rate limit.

        Atomically adds current timestamp and checks count.

        Args:
            identifier: Unique identifier (IP, user ID, API key, etc.)

        Returns:
            True if request is allowed, False if rate limited.
        """
        key = self._get_key(identifier)
        count = await self._backend.sliding_window_add(
            key,
            time.time(),
            self._window_seconds,
        )
        return count <= self._max_requests

    async def check_and_update(self, identifier: str) -> Tuple[bool, int, int]:
        """Check if allowed and return details.

        Args:
            identifier: Unique identifier.

        Returns:
            Tuple of (is_allowed, current_count, remaining).
        """
        key = self._get_key(identifier)
        count = await self._backend.sliding_window_add(
            key,
            time.time(),
            self._window_seconds,
        )
        is_allowed = count <= self._max_requests
        remaining = max(0, self._max_requests - count)
        return is_allowed, count, remaining

    async def get_remaining(self, identifier: str) -> int:
        """Get remaining requests in current window.

        Does not count as a request.

        Args:
            identifier: Unique identifier.

        Returns:
            Number of remaining requests allowed.
        """
        key = self._get_key(identifier)
        count = await self._backend.sliding_window_count(key, self._window_seconds)
        return max(0, self._max_requests - count)

    async def get_current_count(self, identifier: str) -> int:
        """Get current request count in window.

        Args:
            identifier: Unique identifier.

        Returns:
            Current number of requests in window.
        """
        key = self._get_key(identifier)
        return await self._backend.sliding_window_count(key, self._window_seconds)

    async def reset(self, identifier: str) -> bool:
        """Reset rate limit for identifier.

        Args:
            identifier: Unique identifier.

        Returns:
            True if key existed and was reset.
        """
        key = self._get_key(identifier)
        return await self._backend.delete(key)

    async def reset_all(self) -> int:
        """Reset all rate limits.

        Returns:
            Number of keys deleted.
        """
        return await self._backend.delete_prefix(self._key_prefix)

    @property
    def max_requests(self) -> int:
        """Get maximum requests per window."""
        return self._max_requests

    @property
    def window_seconds(self) -> int:
        """Get window size in seconds."""
        return self._window_seconds


class AsyncIPRateLimiter(AsyncRateLimiter):
    """IP-based rate limiter with sensible defaults.

    Convenience class for common IP-based rate limiting.
    """

    def __init__(
        self,
        backend: StateBackendProtocol,
        max_requests: int = 100,
        window_seconds: int = 60,
    ):
        """Initialize IP rate limiter."""
        super().__init__(
            backend=backend,
            max_requests=max_requests,
            window_seconds=window_seconds,
            key_prefix="ratelimit:ip:",
        )

    async def is_allowed_ip(self, ip_address: str) -> bool:
        """Check if IP is allowed.

        Args:
            ip_address: Client IP address.

        Returns:
            True if allowed.
        """
        # Normalize IPv6 addresses
        if ip_address.startswith("::ffff:"):
            ip_address = ip_address[7:]
        return await self.is_allowed(ip_address)


class AsyncUserRateLimiter(AsyncRateLimiter):
    """User-based rate limiter.

    Rate limits per user ID.
    """

    def __init__(
        self,
        backend: StateBackendProtocol,
        max_requests: int = 1000,
        window_seconds: int = 60,
    ):
        """Initialize user rate limiter."""
        super().__init__(
            backend=backend,
            max_requests=max_requests,
            window_seconds=window_seconds,
            key_prefix="ratelimit:user:",
        )

    async def is_allowed_user(self, user_id: int) -> bool:
        """Check if user is allowed.

        Args:
            user_id: User ID.

        Returns:
            True if allowed.
        """
        return await self.is_allowed(str(user_id))
