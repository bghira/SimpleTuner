"""State Backend Protocol Definition.

Defines the interface for pluggable state storage backends.
Supports SQLite, PostgreSQL, MySQL, Redis, and future backends.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class StateBackendProtocol(Protocol):
    """Protocol for pluggable state storage backends.

    All methods are async-first for FastAPI compatibility.
    Implementations MUST be thread-safe for multi-worker deployments.

    Supported backends:
        - SQLite (default, single-node)
        - PostgreSQL (multi-node, connection pooling)
        - MySQL/MariaDB (multi-node, connection pooling)
        - Redis (distributed cache, native TTL)
        - Memory (testing only)
    """

    # --- Basic Key-Value Operations ---

    async def get(self, key: str) -> Optional[bytes]:
        """Get a value by key.

        Args:
            key: The key to retrieve.

        Returns:
            The value as bytes, or None if key does not exist or has expired.
        """
        ...

    async def set(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a value with optional TTL in seconds.

        Args:
            key: The key to set.
            value: The value as bytes.
            ttl: Time-to-live in seconds. None means no expiration.

        If key exists, overwrites it.
        """
        ...

    async def delete(self, key: str) -> bool:
        """Delete a key.

        Args:
            key: The key to delete.

        Returns:
            True if key existed and was deleted, False otherwise.
        """
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired.

        Args:
            key: The key to check.

        Returns:
            True if key exists and is valid, False otherwise.
        """
        ...

    # --- Atomic Counter Operations (for rate limiting) ---

    async def incr(self, key: str, ttl: Optional[int] = None) -> int:
        """Atomically increment a counter.

        Creates key with value 1 if not exists.
        If TTL is provided and key is new, sets expiration.

        Args:
            key: The counter key.
            ttl: Optional TTL for new keys.

        Returns:
            The new counter value.
        """
        ...

    async def get_counter(self, key: str) -> int:
        """Get counter value.

        Args:
            key: The counter key.

        Returns:
            The counter value, or 0 if key doesn't exist.
        """
        ...

    # --- Sliding Window Operations (for rate limiting) ---

    async def sliding_window_add(
        self,
        key: str,
        timestamp: float,
        window_seconds: int,
    ) -> int:
        """Add a timestamp to a sliding window and return count in window.

        Automatically removes timestamps older than window_seconds.

        Args:
            key: The window key.
            timestamp: The timestamp to add (typically time.time()).
            window_seconds: The window size in seconds.

        Returns:
            The count of timestamps in the current window (including the new one).
        """
        ...

    async def sliding_window_count(
        self,
        key: str,
        window_seconds: int,
    ) -> int:
        """Count timestamps in the current sliding window.

        Args:
            key: The window key.
            window_seconds: The window size in seconds.

        Returns:
            The count of timestamps in the current window.
        """
        ...

    # --- Hash Operations (for structured data) ---

    async def hget(self, key: str, field: str) -> Optional[bytes]:
        """Get a field from a hash.

        Args:
            key: The hash key.
            field: The field name.

        Returns:
            The field value as bytes, or None if not found.
        """
        ...

    async def hset(self, key: str, field: str, value: bytes) -> None:
        """Set a field in a hash.

        Args:
            key: The hash key.
            field: The field name.
            value: The field value as bytes.
        """
        ...

    async def hgetall(self, key: str) -> Dict[str, bytes]:
        """Get all fields and values from a hash.

        Args:
            key: The hash key.

        Returns:
            Dictionary of field -> value mappings.
        """
        ...

    async def hdel(self, key: str, field: str) -> bool:
        """Delete a field from a hash.

        Args:
            key: The hash key.
            field: The field name.

        Returns:
            True if field existed and was deleted.
        """
        ...

    async def hset_with_ttl(
        self,
        key: str,
        field: str,
        value: bytes,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a field in a hash with TTL on the entire hash.

        Args:
            key: The hash key.
            field: The field name.
            value: The field value as bytes.
            ttl: TTL for the entire hash key.
        """
        ...

    # --- Set Operations (for tracking) ---

    async def sadd(self, key: str, *members: str) -> int:
        """Add members to a set.

        Args:
            key: The set key.
            *members: Members to add.

        Returns:
            Number of new members added.
        """
        ...

    async def srem(self, key: str, *members: str) -> int:
        """Remove members from a set.

        Args:
            key: The set key.
            *members: Members to remove.

        Returns:
            Number of members removed.
        """
        ...

    async def smembers(self, key: str) -> set[str]:
        """Get all members of a set.

        Args:
            key: The set key.

        Returns:
            Set of all members.
        """
        ...

    async def sismember(self, key: str, member: str) -> bool:
        """Check if member exists in set.

        Args:
            key: The set key.
            member: The member to check.

        Returns:
            True if member exists in set.
        """
        ...

    # --- Batch Operations ---

    async def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """Get multiple keys at once.

        Args:
            keys: Sequence of keys to retrieve.

        Returns:
            List of values in same order as keys. None for missing keys.
        """
        ...

    async def mset(
        self,
        mapping: Dict[str, bytes],
        ttl: Optional[int] = None,
    ) -> None:
        """Set multiple keys at once with optional TTL.

        Args:
            mapping: Dictionary of key -> value.
            ttl: Optional TTL for all keys.
        """
        ...

    async def delete_prefix(self, prefix: str) -> int:
        """Delete all keys with given prefix.

        Args:
            prefix: The key prefix to match.

        Returns:
            Number of keys deleted.
        """
        ...

    # --- Connection Management ---

    async def ping(self) -> bool:
        """Check if backend is healthy and responsive.

        Returns:
            True if backend is healthy.

        Raises:
            Exception if backend is unreachable.
        """
        ...

    async def close(self) -> None:
        """Close connections and cleanup resources."""
        ...

    async def flush_expired(self) -> int:
        """Remove expired entries.

        Called periodically by background task.

        Returns:
            Number of entries removed.
        """
        ...

    # --- Transaction Support (optional) ---

    async def execute_atomic(
        self,
        operations: List[tuple[str, tuple[Any, ...]]],
    ) -> List[Any]:
        """Execute multiple operations atomically.

        Each operation is (method_name, args_tuple).

        Args:
            operations: List of (method_name, args) tuples.

        Returns:
            List of results from each operation.

        Raises:
            Exception on failure (rolls back all operations).

        Note:
            Not all backends support this; may fall back to sequential execution.
        """
        ...
