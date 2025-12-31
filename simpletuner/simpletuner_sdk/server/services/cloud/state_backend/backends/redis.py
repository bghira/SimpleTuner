"""Redis State Backend.

Uses redis.asyncio for async Redis operations.
Optimal choice for multi-node deployments with native support
for all operations (hashes, sets, sliding windows, TTL).

Requires: pip install redis
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set

from ..config import StateBackendConfig
from ..protocols import StateBackendProtocol

logger = logging.getLogger(__name__)


class RedisStateBackend(StateBackendProtocol):
    """Redis implementation of StateBackendProtocol.

    Uses redis.asyncio (redis-py 5+) for native async operations.
    Most efficient backend for multi-node deployments as Redis
    natively supports all required operations.
    """

    def __init__(self, config: StateBackendConfig):
        """Initialize Redis backend.

        Args:
            config: Backend configuration with Redis connection URL.
        """
        self._config = config
        self._client = None
        self._init_lock = asyncio.Lock()
        self._key_prefix = config.key_prefix

    async def _get_client(self):
        """Get or create the Redis client."""
        if self._client is not None:
            return self._client

        async with self._init_lock:
            if self._client is not None:
                return self._client

            try:
                import redis.asyncio as redis
            except ImportError as exc:
                raise ImportError(
                    "redis is required for Redis backend. "
                    "Install with: pip install 'simpletuner[state-redis]' or pip install redis"
                ) from exc

            url = self._config.get_connection_url()

            # Create connection pool
            self._client = redis.from_url(
                url,
                max_connections=self._config.pool_size,
                socket_timeout=self._config.timeout,
                socket_connect_timeout=self._config.timeout,
                decode_responses=False,  # We handle bytes
            )

            return self._client

    def _full_key(self, key: str) -> str:
        """Get full key with prefix."""
        return f"{self._key_prefix}{key}"

    async def get(self, key: str) -> Optional[bytes]:
        """Get a value by key."""
        client = await self._get_client()
        full_key = self._full_key(key)
        return await client.get(full_key)

    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        """Set a value with optional TTL."""
        client = await self._get_client()
        full_key = self._full_key(key)

        if ttl is not None:
            await client.setex(full_key, ttl, value)
        else:
            await client.set(full_key, value)

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        client = await self._get_client()
        full_key = self._full_key(key)
        result = await client.delete(full_key)
        return result > 0

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        client = await self._get_client()
        full_key = self._full_key(key)
        result = await client.exists(full_key)
        return result > 0

    async def incr(self, key: str, amount: int = 1) -> int:
        """Atomically increment a counter."""
        client = await self._get_client()
        full_key = self._full_key(key)
        return await client.incrby(full_key, amount)

    async def get_counter(self, key: str) -> int:
        """Get counter value."""
        client = await self._get_client()
        full_key = self._full_key(key)
        value = await client.get(full_key)

        if value is None:
            return 0

        try:
            return int(value)
        except ValueError:
            return 0

    async def sliding_window_add(self, key: str, timestamp: float, window_seconds: int) -> int:
        """Add timestamp to sliding window and return count.

        Uses Redis sorted set (ZSET) for efficient sliding window.
        Score = timestamp, member = unique identifier.
        """
        client = await self._get_client()
        full_key = self._full_key(key)
        cutoff = timestamp - window_seconds

        # Use pipeline for atomicity
        pipe = client.pipeline()

        # Remove old entries
        pipe.zremrangebyscore(full_key, "-inf", cutoff)

        # Add new entry (use timestamp as both score and member for uniqueness)
        # Append random bytes to ensure uniqueness
        import os

        unique_member = f"{timestamp}:{os.urandom(8).hex()}".encode()
        pipe.zadd(full_key, {unique_member: timestamp})

        # Get count
        pipe.zcard(full_key)

        # Set TTL to clean up eventually
        pipe.expire(full_key, window_seconds * 2)

        results = await pipe.execute()
        return results[2]  # zcard result

    async def sliding_window_count(self, key: str, window_seconds: int) -> int:
        """Get count of entries in sliding window."""
        client = await self._get_client()
        full_key = self._full_key(key)
        cutoff = time.time() - window_seconds

        # Count entries within window
        return await client.zcount(full_key, cutoff, "+inf")

    async def hget(self, key: str, field: str) -> Optional[bytes]:
        """Get a field from a hash."""
        client = await self._get_client()
        full_key = self._full_key(key)
        return await client.hget(full_key, field)

    async def hset(self, key: str, field: str, value: bytes) -> None:
        """Set a field in a hash."""
        client = await self._get_client()
        full_key = self._full_key(key)
        await client.hset(full_key, field, value)

    async def hset_with_ttl(self, key: str, field: str, value: bytes, ttl: int) -> None:
        """Set a field in a hash with TTL for the whole hash."""
        client = await self._get_client()
        full_key = self._full_key(key)

        pipe = client.pipeline()
        pipe.hset(full_key, field, value)
        pipe.expire(full_key, ttl)
        await pipe.execute()

    async def hgetall(self, key: str) -> Dict[bytes, bytes]:
        """Get all fields from a hash."""
        client = await self._get_client()
        full_key = self._full_key(key)
        return await client.hgetall(full_key)

    async def hdel(self, key: str, field: str) -> bool:
        """Delete a field from a hash."""
        client = await self._get_client()
        full_key = self._full_key(key)
        result = await client.hdel(full_key, field)
        return result > 0

    async def sadd(self, key: str, *members: bytes) -> int:
        """Add members to a set."""
        if not members:
            return 0
        client = await self._get_client()
        full_key = self._full_key(key)
        return await client.sadd(full_key, *members)

    async def srem(self, key: str, *members: bytes) -> int:
        """Remove members from a set."""
        if not members:
            return 0
        client = await self._get_client()
        full_key = self._full_key(key)
        return await client.srem(full_key, *members)

    async def smembers(self, key: str) -> Set[bytes]:
        """Get all members of a set."""
        client = await self._get_client()
        full_key = self._full_key(key)
        return await client.smembers(full_key)

    async def sismember(self, key: str, member: bytes) -> bool:
        """Check if member is in set."""
        client = await self._get_client()
        full_key = self._full_key(key)
        return await client.sismember(full_key, member)

    async def mget(self, *keys: str) -> List[Optional[bytes]]:
        """Get multiple keys at once."""
        if not keys:
            return []
        client = await self._get_client()
        full_keys = [self._full_key(k) for k in keys]
        return await client.mget(*full_keys)

    async def mset(self, mapping: Dict[str, bytes], ttl: Optional[int] = None) -> None:
        """Set multiple keys at once."""
        if not mapping:
            return

        client = await self._get_client()
        prefixed = {self._full_key(k): v for k, v in mapping.items()}

        if ttl is None:
            await client.mset(prefixed)
        else:
            # MSET doesn't support TTL, use pipeline
            pipe = client.pipeline()
            for key, value in prefixed.items():
                pipe.setex(key, ttl, value)
            await pipe.execute()

    async def delete_prefix(self, prefix: str) -> int:
        """Delete all keys with a given prefix.

        Note: Uses SCAN for safety (doesn't block server).
        """
        client = await self._get_client()
        full_prefix = self._full_key(prefix)
        count = 0

        # Use SCAN to find keys (non-blocking)
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=f"{full_prefix}*", count=100)

            if keys:
                deleted = await client.delete(*keys)
                count += deleted

            if cursor == 0:
                break

        return count

    async def ping(self) -> bool:
        """Check if Redis is accessible."""
        try:
            client = await self._get_client()
            return await client.ping()
        except Exception as exc:
            logger.warning("Redis ping failed: %s", exc)
            return False

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def flush_expired(self) -> int:
        """Remove all expired entries.

        Redis handles TTL automatically, so this is a no-op.
        Returns 0 as Redis handles expiration internally.
        """
        # Redis handles TTL-based expiration automatically
        return 0

    async def execute_atomic(self, operations: List[tuple]) -> List[Any]:
        """Execute multiple operations atomically using MULTI/EXEC.

        Note: For complex operations, this uses individual calls within
        a pipeline. True atomicity requires Lua scripts for complex ops.
        """
        client = await self._get_client()
        results = []

        # For simple get/set/delete, we can use pipeline
        pipe = client.pipeline()

        for op in operations:
            op_name = op[0]
            op_args = op[1:]

            if op_name == "get":
                pipe.get(self._full_key(op_args[0]))
            elif op_name == "set":
                key, value = op_args[0], op_args[1]
                ttl = op_args[2] if len(op_args) > 2 else None
                if ttl:
                    pipe.setex(self._full_key(key), ttl, value)
                else:
                    pipe.set(self._full_key(key), value)
            elif op_name == "delete":
                pipe.delete(self._full_key(op_args[0]))
            elif op_name == "incr":
                key = op_args[0]
                amount = op_args[1] if len(op_args) > 1 else 1
                pipe.incrby(self._full_key(key), amount)
            elif op_name == "hget":
                pipe.hget(self._full_key(op_args[0]), op_args[1])
            elif op_name == "hset":
                pipe.hset(self._full_key(op_args[0]), op_args[1], op_args[2])
            else:
                # For complex operations, fall back to sequential calls
                method = getattr(self, op_name, None)
                if method is None:
                    raise ValueError(f"Unknown operation: {op_name}")

                # Execute pipeline so far
                if len(pipe) > 0:
                    results.extend(await pipe.execute())
                    pipe = client.pipeline()

                result = await method(*op_args)
                results.append(result)
                continue

        # Execute remaining pipeline
        if len(pipe) > 0:
            results.extend(await pipe.execute())

        return results

    # Redis-specific optimized methods

    async def set_with_nx(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Set value only if key doesn't exist (NX).

        Useful for distributed locks.

        Args:
            key: Key to set.
            value: Value to set.
            ttl: Optional TTL in seconds.

        Returns:
            True if set (key didn't exist), False otherwise.
        """
        client = await self._get_client()
        full_key = self._full_key(key)

        if ttl is not None:
            return await client.set(full_key, value, nx=True, ex=ttl)
        return await client.set(full_key, value, nx=True)

    async def set_with_xx(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Set value only if key exists (XX).

        Args:
            key: Key to set.
            value: Value to set.
            ttl: Optional TTL in seconds.

        Returns:
            True if set (key existed), False otherwise.
        """
        client = await self._get_client()
        full_key = self._full_key(key)

        if ttl is not None:
            return await client.set(full_key, value, xx=True, ex=ttl)
        return await client.set(full_key, value, xx=True)

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on an existing key.

        Args:
            key: Key to expire.
            ttl: TTL in seconds.

        Returns:
            True if TTL was set, False if key doesn't exist.
        """
        client = await self._get_client()
        full_key = self._full_key(key)
        return await client.expire(full_key, ttl)

    async def ttl(self, key: str) -> int:
        """Get remaining TTL for a key.

        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist.
        """
        client = await self._get_client()
        full_key = self._full_key(key)
        return await client.ttl(full_key)
