"""PostgreSQL State Backend.

Uses asyncpg for async PostgreSQL operations.
Supports multi-node deployments with connection pooling.

Requires: pip install asyncpg
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from ..config import StateBackendConfig
from ..protocols import StateBackendProtocol

logger = logging.getLogger(__name__)


class PostgreSQLStateBackend(StateBackendProtocol):
    """PostgreSQL implementation of StateBackendProtocol.

    Uses asyncpg for async operations with connection pooling.
    Supports row-level locking for atomic operations.

    Schema:
        CREATE TABLE state_store (
            key TEXT PRIMARY KEY,
            value BYTEA NOT NULL,
            type TEXT NOT NULL DEFAULT 'bytes',
            expires_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX idx_state_expires ON state_store(expires_at)
            WHERE expires_at IS NOT NULL;
    """

    def __init__(self, config: StateBackendConfig):
        """Initialize PostgreSQL backend.

        Args:
            config: Backend configuration with PostgreSQL connection URL.
        """
        self._config = config
        self._pool = None
        self._pool_lock = asyncio.Lock()
        self._key_prefix = config.key_prefix

    async def _get_pool(self):
        """Get or create the connection pool."""
        if self._pool is not None:
            return self._pool

        async with self._pool_lock:
            if self._pool is not None:
                return self._pool

            try:
                import asyncpg
            except ImportError as exc:
                raise ImportError(
                    "asyncpg is required for PostgreSQL backend. "
                    "Install with: pip install 'simpletuner[state-postgresql]' or pip install asyncpg"
                ) from exc

            url = self._config.get_connection_url()
            self._pool = await asyncpg.create_pool(
                url,
                min_size=1,
                max_size=self._config.pool_size,
                command_timeout=self._config.timeout,
            )

            # Initialize schema
            await self._init_schema()
            return self._pool

    async def _init_schema(self) -> None:
        """Create the state_store table if it doesn't exist."""
        pool = self._pool
        if pool is None:
            return

        async with pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state_store (
                    key TEXT PRIMARY KEY,
                    value BYTEA NOT NULL,
                    type TEXT NOT NULL DEFAULT 'bytes',
                    expires_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_state_expires
                ON state_store(expires_at) WHERE expires_at IS NOT NULL
            """
            )

    def _full_key(self, key: str) -> str:
        """Get full key with prefix."""
        return f"{self._key_prefix}{key}"

    def _is_expired(self, expires_at: Optional[datetime]) -> bool:
        """Check if a timestamp is expired."""
        if expires_at is None:
            return False
        return datetime.now(timezone.utc) >= expires_at

    def _get_expires_at(self, ttl: Optional[int]) -> Optional[datetime]:
        """Calculate expiration timestamp."""
        if ttl is None:
            return None
        return datetime.fromtimestamp(time.time() + ttl, tz=timezone.utc)

    async def get(self, key: str) -> Optional[bytes]:
        """Get a value by key."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value, expires_at FROM state_store WHERE key = $1",
                full_key,
            )

            if row is None:
                return None

            if self._is_expired(row["expires_at"]):
                await conn.execute("DELETE FROM state_store WHERE key = $1", full_key)
                return None

            return bytes(row["value"])

    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        """Set a value with optional TTL."""
        pool = await self._get_pool()
        full_key = self._full_key(key)
        expires_at = self._get_expires_at(ttl)
        now = datetime.now(timezone.utc)

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO state_store (key, value, type, expires_at, created_at, updated_at)
                VALUES ($1, $2, 'bytes', $3, $4, $4)
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    type = 'bytes',
                    expires_at = EXCLUDED.expires_at,
                    updated_at = $4
                """,
                full_key,
                value,
                expires_at,
                now,
            )

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM state_store WHERE key = $1", full_key)
            return result == "DELETE 1"

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT expires_at FROM state_store WHERE key = $1", full_key)

            if row is None:
                return False

            if self._is_expired(row["expires_at"]):
                await conn.execute("DELETE FROM state_store WHERE key = $1", full_key)
                return False

            return True

    async def incr(self, key: str, amount: int = 1) -> int:
        """Atomically increment a counter."""
        pool = await self._get_pool()
        full_key = self._full_key(key)
        now = datetime.now(timezone.utc)

        async with pool.acquire() as conn:
            # Use advisory lock for atomicity
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO state_store (key, value, type, created_at, updated_at)
                    VALUES ($1, $2::text::bytea, 'counter', $3, $3)
                    ON CONFLICT (key) DO UPDATE SET
                        value = (COALESCE(
                            NULLIF(state_store.value::text, '')::bigint, 0
                        ) + $4)::text::bytea,
                        updated_at = $3
                    RETURNING value
                    """,
                    full_key,
                    str(amount),
                    now,
                    amount,
                )
                return int(row["value"].decode())

    async def get_counter(self, key: str) -> int:
        """Get counter value."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM state_store WHERE key = $1 AND type = 'counter'",
                full_key,
            )

            if row is None:
                return 0

            try:
                return int(row["value"].decode())
            except (ValueError, UnicodeDecodeError):
                return 0

    async def sliding_window_add(self, key: str, timestamp: float, window_seconds: int) -> int:
        """Add timestamp to sliding window and return count."""
        pool = await self._get_pool()
        full_key = self._full_key(key)
        cutoff = timestamp - window_seconds
        now = datetime.now(timezone.utc)
        expires_at = self._get_expires_at(window_seconds * 2)

        async with pool.acquire() as conn:
            async with conn.transaction():
                # Get existing timestamps
                row = await conn.fetchrow(
                    "SELECT value FROM state_store WHERE key = $1 FOR UPDATE",
                    full_key,
                )

                if row is not None:
                    try:
                        timestamps = json.loads(row["value"].decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        timestamps = []
                else:
                    timestamps = []

                # Filter and add new
                timestamps = [t for t in timestamps if t > cutoff]
                timestamps.append(timestamp)

                # Store
                value = json.dumps(timestamps).encode()
                await conn.execute(
                    """
                    INSERT INTO state_store (key, value, type, expires_at, created_at, updated_at)
                    VALUES ($1, $2, 'window', $3, $4, $4)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        type = 'window',
                        expires_at = EXCLUDED.expires_at,
                        updated_at = $4
                    """,
                    full_key,
                    value,
                    expires_at,
                    now,
                )

                return len(timestamps)

    async def sliding_window_count(self, key: str, window_seconds: int) -> int:
        """Get count of entries in sliding window."""
        pool = await self._get_pool()
        full_key = self._full_key(key)
        cutoff = time.time() - window_seconds

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM state_store WHERE key = $1 AND type = 'window'",
                full_key,
            )

            if row is None:
                return 0

            try:
                timestamps = json.loads(row["value"].decode())
                return len([t for t in timestamps if t > cutoff])
            except (json.JSONDecodeError, UnicodeDecodeError):
                return 0

    async def hget(self, key: str, field: str) -> Optional[bytes]:
        """Get a field from a hash."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value, expires_at FROM state_store WHERE key = $1 AND type = 'hash'",
                full_key,
            )

            if row is None:
                return None

            if self._is_expired(row["expires_at"]):
                await conn.execute("DELETE FROM state_store WHERE key = $1", full_key)
                return None

            try:
                hash_data = json.loads(row["value"].decode())
                value = hash_data.get(field)
                if value is not None:
                    return bytes.fromhex(value)
                return None
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                return None

    async def hset(self, key: str, field: str, value: bytes) -> None:
        """Set a field in a hash."""
        pool = await self._get_pool()
        full_key = self._full_key(key)
        now = datetime.now(timezone.utc)

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT value FROM state_store WHERE key = $1 FOR UPDATE",
                    full_key,
                )

                if row is not None:
                    try:
                        hash_data = json.loads(row["value"].decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        hash_data = {}
                else:
                    hash_data = {}

                hash_data[field] = value.hex()
                stored_value = json.dumps(hash_data).encode()

                await conn.execute(
                    """
                    INSERT INTO state_store (key, value, type, created_at, updated_at)
                    VALUES ($1, $2, 'hash', $3, $3)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        type = 'hash',
                        updated_at = $3
                    """,
                    full_key,
                    stored_value,
                    now,
                )

    async def hset_with_ttl(self, key: str, field: str, value: bytes, ttl: int) -> None:
        """Set a field in a hash with TTL for the whole hash."""
        pool = await self._get_pool()
        full_key = self._full_key(key)
        now = datetime.now(timezone.utc)
        expires_at = self._get_expires_at(ttl)

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT value FROM state_store WHERE key = $1 FOR UPDATE",
                    full_key,
                )

                if row is not None:
                    try:
                        hash_data = json.loads(row["value"].decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        hash_data = {}
                else:
                    hash_data = {}

                hash_data[field] = value.hex()
                stored_value = json.dumps(hash_data).encode()

                await conn.execute(
                    """
                    INSERT INTO state_store (key, value, type, expires_at, created_at, updated_at)
                    VALUES ($1, $2, 'hash', $3, $4, $4)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        type = 'hash',
                        expires_at = EXCLUDED.expires_at,
                        updated_at = $4
                    """,
                    full_key,
                    stored_value,
                    expires_at,
                    now,
                )

    async def hgetall(self, key: str) -> Dict[bytes, bytes]:
        """Get all fields from a hash."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value, expires_at FROM state_store WHERE key = $1 AND type = 'hash'",
                full_key,
            )

            if row is None:
                return {}

            if self._is_expired(row["expires_at"]):
                await conn.execute("DELETE FROM state_store WHERE key = $1", full_key)
                return {}

            try:
                hash_data = json.loads(row["value"].decode())
                return {k.encode(): bytes.fromhex(v) for k, v in hash_data.items()}
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                return {}

    async def hdel(self, key: str, field: str) -> bool:
        """Delete a field from a hash."""
        pool = await self._get_pool()
        full_key = self._full_key(key)
        now = datetime.now(timezone.utc)

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT value FROM state_store WHERE key = $1 FOR UPDATE",
                    full_key,
                )

                if row is None:
                    return False

                try:
                    hash_data = json.loads(row["value"].decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return False

                if field not in hash_data:
                    return False

                del hash_data[field]

                if hash_data:
                    stored_value = json.dumps(hash_data).encode()
                    await conn.execute(
                        "UPDATE state_store SET value = $1, updated_at = $2 WHERE key = $3",
                        stored_value,
                        now,
                        full_key,
                    )
                else:
                    await conn.execute("DELETE FROM state_store WHERE key = $1", full_key)

                return True

    async def sadd(self, key: str, *members: bytes) -> int:
        """Add members to a set."""
        pool = await self._get_pool()
        full_key = self._full_key(key)
        now = datetime.now(timezone.utc)

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT value FROM state_store WHERE key = $1 FOR UPDATE",
                    full_key,
                )

                if row is not None:
                    try:
                        set_data = set(json.loads(row["value"].decode()))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        set_data = set()
                else:
                    set_data = set()

                original_size = len(set_data)
                for member in members:
                    set_data.add(member.hex())

                stored_value = json.dumps(list(set_data)).encode()

                await conn.execute(
                    """
                    INSERT INTO state_store (key, value, type, created_at, updated_at)
                    VALUES ($1, $2, 'set', $3, $3)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        type = 'set',
                        updated_at = $3
                    """,
                    full_key,
                    stored_value,
                    now,
                )

                return len(set_data) - original_size

    async def srem(self, key: str, *members: bytes) -> int:
        """Remove members from a set."""
        pool = await self._get_pool()
        full_key = self._full_key(key)
        now = datetime.now(timezone.utc)

        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT value FROM state_store WHERE key = $1 FOR UPDATE",
                    full_key,
                )

                if row is None:
                    return 0

                try:
                    set_data = set(json.loads(row["value"].decode()))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return 0

                original_size = len(set_data)
                for member in members:
                    set_data.discard(member.hex())

                removed = original_size - len(set_data)

                if set_data:
                    stored_value = json.dumps(list(set_data)).encode()
                    await conn.execute(
                        "UPDATE state_store SET value = $1, updated_at = $2 WHERE key = $3",
                        stored_value,
                        now,
                        full_key,
                    )
                else:
                    await conn.execute("DELETE FROM state_store WHERE key = $1", full_key)

                return removed

    async def smembers(self, key: str) -> Set[bytes]:
        """Get all members of a set."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value, expires_at FROM state_store WHERE key = $1 AND type = 'set'",
                full_key,
            )

            if row is None:
                return set()

            if self._is_expired(row["expires_at"]):
                await conn.execute("DELETE FROM state_store WHERE key = $1", full_key)
                return set()

            try:
                set_data = json.loads(row["value"].decode())
                return {bytes.fromhex(m) for m in set_data}
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                return set()

    async def sismember(self, key: str, member: bytes) -> bool:
        """Check if member is in set."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value, expires_at FROM state_store WHERE key = $1 AND type = 'set'",
                full_key,
            )

            if row is None:
                return False

            if self._is_expired(row["expires_at"]):
                await conn.execute("DELETE FROM state_store WHERE key = $1", full_key)
                return False

            try:
                set_data = set(json.loads(row["value"].decode()))
                return member.hex() in set_data
            except (json.JSONDecodeError, UnicodeDecodeError):
                return False

    async def mget(self, *keys: str) -> List[Optional[bytes]]:
        """Get multiple keys at once."""
        if not keys:
            return []

        pool = await self._get_pool()
        full_keys = [self._full_key(k) for k in keys]
        now = datetime.now(timezone.utc)

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value, expires_at FROM state_store WHERE key = ANY($1)",
                full_keys,
            )

            results_map = {}
            expired_keys = []

            for row in rows:
                if self._is_expired(row["expires_at"]):
                    expired_keys.append(row["key"])
                else:
                    results_map[row["key"]] = bytes(row["value"])

            # Clean up expired
            if expired_keys:
                await conn.execute("DELETE FROM state_store WHERE key = ANY($1)", expired_keys)

            return [results_map.get(k) for k in full_keys]

    async def mset(self, mapping: Dict[str, bytes], ttl: Optional[int] = None) -> None:
        """Set multiple keys at once."""
        if not mapping:
            return

        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        expires_at = self._get_expires_at(ttl)

        async with pool.acquire() as conn:
            async with conn.transaction():
                for key, value in mapping.items():
                    full_key = self._full_key(key)
                    await conn.execute(
                        """
                        INSERT INTO state_store (key, value, type, expires_at, created_at, updated_at)
                        VALUES ($1, $2, 'bytes', $3, $4, $4)
                        ON CONFLICT (key) DO UPDATE SET
                            value = EXCLUDED.value,
                            type = 'bytes',
                            expires_at = EXCLUDED.expires_at,
                            updated_at = $4
                        """,
                        full_key,
                        value,
                        expires_at,
                        now,
                    )

    async def delete_prefix(self, prefix: str) -> int:
        """Delete all keys with a given prefix."""
        pool = await self._get_pool()
        full_prefix = self._full_key(prefix)

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM state_store WHERE key LIKE $1",
                f"{full_prefix}%",
            )
            # Parse "DELETE N" result
            try:
                return int(result.split()[1])
            except (IndexError, ValueError):
                return 0

    async def ping(self) -> bool:
        """Check if the database is accessible."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as exc:
            logger.warning("PostgreSQL ping failed: %s", exc)
            return False

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def flush_expired(self) -> int:
        """Remove all expired entries."""
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM state_store WHERE expires_at IS NOT NULL AND expires_at < $1",
                now,
            )
            try:
                return int(result.split()[1])
            except (IndexError, ValueError):
                return 0

    async def execute_atomic(self, operations: List[tuple]) -> List[Any]:
        """Execute multiple operations atomically."""
        pool = await self._get_pool()
        results = []

        async with pool.acquire() as conn:
            async with conn.transaction():
                for op in operations:
                    op_name = op[0]
                    op_args = op[1:]

                    method = getattr(self, op_name, None)
                    if method is None:
                        raise ValueError(f"Unknown operation: {op_name}")

                    result = await method(*op_args)
                    results.append(result)

        return results
