"""MySQL/MariaDB State Backend.

Uses aiomysql for async MySQL operations.
Supports multi-node deployments with connection pooling.

Requires: pip install aiomysql
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from ..config import StateBackendConfig
from ..protocols import StateBackendProtocol

logger = logging.getLogger(__name__)


class MySQLStateBackend(StateBackendProtocol):
    """MySQL/MariaDB implementation of StateBackendProtocol.

    Uses aiomysql for async operations with connection pooling.

    Schema:
        CREATE TABLE state_store (
            `key` VARCHAR(512) PRIMARY KEY,
            value LONGBLOB NOT NULL,
            type VARCHAR(16) NOT NULL DEFAULT 'bytes',
            expires_at DATETIME(6),
            created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
            updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                ON UPDATE CURRENT_TIMESTAMP(6),
            INDEX idx_state_expires (expires_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """

    def __init__(self, config: StateBackendConfig):
        """Initialize MySQL backend.

        Args:
            config: Backend configuration with MySQL connection URL.
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
                import aiomysql
            except ImportError as exc:
                raise ImportError(
                    "aiomysql is required for MySQL backend. "
                    "Install with: pip install 'simpletuner[state-mysql]' or pip install aiomysql"
                ) from exc

            url = self._config.get_connection_url()
            parsed = urlparse(url)

            # Extract connection parameters from URL
            host = parsed.hostname or "localhost"
            port = parsed.port or 3306
            user = parsed.username or "root"
            password = parsed.password or ""
            db = parsed.path.lstrip("/") or "simpletuner"

            self._pool = await aiomysql.create_pool(
                host=host,
                port=port,
                user=user,
                password=password,
                db=db,
                minsize=1,
                maxsize=self._config.pool_size,
                connect_timeout=self._config.timeout,
                autocommit=True,
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
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS state_store (
                        `key` VARCHAR(512) PRIMARY KEY,
                        value LONGBLOB NOT NULL,
                        type VARCHAR(16) NOT NULL DEFAULT 'bytes',
                        expires_at DATETIME(6),
                        created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                        updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                            ON UPDATE CURRENT_TIMESTAMP(6),
                        INDEX idx_state_expires (expires_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                )

    def _full_key(self, key: str) -> str:
        """Get full key with prefix."""
        return f"{self._key_prefix}{key}"

    def _is_expired(self, expires_at: Optional[datetime]) -> bool:
        """Check if a timestamp is expired."""
        if expires_at is None:
            return False
        # MySQL returns naive datetime, compare with naive now
        return datetime.utcnow() >= expires_at

    def _get_expires_at(self, ttl: Optional[int]) -> Optional[datetime]:
        """Calculate expiration timestamp."""
        if ttl is None:
            return None
        return datetime.utcnow().replace(tzinfo=None) + __import__("datetime").timedelta(seconds=ttl)

    async def get(self, key: str) -> Optional[bytes]:
        """Get a value by key."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT value, expires_at FROM state_store WHERE `key` = %s",
                    (full_key,),
                )
                row = await cursor.fetchone()

                if row is None:
                    return None

                value, expires_at = row

                if self._is_expired(expires_at):
                    await cursor.execute("DELETE FROM state_store WHERE `key` = %s", (full_key,))
                    return None

                return bytes(value)

    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        """Set a value with optional TTL."""
        pool = await self._get_pool()
        full_key = self._full_key(key)
        expires_at = self._get_expires_at(ttl)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO state_store (`key`, value, type, expires_at)
                    VALUES (%s, %s, 'bytes', %s)
                    ON DUPLICATE KEY UPDATE
                        value = VALUES(value),
                        type = 'bytes',
                        expires_at = VALUES(expires_at)
                    """,
                    (full_key, value, expires_at),
                )

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("DELETE FROM state_store WHERE `key` = %s", (full_key,))
                return cursor.rowcount > 0

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT expires_at FROM state_store WHERE `key` = %s",
                    (full_key,),
                )
                row = await cursor.fetchone()

                if row is None:
                    return False

                if self._is_expired(row[0]):
                    await cursor.execute("DELETE FROM state_store WHERE `key` = %s", (full_key,))
                    return False

                return True

    async def incr(self, key: str, amount: int = 1) -> int:
        """Atomically increment a counter."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Start transaction
                await cursor.execute("START TRANSACTION")
                try:
                    # Get current value with lock
                    await cursor.execute(
                        "SELECT value FROM state_store WHERE `key` = %s FOR UPDATE",
                        (full_key,),
                    )
                    row = await cursor.fetchone()

                    if row is None:
                        new_value = amount
                        await cursor.execute(
                            """
                            INSERT INTO state_store (`key`, value, type)
                            VALUES (%s, %s, 'counter')
                            """,
                            (full_key, str(new_value).encode()),
                        )
                    else:
                        current = int(row[0].decode())
                        new_value = current + amount
                        await cursor.execute(
                            "UPDATE state_store SET value = %s WHERE `key` = %s",
                            (str(new_value).encode(), full_key),
                        )

                    await cursor.execute("COMMIT")
                    return new_value
                except Exception:
                    await cursor.execute("ROLLBACK")
                    raise

    async def get_counter(self, key: str) -> int:
        """Get counter value."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT value FROM state_store WHERE `key` = %s AND type = 'counter'",
                    (full_key,),
                )
                row = await cursor.fetchone()

                if row is None:
                    return 0

                try:
                    return int(row[0].decode())
                except (ValueError, UnicodeDecodeError):
                    return 0

    async def sliding_window_add(self, key: str, timestamp: float, window_seconds: int) -> int:
        """Add timestamp to sliding window and return count."""
        pool = await self._get_pool()
        full_key = self._full_key(key)
        cutoff = timestamp - window_seconds
        expires_at = self._get_expires_at(window_seconds * 2)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("START TRANSACTION")
                try:
                    await cursor.execute(
                        "SELECT value FROM state_store WHERE `key` = %s FOR UPDATE",
                        (full_key,),
                    )
                    row = await cursor.fetchone()

                    if row is not None:
                        try:
                            timestamps = json.loads(row[0].decode())
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            timestamps = []
                    else:
                        timestamps = []

                    timestamps = [t for t in timestamps if t > cutoff]
                    timestamps.append(timestamp)

                    value = json.dumps(timestamps).encode()

                    await cursor.execute(
                        """
                        INSERT INTO state_store (`key`, value, type, expires_at)
                        VALUES (%s, %s, 'window', %s)
                        ON DUPLICATE KEY UPDATE
                            value = VALUES(value),
                            type = 'window',
                            expires_at = VALUES(expires_at)
                        """,
                        (full_key, value, expires_at),
                    )

                    await cursor.execute("COMMIT")
                    return len(timestamps)
                except Exception:
                    await cursor.execute("ROLLBACK")
                    raise

    async def sliding_window_count(self, key: str, window_seconds: int) -> int:
        """Get count of entries in sliding window."""
        pool = await self._get_pool()
        full_key = self._full_key(key)
        cutoff = time.time() - window_seconds

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT value FROM state_store WHERE `key` = %s AND type = 'window'",
                    (full_key,),
                )
                row = await cursor.fetchone()

                if row is None:
                    return 0

                try:
                    timestamps = json.loads(row[0].decode())
                    return len([t for t in timestamps if t > cutoff])
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return 0

    async def hget(self, key: str, field: str) -> Optional[bytes]:
        """Get a field from a hash."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT value, expires_at FROM state_store WHERE `key` = %s AND type = 'hash'",
                    (full_key,),
                )
                row = await cursor.fetchone()

                if row is None:
                    return None

                value, expires_at = row

                if self._is_expired(expires_at):
                    await cursor.execute("DELETE FROM state_store WHERE `key` = %s", (full_key,))
                    return None

                try:
                    hash_data = json.loads(value.decode())
                    field_value = hash_data.get(field)
                    if field_value is not None:
                        return bytes.fromhex(field_value)
                    return None
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    return None

    async def hset(self, key: str, field: str, value: bytes) -> None:
        """Set a field in a hash."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("START TRANSACTION")
                try:
                    await cursor.execute(
                        "SELECT value FROM state_store WHERE `key` = %s FOR UPDATE",
                        (full_key,),
                    )
                    row = await cursor.fetchone()

                    if row is not None:
                        try:
                            hash_data = json.loads(row[0].decode())
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            hash_data = {}
                    else:
                        hash_data = {}

                    hash_data[field] = value.hex()
                    stored_value = json.dumps(hash_data).encode()

                    await cursor.execute(
                        """
                        INSERT INTO state_store (`key`, value, type)
                        VALUES (%s, %s, 'hash')
                        ON DUPLICATE KEY UPDATE
                            value = VALUES(value),
                            type = 'hash'
                        """,
                        (full_key, stored_value),
                    )

                    await cursor.execute("COMMIT")
                except Exception:
                    await cursor.execute("ROLLBACK")
                    raise

    async def hset_with_ttl(self, key: str, field: str, value: bytes, ttl: int) -> None:
        """Set a field in a hash with TTL for the whole hash."""
        pool = await self._get_pool()
        full_key = self._full_key(key)
        expires_at = self._get_expires_at(ttl)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("START TRANSACTION")
                try:
                    await cursor.execute(
                        "SELECT value FROM state_store WHERE `key` = %s FOR UPDATE",
                        (full_key,),
                    )
                    row = await cursor.fetchone()

                    if row is not None:
                        try:
                            hash_data = json.loads(row[0].decode())
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            hash_data = {}
                    else:
                        hash_data = {}

                    hash_data[field] = value.hex()
                    stored_value = json.dumps(hash_data).encode()

                    await cursor.execute(
                        """
                        INSERT INTO state_store (`key`, value, type, expires_at)
                        VALUES (%s, %s, 'hash', %s)
                        ON DUPLICATE KEY UPDATE
                            value = VALUES(value),
                            type = 'hash',
                            expires_at = VALUES(expires_at)
                        """,
                        (full_key, stored_value, expires_at),
                    )

                    await cursor.execute("COMMIT")
                except Exception:
                    await cursor.execute("ROLLBACK")
                    raise

    async def hgetall(self, key: str) -> Dict[bytes, bytes]:
        """Get all fields from a hash."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT value, expires_at FROM state_store WHERE `key` = %s AND type = 'hash'",
                    (full_key,),
                )
                row = await cursor.fetchone()

                if row is None:
                    return {}

                value, expires_at = row

                if self._is_expired(expires_at):
                    await cursor.execute("DELETE FROM state_store WHERE `key` = %s", (full_key,))
                    return {}

                try:
                    hash_data = json.loads(value.decode())
                    return {k.encode(): bytes.fromhex(v) for k, v in hash_data.items()}
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    return {}

    async def hdel(self, key: str, field: str) -> bool:
        """Delete a field from a hash."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("START TRANSACTION")
                try:
                    await cursor.execute(
                        "SELECT value FROM state_store WHERE `key` = %s FOR UPDATE",
                        (full_key,),
                    )
                    row = await cursor.fetchone()

                    if row is None:
                        await cursor.execute("COMMIT")
                        return False

                    try:
                        hash_data = json.loads(row[0].decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        await cursor.execute("COMMIT")
                        return False

                    if field not in hash_data:
                        await cursor.execute("COMMIT")
                        return False

                    del hash_data[field]

                    if hash_data:
                        stored_value = json.dumps(hash_data).encode()
                        await cursor.execute(
                            "UPDATE state_store SET value = %s WHERE `key` = %s",
                            (stored_value, full_key),
                        )
                    else:
                        await cursor.execute("DELETE FROM state_store WHERE `key` = %s", (full_key,))

                    await cursor.execute("COMMIT")
                    return True
                except Exception:
                    await cursor.execute("ROLLBACK")
                    raise

    async def sadd(self, key: str, *members: bytes) -> int:
        """Add members to a set."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("START TRANSACTION")
                try:
                    await cursor.execute(
                        "SELECT value FROM state_store WHERE `key` = %s FOR UPDATE",
                        (full_key,),
                    )
                    row = await cursor.fetchone()

                    if row is not None:
                        try:
                            set_data = set(json.loads(row[0].decode()))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            set_data = set()
                    else:
                        set_data = set()

                    original_size = len(set_data)
                    for member in members:
                        set_data.add(member.hex())

                    stored_value = json.dumps(list(set_data)).encode()

                    await cursor.execute(
                        """
                        INSERT INTO state_store (`key`, value, type)
                        VALUES (%s, %s, 'set')
                        ON DUPLICATE KEY UPDATE
                            value = VALUES(value),
                            type = 'set'
                        """,
                        (full_key, stored_value),
                    )

                    await cursor.execute("COMMIT")
                    return len(set_data) - original_size
                except Exception:
                    await cursor.execute("ROLLBACK")
                    raise

    async def srem(self, key: str, *members: bytes) -> int:
        """Remove members from a set."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("START TRANSACTION")
                try:
                    await cursor.execute(
                        "SELECT value FROM state_store WHERE `key` = %s FOR UPDATE",
                        (full_key,),
                    )
                    row = await cursor.fetchone()

                    if row is None:
                        await cursor.execute("COMMIT")
                        return 0

                    try:
                        set_data = set(json.loads(row[0].decode()))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        await cursor.execute("COMMIT")
                        return 0

                    original_size = len(set_data)
                    for member in members:
                        set_data.discard(member.hex())

                    removed = original_size - len(set_data)

                    if set_data:
                        stored_value = json.dumps(list(set_data)).encode()
                        await cursor.execute(
                            "UPDATE state_store SET value = %s WHERE `key` = %s",
                            (stored_value, full_key),
                        )
                    else:
                        await cursor.execute("DELETE FROM state_store WHERE `key` = %s", (full_key,))

                    await cursor.execute("COMMIT")
                    return removed
                except Exception:
                    await cursor.execute("ROLLBACK")
                    raise

    async def smembers(self, key: str) -> Set[bytes]:
        """Get all members of a set."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT value, expires_at FROM state_store WHERE `key` = %s AND type = 'set'",
                    (full_key,),
                )
                row = await cursor.fetchone()

                if row is None:
                    return set()

                value, expires_at = row

                if self._is_expired(expires_at):
                    await cursor.execute("DELETE FROM state_store WHERE `key` = %s", (full_key,))
                    return set()

                try:
                    set_data = json.loads(value.decode())
                    return {bytes.fromhex(m) for m in set_data}
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    return set()

    async def sismember(self, key: str, member: bytes) -> bool:
        """Check if member is in set."""
        pool = await self._get_pool()
        full_key = self._full_key(key)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT value, expires_at FROM state_store WHERE `key` = %s AND type = 'set'",
                    (full_key,),
                )
                row = await cursor.fetchone()

                if row is None:
                    return False

                value, expires_at = row

                if self._is_expired(expires_at):
                    await cursor.execute("DELETE FROM state_store WHERE `key` = %s", (full_key,))
                    return False

                try:
                    set_data = set(json.loads(value.decode()))
                    return member.hex() in set_data
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return False

    async def mget(self, *keys: str) -> List[Optional[bytes]]:
        """Get multiple keys at once."""
        if not keys:
            return []

        pool = await self._get_pool()
        full_keys = [self._full_key(k) for k in keys]
        placeholders = ", ".join(["%s"] * len(full_keys))

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    f"SELECT `key`, value, expires_at FROM state_store WHERE `key` IN ({placeholders})",
                    full_keys,
                )
                rows = await cursor.fetchall()

                results_map = {}
                expired_keys = []

                for row in rows:
                    key, value, expires_at = row
                    if self._is_expired(expires_at):
                        expired_keys.append(key)
                    else:
                        results_map[key] = bytes(value)

                if expired_keys:
                    exp_placeholders = ", ".join(["%s"] * len(expired_keys))
                    await cursor.execute(
                        f"DELETE FROM state_store WHERE `key` IN ({exp_placeholders})",
                        expired_keys,
                    )

                return [results_map.get(k) for k in full_keys]

    async def mset(self, mapping: Dict[str, bytes], ttl: Optional[int] = None) -> None:
        """Set multiple keys at once."""
        if not mapping:
            return

        pool = await self._get_pool()
        expires_at = self._get_expires_at(ttl)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("START TRANSACTION")
                try:
                    for key, value in mapping.items():
                        full_key = self._full_key(key)
                        await cursor.execute(
                            """
                            INSERT INTO state_store (`key`, value, type, expires_at)
                            VALUES (%s, %s, 'bytes', %s)
                            ON DUPLICATE KEY UPDATE
                                value = VALUES(value),
                                type = 'bytes',
                                expires_at = VALUES(expires_at)
                            """,
                            (full_key, value, expires_at),
                        )
                    await cursor.execute("COMMIT")
                except Exception:
                    await cursor.execute("ROLLBACK")
                    raise

    async def delete_prefix(self, prefix: str) -> int:
        """Delete all keys with a given prefix."""
        pool = await self._get_pool()
        full_prefix = self._full_key(prefix)

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "DELETE FROM state_store WHERE `key` LIKE %s",
                    (f"{full_prefix}%",),
                )
                return cursor.rowcount

    async def ping(self) -> bool:
        """Check if the database is accessible."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
            return True
        except Exception as exc:
            logger.warning("MySQL ping failed: %s", exc)
            return False

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    async def flush_expired(self) -> int:
        """Remove all expired entries."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("DELETE FROM state_store WHERE expires_at IS NOT NULL AND expires_at < NOW()")
                return cursor.rowcount

    async def execute_atomic(self, operations: List[tuple]) -> List[Any]:
        """Execute multiple operations atomically."""
        pool = await self._get_pool()
        results = []

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("START TRANSACTION")
                try:
                    for op in operations:
                        op_name = op[0]
                        op_args = op[1:]

                        method = getattr(self, op_name, None)
                        if method is None:
                            raise ValueError(f"Unknown operation: {op_name}")

                        result = await method(*op_args)
                        results.append(result)

                    await cursor.execute("COMMIT")
                except Exception:
                    await cursor.execute("ROLLBACK")
                    raise

        return results
