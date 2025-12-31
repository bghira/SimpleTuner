"""SQLite State Backend.

Default backend for single-node deployments.
Uses aiosqlite for async operations with WAL mode for concurrency.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import aiosqlite

from ..config import StateBackendConfig

SCHEMA_VERSION = 1


class SQLiteStateBackend:
    """SQLite state backend using aiosqlite.

    Features:
        - WAL mode for multi-worker concurrency
        - Async operations via aiosqlite
        - JSON storage for complex types (hash, set, sliding window)
        - Automatic TTL expiration
        - Singleton pattern with async initialization

    Schema:
        state_store(key, value, type, expires_at, created_at, updated_at)
    """

    _instance: Optional["SQLiteStateBackend"] = None
    _init_lock: Optional[asyncio.Lock] = None
    _init_lock_loop: Optional[asyncio.AbstractEventLoop] = None

    def __init__(self, config: Optional[StateBackendConfig] = None):
        """Initialize SQLite backend.

        Args:
            config: Backend configuration. If None, uses defaults.
        """
        self._config = config or StateBackendConfig()
        self._db_path = self._config.get_sqlite_path()
        self._connection: Optional[aiosqlite.Connection] = None
        self._write_lock = asyncio.Lock()
        self._initialized = False

    @classmethod
    async def get_instance(cls, config: Optional[StateBackendConfig] = None) -> "SQLiteStateBackend":
        """Get or create singleton instance.

        Args:
            config: Backend configuration.

        Returns:
            SQLiteStateBackend instance.
        """
        # Fast path - instance already exists
        if cls._instance is not None:
            return cls._instance

        # Recreate lock if bound to a different event loop
        current_loop = asyncio.get_running_loop()
        if cls._init_lock is None or cls._init_lock_loop is not current_loop:
            cls._init_lock = asyncio.Lock()
            cls._init_lock_loop = current_loop

        async with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(config)
                await cls._instance._ensure_initialized()
            return cls._instance

    @classmethod
    async def reset_instance(cls) -> None:
        """Reset singleton (for testing).

        Properly closes the connection before resetting to avoid ResourceWarning.
        """
        # Ensure lock is valid for current event loop
        current_loop = asyncio.get_running_loop()
        if cls._init_lock is None or cls._init_lock_loop is not current_loop:
            cls._init_lock = asyncio.Lock()
            cls._init_lock_loop = current_loop

        async with cls._init_lock:
            if cls._instance is not None:
                await cls._instance.close()
            cls._instance = None

    async def _ensure_initialized(self) -> None:
        """Ensure database is initialized."""
        if self._initialized:
            return

        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Open connection
        self._connection = await aiosqlite.connect(
            str(self._db_path),
            timeout=self._config.timeout,
        )
        self._connection.row_factory = aiosqlite.Row

        # Configure SQLite
        if self._config.sqlite_wal_mode:
            await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA foreign_keys=ON")
        await self._connection.execute("PRAGMA synchronous=NORMAL")
        await self._connection.execute(f"PRAGMA busy_timeout={self._config.sqlite_busy_timeout}")

        # Initialize schema
        await self._init_schema()
        self._initialized = True

    async def _init_schema(self) -> None:
        """Initialize database schema."""
        assert self._connection is not None

        # Create schema version table
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
        """
        )

        # Check current version
        cursor = await self._connection.execute("SELECT version FROM schema_version LIMIT 1")
        row = await cursor.fetchone()
        current_version = row[0] if row else 0

        if current_version < SCHEMA_VERSION:
            await self._run_migrations(current_version)

        await self._connection.commit()

    async def _run_migrations(self, from_version: int) -> None:
        """Run schema migrations."""
        assert self._connection is not None

        if from_version < 1:
            # Initial schema
            await self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS state_store (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    type TEXT NOT NULL DEFAULT 'bytes',
                    expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            )
            await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_state_expires ON state_store(expires_at)")
            await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_state_type ON state_store(type)")

        # Update version
        await self._connection.execute("DELETE FROM schema_version")
        await self._connection.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )

    def _now_iso(self) -> str:
        """Get current time as ISO8601 string."""
        return datetime.now(timezone.utc).isoformat()

    def _is_expired(self, expires_at: Optional[str]) -> bool:
        """Check if entry is expired."""
        if expires_at is None:
            return False
        try:
            exp_time = datetime.fromisoformat(expires_at)
            return datetime.now(timezone.utc) > exp_time
        except ValueError:
            return False

    def _ttl_to_expires(self, ttl: Optional[int]) -> Optional[str]:
        """Convert TTL seconds to expiration timestamp."""
        if ttl is None:
            return None
        exp_time = datetime.now(timezone.utc).timestamp() + ttl
        return datetime.fromtimestamp(exp_time, timezone.utc).isoformat()

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get database connection, ensuring initialization."""
        if not self._initialized:
            await self._ensure_initialized()
        assert self._connection is not None
        return self._connection

    # --- Basic Key-Value Operations ---

    async def get(self, key: str) -> Optional[bytes]:
        """Get a value by key."""
        conn = await self._get_connection()
        prefixed_key = self._config.key_prefix + key

        cursor = await conn.execute(
            "SELECT value, expires_at FROM state_store WHERE key = ?",
            (prefixed_key,),
        )
        row = await cursor.fetchone()

        if row is None:
            return None

        if self._is_expired(row["expires_at"]):
            # Clean up expired entry
            await conn.execute("DELETE FROM state_store WHERE key = ?", (prefixed_key,))
            await conn.commit()
            return None

        return row["value"]

    async def set(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a value with optional TTL."""
        conn = await self._get_connection()
        prefixed_key = self._config.key_prefix + key
        now = self._now_iso()
        expires_at = self._ttl_to_expires(ttl)

        async with self._write_lock:
            await conn.execute(
                """
                INSERT INTO state_store (key, value, type, expires_at, created_at, updated_at)
                VALUES (?, ?, 'bytes', ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (prefixed_key, value, expires_at, now, now),
            )
            await conn.commit()

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        conn = await self._get_connection()
        prefixed_key = self._config.key_prefix + key

        async with self._write_lock:
            cursor = await conn.execute("DELETE FROM state_store WHERE key = ?", (prefixed_key,))
            await conn.commit()
            return cursor.rowcount > 0

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        return await self.get(key) is not None

    # --- Atomic Counter Operations ---

    async def incr(self, key: str, amount: int = 1) -> int:
        """Atomically increment a counter."""
        conn = await self._get_connection()
        prefixed_key = self._config.key_prefix + key
        now = self._now_iso()

        async with self._write_lock:
            # Get current value
            cursor = await conn.execute(
                "SELECT value, expires_at FROM state_store WHERE key = ? AND type = 'counter'",
                (prefixed_key,),
            )
            row = await cursor.fetchone()

            if row is None or self._is_expired(row["expires_at"]):
                # Create new counter
                new_value = amount
                await conn.execute(
                    """
                    INSERT INTO state_store (key, value, type, created_at, updated_at)
                    VALUES (?, ?, 'counter', ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        type = 'counter',
                        updated_at = excluded.updated_at
                    """,
                    (prefixed_key, str(new_value).encode(), now, now),
                )
            else:
                # Increment existing
                current = int(row["value"].decode())
                new_value = current + amount
                await conn.execute(
                    "UPDATE state_store SET value = ?, updated_at = ? WHERE key = ?",
                    (str(new_value).encode(), now, prefixed_key),
                )

            await conn.commit()
            return new_value

    async def get_counter(self, key: str) -> int:
        """Get counter value."""
        value = await self.get(key)
        if value is None:
            return 0
        try:
            return int(value.decode())
        except ValueError:
            return 0

    # --- Sliding Window Operations ---

    async def sliding_window_add(
        self,
        key: str,
        timestamp: float,
        window_seconds: int,
    ) -> int:
        """Add timestamp to sliding window and return count."""
        conn = await self._get_connection()
        prefixed_key = self._config.key_prefix + key
        now = self._now_iso()
        cutoff = timestamp - window_seconds

        async with self._write_lock:
            # Get current window
            cursor = await conn.execute(
                "SELECT value FROM state_store WHERE key = ? AND type = 'window'",
                (prefixed_key,),
            )
            row = await cursor.fetchone()

            if row is None:
                timestamps = []
            else:
                try:
                    timestamps = json.loads(row["value"].decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    timestamps = []

            # Filter out old timestamps and add new one
            timestamps = [ts for ts in timestamps if ts > cutoff]
            timestamps.append(timestamp)

            # Save updated window
            value = json.dumps(timestamps).encode()
            await conn.execute(
                """
                INSERT INTO state_store (key, value, type, expires_at, created_at, updated_at)
                VALUES (?, ?, 'window', NULL, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    type = 'window',
                    updated_at = excluded.updated_at
                """,
                (prefixed_key, value, now, now),
            )
            await conn.commit()

            return len(timestamps)

    async def sliding_window_count(
        self,
        key: str,
        window_seconds: int,
    ) -> int:
        """Count timestamps in sliding window."""
        conn = await self._get_connection()
        prefixed_key = self._config.key_prefix + key
        cutoff = time.time() - window_seconds

        cursor = await conn.execute(
            "SELECT value FROM state_store WHERE key = ? AND type = 'window'",
            (prefixed_key,),
        )
        row = await cursor.fetchone()

        if row is None:
            return 0

        try:
            timestamps = json.loads(row["value"].decode())
            return len([ts for ts in timestamps if ts > cutoff])
        except (json.JSONDecodeError, UnicodeDecodeError):
            return 0

    # --- Hash Operations ---

    async def hget(self, key: str, field: str) -> Optional[bytes]:
        """Get a field from a hash."""
        conn = await self._get_connection()
        prefixed_key = self._config.key_prefix + key

        cursor = await conn.execute(
            "SELECT value, expires_at FROM state_store WHERE key = ? AND type = 'hash'",
            (prefixed_key,),
        )
        row = await cursor.fetchone()

        if row is None or self._is_expired(row["expires_at"]):
            return None

        try:
            hash_data = json.loads(row["value"].decode())
            field_value = hash_data.get(field)
            if field_value is None:
                return None
            # Values stored as base64 or hex for binary safety
            return bytes.fromhex(field_value)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return None

    async def hset(self, key: str, field: str, value: bytes) -> None:
        """Set a field in a hash."""
        await self.hset_with_ttl(key, field, value, None)

    async def hset_with_ttl(
        self,
        key: str,
        field: str,
        value: bytes,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a field in a hash with TTL."""
        conn = await self._get_connection()
        prefixed_key = self._config.key_prefix + key
        now = self._now_iso()
        expires_at = self._ttl_to_expires(ttl)

        async with self._write_lock:
            # Get current hash
            cursor = await conn.execute(
                "SELECT value FROM state_store WHERE key = ? AND type = 'hash'",
                (prefixed_key,),
            )
            row = await cursor.fetchone()

            if row is None:
                hash_data = {}
            else:
                try:
                    hash_data = json.loads(row["value"].decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    hash_data = {}

            # Update field (store as hex for binary safety)
            hash_data[field] = value.hex()

            # Save updated hash
            hash_value = json.dumps(hash_data).encode()
            await conn.execute(
                """
                INSERT INTO state_store (key, value, type, expires_at, created_at, updated_at)
                VALUES (?, ?, 'hash', ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    type = 'hash',
                    expires_at = COALESCE(excluded.expires_at, state_store.expires_at),
                    updated_at = excluded.updated_at
                """,
                (prefixed_key, hash_value, expires_at, now, now),
            )
            await conn.commit()

    async def hgetall(self, key: str) -> Dict[str, bytes]:
        """Get all fields from a hash."""
        conn = await self._get_connection()
        prefixed_key = self._config.key_prefix + key

        cursor = await conn.execute(
            "SELECT value, expires_at FROM state_store WHERE key = ? AND type = 'hash'",
            (prefixed_key,),
        )
        row = await cursor.fetchone()

        if row is None or self._is_expired(row["expires_at"]):
            return {}

        try:
            hash_data = json.loads(row["value"].decode())
            return {k: bytes.fromhex(v) for k, v in hash_data.items()}
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return {}

    async def hdel(self, key: str, field: str) -> bool:
        """Delete a field from a hash."""
        conn = await self._get_connection()
        prefixed_key = self._config.key_prefix + key
        now = self._now_iso()

        async with self._write_lock:
            cursor = await conn.execute(
                "SELECT value FROM state_store WHERE key = ? AND type = 'hash'",
                (prefixed_key,),
            )
            row = await cursor.fetchone()

            if row is None:
                return False

            try:
                hash_data = json.loads(row["value"].decode())
                if field not in hash_data:
                    return False
                del hash_data[field]

                # Save or delete if empty
                if hash_data:
                    hash_value = json.dumps(hash_data).encode()
                    await conn.execute(
                        "UPDATE state_store SET value = ?, updated_at = ? WHERE key = ?",
                        (hash_value, now, prefixed_key),
                    )
                else:
                    await conn.execute("DELETE FROM state_store WHERE key = ?", (prefixed_key,))

                await conn.commit()
                return True
            except (json.JSONDecodeError, UnicodeDecodeError):
                return False

    # --- Set Operations ---

    async def sadd(self, key: str, *members: str) -> int:
        """Add members to a set."""
        conn = await self._get_connection()
        prefixed_key = self._config.key_prefix + key
        now = self._now_iso()

        async with self._write_lock:
            cursor = await conn.execute(
                "SELECT value FROM state_store WHERE key = ? AND type = 'set'",
                (prefixed_key,),
            )
            row = await cursor.fetchone()

            if row is None:
                current_set: set[str] = set()
            else:
                try:
                    current_set = set(json.loads(row["value"].decode()))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    current_set = set()

            original_size = len(current_set)
            current_set.update(members)
            added = len(current_set) - original_size

            set_value = json.dumps(list(current_set)).encode()
            await conn.execute(
                """
                INSERT INTO state_store (key, value, type, expires_at, created_at, updated_at)
                VALUES (?, ?, 'set', NULL, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    type = 'set',
                    updated_at = excluded.updated_at
                """,
                (prefixed_key, set_value, now, now),
            )
            await conn.commit()

            return added

    async def srem(self, key: str, *members: str) -> int:
        """Remove members from a set."""
        conn = await self._get_connection()
        prefixed_key = self._config.key_prefix + key
        now = self._now_iso()

        async with self._write_lock:
            cursor = await conn.execute(
                "SELECT value FROM state_store WHERE key = ? AND type = 'set'",
                (prefixed_key,),
            )
            row = await cursor.fetchone()

            if row is None:
                return 0

            try:
                current_set = set(json.loads(row["value"].decode()))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return 0

            original_size = len(current_set)
            current_set -= set(members)
            removed = original_size - len(current_set)

            if current_set:
                set_value = json.dumps(list(current_set)).encode()
                await conn.execute(
                    "UPDATE state_store SET value = ?, updated_at = ? WHERE key = ?",
                    (set_value, now, prefixed_key),
                )
            else:
                await conn.execute("DELETE FROM state_store WHERE key = ?", (prefixed_key,))

            await conn.commit()
            return removed

    async def smembers(self, key: str) -> set[str]:
        """Get all members of a set."""
        conn = await self._get_connection()
        prefixed_key = self._config.key_prefix + key

        cursor = await conn.execute(
            "SELECT value FROM state_store WHERE key = ? AND type = 'set'",
            (prefixed_key,),
        )
        row = await cursor.fetchone()

        if row is None:
            return set()

        try:
            return set(json.loads(row["value"].decode()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return set()

    async def sismember(self, key: str, member: str) -> bool:
        """Check if member exists in set."""
        members = await self.smembers(key)
        return member in members

    # --- Batch Operations ---

    async def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """Get multiple keys at once."""
        return [await self.get(key) for key in keys]

    async def mset(
        self,
        mapping: Dict[str, bytes],
        ttl: Optional[int] = None,
    ) -> None:
        """Set multiple keys at once."""
        for key, value in mapping.items():
            await self.set(key, value, ttl)

    async def delete_prefix(self, prefix: str) -> int:
        """Delete all keys with given prefix."""
        conn = await self._get_connection()
        full_prefix = self._config.key_prefix + prefix

        async with self._write_lock:
            cursor = await conn.execute(
                "DELETE FROM state_store WHERE key LIKE ?",
                (f"{full_prefix}%",),
            )
            await conn.commit()
            return cursor.rowcount

    # --- Connection Management ---

    async def ping(self) -> bool:
        """Check if backend is healthy."""
        try:
            conn = await self._get_connection()
            await conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close connection."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            self._initialized = False

    async def flush_expired(self) -> int:
        """Remove expired entries."""
        conn = await self._get_connection()
        now = self._now_iso()

        async with self._write_lock:
            cursor = await conn.execute(
                """
                DELETE FROM state_store
                WHERE expires_at IS NOT NULL AND expires_at < ?
                LIMIT ?
                """,
                (now, self._config.cleanup_batch_size),
            )
            await conn.commit()
            return cursor.rowcount

    # --- Transaction Support ---

    async def execute_atomic(
        self,
        operations: List[tuple[str, tuple[Any, ...]]],
    ) -> List[Any]:
        """Execute operations atomically."""
        results = []
        async with self._write_lock:
            conn = await self._get_connection()
            try:
                for method_name, args in operations:
                    method = getattr(self, method_name)
                    result = await method(*args)
                    results.append(result)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return results
