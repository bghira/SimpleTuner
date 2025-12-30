"""Async base class for SQLite-backed storage using aiosqlite.

Provides native async database access without run_in_executor overhead.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, TypeVar

try:
    import aiosqlite
except ImportError:
    aiosqlite = None  # type: ignore

from .base import get_default_config_dir, get_default_db_path

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncSQLiteStore:
    """Base class for async SQLite-backed storage using aiosqlite.

    Provides:
    - Native async database operations (no thread pool overhead)
    - Connection pooling via connection reuse
    - WAL mode for better concurrency
    - Transaction management
    - Busy timeout for multi-worker scenarios

    Migration from BaseSQLiteStore:
    - Replace `with self.transaction()` with `async with self.transaction()`
    - Replace `cursor.execute()` with `await cursor.execute()`
    - Replace `cursor.fetchone()` with `await cursor.fetchone()`
    - Replace `cursor.fetchall()` with `await cursor.fetchall()`
    """

    _instances: Dict[type, "AsyncSQLiteStore"] = {}
    _init_lock: Optional[asyncio.Lock] = None
    _init_lock_loop: Optional[asyncio.AbstractEventLoop] = None

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the store.

        Note: Actual async initialization happens in ensure_initialized().
        Call `await store.ensure_initialized()` before first use.

        Args:
            db_path: Path to the SQLite database.
        """
        self._db_path = Path(db_path) if db_path else self._get_default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[aiosqlite.Connection] = None
        self._initialized = False
        self._write_lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls, db_path: Optional[Path] = None) -> "AsyncSQLiteStore":
        """Get or create the singleton instance.

        This is the preferred way to get a store instance as it ensures
        async initialization is complete.
        """
        # Fast path - instance already exists
        if cls in cls._instances:
            return cls._instances[cls]

        # Recreate lock if bound to a different event loop
        current_loop = asyncio.get_running_loop()
        if cls._init_lock is None or cls._init_lock_loop is not current_loop:
            cls._init_lock = asyncio.Lock()
            cls._init_lock_loop = current_loop

        async with cls._init_lock:
            if cls not in cls._instances:
                instance = cls(db_path)
                await instance.ensure_initialized()
                cls._instances[cls] = instance
            return cls._instances[cls]

    def _get_default_db_path(self) -> Path:
        """Get the default database path for this store.

        Subclasses should override this to provide their default path.
        """
        return get_default_db_path()

    async def ensure_initialized(self) -> None:
        """Ensure the store is initialized.

        Must be called before first use if not using get_instance().
        """
        if self._initialized:
            return

        if aiosqlite is None:
            raise ImportError("aiosqlite is required for async SQLite support. " "Install with: pip install aiosqlite")

        await self._init_schema()
        self._initialized = True

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get a database connection.

        Returns a persistent connection for connection reuse.
        """
        if self._connection is None:
            self._connection = await aiosqlite.connect(
                str(self._db_path),
                timeout=30.0,
            )
            self._connection.row_factory = aiosqlite.Row
            # Enable pragmas
            await self._connection.execute("PRAGMA foreign_keys = ON")
            await self._connection.execute("PRAGMA journal_mode = WAL")
            await self._connection.execute("PRAGMA synchronous = NORMAL")
            await self._connection.execute("PRAGMA busy_timeout = 30000")

        return self._connection

    async def _init_schema(self) -> None:
        """Initialize the database schema.

        Subclasses must implement this to create their tables.
        """
        raise NotImplementedError("Subclasses must implement _init_schema")

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Async context manager for transactional database operations.

        All operations within the context are committed together.
        On exception, the transaction is rolled back.

        Usage:
            async with store.transaction() as conn:
                await conn.execute("INSERT INTO ...")
            # Committed automatically on successful exit
        """
        async with self._write_lock:
            conn = await self._get_connection()
            try:
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def execute(
        self,
        sql: str,
        parameters: tuple = (),
    ) -> aiosqlite.Cursor:
        """Execute a single SQL statement.

        For write operations, use transaction() instead.
        """
        conn = await self._get_connection()
        return await conn.execute(sql, parameters)

    async def execute_many(
        self,
        sql: str,
        parameters: List[tuple],
    ) -> None:
        """Execute a SQL statement with multiple parameter sets."""
        async with self.transaction() as conn:
            await conn.executemany(sql, parameters)

    async def fetch_one(
        self,
        sql: str,
        parameters: tuple = (),
    ) -> Optional[aiosqlite.Row]:
        """Execute a query and fetch one result."""
        cursor = await self.execute(sql, parameters)
        return await cursor.fetchone()

    async def fetch_all(
        self,
        sql: str,
        parameters: tuple = (),
    ) -> List[aiosqlite.Row]:
        """Execute a query and fetch all results."""
        cursor = await self.execute(sql, parameters)
        return await cursor.fetchall()

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    @classmethod
    async def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        # Ensure lock is valid for current event loop
        current_loop = asyncio.get_running_loop()
        if cls._init_lock is None or cls._init_lock_loop is not current_loop:
            cls._init_lock = asyncio.Lock()
            cls._init_lock_loop = current_loop

        async with cls._init_lock:
            if cls in cls._instances:
                instance = cls._instances[cls]
                await instance.close()
                del cls._instances[cls]

    @classmethod
    async def close_all_instances(cls) -> None:
        """Close all AsyncSQLiteStore instances.

        Should be called during application shutdown to release
        all database connections and allow clean exit.
        """
        # Ensure lock is valid for current event loop
        current_loop = asyncio.get_running_loop()
        if cls._init_lock is None or cls._init_lock_loop is not current_loop:
            cls._init_lock = asyncio.Lock()
            cls._init_lock_loop = current_loop

        async with cls._init_lock:
            for store_cls, instance in list(cls._instances.items()):
                try:
                    await instance.close()
                except Exception:
                    pass  # Best effort cleanup
            cls._instances.clear()

    def get_database_path(self) -> Path:
        """Get the path to the database file."""
        return self._db_path


# Helper for gradual migration - wraps sync store methods
class AsyncStoreMixin:
    """Mixin to add async wrappers to sync stores during migration.

    Add this mixin to existing sync stores to provide async interfaces
    while keeping sync internals. Useful for gradual migration.

    Usage:
        class MyStore(BaseSQLiteStore, AsyncStoreMixin):
            async def my_async_method(self):
                return await self.run_async(self._sync_method, arg1, arg2)
    """

    async def run_async(self, func, *args, **kwargs):
        """Run a sync function in the thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
