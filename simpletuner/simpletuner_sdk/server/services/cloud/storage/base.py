"""Base class for SQLite-backed storage.

Provides common connection management, transaction handling, and
cross-platform path resolution.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_default_config_dir() -> Path:
    """Get the default configuration directory in a cross-platform manner.

    Returns:
        Path to the configuration directory:
        - Windows: %APPDATA%/SimpleTuner
        - macOS: ~/Library/Application Support/SimpleTuner
        - Linux: ~/.config/simpletuner or ~/.simpletuner (legacy)
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "SimpleTuner"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "SimpleTuner"
    else:
        # Linux/Unix - check XDG first, fall back to legacy
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            return Path(xdg_config) / "simpletuner"
        # Check for legacy path
        legacy_path = Path.home() / ".simpletuner"
        if legacy_path.exists():
            return legacy_path
        # Use XDG default
        return Path.home() / ".config" / "simpletuner"


def get_default_db_path(db_name: str = "jobs.db") -> Path:
    """Get the default database path.

    Args:
        db_name: Name of the database file.

    Returns:
        Full path to the database file.
    """
    return get_default_config_dir() / "cloud" / db_name


class BaseSQLiteStore:
    """Base class for SQLite-backed storage with connection pooling.

    Provides:
    - Thread-safe singleton pattern
    - WAL mode for better concurrency
    - Transaction management (sync and async)
    - Busy timeout for multi-worker scenarios
    """

    _instances: dict[type, "BaseSQLiteStore"] = {}
    _init_lock = threading.Lock()

    def __new__(cls, db_path: Optional[Path] = None) -> "BaseSQLiteStore":
        """Singleton pattern - db_path only used on first instantiation."""
        with cls._init_lock:
            if cls not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                instance._pending_db_path = db_path
                cls._instances[cls] = instance
            elif db_path is not None:
                existing = getattr(cls._instances[cls], "_db_path", None)
                if existing and Path(db_path).resolve() != existing.resolve():
                    logger.warning(
                        "%s already initialized with db_path=%s, ignoring new path=%s",
                        cls.__name__,
                        existing,
                        db_path,
                    )
            return cls._instances[cls]

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the store.

        Args:
            db_path: Path to the SQLite database. Only used on first instantiation.
        """
        if getattr(self, "_initialized", False):
            return

        pending = getattr(self, "_pending_db_path", None)
        if pending is not None:
            db_path = pending
            self._pending_db_path = None

        if db_path is None:
            db_path = self._get_default_db_path()

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Async lock for coordinating database access (lazy initialized)
        self.__lock: Optional[asyncio.Lock] = None
        self.__lock_loop: Optional[asyncio.AbstractEventLoop] = None

        # Initialize database schema
        self._init_schema()
        self._initialized = True

    @property
    def _lock(self) -> asyncio.Lock:
        """Get the async lock, creating one for the current event loop if needed."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - create lock without binding
            if self.__lock is None:
                self.__lock = asyncio.Lock()
            return self.__lock

        # Recreate lock if bound to a different loop
        if self.__lock is None or self.__lock_loop is not current_loop:
            self.__lock = asyncio.Lock()
            self.__lock_loop = current_loop
        return self.__lock

    def _get_default_db_path(self) -> Path:
        """Get the default database path for this store.

        Subclasses should override this to provide their default path.
        """
        return get_default_db_path()

    def _init_schema(self) -> None:
        """Initialize the database schema.

        Subclasses must implement this to create their tables.
        """
        raise NotImplementedError("Subclasses must implement _init_schema")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection with proper settings.

        Multi-worker safe: Uses WAL mode and busy timeout for lock contention.
        """
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=30.0,
            isolation_level="IMMEDIATE",
        )
        conn.row_factory = sqlite3.Row
        # Enable foreign keys and WAL mode for better concurrency
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        # Busy timeout for multi-worker lock contention (30 seconds)
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for transactional database operations.

        All operations within the context use the same connection and are
        committed together. On exception, the transaction is rolled back.

        Usage (sync):
            with store.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO ...")
            # Committed automatically on successful exit
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @asynccontextmanager
    async def transaction_async(self) -> Generator[sqlite3.Connection, None, None]:
        """Async context manager for transactional database operations.

        All operations within the context use the same connection and are
        committed together. On exception, the transaction is rolled back.

        Usage:
            async with store.transaction_async() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO ...")
            # Committed automatically on successful exit
        """
        loop = asyncio.get_running_loop()

        def _begin():
            return self._get_connection()

        def _commit(conn):
            conn.commit()
            conn.close()

        def _rollback(conn):
            conn.rollback()
            conn.close()

        async with self._lock:
            conn = await loop.run_in_executor(None, _begin)
            try:
                yield conn
                await loop.run_in_executor(None, _commit, conn)
            except Exception:
                await loop.run_in_executor(None, _rollback, conn)
                raise

    async def execute_in_transaction(
        self,
        operations: List[Callable[[sqlite3.Connection], T]],
    ) -> List[T]:
        """Execute multiple operations in a single transaction.

        Args:
            operations: List of callables that take a connection and return a result

        Returns:
            List of results from each operation
        """
        loop = asyncio.get_running_loop()

        def _run():
            conn = self._get_connection()
            try:
                results = []
                for op in operations:
                    results.append(op(conn))
                conn.commit()
                return results
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _run)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._init_lock:
            if cls in cls._instances:
                del cls._instances[cls]

    def get_database_path(self) -> Path:
        """Get the path to the database file."""
        return self._db_path
