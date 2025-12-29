"""Base class for auth stores sharing a database connection.

All auth stores inherit from this base to share connection infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_default_db_path() -> Path:
    """Get the default database path for auth stores."""
    return Path.home() / ".simpletuner" / "config" / "cloud" / "jobs.db"


class BaseAuthStore:
    """Base class for auth stores with shared connection management.

    Provides:
    - Thread-safe connection handling
    - Async lock for write operations
    - WAL mode for better concurrency
    """

    _instances: dict[type, "BaseAuthStore"] = {}
    _init_lock = threading.Lock()

    def __new__(cls, db_path: Optional[Path] = None) -> "BaseAuthStore":
        """Singleton pattern per class."""
        with cls._init_lock:
            if cls not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                instance._pending_db_path = db_path
                cls._instances[cls] = instance
            return cls._instances[cls]

    def __init__(self, db_path: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return

        pending = getattr(self, "_pending_db_path", None)
        if pending is not None:
            db_path = pending
            self._pending_db_path = None

        if db_path is None:
            db_path = self._resolve_db_path()

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialized = True

    def _resolve_db_path(self) -> Path:
        """Resolve database path, preferring AsyncJobStore's path if available."""
        try:
            from ...async_job_store import AsyncJobStore

            if AsyncJobStore._instance is not None:
                return AsyncJobStore._instance._db_path
        except Exception:
            pass
        return get_default_db_path()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection with proper settings."""
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=30.0,
            isolation_level="IMMEDIATE",
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._init_lock:
            if cls in cls._instances:
                del cls._instances[cls]

    def get_database_path(self) -> Path:
        """Get the path to the database file."""
        return self._db_path
