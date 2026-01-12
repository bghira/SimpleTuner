"""Database backup and restore manager.

Provides backup, restore, and database maintenance operations
for cloud training data.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import aiosqlite
except ImportError:
    aiosqlite = None  # type: ignore

from .base import get_default_db_path

logger = logging.getLogger(__name__)


class BackupManager:
    """Manager for database backup and restore operations.

    Provides:
    - Point-in-time backups with WAL checkpoint
    - Restore from backup with validation
    - Backup listing and cleanup
    - Database health information
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the backup manager.

        Args:
            db_path: Path to the database file. Defaults to standard location.
        """
        self._db_path = Path(db_path) if db_path else get_default_db_path()
        self._backup_dir = self._db_path.parent
        self._connection: Optional[aiosqlite.Connection] = None

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get a database connection for checkpoint operations."""
        if aiosqlite is None:
            raise ImportError("aiosqlite required for backup operations")

        if self._connection is None:
            self._connection = await aiosqlite.connect(str(self._db_path), timeout=30.0)
            await self._connection.execute("PRAGMA busy_timeout = 30000")

        return self._connection

    async def backup(self, backup_path: Optional[Path] = None) -> Path:
        """Create a backup of the database.

        Performs a WAL checkpoint before backup to ensure all data is written.

        Args:
            backup_path: Custom backup path. If not provided, creates a
                        timestamped backup in the cloud directory.

        Returns:
            Path to the created backup file.
        """
        if backup_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = self._backup_dir / f"jobs_backup_{timestamp}.db"

        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Checkpoint WAL to ensure all data is in main db
        conn = await self._get_connection()
        await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        # Copy database file (blocking I/O, run in executor)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, shutil.copy2, self._db_path, backup_path)

        # Copy WAL file if it exists
        wal_path = self._db_path.with_suffix(".db-wal")
        if wal_path.exists():
            backup_wal = backup_path.with_suffix(".db-wal")
            await loop.run_in_executor(None, shutil.copy2, wal_path, backup_wal)

        logger.info("Database backed up to: %s", backup_path)
        return backup_path

    async def restore(self, backup_path: Path) -> bool:
        """Restore the database from a backup.

        WARNING: This overwrites the current database!

        Args:
            backup_path: Path to the backup file.

        Returns:
            True if restore was successful.

        Raises:
            FileNotFoundError: If backup file doesn't exist.
        """
        backup_path = Path(backup_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        # Close existing connection
        await self.close()

        loop = asyncio.get_running_loop()

        # Remove existing database files
        try:
            for suffix in ("", "-wal", "-shm"):
                path = self._db_path.with_suffix(f".db{suffix}")
                if path.exists():
                    await loop.run_in_executor(None, path.unlink)
        except OSError as exc:
            logger.error("Failed to remove existing database: %s", exc)
            return False

        # Copy backup to database location
        await loop.run_in_executor(None, shutil.copy2, backup_path, self._db_path)

        # Copy WAL if it exists in backup
        backup_wal = backup_path.with_suffix(".db-wal")
        if backup_wal.exists():
            dest_wal = self._db_path.with_suffix(".db-wal")
            await loop.run_in_executor(None, shutil.copy2, backup_wal, dest_wal)

        logger.info("Database restored from: %s", backup_path)
        return True

    def list_backups(self) -> List[Path]:
        """List available backup files.

        Returns:
            List of backup file paths, sorted newest first.
        """
        backups = list(self._backup_dir.glob("jobs_backup_*.db"))
        backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return backups

    async def cleanup_old_backups(self, keep_count: int = 5) -> int:
        """Remove old backups, keeping the most recent ones.

        Args:
            keep_count: Number of recent backups to keep.

        Returns:
            Number of backups deleted.
        """
        backups = self.list_backups()

        if len(backups) <= keep_count:
            return 0

        to_delete = backups[keep_count:]
        loop = asyncio.get_running_loop()
        deleted = 0

        for backup in to_delete:
            try:
                await loop.run_in_executor(None, backup.unlink)
                # Also delete associated WAL/SHM files
                for suffix in ("-wal", "-shm"):
                    assoc = backup.with_suffix(f".db{suffix}")
                    if assoc.exists():
                        await loop.run_in_executor(None, assoc.unlink)
                deleted += 1
            except OSError as exc:
                logger.warning("Failed to delete backup %s: %s", backup, exc)

        if deleted:
            logger.info("Cleaned up %d old backups", deleted)

        return deleted

    async def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database.

        Returns:
            Dict with database size, counts, and health info.
        """
        info = {
            "path": str(self._db_path),
            "size_bytes": 0,
            "size_mb": 0.0,
            "job_count": 0,
            "audit_log_count": 0,
            "exists": self._db_path.exists(),
        }

        if not info["exists"]:
            return info

        info["size_bytes"] = self._db_path.stat().st_size
        info["size_mb"] = round(info["size_bytes"] / (1024 * 1024), 2)

        # Include WAL size if present
        wal_path = self._db_path.with_suffix(".db-wal")
        if wal_path.exists():
            wal_size = wal_path.stat().st_size
            info["wal_size_bytes"] = wal_size
            info["total_size_mb"] = round((info["size_bytes"] + wal_size) / (1024 * 1024), 2)

        try:
            conn = await self._get_connection()

            cursor = await conn.execute("SELECT COUNT(*) FROM jobs")
            row = await cursor.fetchone()
            info["job_count"] = row[0] if row else 0

            # Check if audit_log table exists
            cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'")
            if await cursor.fetchone():
                cursor = await conn.execute("SELECT COUNT(*) FROM audit_log")
                row = await cursor.fetchone()
                info["audit_log_count"] = row[0] if row else 0

            # Get schema version if available
            cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
            if await cursor.fetchone():
                cursor = await conn.execute("SELECT version FROM schema_version LIMIT 1")
                row = await cursor.fetchone()
                if row:
                    info["schema_version"] = row[0]

        except Exception as exc:
            logger.warning("Failed to get database counts: %s", exc)
            info["error"] = str(exc)

        return info

    async def vacuum(self) -> bool:
        """Compact the database to reclaim space.

        Returns:
            True if successful.
        """
        try:
            conn = await self._get_connection()
            await conn.execute("VACUUM")
            logger.info("Database vacuumed successfully")
            return True
        except Exception as exc:
            logger.error("Failed to vacuum database: %s", exc)
            return False

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None


# Singleton access
_instance: Optional[BackupManager] = None


async def get_backup_manager(db_path: Optional[Path] = None) -> BackupManager:
    """Get the singleton BackupManager instance."""
    global _instance
    if _instance is None:
        _instance = BackupManager(db_path)
    return _instance
