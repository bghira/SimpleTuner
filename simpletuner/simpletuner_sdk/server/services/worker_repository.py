"""Worker repository for CRUD operations on GPU workers.

This module provides the WorkerRepository class for managing worker persistence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.worker import Worker, WorkerStatus, WorkerType
from .cloud.storage.base import BaseSQLiteStore, get_default_db_path

logger = logging.getLogger(__name__)

# Schema version for this repository
SCHEMA_VERSION = 1


class WorkerRepository(BaseSQLiteStore):
    """Repository for worker persistence.

    Handles CRUD operations for GPU workers.
    """

    def _get_default_db_path(self) -> Path:
        return get_default_db_path("workers.db")

    def _init_schema(self) -> None:
        """Initialize the workers table schema."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Create workers table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    worker_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    user_id INTEGER NOT NULL,
                    gpu_info TEXT,
                    provider TEXT,
                    labels TEXT,
                    current_job_id TEXT,
                    last_heartbeat TEXT,
                    created_at TEXT NOT NULL
                )
            """
            )

            # Create indexes for common queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workers_user_id ON workers(user_id)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workers_token_hash ON workers(token_hash)
            """
            )

            # Create schema_version table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """
            )

            # Check and initialize schema version
            cursor.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()
            if row is None:
                cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            else:
                current_version = row["version"]
                if current_version < SCHEMA_VERSION:
                    self._run_migrations(cursor, current_version, SCHEMA_VERSION)
                    cursor.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

            conn.commit()
        except Exception as exc:
            logger.error("Failed to initialize workers schema: %s", exc)
            raise
        finally:
            conn.close()

    def _run_migrations(self, cursor, from_version: int, to_version: int) -> None:
        """Run schema migrations."""
        logger.info("Running worker schema migrations from v%d to v%d", from_version, to_version)
        # No migrations yet (initial version is v1)

    async def create_worker(self, worker: Worker) -> Worker:
        """Create a new worker.

        Args:
            worker: The worker to create.

        Returns:
            The created worker.

        Raises:
            ValueError: If a worker with the same ID or token hash already exists.
        """
        loop = asyncio.get_running_loop()

        def _insert():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO workers (
                        worker_id, name, worker_type, status, token_hash, user_id,
                        gpu_info, provider, labels, current_job_id, last_heartbeat, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        worker.worker_id,
                        worker.name,
                        worker.worker_type.value,
                        worker.status.value,
                        worker.token_hash,
                        worker.user_id,
                        json.dumps(worker.gpu_info) if worker.gpu_info else None,
                        worker.provider,
                        json.dumps(worker.labels) if worker.labels else None,
                        worker.current_job_id,
                        worker.last_heartbeat.isoformat() if worker.last_heartbeat else None,
                        worker.created_at.isoformat(),
                    ),
                )
                conn.commit()
                return worker
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _insert)

    async def get_worker(self, worker_id: str) -> Optional[Worker]:
        """Get a worker by ID.

        Args:
            worker_id: The worker ID to retrieve.

        Returns:
            The worker if found, None otherwise.
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM workers WHERE worker_id = ?", (worker_id,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_worker(row)
                return None
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_worker_by_token_hash(self, token_hash: str) -> Optional[Worker]:
        """Get a worker by its token hash (for authentication).

        Args:
            token_hash: The token hash to search for.

        Returns:
            The worker if found, None otherwise.
        """
        if not token_hash:
            return None

        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM workers WHERE token_hash = ?", (token_hash,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_worker(row)
                return None
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def list_workers(
        self,
        user_id: Optional[int] = None,
        status: Optional[WorkerStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Worker]:
        """List workers with optional filtering and pagination.

        Args:
            user_id: Filter by user ID.
            status: Filter by worker status.
            limit: Maximum number of workers to return.
            offset: Number of workers to skip for pagination.

        Returns:
            List of Worker objects.
        """
        loop = asyncio.get_running_loop()

        def _list():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                query = "SELECT * FROM workers WHERE 1=1"
                params: List[Any] = []

                if user_id is not None:
                    query += " AND user_id = ?"
                    params.append(user_id)
                if status:
                    query += " AND status = ?"
                    params.append(status.value)

                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.append(limit)
                params.append(offset)

                cursor.execute(query, params)
                return [self._row_to_worker(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _list)

    async def update_worker(self, worker_id: str, updates: Dict[str, Any]) -> Optional[Worker]:
        """Update an existing worker.

        Args:
            worker_id: The worker ID to update.
            updates: Dictionary of field updates.

        Returns:
            The updated worker if found, None otherwise.
        """
        loop = asyncio.get_running_loop()

        def _update():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                set_clauses = []
                values = []
                for key, value in updates.items():
                    # Handle special types
                    if key in ("gpu_info", "labels") and isinstance(value, dict):
                        set_clauses.append(f"{key} = ?")
                        values.append(json.dumps(value))
                    elif key == "last_heartbeat" and isinstance(value, datetime):
                        set_clauses.append(f"{key} = ?")
                        values.append(value.isoformat())
                    elif key == "status" and isinstance(value, WorkerStatus):
                        set_clauses.append(f"{key} = ?")
                        values.append(value.value)
                    elif key == "worker_type" and isinstance(value, WorkerType):
                        set_clauses.append(f"{key} = ?")
                        values.append(value.value)
                    else:
                        set_clauses.append(f"{key} = ?")
                        values.append(value)

                if not set_clauses:
                    return None

                values.append(worker_id)
                query = f"UPDATE workers SET {', '.join(set_clauses)} WHERE worker_id = ?"
                cursor.execute(query, values)
                conn.commit()

                if cursor.rowcount > 0:
                    # Fetch and return updated worker
                    cursor.execute("SELECT * FROM workers WHERE worker_id = ?", (worker_id,))
                    row = cursor.fetchone()
                    if row:
                        return self._row_to_worker(row)
                return None
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _update)

    async def delete_worker(self, worker_id: str) -> bool:
        """Delete a worker from the repository.

        Args:
            worker_id: The worker ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        loop = asyncio.get_running_loop()

        def _delete():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM workers WHERE worker_id = ?", (worker_id,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _delete)

    async def get_idle_worker_for_job(
        self, gpu_requirements: Optional[Dict[str, Any]] = None, labels: Optional[Dict[str, str]] = None
    ) -> Optional[Worker]:
        """Get an idle worker that matches job requirements.

        Args:
            gpu_requirements: GPU requirements (e.g., {"vram_gb": 40})
            labels: Required labels that worker must have

        Returns:
            An idle worker that matches requirements, or None if no match found.
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM workers WHERE status = ?", (WorkerStatus.IDLE.value,))
                rows = cursor.fetchall()

                for row in rows:
                    worker = self._row_to_worker(row)

                    # Check GPU VRAM requirement
                    if gpu_requirements and "vram_gb" in gpu_requirements:
                        required_vram = gpu_requirements["vram_gb"]
                        worker_vram = worker.gpu_info.get("vram_gb", 0)
                        if worker_vram < required_vram:
                            continue

                    # Check labels - worker must have all required labels
                    if labels:
                        worker_labels = worker.labels or {}
                        if not all(worker_labels.get(k) == v for k, v in labels.items()):
                            continue

                    # Found a match
                    return worker

                return None
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def update_heartbeat(self, worker_id: str) -> bool:
        """Update the last heartbeat timestamp for a worker.

        Args:
            worker_id: The worker ID to update.

        Returns:
            True if updated, False if not found.
        """
        now = datetime.now(timezone.utc)
        worker = await self.update_worker(worker_id, {"last_heartbeat": now})
        return worker is not None

    async def get_stale_workers(self, timeout_seconds: int = 300) -> List[Worker]:
        """Get workers that haven't sent a heartbeat recently.

        Args:
            timeout_seconds: Number of seconds without heartbeat to consider stale.

        Returns:
            List of stale workers.
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # Get workers with status != OFFLINE that have stale heartbeats
                cursor.execute(
                    """
                    SELECT * FROM workers
                    WHERE status != ?
                    AND (last_heartbeat IS NULL OR last_heartbeat < datetime('now', '-' || ? || ' seconds'))
                    """,
                    (WorkerStatus.OFFLINE.value, timeout_seconds),
                )
                return [self._row_to_worker(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    def _row_to_worker(self, row) -> Worker:
        """Convert a database row to a Worker."""
        gpu_info = {}
        if row["gpu_info"]:
            try:
                gpu_info = json.loads(row["gpu_info"])
            except json.JSONDecodeError:
                # Malformed GPU info JSON; use empty dict
                pass

        labels = {}
        if row["labels"]:
            try:
                labels = json.loads(row["labels"])
            except json.JSONDecodeError:
                # Malformed labels JSON; use empty dict
                pass

        last_heartbeat = None
        if row["last_heartbeat"]:
            try:
                last_heartbeat = datetime.fromisoformat(row["last_heartbeat"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                # Invalid timestamp format; leave as None
                pass

        created_at = datetime.now(timezone.utc)
        if row["created_at"]:
            try:
                created_at = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                # Invalid timestamp format; use current time
                pass

        return Worker(
            worker_id=row["worker_id"],
            name=row["name"],
            worker_type=WorkerType(row["worker_type"]),
            status=WorkerStatus(row["status"]),
            token_hash=row["token_hash"],
            user_id=row["user_id"],
            gpu_info=gpu_info,
            provider=row["provider"],
            labels=labels,
            current_job_id=row["current_job_id"],
            last_heartbeat=last_heartbeat,
            created_at=created_at,
        )

    @staticmethod
    def hash_token(token: str) -> str:
        """Hash a worker token using SHA256.

        Args:
            token: The plaintext token to hash.

        Returns:
            The SHA256 hex digest of the token.
        """
        return hashlib.sha256(token.encode()).hexdigest()


# Singleton accessor
_worker_repository: Optional[WorkerRepository] = None


def get_worker_repository() -> WorkerRepository:
    """Get the singleton WorkerRepository instance."""
    global _worker_repository
    if _worker_repository is None:
        _worker_repository = WorkerRepository()
    return _worker_repository
