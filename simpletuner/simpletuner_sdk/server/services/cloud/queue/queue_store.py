"""SQLite-based queue storage for cloud training jobs."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .models import QueueEntry, QueuePriority, QueueStatus

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3


class QueueStore:
    """SQLite-based storage for the job queue.

    Provides FIFO queuing with priority and fair scheduling support.
    Thread-safe with WAL mode for concurrent access.
    """

    _instance: Optional["QueueStore"] = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[Path] = None) -> "QueueStore":
        """Singleton pattern - one queue store per process."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the queue store."""
        if getattr(self, "_initialized", False):
            return

        if db_path is None:
            from ..container import get_job_store

            job_store = get_job_store()
            db_path = job_store._db_path.parent / "queue.db"

        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_schema()
        self._initialized = True

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "connection"):
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.connection = conn
        return self._local.connection

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Check current schema version
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
            """
        )

        row = cursor.execute("SELECT version FROM schema_version").fetchone()
        current_version = row["version"] if row else 0

        if current_version < SCHEMA_VERSION:
            self._run_migrations(cursor, current_version, SCHEMA_VERSION)
            cursor.execute("DELETE FROM schema_version")
            cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            conn.commit()

    async def _run_query(self, func, *args, **kwargs):
        """Run a blocking database query in an executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def _run_migrations(self, cursor: sqlite3.Cursor, from_version: int, to_version: int) -> None:
        """Run schema migrations."""
        if from_version < 1 <= to_version:
            # Initial schema
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    user_id INTEGER,
                    provider TEXT NOT NULL DEFAULT 'replicate',
                    config_name TEXT,
                    priority INTEGER NOT NULL DEFAULT 10,
                    status TEXT NOT NULL DEFAULT 'pending',
                    position INTEGER NOT NULL DEFAULT 0,
                    queued_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    estimated_cost REAL DEFAULT 0.0,
                    requires_approval INTEGER DEFAULT 0,
                    approval_id INTEGER,
                    attempt INTEGER DEFAULT 1,
                    max_attempts INTEGER DEFAULT 3,
                    error_message TEXT,
                    metadata TEXT DEFAULT '{}'
                )
                """
            )

            # Indexes for efficient querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON queue(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queue_user_id ON queue(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queue_priority ON queue(priority DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queue_position ON queue(position)")
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_queue_scheduling
                ON queue(status, priority DESC, queued_at ASC)
                """
            )

            logger.info("Created queue schema v1")

        if from_version < 2 <= to_version:
            # Add team_id and priority_override for fair-share scheduling
            try:
                cursor.execute("ALTER TABLE queue ADD COLUMN team_id TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                cursor.execute("ALTER TABLE queue ADD COLUMN priority_override INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Index for team-based queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queue_team_id ON queue(team_id)")

            logger.info("Migrated queue schema to v2 (added team_id, priority_override)")

        if from_version < 3 <= to_version:
            # Add GPU allocation tracking for local jobs
            try:
                cursor.execute("ALTER TABLE queue ADD COLUMN allocated_gpus TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                cursor.execute("ALTER TABLE queue ADD COLUMN job_type TEXT DEFAULT 'cloud'")
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                cursor.execute("ALTER TABLE queue ADD COLUMN num_processes INTEGER DEFAULT 1")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Index for local job queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queue_job_type ON queue(job_type)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_queue_local_running "
                "ON queue(job_type, status) WHERE job_type = 'local' AND status = 'running'"
            )

            logger.info("Migrated queue schema to v3 (added allocated_gpus, job_type, num_processes)")

    async def add_to_queue(
        self,
        job_id: str,
        user_id: Optional[int] = None,
        team_id: Optional[str] = None,
        provider: str = "replicate",
        config_name: Optional[str] = None,
        priority: QueuePriority = QueuePriority.NORMAL,
        priority_override: Optional[int] = None,
        estimated_cost: float = 0.0,
        requires_approval: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        allocated_gpus: Optional[List[int]] = None,
        job_type: str = "cloud",
        num_processes: int = 1,
    ) -> QueueEntry:
        """Add a job to the queue."""

        def _add():
            conn = self._get_connection()
            cursor = conn.cursor()

            now = datetime.now(timezone.utc).isoformat()
            metadata_json = json.dumps(metadata or {})
            allocated_gpus_json = json.dumps(allocated_gpus) if allocated_gpus else None

            # Calculate position
            cursor.execute(
                """
                SELECT COALESCE(MAX(position), 0) + 1 as next_pos
                FROM queue
                WHERE status IN ('pending', 'ready', 'blocked')
                """
            )
            next_position = cursor.fetchone()["next_pos"]

            status = QueueStatus.BLOCKED if requires_approval else QueueStatus.PENDING

            cursor.execute(
                """
                INSERT INTO queue (
                    job_id, user_id, team_id, provider, config_name, priority,
                    priority_override, status, position, queued_at, estimated_cost,
                    requires_approval, metadata, allocated_gpus, job_type, num_processes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    user_id,
                    team_id,
                    provider,
                    config_name,
                    priority.value,
                    priority_override,
                    status.value,
                    next_position,
                    now,
                    estimated_cost,
                    1 if requires_approval else 0,
                    metadata_json,
                    allocated_gpus_json,
                    job_type,
                    num_processes,
                ),
            )
            conn.commit()
            return cursor.lastrowid, next_position, status, now

        entry_id, next_position, status, now = await self._run_query(_add)

        return QueueEntry(
            id=entry_id,
            job_id=job_id,
            user_id=user_id,
            team_id=team_id,
            provider=provider,
            config_name=config_name,
            priority=priority,
            priority_override=priority_override,
            status=status,
            position=next_position,
            queued_at=now,
            estimated_cost=estimated_cost,
            requires_approval=requires_approval,
            metadata=metadata or {},
            allocated_gpus=allocated_gpus,
            job_type=job_type,
            num_processes=num_processes,
        )

    async def get_entry(self, queue_id: int) -> Optional[QueueEntry]:
        """Get a queue entry by ID."""

        def _get():
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM queue WHERE id = ?", (queue_id,))
            return cursor.fetchone()

        row = await self._run_query(_get)
        return self._row_to_entry(row) if row else None

    async def get_entry_by_job_id(self, job_id: str) -> Optional[QueueEntry]:
        """Get a queue entry by job ID."""

        def _get():
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM queue WHERE job_id = ?", (job_id,))
            return cursor.fetchone()

        row = await self._run_query(_get)
        return self._row_to_entry(row) if row else None

    async def get_positions_batch(self, job_ids: List[str]) -> Dict[str, int]:
        """Get queue positions for multiple jobs."""
        if not job_ids:
            return {}

        def _get_batch():
            conn = self._get_connection()
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(job_ids))
            cursor.execute(
                f"SELECT job_id, position FROM queue WHERE job_id IN ({placeholders})",
                job_ids,
            )
            return cursor.fetchall()

        rows = await self._run_query(_get_batch)
        return {row["job_id"]: row["position"] for row in rows}

    async def get_positions_with_eta_batch(self, job_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get queue positions with estimated wait time."""
        if not job_ids:
            return {}

        def _get_batch():
            conn = self._get_connection()
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(job_ids))
            cursor.execute(
                f"SELECT job_id, position, status FROM queue WHERE job_id IN ({placeholders})",
                job_ids,
            )
            return cursor.fetchall()

        rows = await self._run_query(_get_batch)
        positions = {row["job_id"]: {"position": row["position"], "status": row["status"]} for row in rows}

        if not positions:
            return {}

        stats = await self.get_queue_stats()
        avg_wait = stats.get("avg_wait_seconds")
        running = stats.get("running", 0)

        result = {}
        for job_id, data in positions.items():
            position = data["position"]
            status = data["status"]
            estimated_wait_seconds = None
            if status in ("pending", "ready", "blocked") and position > 0:
                if avg_wait is not None and avg_wait > 0:
                    effective_concurrency = max(running, 1)
                    estimated_wait_seconds = (position * avg_wait) / effective_concurrency

            result[job_id] = {
                "position": position,
                "estimated_wait_seconds": estimated_wait_seconds,
            }

        return result

    async def update_entry(self, queue_id: int, updates: Dict[str, Any]) -> bool:
        """Update a queue entry."""
        if not updates:
            return True

        def _update():
            conn = self._get_connection()
            cursor = conn.cursor()

            # Create a copy to avoid mutating original
            local_updates = dict(updates)
            if "priority" in local_updates and isinstance(local_updates["priority"], QueuePriority):
                local_updates["priority"] = local_updates["priority"].value
            if "status" in local_updates and isinstance(local_updates["status"], QueueStatus):
                local_updates["status"] = local_updates["status"].value
            if "metadata" in local_updates and isinstance(local_updates["metadata"], dict):
                local_updates["metadata"] = json.dumps(local_updates["metadata"])

            set_clause = ", ".join(f"{k} = ?" for k in local_updates.keys())
            values = list(local_updates.values()) + [queue_id]

            cursor.execute(f"UPDATE queue SET {set_clause} WHERE id = ?", values)
            conn.commit()
            return cursor.rowcount > 0

        return await self._run_query(_update)

    async def remove_entry(self, queue_id: int) -> bool:
        """Remove an entry from the queue."""

        def _remove():
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM queue WHERE id = ?", (queue_id,))
            conn.commit()
            return cursor.rowcount > 0

        return await self._run_query(_remove)

    async def list_entries(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[QueueStatus] = None,
        user_id: Optional[int] = None,
        include_completed: bool = False,
    ) -> List[QueueEntry]:
        """List queue entries."""

        def _list():
            conn = self._get_connection()
            cursor = conn.cursor()
            query = "SELECT * FROM queue WHERE 1=1"
            params: List[Any] = []

            if status:
                query += " AND status = ?"
                params.append(status.value)
            elif not include_completed:
                query += " AND status NOT IN ('completed', 'failed', 'cancelled')"

            if user_id is not None:
                query += " AND user_id = ?"
                params.append(user_id)

            query += " ORDER BY priority DESC, queued_at ASC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            return cursor.fetchall()

        rows = await self._run_query(_list)
        return [self._row_to_entry(row) for row in rows]

    async def count_running(self) -> int:
        """Count total running jobs."""

        def _count():
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM queue WHERE status = 'running'")
            return cursor.fetchone()["cnt"]

        return await self._run_query(_count)

    async def count_running_by_user(self) -> Dict[int, int]:
        """Get running job count per user."""

        def _count():
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_id, COUNT(*) as running_count
                FROM queue
                WHERE status = 'running' AND user_id IS NOT NULL
                GROUP BY user_id
                """
            )
            return {row["user_id"]: row["running_count"] for row in cursor.fetchall()}

        return await self._run_query(_count)

    async def count_running_by_team(self) -> Dict[str, int]:
        """Get running job count per team."""

        def _count():
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT team_id, COUNT(*) as running_count
                FROM queue
                WHERE status = 'running' AND team_id IS NOT NULL
                GROUP BY team_id
                """
            )
            return {row["team_id"]: row["running_count"] for row in cursor.fetchall()}

        return await self._run_query(_count)

    async def list_pending_by_priority(self, limit: int = 50) -> List[QueueEntry]:
        """List pending entries by priority."""

        def _list():
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM queue
                WHERE status IN ('pending', 'ready')
                ORDER BY priority DESC, queued_at ASC
                LIMIT ?
                """,
                (limit,),
            )
            return cursor.fetchall()

        rows = await self._run_query(_list)
        return [self._row_to_entry(row) for row in rows]

    async def mark_running(self, queue_id: int) -> bool:
        """Mark an entry as running."""
        now = datetime.now(timezone.utc).isoformat()
        return await self.update_entry(queue_id, {"status": QueueStatus.RUNNING.value, "started_at": now})

    async def mark_completed(self, queue_id: int) -> bool:
        """Mark an entry as completed."""
        now = datetime.now(timezone.utc).isoformat()
        return await self.update_entry(queue_id, {"status": QueueStatus.COMPLETED.value, "completed_at": now})

    async def mark_failed(self, queue_id: int, error: str) -> bool:
        """Mark an entry as failed."""
        now = datetime.now(timezone.utc).isoformat()
        return await self.update_entry(
            queue_id,
            {"status": QueueStatus.FAILED.value, "completed_at": now, "error_message": error},
        )

    async def mark_cancelled(self, queue_id: int) -> bool:
        """Mark an entry as cancelled."""
        now = datetime.now(timezone.utc).isoformat()
        return await self.update_entry(queue_id, {"status": QueueStatus.CANCELLED.value, "completed_at": now})

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""

        def _get_stats():
            conn = self._get_connection()
            cursor = conn.cursor()
            stats = {}

            cursor.execute("SELECT status, COUNT(*) as cnt FROM queue GROUP BY status")
            stats["by_status"] = {row["status"]: row["cnt"] for row in cursor.fetchall()}

            cursor.execute(
                """
                SELECT user_id, COUNT(*) as cnt
                FROM queue
                WHERE status NOT IN ('completed', 'failed', 'cancelled')
                GROUP BY user_id
                ORDER BY cnt DESC
                LIMIT 10
                """
            )
            stats["by_user"] = {row["user_id"]: row["cnt"] for row in cursor.fetchall()}

            cursor.execute(
                """
                SELECT AVG(JULIANDAY(started_at) - JULIANDAY(queued_at)) * 86400 as avg_wait
                FROM queue
                WHERE status = 'completed' AND started_at IS NOT NULL
                """
            )
            row = cursor.fetchone()
            stats["avg_wait_seconds"] = row["avg_wait"] if row else None

            cursor.execute("SELECT COUNT(*) as depth FROM queue WHERE status IN ('pending', 'ready', 'blocked')")
            stats["queue_depth"] = cursor.fetchone()["depth"]
            stats["running"] = stats["by_status"].get("running", 0)
            return stats

        return await self._run_query(_get_stats)

    async def get_user_position(self, user_id: int) -> Optional[int]:
        """Get a user's position in the queue."""

        def _get():
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT MIN(position) as pos FROM queue
                WHERE user_id = ? AND status IN ('pending', 'ready')
                """,
                (user_id,),
            )
            row = cursor.fetchone()
            return row["pos"] if row and row["pos"] is not None else None

        return await self._run_query(_get)

    async def cleanup_old_entries(self, days: int = 30) -> int:
        """Remove completed/failed entries."""

        def _cleanup():
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM queue
                WHERE status IN ('completed', 'failed', 'cancelled')
                AND datetime(completed_at) < datetime('now', ?)
                """,
                (f"-{days} days",),
            )
            conn.commit()
            return cursor.rowcount

        deleted = await self._run_query(_cleanup)
        if deleted > 0:
            logger.info("Cleaned up %d old queue entries", deleted)
        return deleted

    def _row_to_entry(self, row: sqlite3.Row) -> QueueEntry:
        """Convert a database row to a QueueEntry."""
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        # Handle new columns that may not exist in older databases
        team_id = None
        priority_override = None
        allocated_gpus = None
        job_type = "cloud"
        num_processes = 1
        try:
            team_id = row["team_id"]
            priority_override = row["priority_override"]
        except (IndexError, KeyError):
            pass

        # Handle v3 columns
        try:
            allocated_gpus_str = row["allocated_gpus"]
            if allocated_gpus_str:
                allocated_gpus = json.loads(allocated_gpus_str)
            job_type = row["job_type"] or "cloud"
            num_processes = row["num_processes"] or 1
        except (IndexError, KeyError):
            pass

        return QueueEntry(
            id=row["id"],
            job_id=row["job_id"],
            user_id=row["user_id"],
            team_id=team_id,
            provider=row["provider"],
            config_name=row["config_name"],
            priority=QueuePriority(row["priority"]),
            priority_override=priority_override,
            status=QueueStatus(row["status"]),
            position=row["position"],
            queued_at=row["queued_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            estimated_cost=row["estimated_cost"] or 0.0,
            requires_approval=bool(row["requires_approval"]),
            approval_id=row["approval_id"],
            attempt=row["attempt"] or 1,
            max_attempts=row["max_attempts"] or 3,
            error_message=row["error_message"],
            metadata=metadata,
            allocated_gpus=allocated_gpus,
            job_type=job_type,
            num_processes=num_processes,
        )

    # --- GPU Allocation Methods ---

    async def get_allocated_gpus(self) -> Set[int]:
        """Get all GPU indices currently allocated to running local jobs."""

        def _get():
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT allocated_gpus FROM queue
                WHERE job_type = 'local' AND status = 'running' AND allocated_gpus IS NOT NULL
                """
            )
            allocated: Set[int] = set()
            for row in cursor.fetchall():
                gpus = json.loads(row["allocated_gpus"])
                if isinstance(gpus, list):
                    allocated.update(gpus)
            return allocated

        return await self._run_query(_get)

    async def get_running_local_jobs(self) -> List[QueueEntry]:
        """Get all running local jobs."""

        def _get():
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM queue
                WHERE job_type = 'local' AND status = 'running'
                ORDER BY started_at ASC
                """
            )
            return cursor.fetchall()

        rows = await self._run_query(_get)
        return [self._row_to_entry(row) for row in rows]

    async def get_pending_local_jobs(self) -> List[QueueEntry]:
        """Get all pending local jobs awaiting GPU allocation."""

        def _get():
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM queue
                WHERE job_type = 'local' AND status IN ('pending', 'ready')
                ORDER BY priority DESC, queued_at ASC
                """
            )
            return cursor.fetchall()

        rows = await self._run_query(_get)
        return [self._row_to_entry(row) for row in rows]

    async def update_allocated_gpus(self, job_id: str, gpus: Optional[List[int]]) -> bool:
        """Update the allocated GPUs for a job."""

        def _update():
            conn = self._get_connection()
            cursor = conn.cursor()
            gpus_json = json.dumps(gpus) if gpus else None
            cursor.execute(
                "UPDATE queue SET allocated_gpus = ? WHERE job_id = ?",
                (gpus_json, job_id),
            )
            conn.commit()
            return cursor.rowcount > 0

        return await self._run_query(_update)

    async def count_running_local_jobs(self) -> int:
        """Count the number of running local jobs."""

        def _count():
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM queue WHERE job_type = 'local' AND status = 'running'")
            return cursor.fetchone()["cnt"]

        return await self._run_query(_count)
