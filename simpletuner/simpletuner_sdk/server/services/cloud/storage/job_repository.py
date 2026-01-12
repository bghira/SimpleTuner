"""Job repository for CRUD operations on training jobs.

This module provides the JobRepository class for managing job persistence,
separated from provider config and audit logging concerns.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import CloudJobStatus, JobType, UnifiedJob
from .base import BaseSQLiteStore, get_default_db_path

logger = logging.getLogger(__name__)

# Schema version for this repository
# v5: Merged queue fields from QueueStore
SCHEMA_VERSION = 5


class JobRepository(BaseSQLiteStore):
    """Repository for job persistence.

    Handles CRUD operations for training jobs, separated from
    provider configuration and audit logging.
    """

    def _get_default_db_path(self) -> Path:
        return get_default_db_path("jobs.db")

    def _init_schema(self) -> None:
        """Initialize the jobs table schema."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Create jobs table (unified with queue fields as of v5)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    provider TEXT,
                    status TEXT NOT NULL,
                    config_name TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    cost_usd REAL,
                    hardware_type TEXT,
                    error_message TEXT,
                    output_url TEXT,
                    upload_token TEXT,
                    user_id INTEGER,
                    metadata TEXT DEFAULT '{}',
                    -- Queue/scheduling fields (merged from QueueStore in v5)
                    priority INTEGER NOT NULL DEFAULT 10,
                    priority_override INTEGER,
                    queue_position INTEGER NOT NULL DEFAULT 0,
                    queued_at TEXT,
                    requires_approval INTEGER DEFAULT 0,
                    approval_id INTEGER,
                    attempt INTEGER DEFAULT 1,
                    max_attempts INTEGER DEFAULT 3,
                    team_id TEXT,
                    org_id INTEGER,
                    estimated_cost REAL DEFAULT 0.0,
                    allocated_gpus TEXT,
                    num_processes INTEGER DEFAULT 1
                )
            """
            )

            # Create indexes for common queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_upload_token ON jobs(upload_token)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id)
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

            # Check and run schema migrations
            cursor.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()
            if row is None:
                cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            else:
                current_version = row["version"]
                if current_version < SCHEMA_VERSION:
                    self._run_migrations(cursor, current_version, SCHEMA_VERSION)
                    cursor.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

            # Queue-related indexes (v5) - created after migrations ensure columns exist
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority DESC)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_queue_position ON jobs(queue_position)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_scheduling ON jobs(status, priority DESC, queued_at ASC)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_team_id ON jobs(team_id)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_org_id ON jobs(org_id)
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_local_running ON jobs(job_type, status)
                WHERE job_type = 'local' AND status = 'running'
            """
            )

            conn.commit()
        except Exception as exc:
            logger.error("Failed to initialize jobs schema: %s", exc)
            raise
        finally:
            conn.close()

    def _run_migrations(self, cursor, from_version: int, to_version: int) -> None:
        """Run schema migrations."""
        logger.info("Running job schema migrations from v%d to v%d", from_version, to_version)

        if from_version < 3 <= to_version:
            # v3: Add user_id column
            cursor.execute("PRAGMA table_info(jobs)")
            existing_columns = {row["name"] for row in cursor.fetchall()}

            if "user_id" not in existing_columns:
                cursor.execute("ALTER TABLE jobs ADD COLUMN user_id INTEGER")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id)")
                logger.info("Added user_id column to jobs table")

        if from_version < 5 <= to_version:
            # v5: Merge queue fields from QueueStore
            cursor.execute("PRAGMA table_info(jobs)")
            existing_columns = {row["name"] for row in cursor.fetchall()}

            queue_columns = [
                ("priority", "INTEGER NOT NULL DEFAULT 10"),
                ("priority_override", "INTEGER"),
                ("queue_position", "INTEGER NOT NULL DEFAULT 0"),
                ("queued_at", "TEXT"),
                ("requires_approval", "INTEGER DEFAULT 0"),
                ("approval_id", "INTEGER"),
                ("attempt", "INTEGER DEFAULT 1"),
                ("max_attempts", "INTEGER DEFAULT 3"),
                ("team_id", "TEXT"),
                ("org_id", "INTEGER"),
                ("estimated_cost", "REAL DEFAULT 0.0"),
                ("allocated_gpus", "TEXT"),
                ("num_processes", "INTEGER DEFAULT 1"),
            ]

            for col_name, col_def in queue_columns:
                if col_name not in existing_columns:
                    cursor.execute(f"ALTER TABLE jobs ADD COLUMN {col_name} {col_def}")
                    logger.info("Added %s column to jobs table", col_name)

            # Create new indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_queue_position ON jobs(queue_position)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_scheduling ON jobs(status, priority DESC, queued_at ASC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_team_id ON jobs(team_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_org_id ON jobs(org_id)")
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_local_running ON jobs(job_type, status)
                WHERE job_type = 'local' AND status = 'running'
            """
            )
            logger.info("Queue fields merged into jobs table (v5 migration complete)")

    async def add(self, job: UnifiedJob) -> bool:
        """Add a new job to the repository.

        Args:
            job: The job to add.

        Returns:
            True if inserted, False if already exists.
        """
        loop = asyncio.get_running_loop()

        def _insert():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO jobs (
                        job_id, job_type, provider, status, config_name,
                        created_at, started_at, completed_at, cost_usd,
                        hardware_type, error_message, output_url, upload_token, user_id, metadata,
                        priority, priority_override, queue_position, queued_at,
                        requires_approval, approval_id, attempt, max_attempts,
                        team_id, org_id, estimated_cost, allocated_gpus, num_processes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        job.job_id,
                        job.job_type.value,
                        job.provider,
                        job.status,
                        job.config_name,
                        job.created_at,
                        job.started_at,
                        job.completed_at,
                        job.cost_usd,
                        job.hardware_type,
                        job.error_message,
                        job.output_url,
                        job.upload_token,
                        job.user_id,
                        json.dumps(job.metadata),
                        job.priority,
                        job.priority_override,
                        job.queue_position,
                        job.queued_at,
                        1 if job.requires_approval else 0,
                        job.approval_id,
                        job.attempt,
                        job.max_attempts,
                        job.team_id,
                        job.org_id,
                        job.estimated_cost,
                        json.dumps(job.allocated_gpus) if job.allocated_gpus else None,
                        job.num_processes,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            inserted = await loop.run_in_executor(None, _insert)
            if not inserted:
                logger.debug("Job %s already exists, skipping add", job.job_id)
            return inserted

    async def update(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing job.

        Args:
            job_id: The job ID to update.
            updates: Dictionary of field updates.

        Returns:
            True if updated, False if not found.
        """
        loop = asyncio.get_running_loop()

        def _update():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                set_clauses = []
                values = []
                for key, value in updates.items():
                    if key == "metadata":
                        set_clauses.append("metadata = json_patch(metadata, ?)")
                        values.append(json.dumps(value))
                    else:
                        set_clauses.append(f"{key} = ?")
                        values.append(value)

                if not set_clauses:
                    return False

                values.append(job_id)
                query = f"UPDATE jobs SET {', '.join(set_clauses)} WHERE job_id = ?"
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _update)

    async def get(self, job_id: str) -> Optional[UnifiedJob]:
        """Get a job by ID.

        Args:
            job_id: The job ID to retrieve.

        Returns:
            The job if found, None otherwise.
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_job(row)
                return None
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_by_upload_token(self, upload_token: str) -> Optional[UnifiedJob]:
        """Get a job by its upload token (for S3 auth validation).

        Args:
            upload_token: The upload token to search for.

        Returns:
            The job if found and active, None otherwise.
        """
        if not upload_token:
            return None

        active_statuses = (
            CloudJobStatus.PENDING.value,
            CloudJobStatus.UPLOADING.value,
            CloudJobStatus.QUEUED.value,
            CloudJobStatus.RUNNING.value,
        )

        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                placeholders = ",".join("?" * len(active_statuses))
                cursor.execute(
                    f"SELECT * FROM jobs WHERE upload_token = ? AND status IN ({placeholders})",
                    (upload_token, *active_statuses),
                )
                row = cursor.fetchone()
                if row:
                    return self._row_to_job(row)
                return None
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        job_type: Optional[JobType] = None,
        status: Optional[str] = None,
        user_id: Optional[int] = None,
        provider: Optional[str] = None,
    ) -> List[UnifiedJob]:
        """List jobs with optional filtering and pagination.

        Args:
            limit: Maximum number of jobs to return.
            offset: Number of jobs to skip for pagination.
            job_type: Filter by job type (local/cloud).
            status: Filter by job status.
            user_id: Filter by user ID (for job isolation).
            provider: Filter by cloud provider (e.g., 'runpod', 'lambda').

        Returns:
            List of UnifiedJob objects.
        """
        loop = asyncio.get_running_loop()

        def _list():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                query = "SELECT * FROM jobs WHERE 1=1"
                params: List[Any] = []

                if job_type:
                    query += " AND job_type = ?"
                    params.append(job_type.value)
                if status:
                    query += " AND status = ?"
                    params.append(status)
                if user_id is not None:
                    query += " AND user_id = ?"
                    params.append(user_id)
                if provider:
                    query += " AND provider = ?"
                    params.append(provider)

                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.append(limit)
                params.append(offset)

                cursor.execute(query, params)
                return [self._row_to_job(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _list)

    async def delete(self, job_id: str) -> bool:
        """Delete a job from the repository.

        Args:
            job_id: The job ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        loop = asyncio.get_running_loop()

        def _delete():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _delete)

    async def cleanup_old(self, retention_days: int = 90) -> int:
        """Remove jobs older than retention_days.

        Args:
            retention_days: Number of days to retain jobs.

        Returns:
            Number of jobs removed.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_iso = cutoff.isoformat()

        loop = asyncio.get_running_loop()

        def _cleanup():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff_iso,))
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _cleanup)

    async def get_history_for_chart(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get job history aggregated by day for chart display.

        Args:
            days: Number of days of history.

        Returns:
            List of daily aggregates.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_iso = cutoff.isoformat()

        loop = asyncio.get_running_loop()

        def _aggregate():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        date(created_at) as date_key,
                        COUNT(*) as total_jobs,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled,
                        SUM(CASE WHEN job_type = 'local' THEN 1 ELSE 0 END) as local_jobs,
                        SUM(CASE WHEN job_type = 'cloud' THEN 1 ELSE 0 END) as cloud_jobs,
                        COALESCE(SUM(cost_usd), 0) as total_cost_usd
                    FROM jobs
                    WHERE created_at >= ?
                    GROUP BY date(created_at)
                    ORDER BY date_key
                """,
                    (cutoff_iso,),
                )

                return [
                    {
                        "date": row["date_key"],
                        "total_jobs": row["total_jobs"],
                        "completed": row["completed"],
                        "failed": row["failed"],
                        "cancelled": row["cancelled"],
                        "local_jobs": row["local_jobs"],
                        "cloud_jobs": row["cloud_jobs"],
                        "total_cost_usd": row["total_cost_usd"],
                    }
                    for row in cursor.fetchall()
                ]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _aggregate)

    async def get_metrics_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary metrics for the dashboard.

        Args:
            days: Number of days to summarize.

        Returns:
            Dictionary of metrics.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_iso = cutoff.isoformat()

        loop = asyncio.get_running_loop()

        def _summarize():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total_jobs,
                        COALESCE(SUM(cost_usd), 0) as total_cost
                    FROM jobs
                    WHERE created_at >= ?
                """,
                    (cutoff_iso,),
                )
                totals = cursor.fetchone()

                cursor.execute(
                    """
                    SELECT status, COUNT(*) as count
                    FROM jobs
                    WHERE created_at >= ?
                    GROUP BY status
                """,
                    (cutoff_iso,),
                )
                status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

                cursor.execute(
                    """
                    SELECT AVG(
                        (julianday(completed_at) - julianday(started_at)) * 86400
                    ) as avg_duration
                    FROM jobs
                    WHERE created_at >= ?
                      AND status = 'completed'
                      AND started_at IS NOT NULL
                      AND completed_at IS NOT NULL
                """,
                    (cutoff_iso,),
                )
                avg_row = cursor.fetchone()
                avg_duration = avg_row["avg_duration"] if avg_row else None

                return {
                    "total_cost_30d": totals["total_cost"],
                    "job_count_30d": totals["total_jobs"],
                    "avg_job_duration_seconds": avg_duration,
                    "jobs_by_status": status_counts,
                }
            finally:
                conn.close()

        return await loop.run_in_executor(None, _summarize)

    def _row_to_job(self, row) -> UnifiedJob:
        """Convert a database row to a UnifiedJob."""
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass

        # Handle columns that may not exist in older databases
        def safe_get(key, default=None):
            try:
                val = row[key]
                return val if val is not None else default
            except (IndexError, KeyError):
                return default

        # Parse allocated_gpus from JSON
        allocated_gpus = None
        allocated_gpus_str = safe_get("allocated_gpus")
        if allocated_gpus_str:
            try:
                allocated_gpus = json.loads(allocated_gpus_str)
            except json.JSONDecodeError:
                pass

        return UnifiedJob(
            job_id=row["job_id"],
            job_type=JobType(row["job_type"]),
            provider=row["provider"],
            status=row["status"],
            config_name=row["config_name"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            cost_usd=row["cost_usd"],
            hardware_type=row["hardware_type"],
            error_message=row["error_message"],
            output_url=row["output_url"],
            upload_token=row["upload_token"],
            user_id=safe_get("user_id"),
            metadata=metadata,
            # Queue/scheduling fields (v5)
            priority=safe_get("priority", 10),
            priority_override=safe_get("priority_override"),
            queue_position=safe_get("queue_position", 0),
            queued_at=safe_get("queued_at"),
            requires_approval=bool(safe_get("requires_approval", 0)),
            approval_id=safe_get("approval_id"),
            attempt=safe_get("attempt", 1),
            max_attempts=safe_get("max_attempts", 3),
            team_id=safe_get("team_id"),
            org_id=safe_get("org_id"),
            estimated_cost=safe_get("estimated_cost", 0.0),
            allocated_gpus=allocated_gpus,
            num_processes=safe_get("num_processes", 1),
        )

    # --- Queue/Scheduling Methods (merged from QueueStore) ---

    async def mark_running(self, job_id: str) -> bool:
        """Mark a job as running and set started_at timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        return await self.update(job_id, {"status": CloudJobStatus.RUNNING.value, "started_at": now})

    async def mark_completed(self, job_id: str) -> bool:
        """Mark a job as completed and set completed_at timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        return await self.update(job_id, {"status": CloudJobStatus.COMPLETED.value, "completed_at": now})

    async def mark_failed(self, job_id: str, error: str) -> bool:
        """Mark a job as failed with error message."""
        now = datetime.now(timezone.utc).isoformat()
        return await self.update(
            job_id,
            {"status": CloudJobStatus.FAILED.value, "completed_at": now, "error_message": error},
        )

    async def mark_cancelled(self, job_id: str) -> bool:
        """Mark a job as cancelled."""
        now = datetime.now(timezone.utc).isoformat()
        return await self.update(job_id, {"status": CloudJobStatus.CANCELLED.value, "completed_at": now})

    async def get_pending_local_jobs(self) -> List[UnifiedJob]:
        """Get all pending local jobs awaiting GPU allocation."""
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM jobs
                    WHERE job_type = 'local' AND status IN ('pending', 'queued')
                    ORDER BY priority DESC, queued_at ASC
                    """
                )
                return [self._row_to_job(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_running_local_jobs(self) -> List[UnifiedJob]:
        """Get all running local jobs."""
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM jobs
                    WHERE job_type = 'local' AND status = 'running'
                    ORDER BY started_at ASC
                    """
                )
                return [self._row_to_job(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def get_allocated_gpus(self) -> set:
        """Get all GPU indices currently allocated to running local jobs."""
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT allocated_gpus FROM jobs
                    WHERE job_type = 'local' AND status = 'running' AND allocated_gpus IS NOT NULL
                    """
                )
                allocated: set = set()
                for row in cursor.fetchall():
                    gpus = json.loads(row["allocated_gpus"])
                    if isinstance(gpus, list):
                        allocated.update(gpus)
                return allocated
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    async def update_allocated_gpus(self, job_id: str, gpus: Optional[List[int]]) -> bool:
        """Update the allocated GPUs for a job."""
        gpus_json = json.dumps(gpus) if gpus else None
        return await self.update(job_id, {"allocated_gpus": gpus_json})

    async def count_running_local_jobs(self) -> int:
        """Count the number of running local jobs."""
        loop = asyncio.get_running_loop()

        def _count():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as cnt FROM jobs WHERE job_type = 'local' AND status = 'running'")
                return cursor.fetchone()["cnt"]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _count)

    async def count_running(self) -> int:
        """Count total running jobs (all types)."""
        loop = asyncio.get_running_loop()

        def _count():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as cnt FROM jobs WHERE status = 'running'")
                return cursor.fetchone()["cnt"]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _count)

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics for monitoring."""
        loop = asyncio.get_running_loop()

        def _get_stats():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                stats = {}

                # Count by status
                cursor.execute("SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status")
                stats["by_status"] = {row["status"]: row["cnt"] for row in cursor.fetchall()}

                # Queue depth (pending/queued jobs)
                cursor.execute("SELECT COUNT(*) as depth FROM jobs WHERE status IN ('pending', 'queued')")
                stats["queue_depth"] = cursor.fetchone()["depth"]

                # Running count
                stats["running"] = stats["by_status"].get("running", 0)

                # Average wait time (for completed jobs)
                cursor.execute(
                    """
                    SELECT AVG(JULIANDAY(started_at) - JULIANDAY(queued_at)) * 86400 as avg_wait
                    FROM jobs
                    WHERE status = 'completed' AND started_at IS NOT NULL AND queued_at IS NOT NULL
                    """
                )
                row = cursor.fetchone()
                stats["avg_wait_seconds"] = row["avg_wait"] if row else None

                return stats
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get_stats)

    async def count_gpus_by_org(self) -> Dict[int, int]:
        """Get count of allocated GPUs per organization for running local jobs."""
        loop = asyncio.get_running_loop()

        def _count():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT org_id, allocated_gpus FROM jobs
                    WHERE job_type = 'local' AND status = 'running'
                    AND org_id IS NOT NULL AND allocated_gpus IS NOT NULL
                    """
                )
                org_gpu_counts: Dict[int, int] = {}
                for row in cursor.fetchall():
                    org_id = row["org_id"]
                    gpus = json.loads(row["allocated_gpus"])
                    if isinstance(gpus, list):
                        org_gpu_counts[org_id] = org_gpu_counts.get(org_id, 0) + len(gpus)
                return org_gpu_counts
            finally:
                conn.close()

        return await loop.run_in_executor(None, _count)

    async def get_org_gpu_usage(self, org_id: int) -> int:
        """Get the number of GPUs currently used by a specific organization."""
        loop = asyncio.get_running_loop()

        def _count():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT allocated_gpus FROM jobs
                    WHERE job_type = 'local' AND status = 'running'
                    AND org_id = ? AND allocated_gpus IS NOT NULL
                    """,
                    (org_id,),
                )
                total_gpus = 0
                for row in cursor.fetchall():
                    gpus = json.loads(row["allocated_gpus"])
                    if isinstance(gpus, list):
                        total_gpus += len(gpus)
                return total_gpus
            finally:
                conn.close()

        return await loop.run_in_executor(None, _count)

    async def list_pending_by_priority(self, limit: int = 50) -> List[UnifiedJob]:
        """List pending cloud jobs by priority for scheduling."""
        loop = asyncio.get_running_loop()

        def _list():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM jobs
                    WHERE status IN ('pending', 'queued')
                      AND job_type != 'local'
                    ORDER BY priority DESC, queued_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                )
                return [self._row_to_job(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _list)

    async def count_running_by_user(self) -> Dict[int, int]:
        """Get running job count per user."""
        loop = asyncio.get_running_loop()

        def _count():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT user_id, COUNT(*) as running_count
                    FROM jobs
                    WHERE status = 'running' AND user_id IS NOT NULL
                    GROUP BY user_id
                    """
                )
                return {row["user_id"]: row["running_count"] for row in cursor.fetchall()}
            finally:
                conn.close()

        return await loop.run_in_executor(None, _count)

    async def count_running_by_team(self) -> Dict[str, int]:
        """Get running job count per team."""
        loop = asyncio.get_running_loop()

        def _count():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT team_id, COUNT(*) as running_count
                    FROM jobs
                    WHERE status = 'running' AND team_id IS NOT NULL
                    GROUP BY team_id
                    """
                )
                return {row["team_id"]: row["running_count"] for row in cursor.fetchall()}
            finally:
                conn.close()

        return await loop.run_in_executor(None, _count)

    async def get_positions_batch(self, job_ids: List[str]) -> Dict[str, int]:
        """Get queue positions for multiple jobs."""
        if not job_ids:
            return {}

        loop = asyncio.get_running_loop()

        def _get_batch():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                placeholders = ",".join("?" * len(job_ids))
                cursor.execute(
                    f"SELECT job_id, queue_position FROM jobs WHERE job_id IN ({placeholders})",
                    job_ids,
                )
                return {row["job_id"]: row["queue_position"] for row in cursor.fetchall()}
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get_batch)

    async def recalculate_queue_positions(self) -> int:
        """Recalculate queue positions for pending jobs."""
        loop = asyncio.get_running_loop()

        def _recalculate():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # Get pending jobs ordered by priority and queue time
                cursor.execute(
                    """
                    SELECT job_id FROM jobs
                    WHERE status IN ('pending', 'queued')
                    ORDER BY priority DESC, queued_at ASC
                    """
                )
                job_ids = [row["job_id"] for row in cursor.fetchall()]

                # Update positions
                for i, job_id in enumerate(job_ids, start=1):
                    cursor.execute(
                        "UPDATE jobs SET queue_position = ? WHERE job_id = ?",
                        (i, job_id),
                    )
                conn.commit()
                return len(job_ids)
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _recalculate)

    async def cleanup_old_entries(self, days: int = 30) -> int:
        """Clean up old completed/failed/cancelled jobs.

        Args:
            days: Delete terminal jobs older than this many days

        Returns:
            Number of jobs deleted
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_iso = cutoff.isoformat()

        loop = asyncio.get_running_loop()

        def _cleanup():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    DELETE FROM jobs
                    WHERE completed_at < ?
                    AND status IN ('completed', 'failed', 'cancelled')
                    """,
                    (cutoff_iso,),
                )
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

        async with self._lock:
            return await loop.run_in_executor(None, _cleanup)


# Singleton accessor
_job_repository: Optional[JobRepository] = None


def get_job_repository() -> JobRepository:
    """Get the singleton JobRepository instance."""
    global _job_repository
    if _job_repository is None:
        _job_repository = JobRepository()
    return _job_repository
