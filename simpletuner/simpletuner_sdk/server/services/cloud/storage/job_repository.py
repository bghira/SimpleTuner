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
SCHEMA_VERSION = 4


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

            # Create jobs table
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
                    metadata TEXT DEFAULT '{}'
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

            # Check and set schema version
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
                        hardware_type, error_message, output_url, upload_token, user_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

        user_id = None
        try:
            user_id = row["user_id"]
        except (IndexError, KeyError):
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
            user_id=user_id,
            metadata=metadata,
        )


# Singleton accessor
_job_repository: Optional[JobRepository] = None


def get_job_repository() -> JobRepository:
    """Get the singleton JobRepository instance."""
    global _job_repository
    if _job_repository is None:
        _job_repository = JobRepository()
    return _job_repository
