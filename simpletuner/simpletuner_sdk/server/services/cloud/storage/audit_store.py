"""Audit log storage for tracking cloud training operations.

Provides storage and retrieval of audit events for compliance,
debugging, and operational visibility.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .async_base import AsyncSQLiteStore

logger = logging.getLogger(__name__)


class AuditStore(AsyncSQLiteStore):
    """Store for audit log entries.

    Records significant events like job submissions, status changes,
    configuration updates, and user actions for audit trails.
    """

    async def _init_schema(self) -> None:
        """Initialize the audit_log table."""
        conn = await self._get_connection()

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                job_id TEXT,
                provider TEXT,
                config_name TEXT,
                user_ip TEXT,
                user_id TEXT,
                details TEXT DEFAULT '{}'
            )
        """
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_job_id ON audit_log(job_id)")
        await conn.commit()

    async def log_event(
        self,
        action: str,
        job_id: Optional[str] = None,
        provider: Optional[str] = None,
        config_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        user_ip: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """Log an audit event.

        Args:
            action: The action being logged (e.g., "job_submitted", "job_cancelled")
            job_id: Associated job ID if applicable
            provider: Cloud provider if applicable
            config_name: Configuration name if applicable
            details: Additional event details as a dict
            user_ip: IP address of the user
            user_id: User ID (as string for flexibility)

        Returns:
            The ID of the created audit entry
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        async with self.transaction() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO audit_log
                (timestamp, action, job_id, provider, config_name, user_ip, user_id, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    action,
                    job_id,
                    provider,
                    config_name,
                    user_ip,
                    user_id,
                    json.dumps(details or {}),
                ),
            )
            return cursor.lastrowid

    async def get_entries(
        self,
        limit: int = 100,
        offset: int = 0,
        action: Optional[str] = None,
        job_id: Optional[str] = None,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries with optional filtering.

        Args:
            limit: Maximum number of entries to return
            offset: Number of entries to skip
            action: Filter by action type
            job_id: Filter by job ID
            user_id: Filter by user ID
            since: Only return entries after this timestamp

        Returns:
            List of audit entries as dicts
        """
        query = """
            SELECT timestamp, action, job_id, provider, config_name, user_ip, user_id, details
            FROM audit_log
            WHERE 1=1
        """
        params: List[Any] = []

        if action:
            query += " AND action = ?"
            params.append(action)
        if job_id:
            query += " AND job_id = ?"
            params.append(job_id)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if since:
            query += " AND timestamp > ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = await self.fetch_all(query, tuple(params))

        entries = []
        for row in rows:
            entry = {
                "timestamp": row["timestamp"],
                "action": row["action"],
            }
            # Only include non-null fields
            for key in ("job_id", "provider", "config_name", "user_ip", "user_id"):
                if row[key]:
                    entry[key] = row[key]
            if row["details"]:
                try:
                    entry["details"] = json.loads(row["details"])
                except json.JSONDecodeError:
                    pass
            entries.append(entry)

        return entries

    async def get_job_history(self, job_id: str) -> List[Dict[str, Any]]:
        """Get all audit entries for a specific job.

        Args:
            job_id: The job ID to get history for

        Returns:
            List of audit entries in chronological order
        """
        rows = await self.fetch_all(
            """
            SELECT timestamp, action, provider, config_name, user_ip, user_id, details
            FROM audit_log
            WHERE job_id = ?
            ORDER BY timestamp ASC
            """,
            (job_id,),
        )

        entries = []
        for row in rows:
            entry = {
                "timestamp": row["timestamp"],
                "action": row["action"],
            }
            for key in ("provider", "config_name", "user_ip", "user_id"):
                if row[key]:
                    entry[key] = row[key]
            if row["details"]:
                try:
                    entry["details"] = json.loads(row["details"])
                except json.JSONDecodeError:
                    pass
            entries.append(entry)

        return entries

    async def cleanup(self, max_age_days: int = 90) -> int:
        """Remove old audit entries.

        Args:
            max_age_days: Delete entries older than this many days

        Returns:
            Number of entries deleted
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()

        async with self.transaction() as conn:
            cursor = await conn.execute("DELETE FROM audit_log WHERE timestamp < ?", (cutoff,))
            deleted = cursor.rowcount

        if deleted > 0:
            logger.info("Cleaned up %d audit entries older than %d days", deleted, max_age_days)

        return deleted

    async def get_entry_count(self) -> int:
        """Get total number of audit entries.

        Returns:
            Total entry count
        """
        row = await self.fetch_one("SELECT COUNT(*) FROM audit_log")
        return row[0] if row else 0


# Singleton access
_instance: Optional[AuditStore] = None


async def get_audit_store() -> AuditStore:
    """Get the singleton AuditStore instance."""
    global _instance
    if _instance is None:
        _instance = await AuditStore.get_instance()
    return _instance
