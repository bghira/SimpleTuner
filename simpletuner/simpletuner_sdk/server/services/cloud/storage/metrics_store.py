"""Metrics storage for aggregated job statistics.

Provides aggregated metrics queries for dashboards, billing,
and operational monitoring.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .async_base import AsyncSQLiteStore

logger = logging.getLogger(__name__)


class MetricsStore(AsyncSQLiteStore):
    """Store for aggregated job metrics.

    Provides pre-computed aggregations over job data for efficient
    dashboard rendering and reporting.
    """

    async def _init_schema(self) -> None:
        """No additional tables needed - queries against jobs table."""
        # Metrics are computed from the jobs table, no separate schema needed.
        # Just verify connection works.
        await self._get_connection()

    async def get_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary metrics for the specified period.

        Args:
            days: Number of days to include in the summary

        Returns:
            Dict with total_cost, job_count, avg_duration, and status breakdown
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        # Total jobs and cost
        totals_row = await self.fetch_one(
            """
            SELECT COUNT(*) as total_jobs, COALESCE(SUM(cost_usd), 0) as total_cost
            FROM jobs WHERE created_at >= ?
            """,
            (cutoff,),
        )

        # Jobs by status
        status_rows = await self.fetch_all(
            """
            SELECT status, COUNT(*) as count FROM jobs
            WHERE created_at >= ? GROUP BY status
            """,
            (cutoff,),
        )
        status_counts = {row["status"]: row["count"] for row in status_rows}

        # Average duration for completed jobs
        avg_row = await self.fetch_one(
            """
            SELECT AVG((julianday(completed_at) - julianday(started_at)) * 86400) as avg_duration
            FROM jobs
            WHERE created_at >= ? AND status = 'completed'
              AND started_at IS NOT NULL AND completed_at IS NOT NULL
            """,
            (cutoff,),
        )

        return {
            "total_cost_usd": totals_row["total_cost"] if totals_row else 0,
            "job_count": totals_row["total_jobs"] if totals_row else 0,
            "avg_job_duration_seconds": avg_row["avg_duration"] if avg_row else None,
            "jobs_by_status": status_counts,
            "period_days": days,
        }

    async def get_daily_breakdown(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get job statistics aggregated by day.

        Args:
            days: Number of days to include

        Returns:
            List of daily stats dicts with date, counts, and costs
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        rows = await self.fetch_all(
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
            (cutoff,),
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
            for row in rows
        ]

    async def get_user_summary(
        self,
        user_id: int,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get metrics summary for a specific user.

        Args:
            user_id: User to get metrics for
            days: Number of days to include

        Returns:
            Dict with user-specific metrics
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        totals_row = await self.fetch_one(
            """
            SELECT COUNT(*) as total_jobs, COALESCE(SUM(cost_usd), 0) as total_cost
            FROM jobs WHERE user_id = ? AND created_at >= ?
            """,
            (user_id, cutoff),
        )

        status_rows = await self.fetch_all(
            """
            SELECT status, COUNT(*) as count FROM jobs
            WHERE user_id = ? AND created_at >= ? GROUP BY status
            """,
            (user_id, cutoff),
        )
        status_counts = {row["status"]: row["count"] for row in status_rows}

        # Active jobs count
        active_row = await self.fetch_one(
            """
            SELECT COUNT(*) as active FROM jobs
            WHERE user_id = ? AND status IN ('pending', 'uploading', 'queued', 'running')
            """,
            (user_id,),
        )

        return {
            "total_cost_usd": totals_row["total_cost"] if totals_row else 0,
            "job_count": totals_row["total_jobs"] if totals_row else 0,
            "jobs_by_status": status_counts,
            "active_jobs": active_row["active"] if active_row else 0,
            "period_days": days,
        }

    async def get_provider_breakdown(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get job statistics grouped by provider.

        Args:
            days: Number of days to include

        Returns:
            List of per-provider stats
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        rows = await self.fetch_all(
            """
            SELECT
                provider,
                COUNT(*) as total_jobs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                COALESCE(SUM(cost_usd), 0) as total_cost_usd,
                AVG(CASE
                    WHEN status = 'completed' AND started_at IS NOT NULL AND completed_at IS NOT NULL
                    THEN (julianday(completed_at) - julianday(started_at)) * 86400
                    ELSE NULL
                END) as avg_duration_seconds
            FROM jobs
            WHERE created_at >= ? AND provider IS NOT NULL
            GROUP BY provider
            ORDER BY total_jobs DESC
            """,
            (cutoff,),
        )

        return [
            {
                "provider": row["provider"],
                "total_jobs": row["total_jobs"],
                "completed": row["completed"],
                "failed": row["failed"],
                "total_cost_usd": row["total_cost_usd"],
                "avg_duration_seconds": row["avg_duration_seconds"],
            }
            for row in rows
        ]

    async def get_cost_by_period(
        self,
        period: str = "day",
        days: int = 30,
        user_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get cost aggregated by time period.

        Args:
            period: Aggregation period ('day', 'week', 'month')
            days: Number of days to include
            user_id: Optional user filter

        Returns:
            List of period costs
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        # SQLite date formatting for different periods
        if period == "week":
            date_fmt = "strftime('%Y-W%W', created_at)"
        elif period == "month":
            date_fmt = "strftime('%Y-%m', created_at)"
        else:
            date_fmt = "date(created_at)"

        query = f"""
            SELECT
                {date_fmt} as period_key,
                COALESCE(SUM(cost_usd), 0) as total_cost_usd,
                COUNT(*) as job_count
            FROM jobs
            WHERE created_at >= ?
        """
        params: List[Any] = [cutoff]

        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)

        query += f" GROUP BY {date_fmt} ORDER BY period_key"

        rows = await self.fetch_all(query, tuple(params))

        return [
            {
                "period": row["period_key"],
                "total_cost_usd": row["total_cost_usd"],
                "job_count": row["job_count"],
            }
            for row in rows
        ]


# Singleton access
_instance: Optional[MetricsStore] = None


async def get_metrics_store() -> MetricsStore:
    """Get the singleton MetricsStore instance."""
    global _instance
    if _instance is None:
        _instance = await MetricsStore.get_instance()
    return _instance
