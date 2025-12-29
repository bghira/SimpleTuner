"""Cloud metrics aggregation service."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import DashboardMetrics
from .container import get_job_store

logger = logging.getLogger(__name__)


class MetricsService:
    """
    Service for aggregating cloud dashboard metrics.

    Combines data from:
    - AsyncJobStore (historical job data)
    - Cloud provider APIs (billing info)
    """

    def __init__(self, job_store=None):
        """
        Initialize the metrics service.

        Args:
            job_store: AsyncJobStore instance. If None, uses container singleton.
        """
        self._job_store = job_store or get_job_store()

    async def get_dashboard_metrics(
        self,
        days: int = 30,
        include_billing: bool = True,
    ) -> DashboardMetrics:
        """
        Get aggregated dashboard metrics.

        Args:
            days: Number of days to include in summaries
            include_billing: Whether to fetch billing info from providers

        Returns:
            DashboardMetrics with all aggregated data
        """
        # Get metrics from job store
        summary = await self._job_store.get_metrics_summary(days=days)
        cost_by_day = await self._job_store.get_job_history_for_chart(days=days)

        # Initialize metrics
        metrics = DashboardMetrics(
            total_cost_30d=summary.get("total_cost_30d", 0.0),
            job_count_30d=summary.get("job_count_30d", 0),
            avg_job_duration_seconds=summary.get("avg_job_duration_seconds"),
            jobs_by_status=summary.get("jobs_by_status", {}),
            cost_by_day=cost_by_day,
        )

        # Fetch billing info if requested
        if include_billing:
            billing = await self._get_provider_billing()
            metrics.credit_balance = billing.get("credit_balance")
            metrics.estimated_jobs_remaining = billing.get("estimated_jobs")

        return metrics

    async def _get_provider_billing(self) -> Dict[str, Any]:
        """Fetch billing info from cloud providers.

        Note: Replicate's billing API requires browser session auth,
        not API tokens, so we don't attempt to fetch it here.
        """
        return {}

    async def get_job_history_chart_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get job history formatted for Chart.js.

        Returns data in format suitable for Chart.js line/bar charts:
        [
            {"date": "2025-01-01", "total": 5, "completed": 3, "failed": 1, ...},
            ...
        ]
        """
        return await self._job_store.get_job_history_for_chart(days=days)

    async def get_cost_over_time(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get daily cost data for Chart.js line chart.

        Returns:
            List of {"date": str, "cost": float} dicts
        """
        history = await self._job_store.get_job_history_for_chart(days=days)
        return [{"date": day["date"], "cost": day.get("total_cost_usd", 0.0)} for day in history]

    async def get_jobs_by_status_chart_data(self, days: int = 30) -> Dict[str, int]:
        """
        Get job counts by status for Chart.js pie/doughnut chart.

        Returns:
            Dict mapping status name to count
        """
        summary = await self._job_store.get_metrics_summary(days=days)
        return summary.get("jobs_by_status", {})
