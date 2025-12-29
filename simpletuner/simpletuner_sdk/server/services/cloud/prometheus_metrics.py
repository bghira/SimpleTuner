"""Prometheus-compatible metrics for cloud training.

Provides metrics export in Prometheus text format without requiring
the prometheus_client library. Metrics are collected from JobStore
and other sources.

Categories:
- jobs: Job counts, status, queue depth, duration, costs
- http: HTTP requests, errors, latency
- rate_limits: Rate limit violations, tracked clients
- approvals: Approval request counts by status
- audit: Audit log entries by type
- health: Uptime, database latency, component status
- circuit_breakers: Circuit breaker states and counts
- provider: Cost limits, credit balance
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# All available categories
ALL_CATEGORIES = {
    "jobs",
    "http",
    "rate_limits",
    "approvals",
    "audit",
    "health",
    "circuit_breakers",
    "provider",
}


@dataclass
class MetricValue:
    """A single metric value with labels."""

    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp_ms: Optional[int] = None


@dataclass
class Metric:
    """A Prometheus metric with multiple values."""

    name: str
    help_text: str
    metric_type: str  # counter, gauge, histogram, summary
    values: List[MetricValue] = field(default_factory=list)

    def to_prometheus(self) -> str:
        """Convert to Prometheus text format."""
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} {self.metric_type}",
        ]

        for mv in self.values:
            if mv.labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in sorted(mv.labels.items()))
                metric_line = f"{self.name}{{{label_str}}} {mv.value}"
            else:
                metric_line = f"{self.name} {mv.value}"

            if mv.timestamp_ms:
                metric_line += f" {mv.timestamp_ms}"

            lines.append(metric_line)

        return "\n".join(lines)


class CloudMetricsCollector:
    """Collects and exports cloud training metrics in Prometheus format."""

    _instance: Optional["CloudMetricsCollector"] = None
    _lock = threading.Lock()

    # In-memory counters for request tracking
    _request_counts: Dict[str, int] = {}
    _error_counts: Dict[str, int] = {}
    _latency_sums: Dict[str, float] = {}
    _latency_counts: Dict[str, int] = {}

    # Rate limit violation tracking
    _rate_limit_violations: Dict[str, int] = {}

    def __new__(cls) -> "CloudMetricsCollector":
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self._start_time = time.time()
        self._initialized = True

    def record_request(self, endpoint: str, method: str, status_code: int, latency_ms: float) -> None:
        """Record an HTTP request for metrics."""
        key = f"{method}_{endpoint}"
        with self._lock:
            self._request_counts[key] = self._request_counts.get(key, 0) + 1
            self._latency_sums[key] = self._latency_sums.get(key, 0.0) + latency_ms
            self._latency_counts[key] = self._latency_counts.get(key, 0) + 1

            if status_code >= 400:
                error_key = f"{key}_{status_code}"
                self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1

    def record_rate_limit_violation(self, endpoint: str) -> None:
        """Record a rate limit violation for metrics."""
        with self._lock:
            self._rate_limit_violations[endpoint] = self._rate_limit_violations.get(endpoint, 0) + 1

    async def collect(self, categories: Optional[List[str]] = None) -> List[Metric]:
        """Collect metrics for specified categories.

        Args:
            categories: List of category names to collect. If None, collects all.

        Returns:
            List of Metric objects.
        """
        # Determine which categories to collect
        if categories is None:
            cats = ALL_CATEGORIES
        else:
            cats = set(categories) & ALL_CATEGORIES

        metrics: List[Metric] = []

        # Collect each category
        if "health" in cats:
            metrics.extend(await self._collect_health_metrics())

        if "jobs" in cats:
            metrics.extend(await self._collect_jobs_metrics())

        if "http" in cats:
            metrics.extend(self._collect_http_metrics())

        if "rate_limits" in cats:
            metrics.extend(self._collect_rate_limit_metrics())

        if "approvals" in cats:
            metrics.extend(await self._collect_approval_metrics())

        if "audit" in cats:
            metrics.extend(await self._collect_audit_metrics())

        if "circuit_breakers" in cats:
            metrics.extend(self._collect_circuit_breaker_metrics())

        if "provider" in cats:
            metrics.extend(await self._collect_provider_metrics())

        return metrics

    async def _collect_health_metrics(self) -> List[Metric]:
        """Collect health and uptime metrics."""
        metrics: List[Metric] = []

        # Server uptime
        uptime = time.time() - self._start_time
        metrics.append(
            Metric(
                name="simpletuner_uptime_seconds",
                help_text="Server uptime in seconds",
                metric_type="gauge",
                values=[MetricValue(value=uptime)],
            )
        )

        # Database latency check
        try:
            from .container import get_job_store

            start = time.time()
            store = get_job_store()
            await store.list_jobs(limit=1)
            latency_ms = (time.time() - start) * 1000

            metrics.append(
                Metric(
                    name="simpletuner_health_database_latency_ms",
                    help_text="Database query latency in milliseconds",
                    metric_type="gauge",
                    values=[MetricValue(value=latency_ms)],
                )
            )

            metrics.append(
                Metric(
                    name="simpletuner_health_component_status",
                    help_text="Component health status (0=unhealthy, 1=degraded, 2=healthy)",
                    metric_type="gauge",
                    values=[MetricValue(value=2, labels={"component": "database"})],
                )
            )
        except Exception as exc:
            logger.warning("Error checking database health: %s", exc)
            metrics.append(
                Metric(
                    name="simpletuner_health_component_status",
                    help_text="Component health status (0=unhealthy, 1=degraded, 2=healthy)",
                    metric_type="gauge",
                    values=[MetricValue(value=0, labels={"component": "database"})],
                )
            )

        return metrics

    async def _collect_jobs_metrics(self) -> List[Metric]:
        """Collect job-related metrics."""
        from .base import CloudJobStatus
        from .container import get_job_store

        metrics: List[Metric] = []

        try:
            store = get_job_store()
            summary = await store.get_metrics_summary(days=30)

            # Total jobs
            metrics.append(
                Metric(
                    name="simpletuner_jobs_total",
                    help_text="Total number of jobs in the last 30 days",
                    metric_type="gauge",
                    values=[MetricValue(value=summary.get("job_count_30d", 0))],
                )
            )

            # Total cost
            metrics.append(
                Metric(
                    name="simpletuner_cost_usd_total",
                    help_text="Total cost in USD in the last 30 days",
                    metric_type="gauge",
                    values=[MetricValue(value=summary.get("total_cost_30d", 0.0))],
                )
            )

            # Jobs by status
            jobs_by_status = summary.get("jobs_by_status", {})
            if jobs_by_status:
                status_values = [
                    MetricValue(value=count, labels={"status": status}) for status, count in jobs_by_status.items()
                ]
                metrics.append(
                    Metric(
                        name="simpletuner_jobs_by_status",
                        help_text="Number of jobs by status",
                        metric_type="gauge",
                        values=status_values,
                    )
                )

            # Average job duration
            avg_duration = summary.get("avg_job_duration_seconds")
            if avg_duration is not None:
                metrics.append(
                    Metric(
                        name="simpletuner_job_duration_seconds_avg",
                        help_text="Average job duration in seconds",
                        metric_type="gauge",
                        values=[MetricValue(value=avg_duration)],
                    )
                )

            # Active jobs (queue depth)
            active_statuses = [
                CloudJobStatus.PENDING.value,
                CloudJobStatus.UPLOADING.value,
                CloudJobStatus.QUEUED.value,
                CloudJobStatus.RUNNING.value,
            ]
            active_count = sum(jobs_by_status.get(s, 0) for s in active_statuses)
            metrics.append(
                Metric(
                    name="simpletuner_jobs_active",
                    help_text="Number of active (non-terminal) jobs",
                    metric_type="gauge",
                    values=[MetricValue(value=active_count)],
                )
            )

        except Exception as exc:
            logger.warning("Error collecting job metrics: %s", exc)

        return metrics

    def _collect_http_metrics(self) -> List[Metric]:
        """Collect HTTP request metrics."""
        metrics: List[Metric] = []

        with self._lock:
            if self._request_counts:
                request_values = [
                    MetricValue(value=count, labels={"endpoint": key}) for key, count in self._request_counts.items()
                ]
                metrics.append(
                    Metric(
                        name="simpletuner_http_requests_total",
                        help_text="Total HTTP requests by endpoint",
                        metric_type="counter",
                        values=request_values,
                    )
                )

            if self._error_counts:
                error_values = [
                    MetricValue(value=count, labels={"endpoint_status": key}) for key, count in self._error_counts.items()
                ]
                metrics.append(
                    Metric(
                        name="simpletuner_http_errors_total",
                        help_text="Total HTTP errors by endpoint and status",
                        metric_type="counter",
                        values=error_values,
                    )
                )

            if self._latency_counts:
                latency_values = []
                for key, count in self._latency_counts.items():
                    avg_latency = self._latency_sums.get(key, 0) / count if count > 0 else 0
                    latency_values.append(MetricValue(value=avg_latency, labels={"endpoint": key}))
                if latency_values:
                    metrics.append(
                        Metric(
                            name="simpletuner_http_request_latency_ms_avg",
                            help_text="Average HTTP request latency in milliseconds",
                            metric_type="gauge",
                            values=latency_values,
                        )
                    )

        return metrics

    def _collect_rate_limit_metrics(self) -> List[Metric]:
        """Collect rate limiting metrics."""
        metrics: List[Metric] = []

        with self._lock:
            if self._rate_limit_violations:
                violation_values = [
                    MetricValue(value=count, labels={"endpoint": endpoint})
                    for endpoint, count in self._rate_limit_violations.items()
                ]
                metrics.append(
                    Metric(
                        name="simpletuner_rate_limit_violations_total",
                        help_text="Total rate limit violations by endpoint",
                        metric_type="counter",
                        values=violation_values,
                    )
                )

                # Total violations
                total_violations = sum(self._rate_limit_violations.values())
                metrics.append(
                    Metric(
                        name="simpletuner_rate_limit_tracked_clients",
                        help_text="Number of endpoints with rate limit violations",
                        metric_type="gauge",
                        values=[MetricValue(value=len(self._rate_limit_violations))],
                    )
                )

        return metrics

    async def _collect_approval_metrics(self) -> List[Metric]:
        """Collect approval request metrics."""
        metrics: List[Metric] = []

        try:
            from .approval.approval_store import ApprovalStore

            store = ApprovalStore()
            pending_count = await store.get_pending_count()

            metrics.append(
                Metric(
                    name="simpletuner_approval_requests_pending",
                    help_text="Number of pending approval requests",
                    metric_type="gauge",
                    values=[MetricValue(value=pending_count)],
                )
            )

            # Get counts by status
            status_counts = await store.get_status_counts()
            if status_counts:
                status_values = [
                    MetricValue(value=count, labels={"status": status}) for status, count in status_counts.items()
                ]
                metrics.append(
                    Metric(
                        name="simpletuner_approval_requests_by_status",
                        help_text="Approval requests by status",
                        metric_type="gauge",
                        values=status_values,
                    )
                )

        except Exception as exc:
            logger.debug("Approval metrics not available: %s", exc)

        return metrics

    async def _collect_audit_metrics(self) -> List[Metric]:
        """Collect audit log metrics."""
        metrics: List[Metric] = []

        try:
            from .audit import AuditStore

            store = AuditStore()
            stats = await store.get_stats()

            # Total entries
            metrics.append(
                Metric(
                    name="simpletuner_audit_log_entries_total",
                    help_text="Total audit log entries",
                    metric_type="gauge",
                    values=[MetricValue(value=stats.get("total_entries", 0))],
                )
            )

            # Entries by type
            by_type = stats.get("by_type", {})
            if by_type:
                type_values = [
                    MetricValue(value=count, labels={"event_type": event_type}) for event_type, count in by_type.items()
                ]
                metrics.append(
                    Metric(
                        name="simpletuner_audit_log_entries_by_type",
                        help_text="Audit log entries by event type",
                        metric_type="gauge",
                        values=type_values,
                    )
                )

            # Last 24 hours
            metrics.append(
                Metric(
                    name="simpletuner_audit_log_entries_24h",
                    help_text="Audit log entries in the last 24 hours",
                    metric_type="gauge",
                    values=[MetricValue(value=stats.get("last_24h", 0))],
                )
            )

        except Exception as exc:
            logger.debug("Audit metrics not available: %s", exc)

        return metrics

    def _collect_circuit_breaker_metrics(self) -> List[Metric]:
        """Collect circuit breaker metrics."""
        metrics: List[Metric] = []

        try:
            from .resilience import get_all_circuit_breaker_health

            breaker_health = get_all_circuit_breaker_health()

            state_values = []
            failure_values = []
            success_values = []

            for name, info in breaker_health.items():
                state = info.get("state", "unknown")
                # Map state to numeric value: closed=0, half_open=1, open=2
                state_num = {"closed": 0, "half_open": 1, "open": 2}.get(state, -1)
                state_values.append(MetricValue(value=state_num, labels={"breaker": name}))

                failure_count = info.get("failure_count", 0)
                failure_values.append(MetricValue(value=failure_count, labels={"breaker": name}))

                success_count = info.get("success_count", 0)
                success_values.append(MetricValue(value=success_count, labels={"breaker": name}))

            if state_values:
                metrics.append(
                    Metric(
                        name="simpletuner_circuit_breaker_state",
                        help_text="Circuit breaker state (0=closed, 1=half_open, 2=open)",
                        metric_type="gauge",
                        values=state_values,
                    )
                )

            if failure_values:
                metrics.append(
                    Metric(
                        name="simpletuner_circuit_breaker_failures_total",
                        help_text="Total circuit breaker failures",
                        metric_type="counter",
                        values=failure_values,
                    )
                )

            if success_values:
                metrics.append(
                    Metric(
                        name="simpletuner_circuit_breaker_successes_total",
                        help_text="Total circuit breaker successes",
                        metric_type="counter",
                        values=success_values,
                    )
                )

        except Exception as exc:
            logger.debug("Circuit breaker metrics not available: %s", exc)

        return metrics

    async def _collect_provider_metrics(self) -> List[Metric]:
        """Collect provider-specific metrics (cost limits, credit balance)."""
        metrics: List[Metric] = []

        try:
            from .container import get_job_store
            from .storage.provider_config_store import ProviderConfigStore

            config_store = ProviderConfigStore()
            job_store = get_job_store()

            for provider in ["replicate"]:
                config = await config_store.get(provider)

                # Cost limit metrics
                if config.get("cost_limit_enabled"):
                    limit_amount = config.get("cost_limit_amount", 0)
                    period = config.get("cost_limit_period", "monthly")

                    # Get current spend
                    days = {"daily": 1, "weekly": 7, "monthly": 30}.get(period, 30)
                    summary = await job_store.get_metrics_summary(days=days)
                    current_spend = summary.get("total_cost_30d", 0.0)

                    percent_used = (current_spend / limit_amount) * 100 if limit_amount > 0 else 0

                    metrics.append(
                        Metric(
                            name="simpletuner_cost_limit_percent_used",
                            help_text="Cost limit usage percentage",
                            metric_type="gauge",
                            values=[MetricValue(value=percent_used, labels={"provider": provider})],
                        )
                    )

        except Exception as exc:
            logger.debug("Provider metrics not available: %s", exc)

        return metrics

    async def export_prometheus(self, categories: Optional[List[str]] = None) -> str:
        """Export metrics in Prometheus text format.

        Args:
            categories: List of categories to export. If None, exports all.

        Returns:
            Prometheus-formatted metrics string.
        """
        metrics = await self.collect(categories=categories)
        if not metrics:
            return "# No metrics available\n"
        return "\n\n".join(m.to_prometheus() for m in metrics) + "\n"

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None
            cls._request_counts = {}
            cls._error_counts = {}
            cls._latency_sums = {}
            cls._latency_counts = {}
            cls._rate_limit_violations = {}


def get_metrics_collector() -> CloudMetricsCollector:
    """Get the global metrics collector instance."""
    return CloudMetricsCollector()
