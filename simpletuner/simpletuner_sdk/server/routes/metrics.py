"""Metrics and monitoring endpoints.

NOTE: This module was moved from routes/cloud/metrics.py to become a top-level
global route, as metrics apply to all jobs, not just cloud jobs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from simpletuner.simpletuner_sdk.server.services.cloud.auth.middleware import get_current_user
from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User

from .cloud._shared import (
    BillingResponse,
    CostLimitStatusResponse,
    HealthCheckComponent,
    HealthCheckResponse,
    MetricsResponse,
    SystemStatusResponse,
    get_job_store,
    get_period_days,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/metrics", tags=["metrics"])

# Include metrics configuration sub-router at /config
from .cloud.metrics_config import router as metrics_config_router

router.include_router(metrics_config_router, prefix="/config")


async def check_cost_limit(store, provider: str) -> CostLimitStatusResponse:
    """Check cost limit status for a provider."""
    config = await store.get_provider_config(provider)

    enabled = config.get("cost_limit_enabled", False)
    if not enabled:
        return CostLimitStatusResponse(enabled=False, message="Cost limits not configured")

    limit_amount = config.get("cost_limit_amount")
    period = config.get("cost_limit_period", "monthly")
    action = config.get("cost_limit_action", "warn")

    if not limit_amount or limit_amount <= 0:
        return CostLimitStatusResponse(enabled=False, message="No limit amount set")

    days = get_period_days(period)
    summary = await store.get_metrics_summary(days=days)
    current_spend = summary.get("total_cost_30d", 0.0)

    percent_used = (current_spend / limit_amount) * 100 if limit_amount > 0 else 0
    is_exceeded = current_spend >= limit_amount
    is_warning = percent_used >= 80 and not is_exceeded

    message = None
    if is_exceeded:
        message = f"Spending limit exceeded: ${current_spend:.2f} / ${limit_amount:.2f} ({period})"
    elif is_warning:
        message = f"Approaching limit: ${current_spend:.2f} / ${limit_amount:.2f} ({period})"

    return CostLimitStatusResponse(
        enabled=True,
        limit_amount=limit_amount,
        period=period,
        action=action,
        current_spend=current_spend,
        percent_used=round(percent_used, 1),
        is_exceeded=is_exceeded,
        is_warning=is_warning,
        message=message,
    )


@router.get("", response_model=MetricsResponse)
async def get_metrics(
    days: int = Query(30, ge=1, le=365, description="Number of days for metrics period"),
    _user: User = Depends(get_current_user),
) -> MetricsResponse:
    """Get dashboard metrics (without billing - use /billing/refresh for that)."""
    store = get_job_store()

    summary = await store.get_metrics_summary(days=days)
    cost_by_day = await store.get_job_history_for_chart(days=days)

    return MetricsResponse(
        credit_balance=None,
        estimated_jobs_remaining=None,
        total_cost=summary.get("total_cost_30d", 0.0),
        job_count=summary.get("job_count_30d", 0),
        avg_job_duration_seconds=summary.get("avg_job_duration_seconds"),
        jobs_by_status=summary.get("jobs_by_status", {}),
        cost_by_day=cost_by_day,
        period_days=days,
    )


@router.post("/billing/refresh", response_model=BillingResponse)
async def refresh_billing(
    _user: User = Depends(get_current_user),
) -> BillingResponse:
    """Refresh credit balance from cloud provider (user-triggered)."""
    try:
        from ..services.cloud.factory import ProviderFactory
        from ..services.cloud.replicate_client import get_default_hardware_cost_per_hour

        store = get_job_store()
        client = ProviderFactory.get_provider("replicate")

        billing = await client.get_billing_info()
        credit_balance = billing.get("balance")

        estimated_jobs = None
        if credit_balance is not None and credit_balance > 0:
            provider_config = await store.get_provider_config("replicate")
            default_cost = get_default_hardware_cost_per_hour(store)
            avg_job_cost = provider_config.get("avg_job_cost_usd", default_cost)
            estimated_jobs = int(credit_balance / avg_job_cost)

        return BillingResponse(
            credit_balance=credit_balance,
            estimated_jobs_remaining=estimated_jobs,
        )
    except Exception as exc:
        logger.warning("Failed to refresh billing: %s", exc)
        return BillingResponse(error=str(exc))


@router.get("/cost-limit/status", response_model=CostLimitStatusResponse)
async def get_cost_limit_status(
    provider: str = "replicate",
    _user: User = Depends(get_current_user),
) -> CostLimitStatusResponse:
    """Get the current cost limit status for a provider."""
    store = get_job_store()
    return await check_cost_limit(store, provider)


@router.get("/replicate/status", response_model=SystemStatusResponse)
async def get_replicate_status(
    _user: User = Depends(get_current_user),
) -> SystemStatusResponse:
    """Get Replicate system status from their status page API."""
    from ..services.cloud.http_client import get_async_client

    try:
        url = "https://www.replicatestatus.com/api/v1/summary"

        async with get_async_client(timeout=10.0) as client:
            response = await client.get(url, headers={"Accept": "*/*"})
            response.raise_for_status()
            data = response.json()

        ongoing = data.get("ongoing_incidents", [])
        in_progress = data.get("in_progress_maintenances", [])

        return SystemStatusResponse(
            operational=len(ongoing) == 0 and len(in_progress) == 0,
            ongoing_incidents=ongoing,
            in_progress_maintenances=in_progress,
            scheduled_maintenances=data.get("scheduled_maintenances", []),
            status_page_url=data.get("page_url"),
        )
    except Exception as exc:
        logger.warning("Failed to fetch Replicate status: %s", exc)
        return SystemStatusResponse(error=str(exc))


# Track server start time for uptime
_server_start_time: Optional[float] = None


def _get_server_start_time() -> float:
    """Get or initialize server start time."""
    global _server_start_time
    if _server_start_time is None:
        import time

        _server_start_time = time.time()
    return _server_start_time


class HealthCheckBuilder:
    """Helper class for building health check responses with less repetition."""

    def __init__(self) -> None:
        self.components: list[HealthCheckComponent] = []
        self.overall_status = "healthy"

    def add(
        self,
        name: str,
        status: str,
        message: str,
        latency_ms: Optional[float] = None,
        degrade_on_unhealthy: bool = True,
    ) -> None:
        """Add a health check component.

        Args:
            name: Component name
            status: "healthy", "degraded", or "unhealthy"
            message: Status message
            latency_ms: Optional latency measurement
            degrade_on_unhealthy: If True, degrades overall status on non-healthy
        """
        self.components.append(
            HealthCheckComponent(
                name=name,
                status=status,
                message=message,
                latency_ms=round(latency_ms, 2) if latency_ms is not None else None,
            )
        )

        if degrade_on_unhealthy:
            if status == "unhealthy":
                self.overall_status = "unhealthy"
            elif status == "degraded" and self.overall_status == "healthy":
                self.overall_status = "degraded"


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    include_replicate: bool = Query(False, description="Include Replicate API connectivity check"),
    _user: User = Depends(get_current_user),
) -> HealthCheckResponse:
    """Health check endpoint for monitoring.

    Returns overall health status and individual component statuses.
    Use include_replicate=true to also check Replicate API connectivity (adds latency).

    Returns:
        - status: "healthy", "degraded", or "unhealthy"
        - components: List of component health statuses
        - uptime_seconds: Server uptime
        - timestamp: Current ISO timestamp
    """
    import time

    from ..services.cloud.secrets import get_secrets_manager

    builder = HealthCheckBuilder()

    # Check database connectivity
    db_start = time.time()
    try:
        store = get_job_store()
        await store.list_jobs(limit=1)
        builder.add("database", "healthy", "SQLite database accessible", latency_ms=(time.time() - db_start) * 1000)
    except Exception as exc:
        builder.add("database", "unhealthy", f"Database error: {exc}", latency_ms=(time.time() - db_start) * 1000)

    # Check secrets manager
    try:
        secrets = get_secrets_manager()
        has_token = secrets.get_replicate_token() is not None
        builder.add(
            "secrets",
            "healthy" if has_token else "degraded",
            "API token configured" if has_token else "No Replicate API token configured",
        )
    except Exception as exc:
        builder.add("secrets", "unhealthy", f"Secrets error: {exc}")

    # Check circuit breakers
    try:
        from ..services.cloud.resilience import get_all_circuit_breaker_health

        for name, info in get_all_circuit_breaker_health().items():
            state = info.get("state", "unknown")
            if state == "closed":
                builder.add(
                    f"circuit_breaker_{name}", "healthy", "Circuit closed - normal operation", degrade_on_unhealthy=False
                )
            elif state == "half_open":
                builder.add(f"circuit_breaker_{name}", "degraded", "Circuit half-open - testing recovery")
            else:
                builder.add(
                    f"circuit_breaker_{name}",
                    "degraded",
                    f"Circuit open - blocking requests (failures: {info.get('failure_count', 0)})",
                )
    except Exception as exc:
        logger.debug("Could not check circuit breakers: %s", exc)

    # Optionally check Replicate API connectivity
    if include_replicate:
        from ..services.cloud.http_client import get_async_client

        api_start = time.time()
        try:
            secrets = get_secrets_manager()
            token = secrets.get_replicate_token()
            if token:
                async with get_async_client(timeout=5.0) as client:
                    response = await client.get(
                        "https://api.replicate.com/v1/account",
                        headers={"Authorization": f"Bearer {token}"},
                    )
                api_latency = (time.time() - api_start) * 1000
                if response.status_code == 200:
                    builder.add("replicate_api", "healthy", "Replicate API accessible", latency_ms=api_latency)
                else:
                    builder.add(
                        "replicate_api", "degraded", f"Replicate API returned {response.status_code}", latency_ms=api_latency
                    )
            else:
                builder.add("replicate_api", "degraded", "Skipped - no API token configured", degrade_on_unhealthy=False)
        except Exception as exc:
            builder.add(
                "replicate_api", "degraded", f"Replicate API error: {exc}", latency_ms=(time.time() - api_start) * 1000
            )

    uptime = time.time() - _get_server_start_time()
    return HealthCheckResponse(
        status=builder.overall_status,
        uptime_seconds=round(uptime, 2),
        components=builder.components,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/health/live")
async def liveness_check(
    _user: User = Depends(get_current_user),
) -> dict:
    """Kubernetes-style liveness probe.

    Returns 200 if the server is running. Use for restart decisions.
    """
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness_check(
    _user: User = Depends(get_current_user),
) -> dict:
    """Kubernetes-style readiness probe.

    Returns 200 if the server is ready to accept requests.
    Checks database connectivity.
    """
    try:
        store = get_job_store()
        await store.list_jobs(limit=1)
        return {"status": "ready"}
    except Exception as exc:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Not ready: {exc}",
        )


@router.get("/prometheus")
async def prometheus_metrics(
    _user: User = Depends(get_current_user),
):
    """Export metrics in Prometheus text format.

    Use this endpoint for Prometheus scraping:
    ```yaml
    scrape_configs:
      - job_name: 'simpletuner'
        static_configs:
          - targets: ['localhost:8080']
        metrics_path: '/api/metrics/prometheus'
    ```

    Metrics are exported based on configuration in the Metrics tab.
    Enable Prometheus export and select categories to control what is exposed.

    Available metric categories:
    - jobs: Job counts, status distribution, costs, duration
    - http: HTTP requests, errors, latency by endpoint
    - rate_limits: Rate limit violations
    - approvals: Approval request counts by status
    - audit: Audit log entries by type
    - health: Uptime, database latency, component status
    - circuit_breakers: Circuit breaker states
    - provider: Cost limits, credit balance
    """
    from fastapi.responses import PlainTextResponse

    from ..services.cloud.prometheus_metrics import get_metrics_collector
    from ..services.webui_state import WebUIStateStore

    # Check if Prometheus export is enabled
    store = WebUIStateStore()
    defaults = store.load_defaults()

    if not defaults.metrics_prometheus_enabled:
        return PlainTextResponse(
            content="# Prometheus export disabled. Enable in Metrics tab.\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # Get configured categories
    categories = defaults.metrics_prometheus_categories

    collector = get_metrics_collector()
    content = await collector.export_prometheus(categories=categories)

    return PlainTextResponse(
        content=content,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/circuit-breakers")
async def get_circuit_breaker_status(
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get current status of all circuit breakers.

    Returns circuit breaker state for each external service, useful for
    monitoring resilience patterns and troubleshooting connectivity issues.

    States:
    - closed: Normal operation, requests flowing
    - half_open: Testing recovery after failures
    - open: Blocking requests due to failures
    """
    from ..services.cloud.resilience import get_all_circuit_breaker_health

    try:
        breakers = get_all_circuit_breaker_health()
        return {
            "circuit_breakers": [
                {
                    "name": name,
                    "state": info.get("state", "unknown"),
                    "failure_count": info.get("failure_count", 0),
                    "success_count": info.get("success_count", 0),
                    "last_failure": info.get("last_failure_time"),
                    "retry_after": info.get("retry_after"),
                }
                for name, info in breakers.items()
            ],
            "total": len(breakers),
        }
    except Exception as exc:
        logger.warning("Failed to get circuit breaker status: %s", exc)
        return {"circuit_breakers": [], "total": 0, "error": str(exc)}


@router.get("/gpu-health")
async def get_gpu_health_status(
    _user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get GPU health status including thermal throttling warnings.

    Returns per-GPU status with temperature and throttling info,
    useful for displaying thermal warnings in the WebUI.
    """
    from simpletuner.helpers.training.gpu_circuit_breaker import get_gpu_circuit_breaker

    try:
        breaker = get_gpu_circuit_breaker()
        gpu_statuses = breaker.get_gpu_thermal_status()
        return {
            "gpus": gpu_statuses,
            "total": len(gpu_statuses),
        }
    except Exception as exc:
        logger.warning("Failed to get GPU health status: %s", exc)
        return {"gpus": [], "total": 0, "error": str(exc)}
