"""Metrics configuration endpoints for Prometheus export."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ...services.webui_state import WebUIStateStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics-config"])


# --- Category Definitions ---

METRIC_CATEGORIES = {
    "jobs": {
        "name": "Jobs & Costs",
        "description": "Job counts, status distribution, queue depth, costs, and duration",
        "metrics": [
            "simpletuner_jobs_total",
            "simpletuner_jobs_by_status",
            "simpletuner_jobs_active",
            "simpletuner_job_duration_seconds_avg",
            "simpletuner_cost_usd_total",
        ],
    },
    "http": {
        "name": "HTTP Requests",
        "description": "Request counts, errors, and latency by endpoint",
        "metrics": [
            "simpletuner_http_requests_total",
            "simpletuner_http_errors_total",
            "simpletuner_http_request_latency_ms_avg",
        ],
    },
    "rate_limits": {
        "name": "Rate Limiting",
        "description": "Rate limit violations and tracked clients",
        "metrics": [
            "simpletuner_rate_limit_violations_total",
            "simpletuner_rate_limit_tracked_clients",
        ],
    },
    "approvals": {
        "name": "Approvals",
        "description": "Pending, approved, rejected, and expired approval requests",
        "metrics": [
            "simpletuner_approval_requests_pending",
            "simpletuner_approval_requests_by_status",
        ],
    },
    "audit": {
        "name": "Audit Log",
        "description": "Audit log entries by event type and 24-hour activity",
        "metrics": [
            "simpletuner_audit_log_entries_total",
            "simpletuner_audit_log_entries_by_type",
            "simpletuner_audit_log_entries_24h",
        ],
    },
    "health": {
        "name": "Health & Uptime",
        "description": "Server uptime, database latency, component health",
        "metrics": [
            "simpletuner_uptime_seconds",
            "simpletuner_health_database_latency_ms",
            "simpletuner_health_component_status",
        ],
    },
    "circuit_breakers": {
        "name": "Circuit Breakers",
        "description": "Circuit breaker state, failures, and recovery for providers",
        "metrics": [
            "simpletuner_circuit_breaker_state",
            "simpletuner_circuit_breaker_failures_total",
            "simpletuner_circuit_breaker_successes_total",
        ],
    },
    "provider": {
        "name": "Provider Status",
        "description": "Cost limits, credit balance, and estimated jobs remaining",
        "metrics": [
            "simpletuner_cost_limit_percent_used",
            "simpletuner_credit_balance_usd",
            "simpletuner_estimated_jobs_remaining",
        ],
    },
}


# --- Templates ---

METRIC_TEMPLATES = {
    "minimal": {
        "name": "Minimal",
        "description": "Just job counts - lightweight monitoring",
        "categories": ["jobs"],
    },
    "standard": {
        "name": "Standard (Recommended)",
        "description": "Jobs, HTTP metrics, and health checks",
        "categories": ["jobs", "http", "health"],
    },
    "security": {
        "name": "Security",
        "description": "Include rate limiting, audit, and approval tracking",
        "categories": ["jobs", "http", "rate_limits", "audit", "approvals"],
    },
    "full": {
        "name": "Full",
        "description": "All available metrics",
        "categories": list(METRIC_CATEGORIES.keys()),
    },
}


# --- Request/Response Models ---


class MetricsConfigResponse(BaseModel):
    """Current metrics configuration."""

    prometheus_enabled: bool
    prometheus_categories: List[str]
    tensorboard_enabled: bool
    endpoint_url: str = "/api/metrics/prometheus"


class MetricsConfigUpdate(BaseModel):
    """Update metrics configuration."""

    prometheus_enabled: Optional[bool] = None
    prometheus_categories: Optional[List[str]] = None


class CategoryInfo(BaseModel):
    """Information about a metric category."""

    id: str
    name: str
    description: str
    metrics: List[str]


class CategoriesResponse(BaseModel):
    """List of available metric categories."""

    categories: List[CategoryInfo]


class TemplateInfo(BaseModel):
    """Information about a preset template."""

    id: str
    name: str
    description: str
    categories: List[str]


class TemplatesResponse(BaseModel):
    """List of available preset templates."""

    templates: List[TemplateInfo]


class PreviewResponse(BaseModel):
    """Preview of Prometheus export output."""

    content: str
    metric_count: int
    categories_used: List[str]


# --- Endpoints ---


@router.get("", response_model=MetricsConfigResponse)
async def get_metrics_config() -> MetricsConfigResponse:
    """Get current metrics export configuration."""
    store = WebUIStateStore()
    defaults = store.load_defaults()

    return MetricsConfigResponse(
        prometheus_enabled=defaults.metrics_prometheus_enabled,
        prometheus_categories=defaults.metrics_prometheus_categories,
        tensorboard_enabled=defaults.metrics_tensorboard_enabled,
    )


@router.put("", response_model=MetricsConfigResponse)
async def update_metrics_config(update: MetricsConfigUpdate) -> MetricsConfigResponse:
    """Update metrics export configuration."""
    store = WebUIStateStore()
    defaults = store.load_defaults()

    if update.prometheus_enabled is not None:
        defaults.metrics_prometheus_enabled = update.prometheus_enabled

    if update.prometheus_categories is not None:
        # Validate categories
        valid = set(METRIC_CATEGORIES.keys())
        categories = [c for c in update.prometheus_categories if c in valid]
        defaults.metrics_prometheus_categories = categories

    store.save_defaults(defaults)

    return MetricsConfigResponse(
        prometheus_enabled=defaults.metrics_prometheus_enabled,
        prometheus_categories=defaults.metrics_prometheus_categories,
        tensorboard_enabled=defaults.metrics_tensorboard_enabled,
    )


@router.get("/categories", response_model=CategoriesResponse)
async def list_categories() -> CategoriesResponse:
    """List available metric categories with descriptions."""
    categories = [
        CategoryInfo(
            id=cat_id,
            name=cat_info["name"],
            description=cat_info["description"],
            metrics=cat_info["metrics"],
        )
        for cat_id, cat_info in METRIC_CATEGORIES.items()
    ]
    return CategoriesResponse(categories=categories)


@router.get("/templates", response_model=TemplatesResponse)
async def list_templates() -> TemplatesResponse:
    """List available preset templates."""
    templates = [
        TemplateInfo(
            id=tmpl_id,
            name=tmpl_info["name"],
            description=tmpl_info["description"],
            categories=tmpl_info["categories"],
        )
        for tmpl_id, tmpl_info in METRIC_TEMPLATES.items()
    ]
    return TemplatesResponse(templates=templates)


@router.post("/apply-template/{template_id}", response_model=MetricsConfigResponse)
async def apply_template(template_id: str) -> MetricsConfigResponse:
    """Apply a preset template to metrics configuration."""
    if template_id not in METRIC_TEMPLATES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template not found: {template_id}",
        )

    template = METRIC_TEMPLATES[template_id]
    store = WebUIStateStore()
    defaults = store.load_defaults()

    defaults.metrics_prometheus_enabled = True
    defaults.metrics_prometheus_categories = template["categories"].copy()

    store.save_defaults(defaults)

    return MetricsConfigResponse(
        prometheus_enabled=defaults.metrics_prometheus_enabled,
        prometheus_categories=defaults.metrics_prometheus_categories,
        tensorboard_enabled=defaults.metrics_tensorboard_enabled,
    )


@router.post("/preview", response_model=PreviewResponse)
async def preview_export(categories: Optional[List[str]] = None) -> PreviewResponse:
    """Preview Prometheus export output.

    If categories is not provided, uses the current configuration.
    """
    from ...services.cloud.prometheus_metrics import get_metrics_collector
    from ..cloud.metrics import get_job_store

    store = WebUIStateStore()
    defaults = store.load_defaults()

    # Use provided categories or fall back to configured
    cats = categories if categories is not None else defaults.metrics_prometheus_categories

    # Validate categories
    valid = set(METRIC_CATEGORIES.keys())
    cats = [c for c in cats if c in valid]

    # Get metrics collector and export
    collector = get_metrics_collector()
    content = await collector.export_prometheus(categories=cats)

    # Count metrics (lines starting with simpletuner_ that aren't HELP or TYPE)
    metric_count = sum(1 for line in content.split("\n") if line.startswith("simpletuner_") and not line.startswith("# "))

    return PreviewResponse(
        content=content,
        metric_count=metric_count,
        categories_used=cats,
    )


# --- Metrics Hint Management ---
# NOTE: These endpoints handle metrics-specific hints (hero CTAs, etc.).
# Cloud/admin hints are handled in settings.py with separate storage.
# This separation keeps each feature area's hints independent.


@router.post("/dismiss-hint/{hint_name}")
async def dismiss_hint(hint_name: str) -> Dict[str, Any]:
    """Dismiss a metrics tab hint (for hero CTA)."""
    store = WebUIStateStore()
    defaults = store.load_defaults()

    if hint_name not in defaults.metrics_dismissed_hints:
        defaults.metrics_dismissed_hints.append(hint_name)
        store.save_defaults(defaults)

    return {"success": True, "hint": hint_name}


@router.post("/show-hint/{hint_name}")
async def show_hint(hint_name: str) -> Dict[str, Any]:
    """Re-show a metrics tab hint."""
    store = WebUIStateStore()
    defaults = store.load_defaults()

    if hint_name in defaults.metrics_dismissed_hints:
        defaults.metrics_dismissed_hints.remove(hint_name)
        store.save_defaults(defaults)

    return {"success": True, "hint": hint_name}
