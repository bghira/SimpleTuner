"""Shared utilities, models, and helpers for cloud routes.

This module re-exports from specialized submodules for backwards compatibility.
New code should import directly from the specific modules:
- dependencies: get_job_store, get_async_job_store
- helpers: emit_cloud_event, get_client_ip, check_ip_allowlist, etc.
- schemas: All Pydantic request/response models

Note: Rate limiting is handled by RateLimitMiddleware in security_middleware.py.
The rate_limiting module is deprecated.
"""

from __future__ import annotations

# Re-export from services for backwards compatibility
from ...services.cloud import AsyncJobStore, CloudJobStatus, JobType, UnifiedJob
from ...services.cloud.protocols import JobStore

# Re-export dependencies
from .dependencies import get_async_job_store, get_job_store

# Re-export helpers
from .helpers import (
    check_ip_allowlist,
    emit_cloud_event,
    enrich_jobs_with_queue_info,
    get_active_config,
    get_client_ip,
    get_hf_token,
    get_local_upload_dir,
    get_period_days,
    validate_webhook_url,
)

# Re-export all schemas
from .schemas import (
    AvailableConfigsResponse,
    BillingResponse,
    CostEstimateResponse,
    CostLimitStatusResponse,
    DataConsentSettingResponse,
    DataUploadPreviewResponse,
    ExternalWebhookTestResponse,
    HardwareOption,
    HardwareOptionsResponse,
    HealthCheckComponent,
    HealthCheckResponse,
    HintsStatusResponse,
    JobListResponse,
    JobResponse,
    LocalUploadConfigResponse,
    MetricsResponse,
    ModelVersionInfo,
    ModelVersionsResponse,
    PreSubmitCheckResponse,
    ProviderConfigResponse,
    ProviderConfigUpdate,
    ProvidersListResponse,
    PublishingStatusResponse,
    ReplicateWebhookPayload,
    SubmitJobRequest,
    SubmitJobResponse,
    SystemStatusResponse,
    ValidateResponse,
    WebhookTestResponse,
)

__all__ = [
    # Services types
    "AsyncJobStore",
    "CloudJobStatus",
    "JobStore",
    "JobType",
    "UnifiedJob",
    # Dependencies
    "get_async_job_store",
    "get_job_store",
    # Helpers
    "check_ip_allowlist",
    "emit_cloud_event",
    "enrich_jobs_with_queue_info",
    "get_active_config",
    "get_client_ip",
    "get_hf_token",
    "get_local_upload_dir",
    "get_period_days",
    "validate_webhook_url",
    # Schemas
    "AvailableConfigsResponse",
    "BillingResponse",
    "CostEstimateResponse",
    "CostLimitStatusResponse",
    "DataConsentSettingResponse",
    "DataUploadPreviewResponse",
    "ExternalWebhookTestResponse",
    "HardwareOption",
    "HardwareOptionsResponse",
    "HealthCheckComponent",
    "HealthCheckResponse",
    "HintsStatusResponse",
    "JobListResponse",
    "JobResponse",
    "LocalUploadConfigResponse",
    "MetricsResponse",
    "ModelVersionInfo",
    "ModelVersionsResponse",
    "PreSubmitCheckResponse",
    "ProviderConfigResponse",
    "ProviderConfigUpdate",
    "ProvidersListResponse",
    "PublishingStatusResponse",
    "ReplicateWebhookPayload",
    "SubmitJobRequest",
    "SubmitJobResponse",
    "SystemStatusResponse",
    "ValidateResponse",
    "WebhookTestResponse",
]
