"""Pydantic schemas for cloud routes.

All request/response models for the cloud API endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class JobListResponse(BaseModel):
    """Response for job listing."""

    jobs: List[Dict[str, Any]]
    total: int


class JobResponse(BaseModel):
    """Response for a single job."""

    job: Dict[str, Any]


class AvailableConfigsResponse(BaseModel):
    """Response for available configs list."""

    configs: List[Dict[str, Any]] = []
    active_config: Optional[str] = None


class ProviderConfigResponse(BaseModel):
    """Response for provider configuration."""

    provider: str
    config: Dict[str, Any]


class ProviderConfigUpdate(BaseModel):
    """Request to update provider configuration."""

    version_override: Optional[str] = None
    webhook_url: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    cost_limit_enabled: Optional[bool] = None
    cost_limit_amount: Optional[float] = None
    cost_limit_period: Optional[str] = None
    cost_limit_action: Optional[str] = None
    hardware_info: Optional[Dict[str, Dict[str, Any]]] = None
    webhook_require_signature: Optional[bool] = None
    webhook_allowed_ips: Optional[List[str]] = None
    ssl_verify: Optional[bool] = None
    ssl_ca_bundle: Optional[str] = None
    proxy_url: Optional[str] = None
    http_timeout: Optional[float] = None
    # Rate limiting configuration
    webhook_rate_limit_max: Optional[int] = Field(None, ge=1, le=10000)
    webhook_rate_limit_window: Optional[int] = Field(None, ge=1, le=3600)
    s3_rate_limit_max: Optional[int] = Field(None, ge=1, le=10000)
    s3_rate_limit_window: Optional[int] = Field(None, ge=1, le=3600)
    # SimpleTuner.io configuration
    org_id: Optional[str] = None
    api_base_url: Optional[str] = None
    max_runtime_minutes: Optional[int] = Field(None, ge=1)


class ValidateResponse(BaseModel):
    """Response for API key validation."""

    valid: bool
    user_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ProvidersListResponse(BaseModel):
    """Response for listing available providers."""

    providers: List[Dict[str, Any]]


class SubmitJobRequest(BaseModel):
    """Request to submit a cloud training job."""

    config: Optional[Dict[str, Any]] = None
    dataloader_config: Optional[List[Dict[str, Any]]] = None
    config_name_to_load: Optional[str] = None
    config_name: Optional[str] = None
    webhook_url: Optional[str] = None
    snapshot_name: Optional[str] = None
    snapshot_message: Optional[str] = None
    tracker_run_name: Optional[str] = None
    upload_id: Optional[str] = None
    idempotency_key: Optional[str] = Field(
        None,
        description="Client-generated unique key for deduplication. "
        "If a job with this key was submitted in the last 24h, returns the existing job.",
    )


class PreSubmitCheckResponse(BaseModel):
    """Response for pre-submit check endpoint."""

    git_available: bool = False
    repo_present: bool = False
    is_dirty: bool = False
    dirty_count: int = 0
    dirty_paths: List[str] = []
    current_commit: Optional[str] = None
    current_abbrev: Optional[str] = None
    current_branch: Optional[str] = None
    tracker_run_name: Optional[str] = None
    config_name: Optional[str] = None


class PublishingStatusResponse(BaseModel):
    """Response for publishing status check."""

    hf_configured: bool = False
    hf_token_valid: bool = False
    hf_username: Optional[str] = None
    hub_model_id: Optional[str] = None
    push_to_hub: bool = False
    s3_configured: bool = False
    local_upload_available: bool = False
    local_upload_dir: Optional[str] = None
    message: Optional[str] = None


class SubmitJobResponse(BaseModel):
    """Response for job submission."""

    success: bool
    job_id: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None
    data_uploaded: bool = False
    upload_id: Optional[str] = None
    cost_limit_warning: Optional[str] = None
    quota_warnings: List[str] = Field(default_factory=list)
    idempotent_hit: bool = Field(
        default=False, description="True if this response is for an existing job matched by idempotency_key"
    )


class MetricsResponse(BaseModel):
    """Response for dashboard metrics."""

    credit_balance: Optional[float] = None
    estimated_jobs_remaining: Optional[int] = None
    total_cost: float = 0.0
    job_count: int = 0
    avg_job_duration_seconds: Optional[float] = None
    jobs_by_status: Dict[str, int] = {}
    cost_by_day: List[Dict[str, Any]] = []
    period_days: int = 30


class BillingResponse(BaseModel):
    """Response for billing refresh."""

    credit_balance: Optional[float] = None
    estimated_jobs_remaining: Optional[int] = None
    error: Optional[str] = None


class CostLimitStatusResponse(BaseModel):
    """Response for cost limit status check."""

    enabled: bool = False
    limit_amount: Optional[float] = None
    period: Optional[str] = None
    action: Optional[str] = None
    current_spend: float = 0.0
    percent_used: float = 0.0
    is_exceeded: bool = False
    is_warning: bool = False
    message: Optional[str] = None


class SystemStatusResponse(BaseModel):
    """Response for system status check."""

    operational: bool = True
    ongoing_incidents: List[Dict[str, Any]] = []
    in_progress_maintenances: List[Dict[str, Any]] = []
    scheduled_maintenances: List[Dict[str, Any]] = []
    status_page_url: Optional[str] = None
    error: Optional[str] = None


class HealthCheckComponent(BaseModel):
    """Health status of a single component."""

    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    latency_ms: Optional[float] = None
    message: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint."""

    status: str  # "healthy", "degraded", "unhealthy"
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None
    components: List[HealthCheckComponent] = []
    timestamp: str


class WebhookTestResponse(BaseModel):
    """Response for webhook connection test."""

    success: bool
    message: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class ExternalWebhookTestResponse(BaseModel):
    """Response for external webhook reachability test."""

    success: bool
    message: str
    latency_ms: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    prediction_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ModelVersionInfo(BaseModel):
    """Info about a model version."""

    id: str
    created_at: Optional[str] = None
    model_id: str
    full_version: str


class ModelVersionsResponse(BaseModel):
    """Response for model versions list."""

    versions: List[ModelVersionInfo] = []
    current_version: Optional[str] = None
    error: Optional[str] = None


class HardwareOption(BaseModel):
    """Hardware option for cloud training."""

    id: str
    name: str
    cost_per_second: float
    cost_per_hour: float
    available: bool = True


class HardwareOptionsResponse(BaseModel):
    """Response for available hardware options."""

    hardware: List[HardwareOption] = []
    default_hardware: str = "gpu-l40s"
    message: Optional[str] = None


class CostEstimateResponse(BaseModel):
    """Response for cost estimate."""

    has_estimate: bool = False
    estimated_cost_usd: Optional[float] = None
    avg_duration_seconds: Optional[float] = None
    hardware_cost_per_hour: Optional[float] = None
    data_size_mb: float = 0.0
    message: Optional[str] = None


class HintsStatusResponse(BaseModel):
    """Response for hints status."""

    dataloader_dismissed: bool = False
    git_dismissed: bool = False


class LocalUploadConfigResponse(BaseModel):
    """Response for local upload configuration."""

    enabled: bool = False
    webhook_url: Optional[str] = None
    upload_dir: Optional[str] = None
    s3_endpoint_url: Optional[str] = None
    message: Optional[str] = None


class DataConsentSettingResponse(BaseModel):
    """Response for data consent setting."""

    consent_given: bool = False


class DataUploadPreviewResponse(BaseModel):
    """Response for data upload preview."""

    files: List[Dict[str, Any]] = []
    total_size_bytes: int = 0
    total_size_mb: float = 0.0
    file_count: int = 0
    message: Optional[str] = None


# Webhook payload validation model
class ReplicateWebhookPayload(BaseModel):
    """Validated payload from Replicate webhook."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    event_type: Optional[str] = Field(None, alias="type")
    job_id: Optional[str] = Field(None, alias="prediction_id")
    status: Optional[str] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None
    logs: Optional[str] = None
