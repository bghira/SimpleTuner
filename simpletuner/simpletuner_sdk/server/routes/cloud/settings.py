"""Settings, hints, and data consent endpoints for cloud training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ._shared import get_job_store, get_local_upload_dir

logger = logging.getLogger(__name__)


# --- Provider Helpers ---


def get_default_provider() -> str:
    """Get the default cloud provider ID.

    Returns the first available (non-coming-soon) provider.
    Configurable via DEFAULT_CLOUD_PROVIDER env var.
    """
    import os

    # Allow override via environment
    env_provider = os.environ.get("SIMPLETUNER_DEFAULT_PROVIDER")
    if env_provider:
        return env_provider

    # Use first available provider from registry
    from ...services.cloud.provider_registry import get_available_provider_ids

    available = get_available_provider_ids()
    return available[0] if available else "replicate"


router = APIRouter()


# --- Pydantic Models ---


class HintsStatusResponse(BaseModel):
    """Response for UI hints status."""

    dataloader_hint_dismissed: bool = False
    git_hint_dismissed: bool = False


class LocalUploadConfigResponse(BaseModel):
    """Response with local upload configuration for publishing."""

    enabled: bool = False
    endpoint_url: Optional[str] = None
    bucket: str = "outputs"
    config_json: Optional[str] = None
    outputs_dir: Optional[str] = None


class DataConsentSettingResponse(BaseModel):
    """Response for data consent setting."""

    consent: str  # "ask", "allow", or "deny"


class DataConsentSettingUpdate(BaseModel):
    """Request to update data consent setting."""

    consent: str


class PollingStatusResponse(BaseModel):
    """Response for job polling status."""

    is_active: bool
    preference: Optional[bool] = None


class PollingSettingUpdate(BaseModel):
    """Request to update polling setting."""

    enabled: Optional[bool] = None


class DatasetUploadInfo(BaseModel):
    """Info about a dataset that would be uploaded."""

    id: str
    type: str
    instance_data_dir: Optional[str] = None
    file_count: Optional[int] = None
    total_size_mb: Optional[float] = None


class DataUploadPreviewResponse(BaseModel):
    """Response showing what data would be uploaded."""

    requires_upload: bool
    consent_mode: str
    datasets: List[DatasetUploadInfo] = []
    total_files: int = 0
    total_size_mb: float = 0.0
    message: Optional[str] = None


# --- Hints Endpoints ---


class HintsResponse(BaseModel):
    """Response for all hints (unified format for admin and cloud dashboard)."""

    dismissed_hints: List[str] = []


@router.get("/hints")
async def get_all_hints() -> HintsResponse:
    """Get all dismissed hints as a list (unified format for admin component)."""
    from ...services.webui_state import WebUIStateStore

    defaults = WebUIStateStore().load_defaults()
    dismissed = []

    # Cloud dashboard hints
    if defaults.cloud_dataloader_hint_dismissed:
        dismissed.append("dataloader")
    if defaults.cloud_git_hint_dismissed:
        dismissed.append("git")

    # Admin hints (stored as admin_*)
    admin_hints = getattr(defaults, "admin_dismissed_hints", []) or []
    for hint in admin_hints:
        dismissed.append(f"admin_{hint}")

    return HintsResponse(dismissed_hints=dismissed)


# --- Hint Management ---
# NOTE: This handles cloud dashboard and admin hints.
# Metrics hints are handled separately in metrics_config.py with their own storage.
# This separation is intentional - different feature areas use different hint lists.


@router.get("/hints/status", response_model=HintsStatusResponse)
async def get_hints_status() -> HintsStatusResponse:
    """Get status of dismissible UI hints."""
    from ...services.webui_state import WebUIStateStore

    defaults = WebUIStateStore().load_defaults()
    return HintsStatusResponse(
        dataloader_hint_dismissed=defaults.cloud_dataloader_hint_dismissed,
        git_hint_dismissed=defaults.cloud_git_hint_dismissed,
    )


@router.post("/hints/dismiss/{hint_name}")
async def dismiss_hint(hint_name: str) -> Dict[str, Any]:
    """Dismiss a UI hint permanently."""
    from ...services.webui_state import WebUIStateStore

    store = WebUIStateStore()
    defaults = store.load_defaults()

    # Handle cloud dashboard hints
    if hint_name == "dataloader":
        defaults.cloud_dataloader_hint_dismissed = True
    elif hint_name == "git":
        defaults.cloud_git_hint_dismissed = True
    # Handle admin hints (admin_overview, admin_users, etc.)
    elif hint_name.startswith("admin_"):
        admin_hints = getattr(defaults, "admin_dismissed_hints", None) or []
        admin_hint_name = hint_name[6:]  # Remove "admin_" prefix
        if admin_hint_name not in admin_hints:
            admin_hints.append(admin_hint_name)
        defaults.admin_dismissed_hints = admin_hints
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown hint: {hint_name}",
        )

    store.save_defaults(defaults)
    logger.info("Dismissed hint: %s", hint_name)

    return {"success": True, "hint": hint_name}


@router.post("/hints/show/{hint_name}")
async def show_hint(hint_name: str) -> Dict[str, Any]:
    """Show a previously dismissed UI hint."""
    from ...services.webui_state import WebUIStateStore

    store = WebUIStateStore()
    defaults = store.load_defaults()

    # Handle cloud dashboard hints
    if hint_name == "dataloader":
        defaults.cloud_dataloader_hint_dismissed = False
    elif hint_name == "git":
        defaults.cloud_git_hint_dismissed = False
    # Handle admin hints (admin_overview, admin_users, etc.)
    elif hint_name.startswith("admin_"):
        admin_hints = getattr(defaults, "admin_dismissed_hints", None) or []
        admin_hint_name = hint_name[6:]  # Remove "admin_" prefix
        if admin_hint_name in admin_hints:
            admin_hints.remove(admin_hint_name)
        defaults.admin_dismissed_hints = admin_hints
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown hint: {hint_name}",
        )

    store.save_defaults(defaults)
    logger.info("Showed hint: %s", hint_name)

    return {"success": True, "hint": hint_name}


# --- Local Upload Config ---


@router.get("/local-upload/config", response_model=LocalUploadConfigResponse)
async def get_local_upload_config() -> LocalUploadConfigResponse:
    """Get the local upload configuration for use in publishing_config."""
    store = get_job_store()
    provider_config = await store.get_provider_config(get_default_provider())
    webhook_url = provider_config.get("webhook_url")

    if not webhook_url:
        return LocalUploadConfigResponse(enabled=False)

    parsed = urlparse(webhook_url)
    s3_path = "/api/cloud/storage"
    s3_url = urlunparse((parsed.scheme, parsed.netloc, s3_path, "", "", ""))

    bucket = "outputs"
    upload_dir = get_local_upload_dir()

    config = {
        "provider": "s3",
        "bucket": bucket,
        "endpoint_url": s3_url,
        "access_key": "local",
        "secret_key": "local",
        "use_ssl": parsed.scheme == "https",
    }

    return LocalUploadConfigResponse(
        enabled=True,
        endpoint_url=s3_url,
        bucket=bucket,
        config_json=json.dumps([config]),
        outputs_dir=str(upload_dir / bucket),
    )


# --- Data Consent Endpoints ---


@router.get("/data-consent/setting", response_model=DataConsentSettingResponse)
async def get_data_consent_setting() -> DataConsentSettingResponse:
    """Get the current data consent setting."""
    from ...services.webui_state import WebUIStateStore

    defaults = WebUIStateStore().load_defaults()
    return DataConsentSettingResponse(consent=defaults.cloud_data_consent)


@router.put("/data-consent/setting")
async def set_data_consent_setting(request: DataConsentSettingUpdate) -> Dict[str, Any]:
    """Update the data consent setting."""
    from ...services.webui_state import WebUIStateStore

    valid_values = {"ask", "allow", "deny"}
    if request.consent not in valid_values:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid consent value: {request.consent}. Must be one of: {valid_values}",
        )

    store = WebUIStateStore()
    defaults = store.load_defaults()
    defaults.cloud_data_consent = request.consent
    store.save_defaults(defaults)

    logger.info("Updated cloud data consent setting to: %s", request.consent)
    return {"success": True, "consent": request.consent}


@router.post("/data-consent/preview", response_model=DataUploadPreviewResponse)
async def preview_data_upload(
    dataloader_config: List[Dict[str, Any]],
) -> DataUploadPreviewResponse:
    """Preview what data would be uploaded for a job submission."""
    from ...services.webui_state import WebUIStateStore

    defaults = WebUIStateStore().load_defaults()
    consent_mode = defaults.cloud_data_consent

    if consent_mode == "deny":
        return DataUploadPreviewResponse(
            requires_upload=False,
            consent_mode=consent_mode,
            message="Cloud uploads are disabled. Change 'Data Upload Consent' in settings to enable.",
        )

    datasets = []
    total_files = 0
    total_size_bytes = 0

    for ds_config in dataloader_config:
        ds_type = ds_config.get("type", "local")
        ds_id = ds_config.get("id", "unknown")

        if ds_type != "local":
            continue

        instance_dir = ds_config.get("instance_data_dir")
        if not instance_dir:
            continue

        instance_path = Path(instance_dir)
        if not instance_path.exists():
            datasets.append(
                DatasetUploadInfo(id=ds_id, type=ds_type, instance_data_dir=instance_dir, file_count=0, total_size_mb=0.0)
            )
            continue

        file_count = 0
        size_bytes = 0

        try:
            for item in instance_path.rglob("*"):
                if item.is_file():
                    file_count += 1
                    try:
                        size_bytes += item.stat().st_size
                    except OSError:
                        pass
        except OSError as e:
            logger.warning("Could not scan dataset %s: %s", ds_id, e)

        total_files += file_count
        total_size_bytes += size_bytes

        datasets.append(
            DatasetUploadInfo(
                id=ds_id,
                type=ds_type,
                instance_data_dir=instance_dir,
                file_count=file_count,
                total_size_mb=round(size_bytes / (1024 * 1024), 2),
            )
        )

    requires_upload = len(datasets) > 0
    total_size_mb = round(total_size_bytes / (1024 * 1024), 2)

    message = None
    if requires_upload:
        if consent_mode == "allow":
            message = f"{len(datasets)} dataset(s) will be uploaded automatically."
        elif consent_mode == "ask":
            message = f"{len(datasets)} dataset(s) will be uploaded. Please review and confirm."
    else:
        message = "No local data needs to be uploaded."

    return DataUploadPreviewResponse(
        requires_upload=requires_upload,
        consent_mode=consent_mode,
        datasets=datasets,
        total_files=total_files,
        total_size_mb=total_size_mb,
        message=message,
    )


# --- Polling Endpoints ---


@router.get("/polling/status", response_model=PollingStatusResponse)
async def get_polling_status() -> PollingStatusResponse:
    """Get status of job polling."""
    from ...services.cloud.background_tasks import get_task_manager
    from ...services.webui_state import WebUIStateStore

    defaults = WebUIStateStore().load_defaults()
    manager = get_task_manager()

    return PollingStatusResponse(
        is_active=manager.is_polling_active,
        preference=defaults.cloud_job_polling_enabled,
    )


@router.put("/polling/setting")
async def update_polling_setting(request: PollingSettingUpdate) -> Dict[str, Any]:
    """Update job polling setting."""
    from ...services.cloud.background_tasks import get_task_manager
    from ...services.webui_state import WebUIStateStore

    store = WebUIStateStore()
    defaults = store.load_defaults()
    defaults.cloud_job_polling_enabled = request.enabled
    store.save_defaults(defaults)

    # Refresh background tasks
    manager = get_task_manager()
    is_active = await manager.refresh_polling_state()

    logger.info("Updated polling setting: %s (active: %s)", request.enabled, is_active)
    return {"success": True, "enabled": request.enabled, "is_active": is_active}


# --- System Status Endpoints ---


class SystemStatusResponse(BaseModel):
    """Response for Replicate system status."""

    operational: bool = True
    ongoing_incidents: List[Dict[str, Any]] = []
    in_progress_maintenances: List[Dict[str, Any]] = []
    scheduled_maintenances: List[Dict[str, Any]] = []
    status_page_url: Optional[str] = None
    error: Optional[str] = None


@router.get("/system-status", response_model=SystemStatusResponse)
async def get_system_status() -> SystemStatusResponse:
    """Get Replicate system status from their status page."""
    from ...services.cloud.http_client import get_async_client

    try:
        async with get_async_client(timeout=5.0) as client:
            response = await client.get(
                "https://status.replicate.com/api/v2/summary.json",
            )
        if response.status_code == 200:
            data = response.json()
            status_data = data.get("status", {})
            components = data.get("components", [])
            incidents = data.get("incidents", [])
            maintenances = data.get("scheduled_maintenances", [])

            # Check if operational
            operational = status_data.get("indicator", "none") in ["none", "minor"]

            # Filter active incidents
            ongoing = [i for i in incidents if i.get("status") not in ["resolved", "postmortem"]]

            # Filter maintenances
            in_progress = [m for m in maintenances if m.get("status") == "in_progress"]
            scheduled = [m for m in maintenances if m.get("status") == "scheduled"]

            return SystemStatusResponse(
                operational=operational,
                ongoing_incidents=ongoing,
                in_progress_maintenances=in_progress,
                scheduled_maintenances=scheduled,
                status_page_url="https://status.replicate.com",
            )
        else:
            return SystemStatusResponse(error="Failed to fetch status")
    except Exception as e:
        logger.warning("Failed to fetch Replicate status: %s", e)
        return SystemStatusResponse(error=str(e))


# --- Publishing Status Endpoints ---


class PublishingStatusResponse(BaseModel):
    """Response for publishing/output configuration status."""

    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    s3_configured: bool = False
    s3_bucket: Optional[str] = None
    local_upload_configured: bool = False


@router.get("/publishing-status", response_model=PublishingStatusResponse)
async def get_publishing_status_compat() -> PublishingStatusResponse:
    """Get current publishing/output configuration status."""
    from ._shared import get_active_config

    try:
        config_resp = await get_active_config()

        push_to_hub = False
        hub_model_id = None
        s3_configured = False
        s3_bucket = None

        if config_resp:
            push_to_hub = config_resp.get("push_to_hub") or config_resp.get("--push_to_hub", False)
            hub_model_id = config_resp.get("hub_model_id") or config_resp.get("--hub_model_id")
            publishing_config = config_resp.get("publishing_config") or config_resp.get("--publishing_config")
            if publishing_config:
                try:
                    if isinstance(publishing_config, str):
                        configs = json.loads(publishing_config)
                    else:
                        configs = publishing_config
                    if isinstance(configs, list) and len(configs) > 0:
                        s3_configured = True
                        s3_bucket = configs[0].get("bucket")
                except (json.JSONDecodeError, TypeError):
                    pass

        # Check local upload config
        store = get_job_store()
        provider_config = await store.get_provider_config(get_default_provider())
        webhook_url = provider_config.get("webhook_url")
        local_upload_configured = bool(webhook_url)

        return PublishingStatusResponse(
            push_to_hub=bool(push_to_hub),
            hub_model_id=hub_model_id if push_to_hub else None,
            s3_configured=s3_configured,
            s3_bucket=s3_bucket,
            local_upload_configured=local_upload_configured,
        )
    except Exception as e:
        logger.warning("Failed to get publishing status: %s", e)
        return PublishingStatusResponse()


# --- Credential Security Settings ---


class CredentialSecuritySettings(BaseModel):
    """Credential security settings."""

    stale_threshold_days: int = 90
    early_warning_enabled: bool = False
    early_warning_percent: int = 75


class CredentialSecurityUpdate(BaseModel):
    """Update for credential security settings."""

    stale_threshold_days: Optional[int] = None
    early_warning_enabled: Optional[bool] = None
    early_warning_percent: Optional[int] = None


@router.get("/settings/credentials", response_model=CredentialSecuritySettings)
async def get_credential_settings() -> CredentialSecuritySettings:
    """Get credential security settings."""
    from ...services.webui_state import WebUIStateStore

    defaults = WebUIStateStore().load_defaults()
    return CredentialSecuritySettings(
        stale_threshold_days=getattr(defaults, "credential_rotation_threshold_days", 90),
        early_warning_enabled=getattr(defaults, "credential_early_warning_enabled", False),
        early_warning_percent=getattr(defaults, "credential_early_warning_percent", 75),
    )


@router.put("/settings/credentials", response_model=CredentialSecuritySettings)
async def update_credential_settings(update: CredentialSecurityUpdate) -> CredentialSecuritySettings:
    """Update credential security settings."""
    from ...services.webui_state import WebUIStateStore

    store = WebUIStateStore()
    defaults = store.load_defaults()

    if update.stale_threshold_days is not None:
        defaults.credential_rotation_threshold_days = update.stale_threshold_days
    if update.early_warning_enabled is not None:
        defaults.credential_early_warning_enabled = update.early_warning_enabled
    if update.early_warning_percent is not None:
        defaults.credential_early_warning_percent = update.early_warning_percent

    store.save_defaults(defaults)

    return CredentialSecuritySettings(
        stale_threshold_days=defaults.credential_rotation_threshold_days,
        early_warning_enabled=defaults.credential_early_warning_enabled,
        early_warning_percent=defaults.credential_early_warning_percent,
    )
