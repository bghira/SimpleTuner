"""Helper functions for cloud routes.

Provides utility functions for request handling, event emission, and configuration.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)


def get_client_ip(request: Request) -> str:
    """Get the client IP address from request, handling proxies."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_ip_allowlist(client_ip: str, allowed_ips: List[str]) -> bool:
    """Check if client IP is in the allowlist."""
    import ipaddress

    try:
        client_addr = ipaddress.ip_address(client_ip)
    except ValueError:
        return False

    for allowed in allowed_ips:
        try:
            if "/" in allowed:
                if client_addr in ipaddress.ip_network(allowed, strict=False):
                    return True
            else:
                if client_addr == ipaddress.ip_address(allowed):
                    return True
        except ValueError:
            continue

    return False


def emit_cloud_event(
    event_type: str,
    job_id: str,
    message: str,
    severity: str = "info",
    **extra: Any,
) -> None:
    """Emit a cloud job event to the event log dock and notification system."""
    try:
        from ...services.callback_service import get_default_callback_service

        service = get_default_callback_service()
        payload = {
            "type": "notification",
            "message_type": event_type,
            "message": message,
            "severity": severity,
            "job_id": job_id,
            "source": "cloud",
            **extra,
        }
        service.handle_incoming(payload)
    except Exception as exc:
        logger.debug("Could not emit cloud event: %s", exc)

    # Dispatch to notification system (non-blocking)
    _dispatch_notification(event_type, job_id, message, severity, extra)


def _dispatch_notification(
    event_type: str,
    job_id: str,
    message: str,
    severity: str,
    extra: dict,
) -> None:
    """Dispatch event to notification system in background."""
    try:
        from ...services.cloud.notification import NotificationEventType, get_notifier

        # Map cloud event types to notification event types
        event_mapping = {
            "job_submitted": NotificationEventType.JOB_SUBMITTED,
            "job_started": NotificationEventType.JOB_STARTED,
            "job_completed": NotificationEventType.JOB_COMPLETED,
            "job_failed": NotificationEventType.JOB_FAILED,
            "job_cancelled": NotificationEventType.JOB_CANCELLED,
            "approval_required": NotificationEventType.APPROVAL_REQUIRED,
            "approval_granted": NotificationEventType.APPROVAL_GRANTED,
            "approval_rejected": NotificationEventType.APPROVAL_REJECTED,
            "approval_expired": NotificationEventType.APPROVAL_EXPIRED,
            "quota_warning": NotificationEventType.QUOTA_WARNING,
            "quota_exceeded": NotificationEventType.QUOTA_EXCEEDED,
            "cost_limit_warning": NotificationEventType.COST_LIMIT_WARNING,
            "cost_limit_exceeded": NotificationEventType.COST_LIMIT_EXCEEDED,
            "provider_error": NotificationEventType.PROVIDER_ERROR,
            "provider_degraded": NotificationEventType.PROVIDER_DEGRADED,
            "connection_restored": NotificationEventType.CONNECTION_RESTORED,
            "webhook_failure": NotificationEventType.WEBHOOK_FAILURE,
        }

        notification_type = event_mapping.get(event_type)
        if notification_type is None:
            return

        # Build notification payload with severity
        payload = {
            "job_id": job_id,
            "message": message,
            "severity": severity,
            **extra,
        }

        # Get or create event loop and dispatch
        try:
            loop = asyncio.get_running_loop()
            # Already in async context, create task
            loop.create_task(get_notifier().notify(notification_type, payload))
        except RuntimeError:
            # No running event loop - we're in a sync context
            # Notifications are best-effort; log and continue
            logger.debug(
                "Skipping notification dispatch (no event loop): type=%s job=%s",
                notification_type,
                job_id,
            )
    except ImportError:
        # Notification module not available
        pass
    except Exception as exc:
        logger.debug("Could not dispatch notification: %s", exc)


def validate_webhook_url(url: Optional[str]) -> Optional[str]:
    """Validate and normalize a webhook URL."""
    if not url:
        return None

    url = url.strip()
    if not url:
        return None

    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL format")
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must use http or https scheme")
        return url
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid webhook URL: {exc}",
        )


async def get_active_config() -> Optional[Dict[str, Any]]:
    """Helper to fetch the active config."""
    try:
        from ...services.config_store import ConfigStore
        from ...services.webui_state import WebUIStateStore

        defaults = WebUIStateStore().load_defaults()
        if not defaults.configs_dir:
            logger.debug("No configs_dir set in defaults")
            return None
        if not defaults.active_config:
            logger.debug("No active_config set in defaults")
            return None

        store = ConfigStore(config_dir=Path(defaults.configs_dir).expanduser(), config_type="model")
        config, _ = store.load_config(defaults.active_config)
        return config
    except Exception as exc:
        logger.warning("Failed to load active config: %s", exc)
        return None


def get_local_upload_dir() -> Path:
    """Get the directory for local cloud uploads."""
    from ...services.webui_state import WebUIStateStore

    defaults = WebUIStateStore().load_defaults()

    if defaults.cloud_outputs_dir:
        base = Path(defaults.cloud_outputs_dir).expanduser()
    else:
        base = Path.home() / ".simpletuner" / "cloud_outputs"

    base.mkdir(parents=True, exist_ok=True)
    return base


def get_hf_token() -> Optional[str]:
    """Read HF token from local cache."""
    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
        if token:
            return token
    except ImportError:
        pass

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def get_period_days(period: str) -> int:
    """Convert period string to number of days."""
    period_map = {
        "daily": 1,
        "weekly": 7,
        "monthly": 30,
    }
    return period_map.get(period, 30)


async def enrich_jobs_with_queue_info(
    jobs: list,
    pending_statuses: tuple = ("pending", "queued"),
) -> list:
    """Enrich job dicts with queue position and ETA.

    Args:
        jobs: List of UnifiedJob objects or job dicts with job_id and status
        pending_statuses: Status values that indicate a job is in the queue

    Returns:
        List of job dicts enriched with queue_position and estimated_wait_seconds
    """
    from ...services.cloud import CloudJobStatus

    # Convert to dicts if needed and collect pending job IDs
    job_dicts = []
    pending_job_ids = []

    for j in jobs:
        jd = j.to_dict() if hasattr(j, "to_dict") else dict(j)
        job_dicts.append(jd)

        status = jd.get("status", "")
        if status in (CloudJobStatus.PENDING.value, CloudJobStatus.QUEUED.value):
            pending_job_ids.append(jd.get("job_id"))

    # Batch fetch queue info for pending jobs
    if pending_job_ids:
        try:
            from ...services.cloud.queue import QueueStore

            queue_store = QueueStore()
            queue_info = await queue_store.get_positions_with_eta_batch(pending_job_ids)

            for jd in job_dicts:
                job_id = jd.get("job_id")
                if job_id in queue_info:
                    info = queue_info[job_id]
                    jd["queue_position"] = info["position"]
                    jd["estimated_wait_seconds"] = info["estimated_wait_seconds"]
        except Exception:
            pass  # Queue info is optional enrichment

    return job_dicts
