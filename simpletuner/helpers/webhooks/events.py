"""Helpers for building structured webhook events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from simpletuner.simpletuner_sdk.server.services.callback_events import (
    CallbackEvent,
    CheckpointData,
    EventSeverity,
    EventType,
    ProgressData,
    StageData,
    StageStatus,
    ValidationAsset,
    ValidationData,
)


def _progress_payload(
    *,
    label: str | None = None,
    current: float | None = None,
    total: float | None = None,
    percent: float | None = None,
    eta_seconds: float | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if label:
        payload["label"] = label
    if current is not None:
        payload["current"] = current
    if total is not None:
        payload["total"] = total
    if percent is not None:
        payload["percent"] = percent
    if eta_seconds is not None:
        payload["eta_seconds"] = eta_seconds
    if metrics:
        payload["metrics"] = dict(metrics)
    return payload


def lifecycle_stage_event(
    key: str,
    *,
    label: str | None = None,
    status: str = "running",
    current: float | None = None,
    total: float | None = None,
    percent: float | None = None,
    eta_seconds: float | None = None,
    metrics: Mapping[str, Any] | None = None,
    message: str | None = None,
    job_id: str | None = None,
    severity: str = "info",
    timestamp: datetime | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "lifecycle.stage",
        "severity": severity,
        "job_id": job_id,
        "stage": {
            "key": key,
            "label": label,
            "status": status,
        },
    }
    if message:
        event["message"] = message
        if not event.get("title"):
            event["title"] = message.splitlines()[0]
    if timestamp:
        event["timestamp"] = timestamp.isoformat()
    if extra:
        event["data"] = dict(extra)
    progress = _progress_payload(
        label=label,
        current=current,
        total=total,
        percent=percent,
        eta_seconds=eta_seconds,
        metrics=metrics,
    )
    if progress:
        event["stage"]["progress"] = progress
    return event


def training_status_event(
    status: str,
    *,
    message: str | None = None,
    job_id: str | None = None,
    severity: str | None = None,
    progress: Mapping[str, Any] | None = None,
    timestamp: datetime | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "training.status",
        "job_id": job_id,
        "title": status.replace("_", " ").title(),
        "data": {"status": status},
    }
    if message:
        event["message"] = message
    if severity:
        event["severity"] = severity
    if timestamp:
        event["timestamp"] = timestamp.isoformat()
    if progress:
        event["progress"] = dict(progress)
    if extra:
        data = event.setdefault("data", {})
        data.update(dict(extra))
    return event


def notification_event(
    message: str,
    *,
    title: str | None = None,
    severity: str = "info",
    job_id: str | None = None,
    timestamp: datetime | None = None,
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "notification",
        "severity": severity,
        "job_id": job_id,
        "message": message,
        "title": title or message.splitlines()[0],
    }
    if timestamp:
        event["timestamp"] = timestamp.isoformat()
    if data:
        event["data"] = dict(data)
    return event


def error_event(
    message: str,
    *,
    job_id: str | None = None,
    title: str | None = None,
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return notification_event(
        message,
        title=title or "Error",
        severity="error",
        job_id=job_id,
        data=data,
    )


def checkpoint_event(
    *,
    path: str | None,
    label: str | None = None,
    job_id: str | None = None,
    is_final: bool = False,
    severity: str = "info",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "training.checkpoint",
        "severity": severity,
        "job_id": job_id,
        "checkpoint": {
            "path": path,
            "label": label,
            "is_final": is_final,
        },
    }
    if label:
        payload["title"] = label
    return payload


def attach_timestamp(event: dict[str, Any]) -> dict[str, Any]:
    """Ensure *event* has an ISO timestamp."""
    if "timestamp" not in event:
        event["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
    return event


__all__ = [
    "CallbackEvent",
    "EventSeverity",
    "EventType",
    "ProgressData",
    "StageData",
    "StageStatus",
    "ValidationData",
    "ValidationAsset",
    "CheckpointData",
    "lifecycle_stage_event",
    "training_status_event",
    "notification_event",
    "error_event",
    "checkpoint_event",
    "attach_timestamp",
]
