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
    label: str | None = None,
    current: float | None = None,
    total: float | None = None,
    percent: float | None = None,
    eta_seconds: float | None = None,
    metrics: Mapping[str, Any] | None = None,
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
    progress_payload = progress or _progress_payload(
        label=label,
        current=current,
        total=total,
        percent=percent,
        eta_seconds=eta_seconds,
        metrics=metrics,
    )
    if progress_payload:
        event["progress"] = dict(progress_payload)
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


def gpu_fault_event(
    fault_type: str,
    *,
    message: str,
    gpu_index: int | None = None,
    gpu_name: str | None = None,
    job_id: str | None = None,
    severity: str = "critical",
    temperature_celsius: float | None = None,
    ecc_errors_single: int | None = None,
    ecc_errors_double: int | None = None,
    throttle_reasons: list[str] | None = None,
    memory_used_percent: float | None = None,
    action_taken: str | None = None,
    exception_type: str | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build a GPU fault event for webhook emission.

    Args:
        fault_type: Type of fault (e.g., "cuda_error", "ecc_error", "thermal",
                    "throttling", "circuit_open", "health_warning")
        message: Human-readable description of the fault
        gpu_index: GPU device index (0-based)
        gpu_name: GPU device name (e.g., "NVIDIA RTX 5090")
        job_id: Training job identifier
        severity: Event severity ("warning", "error", "critical")
        temperature_celsius: GPU temperature at time of fault
        ecc_errors_single: Count of correctable single-bit ECC errors
        ecc_errors_double: Count of uncorrectable double-bit ECC errors
        throttle_reasons: List of active throttle reasons
        memory_used_percent: GPU memory utilization percentage
        action_taken: Action taken in response (e.g., "circuit_opened", "training_terminated")
        exception_type: Python exception class name if triggered by exception
        timestamp: Event timestamp (defaults to now)

    Returns:
        Structured event dict for webhook emission
    """
    gpu_info: dict[str, Any] = {}
    if gpu_index is not None:
        gpu_info["index"] = gpu_index
    if gpu_name:
        gpu_info["name"] = gpu_name
    if temperature_celsius is not None:
        gpu_info["temperature_celsius"] = temperature_celsius
    if ecc_errors_single is not None:
        gpu_info["ecc_errors_single"] = ecc_errors_single
    if ecc_errors_double is not None:
        gpu_info["ecc_errors_double"] = ecc_errors_double
    if throttle_reasons:
        gpu_info["throttle_reasons"] = list(throttle_reasons)
    if memory_used_percent is not None:
        gpu_info["memory_used_percent"] = round(memory_used_percent, 1)

    event: dict[str, Any] = {
        "type": "gpu.fault",
        "severity": severity,
        "job_id": job_id,
        "title": f"GPU Fault: {fault_type}",
        "message": message,
        "fault": {
            "type": fault_type,
        },
    }

    if gpu_info:
        event["fault"]["gpu"] = gpu_info
    if action_taken:
        event["fault"]["action_taken"] = action_taken
    if exception_type:
        event["fault"]["exception_type"] = exception_type
    if timestamp:
        event["timestamp"] = timestamp.isoformat()

    return event


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
    "gpu_fault_event",
    "attach_timestamp",
]
