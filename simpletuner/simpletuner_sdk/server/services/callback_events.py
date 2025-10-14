"""Unified webhook event models and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


class EventType(str, Enum):
    """Logical event groupings understood by the WebUI."""

    LIFECYCLE_STAGE = "lifecycle.stage"
    TRAINING_PROGRESS = "training.progress"
    TRAINING_STATUS = "training.status"
    TRAINING_SUMMARY = "training.summary"
    NOTIFICATION = "notification"
    CHECKPOINT = "training.checkpoint"
    VALIDATION = "training.validation"
    METRIC = "training.metric"
    ERROR = "error"
    DEBUG = "debug"
    DEBUG_IMAGE = "debug.image"
    VALIDATION_IMAGE = "validation.image"


class EventSeverity(str, Enum):
    """Shared severity scale for surfaced events."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class StageStatus(str, Enum):
    """Lifecycle stage states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ProgressData:
    """Normalized progress details for rendering progress bars and metrics."""

    label: str | None = None
    current: float | None = None
    total: float | None = None
    percent: float | None = None
    eta_seconds: float | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def normalized_percent(self) -> float | None:
        percent = self.percent
        if percent is None and self.current is not None and self.total:
            try:
                percent = (float(self.current) / float(self.total)) * 100
            except ZeroDivisionError:
                percent = None
        if percent is None:
            return None
        numeric = float(percent)
        if numeric != numeric:  # NaN guard
            return None
        return max(0.0, min(100.0, numeric))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "current": self.current,
            "total": self.total,
            "percent": self.normalized_percent,
            "eta_seconds": self.eta_seconds,
        }
        if self.metrics:
            payload["metrics"] = dict(self.metrics)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> ProgressData | None:
        if not isinstance(payload, Mapping):
            return None

        label = _safe_str(
            payload.get("label") or payload.get("headline") or payload.get("title") or payload.get("readable_type")
        )
        current = _maybe_number(
            payload.get("current")
            or payload.get("step")
            or payload.get("current_step")
            or payload.get("current_estimated_index")
        )
        total = _maybe_number(
            payload.get("total")
            or payload.get("max_steps")
            or payload.get("total_steps")
            or payload.get("total_num_steps")
            or payload.get("total_elements")
        )

        percent = _maybe_number(payload.get("percent") or payload.get("progress"))
        if percent is not None and percent <= 1 and payload.get("progress") is not None:
            percent *= 100

        eta_seconds = _maybe_number(payload.get("eta_seconds") or payload.get("eta"))

        metrics_raw = payload.get("metrics") or {}
        metrics = dict(metrics_raw) if isinstance(metrics_raw, Mapping) else {}

        extra_metrics = {}
        for candidate in ("loss", "learning_rate", "lr", "accuracy"):
            value = payload.get(candidate)
            if value is not None and candidate not in metrics:
                extra_metrics[candidate] = value

        if extra_metrics:
            metrics.update(extra_metrics)

        if all(value is None for value in (label, current, total, percent, eta_seconds)) and not metrics:
            return None

        return cls(label=label, current=current, total=total, percent=percent, eta_seconds=eta_seconds, metrics=metrics)


@dataclass(frozen=True)
class StageData:
    """Lifecycle stage definition."""

    key: str
    status: StageStatus = StageStatus.RUNNING
    label: str | None = None
    progress: ProgressData | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self.key,
            "status": self.status.value,
            "label": self.label,
        }
        if self.progress:
            progress_dict = self.progress.to_dict()
            payload.update(
                {
                    "percent": progress_dict.get("percent"),
                    "current": progress_dict.get("current"),
                    "total": progress_dict.get("total"),
                }
            )
            payload["progress"] = progress_dict
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> StageData | None:
        if not isinstance(payload, Mapping):
            return None
        key = payload.get("key") or payload.get("progress_type") or payload.get("stage")
        key = _safe_key(key)
        if not key:
            return None
        status_value = payload.get("status") or payload.get("state") or "running"
        status = _coerce_stage_status(status_value)
        label = _safe_str(payload.get("label") or payload.get("readable_type"))
        if not label and key:
            label = _prettify_key(key)
        progress = ProgressData.from_mapping(payload.get("progress") or payload)
        metadata = payload.get("metadata") or {}
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        return cls(key=key, status=status, label=label, progress=progress, metadata=metadata)


@dataclass(frozen=True)
class CheckpointData:
    """Metadata about checkpoint operations."""

    path: str | None = None
    label: str | None = None
    is_final: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "label": self.label,
            "is_final": self.is_final,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> CheckpointData | None:
        if not isinstance(payload, Mapping):
            return None
        path = _safe_str(payload.get("path") or payload.get("checkpoint_path"))
        label = _safe_str(payload.get("label") or payload.get("message"))
        is_final = bool(payload.get("is_final") or payload.get("final"))
        extra = payload.get("extra") or {}
        extra = dict(extra) if isinstance(extra, Mapping) else {}
        if not any((path, label, extra)):
            return None
        return cls(path=path, label=label, is_final=is_final, extra=extra)


@dataclass(frozen=True)
class ValidationAsset:
    """Single validation attachment (image, video, etc.)."""

    url: str
    alt: str | None = None
    kind: str = "image"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"url": self.url, "kind": self.kind}
        if self.alt:
            payload["alt"] = self.alt
        return payload


@dataclass(frozen=True)
class ValidationData:
    """Metadata and assets produced by validation runs."""

    title: str | None = None
    prompt: str | None = None
    score: float | str | None = None
    assets: Sequence[ValidationAsset] = field(default_factory=tuple)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "title": self.title,
            "prompt": self.prompt,
            "score": self.score,
            "assets": [asset.to_dict() for asset in self.assets],
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> ValidationData | None:
        if not isinstance(payload, Mapping):
            return None
        title = _safe_str(payload.get("title") or payload.get("headline") or payload.get("message"))
        prompt = _safe_str(payload.get("prompt"))
        score = payload.get("score") or payload.get("evaluation_score")
        images = payload.get("images") or payload.get("assets") or ()
        assets: list[ValidationAsset] = []
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
            for item in images:
                if isinstance(item, Mapping):
                    src = _safe_str(item.get("url") or item.get("src"))
                    if src:
                        assets.append(ValidationAsset(url=src, alt=title, kind=_safe_str(item.get("kind")) or "image"))
                else:
                    src = _safe_str(item)
                    if src:
                        assets.append(ValidationAsset(url=src, alt=title, kind="image"))
        extra = payload.get("extra") or {}
        extra = dict(extra) if isinstance(extra, Mapping) else {}
        if not any((title, prompt, score, assets, extra)):
            return None
        return cls(title=title, prompt=prompt, score=score, assets=tuple(assets), extra=extra)


@dataclass(frozen=True)
class CallbackEvent:
    """Unified representation of webhook callbacks."""

    type: EventType
    severity: EventSeverity = EventSeverity.INFO
    title: str | None = None
    message: str | None = None
    job_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    progress: ProgressData | None = None
    stage: StageData | None = None
    checkpoint: CheckpointData | None = None
    validation: ValidationData | None = None
    images: Sequence[str] = field(default_factory=tuple)
    data: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)
    index: int | None = None
    reset_history: bool = False

    def with_index(self, index: int) -> CallbackEvent:
        return replace(self, index=index)

    @property
    def id(self) -> str | None:
        return str(self.index) if self.index is not None else None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat(),
            "images": list(self.images),
        }
        if self.progress:
            payload["progress"] = self.progress.to_dict()
        if self.stage:
            payload["stage"] = self.stage.to_dict()
        if self.checkpoint:
            payload["checkpoint"] = self.checkpoint.to_dict()
        if self.validation:
            payload["validation"] = self.validation.to_dict()
        if self.data:
            payload["data"] = dict(self.data)
        if self.reset_history:
            payload["reset_history"] = True
        if "headline" not in payload and self.title:
            payload["headline"] = self.title
        if "body" not in payload and self.message:
            payload["body"] = self.message
        payload.setdefault("message_type", self.type.value)
        return payload

    @classmethod
    def from_message(cls, raw: Mapping[str, Any]) -> CallbackEvent:
        if not isinstance(raw, Mapping):
            raise TypeError("raw payload must be a mapping")

        event_type = _coerce_event_type(raw)
        severity = _coerce_severity(raw.get("severity") or raw.get("level"))

        title = _safe_str(raw.get("title") or raw.get("headline") or raw.get("label"))
        message = _safe_str(raw.get("message") or raw.get("body"))

        job_id = _safe_str(
            raw.get("job_id")
            or raw.get("job")
            or raw.get("extra", {}).get("job_id")
            or raw.get("progress", {}).get("job_id")
        )

        timestamp = _coerce_timestamp(raw.get("timestamp"))

        progress_payload = raw.get("progress")
        stage_payload = raw.get("stage")
        checkpoint_payload = raw.get("checkpoint")
        validation_payload = raw.get("validation")

        # Legacy compatibility for older webhook payloads
        if event_type == EventType.LIFECYCLE_STAGE and not stage_payload:
            stage_payload = raw
        if event_type == EventType.TRAINING_PROGRESS and not progress_payload:
            progress_payload = raw.get("extra") or raw

        progress = ProgressData.from_mapping(progress_payload)
        stage = StageData.from_mapping(stage_payload)
        checkpoint = CheckpointData.from_mapping(checkpoint_payload or raw if event_type == EventType.CHECKPOINT else None)
        validation = ValidationData.from_mapping(validation_payload or raw if event_type == EventType.VALIDATION else None)

        images = []
        raw_images = raw.get("images")
        if isinstance(raw_images, Sequence) and not isinstance(raw_images, (str, bytes, bytearray)):
            images = [_safe_str(image) for image in raw_images if _safe_str(image)]

        data = raw.get("data") or raw.get("extras") or {}
        data = dict(data) if isinstance(data, Mapping) else {}

        reset_history = bool(raw.get("reset_history"))

        if not title and message:
            title = _derive_title(message)

        return cls(
            type=event_type,
            severity=severity,
            title=title,
            message=message,
            job_id=job_id,
            timestamp=timestamp,
            progress=progress,
            stage=stage,
            checkpoint=checkpoint,
            validation=validation,
            images=tuple(images),
            data=data,
            raw=dict(raw),
            reset_history=reset_history,
        )


def _coerce_event_type(raw: Mapping[str, Any]) -> EventType:
    """
    Coerce event type from explicit type field in payload.

    All event sources MUST send an explicit 'type' field.
    No heuristic detection or fallback mappings are performed.
    """
    candidate = raw.get("type") or raw.get("event_type") or raw.get("message_type") or raw.get("category")
    if isinstance(candidate, EventType):
        return candidate
    if isinstance(candidate, str):
        value = candidate.strip().lower()
        if value in EventType._value2member_map_:
            return EventType(value)

    # If no valid type was provided, default to notification
    # Event sources should be fixed to send explicit types
    return EventType.NOTIFICATION


def _coerce_severity(value: Any) -> EventSeverity:
    if isinstance(value, EventSeverity):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in EventSeverity._value2member_map_:
            return EventSeverity(lowered)
    return EventSeverity.INFO


def _coerce_stage_status(value: Any) -> StageStatus:
    if isinstance(value, StageStatus):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in StageStatus._value2member_map_:
            return StageStatus(lowered)
        if lowered in {"complete", "completed", "done"}:
            return StageStatus.COMPLETED
        if lowered in {"fail", "failed", "error"}:
            return StageStatus.FAILED
        if lowered in {"queued", "pending"}:
            return StageStatus.PENDING
    return StageStatus.RUNNING


def _coerce_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(tz=timezone.utc)


def _maybe_number(value: Any) -> float | None:
    if value in (None, "", False):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:  # NaN
        return None
    return numeric


def _safe_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _safe_key(value: Any) -> str | None:
    key = _safe_str(value)
    if not key:
        return None
    return key.strip().lower().replace(" ", "_")


def _prettify_key(value: str) -> str:
    return " ".join(part.capitalize() for part in value.replace("_", " ").split())


def _derive_title(message: str, *, max_length: int = 120) -> str:
    first_line = message.strip().splitlines()[0]
    return first_line[:max_length]


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
]
