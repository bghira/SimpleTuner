"""Typed callback event domain models for webhook ingestion."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


class CallbackCategory(str, Enum):
    """High-level grouping for callback events."""

    JOB = "job"
    PROGRESS = "progress"
    CHECKPOINT = "checkpoint"
    VALIDATION = "validation"
    ALERT = "alert"
    STATUS = "status"
    DEBUG = "debug"


class CallbackSeverity(str, Enum):
    """Severity level for surfaced events."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ProgressPayload:
    """Normalized progress details for rendering progress bars and metrics."""

    label: str | None = None
    current: float | None = None
    total: float | None = None
    percent: float | None = None
    eta_seconds: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "current": self.current,
            "total": self.total,
            "percent": self.percent,
            "eta_seconds": self.eta_seconds,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


@dataclass(frozen=True)
class CheckpointPayload:
    """Metadata about checkpoint operations."""

    path: str | None = None
    is_final: bool = False
    label: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "is_final": self.is_final,
            "label": self.label,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


@dataclass(frozen=True)
class ValidationAsset:
    """Single validation attachment (image, video, etc)."""

    url: str
    alt: str | None = None
    kind: str = "image"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"url": self.url, "kind": self.kind}
        if self.alt:
            payload["alt"] = self.alt
        return payload


@dataclass(frozen=True)
class ValidationPayload:
    """Metadata and assets produced by validation runs."""

    headline: str | None = None
    prompt: str | None = None
    score: float | str | None = None
    assets: Sequence[ValidationAsset] = field(default_factory=tuple)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "headline": self.headline,
            "prompt": self.prompt,
            "score": self.score,
            "assets": [asset.to_dict() for asset in self.assets],
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


class CallbackEventFactory(Protocol):
    """Protocol for factory callables that turn raw payloads into typed events."""

    def __call__(self, raw: Mapping[str, Any]) -> "CallbackEvent":
        ...


@dataclass(frozen=True)
class CallbackEvent:
    """Typed representation of a webhook callback event."""

    category: CallbackCategory
    severity: CallbackSeverity
    headline: str
    body: str | None
    job_id: str | None
    timestamp: datetime
    message_type: str | None = None
    progress: ProgressPayload | None = None
    checkpoint: CheckpointPayload | None = None
    validation: ValidationPayload | None = None
    images: Sequence[str] = field(default_factory=tuple)
    extras: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)
    index: int | None = None
    reset_history: bool = False
    dedupe_key: str | None = None

    def with_index(self, index: int) -> "CallbackEvent":
        """Return a copy of the event with the datastore index set."""
        return replace(self, index=index)

    @property
    def id(self) -> str | None:
        """Stable identifier for rendering or anchors."""
        return str(self.index) if self.index is not None else None

    def to_payload(self) -> dict[str, Any]:
        """Render the event into a presentation-friendly dictionary."""
        payload: dict[str, Any] = {
            "id": self.id,
            "category": self.category.value,
            "severity": self.severity.value,
            "headline": self.headline,
            "body": self.body,
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat(),
            "images": list(self.images),
        }
        if self.progress:
            payload["progress"] = self.progress.to_dict()
        if self.checkpoint:
            payload["checkpoint"] = self.checkpoint.to_dict()
        if self.validation:
            payload["validation"] = self.validation.to_dict()
        if self.extras:
            payload["extras"] = dict(self.extras)
        if self.message_type:
            payload["message_type"] = self.message_type
        return payload

    @classmethod
    def from_message(
        cls,
        raw: Mapping[str, Any],
        *,
        registry: "CallbackEventRegistry | None" = None,
    ) -> "CallbackEvent":
        """Build a typed event using the supplied registry or the default registry."""
        from .callback_registry import default_callback_registry

        active_registry = registry or default_callback_registry
        factory = active_registry.resolve(raw)
        if factory:
            return factory(raw)
        return cls._fallback(raw)

    @classmethod
    def _fallback(cls, raw: Mapping[str, Any]) -> "CallbackEvent":
        """Generic mapper for unknown callbacks."""
        message = safe_get(raw, "message")
        timestamp = coerce_timestamp(raw.get("timestamp"))
        message_type = raw.get("message_type")
        headline = derive_headline(message) if message else prettify_message_type(message_type)
        body = message if message and message != headline else None
        extras = {key: value for key, value in raw.items() if key not in {"message", "timestamp", "message_type", "job_id"}}
        return cls(
            category=CallbackCategory.DEBUG,
            severity=CallbackSeverity.INFO,
            headline=headline or "Callback Event",
            body=body,
            job_id=raw.get("job_id"),
            timestamp=timestamp,
            message_type=message_type,
            extras=extras,
            raw=dict(raw),
        )


def coerce_timestamp(value: Any) -> datetime:
    """Normalize timestamp inputs to timezone-aware datetimes."""
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

    # Fallback to now if malformed or missing
    return datetime.now(tz=timezone.utc)


def derive_headline(message: str | None, *, max_length: int = 120) -> str | None:
    if not message:
        return None
    first_line = message.strip().splitlines()[0]
    return first_line[:max_length]


def prettify_message_type(message_type: str | None) -> str:
    if not message_type:
        return "Update"
    parts = message_type.replace("_", " ").strip().split()
    return " ".join(word.capitalize() for word in parts) if parts else "Update"


def safe_get(mapping: Mapping[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except Exception:
        return None


# Forward declaration for type checkers
class CallbackEventRegistry(Protocol):
    def resolve(self, raw: Mapping[str, Any]) -> CallbackEventFactory | None:
        ...
