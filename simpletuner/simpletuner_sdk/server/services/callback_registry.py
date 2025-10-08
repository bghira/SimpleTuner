"""Declarative registry that maps raw webhook payloads into typed events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

from .callback_events import (
    CallbackCategory,
    CallbackEvent,
    CallbackEventFactory,
    CallbackSeverity,
    CheckpointPayload,
    ProgressPayload,
    ValidationAsset,
    ValidationPayload,
    coerce_timestamp,
    derive_headline,
    prettify_message_type,
)

MessageMapper = Callable[[Mapping[str, Any]], str | None]
ExtrasBuilder = Callable[[Mapping[str, Any]], Mapping[str, Any]]
SequenceBuilder = Callable[[Mapping[str, Any]], Sequence[Any]]
SeverityBuilder = Callable[[Mapping[str, Any]], CallbackSeverity]


def _safe_message(raw: Mapping[str, Any]) -> str | None:
    value = raw.get("message")
    return value if isinstance(value, str) else None


def _clamp_percent(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not (numeric == numeric):  # NaN check
        return None
    return max(0.0, min(100.0, numeric))


def _default_extras(raw: Mapping[str, Any], *, message_field: str | None) -> dict[str, Any]:
    ignore = {"timestamp", "message_type", "job_id", "images"}
    if message_field:
        ignore.add(message_field)
    extras = {k: v for k, v in raw.items() if k not in ignore}
    return extras


@dataclass(frozen=True)
class CallbackEventDefinition:
    """Declarative description of how to render a raw webhook message."""

    category: CallbackCategory
    severity: CallbackSeverity | SeverityBuilder = CallbackSeverity.INFO
    headline: str | MessageMapper | None = None
    body: str | MessageMapper | None = None
    message_field: str | None = "message"
    extras_builder: ExtrasBuilder | None = None
    progress_builder: Callable[[Mapping[str, Any]], ProgressPayload | None] | None = None
    checkpoint_builder: Callable[[Mapping[str, Any]], CheckpointPayload | None] | None = None
    validation_builder: Callable[[Mapping[str, Any]], ValidationPayload | None] | None = None
    images_builder: SequenceBuilder | None = None
    reset_history: bool = False
    dedupe_builder: Callable[[Mapping[str, Any]], str | None] | None = None

    def build(self, raw: Mapping[str, Any]) -> CallbackEvent:
        message_type = raw.get("message_type")
        timestamp = coerce_timestamp(raw.get("timestamp"))
        message = raw.get(self.message_field) if self.message_field else None

        headline = self._evaluate_text(self.headline, raw)
        if not headline:
            if isinstance(message, str) and message:
                headline = derive_headline(message)
            else:
                headline = prettify_message_type(message_type)

        body = self._evaluate_text(self.body, raw)
        if body is None and isinstance(message, str) and message != headline:
            body = message

        extras = self.extras_builder(raw) if self.extras_builder else _default_extras(raw, message_field=self.message_field)
        if self.images_builder:
            images = tuple(str(item) for item in self.images_builder(raw) or ())
        else:
            images_value = raw.get("images")
            if isinstance(images_value, Sequence) and not isinstance(images_value, (str, bytes, bytearray)):
                images = tuple(str(item) for item in images_value)
            else:
                images = tuple()

        progress = self.progress_builder(raw) if self.progress_builder else None
        checkpoint = self.checkpoint_builder(raw) if self.checkpoint_builder else None
        validation = self.validation_builder(raw) if self.validation_builder else None
        dedupe_key = self.dedupe_builder(raw) if self.dedupe_builder else None

        resolved_severity = self._resolve_severity(raw)

        return CallbackEvent(
            category=self.category,
            severity=resolved_severity,
            headline=headline or prettify_message_type(message_type),
            body=body,
            job_id=raw.get("job_id"),
            timestamp=timestamp,
            message_type=message_type,
            progress=progress,
            checkpoint=checkpoint,
            validation=validation,
            images=images,
            extras=extras,
            raw=dict(raw),
            reset_history=self.reset_history,
            dedupe_key=dedupe_key,
        )

    @staticmethod
    def _evaluate_text(value: str | MessageMapper | None, raw: Mapping[str, Any]) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return value(raw)
        except Exception:
            return None

    def _resolve_severity(self, raw: Mapping[str, Any]) -> CallbackSeverity:
        if isinstance(self.severity, CallbackSeverity):
            return self.severity
        try:
            return self.severity(raw)
        except Exception:
            return CallbackSeverity.INFO


class CallbackEventRegistry:
    """Registry containing mapping rules from raw payloads to typed events."""

    def __init__(self) -> None:
        self._by_type: Dict[str, CallbackEventDefinition] = {}
        self._predicates: list[tuple[Callable[[Mapping[str, Any]], bool], CallbackEventDefinition]] = []

    def register_type(self, message_type: str, definition: CallbackEventDefinition) -> None:
        self._by_type[message_type] = definition

    def register_many(self, mapping: Mapping[str, CallbackEventDefinition]) -> None:
        for message_type, definition in mapping.items():
            self.register_type(message_type, definition)

    def register_predicate(
        self,
        predicate: Callable[[Mapping[str, Any]], bool],
        definition: CallbackEventDefinition,
    ) -> None:
        self._predicates.append((predicate, definition))

    def resolve(self, raw: Mapping[str, Any]) -> CallbackEventFactory | None:
        message_type = raw.get("message_type")
        if message_type and message_type in self._by_type:
            definition = self._by_type[message_type]
            return definition.build
        for predicate, definition in self._predicates:
            try:
                if predicate(raw):
                    return definition.build
            except Exception:
                continue
        return None


def _progress_from_mixin(raw: Mapping[str, Any]) -> ProgressPayload | None:
    payload = raw.get("message")
    if not isinstance(payload, Mapping):
        return None
    progress_type = payload.get("progress_type") or raw.get("progress_type")
    readable = raw.get("readable_type") or payload.get("readable_type")
    label = readable or progress_type
    current = payload.get("current_estimated_index")
    total = payload.get("total_elements")
    progress_value = payload.get("progress")
    percent: float | None = None
    if isinstance(progress_value, (int, float)):
        if progress_value <= 1:
            percent = float(progress_value) * 100
        elif progress_value <= 100:
            percent = float(progress_value)
    if percent is None and isinstance(current, (int, float)) and isinstance(total, (int, float)) and total:
        percent = float(current) / float(total) * 100
    percent = _clamp_percent(percent)
    if percent is not None:
        percent = round(percent)
    extra = {k: v for k, v in payload.items() if k not in {"current_estimated_index", "total_elements"}}
    if readable:
        extra.setdefault("readable_type", readable)
    if progress_type is not None:
        extra.setdefault("progress_type", progress_type)
    current_value = _maybe_float(current) if current is not None else _maybe_float(raw.get("current_estimated_index"))
    total_value = _maybe_float(total) if total is not None else _maybe_float(raw.get("total_elements"))
    return ProgressPayload(
        label=label,
        current=current_value,
        total=total_value,
        percent=percent,
        extra=extra,
    )


def _progress_from_training_status(raw: Mapping[str, Any]) -> ProgressPayload | None:
    # New format with top-level fields (coupled client-server)
    current = raw.get("global_step")
    total = raw.get("total_num_steps") or raw.get("steps_remaining_at_start")
    extra = {key: value for key, value in raw.items() if key not in {"message_type", "timestamp", "job_id"}}

    percent: float | None = None
    if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total:
        percent = float(current) / float(total) * 100
    percent = _clamp_percent(percent)
    return ProgressPayload(
        label="Training progress",
        current=_maybe_float(current),
        total=_maybe_float(total),
        percent=percent,
        extra=extra,
    )


def _checkpoint_payload(raw: Mapping[str, Any]) -> CheckpointPayload | None:
    message = _safe_message(raw)
    checkpoint_path = raw.get("checkpoint_path")
    extra = {
        key: value
        for key, value in raw.items()
        if key not in {"message", "timestamp", "message_type", "job_id", "checkpoint_path"}
    }
    return CheckpointPayload(
        path=str(checkpoint_path) if checkpoint_path else None,
        label=message,
        is_final=raw.get("message_type") == "checkpoint_state_save_finalized",
        extra=extra,
    )


def _validation_payload(raw: Mapping[str, Any]) -> ValidationPayload | None:
    message = _safe_message(raw)
    prompt = raw.get("prompt")
    score = raw.get("score") or raw.get("evaluation_score")
    assets = []
    images = raw.get("images")
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
        for item in images:
            assets.append(
                ValidationAsset(
                    url=str(item),
                    alt=message,
                )
            )
    extra = {
        key: value
        for key, value in raw.items()
        if key not in {"message", "timestamp", "message_type", "job_id", "prompt", "score", "evaluation_score", "images"}
    }
    return ValidationPayload(
        headline=message,
        prompt=str(prompt) if prompt else None,
        score=score,
        assets=tuple(assets),
        extra=extra,
    )


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _dedupe_progress_key(raw: Mapping[str, Any]) -> str | None:
    payload = raw.get("message")
    if not isinstance(payload, Mapping):
        return None
    job_id = raw.get("job_id")
    label = payload.get("progress_type")
    progress_value = payload.get("progress")
    return f"{job_id}:{label}:{progress_value}"


def _status_headline(raw: Mapping[str, Any]) -> str | None:
    status = raw.get("status")
    if isinstance(status, str):
        return prettify_message_type(status)
    return None


def _status_severity(raw: Mapping[str, Any]) -> CallbackSeverity:
    status = raw.get("status")
    if isinstance(status, str):
        lowered = status.lower()
        if lowered in {"failed", "error", "fatal"}:
            return CallbackSeverity.ERROR
        if lowered in {"warning", "degraded", "interrupted"}:
            return CallbackSeverity.WARNING
        if lowered in {"success", "completed", "available", "ready"}:
            return CallbackSeverity.SUCCESS
    return CallbackSeverity.INFO


default_callback_registry = CallbackEventRegistry()


_basic_message_types: dict[str, CallbackEventDefinition] = {}

for message_type in (
    "_send_webhook_msg",
    "init_load_base_model_begin",
    "init_load_base_model_completed",
    "init_data_backend_begin",
    "init_data_backend_completed",
    "init_data_backend",
    "init_prepare_models_begin",
    "init_prepare_models_completed",
    "init_benchmark_base_model_begin",
    "init_benchmark_base_model_completed",
    "init_resume_checkpoint",
    "init_resume_checkpoint_details",
    "_train_initial_msg",
    "exit",
    "model_save_start",
    "run_complete",
):
    _basic_message_types[message_type] = CallbackEventDefinition(
        category=CallbackCategory.JOB,
        severity=CallbackSeverity.INFO,
        body=_safe_message,
    )

_basic_message_types["configure_webhook"] = CallbackEventDefinition(
    category=CallbackCategory.JOB,
    severity=CallbackSeverity.INFO,
    headline="Training configuration started",
    body=_safe_message,
    reset_history=True,
)

_basic_message_types["fatal_error"] = CallbackEventDefinition(
    category=CallbackCategory.ALERT,
    severity=CallbackSeverity.ERROR,
    body=_safe_message,
)

_basic_message_types["error"] = CallbackEventDefinition(
    category=CallbackCategory.ALERT,
    severity=CallbackSeverity.ERROR,
    body=_safe_message,
)

_basic_message_types["warning"] = CallbackEventDefinition(
    category=CallbackCategory.ALERT,
    severity=CallbackSeverity.WARNING,
    body=_safe_message,
)

_basic_message_types["training_config"] = CallbackEventDefinition(
    category=CallbackCategory.JOB,
    severity=CallbackSeverity.INFO,
    headline="Training configuration",
    body=None,
    message_field=None,
)

_basic_message_types["checkpoint_state_save"] = CallbackEventDefinition(
    category=CallbackCategory.CHECKPOINT,
    severity=CallbackSeverity.INFO,
    body=_safe_message,
    checkpoint_builder=_checkpoint_payload,
)

_basic_message_types["checkpoint_state_save_completed"] = CallbackEventDefinition(
    category=CallbackCategory.CHECKPOINT,
    severity=CallbackSeverity.SUCCESS,
    body=_safe_message,
    checkpoint_builder=_checkpoint_payload,
)

_basic_message_types["checkpoint_state_save_distiller"] = CallbackEventDefinition(
    category=CallbackCategory.CHECKPOINT,
    severity=CallbackSeverity.INFO,
    body=_safe_message,
)

_basic_message_types["checkpoint_state_save_distiller_completed"] = CallbackEventDefinition(
    category=CallbackCategory.CHECKPOINT,
    severity=CallbackSeverity.SUCCESS,
    body=_safe_message,
)

_basic_message_types["checkpoint_state_save_finalized"] = CallbackEventDefinition(
    category=CallbackCategory.CHECKPOINT,
    severity=CallbackSeverity.SUCCESS,
    body=_safe_message,
    checkpoint_builder=_checkpoint_payload,
)

_basic_message_types["progress_update"] = CallbackEventDefinition(
    category=CallbackCategory.PROGRESS,
    severity=CallbackSeverity.INFO,
    body=None,
    message_field=None,
    progress_builder=_progress_from_mixin,
    dedupe_builder=_dedupe_progress_key,
)

_basic_message_types["train_status"] = CallbackEventDefinition(
    category=CallbackCategory.STATUS,
    severity=_status_severity,
    headline=_status_headline,
    body=_safe_message,
    progress_builder=_progress_from_training_status,
)

_basic_message_types["training_complete"] = CallbackEventDefinition(
    category=CallbackCategory.PROGRESS,
    severity=CallbackSeverity.SUCCESS,
    body=None,
    message_field=None,
    progress_builder=_progress_from_training_status,
)

_basic_message_types["validation_start"] = CallbackEventDefinition(
    category=CallbackCategory.VALIDATION,
    severity=CallbackSeverity.INFO,
    body=_safe_message,
)

_basic_message_types["validation_log"] = CallbackEventDefinition(
    category=CallbackCategory.VALIDATION,
    severity=CallbackSeverity.INFO,
    body=_safe_message,
    validation_builder=_validation_payload,
)

_basic_message_types["prepared_sample"] = CallbackEventDefinition(
    category=CallbackCategory.DEBUG,
    severity=CallbackSeverity.INFO,
    body=_safe_message,
)

default_callback_registry.register_many(_basic_message_types)
default_callback_registry.register_type(
    "init_resume_checkpoint_details",
    CallbackEventDefinition(
        category=CallbackCategory.JOB,
        severity=CallbackSeverity.INFO,
        headline="Resume checkpoint details",
        body=None,
        message_field=None,
    ),
)

default_callback_registry.register_type(
    "run_complete",
    CallbackEventDefinition(
        category=CallbackCategory.JOB,
        severity=CallbackSeverity.SUCCESS,
        body=_safe_message,
    ),
)

default_callback_registry.register_type(
    "exit",
    CallbackEventDefinition(
        category=CallbackCategory.ALERT,
        severity=CallbackSeverity.WARNING,
        body=_safe_message,
    ),
)

default_callback_registry.register_type(
    "checkpoint_upload",
    CallbackEventDefinition(
        category=CallbackCategory.CHECKPOINT,
        severity=CallbackSeverity.INFO,
        body=_safe_message,
    ),
)
