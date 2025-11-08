"""Service layer responsible for normalising and persisting webhook callbacks."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections import deque
from threading import Lock
from typing import Any, Mapping, Sequence

from ...api_state import APIState
from ...process_keeper import append_external_event
from .callback_events import CallbackEvent, EventSeverity, EventType, ProgressData, StageData, StageStatus
from .event_store import EventStore
from .sse_manager import get_sse_manager

logger = logging.getLogger(__name__)

_default_service: CallbackService | None = None
_service_lock = Lock()


class CallbackService:
    """Wraps the event store to provide typed callback events and read APIs."""

    def __init__(self, event_store: EventStore) -> None:
        self._store = event_store
        self._lock = Lock()
        self._typed_by_index: dict[int, CallbackEvent] = {}
        self._index_order: deque[int] = deque(maxlen=event_store.max_events)
        self._job_status: dict[str, str] = {}
        self._bootstrap_from_store()

    def handle_incoming(self, raw_payload: Mapping[str, Any]) -> CallbackEvent | None:
        """Normalise and persist a raw webhook payload."""
        if not isinstance(raw_payload, Mapping):
            raise TypeError("raw_payload must be a mapping")

        normalized_payload = dict(raw_payload)
        message_type = str(normalized_payload.get("message_type") or normalized_payload.get("type") or "").lower()

        if message_type == "configure_webhook":
            normalized_payload.setdefault("reset_history", True)

        event = CallbackEvent.from_message(normalized_payload)

        if self._should_suppress_event(event):
            return None

        if event.reset_history:
            with self._lock:
                self._store.clear()
                self._typed_by_index.clear()
                self._index_order.clear()

        with self._lock:
            record = self._prepare_record(normalized_payload)
            index = self._store.add_event(record)
            typed_event = event.with_index(index)
            record["typed"] = typed_event.to_payload()
            record["type"] = typed_event.type.value
            record["severity"] = typed_event.severity.value
            record["timestamp"] = typed_event.timestamp.isoformat()

            self._typed_by_index[index] = typed_event
            self._append_index(index)

            # Debug logging for validation events
            if typed_event.type in {EventType.VALIDATION, EventType.VALIDATION_IMAGE}:
                logger.debug(
                    f"Stored {typed_event.type.value} event at index {index}, " f"has_stage={typed_event.stage is not None}"
                )

        self._update_training_state(typed_event)
        self._mirror_to_process_keeper(typed_event)
        return typed_event

    def get_recent(self, limit: int = 10) -> list[CallbackEvent]:
        """Return the most recent *limit* events, newest first."""
        if limit <= 0:
            return []
        with self._lock:
            indices = list(self._index_order)[-limit:]
            snapshot = {idx: self._typed_by_index.get(idx) for idx in indices}
        return [
            event
            for idx in reversed(indices)
            if (event := snapshot.get(idx)) is not None and not self._should_suppress_event(event)
        ]

    def stream_since(self, index: int) -> list[CallbackEvent]:
        """Return all events newer than the provided index."""
        with self._lock:
            matching_indices = [idx for idx in self._index_order if idx > index]
            snapshot = {idx: self._typed_by_index.get(idx) for idx in matching_indices}
        return [
            event
            for idx in matching_indices
            if (event := snapshot.get(idx)) is not None and not self._should_suppress_event(event)
        ]

    def latest_for_job(self, job_id: str | None) -> CallbackEvent | None:
        """Return the newest event for a specific job identifier."""
        if not job_id:
            return None
        with self._lock:
            indices = list(reversed(self._index_order))
            snapshot = {idx: self._typed_by_index.get(idx) for idx in indices}
        for idx in indices:
            event = snapshot.get(idx)
            if event and event.job_id == job_id:
                return event
        return None

    def latest_index(self) -> int | None:
        """Return the highest event index currently stored."""
        with self._lock:
            if not self._index_order:
                return None
            return self._index_order[-1]

    def as_payloads(self, events: Sequence[CallbackEvent]) -> list[dict[str, Any]]:
        """Convert events into serialisable dictionaries for transport."""
        return [event.to_payload() for event in events]

    def _append_index(self, index: int) -> None:
        if self._index_order.maxlen and len(self._index_order) == self._index_order.maxlen:
            oldest = self._index_order[0]
            self._index_order.append(index)
            self._evict_index(oldest)
        else:
            self._index_order.append(index)

    def _prepare_record(self, raw_payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            payload_copy = copy.deepcopy(dict(raw_payload))
        except Exception:
            payload_copy = dict(raw_payload)
        return {
            "raw": payload_copy,
            "type": payload_copy.get("type") or payload_copy.get("message_type"),
            "severity": payload_copy.get("severity"),
            "job_id": payload_copy.get("job_id"),
            "timestamp": payload_copy.get("timestamp"),
        }

    def _bootstrap_from_store(self) -> None:
        existing_events = self._store.get_all_events()
        with self._lock:
            for record in existing_events:
                self._ingest_existing(record)

    def _ingest_existing(self, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            return
        index = record.get("_index")
        if index is None:
            return
        raw_payload = record.get("raw") if isinstance(record.get("raw"), Mapping) else record
        try:
            event = CallbackEvent.from_message(raw_payload).with_index(index)
        except Exception:
            return

        self._typed_by_index[index] = event
        if self._index_order.maxlen and len(self._index_order) == self._index_order.maxlen:
            oldest = self._index_order[0]
            self._index_order.append(index)
            self._evict_index(oldest)
        else:
            self._index_order.append(index)

        if isinstance(record, dict):
            record.setdefault("raw", event.raw)
            record.setdefault("typed", event.to_payload())
            record.setdefault("type", event.type.value)
            record.setdefault("severity", event.severity.value)
            record.setdefault("timestamp", event.timestamp.isoformat())

    def _evict_index(self, index: int) -> None:
        self._typed_by_index.pop(index, None)

    @staticmethod
    def _merge_progress_state(
        previous: Mapping[str, Any] | None,
        progress: ProgressData,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = dict(previous) if isinstance(previous, Mapping) else {}

        percent = progress.normalized_percent
        if percent is not None:
            merged["percent"] = max(float(merged.get("percent") or 0), percent)
        elif "percent" not in merged:
            merged["percent"] = 0.0

        if progress.current is not None:
            merged["step"] = int(progress.current)
        else:
            merged.setdefault("step", 0)

        if progress.total is not None:
            merged["total_steps"] = int(progress.total)
        else:
            merged.setdefault("total_steps", 0)

        merged.setdefault("epoch", 0)
        merged.setdefault("total_epochs", 0)

        metrics = dict(progress.metrics or {})
        if isinstance(extra, Mapping):
            for key in ("loss", "learning_rate", "lr", "epoch", "total_epochs", "final_epoch"):
                if key in extra and key not in metrics:
                    metrics[key] = extra[key]

        epoch = _maybe_int(metrics.pop("epoch", None))
        if epoch is not None:
            merged["epoch"] = epoch

        total_epochs = _maybe_int(metrics.pop("total_epochs", None) or metrics.pop("final_epoch", None))
        if total_epochs is not None:
            merged["total_epochs"] = total_epochs

        loss = _maybe_float(metrics.pop("loss", None))
        if loss is not None:
            merged["loss"] = loss
        else:
            merged.setdefault("loss", None)

        lr = _maybe_float(metrics.pop("learning_rate", None) or metrics.pop("lr", None))
        if lr is not None:
            merged["learning_rate"] = lr
        else:
            merged.setdefault("learning_rate", None)

        if metrics:
            extra_metrics = merged.setdefault("metrics", {})
            if isinstance(extra_metrics, Mapping):
                extra_metrics = dict(extra_metrics)
            extra_metrics.update(metrics)
            merged["metrics"] = extra_metrics

        return merged

    def _update_startup_stage(
        self,
        stage: StageData,
        *,
        job_id: str | None,
        job_changed: bool = False,
    ) -> None:
        if isinstance(stage.key, str) and stage.key.startswith("stage_") and stage.key[6:].isdigit():
            return

        stages_state = APIState.get_state("training_startup_stages") or {}
        stages = dict(stages_state) if isinstance(stages_state, Mapping) else {}
        if job_changed:
            stages = {}

        progress = stage.progress or ProgressData()
        percent = progress.normalized_percent
        percent_value = int(round(percent)) if percent is not None else 0
        current = int(progress.current or 0)
        total = int(progress.total or 0)

        state = {
            "label": stage.label or _prettify(stage.key),
            "progress_type": stage.key,
            "status": stage.status.value,
            "percent": percent_value,
            "current": current,
            "total": total,
        }

        should_remove = stage.status == StageStatus.COMPLETED or stage.status == StageStatus.FAILED
        should_remove = should_remove or (total and current >= total) or percent_value >= 100

        if should_remove and stage.status != StageStatus.FAILED:
            state["status"] = StageStatus.COMPLETED.value
            state["percent"] = 100
            state["current"] = total or state["current"]
        elif stage.status == StageStatus.FAILED:
            state["status"] = StageStatus.FAILED.value

        stages[stage.key] = state

        APIState.set_state("training_startup_stages", stages)
        self._broadcast_startup_stage(job_id, state)

    def _handle_progress_event(self, event: CallbackEvent, job_id: str | None, job_changed: bool) -> None:
        if event.progress is None:
            return

        APIState.set_state("training_status", "running")
        if job_id:
            self._job_status[job_id] = "running"
        previous_progress = APIState.get_state("training_progress") or {}
        if job_changed:
            previous_progress = {}
        merged = self._merge_progress_state(previous_progress, event.progress, event.data)
        APIState.set_state("training_progress", merged)
        self._broadcast_training_progress(job_id, merged)

    def _handle_status_event(self, event: CallbackEvent, job_id: str | None) -> None:
        raw_status = None
        if isinstance(event.data, Mapping):
            raw_status = event.data.get("status") or event.data.get("state")
        if not raw_status and isinstance(event.raw, Mapping):
            raw_status = event.raw.get("status") or event.raw.get("state")
        if not raw_status:
            raw_status = event.message or event.title or "running"
        status = str(raw_status).strip().lower()
        if not status:
            status = "running"

        APIState.set_state("training_status", status)
        if job_id:
            self._job_status[job_id] = status

        if status in {"failed", "error", "fatal", "cancelled", "stopped"}:
            APIState.set_state("training_progress", None)
            self._clear_startup_stages()  # Clear all stages on error
            self._broadcast_progress_reset(job_id, status=status)
            current_job = APIState.get_state("current_job_id")
            if job_id and current_job == job_id:
                APIState.set_state("current_job_id", None)
        elif status in {"completed", "success"}:
            progress_state = APIState.get_state("training_progress") or {}
            if isinstance(progress_state, Mapping):
                progress_state = dict(progress_state)
                progress_state["percent"] = 100
                APIState.set_state("training_progress", progress_state)
            self._clear_startup_stages()  # Clear all stages on completion
            current_job = APIState.get_state("current_job_id")
            if job_id and current_job == job_id:
                APIState.set_state("current_job_id", None)
        elif status == "running":
            # Training has started running - clear initialization lifecycle stages
            self._clear_startup_stages()

    def _handle_summary_event(self, event: CallbackEvent, job_id: str | None) -> None:
        progress_state = APIState.get_state("training_progress") or {}
        if isinstance(progress_state, Mapping):
            progress_state = dict(progress_state)
            progress_state["percent"] = 100
            APIState.set_state("training_progress", progress_state)
        APIState.set_state("training_status", "completed")
        if job_id:
            self._job_status[job_id] = "completed"
        self._clear_startup_stages()

    def _handle_error_event(self, event: CallbackEvent, job_id: str | None) -> None:
        APIState.set_state("training_status", "error")
        APIState.set_state("training_progress", None)
        self._clear_startup_stages()
        self._broadcast_progress_reset(job_id, status="failed")
        if job_id:
            self._job_status[job_id] = "failed"
        current_job = APIState.get_state("current_job_id")
        if job_id and current_job == job_id:
            APIState.set_state("current_job_id", None)

    def _update_training_state(self, event: CallbackEvent) -> None:
        job_id = self._derive_job_id(event)
        previous_job_id = APIState.get_state("current_job_id")
        job_changed = bool(job_id and job_id != previous_job_id)

        if job_id:
            APIState.set_state("current_job_id", job_id)

        if event.stage:
            # Only mark training as complete for the actual training_complete stage
            stage_key = event.stage.key or ""

            if stage_key == "training_complete":
                # Training has completed - update API state to reflect this
                APIState.set_state("training_status", "completed")
                if job_id:
                    self._job_status[job_id] = "completed"
                # Ensure progress shows 100%
                progress_state = APIState.get_state("training_progress") or {}
                if isinstance(progress_state, Mapping):
                    progress_state = dict(progress_state)
                    progress_state["percent"] = 100
                    APIState.set_state("training_progress", progress_state)
                self._clear_startup_stages()
                self._update_startup_stage(event.stage, job_id=job_id, job_changed=job_changed)
                return

            # For all other stages, just update the stage tracking
            current_status = str(APIState.get_state("training_status") or "").lower()
            if current_status not in {"running", "completed", "success", "failed", "error", "fatal", "cancelled", "stopped"}:
                APIState.set_state("training_status", "starting")
                current_status = "starting"
            if job_id and current_status not in {"completed", "failed", "error", "cancelled", "stopped"}:
                self._job_status[job_id] = current_status
            self._update_startup_stage(event.stage, job_id=job_id, job_changed=job_changed)
            return

        if event.type == EventType.TRAINING_PROGRESS:
            self._handle_progress_event(event, job_id, job_changed)
            return

        if event.type == EventType.TRAINING_STATUS:
            self._handle_status_event(event, job_id)
            return

        if event.type == EventType.TRAINING_SUMMARY:
            self._handle_summary_event(event, job_id)
            return

        if event.type == EventType.ERROR or event.severity in {EventSeverity.ERROR, EventSeverity.CRITICAL}:
            self._handle_error_event(event, job_id)
            return

        if event.type in {EventType.CHECKPOINT, EventType.VALIDATION}:
            APIState.set_state("training_status", "running")
            if job_id:
                self._job_status[job_id] = "running"
            return

        if event.type == EventType.NOTIFICATION and event.progress:
            self._handle_progress_event(event, job_id, job_changed)

    def _clear_startup_stages(self) -> None:
        """Clear all startup stages from state."""
        APIState.set_state("training_startup_stages", {})

    def _broadcast_startup_stage(self, job_id: str | None, stage_state: Mapping[str, Any]) -> None:
        payload = {
            "type": "lifecycle.stage",
            "job_id": job_id,
            "stage": {
                "key": stage_state.get("progress_type"),
                "label": stage_state.get("label"),
                "status": stage_state.get("status", "running"),
                "percent": stage_state.get("percent", 0),
                "current": stage_state.get("current", 0),
                "total": stage_state.get("total", 0),
            },
        }
        logger.debug(
            f"Broadcasting lifecycle stage: {stage_state.get('progress_type')} (status={stage_state.get('status')}) for job {job_id}"
        )

        async def _broadcast():
            manager = get_sse_manager()
            await manager.broadcast(payload, event_type="lifecycle.stage")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.create_task(_broadcast())
        else:
            asyncio.run(_broadcast())

    def _broadcast_progress_reset(self, job_id: str | None, *, status: str) -> None:
        if job_id:
            self._job_status[job_id] = status.lower()

        payload = {
            "type": "training_progress",
            "job_id": job_id,
            "status": status,
            "reset": True,
            "percent": 0,
            "step": 0,
            "total_steps": 0,
            "epoch": 0,
            "total_epochs": 0,
        }

        async def _broadcast():
            manager = get_sse_manager()
            await manager.broadcast(payload, event_type="training_progress")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.create_task(_broadcast())
        else:
            asyncio.run(_broadcast())

    def _broadcast_training_progress(self, job_id: str | None, progress_state: Mapping[str, Any]) -> None:
        """Broadcast real-time training progress updates via SSE."""
        payload = {
            "type": "training.progress",
            "job_id": job_id,
            "percent": progress_state.get("percent", 0),
            "step": progress_state.get("step", 0),
            "total_steps": progress_state.get("total_steps", 0),
            "epoch": progress_state.get("epoch", 0),
            "total_epochs": progress_state.get("total_epochs", 0),
            "loss": progress_state.get("loss"),
            "learning_rate": progress_state.get("learning_rate"),
        }

        # Include any additional metrics if present
        if "metrics" in progress_state and isinstance(progress_state["metrics"], Mapping):
            payload["metrics"] = progress_state["metrics"]

        logger.debug(
            f"Broadcasting training progress: step {payload['step']}/{payload['total_steps']} "
            f"({payload['percent']:.1f}%) for job {job_id}"
        )

        async def _broadcast():
            manager = get_sse_manager()
            await manager.broadcast(payload, event_type="training.progress")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.create_task(_broadcast())
        else:
            asyncio.run(_broadcast())

    def _should_suppress_event(self, event: CallbackEvent) -> bool:
        if not event:
            return False

        job_id = self._derive_job_id(event)
        if not job_id:
            return False

        status = self._job_status.get(job_id, "")
        if status in {"failed", "error", "completed", "cancelled", "stopped"}:
            if event.type in {EventType.TRAINING_PROGRESS, EventType.LIFECYCLE_STAGE}:
                return True
            if event.type == EventType.TRAINING_STATUS:
                return True
        return False

    def _mirror_to_process_keeper(self, event: CallbackEvent) -> None:
        if not event or not event.job_id:
            return

        if event.type not in {EventType.NOTIFICATION, EventType.ERROR, EventType.DEBUG}:
            return

        message = event.message or event.title
        if not message:
            return

        if isinstance(event.severity, EventSeverity):
            severity = event.severity.value
        else:
            severity = str(event.severity or "")

        append_external_event(
            event.job_id,
            {
                "type": event.type.value,
                "message": message,
                "severity": severity,
                "timestamp": getattr(event, "timestamp", None) or time.time(),
                "data": event.data or {},
            },
        )

    def _derive_job_id(self, event: CallbackEvent) -> str | None:
        if not event:
            return None

        candidates: list[str | None] = [
            event.job_id,
        ]

        if isinstance(event.data, Mapping):
            candidates.append(event.data.get("job_id"))

        if isinstance(event.raw, Mapping):
            candidates.append(event.raw.get("job_id"))
            raw_progress = event.raw.get("progress")
            if isinstance(raw_progress, Mapping):
                candidates.append(raw_progress.get("job_id"))

        stage = event.stage
        if stage and isinstance(stage.metadata, Mapping):
            candidates.append(stage.metadata.get("job_id"))

        progress = event.progress
        if progress and isinstance(progress.metrics, Mapping):
            candidates.append(progress.metrics.get("job_id"))

        current_job = APIState.get_state("current_job_id")
        if current_job:
            candidates.append(current_job)

        for candidate in candidates:
            if isinstance(candidate, str):
                normalized = candidate.strip()
                if normalized:
                    return normalized
        return None


__all__ = ["CallbackService"]


def get_default_callback_service() -> CallbackService:
    """Lazily create a shared CallbackService backed by the default event store."""
    global _default_service
    if _default_service is not None:
        return _default_service
    with _service_lock:
        if _default_service is None:
            from .event_store import get_default_store

            _default_service = CallbackService(get_default_store())
        return _default_service


def _maybe_int(value: Any) -> int | None:
    numeric = _maybe_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _maybe_float(value: Any) -> float | None:
    if value in (None, "", False):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:  # NaN
        return None
    return numeric


def _prettify(value: str) -> str:
    return " ".join(part.capitalize() for part in value.replace("_", " ").split())
