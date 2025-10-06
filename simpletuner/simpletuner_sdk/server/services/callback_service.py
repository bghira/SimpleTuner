"""Service layer responsible for normalising and persisting webhook callbacks."""

from __future__ import annotations

import copy
from collections import deque
from threading import Lock
from typing import Any, Iterable, Mapping, Sequence

from ...api_state import APIState
from .callback_events import CallbackCategory, CallbackEvent, CallbackSeverity
from .callback_registry import CallbackEventRegistry, default_callback_registry
from .event_store import EventStore

_default_service: CallbackService | None = None
_service_lock = Lock()


class CallbackService:
    """Wraps the event store to provide typed callback events and read APIs."""

    def __init__(
        self,
        event_store: EventStore,
        *,
        registry: CallbackEventRegistry | None = None,
    ) -> None:
        self._store = event_store
        self._registry = registry or default_callback_registry
        self._lock = Lock()
        self._typed_by_index: dict[int, CallbackEvent] = {}
        self._index_order: deque[int] = deque(maxlen=event_store.max_events)
        self._dedupe_cache: dict[str, int] = {}
        self._bootstrap_from_store()

    def handle_incoming(self, raw_payload: Mapping[str, Any]) -> CallbackEvent | None:
        """Normalise and persist a raw webhook payload."""
        if not isinstance(raw_payload, Mapping):
            raise TypeError("raw_payload must be a mapping")

        event = CallbackEvent.from_message(raw_payload, registry=self._registry)

        typed_event = None
        with self._lock:
            if event.reset_history:
                self._store.clear()
                self._typed_by_index.clear()
                self._index_order.clear()
                self._dedupe_cache.clear()

            if event.dedupe_key:
                existing_index = self._dedupe_cache.get(event.dedupe_key)
                if existing_index is not None:
                    existing_event = self._typed_by_index.get(existing_index)
                    if existing_event:
                        return existing_event

            record = self._prepare_record(raw_payload)
            index = self._store.add_event(record)
            typed_event = event.with_index(index)
            record["typed"] = typed_event.to_payload()
            record["category"] = typed_event.category.value
            record["severity"] = typed_event.severity.value
            record["timestamp"] = typed_event.timestamp.isoformat()

            self._typed_by_index[index] = typed_event
            self._append_index(index)

            if event.dedupe_key:
                self._dedupe_cache[event.dedupe_key] = index

        # Update training state outside the lock to avoid blocking
        if typed_event:
            self._update_training_state(typed_event)

        return typed_event

    def get_recent(self, limit: int = 10) -> list[CallbackEvent]:
        """Return the most recent *limit* events, newest first."""
        if limit <= 0:
            return []
        # Snapshot indices while holding lock, then release before iteration
        with self._lock:
            indices = list(self._index_order)[-limit:]
            snapshot = {idx: self._typed_by_index.get(idx) for idx in indices}
        # Iterate over snapshot without holding lock
        return [snapshot[idx] for idx in reversed(indices) if snapshot.get(idx) is not None]

    def stream_since(self, index: int) -> list[CallbackEvent]:
        """Return all events newer than the provided index."""
        # Snapshot indices and events while holding lock, then release before iteration
        with self._lock:
            matching_indices = [idx for idx in self._index_order if idx > index]
            snapshot = {idx: self._typed_by_index.get(idx) for idx in matching_indices}
        # Build result from snapshot without holding lock
        return [snapshot[idx] for idx in matching_indices if snapshot.get(idx) is not None]

    def latest_for_job(self, job_id: str | None) -> CallbackEvent | None:
        """Return the newest event for a specific job identifier."""
        if not job_id:
            return None
        # Snapshot indices while holding lock, then search without lock
        with self._lock:
            indices = list(reversed(self._index_order))
            snapshot = {idx: self._typed_by_index.get(idx) for idx in indices}
        # Search snapshot without holding lock
        for idx in indices:
            event = snapshot.get(idx)
            if event and event.job_id == job_id:
                return event
        return None

    def as_payloads(self, events: Sequence[CallbackEvent]) -> list[dict[str, Any]]:
        """Convert events into serialisable dictionaries for transport."""
        return [event.to_payload() for event in events]

    def _append_index(self, index: int) -> None:
        # Check if deque is at capacity and will evict
        if self._index_order.maxlen and len(self._index_order) == self._index_order.maxlen:
            oldest = self._index_order[0]
            # Deque will automatically evict oldest when we append
            self._index_order.append(index)
            # Clean up evicted index from tracking structures
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
            "message_type": payload_copy.get("message_type"),
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
        event = CallbackEvent.from_message(raw_payload, registry=self._registry).with_index(index)
        self._typed_by_index[index] = event
        if event.dedupe_key:
            self._dedupe_cache[event.dedupe_key] = index

        # Check if deque is at capacity and will evict
        if self._index_order.maxlen and len(self._index_order) == self._index_order.maxlen:
            oldest = self._index_order[0]
            # Deque will automatically evict oldest when we append
            self._index_order.append(index)
            # Clean up evicted index from tracking structures
            self._evict_index(oldest)
        else:
            self._index_order.append(index)

        # Ensure record has normalised typed payload for downstream consumers
        if isinstance(record, dict):
            record.setdefault("raw", event.raw)
            record.setdefault("typed", event.to_payload())
            record.setdefault("category", event.category.value)
            record.setdefault("severity", event.severity.value)
            record.setdefault("timestamp", event.timestamp.isoformat())

    def _evict_index(self, index: int) -> None:
        self._typed_by_index.pop(index, None)
        stale_keys = [key for key, value in self._dedupe_cache.items() if value == index]
        for key in stale_keys:
            self._dedupe_cache.pop(key, None)

    @staticmethod
    def _merge_progress_state(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}

        if isinstance(previous, Mapping):
            merged.update(previous)

        new_percent = CallbackService._clamp_percent(current.get("percent"))
        if new_percent is not None:
            prev_percent = CallbackService._clamp_percent(merged.get("percent")) or 0
            merged["percent"] = max(prev_percent, new_percent)
        elif "percent" not in merged:
            merged["percent"] = 0

        for key in ("step", "total_steps", "epoch", "total_epochs"):
            value = current.get(key)
            if value:
                merged[key] = value
            elif key not in merged:
                merged[key] = 0

        for key in ("loss", "learning_rate"):
            value = current.get(key)
            if value is not None:
                merged[key] = value
            elif key not in merged:
                merged[key] = None

        return merged

    @staticmethod
    def _extract_progress_from_extras(extras: Mapping[str, Any]) -> dict[str, Any] | None:
        if not isinstance(extras, Mapping):
            return None

        state = extras.get("state")
        if not isinstance(state, Mapping):
            return None

        percent = extras.get("percent") or state.get("percent")
        current_step = extras.get("current_step") or state.get("global_step")
        total_steps = extras.get("total_steps") or extras.get("total_num_steps") or state.get("max_steps")
        epoch = state.get("current_epoch")
        total_epochs = state.get("final_epoch")
        loss = extras.get("loss") or state.get("loss")
        learning_rate = extras.get("learning_rate") or extras.get("lr") or state.get("learning_rate")

        return {
            "percent": CallbackService._clamp_percent(percent) or 0,
            "step": current_step or 0,
            "total_steps": total_steps or 0,
            "epoch": epoch or 0,
            "total_epochs": total_epochs or 0,
            "loss": loss,
            "learning_rate": learning_rate,
        }

    @staticmethod
    def _clamp_percent(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not (numeric == numeric):  # NaN check
            return None
        return max(0.0, min(100.0, numeric))

    def _update_training_state(self, event: CallbackEvent) -> None:
        job_id = event.job_id
        message_type = (event.message_type or "").lower()
        previous_job_id = APIState.get_state("current_job_id")
        job_changed = bool(job_id and job_id != previous_job_id)

        if event.category == CallbackCategory.PROGRESS:
            APIState.set_state("training_status", "running")
            if job_id:
                APIState.set_state("current_job_id", job_id)

            progress_payload = event.progress
            if progress_payload:
                state = {}
                extras = dict(progress_payload.extra or {})
                if isinstance(extras.get("state"), dict):
                    state = extras["state"]

                current_step = progress_payload.current
                if current_step is None:
                    current_step = extras.get("current_step") or state.get("global_step")

                total_steps = progress_payload.total
                if total_steps is None:
                    total_steps = (
                        extras.get("total_steps")
                        or extras.get("total_num_steps")
                        or extras.get("max_steps")
                        or state.get("max_steps")
                    )

                epoch = extras.get("epoch") or state.get("current_epoch")
                total_epochs = extras.get("final_epoch") or extras.get("total_epochs") or state.get("final_epoch")

                loss = extras.get("loss") or extras.get("train_loss")
                learning_rate = extras.get("learning_rate") or extras.get("lr")

                progress_state = {
                    "percent": progress_payload.percent or 0,
                    "step": current_step or 0,
                    "total_steps": total_steps or 0,
                    "epoch": epoch or 0,
                    "total_epochs": total_epochs or 0,
                    "loss": loss,
                    "learning_rate": learning_rate,
                }

                previous_progress = APIState.get_state("training_progress") or {}
                if job_changed:
                    previous_progress = {}

                merged_progress = self._merge_progress_state(previous_progress, progress_state)
                APIState.set_state("training_progress", merged_progress)
            return

        # Handle startup messages - these always mean training is starting/running
        if message_type in {"configure_webhook", "_train_initial_msg"}:
            APIState.set_state("training_status", "running" if message_type != "configure_webhook" else "starting")
            if job_id:
                APIState.set_state("current_job_id", job_id)
            return

        # Handle train_status messages - derive status from payload, don't hardcode
        if message_type == "train_status":
            # Try to get status from extras, default to 'running'
            status_from_payload = event.extras.get("status", "running") if event.extras else "running"
            APIState.set_state("training_status", status_from_payload)

            if job_id:
                APIState.set_state("current_job_id", job_id)

            # Extract progress from event.progress (which now contains the training data)
            progress_state = None
            if event.progress:
                # Get progress data from the progress payload
                progress_extra = event.progress.extra or {}
                progress_state = {
                    "percent": event.progress.percent or 0,
                    "step": progress_extra.get("global_step") or 0,
                    "total_steps": progress_extra.get("total_num_steps") or 0,
                    "epoch": progress_extra.get("current_epoch") or progress_extra.get("epoch") or 0,
                    "total_epochs": progress_extra.get("final_epoch") or 0,
                    "loss": progress_extra.get("loss") or progress_extra.get("train_loss"),
                    "learning_rate": progress_extra.get("learning_rate") or progress_extra.get("lr"),
                }

            if progress_state:
                previous_progress = APIState.get_state("training_progress") or {}
                if job_changed:
                    previous_progress = {}
                merged_progress = self._merge_progress_state(previous_progress, progress_state)
                APIState.set_state("training_progress", merged_progress)
            # DON'T return - fall through to severity checks below

        if message_type in {"training_complete", "run_complete"}:
            progress_state = APIState.get_state("training_progress") or {}
            if isinstance(progress_state, dict):
                progress_state = dict(progress_state)
                progress_state["percent"] = 100
                APIState.set_state("training_progress", progress_state)
            APIState.set_state("training_status", "completed")
            APIState.set_state("current_job_id", None)
            return

        if message_type in {"exit", "fatal_error"} or event.severity in {CallbackSeverity.ERROR, CallbackSeverity.CRITICAL}:
            APIState.set_state("training_status", "error")
            APIState.set_state("current_job_id", None)
            return

        if event.category == CallbackCategory.ALERT and event.severity == CallbackSeverity.WARNING:
            APIState.set_state("training_status", "warning")


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


__all__.append("get_default_callback_service")
