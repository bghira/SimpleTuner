"""Service layer responsible for normalising and persisting webhook callbacks."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any, Mapping, Sequence

from ...api_state import APIState
from ...process_keeper import append_external_event
from .callback_events import CallbackEvent, EventSeverity, EventType, ProgressData, StageData, StageStatus, parse_number
from .cloud.auth.models import User
from .event_store import EventStore
from .sse_manager import get_sse_manager

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "success", "failed", "error", "fatal", "crashed", "cancelled", "stopped", "terminated"}


def _update_job_store_status(job_id: str | None, status: str) -> None:
    """Update job status in the unified JobRepository.

    This is called when training completes or fails via webhook callbacks,
    ensuring job status is properly reflected in the Job Queue.
    """
    logger.info("_update_job_store_status called with job_id=%s, status=%s", job_id, status)
    if not job_id:
        logger.warning("_update_job_store_status: no job_id provided, skipping")
        return

    try:
        from .cloud.base import CloudJobStatus, JobType
        from .cloud.storage.job_repository import get_job_repository

        job_repo = get_job_repository()

        # Map callback status to CloudJobStatus
        if status in {"completed", "success"}:
            new_status = CloudJobStatus.COMPLETED.value
        elif status in {"failed", "error", "fatal"}:
            new_status = CloudJobStatus.FAILED.value
        elif status in {"cancelled", "stopped"}:
            new_status = CloudJobStatus.CANCELLED.value
        else:
            return  # Don't update for other statuses

        async def _do_updates():
            logger.info("_do_updates starting for job %s (status: %s)", job_id, new_status)
            try:
                # Get job to check if it's local and current state
                job = await job_repo.get(job_id)
                if not job:
                    logger.warning("_do_updates: job %s not found in repo", job_id)
                    return

                logger.info("_do_updates: job %s is type %s, current status %s", job_id, job.job_type, job.status)

                # Check if job is already in a terminal state (idempotency check)
                # This prevents race conditions with the cancellation API
                terminal_states = {
                    CloudJobStatus.COMPLETED.value,
                    CloudJobStatus.FAILED.value,
                    CloudJobStatus.CANCELLED.value,
                }
                if job.status in terminal_states:
                    logger.info("_do_updates: job %s already in terminal state %s, skipping", job_id, job.status)
                    return

                # Update job status
                if new_status == CloudJobStatus.COMPLETED.value:
                    await job_repo.mark_completed(job_id)
                elif new_status == CloudJobStatus.FAILED.value:
                    await job_repo.mark_failed(job_id, "Job failed")
                elif new_status == CloudJobStatus.CANCELLED.value:
                    await job_repo.mark_cancelled(job_id)

                # Release GPUs and process pending jobs for local jobs
                if (
                    job.job_type == JobType.LOCAL
                    and not (job.metadata or {}).get("worker_id")
                    and job.provider not in {"worker", "kubeflow"}
                ):
                    from .local_gpu_allocator import get_gpu_allocator

                    allocator = get_gpu_allocator()
                    logger.info("_do_updates: releasing GPUs for job %s", job_id)
                    await allocator.release(job_id)
                    # Process pending jobs to start the next one if GPUs are available
                    logger.info("_do_updates: calling process_pending_jobs")
                    started = await allocator.process_pending_jobs()
                    if started:
                        logger.info("Started %d pending jobs after job %s completed", len(started), job_id)
                    else:
                        logger.info("No pending jobs were started after job %s completed", job_id)
            except Exception as exc:
                logger.error("_do_updates failed for job %s: %s", job_id, exc, exc_info=True)

        try:
            loop = asyncio.get_running_loop()
            logger.info("Scheduling _do_updates task for job %s", job_id)
            loop.create_task(_do_updates())
        except RuntimeError:
            # No running event loop, use thread
            import threading

            def _update():
                try:
                    asyncio.run(_do_updates())
                except Exception as thread_exc:
                    # Best-effort background update - log the failure but don't propagate.
                    # During testing, the database or logger may be unavailable during cleanup,
                    # so we wrap logging in its own try/except to prevent traceback noise.
                    try:
                        logger.debug("Background job update failed: %s", thread_exc)
                    except Exception:
                        pass

            threading.Thread(target=_update, daemon=True).start()
    except Exception as exc:
        logger.debug("Failed to update job status in JobRepository: %s", exc)


@lru_cache(maxsize=1024)
def _job_context(job_id: str) -> dict[str, Any]:
    """Resolve display metadata once per job, including the assigned worker."""
    from .cloud.storage.job_repository import get_job_repository
    from .worker_repository import get_worker_repository

    async def _get_job():
        job = await get_job_repository().get(job_id)
        if not job:
            return {}
        metadata = job.metadata or {}
        if job.provider == "captionflow" or metadata.get("handler") == "captionflow":
            return {"captionflow": True}
        worker_id = metadata.get("worker_id")
        worker = await get_worker_repository().get_worker(worker_id) if worker_id else None
        return {
            "config_name": job.config_name,
            "user_id": job.user_id,
            "assigned_worker": bool(worker_id or job.provider in {"worker", "kubeflow"}),
            "local_worker": job.provider == "local" and not worker_id,
            "worker": worker.name if worker else worker_id,
            "gpu": (worker.gpu_info or {}).get("name") if worker else job.hardware_type,
        }

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            job = asyncio.run(_get_job())
        else:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                job = pool.submit(lambda: asyncio.run(_get_job())).result(timeout=5)

        return job
    except Exception:
        logger.debug("Failed to inspect job metadata for %s", job_id, exc_info=True)
        return {}


_default_service: CallbackService | None = None
_service_lock = Lock()


@dataclass
class _TrainingUpdate:
    status: str
    progress: dict | None = None
    stage: dict | None = None


class CallbackService:
    """Wraps the event store to provide typed callback events and read APIs."""

    def __init__(self, event_store: EventStore) -> None:
        self._store = event_store
        self._lock = Lock()
        self._typed_by_index: dict[int, CallbackEvent] = {}
        self._index_order: deque[int] = deque(maxlen=event_store.max_events)
        self._job_status: dict[str, str] = {}
        self._runs: dict[str, dict[str, Any]] = {}
        self._runs_revision = 0
        self._bootstrap_from_store()

    def handle_incoming(
        self, raw_payload: Mapping[str, Any], *, assigned_worker: Mapping[str, Any] | None = None
    ) -> CallbackEvent | None:
        """Normalise and persist a raw webhook payload."""
        if not isinstance(raw_payload, Mapping):
            raise TypeError("raw_payload must be a mapping")

        normalized_payload = dict(raw_payload)
        message_type = str(normalized_payload.get("message_type") or normalized_payload.get("type") or "").lower()

        if message_type == "configure_webhook":
            normalized_payload.setdefault("reset_history", True)

        event = CallbackEvent.from_message(normalized_payload)

        status = self._training_status(event)
        context = assigned_worker or {}
        if assigned_worker is None and event.job_id and status:
            context = _job_context(event.job_id)
        worker_terminal = assigned_worker is not None and status in TERMINAL_STATUSES

        with self._lock:
            if self._should_suppress_event(event) and not worker_terminal:
                return None
            if event.reset_history:
                for index in self._index_order:
                    self._evict_index(index)
                self._store.clear()
                self._typed_by_index.clear()
                self._index_order.clear()
            record = self._prepare_record(normalized_payload)
            index = self._store.add_event(record)
            typed_event = event.with_index(index)
            record["typed"] = typed_event.to_payload()
            record["type"] = typed_event.type.value
            record["severity"] = typed_event.severity.value
            record["timestamp"] = typed_event.timestamp.isoformat()

            self._typed_by_index[index] = typed_event
            self._append_index(index)

            update = self._update_training_state(typed_event, status, context, worker_terminal=worker_terminal)
        if update and assigned_worker is None:
            self._publish_training_update(typed_event, update, context)
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
        if event.job_id and (status := self._training_status(event)):
            self._update_training_state(event, status, _job_context(event.job_id), replay=True)
        self._append_index(index)

        if isinstance(record, dict):
            record.setdefault("raw", event.raw)
            record.setdefault("typed", event.to_payload())
            record.setdefault("type", event.type.value)
            record.setdefault("severity", event.severity.value)
            record.setdefault("timestamp", event.timestamp.isoformat())

    def _evict_index(self, index: int) -> None:
        event = self._typed_by_index.pop(index, None)
        if event and (run := self._runs.get(event.job_id)) and run.get("terminal_index") == index:
            del self._runs[event.job_id]
            self._runs_revision += 1

    @staticmethod
    def _merge_progress_state(
        previous: Mapping[str, Any] | None,
        progress: ProgressData,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = dict(previous) if isinstance(previous, Mapping) else {}

        if progress.current is not None and progress.current < merged.get("step", 0):
            return merged
        percent = progress.normalized_percent
        if percent is None and progress.current is not None and merged.get("total_steps"):
            percent = min(100, 100 * progress.current / merged["total_steps"])
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

        loss = parse_number(metrics.pop("loss", None))
        if loss is not None:
            merged["loss"] = loss
        else:
            merged.setdefault("loss", None)

        # Pop both values to prevent lr from being included in merged["metrics"]
        # Using separate pops instead of `or` to ensure both are removed regardless of values
        learning_rate_val = parse_number(metrics.pop("learning_rate", None))
        lr_val = parse_number(metrics.pop("lr", None))
        lr = learning_rate_val if learning_rate_val is not None else lr_val
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

    @staticmethod
    def _stage_state(stage: StageData) -> dict[str, Any] | None:
        if stage.key.startswith("stage_") and stage.key[6:].isdigit():
            return None

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

        return state

    @staticmethod
    def _extract_event_status(event: CallbackEvent) -> str | None:
        status = None
        if isinstance(event.data, Mapping):
            status = event.data.get("status") or event.data.get("state")
        if not status and isinstance(event.raw, Mapping):
            status = event.raw.get("status") or event.raw.get("state")
        if isinstance(status, str):
            normalized = status.strip().lower()
            return normalized or None
        return None

    @staticmethod
    def _training_status(event: CallbackEvent) -> str | None:
        status = CallbackService._extract_event_status(event)
        if event.stage:
            status = "completed" if event.stage.key == "training_complete" else "starting"
        elif event.type == EventType.TRAINING_SUMMARY:
            status = "completed"
        elif event.type == EventType.ERROR or (
            event.severity in {EventSeverity.ERROR, EventSeverity.CRITICAL}
            and status in {"failed", "error", "fatal", "crashed"}
        ):
            status = "error"
        elif event.type == EventType.TRAINING_STATUS:
            status = status or str(event.message or event.title or "running").lower()
        elif event.type == EventType.TRAINING_PROGRESS or (event.type == EventType.NOTIFICATION and event.progress):
            status = "running"
        elif event.type in {EventType.CHECKPOINT, EventType.VALIDATION, EventType.VALIDATION_IMAGE}:
            status = "running"
        else:
            return None
        return status

    @staticmethod
    def _update_validation(run: dict, event: CallbackEvent, training_resumed: bool, terminal: bool) -> None:
        timestamp = event.timestamp.timestamp()
        if not terminal and timestamp < run.get("validation_at", 0):
            return
        if event.stage and event.stage.key == "validation":
            active = event.stage.status in {StageStatus.PENDING, StageStatus.RUNNING}
            progress = event.stage.progress or ProgressData()
        elif event.type == EventType.VALIDATION_IMAGE:
            active = True
            # Older workers send the sampling total only in the displayed step label.
            total = event.data.get("total_steps")
            if total is None:
                total = str(event.data.get("step_label", "")).partition("/")[2]
            progress = ProgressData(current=parse_number(event.data.get("step")), total=parse_number(total))
        elif (terminal or training_resumed) and run.get("validation") is not None:
            active = False
        else:
            return
        run["validation_at"] = timestamp
        run["validation"] = (
            {"current": progress.current, "total": progress.total, "percent": progress.normalized_percent}
            if active
            else None
        )

    def _update_training_state(
        self,
        event: CallbackEvent,
        status: str | None,
        context: Mapping[str, Any],
        *,
        replay: bool = False,
        worker_terminal: bool = False,
    ) -> _TrainingUpdate | None:
        job_id = event.job_id
        if not job_id or not status:
            return None
        if self._job_status.get(job_id) in TERMINAL_STATUSES:
            if not worker_terminal and not (replay and status in TERMINAL_STATUSES):
                return None
        if context.get("captionflow"):
            return _TrainingUpdate(status)

        run = self._runs.setdefault(job_id, {"job_id": job_id})
        run.update(context)
        local = job_id == APIState.get_state("current_job_id") and not run.get("assigned_worker")
        if local:
            run["local"] = True
        if not context.get("worker") and isinstance(event.data.get("worker"), str) and event.data["worker"]:
            run["worker"] = event.data["worker"]
        if (local or context.get("local_worker")) and not run.get("worker"):
            run["worker"] = "Local"
        if event.stage and status == "starting":
            status = self._job_status.get(job_id, status)

        progress = run.get("progress")
        previous_step = progress["step"] if progress else -1
        if (
            event.progress
            and not event.stage
            and event.type in {EventType.TRAINING_PROGRESS, EventType.TRAINING_STATUS, EventType.NOTIFICATION}
        ):
            progress = self._merge_progress_state(progress, event.progress, event.data)
            run["progress"] = progress
        terminal = status in TERMINAL_STATUSES
        self._update_validation(run, event, progress is not None and progress["step"] > previous_step, terminal)
        failed = terminal and status not in {"completed", "success"}
        if terminal:
            run["terminal_index"] = event.index
            if not failed and progress is not None:
                progress["percent"] = 100
        run["updated_at"] = event.timestamp.timestamp()
        run["status"] = self._job_status[job_id] = status
        self._runs_revision += 1

        stage = self._stage_state(event.stage) if event.stage else None
        if local and not replay and event.type != EventType.VALIDATION_IMAGE:
            APIState.set_state("training_status", status)
            if progress is not None or terminal:
                APIState.set_state("training_progress", None if failed else progress)
            if terminal or (status == "running" and not event.stage):
                APIState.set_state("training_startup_stages", {})
            if stage:
                stages = dict(APIState.get_state("training_startup_stages") or {})
                stages[event.stage.key] = stage
                APIState.set_state("training_startup_stages", stages)
            if terminal:
                APIState.set_state("current_job_id", None)
        return _TrainingUpdate(status, progress if event.progress and not event.stage else None, stage)

    def _publish_training_update(self, event: CallbackEvent, update: _TrainingUpdate, context: Mapping[str, Any]) -> None:
        job_id, status = event.job_id, update.status
        if context.get("captionflow"):
            if status in TERMINAL_STATUSES:
                _update_job_store_status(job_id, status)
            return
        if update.stage:
            self._broadcast_startup_stage(job_id, update.stage)
        if status in TERMINAL_STATUSES:
            if status not in {"completed", "success"}:
                self._broadcast_progress_reset(job_id, status=status)
                if event.type == EventType.ERROR or event.severity in {EventSeverity.CRITICAL, EventSeverity.ERROR}:
                    self._broadcast_error(event, job_id)
            # Registered workers finalize their own jobs (including artifact receipt).
            if not context.get("assigned_worker") and (event.type != EventType.TRAINING_STATUS or self.is_local_job(job_id)):
                _update_job_store_status(job_id, status)
        elif update.progress is not None:
            self._broadcast_training_progress(job_id, update.progress)

    def is_local_job(self, job_id: str | None) -> bool:
        if not job_id:
            return False
        run = self._runs.get(job_id, {})
        current = APIState.get_state("current_job_id")
        return not run.get("assigned_worker") and (job_id == current if current else bool(run.get("local")))

    def training_runs_since(self, revision: int, user: User | None = None) -> tuple[int, list[dict[str, Any]] | None]:
        with self._lock:
            if revision == self._runs_revision:
                return revision, None
            snapshots = []
            view_all = user is None or user.has_permission("job.view.all")
            for run in self._runs.values():
                if not view_all and run.get("user_id") != user.id:
                    continue
                progress = run.get("progress")
                if progress is None:
                    if run.get("validation") is None:
                        continue
                    progress = self._merge_progress_state(None, ProgressData())
                metrics = progress.get("metrics", {})
                stats = {key: progress[key] for key in ("step", "total_steps", "percent", "epoch", "loss", "learning_rate")}
                stats["metrics"] = {
                    key: value
                    for key in (
                        "step_speed_seconds",
                        "iteration_step_time_seconds",
                        "effective_batch_size",
                        "total_batch_size",
                        "steps_per_second",
                        "samples_per_second",
                    )
                    if (value := parse_number(metrics.get(key))) is not None
                }
                summary = {
                    key: run.get(key)
                    for key in (
                        "job_id",
                        "worker",
                        "gpu",
                        "config_name",
                        "updated_at",
                        "status",
                        "validation",
                        "validation_at",
                    )
                }
                snapshots.append({**summary, "progress": stats, "terminal": run["status"] in TERMINAL_STATUSES})
            return self._runs_revision, snapshots

    def _broadcast_startup_stage(self, job_id: str | None, stage_state: Mapping[str, Any]) -> None:
        payload = {
            "type": "lifecycle.stage",
            "local": self.is_local_job(job_id),
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

        manager = get_sse_manager()
        manager.broadcast_threadsafe(payload, event_type="lifecycle.stage")

    def _broadcast_progress_reset(self, job_id: str | None, *, status: str) -> None:
        payload = {
            "type": "training_progress",
            "local": self.is_local_job(job_id),
            "job_id": job_id,
            "status": status,
            "reset": True,
            "percent": 0,
            "step": 0,
            "total_steps": 0,
            "epoch": 0,
            "total_epochs": 0,
        }

        manager = get_sse_manager()
        manager.broadcast_threadsafe(payload, event_type="training_progress")

    def _broadcast_training_progress(self, job_id: str | None, progress_state: Mapping[str, Any]) -> None:
        """Broadcast real-time training progress updates via SSE."""
        payload = {
            "type": "training.progress",
            "local": self.is_local_job(job_id),
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

        manager = get_sse_manager()
        manager.broadcast_threadsafe(payload, event_type="training.progress")

    def _broadcast_error(self, event: CallbackEvent, job_id: str | None) -> None:
        """Broadcast error events immediately via SSE for fast delivery to WebUI."""
        payload = {
            "type": "error",
            "local": self.is_local_job(job_id),
            "job_id": job_id,
            "title": event.title,
            "message": event.message,
            "severity": event.severity.value if hasattr(event.severity, "value") else str(event.severity),
            "timestamp": event.timestamp.isoformat() if event.timestamp else None,
        }

        # Include traceback if available in the raw data
        if isinstance(event.raw, Mapping) and "traceback" in event.raw:
            payload["traceback"] = event.raw["traceback"]

        logger.info(f"Broadcasting error event for job {job_id}: {event.title or event.message}")

        manager = get_sse_manager()
        manager.broadcast_threadsafe(payload, event_type="error")

    def _should_suppress_event(self, event: CallbackEvent) -> bool:
        if not event:
            return False

        job_id = event.job_id
        if not job_id:
            return False

        status = self._job_status.get(job_id, "")
        if status in TERMINAL_STATUSES:
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

        # Skip log.message events - they're already displayed via the callback service
        # and mirroring them would cause duplicates in the WebUI
        raw_type = str(event.raw.get("type") or "").lower() if event.raw else ""
        if raw_type == "log.message":
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
    numeric = parse_number(value)
    if numeric is None:
        return None
    return int(numeric)


def _prettify(value: str) -> str:
    return " ".join(part.capitalize() for part in value.replace("_", " ").split())
