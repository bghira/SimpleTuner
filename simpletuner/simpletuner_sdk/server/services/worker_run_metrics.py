"""Recent training samples from authenticated worker progress reports."""

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from time import monotonic

from .callback_events import ProgressData, parse_number
from .cloud.base import UnifiedJob

WORKER_PROVIDERS = {"worker", "kubeflow"}
ACTIVE_STATUSES = {"starting", "training", "running"}
_METRIC_NAMES = (
    "loss",
    "parent_loss",
    "learning_rate",
    "epoch",
    "steps_per_second",
    "samples_per_second",
    "step_speed_seconds",
    "effective_batch_size",
)
_SAMPLE_INTERVAL = 5
_HISTORY_SECONDS = 30 * 60
_MAX_RUNS = 64


@dataclass
class _RunSeries:
    assignment: tuple[str | None, str | None]
    worker_name: str
    samples: deque = field(default_factory=lambda: deque(maxlen=_HISTORY_SECONDS // _SAMPLE_INTERVAL))


class WorkerRunMetrics:
    def __init__(self):
        self._runs: OrderedDict[str, _RunSeries] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def _assignment(job: UnifiedJob) -> tuple[str | None, str | None]:
        # Dispatch updates started_at. A reconnect for the same assignment keeps it.
        return (job.metadata or {}).get("worker_id"), job.started_at

    def record(self, job: UnifiedJob, worker_name: str, payload: dict) -> None:
        if job.provider not in WORKER_PROVIDERS or job.status not in ACTIVE_STATUSES:
            return
        progress = ProgressData.from_mapping(payload)
        if progress is None or progress.current is None or progress.current <= 0 or not progress.current.is_integer():
            return
        metrics = {key: value for key in _METRIC_NAMES if (value := parse_number(progress.metrics.get(key))) is not None}
        if not metrics:
            return

        step = int(progress.current)
        assignment = self._assignment(job)
        with self._lock:
            now = monotonic()
            series = self._runs.get(job.job_id)
            if series is None or series.assignment != assignment:
                series = self._runs[job.job_id] = _RunSeries(assignment, worker_name)
            if series.samples:
                last_time, last = series.samples[-1]
                if now - last_time < _SAMPLE_INTERVAL or step < last["step"]:
                    return
                if step == last["step"]:
                    series.samples.pop()
            series.samples.append(
                (now, {"step": step, "timestamp": datetime.now(timezone.utc).isoformat(), "metrics": metrics})
            )
            self._runs.move_to_end(job.job_id)
            if len(self._runs) > _MAX_RUNS:
                self._runs.popitem(last=False)

    def job_ids(self) -> list[str]:
        cutoff = monotonic() - _HISTORY_SECONDS
        with self._lock:
            return [job_id for job_id, series in self._runs.items() if series.samples[-1][0] >= cutoff]

    def snapshot(self, job: UnifiedJob) -> tuple[str, list[dict]] | None:
        cutoff = monotonic() - _HISTORY_SECONDS
        with self._lock:
            series = self._runs.get(job.job_id)
            if series is None or series.assignment != self._assignment(job):
                return None
            records = [point for sampled_at, point in series.samples if sampled_at >= cutoff]
            return (series.worker_name, records) if records else None


WORKER_RUN_METRICS = WorkerRunMetrics()
