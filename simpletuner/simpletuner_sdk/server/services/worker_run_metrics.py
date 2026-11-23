"""Training history from authenticated worker progress reports."""

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
_STALE_AFTER_SECONDS = 30 * 60
_RECENT_SAMPLE_LIMIT = 360
_OLDER_SAMPLE_LIMIT = 360
_MAX_RUNS = 64


@dataclass
class _RunSeries:
    assignment: tuple[str | None, str | None]
    worker_name: str
    samples: deque[tuple[float, dict]] = field(default_factory=lambda: deque(maxlen=_RECENT_SAMPLE_LIMIT))
    older_samples: list[tuple[float, dict]] = field(default_factory=list)
    older_sample_interval: int = _SAMPLE_INTERVAL

    def append(self, sampled_at: float, record: dict) -> None:
        if len(self.samples) == _RECENT_SAMPLE_LIMIT:
            oldest = self.samples[0]
            if not self.older_samples or oldest[0] - self.older_samples[-1][0] >= self.older_sample_interval:
                self.older_samples.append(oldest)
                if len(self.older_samples) > _OLDER_SAMPLE_LIMIT:
                    self.older_samples = self.older_samples[::2]
                    self.older_sample_interval *= 2
        self.samples.append((sampled_at, record))


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
            series.append(now, {"step": step, "timestamp": datetime.now(timezone.utc).isoformat(), "metrics": metrics})
            self._runs.move_to_end(job.job_id)
            if len(self._runs) > _MAX_RUNS:
                self._runs.popitem(last=False)

    def job_ids(self) -> list[str]:
        cutoff = monotonic() - _STALE_AFTER_SECONDS
        with self._lock:
            return [job_id for job_id, series in self._runs.items() if series.samples[-1][0] >= cutoff]

    def snapshot(self, job: UnifiedJob) -> tuple[str, list[dict]] | None:
        cutoff = monotonic() - _STALE_AFTER_SECONDS
        with self._lock:
            series = self._runs.get(job.job_id)
            if series is None or series.assignment != self._assignment(job) or series.samples[-1][0] < cutoff:
                return None
            return series.worker_name, [point for _, point in (*series.older_samples, *series.samples)]


WORKER_RUN_METRICS = WorkerRunMetrics()
