"""Track iteration timing information for ETA and rate calculations."""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Callable, Deque, Iterable, Sequence


class IterationTracker:
    """Collects per-step timing information to compute iteration speeds and ETAs."""

    DEFAULT_WINDOWS = (5, 15, 30, 60)

    def __init__(
        self,
        *,
        windows: Sequence[int] | None = None,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        window_candidates = windows or self.DEFAULT_WINDOWS
        cleaned_windows = sorted({int(value) for value in window_candidates if int(value) > 0})
        if not cleaned_windows:
            raise ValueError("IterationTracker requires at least one positive window size.")
        self._windows: tuple[int, ...] = tuple(cleaned_windows)
        self._max_window_seconds: float = float(self._windows[-1] * 60)
        self._history: Deque[tuple[float, float]] = deque()
        self._training_started_at: float | None = None
        self._last_timestamp: float | None = None
        self._latest_step_duration: float | None = None
        self._time_source: Callable[[], float] = time_source or time.monotonic

    def mark_start(self) -> None:
        """Record the timestamp when training begins."""
        if self._training_started_at is None:
            now = self._time_source()
            self._training_started_at = now
            self._last_timestamp = now

    def record_step(self, global_step: float | int) -> None:
        """Add a completed global step to the timing history."""
        now = self._time_source()
        if self._training_started_at is None:
            self._training_started_at = now
        if self._last_timestamp is not None:
            duration = now - self._last_timestamp
            if duration >= 0:
                self._latest_step_duration = duration
        self._last_timestamp = now
        self._history.append((now, float(global_step)))
        cutoff = now - self._max_window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _compute_window_rates(self) -> dict[int, float]:
        """Return iterations-per-minute metrics for the configured windows."""
        if not self._history:
            return {}
        latest_timestamp, latest_step = self._history[-1]
        rates: dict[int, float] = {}
        for window_minutes in self._windows:
            window_seconds = window_minutes * 60
            threshold = latest_timestamp - window_seconds
            anchor: tuple[float, float] | None = None
            for timestamp, recorded_step in self._history:
                if timestamp >= threshold:
                    anchor = (timestamp, recorded_step)
                    break
            if anchor is None:
                continue
            elapsed = latest_timestamp - anchor[0]
            step_delta = latest_step - anchor[1]
            if elapsed <= 0 or step_delta <= 0:
                continue
            rates[window_minutes] = step_delta / (elapsed / 60.0)
        return rates

    def _overall_rate(self) -> float | None:
        if not self._history or self._training_started_at is None:
            return None
        latest_timestamp, latest_step = self._history[-1]
        elapsed = latest_timestamp - self._training_started_at
        if elapsed <= 0 or latest_step <= 0:
            return None
        return latest_step / (elapsed / 60.0)

    def iteration_metrics(self) -> dict[str, float]:
        """Expose a metrics dictionary for logging and webhook payloads."""
        metrics: dict[str, float] = {}
        if self._latest_step_duration is not None:
            metrics["iteration_step_time_seconds"] = self._latest_step_duration
        for window_minutes, rate in self._compute_window_rates().items():
            metrics[f"iterations_per_minute_{window_minutes}m"] = rate
        overall_rate = self._overall_rate()
        if overall_rate is not None:
            metrics["iterations_per_minute_overall"] = overall_rate
        return metrics

    def estimate_eta(self, current_step: float | None, total_steps: float | None) -> float | None:
        """Estimate seconds remaining based on iteration speed."""
        if current_step is None or total_steps is None:
            return None
        steps_remaining = max(float(total_steps) - float(current_step), 0.0)
        if steps_remaining <= 0:
            return 0.0
        rates = self._compute_window_rates()
        steps_per_minute: float | None = None
        for window_minutes in sorted(self._windows, reverse=True):
            rate = rates.get(window_minutes)
            if rate:
                steps_per_minute = rate
                break
        if steps_per_minute is None:
            steps_per_minute = self._overall_rate()
        if steps_per_minute is None or steps_per_minute <= 0:
            return None
        steps_per_second = steps_per_minute / 60.0
        if steps_per_second <= 0:
            return None
        eta_seconds = steps_remaining / steps_per_second
        if math.isinf(eta_seconds) or eta_seconds != eta_seconds:
            return None
        return eta_seconds
