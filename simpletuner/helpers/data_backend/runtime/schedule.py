"""Helpers for dataset activation scheduling."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from simpletuner.helpers.training.state_tracker import StateTracker


def normalize_start_epoch(value: Any) -> int:
    """Clamp a user-supplied start_epoch to an int >= 1."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 1
    return max(parsed, 1)


def normalize_start_step(value: Any) -> int:
    """Clamp a user-supplied start_step to an int >= 0."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _extract_backend_config(backend_id: str, backend: Any) -> Dict[str, Any]:
    if isinstance(backend, dict):
        return backend
    config = getattr(backend, "config", None)
    if isinstance(config, dict):
        return config
    return StateTracker.get_data_backend_config(backend_id) or {}


def _next_optimizer_step(step_hint: int | None = None) -> int:
    """Return the next optimizer step number (1-based)."""
    try:
        current_step = int(StateTracker.get_global_step() or 0)
    except Exception:
        current_step = 0
    return current_step + 1


def dataset_is_active(
    backend_id: str,
    backend: Any,
    *,
    step_hint: int | None = None,
    epoch_hint: int | None = None,
    update_state: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Determine whether a dataset is eligible for sampling based on schedule.

    Returns (is_active, schedule_metadata).
    schedule_metadata contains normalized start_epoch/start_step and activation state.
    """
    backend_config = _extract_backend_config(backend_id, backend)
    start_epoch = normalize_start_epoch(backend_config.get("start_epoch", 1))
    start_step = normalize_start_step(backend_config.get("start_step", 0))

    try:
        current_epoch = int(epoch_hint if epoch_hint is not None else StateTracker.get_epoch() or 1)
    except Exception:
        current_epoch = 1

    next_step = _next_optimizer_step(step_hint)
    is_active = current_epoch >= start_epoch and next_step >= start_step

    schedule_state = StateTracker.get_dataset_schedule(backend_id)
    if not schedule_state:
        StateTracker.set_dataset_schedule(backend_id, start_epoch=start_epoch, start_step=start_step)
        schedule_state = StateTracker.get_dataset_schedule(backend_id)

    if is_active and update_state and not schedule_state.get("reached", False):
        StateTracker.mark_dataset_schedule_reached(
            backend_id,
            epoch=current_epoch,
            step=next_step,
        )
        schedule_state = StateTracker.get_dataset_schedule(backend_id)

    return is_active, schedule_state
