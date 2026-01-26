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


def normalize_end_epoch(value: Any) -> int | None:
    """Parse end_epoch: returns int >= 1 or None for infinite."""
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    # 0 or negative means no end (infinite)
    return parsed if parsed >= 1 else None


def normalize_end_step(value: Any) -> int | None:
    """Parse end_step: returns int >= 1 or None for infinite."""
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    # 0 or negative means no end (infinite)
    return parsed if parsed >= 1 else None


def _extract_backend_config(backend_id: str, backend: Any) -> Dict[str, Any]:
    if isinstance(backend, dict):
        return backend
    config = getattr(backend, "config", None)
    if isinstance(config, dict):
        return config
    return StateTracker.get_data_backend_config(backend_id) or {}


def _next_optimizer_step(step_hint: int | None = None) -> int:
    """Return the next optimizer step number (1-based).

    If step_hint is provided, it's used as the current step value.
    Otherwise, StateTracker.get_global_step() is used.
    """
    if step_hint is not None:
        return step_hint + 1
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
    schedule_metadata contains normalized start/end epoch/step and activation state.

    A dataset is active when:
    - current_epoch >= start_epoch AND next_step >= start_step (has started)
    - AND (end_epoch is None OR current_epoch <= end_epoch) (hasn't ended by epoch)
    - AND (end_step is None OR next_step <= end_step) (hasn't ended by step)
    """
    backend_config = _extract_backend_config(backend_id, backend)
    start_epoch = normalize_start_epoch(backend_config.get("start_epoch", 1))
    start_step = normalize_start_step(backend_config.get("start_step", 0))
    end_epoch = normalize_end_epoch(backend_config.get("end_epoch"))
    end_step = normalize_end_step(backend_config.get("end_step"))

    try:
        current_epoch = int(epoch_hint if epoch_hint is not None else StateTracker.get_epoch() or 1)
    except Exception:
        current_epoch = 1

    next_step = _next_optimizer_step(step_hint)

    # Check start conditions
    has_started = current_epoch >= start_epoch and next_step >= start_step

    # Check end conditions (None means infinite, no end)
    not_ended_by_epoch = end_epoch is None or current_epoch <= end_epoch
    not_ended_by_step = end_step is None or next_step <= end_step

    is_active = has_started and not_ended_by_epoch and not_ended_by_step

    schedule_state = StateTracker.get_dataset_schedule(backend_id)
    if not schedule_state:
        StateTracker.set_dataset_schedule(
            backend_id,
            start_epoch=start_epoch,
            start_step=start_step,
            end_epoch=end_epoch,
            end_step=end_step,
        )
        schedule_state = StateTracker.get_dataset_schedule(backend_id)

    if is_active and update_state and not schedule_state.get("reached", False):
        StateTracker.mark_dataset_schedule_reached(
            backend_id,
            epoch=current_epoch,
            step=next_step,
        )
        schedule_state = StateTracker.get_dataset_schedule(backend_id)

    return is_active, schedule_state
