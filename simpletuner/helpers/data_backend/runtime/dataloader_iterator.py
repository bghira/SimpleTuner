"""Dataloader iterator functions for multi-backend sampling."""

import logging
import os
from typing import Any, Dict, Optional, Union

import torch

from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger("DataBackendFactory")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel(logging.ERROR)

prefetch_log = logging.getLogger("DataBackendPrefetch")
if should_log():
    prefetch_log.setLevel(os.environ.get("SIMPLETUNER_PREFETCH_LOG_LEVEL", "INFO"))
else:
    prefetch_log.setLevel(logging.ERROR)

_SCALED_SAMPLERS: dict[int, "ScaledDatasetSampler"] = {}


def prefetch_log_debug(message: str) -> None:
    from simpletuner.helpers.training.multi_process import rank_info

    prefetch_log.debug(f"{rank_info()} {message}")


def select_dataloader_index(step: int, backends: Dict[str, Any]) -> Optional[str]:
    if not backends:
        raise ValueError("No data backends available for selection.")

    weights = []
    backend_ids = []
    for backend_id, backend in backends.items():
        weight = get_backend_weight(backend_id, backend, step)
        weights.append(weight)
        backend_ids.append(backend_id)

    if not backend_ids:
        raise ValueError("No data backends available for selection.")

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    total = weights_tensor.sum()
    if total <= 0:
        raise ValueError("All backend sampling weights are zero.")
    weights_tensor /= total

    chosen_index = torch.multinomial(weights_tensor, 1).item()
    chosen_backend_id = backend_ids[chosen_index]

    return chosen_backend_id


def get_backend_weight(backend_id: str, backend: Any, step: int) -> float:
    if isinstance(backend, dict):
        backend_config = backend
    else:
        backend_config = getattr(backend, "config", None)
        if not isinstance(backend_config, dict):
            backend_config = StateTracker.get_data_backend_config(backend_id)
    prob = backend_config.get("probability", 1)

    sampling_args = StateTracker.get_args()
    sampling_method = getattr(sampling_args, "data_backend_sampling", "uniform") if sampling_args is not None else "uniform"
    if not isinstance(sampling_method, str):
        sampling_method = "uniform"

    if sampling_method == "uniform":
        return prob
    elif sampling_method == "auto-weighting":
        dataset_length = StateTracker.get_dataset_size(backend_id)

        try:
            total = sum(
                StateTracker.get_dataset_size(b) for b in StateTracker.get_data_backends(_types=["image", "video", "audio"])
            )
        except Exception:
            total = 0
        if not total:
            total = dataset_length or 1
        length_factor = dataset_length / total

        adjusted_prob = prob * length_factor

        disable_step = backend_config.get("disable_after_epoch_step", None)
        if disable_step:
            disable_step = int(disable_step)
        else:
            disable_step = float("inf")
        adjusted_prob = 0 if int(step) > disable_step else max(0, adjusted_prob * (1 - step / disable_step))

        return adjusted_prob
    else:
        raise ValueError(f"Unknown sampling weighting method: {sampling_method}")


def _read_slider_strength(backend_id: str, backend: Any, slider_key: str = "slider_strength") -> Optional[float]:
    """
    Attempt to read a numeric slider strength from the backend or its config.
    """
    strength = None
    candidate = None
    if isinstance(backend, dict):
        candidate = backend.get(slider_key, None)
    else:
        candidate = getattr(backend, slider_key, None)
        if candidate is None:
            backend_config = getattr(backend, "config", None)
            if isinstance(backend_config, dict):
                candidate = backend_config.get(slider_key, None)
            else:
                backend_config = StateTracker.get_data_backend_config(backend_id)
                if isinstance(backend_config, dict):
                    candidate = backend_config.get(slider_key, None)

    if candidate is None:
        return None
    try:
        strength = float(candidate)
    except (TypeError, ValueError):
        return None
    return strength


class ScaledDatasetSampler:
    """
    Alternates sampling across positive, negative, and unscaled (neutral) slider groups.
    Selection within a group still honours backend probabilities via get_backend_weight.
    """

    def __init__(self, slider_key: str = "slider_strength") -> None:
        self.slider_key = slider_key
        self._groups: dict[str, list[str]] = {"positive": [], "negative": [], "neutral": []}
        self._phase_index: int = 0
        self._has_scaled_groups: bool = False

    def rebuild(self, backends: Dict[str, Any]) -> None:
        self._groups = {"positive": [], "negative": [], "neutral": []}
        for backend_id, backend in backends.items():
            strength = _read_slider_strength(backend_id, backend, self.slider_key)
            if strength is None:
                self._groups["neutral"].append(backend_id)
            elif strength < 0:
                self._groups["negative"].append(backend_id)
            elif strength > 0:
                self._groups["positive"].append(backend_id)
            else:
                self._groups["neutral"].append(backend_id)
        self._has_scaled_groups = bool(self._groups["positive"] or self._groups["negative"])
        order = self._phase_order()
        if not order:
            self._phase_index = 0
        else:
            self._phase_index = self._phase_index % len(order)

    def has_scaled_groups(self) -> bool:
        return self._has_scaled_groups

    def remove_backend(self, backend_id: str) -> None:
        for group in self._groups.values():
            if backend_id in group:
                group.remove(backend_id)
        self._groups = {k: v for k, v in self._groups.items()}
        self._has_scaled_groups = bool(self._groups["positive"] or self._groups["negative"])
        if self._phase_index >= len(self._phase_order()):
            self._phase_index = 0

    def _phase_order(self) -> list[str]:
        # Fixed order with empty groups removed.
        return [phase for phase in ("positive", "negative", "neutral") if self._groups.get(phase)]

    def _choose_from_group(self, group: str, step: int, backends: Dict[str, Any]) -> Optional[str]:
        backend_ids = [backend_id for backend_id in self._groups.get(group, []) if backend_id in backends]
        if not backend_ids:
            return None
        weights: list[float] = []
        for backend_id in backend_ids:
            try:
                weights.append(get_backend_weight(backend_id, backends[backend_id], step))
            except Exception:
                weights.append(0.0)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        total = weights_tensor.sum()
        if total <= 0:
            return backend_ids[0]
        weights_tensor /= total
        chosen_index = torch.multinomial(weights_tensor, 1).item()
        return backend_ids[chosen_index]

    def next_backend_id(self, step: int, backends: Dict[str, Any]) -> Optional[str]:
        """
        Return the next backend id following the fixed positive -> negative -> neutral cycle.
        """
        phases = self._phase_order()
        if not phases:
            return None
        attempts = 0
        while attempts < len(phases):
            phase = phases[self._phase_index % len(phases)]
            candidate = self._choose_from_group(phase, step, backends)
            self._phase_index = (self._phase_index + 1) % len(phases)
            if candidate is not None:
                return candidate
            attempts += 1
        return None


def _get_scaled_sampler(backends: Dict[str, Any]) -> Optional[ScaledDatasetSampler]:
    """
    Lazily build and cache a ScaledDatasetSampler keyed by the backends object's id.
    """
    if not backends:
        return None
    key = id(backends)
    sampler = _SCALED_SAMPLERS.get(key)
    if sampler is None:
        sampler = ScaledDatasetSampler()
        _SCALED_SAMPLERS[key] = sampler
    sampler.rebuild(backends)
    if not any(sampler._groups.values()):
        return None
    return sampler


def random_dataloader_iterator(step: int, backends: Dict[str, Any]) -> Union[Any, bool]:
    if not backends:
        raise ValueError("No data backends provided to iterator.")

    prefetch_log_debug("Random dataloader iterator launched.")
    args = StateTracker.get_args()
    grad_steps = getattr(args, "gradient_accumulation_steps", 1) if args is not None else 1
    if isinstance(grad_steps, (int, float)):
        gradient_accumulation_steps = max(1, int(grad_steps))
    else:
        gradient_accumulation_steps = 1
    logger.debug(f"Backends to select from {backends}")
    while backends:
        epoch_step = int(step / gradient_accumulation_steps)
        StateTracker.set_epoch_step(epoch_step)

        sampler = _get_scaled_sampler(backends)
        chosen_backend_id = None
        if sampler and sampler.has_scaled_groups():
            chosen_backend_id = sampler.next_backend_id(step, backends)
        if chosen_backend_id is None:
            chosen_backend_id = select_dataloader_index(step, backends)
        backend = backends.get(chosen_backend_id)
        if backend is None:
            raise KeyError(f"Selected backend {chosen_backend_id} not found.")

        try:
            if hasattr(backend, "get_data_loader"):
                data_loader = backend.get_data_loader()
                return data_loader

            data_iterator = iter(backend)
            return next(data_iterator)
        except MultiDatasetExhausted:
            repeats = StateTracker.get_data_backend_config(chosen_backend_id).get("repeats", False)
            if repeats and repeats > 0 and StateTracker.get_repeats(chosen_backend_id) < repeats:
                StateTracker.increment_repeats(chosen_backend_id)
                logger.debug(
                    f"Dataset (name={chosen_backend_id}) is now sampling its {StateTracker.get_repeats(chosen_backend_id)} repeat out of {repeats} total allowed."
                )
                continue
            logger.debug(
                f"Dataset (name={chosen_backend_id}) is now exhausted after {StateTracker.get_repeats(chosen_backend_id)} repeat(s). Removing from list."
            )
            del backends[chosen_backend_id]
            StateTracker.backend_exhausted(chosen_backend_id)
            StateTracker.set_repeats(data_backend_id=chosen_backend_id, repeats=0)
            if sampler:
                sampler.remove_backend(chosen_backend_id)
        finally:
            if not backends:
                logger.debug("All dataloaders exhausted. Moving to next epoch in main training loop.")
                StateTracker.clear_exhausted_buckets()
                _SCALED_SAMPLERS.pop(id(backends), None)
                return False
