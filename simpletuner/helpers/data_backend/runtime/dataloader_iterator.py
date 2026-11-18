"""Dataloader iterator functions for multi-backend sampling."""

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional, Union

import torch

from simpletuner.helpers.data_backend.runtime.schedule import dataset_is_active
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
_MAX_TORCH_SEED = 2**63 - 1


def prefetch_log_debug(message: str) -> None:
    from simpletuner.helpers.training.multi_process import rank_info

    prefetch_log.debug(f"{rank_info()} {message}")


def _set_epoch_step_for_training_step(step: int) -> None:
    args = StateTracker.get_args()
    grad_steps = getattr(args, "gradient_accumulation_steps", 1) if args is not None else 1
    if isinstance(grad_steps, (int, float)):
        gradient_accumulation_steps = max(1, int(grad_steps))
    else:
        gradient_accumulation_steps = 1
    StateTracker.set_epoch_step(int(step / gradient_accumulation_steps))


def _selection_generator(step: int) -> torch.Generator:
    args = StateTracker.get_args()
    base_seed = getattr(args, "seed", 0) if args is not None else 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed((int(base_seed or 0) + int(step)) % _MAX_TORCH_SEED)
    return generator


def select_dataloader_index(step: int, backends: Dict[str, Any], *, mixer=None) -> Optional[str]:
    if not backends:
        raise ValueError("No data backends available for selection.")

    weights = []
    backend_ids = []
    inactive_ids = []
    for backend_id in sorted(backends):
        backend = backends[backend_id]
        weight = get_backend_weight(backend_id, backend, step, mixer=mixer)
        if weight <= 0:
            inactive_ids.append(backend_id)
            continue
        weights.append(weight)
        backend_ids.append(backend_id)

    if not backend_ids:
        # No active datasets available for sampling
        # Check if any datasets have been exhausted - if so, this is a normal epoch end
        # If no datasets have been exhausted, this is a misconfiguration error
        exhausted = StateTracker.exhausted_backends if mixer is None else mixer.dataset_ids.difference(backends)
        if exhausted:
            return None
        next_step = (StateTracker.get_global_step() or 0) + 1 if mixer is None else mixer.optimizer_step + 1
        current_epoch = StateTracker.get_epoch() if mixer is None else mixer.epoch
        raise ValueError(
            f"No active datasets available for sampling at epoch {current_epoch}, step {next_step}. "
            "Adjust start_epoch/start_step so at least one dataset is active."
        )

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    total = weights_tensor.sum()
    if total <= 0:
        raise ValueError("All backend sampling weights are zero.")
    weights_tensor /= total

    chosen_index = torch.multinomial(weights_tensor, 1, generator=_selection_generator(step)).item()
    chosen_backend_id = backend_ids[chosen_index]

    return chosen_backend_id


def get_backend_weight(backend_id: str, backend: Any, step: int, *, mixer=None) -> float:
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

    epoch = StateTracker.get_epoch() if mixer is None else mixer.epoch
    optimizer_step = step if mixer is None else mixer.optimizer_step
    is_active, _ = dataset_is_active(
        backend_id, backend_config, step_hint=optimizer_step, epoch_hint=epoch, update_state=mixer is None
    )
    if not is_active:
        return 0.0

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
        try:
            disable_step = int(disable_step) if disable_step else None
        except (TypeError, ValueError):
            disable_step = None
        if disable_step is None:
            disable_step = float("inf")
        try:
            current_step = int(step)
        except Exception:
            current_step = 0
        adjusted_prob = 0 if current_step > disable_step else max(0, adjusted_prob * (1 - current_step / disable_step))

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
        for backend_id in sorted(backends):
            backend = backends[backend_id]
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

    def _choose_from_group(self, group: str, step: int, backends: Dict[str, Any], *, mixer=None) -> Optional[str]:
        backend_ids = [backend_id for backend_id in self._groups.get(group, []) if backend_id in backends]
        if not backend_ids:
            return None
        weights: list[float] = []
        filtered_ids: list[str] = []
        for backend_id in backend_ids:
            try:
                weight = get_backend_weight(backend_id, backends[backend_id], step, mixer=mixer)
            except Exception:
                weight = 0.0
            if weight <= 0:
                continue
            filtered_ids.append(backend_id)
            weights.append(weight)
        if not filtered_ids:
            return None
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        total = weights_tensor.sum()
        if total <= 0:
            return None
        weights_tensor /= total
        chosen_index = torch.multinomial(weights_tensor, 1, generator=_selection_generator(step)).item()
        return filtered_ids[chosen_index]

    def next_backend_id(self, step: int, backends: Dict[str, Any], *, mixer=None) -> Optional[str]:
        """
        Return the next backend id following the fixed positive -> negative -> neutral cycle.
        """
        phases = self._phase_order()
        if not phases:
            return None
        attempts = 0
        while attempts < len(phases):
            phase = phases[self._phase_index % len(phases)]
            candidate = self._choose_from_group(phase, step, backends, mixer=mixer)
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


class DatasetMixer:
    """Keep the fetched selection phase separate from the trainer's acknowledged frontier."""

    def __init__(self, backends, epoch, step, saved=None, *, optimizer_step=0, accumulated_microbatches=0):
        self.sampler = ScaledDatasetSampler()
        keys = (
            "probability",
            "slider_strength",
            "start_epoch",
            "end_epoch",
            "start_step",
            "end_step",
            "disable_after_epoch_step",
        )
        settings = {name: {key: dataset.config.get(key) for key in keys} for name, dataset in backends.items()}
        args = StateTracker.get_args()
        self.gradient_accumulation_steps = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
        identity = [settings, args.seed, getattr(args, "data_backend_sampling", "uniform"), self.gradient_accumulation_steps]
        self.revision = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.epoch = epoch
        self.step = step
        self.optimizer_step = optimizer_step
        self.accumulated_microbatches = accumulated_microbatches
        self.dataset_ids = set(backends)
        if saved is not None:
            if saved["version"] != 2 or saved["revision"] != self.revision or saved["epoch"] != epoch:
                raise ValueError("Dataset mixing resume requires the same datasets, epoch, weights and sampling settings.")
            self.step = saved["step"]
            self.optimizer_step = saved["optimizer_step"]
            self.accumulated_microbatches = saved["accumulated_microbatches"]
            for name in saved["exhausted_backends"]:
                del backends[name]
            self.sampler.rebuild(backends)
            self.sampler._phase_index = saved["phase"]

    def advance(self):
        self.step += 1
        completed, self.accumulated_microbatches = divmod(
            self.accumulated_microbatches + 1, self.gradient_accumulation_steps
        )
        self.optimizer_step += completed

    def state_dict(self, backends):
        return {
            "version": 2,
            "revision": self.revision,
            "epoch": self.epoch,
            "step": self.step,
            "optimizer_step": self.optimizer_step,
            "accumulated_microbatches": self.accumulated_microbatches,
            "phase": self.sampler._phase_index,
            "exhausted_backends": sorted(self.dataset_ids.difference(backends)),
        }


def random_dataloader_iterator(
    step: int,
    backends: Dict[str, Any],
    *,
    update_epoch_step: bool = True,
    mixer: Optional[DatasetMixer] = None,
) -> Union[Any, bool]:
    if not backends:
        raise ValueError("No data backends provided to iterator.")

    if mixer is not None:
        step = mixer.step + 1
    prefetch_log_debug("Random dataloader iterator launched.")
    logger.debug(f"Backends to select from {backends}")
    while backends:
        if update_epoch_step:
            _set_epoch_step_for_training_step(step)

        if mixer is None:
            sampler = _get_scaled_sampler(backends)
        else:
            sampler = mixer.sampler
            sampler.rebuild(backends)
        chosen_backend_id = None
        if sampler and sampler.has_scaled_groups():
            chosen_backend_id = sampler.next_backend_id(step, backends, mixer=mixer)
        if chosen_backend_id is None:
            chosen_backend_id = select_dataloader_index(step, backends, mixer=mixer)
        if chosen_backend_id is None:
            # No active backends available - signal epoch end
            logger.info(
                "No active datasets available for sampling. "
                "This may happen when datasets with start_epoch/start_step are waiting to become active. "
                "Ending epoch early."
            )
            if mixer is None:
                StateTracker.clear_exhausted_buckets()
                _SCALED_SAMPLERS.pop(id(backends), None)
            return False
        backend = backends.get(chosen_backend_id)
        if backend is None:
            raise KeyError(f"Selected backend {chosen_backend_id} not found.")

        try:
            if hasattr(backend, "get_data_loader"):
                batch = backend.get_data_loader()
            else:
                batch = next(iter(backend))
            if mixer is not None:
                mixer.advance()
                batch["dataset_mixer_state"] = mixer.state_dict(backends)
            return batch
        except MultiDatasetExhausted:
            config = getattr(backend, "config", None)
            if not isinstance(config, dict):
                config = StateTracker.get_data_backend_config(chosen_backend_id)
            schedule_owns_repeats = (
                getattr(backend, "uses_native_schedule", False) is True
                or getattr(backend, "tracks_consumed_batches", False) is True
            )
            repeats = 0 if schedule_owns_repeats else config.get("repeats", False)
            if repeats and repeats > 0 and StateTracker.get_repeats(chosen_backend_id) < repeats:
                StateTracker.increment_repeats(chosen_backend_id)
                logger.debug(
                    f"Dataset (name={chosen_backend_id}) is now sampling its {StateTracker.get_repeats(chosen_backend_id)} repeat out of {repeats} total allowed."
                )
                continue
            del backends[chosen_backend_id]
            if mixer is None:
                StateTracker.backend_exhausted(chosen_backend_id)
                StateTracker.set_repeats(data_backend_id=chosen_backend_id, repeats=0)
            logger.debug("Dataset %s exhausted its schedule.", chosen_backend_id)
            if sampler:
                sampler.remove_backend(chosen_backend_id)
    if mixer is None:
        StateTracker.clear_exhausted_buckets()
        _SCALED_SAMPLERS.pop(id(backends), None)
    return False
