import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from skrample import scheduling as sk_scheduling


@dataclass(frozen=True)
class FixedDDPMSchedule:
    timesteps: np.ndarray
    regular_sigmas: np.ndarray
    sigma_transform: Any

    def schedule(self, steps: int) -> np.ndarray:
        if steps == len(self.regular_sigmas):
            return np.stack([self.timesteps, self.regular_sigmas], axis=1)

        positions = np.linspace(0, len(self.regular_sigmas) - 1, steps, dtype=np.float64)
        sigmas = np.interp(positions, np.arange(len(self.regular_sigmas), dtype=np.float64), self.regular_sigmas)
        timesteps = np.interp(positions, np.arange(len(self.timesteps), dtype=np.float64), self.timesteps)
        return np.stack([timesteps, sigmas], axis=1)

    def sigmas(self, steps: int) -> np.ndarray:
        return self.schedule(steps)[:, 1]

    def ipoint(self, fraction: float):
        idx = int(float(fraction) * len(self.regular_sigmas))
        idx = max(0, min(idx, len(self.regular_sigmas) - 1))
        regular_sigma = float(self.regular_sigmas[idx])
        sigma_u, sigma_v = self.sigma_transform(regular_sigma)
        return SimpleNamespace(
            timestep=float(self.timesteps[idx]),
            regular_sigma=regular_sigma,
            sigma=sigma_u,
            alpha=sigma_v,
        )


class _Skrample05Sampler:
    def __init__(self, sampler):
        self._sampler = sampler

    @property
    def require_previous(self) -> int:
        return int(getattr(self._sampler, "require_previous", 0))

    def sample(self, current, x_pred, *, step, model_transform, schedule, noise, previous):
        if isinstance(step, tuple):
            step_idx = int(float(step[0]) * len(schedule.regular_sigmas))
        else:
            step_idx = int(step)
        step_idx = max(0, min(step_idx, len(schedule.regular_sigmas) - 1))
        return self._sampler.sample(
            current,
            x_pred,
            step_idx,
            schedule.regular_sigmas,
            schedule.sigma_transform,
            noise,
            previous,
        )


class _Skrample06Sampler:
    def __init__(self, sampler):
        self._sampler = sampler

    @property
    def require_previous(self) -> int:
        return int(getattr(self._sampler, "require_previous", 0))

    def sample(self, current, x_pred, *, step, model_transform, schedule, noise, previous):
        return self._sampler.sample(
            current,
            x_pred,
            step=step,
            model_transform=model_transform,
            schedule=schedule,
            noise=noise,
            previous=previous,
        )


def _make_fixed_schedule(timesteps: np.ndarray, regular_sigmas: np.ndarray):
    if hasattr(sk_scheduling, "FixedSchedule"):
        return sk_scheduling.FixedSchedule.from_regular(
            timesteps=timesteps,
            regular_sigmas=regular_sigmas,
            sigma_space=sk_scheduling.VariancePreserving(),
        )

    from skrample import common as sk_common

    return FixedDDPMSchedule(
        timesteps=timesteps,
        regular_sigmas=regular_sigmas,
        sigma_transform=sk_common.sigma_polar,
    )


def make_sigma_schedule_from_ddpm(noise_schedule):
    """
    Build a fixed skrample schedule from a diffusers DDPMScheduler.
    """
    alphas_cumprod = getattr(noise_schedule, "alphas_cumprod", None)
    if alphas_cumprod is None:
        raise ValueError("Noise schedule does not expose alphas_cumprod for scheduled sampling rollout.")

    if torch.is_tensor(alphas_cumprod):
        alphas = alphas_cumprod.detach().cpu().double().numpy()
    else:
        alphas = np.array(alphas_cumprod, dtype=np.float64)

    regular_sigmas = np.sqrt((1.0 - alphas) / np.maximum(alphas, 1e-12))
    timesteps = np.arange(0, len(regular_sigmas), dtype=np.float64)
    return _make_fixed_schedule(timesteps, regular_sigmas)


def make_data_prediction_transform():
    """
    Scheduled sampling rollouts convert model outputs to x-prediction before calling skrample.
    """
    try:
        from skrample.sampling import models as sk_models
    except ImportError:
        return None
    return sk_models.DataModel()


def make_sampler(name: str, order: int):
    """
    Factory for skrample structured samplers used in short rollouts.
    """
    name = (name or "unipc").strip().lower()
    order = int(max(1, min(order, 9)))

    try:
        structured = importlib.import_module("skrample.sampling.structured")
        wrapper = _Skrample06Sampler
    except ImportError:
        structured = importlib.import_module("skrample.sampling")
        wrapper = _Skrample05Sampler

    if name == "unipc" and hasattr(structured, "UniPC"):
        return wrapper(structured.UniPC(order=order))
    if name == "unip" and hasattr(structured, "UniP"):
        return wrapper(structured.UniP(order=order))
    if name == "dpm" and hasattr(structured, "DPM"):
        max_order = min(order, structured.DPM.max_order())
        return wrapper(structured.DPM(order=max_order))
    if name == "euler" and hasattr(structured, "Euler"):
        return wrapper(structured.Euler())
    if name == "rk4":
        raise ImportError("Scheduled sampling sampler 'rk4' is not available in skrample 0.5.x/0.6.x samplers.")

    raise ValueError(f"Unsupported scheduled sampling sampler: {name}")
