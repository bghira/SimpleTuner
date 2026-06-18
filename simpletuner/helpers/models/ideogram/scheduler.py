"""Logit-normal schedule and Euler flow-matching sampler."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LogitNormalSchedule:
    mean: float
    std: float = 1.5
    logsnr_min: float = -15.0
    logsnr_max: float = 18.0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        # MPS supports neither float64 nor ndtri; the inverse-normal-CDF needs
        # float64 precision in the tails, so run that part on CPU and return the
        # float32 result on the original device. The tensor is tiny (a few
        # timesteps), so the host round-trip is negligible.
        input_device = t.device
        compute_device = torch.device("cpu") if input_device.type == "mps" else input_device
        # Move device first, then cast: a fused .to(device=cpu, dtype=float64) from an
        # MPS source still attempts the unsupported float64 cast on MPS.
        t = t.to(device=compute_device).to(torch.float64)
        z = torch.special.ndtri(t)
        y = self.mean + self.std * z
        t_ = torch.special.expit(y)
        t_ = 1 - t_
        t_min = 1.0 / (1 + math.exp(0.5 * self.logsnr_max))
        t_max = 1.0 / (1 + math.exp(0.5 * self.logsnr_min))
        return t_.clamp(t_min, t_max).to(device=input_device, dtype=torch.float32)


def get_schedule_for_resolution(
    image_resolution: tuple[int, int],
    known_resolution: tuple[int, int] = (512, 512),
    known_mean: float = 0.0,
    std: float = 1.5,
) -> LogitNormalSchedule:
    """Resolution-aware schedule used at eval time."""
    num_pixels = image_resolution[0] * image_resolution[1]
    known_pixels = known_resolution[0] * known_resolution[1]
    mean = known_mean + 0.5 * math.log(num_pixels / known_pixels)
    return LogitNormalSchedule(mean=mean, std=std)


def make_step_intervals(num_steps: int) -> torch.Tensor:
    """Default linear step schedule used by the v4 eval config."""
    return torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32)


@dataclass(frozen=True, kw_only=True)
class SamplerParameters:
    """Bundle of sampling hyperparameters for a named preset.

    ``guidance_schedule`` is in LOOP-INDEX order: index 0 is the LAST sampling
    step (final polish), index ``num_steps - 1`` is the FIRST sampling step.
    ``mu`` and ``std`` are the mean and stddev of the logit-normal noise
    schedule passed to ``get_schedule_for_resolution`` (as ``known_mean`` and
    ``std`` respectively).

    See ``ideogram4.sampler_configs.PRESETS`` for the named preset registry.
    """

    num_steps: int
    guidance_schedule: tuple[float, ...]
    mu: float
    std: float = 1.5

    def __post_init__(self) -> None:
        if len(self.guidance_schedule) != self.num_steps:
            raise ValueError(
                f"guidance_schedule has length {len(self.guidance_schedule)}, " f"expected num_steps={self.num_steps}"
            )
