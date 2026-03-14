# Vendored from diffusers-anima: /src/diffusers-anima/src/diffusers_anima/pipelines/anima/strength_utils.py
# Adapted for SimpleTuner local imports.

"""Strength-based step trimming utilities for img2img workflows."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .pipeline import AnimaPipeline


def _resolve_strength_start_step(
    *,
    total_steps: int,
    strength: float,
) -> int:
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    init_timestep = min(total_steps * strength, total_steps)
    return int(max(total_steps - init_timestep, 0))


def _trim_flowmatch_timesteps_by_strength(
    pipe: AnimaPipeline,
    *,
    num_inference_steps: int,
    strength: float,
) -> torch.Tensor:
    pipe.scheduler.set_timesteps(num_inference_steps, device=pipe.execution_device)
    timesteps = pipe.scheduler.timesteps

    if math.isclose(strength, 1.0):
        return timesteps

    t_start = _resolve_strength_start_step(total_steps=num_inference_steps, strength=strength)
    begin_index = t_start * int(getattr(pipe.scheduler, "order", 1))
    trimmed = timesteps[begin_index:]
    if hasattr(pipe.scheduler, "set_begin_index"):
        pipe.scheduler.set_begin_index(begin_index)
    if len(trimmed) < 1:
        raise ValueError(
            f"After applying strength={strength}, no denoising steps remain. "
            "Increase `strength` or `num_inference_steps`."
        )
    return trimmed


def _trim_sigmas_by_strength(
    *,
    sigmas: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    if math.isclose(strength, 1.0):
        return sigmas

    total_steps = len(sigmas) - 1
    t_start = _resolve_strength_start_step(total_steps=total_steps, strength=strength)
    trimmed = sigmas[t_start:]
    if len(trimmed) < 2:
        raise ValueError(
            f"After applying strength={strength}, fewer than 1 denoising step remain. "
            "Increase `strength` or `num_inference_steps`."
        )
    return trimmed
