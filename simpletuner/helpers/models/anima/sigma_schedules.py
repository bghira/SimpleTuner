# Vendored from diffusers-anima: /src/diffusers-anima/src/diffusers_anima/pipelines/anima/sigma_schedules.py
# Adapted for SimpleTuner local imports.

"""Sigma schedule construction for Anima custom samplers."""

from __future__ import annotations

import math

import numpy as np
import torch

from .constants import ANIMA_SAMPLING_MULTIPLIER, FORGE_BETA_ALPHA, FORGE_BETA_BETA


def _time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    if alpha == 1.0:
        return t
    numerator = torch.mul(t, alpha)
    denominator = torch.add(torch.mul(t, alpha - 1.0), 1.0)
    return torch.div(numerator, denominator)


def build_simple_sigmas(base_sigmas: torch.Tensor, *, steps: int) -> torch.Tensor:
    """Select sigmas for the simple schedule."""
    if steps < 1:
        raise ValueError("steps must be >= 1")

    stride = len(base_sigmas) / float(steps)
    picked = [float(base_sigmas[-(1 + int(i * stride))].item()) for i in range(steps)]
    picked.append(0.0)
    return torch.tensor(picked, device=base_sigmas.device, dtype=torch.float32)


def build_beta_sigmas(
    *,
    num_inference_steps: int,
    num_train_timesteps: int,
    shift: float,
    beta_alpha: float,
    beta_beta: float,
    device: str,
) -> torch.Tensor:
    """Build beta-distribution sigma schedule."""
    from scipy import stats

    t = (
        torch.arange(1, num_train_timesteps + 1, dtype=torch.float32, device=device) / float(num_train_timesteps)
    ) * ANIMA_SAMPLING_MULTIPLIER
    base_sigmas = _time_snr_shift(shift, t)

    total_timesteps = len(base_sigmas) - 1
    ts = 1.0 - np.linspace(0.0, 1.0, num_inference_steps, endpoint=False)
    mapped = stats.beta.ppf(ts, beta_alpha, beta_beta) * float(total_timesteps)
    mapped = np.nan_to_num(mapped, nan=0.0, posinf=float(total_timesteps), neginf=0.0)
    indices = np.clip(np.rint(mapped).astype(np.int64), 0, total_timesteps)

    sigmas: list[float] = []
    last_index: int | None = None
    for index in indices:
        if last_index is None or index != last_index:
            sigmas.append(float(base_sigmas[int(index)].item()))
        last_index = int(index)

    sigmas.append(0.0)
    return torch.tensor(sigmas, device=device, dtype=torch.float32)


def build_normal_sigmas(
    *,
    num_inference_steps: int,
    num_train_timesteps: int,
    shift: float,
    device: str,
) -> torch.Tensor:
    """Build normal (linearly-spaced) sigma schedule."""
    multiplier = float(ANIMA_SAMPLING_MULTIPLIER)

    t = (
        torch.arange(
            1,
            num_train_timesteps + 1,
            dtype=torch.float32,
            device=device,
        )
        / float(num_train_timesteps)
    ) * multiplier
    base_sigmas = _time_snr_shift(shift, t / multiplier)
    sigma_min = base_sigmas[0]
    sigma_max = base_sigmas[-1]

    start = sigma_max * multiplier
    end = sigma_min * multiplier

    append_zero = True
    sigma_at_end = _time_snr_shift(shift, end / multiplier)
    if math.isclose(float(sigma_at_end.item()), 0.0, abs_tol=1e-5):
        num_inference_steps += 1
        append_zero = False

    timesteps = torch.linspace(
        start,
        end,
        num_inference_steps,
        device=device,
        dtype=torch.float32,
    )
    sigmas = _time_snr_shift(shift, timesteps / multiplier).to(dtype=torch.float32)
    if append_zero:
        sigmas = torch.cat(
            [
                sigmas,
                torch.zeros(1, device=device, dtype=torch.float32),
            ]
        )
    return sigmas


def build_sampling_sigmas(
    scheduler: object,
    *,
    num_inference_steps: int,
    sigma_schedule: str,
    beta_alpha: float = FORGE_BETA_ALPHA,
    beta_beta: float = FORGE_BETA_BETA,
    device: str,
) -> torch.Tensor:
    """Build sigma trajectories for Anima custom samplers.

    Args:
        scheduler: An ``AnimaFlowMatchEulerDiscreteScheduler`` instance.
        num_inference_steps: Number of denoising steps.
        sigma_schedule: One of ``beta``, ``simple``, ``normal``, or ``uniform``.
        beta_alpha: Alpha parameter for the beta distribution (used when
            ``sigma_schedule='beta'``).
        beta_beta: Beta parameter for the beta distribution (used when
            ``sigma_schedule='beta'``).
        device: Target device string.

    Returns:
        A 1-D tensor of sigmas with a trailing zero appended.
    """
    shift = float(scheduler.config.shift)  # type: ignore[union-attr]
    num_train_timesteps = int(scheduler.config.num_train_timesteps)  # type: ignore[union-attr]

    if sigma_schedule == "normal":
        return build_normal_sigmas(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            device=device,
        )

    if sigma_schedule == "simple":
        t = (
            torch.arange(
                1,
                num_train_timesteps + 1,
                dtype=torch.float32,
                device=device,
            )
            / float(num_train_timesteps)
        ) * ANIMA_SAMPLING_MULTIPLIER
        base_sigmas = _time_snr_shift(shift, t)
        return build_simple_sigmas(base_sigmas, steps=num_inference_steps)

    if sigma_schedule == "beta":
        return build_beta_sigmas(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            device=device,
        )

    # "uniform" — delegate to the scheduler's own set_timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)  # type: ignore[union-attr]
    return scheduler.sigmas.to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
