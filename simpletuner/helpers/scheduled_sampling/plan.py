from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ScheduledSamplingPlan:
    target_timesteps: torch.Tensor
    source_timesteps: torch.Tensor
    rollout_steps: torch.Tensor


def build_rollout_schedule(
    *,
    num_train_timesteps: int,
    batch_size: int,
    max_step_offset: int,
    device,
    strategy: str = "uniform",
    apply_probability: Optional[float] = None,
) -> ScheduledSamplingPlan:
    """
    Sample target timesteps and optional upstream rollout timesteps for scheduled sampling.
    """
    if max_step_offset <= 0:
        base = torch.randint(0, num_train_timesteps, (batch_size,), device=device)
        return ScheduledSamplingPlan(target_timesteps=base, source_timesteps=base, rollout_steps=torch.zeros_like(base))

    base = torch.randint(0, num_train_timesteps, (batch_size,), device=device)
    if apply_probability is not None and apply_probability < 1.0:
        mask = torch.bernoulli(torch.full((batch_size,), apply_probability, device=device)).bool()
    else:
        mask = torch.ones((batch_size,), device=device, dtype=torch.bool)

    if strategy == "uniform":
        offsets = torch.randint(0, max_step_offset + 1, (batch_size,), device=device)
    elif strategy == "biased_early":
        offsets = torch.round(torch.rand((batch_size,), device=device) ** 2 * max_step_offset).to(torch.long)
    elif strategy == "biased_late":
        offsets = torch.round((1.0 - torch.rand((batch_size,), device=device) ** 2) * max_step_offset).to(torch.long)
    else:
        raise ValueError(f"Unknown scheduled sampling strategy: {strategy}")

    offsets = offsets * mask
    source = torch.clamp(base + offsets, max=num_train_timesteps - 1)
    return ScheduledSamplingPlan(target_timesteps=base, source_timesteps=source, rollout_steps=offsets)
