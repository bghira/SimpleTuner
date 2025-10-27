from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


_EPS = 1e-6


def _ensure_tensor(data: torch.Tensor | float | int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(data):
        tensor = data
    else:
        tensor = torch.tensor(data)
    return tensor.to(device=device, dtype=dtype)


def _match_rank(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    while values.ndim < reference.ndim:
        values = values.unsqueeze(-1)
    return values


@dataclass
class FlowMatchingSchedulerAdapter:
    """
    Wraps a diffusers FlowMatchEulerDiscreteScheduler (or compatible object) to expose
    the WAN flow-matching operations used by the realtime self-forcing stack.
    """

    scheduler: object

    def __post_init__(self) -> None:
        if not hasattr(self.scheduler, "sigmas"):
            raise ValueError("Scheduler must expose `sigmas` and `timesteps` tensors.")
        raw_sigmas = self.scheduler.sigmas
        self.sigmas: torch.Tensor = torch.as_tensor(raw_sigmas).clone().detach()
        if hasattr(self.scheduler, "timesteps"):
            raw_timesteps = self.scheduler.timesteps
            self.timesteps = torch.as_tensor(raw_timesteps).clone().detach()
        else:
            self.timesteps = torch.arange(self.sigmas.shape[0], dtype=self.sigmas.dtype)

    @property
    def device(self) -> torch.device:
        return self.sigmas.device

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "FlowMatchingSchedulerAdapter":
        if device is not None:
            self.sigmas = self.sigmas.to(device=device)
            self.timesteps = self.timesteps.to(device=device)
        if dtype is not None:
            self.sigmas = self.sigmas.to(dtype=dtype)
            self.timesteps = self.timesteps.to(dtype=dtype)
        return self

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _lookup_sigma(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dtype.is_floating_point:
            indices = timesteps.round().long()
        else:
            indices = timesteps.long()
        indices = indices.clamp_(0, self.sigmas.shape[0] - 1)
        flat_sigma = self.sigmas[indices.view(-1)].view_as(indices)
        sigma = flat_sigma.to(device=timesteps.device, dtype=self.sigmas.dtype)
        return sigma

    def add_noise(self, clean_latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sigma = self._lookup_sigma(timesteps).to(device=clean_latents.device, dtype=clean_latents.dtype)
        sigma = _match_rank(sigma, clean_latents)
        return (1.0 - sigma) * clean_latents + sigma * noise

    def convert_flow_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sigma = self._lookup_sigma(timesteps).to(device=flow_pred.device, dtype=flow_pred.dtype)
        sigma = _match_rank(sigma, flow_pred)
        denom = (1.0 - sigma).clamp(min=_EPS)
        return (xt - sigma * flow_pred) / denom

    def convert_x0_to_noise(self, x0: torch.Tensor, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sigma = self._lookup_sigma(timesteps).to(device=x0.device, dtype=x0.dtype)
        sigma = _match_rank(sigma, x0)
        denom = sigma.clamp(min=_EPS)
        return (xt - (1.0 - sigma) * x0) / denom

    def convert_noise_to_x0(self, noise: torch.Tensor, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sigma = self._lookup_sigma(timesteps).to(device=noise.device, dtype=noise.dtype)
        sigma = _match_rank(sigma, noise)
        denom = (1.0 - sigma).clamp(min=_EPS)
        return (xt - sigma * noise) / denom
