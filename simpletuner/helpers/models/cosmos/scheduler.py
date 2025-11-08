"""Rectified Flow scheduler utilities for Cosmos models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers import KDPM2DiscreteScheduler
from diffusers.utils import BaseOutput


def _broadcast_like(reference: torch.Tensor, target_ndim: int) -> torch.Tensor:
    if reference.ndim >= target_ndim:
        return reference
    shape = reference.shape + (1,) * (target_ndim - reference.ndim)
    return reference.view(shape)


def _batch_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    max_ndim = max(x.ndim, y.ndim)
    return _broadcast_like(x, max_ndim) * _broadcast_like(y, max_ndim)


def _phi1(values: torch.Tensor) -> torch.Tensor:
    input_dtype = values.dtype
    values = values.to(dtype=torch.float64)
    return (torch.expm1(values) / values).to(dtype=input_dtype)


def _phi2(values: torch.Tensor) -> torch.Tensor:
    input_dtype = values.dtype
    values = values.to(dtype=torch.float64)
    return ((_phi1(values) - 1.0) / values).to(dtype=input_dtype)


def _res_x0_rk2_step(
    x_s: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    x0_s: torch.Tensor,
    s1: torch.Tensor,
    x0_s1: torch.Tensor,
) -> torch.Tensor:
    s = -torch.log(s)
    t = -torch.log(t)
    mid = -torch.log(s1)

    dt = t - s
    if torch.any(torch.isclose(dt, torch.zeros_like(dt), atol=1e-6)):
        raise AssertionError("Step size is too small")
    if torch.any(torch.isclose(mid - s, torch.zeros_like(dt), atol=1e-6)):
        raise AssertionError("Step size is too small")

    c2 = (mid - s) / dt
    phi1_val, phi2_val = _phi1(-dt), _phi2(-dt)

    b1 = torch.nan_to_num(phi1_val - (1.0 / c2) * phi2_val, nan=0.0)
    b2 = torch.nan_to_num((1.0 / c2) * phi2_val, nan=0.0)

    residual = _batch_mul(b1, x0_s) + _batch_mul(b2, x0_s1)
    return _batch_mul(torch.exp(-dt), x_s) + _batch_mul(dt, residual)


def _reg_x0_euler_step(
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    x0_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    coef_x0 = (s - t) / s
    coef_xs = t / s
    updated = _batch_mul(coef_x0, x0_s) + _batch_mul(coef_xs, x_s)
    return updated, x0_s


@dataclass
class RectifiedFlowAB2SchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor


class RectifiedFlowAB2Scheduler(KDPM2DiscreteScheduler):
    """Adams-Bashforth 2 scheduler tailored for Cosmos Rectified-Flow models."""

    order = 2

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        order: float = 7.0,
        sigma_data: float = 1.0,
        final_sigmas_type: str = "sigma_min",
        t_scaling_factor: float = 1.0,
        use_double_precision: bool = True,
        **kpm2_kwargs,
    ):
        prediction_type = kpm2_kwargs.pop("prediction_type", "epsilon")
        num_train_timesteps = kpm2_kwargs.pop("num_train_timesteps", 1000)
        self._use_fp64 = use_double_precision
        self._sample_eps = 1e-6
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        super().__init__(
            prediction_type=prediction_type,
            num_train_timesteps=num_train_timesteps,
            **kpm2_kwargs,
        )

    def _dtype(self) -> torch.dtype:
        return torch.float64 if self._use_fp64 else torch.float32

    def sample_sigma(
        self,
        batch_size: int,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device = device or (self.sigmas.device if self.sigmas is not None else torch.device("cpu"))
        u = torch.rand(batch_size, device=device, generator=generator)
        u = u.clamp_(self._sample_eps, 1.0 - self._sample_eps)
        log_sigma = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
        sigmas = torch.exp(log_sigma)
        return sigmas.clamp(self.config.sigma_min, self.config.sigma_max)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: torch.device | None = None,
        num_train_timesteps: int | None = None,
        sigmas: torch.Tensor | None = None,
        **_,
    ):
        device = device or torch.device("cpu")
        dtype = self._dtype()

        if sigmas is None and num_inference_steps is None:
            num_inference_steps = num_train_timesteps

        if sigmas is not None:
            sigma_values = torch.as_tensor(sigmas, device=device, dtype=dtype).flatten()
            if sigma_values.numel() < 2:
                raise ValueError("`sigmas` must contain at least two values.")
        else:
            if num_inference_steps is None:
                raise ValueError("`num_inference_steps` must be provided when sigmas are not supplied.")
            n_sigma = num_inference_steps + 1
            i = torch.arange(n_sigma, device=device, dtype=dtype)
            sigma_min = torch.tensor(self.config.sigma_min, device=device, dtype=dtype)
            sigma_max = torch.tensor(self.config.sigma_max, device=device, dtype=dtype)
            order = torch.tensor(self.config.order, device=device, dtype=dtype)
            ramp = sigma_max ** (1.0 / order) + i / (n_sigma - 1) * (sigma_min ** (1.0 / order) - sigma_max ** (1.0 / order))
            sigma_values = ramp**order

        self.sigmas = sigma_values
        self.timesteps = torch.arange(sigma_values.numel() - 1, device=device, dtype=torch.long)
        self.num_inference_steps = self.timesteps.numel()
        return self.timesteps

    def step(
        self,
        x0_pred: torch.Tensor,
        i: int,
        sample: torch.Tensor,
        x0_prev: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> RectifiedFlowAB2SchedulerOutput | tuple[torch.Tensor, torch.Tensor]:
        if self.sigmas is None:
            raise RuntimeError("`set_timesteps` must be called before `step`.")

        dtype_target = sample.dtype
        dtype_work = self._dtype()
        device = sample.device

        x_t = sample.to(dtype=dtype_work)
        x0_t = x0_pred.to(dtype=dtype_work)

        sigma_t = self.sigmas[i].to(device=device, dtype=dtype_work)
        sigma_s = self.sigmas[i + 1].to(device=device, dtype=dtype_work)
        ones = torch.ones(x_t.shape[0], device=device, dtype=dtype_work)

        if x0_prev is None:
            updated, x0_current = _reg_x0_euler_step(x_t, sigma_t * ones, sigma_s * ones, x0_t)
        else:
            x0_prev = x0_prev.to(dtype=dtype_work)
            updated = _res_x0_rk2_step(
                x_t,
                sigma_s * ones,
                sigma_t * ones,
                x0_t,
                self.sigmas[i - 1].to(device=device, dtype=dtype_work) * ones,
                x0_prev,
            )
            x0_current = x0_t

        output = RectifiedFlowAB2SchedulerOutput(
            prev_sample=updated.to(dtype=dtype_target),
            pred_original_sample=x0_current.to(dtype=dtype_target),
        )
        if return_dict:
            return output
        return output.prev_sample, output.pred_original_sample
