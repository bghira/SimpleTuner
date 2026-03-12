# Vendored from diffusers-anima: /src/diffusers-anima/src/diffusers_anima/schedulers/anima_flow_match_euler.py
# Adapted for SimpleTuner local imports.

"""Anima-specific scheduler wrapper on top of FlowMatch Euler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Diffusers custom-component loading checks for `SchedulerMixin` by name in this module.
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import SchedulerMixin as SchedulerMixin  # noqa: F401

from .constants import FORGE_BETA_ALPHA, FORGE_BETA_BETA

SUPPORTED_ANIMA_SAMPLERS = (
    "flowmatch_euler",
    "euler",
    "euler_a_rf",
    "euler_ancestral_rf",
)
SUPPORTED_ANIMA_SIGMA_SCHEDULES = ("beta", "uniform", "simple", "normal")


def _validate_anima_sampler_config(*, sampler: str, sigma_schedule: str) -> None:
    """Validate Anima sampler/sigma schedule combinations."""
    if sampler not in SUPPORTED_ANIMA_SAMPLERS:
        raise ValueError("`sampler` must be one of: flowmatch_euler, euler, euler_a_rf, euler_ancestral_rf.")
    if sigma_schedule not in SUPPORTED_ANIMA_SIGMA_SCHEDULES:
        raise ValueError("`sigma_schedule` must be one of: beta, uniform, simple, normal.")
    if sampler == "flowmatch_euler" and sigma_schedule != "uniform":
        raise ValueError("`flowmatch_euler` requires `sigma_schedule='uniform'`.")


def _scheduler_config_get(config: Any, *, key: str, default: Any) -> Any:
    """Read scheduler config from dict-like or attribute-like objects."""
    if hasattr(config, "get"):
        value = config.get(key, None)
    else:
        value = getattr(config, key, None)
    return default if value is None else value


@dataclass(frozen=True)
class AnimaSamplingConfig:
    """Resolved Anima sampling parameters consumed by the pipeline runtime."""

    sampler: str
    sigma_schedule: str
    beta_alpha: float
    beta_beta: float
    eta: float
    s_noise: float


class AnimaFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """FlowMatch Euler scheduler extended with Anima sampling metadata.

    Sampling knobs are first-class scheduler config fields so they are serialized
    into `scheduler_config.json` and restored by `from_config(...)`.
    """

    SUPPORTED_SAMPLERS = SUPPORTED_ANIMA_SAMPLERS
    SUPPORTED_SIGMA_SCHEDULES = SUPPORTED_ANIMA_SIGMA_SCHEDULES

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: float | None = 0.5,
        max_shift: float | None = 1.15,
        base_image_seq_len: int | None = 256,
        max_image_seq_len: int | None = 4096,
        invert_sigmas: bool = False,
        shift_terminal: float | None = None,
        use_karras_sigmas: bool | None = False,
        use_exponential_sigmas: bool | None = False,
        use_beta_sigmas: bool | None = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
        sampler: str = "euler_a_rf",
        sigma_schedule: str = "beta",
        beta_alpha: float = FORGE_BETA_ALPHA,
        beta_beta: float = FORGE_BETA_BETA,
        eta: float = 1.0,
        s_noise: float = 1.0,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
            base_shift=base_shift,
            max_shift=max_shift,
            base_image_seq_len=base_image_seq_len,
            max_image_seq_len=max_image_seq_len,
            invert_sigmas=invert_sigmas,
            shift_terminal=shift_terminal,
            use_karras_sigmas=use_karras_sigmas,
            use_exponential_sigmas=use_exponential_sigmas,
            use_beta_sigmas=use_beta_sigmas,
            time_shift_type=time_shift_type,
            stochastic_sampling=stochastic_sampling,
        )
        self.set_sampling_config(
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            eta=eta,
            s_noise=s_noise,
        )

    def set_sampling_config(
        self,
        *,
        sampler: str = "euler_a_rf",
        sigma_schedule: str = "beta",
        beta_alpha: float = FORGE_BETA_ALPHA,
        beta_beta: float = FORGE_BETA_BETA,
        eta: float = 1.0,
        s_noise: float = 1.0,
    ) -> None:
        """Persist Anima sampling knobs into scheduler config."""
        _validate_anima_sampler_config(sampler=sampler, sigma_schedule=sigma_schedule)
        self.register_to_config(
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            beta_alpha=float(beta_alpha),
            beta_beta=float(beta_beta),
            eta=float(eta),
            s_noise=float(s_noise),
        )

    def get_sampling_config(self) -> AnimaSamplingConfig:
        """Return validated Anima sampling knobs from scheduler config."""
        config = self.config
        sampler = str(_scheduler_config_get(config, key="sampler", default="euler_a_rf"))
        sigma_schedule = str(_scheduler_config_get(config, key="sigma_schedule", default="beta"))
        beta_alpha = float(_scheduler_config_get(config, key="beta_alpha", default=FORGE_BETA_ALPHA))
        beta_beta = float(_scheduler_config_get(config, key="beta_beta", default=FORGE_BETA_BETA))
        eta = float(_scheduler_config_get(config, key="eta", default=1.0))
        s_noise = float(_scheduler_config_get(config, key="s_noise", default=1.0))
        _validate_anima_sampler_config(sampler=sampler, sigma_schedule=sigma_schedule)
        return AnimaSamplingConfig(
            sampler=sampler,
            sigma_schedule=sigma_schedule,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            eta=eta,
            s_noise=s_noise,
        )
