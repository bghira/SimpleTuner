from __future__ import annotations

from typing import TypeVar

TScheduler = TypeVar("TScheduler")


def fix_flow_match_euler_schedule_bounds(scheduler: TScheduler) -> TScheduler:
    config = getattr(scheduler, "config", None)
    if config is None or getattr(config, "use_dynamic_shifting", False):
        return scheduler

    num_train_timesteps = getattr(config, "num_train_timesteps", None)
    if num_train_timesteps is None:
        return scheduler

    scheduler.sigma_max = float(getattr(config, "sigma_max", 1.0) or 1.0)
    scheduler.sigma_min = 1.0 / float(num_train_timesteps)
    return scheduler
