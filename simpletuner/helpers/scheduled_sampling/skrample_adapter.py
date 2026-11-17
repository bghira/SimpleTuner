import numpy as np
import torch
from skrample import scheduling as sk_scheduling
from skrample.sampling import models as sk_models
from skrample.sampling import structured


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
    return sk_scheduling.FixedSchedule.from_regular(
        timesteps=timesteps,
        regular_sigmas=regular_sigmas,
        sigma_space=sk_scheduling.VariancePreserving(),
    )


def make_data_prediction_transform():
    """
    Scheduled sampling rollouts convert model outputs to x-prediction before calling skrample.
    """
    return sk_models.DataModel()


def make_sampler(name: str, order: int):
    """
    Factory for skrample structured samplers used in short rollouts.
    """
    name = (name or "unipc").strip().lower()
    order = int(max(1, min(order, 9)))

    if name == "unipc":
        return structured.UniPC(order=order)
    if name == "unip":
        return structured.UniP(order=order)
    if name == "dpm":
        max_order = min(order, structured.DPM.max_order())
        return structured.DPM(order=max_order)
    if name == "euler":
        return structured.Euler()
    if name == "rk4":
        raise ImportError("Scheduled sampling sampler 'rk4' is not available in skrample structured samplers.")

    raise ValueError(f"Unsupported scheduled sampling sampler: {name}")
