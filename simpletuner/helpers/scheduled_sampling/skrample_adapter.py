import importlib

import numpy as np
import torch
from skrample import scheduling as sk_scheduling
from skrample.sampling import models as sk_models


def make_sigma_schedule_from_ddpm(noise_schedule) -> sk_scheduling.SkrampleSchedule:
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

    sigmas = np.sqrt((1.0 - alphas) / np.maximum(alphas, 1e-12))
    timesteps = np.arange(0, len(sigmas), dtype=np.float64)
    return sk_scheduling.FixedSchedule.from_regular(
        timesteps=timesteps,
        regular_sigmas=sigmas,
        sigma_space=sk_scheduling.VariancePreserving(),
    )


def make_data_prediction_transform() -> sk_models.DataModel:
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

    try:
        structured = importlib.import_module("skrample.sampling.structured")
    except Exception as exc:
        raise ImportError("No compatible skrample sampler implementations found.") from exc

    if name == "unipc" and hasattr(structured, "UniPC"):
        return structured.UniPC(order=order)
    if name == "unip" and hasattr(structured, "UniP"):
        return structured.UniP(order=order)
    if name == "dpm" and hasattr(structured, "DPM"):
        max_order = min(order, structured.DPM.max_order())
        return structured.DPM(order=max_order)
    if name == "euler" and hasattr(structured, "Euler"):
        return structured.Euler()
    if name == "rk4" and hasattr(structured, "Euler"):
        return structured.Euler()
    if hasattr(structured, "Euler"):
        return structured.Euler()

    raise ImportError("No compatible skrample sampler implementations found.")
