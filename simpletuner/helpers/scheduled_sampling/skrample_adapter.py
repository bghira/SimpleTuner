import importlib

import numpy as np
import torch
from skrample import common as sk_common


def make_sigma_schedule_from_ddpm(noise_schedule) -> tuple[np.ndarray, sk_common.SigmaTransform]:
    """
    Build a Skrample-style schedule [(timestep, sigma), ...] and sigma_transform from a diffusers DDPMScheduler.
    """
    alphas_cumprod = getattr(noise_schedule, "alphas_cumprod", None)
    if alphas_cumprod is None:
        raise ValueError("Noise schedule does not expose alphas_cumprod for scheduled sampling rollout.")
    # alphas_cumprod may be tensor; convert to numpy
    if torch.is_tensor(alphas_cumprod):
        alphas = alphas_cumprod.detach().cpu().float().numpy()
    else:
        alphas = np.array(alphas_cumprod, dtype=np.float32)

    sigmas = np.sqrt((1.0 - alphas) / np.maximum(alphas, 1e-12)).astype(np.float32)
    timesteps = np.arange(0, len(sigmas), dtype=np.float32)
    schedule = np.stack([timesteps, sigmas], axis=1)
    return schedule, sk_common.sigma_polar


def _mark_schedule_expectation(sampler, expects_pairs: bool):
    """
    Annotate sampler with the schedule structure it expects.
    """
    try:
        setattr(sampler, "_expects_pair_schedule", expects_pairs)
    except Exception:
        pass
    return sampler


def make_sampler(name: str, order: int):
    """
    Factory for skrample structured samplers used in short rollouts.
    """
    name = (name or "unipc").strip().lower()
    order = int(max(1, min(order, 9)))

    structured = None
    try:
        structured = importlib.import_module("skrample.sampling.structured")
    except Exception:
        structured = None

    if structured is not None:
        if name == "unipc" and hasattr(structured, "UniPC"):
            return _mark_schedule_expectation(structured.UniPC(order=order), True)
        if name == "unip" and hasattr(structured, "UniP"):
            return _mark_schedule_expectation(structured.UniP(order=order), True)
        if name == "dpm" and hasattr(structured, "DPM"):
            max_order = min(order, structured.DPM.max_order())
            return _mark_schedule_expectation(structured.DPM(order=max_order), True)
        if name == "rk4":
            try:
                rk_module = importlib.import_module("skrample.sampling.functional")
                if hasattr(rk_module, "RKUltra"):
                    return _mark_schedule_expectation(rk_module.RKUltra(order=min(order, 4)), True)
            except Exception:
                pass
        if hasattr(structured, "Euler"):
            return _mark_schedule_expectation(structured.Euler(), True)

    try:
        sampling_mod = importlib.import_module("skrample.sampling")
    except Exception as exc:
        raise ImportError("No compatible skrample sampler implementations found.") from exc

    if name == "unipc" and hasattr(sampling_mod, "UniPC"):
        return _mark_schedule_expectation(sampling_mod.UniPC(order=order), False)
    if name == "unip" and hasattr(sampling_mod, "UniP"):
        return _mark_schedule_expectation(sampling_mod.UniP(order=order), False)
    if name == "dpm" and hasattr(sampling_mod, "DPM"):
        max_order = order
        if hasattr(sampling_mod.DPM, "max_order"):
            try:
                max_order = min(order, sampling_mod.DPM.max_order())
            except Exception:
                max_order = order
        return _mark_schedule_expectation(sampling_mod.DPM(order=max_order), False)
    if name == "rk4" and hasattr(sampling_mod, "Euler"):
        # No RK4 fallback in legacy module; use Euler.
        return _mark_schedule_expectation(sampling_mod.Euler(), False)
    if hasattr(sampling_mod, "Euler"):
        return _mark_schedule_expectation(sampling_mod.Euler(), False)

    raise ImportError("No compatible skrample sampler implementations found.")
