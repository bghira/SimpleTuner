import numpy as np
import torch
from skrample import common as sk_common
from skrample.sampling import structured as sk_structured


def make_sigma_schedule_from_ddpm(noise_schedule) -> tuple[list[tuple[float, float]], sk_common.SigmaTransform]:
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

    sigmas = np.sqrt((1.0 - alphas) / np.maximum(alphas, 1e-12))
    timesteps = np.arange(0, len(sigmas), dtype=np.float32)
    schedule = list(zip(timesteps.tolist(), sigmas.tolist()))
    return schedule, sk_common.sigma_polar


def make_sampler(name: str, order: int):
    """
    Factory for skrample structured samplers used in short rollouts.
    """
    name = (name or "unipc").strip().lower()
    order = int(max(1, min(order, 9)))
    if name == "unipc":
        return sk_structured.UniPC(order=order)
    if name == "unip":
        return sk_structured.UniP(order=order)
    if name == "dpm":
        return sk_structured.DPM(order=min(order, sk_structured.DPM.max_order()))
    if name == "rk4":
        from skrample.sampling.functional import RKUltra

        return RKUltra(order=min(order, 4))
    # Fallback to Euler
    return sk_structured.Euler()
