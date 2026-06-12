from __future__ import annotations

import torch

from simpletuner.helpers.models.zlab_i1.transformer import FLUX2_LATENTS_MEAN, FLUX2_LATENTS_VAR


def pixel_unshuffle_2x(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"i1 latents require even spatial dimensions, got {(height, width)}.")
    latents = latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
    return latents.permute(0, 1, 3, 5, 2, 4).reshape(batch, channels * 4, height // 2, width // 2)


def pixel_shuffle_2x(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    if channels % 4 != 0:
        raise ValueError(f"i1 pixel-shuffle expects channel count divisible by 4, got {channels}.")
    latents = latents.reshape(batch, channels // 4, 2, 2, height, width)
    return latents.permute(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, height * 2, width * 2)


def flux2_latent_stats(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(FLUX2_LATENTS_MEAN, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
    var = torch.tensor(FLUX2_LATENTS_VAR, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
    return mean, torch.sqrt(var + 0.0001)


def normalize_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    packed = pixel_unshuffle_2x(latents)
    mean, std = flux2_latent_stats(packed)
    return pixel_shuffle_2x((packed - mean) / std)


def unscale_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    packed = pixel_unshuffle_2x(latents)
    mean, std = flux2_latent_stats(packed)
    return pixel_shuffle_2x(packed * std + mean)
