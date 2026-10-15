"""FLUX.3 image-model integration."""

import torch
from einops import rearrange


def patchify(latents: torch.Tensor) -> torch.Tensor:
    return rearrange(latents, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=2, pw=2)


def unpatchify(latents: torch.Tensor) -> torch.Tensor:
    return rearrange(latents, "b (c ph pw) h w -> b c (h ph) (w pw)", ph=2, pw=2)


def image_ids(height: int, width: int, device, reference_index: int | None = None) -> torch.Tensor:
    rows, columns = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    ids = torch.zeros(height, width, 4, device=device, dtype=torch.float32)
    ids[..., 1] = rows * 0.5
    ids[..., 2] = columns * 0.5
    if reference_index is not None:
        ids[..., 3] = -1 - reference_index
    return ids.reshape(-1, 4)


def text_ids(length: int, device) -> torch.Tensor:
    ids = torch.zeros(length, 4, device=device, dtype=torch.float32)
    ids[:, 3] = torch.arange(length, device=device)
    return ids


def normalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    mean = vae.bn.running_mean.to(latents).view(1, -1, 1, 1)
    std = (vae.bn.running_var.float() + vae.config.batch_norm_eps).sqrt().to(latents).view(1, -1, 1, 1)
    return (latents - mean) / std


def denormalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    mean = vae.bn.running_mean.to(latents).view(1, -1, 1, 1)
    std = (vae.bn.running_var.float() + vae.config.batch_norm_eps).sqrt().to(latents).view(1, -1, 1, 1)
    return latents * std + mean
