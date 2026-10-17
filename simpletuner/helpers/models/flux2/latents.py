"""Pixel-shuffle packing and batch-normalization transforms for the FLUX.2 VAE."""

import torch
from einops import rearrange


def patchify(latents: torch.Tensor) -> torch.Tensor:
    return rearrange(latents, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=2, pw=2)


def unpatchify(latents: torch.Tensor) -> torch.Tensor:
    return rearrange(latents, "b (c ph pw) h w -> b c (h ph) (w pw)", ph=2, pw=2)


def normalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    mean = vae.bn.running_mean.to(latents).view(1, -1, 1, 1)
    std = (vae.bn.running_var.float() + vae.config.batch_norm_eps).sqrt().to(latents).view(1, -1, 1, 1)
    return (latents - mean) / std


def denormalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    mean = vae.bn.running_mean.to(latents).view(1, -1, 1, 1)
    std = (vae.bn.running_var.float() + vae.config.batch_norm_eps).sqrt().to(latents).view(1, -1, 1, 1)
    return latents * std + mean
