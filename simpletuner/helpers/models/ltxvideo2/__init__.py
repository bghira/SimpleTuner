from typing import Optional, Tuple

import torch


def normalize_ltx2_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float = 1.0,
    reverse: bool = False,
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    if reverse:
        return latents * latents_std / scaling_factor + latents_mean
    return (latents.float() - latents_mean) * scaling_factor / latents_std


def normalize_ltx2_audio_latents(
    latents: torch.Tensor,
    audio_vae,
    reverse: bool = False,
) -> torch.Tensor:
    if hasattr(audio_vae, "per_channel_statistics"):
        if reverse:
            return audio_vae.per_channel_statistics.un_normalize(latents)
        return latents
    latents_mean = getattr(audio_vae, "latents_mean", None)
    latents_std = getattr(audio_vae, "latents_std", None)
    if latents_mean is None or latents_std is None:
        return latents
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    if reverse:
        return latents * latents_std + latents_mean
    return (latents - latents_mean) / latents_std


def pack_ltx2_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    batch_size, _, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def unpack_ltx2_latents(
    latents: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def pack_ltx2_audio_latents(
    latents: torch.Tensor, patch_size: Optional[int] = None, patch_size_t: Optional[int] = None
) -> torch.Tensor:
    if patch_size is not None and patch_size_t is not None:
        batch_size, _, latent_length, latent_mel_bins = latents.shape
        post_patch_latent_length = latent_length / patch_size_t
        post_patch_mel_bins = latent_mel_bins / patch_size
        latents = latents.reshape(batch_size, -1, post_patch_latent_length, patch_size_t, post_patch_mel_bins, patch_size)
        latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        return latents
    return latents.transpose(1, 2).flatten(2, 3)


def unpack_ltx2_audio_latents(
    latents: torch.Tensor,
    latent_length: int,
    num_mel_bins: int,
    patch_size: Optional[int] = None,
    patch_size_t: Optional[int] = None,
) -> torch.Tensor:
    if patch_size is not None and patch_size_t is not None:
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, latent_length, num_mel_bins, -1, patch_size_t, patch_size)
        latents = latents.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
        return latents
    return latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)


__all__ = [
    "normalize_ltx2_audio_latents",
    "normalize_ltx2_latents",
    "pack_ltx2_audio_latents",
    "pack_ltx2_latents",
    "unpack_ltx2_audio_latents",
    "unpack_ltx2_latents",
]
