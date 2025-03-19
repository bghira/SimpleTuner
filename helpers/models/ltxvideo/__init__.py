import torch
import random
from torch import randn as randn_tensor


def normalize_ltx_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float = 1.0,
    reverse=False,
) -> torch.Tensor:
    # Normalize latents across the channel dimension [B, C, F, H, W]
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    if not reverse:
        latents = (latents - latents_mean) * scaling_factor / latents_std
    else:
        latents = latents * latents_std / scaling_factor + latents_mean
    return latents


def unpack_ltx_latents(
    latents: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
    # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
    # what happens in the `_pack_latents` method.
    batch_size = latents.size(0)
    latents = latents.reshape(
        batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size
    )
    latents = (
        latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
        .flatten(6, 7)
        .flatten(4, 5)
        .flatten(2, 3)
    )
    return latents


def pack_ltx_latents(
    latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
    # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
    # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
    # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
    batch_size, num_channels, num_frames, height, width = latents.shape
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


def generate_noise_from_first_unpacked_frame_latent(
    first_frame_latent,
    latent_num_frames,
    vae_spatial_compression_ratio: int = 32,
    vae_temporal_compression_ratio: int = 8,
    generator: torch.Generator = None,
    noise_to_first_frame: float = 0.05,
):
    latent_height, latent_width = first_frame_latent.size(-2), first_frame_latent.size(
        -1
    )
    num_channels_latents = first_frame_latent.size(1)
    batch_size = first_frame_latent.size(0)
    shape = (
        batch_size,
        num_channels_latents,
        latent_num_frames,
        latent_height,
        latent_width,
    )
    mask_shape = (batch_size, 1, latent_num_frames, latent_height, latent_width)
    first_frame_latent = first_frame_latent.repeat(1, 1, latent_num_frames, 1, 1)
    conditioning_mask = torch.zeros(
        mask_shape, device=first_frame_latent.device, dtype=first_frame_latent.dtype
    )
    conditioning_mask[:, :, 0] = 1.0

    rand_noise_ff = random.random() * noise_to_first_frame

    first_frame_mask = conditioning_mask.clone()
    first_frame_mask[:, :, 0] = 1.0 - rand_noise_ff

    noise = randn_tensor(
        shape,
        generator=generator,
        device=first_frame_latent.device,
        dtype=first_frame_latent.dtype,
    )

    latent_noise = first_frame_latent * first_frame_mask + noise * (
        1 - first_frame_mask
    )

    return latent_noise, conditioning_mask
