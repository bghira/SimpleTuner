import math
import random

import torch
from diffusers.pipelines.flux.pipeline_flux import calculate_shift as calculate_shift_flux

from simpletuner.helpers.models.flux.pipeline import FluxPipeline
from simpletuner.helpers.training import steps_remaining_in_epoch


def update_flux_schedule_to_fast(args, noise_scheduler_to_copy):
    if args.flux_fast_schedule and args.model_family.lower() == "flux":
        # 4-step noise schedule [0.7, 0.1, 0.1, 0.1] from SD3-Turbo paper
        for i in range(0, 250):
            noise_scheduler_to_copy.sigmas[i] = 1.0
        for i in range(250, 500):
            noise_scheduler_to_copy.sigmas[i] = 0.3
        for i in range(500, 750):
            noise_scheduler_to_copy.sigmas[i] = 0.2
        for i in range(750, 1000):
            noise_scheduler_to_copy.sigmas[i] = 0.1
    return noise_scheduler_to_copy


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents


def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

    return latents


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
    latent_image_ids = latent_image_ids.reshape(
        batch_size,
        latent_image_id_height * latent_image_id_width,
        latent_image_id_channels,
    )

    return latent_image_ids.to(device=device, dtype=torch.float32)[0]


def build_kontext_inputs(
    cond_latents: list[torch.Tensor] | torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    latent_channels: int,
):
    """
    Args
    ----
    cond_latents : list of (B, C, H, W) or single (B, C, H, W) tensor
                   Each tensor is already VAE-encoded by VAECache
    dtype        : dtype to use (match main latents)
    device       : target device
    latent_channels: number of channels in the latent (16 for Flux)

    Returns
    -------
    packed_cond : (B, S, C*4)   – flattened patch sequence
    cond_ids    : (B, S, 3)     – seq-ids with id[...,0] == 1
    """
    # Handle different input formats
    if isinstance(cond_latents, torch.Tensor):
        # Single tensor, treat as one conditioning image/batch
        cond_latents = [cond_latents]
    elif isinstance(cond_latents, list) and len(cond_latents) == 1:
        # List with single element - no change needed
        pass
    elif isinstance(cond_latents, list):
        # Multiple tensors in list
        # Check if they're batched (all have same batch size)
        batch_sizes = [t.shape[0] if len(t.shape) == 4 else 1 for t in cond_latents]

        if all(bs == batch_sizes[0] for bs in batch_sizes):
            # All have same batch size - this is genuine batched training
            # Keep as list - each element is a different conditioning image set
            pass
        else:
            raise ValueError(f"Inconsistent batch sizes in conditioning latents: {batch_sizes}")

    packed_cond = []
    packed_ids = []

    # Get batch size from first tensor
    first_tensor = cond_latents[0]
    if len(first_tensor.shape) == 3:
        # Single image without batch dim
        batch_size = 1
    else:
        batch_size = first_tensor.shape[0]

    # Process each conditioning tensor
    # this coordinate offsetting algorithm follows the comfyui implementation
    # so that multi-image loras will behave the same there
    # this indexing scheme minimizes max(max_x, max_y)
    x0 = 0
    y0 = 0

    for latent in cond_latents:
        # Ensure 4D shape (B, C, H, W)
        if len(latent.shape) == 3:
            if latent.shape[0] == latent_channels:
                # Shape is (C, H, W) - add batch dimension
                latent = latent.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected 3D tensor shape: {latent.shape}")

        B, C, H, W = latent.shape

        # Verify batch size consistency
        if B != batch_size:
            raise ValueError(
                f"Batch size mismatch: expected {batch_size}, got {B}. "
                f"All conditioning latents must have the same batch size."
            )

        # Verify channel count
        if C != latent_channels:
            raise ValueError(f"Channel mismatch: expected {latent_channels}, got {C}")

        # Pack this conditioning image/batch
        packed_cond.append(pack_latents(latent, B, C, H, W).to(device=device, dtype=dtype))

        # Compute spatial IDs
        x = 0
        y = 0
        if H + y0 > W + x0:
            x = x0
        else:
            y = y0

        # seq-ids: flag-channel==1, rest is y/x indices
        idx_y = torch.arange(H // 2, device=device) + y // 2
        idx_x = torch.arange(W // 2, device=device) + x // 2
        ids = torch.stack(torch.meshgrid(idx_y, idx_x, indexing="ij"), dim=-1)  # (H/2,W/2,2)
        ones = torch.ones_like(ids[..., :1])

        # Shape: (1, H/2*W/2, 3) -> expand to (B, H/2*W/2, 3)
        packed_ids.append(torch.cat([ones, ids], dim=-1).view(1, -1, 3).expand(B, -1, -1).to(dtype))

        x0 = max(x0, W + x)
        y0 = max(y0, H + y)

    # Concatenate along sequence dimension
    packed_cond = torch.cat(packed_cond, dim=1)  # (B, total_seq, C*4)
    packed_ids = torch.cat(packed_ids, dim=1)  # (B, total_seq, 3)

    return packed_cond, packed_ids
