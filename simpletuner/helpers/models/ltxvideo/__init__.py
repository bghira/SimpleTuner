import random
from typing import Optional, Tuple

import torch
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
        latents = (latents.float() - latents_mean) * scaling_factor / latents_std
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
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def pack_ltx_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
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


def apply_first_frame_protection(
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    i2v_conditioning_mask: torch.Tensor,
    protect_first_frame: bool = False,
    first_frame_probability: float = 0.0,
    partial_noise_fraction: float = 0.05,
    return_sigmas: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optionally protect the first frame in a video-like latent tensor, either completely
    or probabilistically.

    Args:
        latents (torch.Tensor):
            The clean latents of shape [B, C, T, H, W].
        timesteps (torch.Tensor):
            The “time” or step-level for each frame, typically derived from sigmas
            via: timesteps = (sigmas * 1000).long() or float.
            Expected shape could be [B], [B, T], or broadcasted [B, 1, T, H, W].
        noise (torch.Tensor):
            The random noise tensor, same shape as latents ([B, C, T, H, W]).
        i2v_conditioning_mask (torch.Tensor):
            A 5D mask [B, 1, T, H, W] that indicates which frames/pixels to protect (1.0)
            vs. not protect (0.0). Typically, i2v logic sets the first frame to 1.0 if
            you want to “preserve” it.
        protect_first_frame (bool):
            If True, always protect the first frame (set timesteps=0 there).
            If False, do not guarantee full protection (see probability).
        first_frame_probability (float):
            A probability in [0, 1] for applying partial protection.
            If random.random() < first_frame_probability, we do a partial noise
            mix on the first frame.
        partial_noise_fraction (float):
            The maximum fraction of noise to introduce in the first frame when
            partial protection is applied. (e.g. 0.05 => up to 5% noise)
        return_sigmas (bool):
            If True, we also return the sigmas after converting timesteps / 1000.
            Otherwise, return None for that slot.

    Returns:
        updated_timesteps (torch.Tensor):
            Possibly masked or partially zeroed timesteps. Same shape as input `timesteps` (after broadcasting).
        updated_noise (torch.Tensor):
            If partial protection is triggered, the first frame in `noise` might get scaled down.
            (If complete protection, you could leave it as-is or zero it for the protected frame.)
        sigmas (Optional[torch.Tensor]):
            If `return_sigmas=True`, this is timesteps.float()/1000.0;
            otherwise, None. Shape matches updated_timesteps.

    Example:
        updated_t, updated_n, s = apply_first_frame_protection(
            latents=latents,
            timesteps=timesteps,
            noise=noise,
            i2v_conditioning_mask=i2v_mask,
            protect_first_frame=True,
            first_frame_probability=0.1,
            partial_noise_fraction=0.05,
        )
    """
    # Make sure timesteps is at least 5D if we want to broadcast it with [B,1,T,H,W].
    # For example, if timesteps is [B], reshape to [B, 1, 1, 1, 1].
    # If timesteps is [B, T], reshape to [B, 1, T, 1, 1], etc.
    # Below is an example if we assume [B] => [B,1,1,1,1].
    if timesteps.ndim == 1:
        bsz = timesteps.shape[0]
        # shape: [B, 1, 1, 1, 1]
        timesteps = timesteps.view(bsz, 1, 1, 1, 1)
    # If you have [B, T], do a different reshape:
    # elif timesteps.ndim == 2:
    #     bsz, t = timesteps.shape
    #     timesteps = timesteps.view(bsz, 1, t, 1, 1)
    # etc.

    # We'll copy noise so we can modify the first frame if partial protection is triggered
    updated_noise = noise.clone()

    # 1. Decide if partial protection triggers
    do_partial = (not protect_first_frame) and (random.random() < first_frame_probability)

    if protect_first_frame:
        # Completely zero out timesteps where i2v_conditioning_mask=1
        updated_timesteps = timesteps * (1 - i2v_conditioning_mask)
        # Optionally also zero out the noise in the protected frames:
        # updated_noise = updated_noise * (1 - i2v_conditioning_mask)
    elif do_partial:
        # PARTIAL PROTECTION => only add partial_noise_fraction * random() to the first frame
        # e.g., if i2v_mask for the first frame is 1.0, let's reduce timesteps or noise for that frame
        # Usually, partial approach might be done at the noise level rather than timesteps,
        # so that we don't disrupt the entire schedule.
        # We'll do: noise_first_frame = alpha * noise + (1-alpha) * latents
        # or timesteps_first_frame = timesteps * something.
        # One approach:
        rand_noise_ff = random.random() * partial_noise_fraction  # e.g. up to 5%
        alpha_mask = 1.0 - (i2v_conditioning_mask * rand_noise_ff)

        # Multiply timesteps by alpha_mask => reduces timesteps in the protected region
        updated_timesteps = timesteps * alpha_mask
        # Or scale the noise in that region
        updated_noise = updated_noise * alpha_mask
    else:
        # No protection at all => leave timesteps as is
        updated_timesteps = timesteps

    # Convert timesteps back to sigmas if requested
    # If timesteps was an integer approximation, you lose decimal precision
    sigmas = None
    if return_sigmas:
        sigmas = updated_timesteps.float() / 1000.0

    return updated_timesteps, updated_noise, sigmas


def make_i2v_conditioning_mask(latents: torch.Tensor, protect_frame_index: int = 0) -> torch.Tensor:
    """
    Create a mask that is 1.0 at the given 'protect_frame_index' frame (e.g., the first frame),
    and 0.0 elsewhere.

    Args:
        latents (torch.Tensor): The latents of shape [B, C, T, H, W].
        protect_frame_index (int): Which frame to protect (default=0 => the very first frame).

    Returns:
        torch.Tensor: An i2v conditioning mask of shape [B, 1, T, H, W].
                      The selected frame is set to 1.0, all others 0.0.
    """
    bsz, _, num_frames, height, width = latents.shape
    mask = torch.zeros((bsz, 1, num_frames, height, width), dtype=latents.dtype, device=latents.device)
    if protect_frame_index < num_frames:
        mask[:, :, protect_frame_index, :, :] = 1.0
    return mask
