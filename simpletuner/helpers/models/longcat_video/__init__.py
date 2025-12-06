import torch


def normalize_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
    """
    Map VAE latents into the diffusion space using the repository-provided mean and std.
    """
    return (latents - latents_mean) * latents_std


def denormalize_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
    """
    Map diffusion-space latents back to the VAE space for decoding.
    """
    return latents / latents_std + latents_mean


def pack_video_latents(latents: torch.Tensor, latents_mean: torch.Tensor = None, latents_std: torch.Tensor = None):
    """
    Prepare latents for the transformer. If statistics are provided, latents are normalized; otherwise, they are
    returned unchanged.
    """
    if latents_mean is not None and latents_std is not None:
        return normalize_latents(latents, latents_mean, latents_std)
    return latents


def unpack_video_latents(latents: torch.Tensor, latents_mean: torch.Tensor = None, latents_std: torch.Tensor = None):
    """
    Prepare latents for decoding or preview callbacks. If statistics are provided, latents are denormalized.
    """
    if latents_mean is not None and latents_std is not None:
        return denormalize_latents(latents, latents_mean, latents_std)
    return latents


def optimized_scale(positive_flat: torch.Tensor, negative_flat: torch.Tensor):
    """
    CFG-zero scaling from the LongCat-Video reference implementation.
    """
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    return dot_product / squared_norm
