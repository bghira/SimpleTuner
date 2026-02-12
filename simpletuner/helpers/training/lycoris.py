"""T-LoRA (Timestep-dependent LoRA) helpers for LyCORIS integration."""

import torch

try:
    from lycoris.modules.tlora import clear_timestep_mask, compute_timestep_mask, set_timestep_mask

    TLORA_AVAILABLE = True
except ImportError:
    TLORA_AVAILABLE = False


def apply_tlora_timestep_mask(
    timesteps: torch.Tensor,
    max_timestep: int,
    max_rank: int,
    min_rank: int = 1,
    alpha: float = 1.0,
) -> None:
    """Compute per-sample timestep masks and set them for T-LoRA modules.

    Args:
        timesteps: Batch of timesteps, shape (batch,).
        max_timestep: Maximum scheduler timestep (e.g. 1000).
        max_rank: LoRA rank dimension (linear_dim from config).
        min_rank: Minimum active ranks at highest noise.
        alpha: Masking exponent (1.0 = linear).
    """
    masks = torch.stack(
        [compute_timestep_mask(int(t), max_timestep, max_rank, min_rank, alpha).squeeze(0) for t in timesteps.tolist()]
    )  # (batch, max_rank)
    set_timestep_mask(masks)


def apply_tlora_inference_mask(
    timestep: int,
    max_timestep: int,
    max_rank: int,
    min_rank: int = 1,
    alpha: float = 1.0,
) -> None:
    """Compute and set a T-LoRA mask for a single inference timestep.

    Unlike apply_tlora_timestep_mask (which takes a batch), this handles
    the single-scalar-timestep case in pipeline denoising loops.
    """
    mask = compute_timestep_mask(timestep, max_timestep, max_rank, min_rank, alpha)  # (1, max_rank)
    set_timestep_mask(mask)


def clear_tlora_mask() -> None:
    """Clear the T-LoRA timestep mask after forward pass."""
    clear_timestep_mask()
