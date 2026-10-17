"""FLUX.3 image-model integration."""

import torch

from simpletuner.helpers.models.flux2.latents import denormalize_latents, normalize_latents, patchify, unpatchify


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
