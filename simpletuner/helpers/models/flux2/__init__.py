# FLUX.2 model implementation for SimpleTuner
# Based on Black Forest Labs FLUX.2-dev architecture

from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor


def compress_time(t_ids: Tensor) -> Tensor:
    """Compress time IDs to contiguous indices."""
    assert t_ids.ndim == 1
    t_ids_max = torch.max(t_ids)
    t_remap = torch.zeros((t_ids_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    t_unique_sorted_ids = torch.unique(t_ids, sorted=True)
    t_remap[t_unique_sorted_ids] = torch.arange(len(t_unique_sorted_ids), device=t_ids.device, dtype=t_ids.dtype)
    t_ids_compressed = t_remap[t_ids]
    return t_ids_compressed


def pack_latents(latents: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Pack latents from (B, C, H, W) to (B, H*W, C) with position IDs.

    Returns:
        packed: Tensor of shape (B, S, C) where S = H * W
        ids: Tensor of shape (B, S, 4) containing (t, h, w, l) coordinates
    """
    batch_size = latents.shape[0]
    results = []

    for i in range(batch_size):
        x = latents[i]  # (C, H, W)
        c, h, w = x.shape

        # Create position IDs
        coords = {
            "t": torch.arange(1, device=x.device),
            "h": torch.arange(h, device=x.device),
            "w": torch.arange(w, device=x.device),
            "l": torch.arange(1, device=x.device),
        }
        x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])

        # Reshape latents
        x_flat = rearrange(x, "c h w -> (h w) c")
        results.append((x_flat, x_ids))

    packed = torch.stack([r[0] for r in results])
    ids = torch.stack([r[1] for r in results])

    return packed, ids


def unpack_latents(packed: Tensor, ids: Tensor) -> Tensor:
    """
    Unpack latents from (B, S, C) back to (B, C, H, W).

    Args:
        packed: Tensor of shape (B, S, C)
        ids: Tensor of shape (B, S, 4) containing (t, h, w, l) coordinates

    Returns:
        Tensor of shape (B, C, T, H, W) squeezed to (B, C, H, W) if T=1
    """
    x_list = []
    for data, pos in zip(packed, ids):
        _, ch = data.shape
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_ids_cmpr = compress_time(t_ids)

        t = torch.max(t_ids_cmpr) + 1
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = t_ids_cmpr * w * h + h_ids * w + w_ids

        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        x_list.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))

    result = torch.cat(x_list, dim=0)
    # Squeeze time dimension if t=1
    if result.shape[2] == 1:
        result = result.squeeze(2)
    return result


def pack_text(text_embeds: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Pack text embeddings with position IDs.

    Args:
        text_embeds: Tensor of shape (B, L, D)

    Returns:
        text: Same tensor
        ids: Tensor of shape (B, L, 4) containing (t, h, w, l) coordinates
    """
    batch_size, seq_len, _ = text_embeds.shape
    device = text_embeds.device

    results = []
    for i in range(batch_size):
        coords = {
            "t": torch.arange(1, device=device),
            "h": torch.arange(1, device=device),  # dummy
            "w": torch.arange(1, device=device),  # dummy
            "l": torch.arange(seq_len, device=device),
        }
        x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
        results.append(x_ids)

    ids = torch.stack(results)
    return text_embeds, ids
