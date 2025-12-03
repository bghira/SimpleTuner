import math
from typing import Dict, Optional

import torch

from simpletuner.helpers.training.attention_backend import AttentionBackendController

# Limit key-chunk width when computing max logits to avoid full seq^2 allocations.
_KEY_CHUNK_SIZE = 1024


def _as_bhsd(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Normalize tensor shape to [batch, heads, seq, dim]."""
    if tensor is None or tensor.ndim != 4:
        return None
    b, d1, d2, d3 = tensor.shape
    # Heuristic: smaller of d1/d2 is likely heads.
    if d1 <= d2:
        return tensor
    return tensor.transpose(1, 2)


def _prepare_mask(
    attn_mask: Optional[torch.Tensor],
    batch: int,
    heads: int,
    q_len: int,
    k_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if attn_mask is None:
        return None

    mask = attn_mask
    while mask.ndim < 4:
        mask = mask.unsqueeze(1)

    try:
        mask = mask.expand(batch, heads, q_len, k_len)
    except Exception:
        return None

    if mask.device != device:
        mask = mask.to(device=device)

    return mask.reshape(batch * heads, q_len, k_len)


def _compute_max_logits(
    q_norm: torch.Tensor, k_norm: torch.Tensor, attn_mask: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Compute per-head max logits using chunked matmul to avoid large allocations."""
    batch, heads, q_len, dim = q_norm.shape
    k_len = k_norm.shape[2]
    scale = 1.0 / math.sqrt(max(1, dim))

    q_flat = q_norm.reshape(batch * heads, q_len, dim).float()
    k_flat = k_norm.reshape(batch * heads, k_len, dim).float()
    mask = _prepare_mask(attn_mask, batch, heads, q_len, k_len, device=q_flat.device)

    max_logits: Optional[torch.Tensor] = None
    for k_start in range(0, k_len, _KEY_CHUNK_SIZE):
        k_end = min(k_len, k_start + _KEY_CHUNK_SIZE)
        logits_chunk = torch.bmm(q_flat, k_flat[:, k_start:k_end, :].transpose(1, 2))
        logits_chunk.mul_(scale)

        if mask is not None:
            mask_chunk = mask[:, :, k_start:k_end]
            if mask_chunk.dtype == torch.bool:
                logits_chunk = logits_chunk.masked_fill(~mask_chunk, float("-inf"))
            else:
                logits_chunk = logits_chunk + mask_chunk

        chunk_max = logits_chunk.amax(dim=2)
        max_logits = chunk_max if max_logits is None else torch.maximum(max_logits, chunk_max)

    if max_logits is None:
        return None

    max_logits = torch.nan_to_num(max_logits, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))
    return max_logits.view(batch, heads, q_len).amax(dim=(0, 2))


@torch.no_grad()
def publish_attention_max_logits(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    q_param: Optional[torch.nn.Parameter],
    k_param: Optional[torch.nn.Parameter],
) -> None:
    """
    Publish per-head max logits for QK-Clip if an active MuonClip optimizer is registered.
    """
    q_name = AttentionBackendController.lookup_param_name(q_param) if q_param is not None else ""
    k_name = AttentionBackendController.lookup_param_name(k_param) if k_param is not None else ""
    if not q_name and not k_name:
        return

    q_norm = _as_bhsd(query)
    k_norm = _as_bhsd(key)
    if q_norm is None or k_norm is None:
        return

    max_per_head = _compute_max_logits(q_norm, k_norm, attn_mask)
    if max_per_head is None:
        return

    payload: Dict[str, torch.Tensor] = {}
    if q_name:
        payload[q_name] = max_per_head
    if k_name:
        payload[k_name] = max_per_head

    if payload:
        AttentionBackendController.publish_attention_max_logits(payload)
