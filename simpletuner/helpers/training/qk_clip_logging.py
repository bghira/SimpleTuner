import math
from typing import Dict, Optional

import torch

from simpletuner.helpers.training.attention_backend import AttentionBackendController


def _as_bhsd(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Normalize tensor shape to [batch, heads, seq, dim]."""
    if tensor is None or tensor.ndim != 4:
        return None
    b, d1, d2, d3 = tensor.shape
    # Heuristic: smaller of d1/d2 is likely heads.
    if d1 <= d2:
        return tensor
    return tensor.transpose(1, 2)


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

    logits = torch.matmul(q_norm.float(), k_norm.float().transpose(-2, -1))
    logits.div_(math.sqrt(max(1, q_norm.shape[-1])))

    if attn_mask is not None:
        mask = attn_mask
        if mask.dtype == torch.bool:
            logits = logits.masked_fill(~mask, float("-inf"))
        else:
            logits = logits + mask

    # Reduce over sequence and batch; keep per-head maxima.
    logits = torch.nan_to_num(logits, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))
    max_per_head = logits.amax(dim=(-1, -2))
    if max_per_head.dim() == 2:
        max_per_head = max_per_head.amax(dim=0)

    payload: Dict[str, torch.Tensor] = {}
    if q_name:
        payload[q_name] = max_per_head
    if k_name:
        payload[k_name] = max_per_head

    if payload:
        AttentionBackendController.publish_attention_max_logits(payload)
