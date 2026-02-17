"""Core dataclasses for the grounding pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BboxEntity:
    """A single entity annotation: label + bounding box + optional mask."""

    label: str  # entity noun phrase
    bbox: tuple[float, float, float, float]  # normalized XYXY (0-1)
    mask_path: Optional[str] = None  # path to mask image (if provided)


@dataclass
class GroundingBatch:
    """Collated grounding data for a training batch.

    Produced by ``GroundingCollate.build_batch`` and consumed by the
    training loop / ``model.prepare_batch``.
    """

    boxes: torch.Tensor  # (B, N, 4) normalized XYXY
    validity_mask: torch.Tensor  # (B, N) which entity slots are real
    spatial_masks: torch.Tensor  # (B, N, H_latent, W_latent) per-entity masks
    text_embeds: torch.Tensor  # (B, N, D_text) pooled per-entity text features
    image_embeds: Optional[torch.Tensor]  # (B, N, D_img) Florence-2 entity crop features
    text_masks: torch.Tensor  # (B, N) per-entity text feature mask after random drop
    image_masks: torch.Tensor  # (B, N) per-entity image feature mask after random drop
    max_entities: int
    num_frames: int = 1  # 1 for images; T for video (shapes become (B, T, N, ...))
