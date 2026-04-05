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

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "GroundingBatch":
        """Return a new ``GroundingBatch`` with all tensors moved to *device*/*dtype*."""

        def _move(t: torch.Tensor | None) -> torch.Tensor | None:
            if t is None:
                return None
            return t.to(device=device, dtype=dtype) if dtype is not None else t.to(device=device)

        return GroundingBatch(
            boxes=_move(self.boxes),
            validity_mask=_move(self.validity_mask),
            spatial_masks=_move(self.spatial_masks),
            text_embeds=_move(self.text_embeds),
            image_embeds=_move(self.image_embeds),
            text_masks=_move(self.text_masks),
            image_masks=_move(self.image_masks),
            max_entities=self.max_entities,
            num_frames=self.num_frames,
        )
