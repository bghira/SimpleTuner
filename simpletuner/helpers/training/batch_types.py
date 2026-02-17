"""Typed batch container for collate_fn output."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional

import torch

from simpletuner.helpers.training.grounding.types import GroundingBatch


@dataclass
class TrainingBatch:
    """Structured representation of a training batch.

    Provides ``__getitem__`` / ``get`` / ``keys`` for backwards compatibility
    with existing code that treats the batch as a plain dict.
    """

    latent_batch: Optional[torch.Tensor]
    latent_metadata: Optional[dict]
    prompts: list[str]
    text_encoder_output: dict
    prompt_embeds: Optional[torch.Tensor]
    add_text_embeds: Optional[torch.Tensor]
    batch_time_ids: Optional[torch.Tensor]
    batch_luminance: Optional[float]
    conditioning_pixel_values: Optional[list]
    conditioning_latents: Optional[list]
    conditioning_image_embeds: Optional[Any]
    conditioning_captions: Optional[list[str]]
    encoder_attention_mask: Optional[torch.Tensor]
    is_regularisation_data: bool
    is_i2v_data: bool
    conditioning_type: Optional[str]
    loss_mask_type: Optional[str]
    audio_latent_batch: Optional[torch.Tensor]
    audio_latent_mask: Optional[torch.Tensor]
    video_latent_mask: Optional[torch.Tensor]
    is_audio_only: bool
    s2v_audio_paths: Optional[list[str]]
    s2v_audio_backend_ids: Optional[list[str]]
    grounding_batch: Optional[GroundingBatch] = None
    slider_strength: Optional[float] = None

    # ------------------------------------------------------------------
    # Dict-like access for backwards compatibility
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> list[str]:
        return [f.name for f in fields(self)]

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)
