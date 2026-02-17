"""GroundingCollate: build per-batch grounding tensors from bbox annotations."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from simpletuner.helpers.training.grounding.types import GroundingBatch

logger = logging.getLogger(__name__)


class GroundingCollate:
    """Collate grounding annotations into padded batch tensors."""

    def __init__(self, max_entities: int, vae_scale_factor: int):
        self.max_entities = max_entities
        self.vae_scale_factor = vae_scale_factor

    def build_batch(
        self,
        examples: list[dict],
        data_backend_id: str,
        text_embed_cache,
    ) -> Optional[GroundingBatch]:
        """Build a ``GroundingBatch`` from per-sample metadata.

        Returns ``None`` if no samples in the batch have grounding annotations.
        """
        from simpletuner.helpers.training.state_tracker import StateTracker

        batch_size = len(examples)
        any_has_grounding = False
        per_sample_entities: list[list[dict]] = []

        for example in examples:
            metadata = example.get("image_metadata") or {}
            if isinstance(metadata, str):
                # metadata might be a path; resolve via StateTracker
                metadata = StateTracker.get_metadata_by_filepath(metadata, data_backend_id) or {}
            bbox_entities = metadata.get("bbox_entities", [])
            per_sample_entities.append(bbox_entities)
            if bbox_entities:
                any_has_grounding = True

        if not any_has_grounding:
            return None

        N = self.max_entities
        all_boxes = []
        all_validity = []
        all_masks = []
        all_text_embeds = []

        for sample_idx, entities in enumerate(per_sample_entities):
            example = examples[sample_idx]
            target_size = (example.get("image_metadata") or {}).get("target_size")
            if target_size is None:
                target_size = (512, 512)
            target_w, target_h = target_size
            latent_h = target_h // self.vae_scale_factor
            latent_w = target_w // self.vae_scale_factor

            sample_boxes = []
            sample_validity = []
            sample_masks = []
            sample_embeds = []

            for ent_idx in range(N):
                if ent_idx < len(entities):
                    entity = entities[ent_idx]
                    bbox = entity.get("bbox", [0, 0, 0, 0])
                    x1, y1, x2, y2 = bbox
                    sample_boxes.append([x1, y1, x2, y2])
                    sample_validity.append(1.0)

                    # Build spatial mask from bbox (mask_path loading is deferred to future phases)
                    mask = self._bbox_to_mask(x1, y1, x2, y2, latent_h, latent_w)
                    sample_masks.append(mask)

                    # Retrieve entity text embedding
                    embed = self._get_entity_embed(example, ent_idx, text_embed_cache, data_backend_id)
                    sample_embeds.append(embed)
                else:
                    sample_boxes.append([0.0, 0.0, 0.0, 0.0])
                    sample_validity.append(0.0)
                    sample_masks.append(torch.zeros(latent_h, latent_w))
                    # Use zero embed with same dim as real entities
                    if sample_embeds:
                        embed_dim = sample_embeds[0].shape[-1]
                    else:
                        embed_dim = 768
                    sample_embeds.append(torch.zeros(embed_dim))

            all_boxes.append(torch.tensor(sample_boxes, dtype=torch.float32))
            all_validity.append(torch.tensor(sample_validity, dtype=torch.float32))
            all_masks.append(torch.stack(sample_masks))
            all_text_embeds.append(torch.stack(sample_embeds))

        return GroundingBatch(
            boxes=torch.stack(all_boxes),
            validity_mask=torch.stack(all_validity),
            spatial_masks=torch.stack(all_masks),
            text_embeds=torch.stack(all_text_embeds),
            max_entities=N,
        )

    def _bbox_to_mask(self, x1: float, y1: float, x2: float, y2: float, h: int, w: int) -> torch.Tensor:
        """Create a binary mask at latent resolution from normalized XYXY coords."""
        mask = torch.zeros(h, w, dtype=torch.float32)
        r1 = int(y1 * h)
        r2 = max(int(y2 * h), r1 + 1)
        c1 = int(x1 * w)
        c2 = max(int(x2 * w), c1 + 1)
        mask[r1:r2, c1:c2] = 1.0
        return mask

    def _get_entity_embed(
        self,
        example: dict,
        entity_idx: int,
        text_embed_cache,
        data_backend_id: str,
    ) -> torch.Tensor:
        """Retrieve a pooled text embedding for a single entity label."""
        from simpletuner.helpers.training.state_tracker import StateTracker
        from simpletuner.helpers.utils.pathing import normalize_data_path

        image_path = example.get("image_path", "")
        dataset_root = StateTracker.get_data_backend(data_backend_id).get("instance_data_dir")
        normalized_id = normalize_data_path(str(image_path), dataset_root)
        entity_key = f"{normalized_id}__bbox_{entity_idx}"

        try:
            text_encoder_output = text_embed_cache.compute_prompt_embeddings_with_model(
                prompt_records=[{"prompt": "", "key": entity_key, "metadata": {}}],
            )
            return self._pool_text_encoder_output(text_encoder_output)
        except Exception as exc:
            logger.debug(f"Failed to retrieve entity embed for key {entity_key}: {exc}")
            return torch.zeros(768)

    @staticmethod
    def _pool_text_encoder_output(text_encoder_output: dict) -> torch.Tensor:
        """Pool text encoder output to a single vector per entity."""
        pooled = text_encoder_output.get("pooled_prompt_embeds")
        if pooled is not None:
            if isinstance(pooled, torch.Tensor):
                return pooled.squeeze(0)

        prompt_embeds = text_encoder_output.get("prompt_embeds")
        if prompt_embeds is None:
            return torch.zeros(768)

        if isinstance(prompt_embeds, torch.Tensor):
            # prompt_embeds shape: (1, seq_len, dim) or (seq_len, dim)
            if prompt_embeds.dim() == 3:
                prompt_embeds = prompt_embeds.squeeze(0)
            attn_mask = text_encoder_output.get("attention_masks")
            if attn_mask is not None and isinstance(attn_mask, torch.Tensor):
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.squeeze(0)
                # Weighted mean pooling
                masked = prompt_embeds * attn_mask.unsqueeze(-1).float()
                denom = attn_mask.sum().clamp(min=1.0)
                return masked.sum(dim=0) / denom
            # Simple mean pooling
            return prompt_embeds.mean(dim=0)

        return torch.zeros(768)

    @staticmethod
    def downsample_mask(mask: torch.Tensor, vae_scale_factor: int) -> torch.Tensor:
        """Downsample a spatial mask to latent resolution via bilinear interpolation."""
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
        h = max(1, mask.shape[-2] // vae_scale_factor)
        w = max(1, mask.shape[-1] // vae_scale_factor)
        return F.interpolate(mask.float(), size=(h, w), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
