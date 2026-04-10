"""GroundingCollate: build per-batch grounding tensors from bbox annotations."""

from __future__ import annotations

import logging
import random
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
        grounding_image_cache=None,
        num_frames: int = 1,
    ) -> Optional[GroundingBatch]:
        """Build a ``GroundingBatch`` from per-sample metadata.

        Args:
            num_frames: Number of temporal frames.  When > 1 (video),
                per-entity tensors are expanded to ``(B, T, N, ...)`` using
                the first frame's annotations for all frames.

        Returns ``None`` if no samples in the batch have grounding annotations.
        """
        from simpletuner.helpers.training.state_tracker import StateTracker

        batch_size = len(examples)
        any_has_grounding = False
        per_sample_entities: list[list[dict]] = []

        for example in examples:
            # In the collate, each example IS the image_metadata dict itself
            # (not wrapped in a nested "image_metadata" key).
            metadata = example
            bbox_entities = metadata.get("bbox_entities", [])
            per_sample_entities.append(bbox_entities)
            if bbox_entities:
                any_has_grounding = True

        if not any_has_grounding:
            return None

        N = self.max_entities
        has_image_cache = grounding_image_cache is not None
        image_embed_dim = grounding_image_cache.embed_dim if has_image_cache else None

        all_boxes = []
        all_validity = []
        all_masks = []
        all_text_embeds = []
        all_image_embeds = [] if has_image_cache else None

        for sample_idx, entities in enumerate(per_sample_entities):
            example = examples[sample_idx]
            target_size = example.get("target_size")
            if target_size is None:
                target_size = (512, 512)
            target_w, target_h = target_size
            latent_h = target_h // self.vae_scale_factor
            latent_w = target_w // self.vae_scale_factor

            image_path = example.get("image_path", "")

            sample_boxes = []
            sample_validity = []
            sample_masks = []
            sample_text_embeds = []
            sample_image_embeds = [] if has_image_cache else None

            for ent_idx in range(N):
                if ent_idx < len(entities):
                    entity = entities[ent_idx]
                    bbox = entity.get("bbox", [0, 0, 0, 0])
                    x1, y1, x2, y2 = bbox
                    sample_boxes.append([x1, y1, x2, y2])
                    sample_validity.append(1.0)

                    # Build spatial mask from bbox
                    mask = self._bbox_to_mask(x1, y1, x2, y2, latent_h, latent_w)
                    sample_masks.append(mask)

                    # Retrieve entity text embedding
                    embed = self._get_entity_embed(example, ent_idx, text_embed_cache, data_backend_id)
                    sample_text_embeds.append(embed)

                    # Retrieve entity image embedding if cache is available
                    if has_image_cache:
                        img_embed = grounding_image_cache.retrieve(str(image_path), ent_idx)
                        if img_embed is None:
                            img_embed = torch.zeros(image_embed_dim)
                        sample_image_embeds.append(img_embed)
                else:
                    sample_boxes.append([0.0, 0.0, 0.0, 0.0])
                    sample_validity.append(0.0)
                    sample_masks.append(torch.zeros(latent_h, latent_w))
                    # Use zero embed with same dim as real entities
                    if sample_text_embeds:
                        text_dim = sample_text_embeds[0].shape[-1]
                    else:
                        text_dim = 768
                    sample_text_embeds.append(torch.zeros(text_dim))
                    if has_image_cache:
                        sample_image_embeds.append(torch.zeros(image_embed_dim))

            all_boxes.append(torch.tensor(sample_boxes, dtype=torch.float32))
            all_validity.append(torch.tensor(sample_validity, dtype=torch.float32))
            all_masks.append(torch.stack(sample_masks))
            all_text_embeds.append(torch.stack(sample_text_embeds))
            if has_image_cache:
                all_image_embeds.append(torch.stack(sample_image_embeds))

        boxes = torch.stack(all_boxes)
        validity_mask = torch.stack(all_validity)
        text_masks, image_masks = self._random_drop_features(validity_mask, has_image_cache)
        text_embeds = torch.stack(all_text_embeds)
        image_embeds_tensor = torch.stack(all_image_embeds) if has_image_cache else None

        # Video: expand (B, N, ...) -> (B, T, N, ...) using first-frame annotations
        if num_frames > 1:

            def _expand_temporal(t):
                if t is None:
                    return None
                return t.unsqueeze(1).expand(-1, num_frames, *[-1] * (t.dim() - 1)).contiguous()

            boxes = _expand_temporal(boxes)
            validity_mask = _expand_temporal(validity_mask)
            text_masks = _expand_temporal(text_masks)
            image_masks = _expand_temporal(image_masks)
            text_embeds = _expand_temporal(text_embeds)
            image_embeds_tensor = _expand_temporal(image_embeds_tensor)

        return GroundingBatch(
            boxes=boxes,
            validity_mask=validity_mask,
            spatial_masks=torch.stack(all_masks),
            text_embeds=text_embeds,
            image_embeds=image_embeds_tensor,
            text_masks=text_masks,
            image_masks=image_masks,
            max_entities=N,
            num_frames=num_frames,
        )

    @staticmethod
    def _random_drop_features(validity_mask: torch.Tensor, has_image: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly drop text or image features per entity for CFG.

        For each valid entity, with 50% probability drop either text or image
        (but never both).  Invalid entities get both masks zeroed.
        When there is no image cache, text_masks == validity_mask and
        image_masks is all zeros.
        """
        text_masks = validity_mask.clone()
        image_masks = validity_mask.clone() if has_image else torch.zeros_like(validity_mask)

        if not has_image:
            return text_masks, image_masks

        B, N = validity_mask.shape
        for b in range(B):
            for i in range(N):
                if validity_mask[b, i] > 0 and random.random() < 0.5:
                    if random.random() < 0.5:
                        text_masks[b, i] = 0
                    else:
                        image_masks[b, i] = 0
        return text_masks, image_masks

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
        """Pool text encoder output to a single vector per entity.

        Prefers mean-pooled ``prompt_embeds`` over ``pooled_prompt_embeds``
        because the hidden-state dimension of ``prompt_embeds`` matches the
        model's ``cross_attention_dim`` (and therefore ``position_net``'s
        ``positive_len``), whereas ``pooled_prompt_embeds`` may have a
        different projection size (e.g. 1280 for SDXL vs cross_attention_dim 2048).
        """
        prompt_embeds = text_encoder_output.get("prompt_embeds")
        if prompt_embeds is not None and isinstance(prompt_embeds, torch.Tensor):
            # prompt_embeds shape: (1, seq_len, dim) or (seq_len, dim)
            if prompt_embeds.dim() == 3:
                prompt_embeds = prompt_embeds.squeeze(0)
            attn_mask = text_encoder_output.get("attention_masks")
            if attn_mask is None:
                attn_mask = text_encoder_output.get("attention_mask")
            if attn_mask is None:
                attn_mask = text_encoder_output.get("prompt_attention_mask")
            if attn_mask is not None and isinstance(attn_mask, torch.Tensor):
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.squeeze(0)
                # Weighted mean pooling
                masked = prompt_embeds * attn_mask.unsqueeze(-1).float()
                denom = attn_mask.sum().clamp(min=1.0)
                return masked.sum(dim=0) / denom
            # Simple mean pooling
            return prompt_embeds.mean(dim=0)

        # Fallback to pooled_prompt_embeds if prompt_embeds unavailable
        pooled = text_encoder_output.get("pooled_prompt_embeds")
        if pooled is not None and isinstance(pooled, torch.Tensor):
            return pooled.squeeze(0)

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
