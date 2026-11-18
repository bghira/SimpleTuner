"""Cache Florence-2 (or other backend) image features for grounding entity crops."""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

import numpy as np
import torch
from atomicwrites import atomic_write
from PIL import Image

from simpletuner.helpers.training.grounding.feature_backend import GroundingFeatureBackend

logger = logging.getLogger(__name__)


class GroundingImageEmbedCache:
    """Pre-compute and cache per-entity image features from bbox annotations.

    For each image with ``.bbox`` annotations, crops each entity region and
    encodes it via the provided ``feature_backend``.  Results are stored as
    individual ``.pt`` files under ``cache_dir``.

    Cache key format: ``{image_hash}__bbox_{entity_idx}.pt``
    """

    def __init__(
        self,
        feature_backend: GroundingFeatureBackend,
        cache_dir: str,
        instance_data_dir: str = "",
    ):
        self.feature_backend = feature_backend
        self.cache_dir = os.path.abspath(cache_dir)
        self.instance_data_dir = instance_data_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _hash_path(image_path: str) -> str:
        return hashlib.sha256(image_path.encode()).hexdigest()[:16]

    def _cache_path(self, image_path: str, entity_idx: int) -> str:
        if self.instance_data_dir and not os.path.isabs(image_path):
            image_path = os.path.join(self.instance_data_dir, image_path)
        h = self._hash_path(image_path)
        return os.path.join(self.cache_dir, f"{h}__bbox_{entity_idx}.pt")

    def retrieve(self, image_path: str, entity_idx: int) -> Optional[torch.Tensor]:
        """Load cached embedding for one entity, or None if not cached."""
        path = self._cache_path(image_path, entity_idx)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=True)
        return None

    @property
    def embed_dim(self) -> int:
        return self.feature_backend.embed_dim

    def process_image(self, image_path: str, bbox_entities: list[dict], *, image=None) -> list[torch.Tensor]:
        """Crop, encode, and cache embeddings for all entities in one image.

        Returns the list of embedding tensors (one per entity).
        """
        results: list[torch.Tensor] = []
        uncached_indices: list[int] = []
        uncached_crops: list[Image.Image] = []

        img = None
        for idx, entity in enumerate(bbox_entities):
            cached = self.retrieve(image_path, idx)
            if cached is not None:
                results.append(cached)
                continue

            if img is None:
                if image is None:
                    with Image.open(os.path.join(self.instance_data_dir, image_path)) as source:
                        img = source.convert("RGB")
                else:
                    first_frame = (
                        Image.fromarray(image[0] if image.ndim == 4 else image) if isinstance(image, np.ndarray) else image
                    )
                    img = first_frame.convert("RGB")

            bbox = entity.get("bbox", [0, 0, 1, 1])
            x1, y1, x2, y2 = bbox
            w, h = img.size
            crop_box = (
                max(0, int(x1 * w)),
                max(0, int(y1 * h)),
                min(w, int(x2 * w)),
                min(h, int(y2 * h)),
            )
            # Ensure minimum 1-pixel crop
            if crop_box[2] <= crop_box[0]:
                crop_box = (crop_box[0], crop_box[1], crop_box[0] + 1, crop_box[3])
            if crop_box[3] <= crop_box[1]:
                crop_box = (crop_box[0], crop_box[1], crop_box[2], crop_box[1] + 1)
            crop = img.crop(crop_box)
            uncached_indices.append(idx)
            uncached_crops.append(crop)
            results.append(torch.zeros(0))  # placeholder

        if uncached_crops:
            embeddings = self.feature_backend.embed_image(uncached_crops)
            for i, idx in enumerate(uncached_indices):
                emb = embeddings[i]
                cache_path = self._cache_path(image_path, idx)
                with atomic_write(cache_path, mode="wb", overwrite=True) as stream:
                    torch.save(emb, stream)
                results[idx] = emb

        return results

    def process_all(self, dataset, *, batch_size=32) -> int:
        """Prepare features for accepted samples with entity annotations."""
        count = 0
        for _, refs in dataset.iter_preparation_batches(batch_size=batch_size):
            pending = {}
            for record in dataset.read_samples(refs):
                entities = record.get("bbox_entities")
                if not entities:
                    continue
                ref = record["image_path"]
                if any(not os.path.isfile(self._cache_path(ref, index)) for index in range(len(entities))):
                    pending[ref] = entities
            if not pending:
                continue
            for record in dataset.read_samples(tuple(pending), fields=("image",)):
                ref = record["image_path"]
                self.process_image(ref, pending[ref], image=record["image"])
                count += 1

        if count:
            logger.info("Grounding image embed cache: processed %d images", count)
        return count
