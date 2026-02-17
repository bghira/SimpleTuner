"""Optional image feature extraction for grounding entity crops."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class GroundingFeatureBackend(ABC):
    """Abstract base for per-entity image feature extraction."""

    @abstractmethod
    def embed_image(self, crops: list[Image.Image]) -> torch.Tensor:
        """Encode entity image crops to (N, D) feature vectors."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimensionality of the output feature vector."""


class DINOFeatureBackend(GroundingFeatureBackend):
    """Uses DINOv2 for visual entity features."""

    def __init__(self, model_name_or_path: str = "facebook/dinov2-large", device: Optional[str] = None):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device).eval()
        self._embed_dim = self.model.config.hidden_size

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @torch.no_grad()
    def embed_image(self, crops: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # Use CLS token as the entity feature
        return outputs.last_hidden_state[:, 0].cpu()


class CLIPFeatureBackend(GroundingFeatureBackend):
    """Uses CLIP ViT for visual entity features."""

    def __init__(self, model_name_or_path: str = "openai/clip-vit-large-patch14", device: Optional[str] = None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.model = CLIPModel.from_pretrained(model_name_or_path).to(self.device).eval()
        self._embed_dim = self.model.config.vision_config.hidden_size

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @torch.no_grad()
    def embed_image(self, crops: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        return outputs.cpu()
