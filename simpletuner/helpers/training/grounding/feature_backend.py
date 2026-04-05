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

    def unload(self):
        """Release model resources."""


class Florence2FeatureBackend(GroundingFeatureBackend):
    """Florence-2 DaViT vision encoder features for entity crops."""

    DEFAULT_MODEL = "florence-community/Florence-2-large"

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL, device: Optional[str] = None):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if "cuda" in str(self.device) else torch.float32

        self._model = (
            Florence2ForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
            .eval()
            .to(self.device)
        )
        self._processor = AutoProcessor.from_pretrained(model_name_or_path)

        self._embed_dim = self._model.config.vision_config.embed_dim[-1]

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @torch.no_grad()
    def embed_image(self, crops: list[Image.Image]) -> torch.Tensor:
        """Encode entity crops through Florence-2 vision encoder, return pooled features."""
        if not crops:
            return torch.zeros(0, self._embed_dim)

        inputs = self._processor(images=crops, return_tensors="pt").to(self.device, self._model.dtype)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            return torch.zeros(len(crops), self._embed_dim)

        vision_tower = getattr(self._model, "vision_tower", None)
        if vision_tower is None:
            vision_tower = getattr(self._model, "model", self._model)
            vision_tower = getattr(vision_tower, "vision_tower", None)
        if vision_tower is None:
            raise RuntimeError("Could not locate Florence-2 vision tower on the loaded model.")

        features = vision_tower(pixel_values)
        if isinstance(features, (tuple, list)):
            features = features[0]
        # Native Florence-2 vision tower outputs (N, C, H, W) — pool over spatial dims.
        pooled = features.mean(dim=(2, 3))
        return pooled.float().cpu()

    def unload(self):
        """Release GPU memory."""
        del self._model
        del self._processor
        self._model = None  # type: ignore[assignment]
        self._processor = None  # type: ignore[assignment]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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

    def unload(self):
        del self.model
        del self.processor
        self.model = None  # type: ignore[assignment]
        self.processor = None  # type: ignore[assignment]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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

    def unload(self):
        del self.model
        del self.processor
        self.model = None  # type: ignore[assignment]
        self.processor = None  # type: ignore[assignment]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
