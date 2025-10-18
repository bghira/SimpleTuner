"""Backend configuration classes for different dataset types."""

from . import validators
from .base import BaseBackendConfig
from .image import ImageBackendConfig
from .image_embed import ImageEmbedBackendConfig
from .text_embed import TextEmbedBackendConfig

try:  # pragma: no cover - graceful fallback when optional module missing
    from .conditioning_image_embed import ConditioningImageEmbedBackendConfig
except ModuleNotFoundError:  # pragma: no cover - legacy environments

    class ConditioningImageEmbedBackendConfig(ImageEmbedBackendConfig):  # type: ignore[misc]
        """Fallback configuration that mirrors ImageEmbed when the specialised class is unavailable."""

        def __post_init__(self):
            super().__post_init__()
            self.dataset_type = "conditioning_image_embeds"

        @classmethod
        def from_dict(cls, backend_dict: dict, args: dict) -> "ConditioningImageEmbedBackendConfig":
            config = super().from_dict(backend_dict, args)
            config.dataset_type = "conditioning_image_embeds"
            return config

        def to_dict(self) -> dict:
            payload = super().to_dict()
            payload["dataset_type"] = "conditioning_image_embeds"
            return payload

__all__ = [
    "BaseBackendConfig",
    "ImageBackendConfig",
    "TextEmbedBackendConfig",
    "ImageEmbedBackendConfig",
    "ConditioningImageEmbedBackendConfig",
    "validators",
    "create_backend_config",
]


def create_backend_config(backend_dict: dict, args: dict) -> BaseBackendConfig:
    dataset_type = backend_dict.get("dataset_type", "image")

    if dataset_type == "text_embeds":
        return TextEmbedBackendConfig.from_dict(backend_dict, args)
    elif dataset_type == "image_embeds":
        return ImageEmbedBackendConfig.from_dict(backend_dict, args)
    elif dataset_type == "conditioning_image_embeds":
        return ConditioningImageEmbedBackendConfig.from_dict(backend_dict, args)
    elif dataset_type in ["image", "conditioning", "eval", "video"]:
        return ImageBackendConfig.from_dict(backend_dict, args)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
