"""Backend configuration classes for different dataset types."""

from .base import BaseBackendConfig
from .image import ImageBackendConfig
from .text_embed import TextEmbedBackendConfig
from .image_embed import ImageEmbedBackendConfig
from . import validators

__all__ = [
    "BaseBackendConfig",
    "ImageBackendConfig",
    "TextEmbedBackendConfig",
    "ImageEmbedBackendConfig",
    "validators",
    "create_backend_config"
]


def create_backend_config(backend_dict: dict, args: dict) -> BaseBackendConfig:
    dataset_type = backend_dict.get("dataset_type", "image")

    if dataset_type == "text_embeds":
        return TextEmbedBackendConfig.from_dict(backend_dict, args)
    elif dataset_type == "image_embeds":
        return ImageEmbedBackendConfig.from_dict(backend_dict, args)
    elif dataset_type in ["image", "conditioning", "eval", "video"]:
        return ImageBackendConfig.from_dict(backend_dict, args)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")