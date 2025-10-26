"""Backend configuration classes for different dataset types."""

from simpletuner.helpers.data_backend.dataset_types import DatasetType, ensure_dataset_type

from . import validators
from .base import BaseBackendConfig
from .distillation_cache import DistillationCacheBackendConfig
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
            self.dataset_type = DatasetType.CONDITIONING_IMAGE_EMBEDS

        @classmethod
        def from_dict(cls, backend_dict: dict, args: dict) -> "ConditioningImageEmbedBackendConfig":
            config = super().from_dict(backend_dict, args)
            config.dataset_type = DatasetType.CONDITIONING_IMAGE_EMBEDS
            return config

        def to_dict(self) -> dict:
            payload = super().to_dict()
            payload["dataset_type"] = DatasetType.CONDITIONING_IMAGE_EMBEDS.value
            return payload


__all__ = [
    "BaseBackendConfig",
    "ImageBackendConfig",
    "TextEmbedBackendConfig",
    "ImageEmbedBackendConfig",
    "DistillationCacheBackendConfig",
    "ConditioningImageEmbedBackendConfig",
    "validators",
    "create_backend_config",
]


def create_backend_config(backend_dict: dict, args: dict) -> BaseBackendConfig:
    dataset_type = ensure_dataset_type(backend_dict.get("dataset_type"), default=DatasetType.IMAGE)

    if dataset_type is DatasetType.TEXT_EMBEDS:
        return TextEmbedBackendConfig.from_dict(backend_dict, args)
    elif dataset_type is DatasetType.IMAGE_EMBEDS:
        return ImageEmbedBackendConfig.from_dict(backend_dict, args)
    elif dataset_type is DatasetType.CONDITIONING_IMAGE_EMBEDS:
        return ConditioningImageEmbedBackendConfig.from_dict(backend_dict, args)
    elif dataset_type is DatasetType.DISTILLATION_CACHE:
        return DistillationCacheBackendConfig.from_dict(backend_dict, args)
    elif dataset_type in {
        DatasetType.IMAGE,
        DatasetType.CONDITIONING,
        DatasetType.EVAL,
        DatasetType.VIDEO,
        DatasetType.CAPTION,
    }:
        return ImageBackendConfig.from_dict(backend_dict, args)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
