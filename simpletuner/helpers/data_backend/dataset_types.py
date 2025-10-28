"""Centralised dataset type enum and helpers."""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional, Sequence


class DatasetType(str, Enum):
    """Supported dataset categories across builders, metadata, and training."""

    IMAGE = "image"
    VIDEO = "video"
    CONDITIONING = "conditioning"
    EVAL = "eval"
    TEXT_EMBEDS = "text_embeds"
    IMAGE_EMBEDS = "image_embeds"
    CONDITIONING_IMAGE_EMBEDS = "conditioning_image_embeds"
    DISTILLATION_CACHE = "distillation_cache"
    CAPTION = "caption"

    @classmethod
    def from_value(cls, value: Optional[object], default: Optional["DatasetType"] = None) -> "DatasetType":
        """Convert strings/enum-like values into a DatasetType."""
        if isinstance(value, cls):
            return value
        if value is None:
            if default is not None:
                return default
            raise ValueError("Dataset type value may not be None without a default.")

        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in cls:
                if member.value == normalized:
                    return member
        raise ValueError(f"Unknown dataset_type: {value}")

    @classmethod
    def normalize_list(
        cls,
        values: Optional[Sequence[object]],
        default: Optional["DatasetType"] = None,
    ) -> list["DatasetType"]:
        if values is None:
            return []
        return [cls.from_value(value, default=default) for value in values]


def ensure_dataset_type(value: Optional[object], default: Optional[DatasetType] = None) -> DatasetType:
    """Helper alias for DatasetType.from_value used throughout the codebase."""
    return DatasetType.from_value(value, default=default)


def dataset_type_in(
    value: Optional[object],
    candidates: Iterable[DatasetType],
    *,
    default: Optional[DatasetType] = None,
) -> bool:
    """Return True if value matches any candidate dataset types."""
    target = ensure_dataset_type(value, default=default)
    return target in set(candidates)
