"""Centralised dataset type enum and helpers."""

from __future__ import annotations

from enum import Enum
from numbers import Integral
from typing import Any, Iterable, Mapping, Optional, Sequence


class DatasetType(str, Enum):
    """Supported dataset categories across builders, metadata, and training."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    CONDITIONING = "conditioning"
    EVAL = "eval"
    TEXT_EMBEDS = "text_embeds"
    IMAGE_EMBEDS = "image_embeds"
    CONDITIONING_IMAGE_EMBEDS = "conditioning_image_embeds"
    DISTILLATION_CACHE = "distillation_cache"
    CAPTION = "caption"
    GROUNDING = "grounding"

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


def get_arg_value(args: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve a value from an args mapping or namespace."""
    if isinstance(args, Mapping):
        return args.get(key, default)
    return getattr(args, key, default)


def parse_positive_train_batch_size(raw_value: Any, backend_id: Optional[str] = None) -> int:
    """Parse a positive integer dataset training batch size."""
    error_message = f"(id={backend_id}) train_batch_size must be a positive integer."

    if isinstance(raw_value, bool):
        raise ValueError(error_message)
    if isinstance(raw_value, Integral):
        batch_size = int(raw_value)
    elif isinstance(raw_value, str) and raw_value.isascii() and raw_value.isdecimal():
        batch_size = int(raw_value)
        if raw_value != str(batch_size):
            raise ValueError(error_message)
    else:
        raise ValueError(error_message)

    if batch_size < 1:
        raise ValueError(error_message)
    return batch_size


def resolve_dataset_train_batch_size(
    backend: Mapping[str, Any],
    args: Any,
    dataset_type: Optional[DatasetType] = None,
    backend_id: Optional[str] = None,
) -> int:
    """Return the effective per-rank training batch size for a dataset."""
    resolved_dataset_type = dataset_type or ensure_dataset_type(backend.get("dataset_type"), default=DatasetType.IMAGE)
    if resolved_dataset_type is DatasetType.EVAL:
        return 1

    raw_value = backend.get("train_batch_size")
    if raw_value in (None, ""):
        raw_value = get_arg_value(args, "train_batch_size", 1)

    resolved_backend_id = backend_id if backend_id is not None else backend.get("id")
    return parse_positive_train_batch_size(raw_value, resolved_backend_id)
