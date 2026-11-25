"""Data backend builder classes."""

from typing import Any, Optional

from .aws import AwsBackendBuilder
from .base import BaseBackendBuilder
from .csv import CsvBackendBuilder
from .huggingface import HuggingfaceBackendBuilder
from .local import LocalBackendBuilder
from .memory import MemoryBackendBuilder
from .webshart import WebshartBackendBuilder

__all__ = [
    "BaseBackendBuilder",
    "LocalBackendBuilder",
    "MemoryBackendBuilder",
    "AwsBackendBuilder",
    "CsvBackendBuilder",
    "HuggingfaceBackendBuilder",
    "WebshartBackendBuilder",
    "create_backend_builder",
    "supports_shard_listing",
    "build_backend_from_config",
]


def _backend_builders() -> dict[str, type[BaseBackendBuilder]]:
    return {
        "local": LocalBackendBuilder,
        "memory": MemoryBackendBuilder,
        "aws": AwsBackendBuilder,
        "csv": CsvBackendBuilder,
        "huggingface": HuggingfaceBackendBuilder,
        "webshart": WebshartBackendBuilder,
    }


def supports_shard_listing(backend_type: str) -> bool:
    builder = _backend_builders().get(backend_type)
    return builder is not None and builder.supports_shard_listing


def create_backend_builder(backend_type: str, accelerator, args: Optional[Any] = None) -> BaseBackendBuilder:
    builders = _backend_builders()
    builder_class = builders.get(backend_type)
    if builder_class is None:
        raise ValueError(f"Unknown backend type: {backend_type}. Supported types: {list(builders)}")

    builder = builder_class(accelerator)
    if args is not None:
        builder.args = args
    return builder


def build_backend_from_config(config, accelerator, args: dict, **kwargs):
    config.validate(args)

    builder = create_backend_builder(config.backend_type, accelerator, args)

    if hasattr(builder, "build_with_metadata"):
        return builder.build_with_metadata(config, args, **kwargs)
    else:
        data_backend = builder.build(config)
        metadata_backend = builder.create_metadata_backend(config, data_backend, args)
        return {
            "id": config.id,
            "data_backend": data_backend,
            "metadata_backend": metadata_backend,
            "instance_data_dir": "",
            "config": config.to_dict()["config"],
        }
