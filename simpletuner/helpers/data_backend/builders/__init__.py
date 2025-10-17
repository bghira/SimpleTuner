"""Data backend builder classes."""

from typing import Any, Optional

from .aws import AwsBackendBuilder
from .base import BaseBackendBuilder
from .csv import CsvBackendBuilder
from .huggingface import HuggingfaceBackendBuilder
from .local import LocalBackendBuilder

__all__ = [
    "BaseBackendBuilder",
    "LocalBackendBuilder",
    "AwsBackendBuilder",
    "CsvBackendBuilder",
    "HuggingfaceBackendBuilder",
    "create_backend_builder",
    "build_backend_from_config",
]


def create_backend_builder(backend_type: str, accelerator, args: Optional[Any] = None) -> BaseBackendBuilder:
    builder_mapping = {
        "local": LocalBackendBuilder,
        "aws": AwsBackendBuilder,
        "csv": CsvBackendBuilder,
        "huggingface": HuggingfaceBackendBuilder,
    }

    if backend_type not in builder_mapping:
        raise ValueError(f"Unknown backend type: {backend_type}. Supported types: {list(builder_mapping.keys())}")

    builder_class = builder_mapping[backend_type]
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
