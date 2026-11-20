"""Text embed backend configuration class."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from simpletuner.helpers.data_backend.dataset_types import DatasetType

from . import validators
from .base import BaseBackendConfig


@dataclass
class TextEmbedBackendConfig(BaseBackendConfig):

    caption_filter_list: Optional[List[str]] = None
    text_cache_ondemand: bool = False
    text_cache_disable: bool = False
    text_cache_encode_only: bool = False
    text_cache_use_listing: bool = False

    def __post_init__(self):
        self.dataset_type = DatasetType.TEXT_EMBEDS
        super().__post_init__()

    @classmethod
    def from_dict(cls, backend_dict: Dict[str, Any], args: Dict[str, Any]) -> "TextEmbedBackendConfig":
        config = cls(
            id=backend_dict["id"],
            backend_type=backend_dict.get("type", "local"),
            dataset_type=DatasetType.TEXT_EMBEDS,
            disabled=backend_dict.get("disabled", backend_dict.get("disable", False)),
            caption_filter_list=backend_dict.get("caption_filter_list", []),
            text_cache_ondemand=bool(backend_dict.get("text_cache_ondemand", False)),
            text_cache_encode_only=bool(backend_dict.get("text_cache_encode_only", False)),
            text_cache_use_listing=backend_dict.get("text_cache_use_listing", False),
            text_cache_disable=bool(backend_dict.get("text_cache_disable", False)),
        )
        if config.text_cache_encode_only:
            config.text_cache_disable = True
        if config.text_cache_disable:
            config.text_cache_ondemand = True

        compress_arg = backend_dict.get("compress_cache", None)
        if compress_arg is None:
            if isinstance(args, dict):
                compress_arg = args.get("compress_disk_cache")
            else:
                compress_arg = getattr(args, "compress_disk_cache", None)
        if compress_arg is not None:
            config.compress_cache = bool(compress_arg)
            config.config["compress_cache"] = config.compress_cache

        config._apply_artifact_storage_settings(backend_dict)

        config.apply_defaults(args)

        return config

    def apply_defaults(self, args: Dict[str, Any]) -> None:
        pass

    def validate(self, args: Dict[str, Any]) -> None:
        validators.validate_backend_id(self.id)

        validators.validate_dataset_type(self.dataset_type, [DatasetType.TEXT_EMBEDS], self.id)

        if not isinstance(self.text_cache_use_listing, bool):
            raise ValueError("text_cache_use_listing must be a boolean.")
        if self.text_cache_use_listing and self.backend_type != "aws":
            raise ValueError("text_cache_use_listing requires an AWS text embedding store.")

        validators.check_for_caption_filter_list_misuse(
            self.dataset_type, self.caption_filter_list is not None and len(self.caption_filter_list) > 0, self.id
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {"id": self.id, "dataset_type": DatasetType.TEXT_EMBEDS.value, "config": {}}

        if self.caption_filter_list is not None:
            result["config"]["caption_filter_list"] = self.caption_filter_list
        if self.text_cache_ondemand:
            result["config"]["text_cache_ondemand"] = True
        if self.text_cache_encode_only:
            result["config"]["text_cache_encode_only"] = True
        if self.text_cache_disable:
            result["config"]["text_cache_disable"] = True
        if self.text_cache_use_listing:
            result["config"]["text_cache_use_listing"] = True

        for key, value in self.config.items():
            result["config"][key] = value

        return result
