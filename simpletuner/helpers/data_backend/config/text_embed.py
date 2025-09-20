"""Text embed backend configuration class."""
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from .base import BaseBackendConfig
from . import validators


@dataclass
class TextEmbedBackendConfig(BaseBackendConfig):

    caption_filter_list: Optional[List[str]] = None

    def __post_init__(self):
        self.dataset_type = "text_embeds"
        super().__post_init__()

    @classmethod
    def from_dict(cls, backend_dict: Dict[str, Any], args: Dict[str, Any]) -> "TextEmbedBackendConfig":
        config = cls(
            id=backend_dict["id"],
            backend_type=backend_dict.get("type", "local"),
            dataset_type="text_embeds",
            disabled=backend_dict.get("disabled", backend_dict.get("disable", False)),
            caption_filter_list=backend_dict.get("caption_filter_list", [])
        )

        compress_arg = backend_dict.get("compress_cache", None)
        if compress_arg is None:
            if isinstance(args, dict):
                compress_arg = args.get("compress_disk_cache")
            else:
                compress_arg = getattr(args, "compress_disk_cache", None)
        if compress_arg is not None:
            config.compress_cache = bool(compress_arg)
            config.config["compress_cache"] = config.compress_cache

        passthrough_keys = {
            "cache_dir",
            "aws_bucket_name",
            "aws_data_prefix",
            "aws_endpoint_url",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_region_name",
        }
        for key in passthrough_keys:
            if key in backend_dict:
                value = backend_dict[key]
                config.config[key] = value
                setattr(config, key, value)

        if backend_dict.get("type") == "aws" and "aws" in backend_dict:
            aws_block = backend_dict.get("aws", {})
            config.config["aws"] = aws_block
            for aws_key, aws_value in aws_block.items():
                setattr(config, aws_key, aws_value)

        config.apply_defaults(args)

        return config

    def apply_defaults(self, args: Dict[str, Any]) -> None:
        pass

    def validate(self, args: Dict[str, Any]) -> None:
        validators.validate_backend_id(self.id)

        validators.validate_dataset_type(self.dataset_type, ["text_embeds"], self.id)

        validators.check_for_caption_filter_list_misuse(
            self.dataset_type,
            self.caption_filter_list is not None and len(self.caption_filter_list) > 0,
            self.id
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "dataset_type": "text_embeds",
            "config": {}
        }

        if self.caption_filter_list is not None:
            result["config"]["caption_filter_list"] = self.caption_filter_list

        for key, value in self.config.items():
            result["config"][key] = value

        return result
