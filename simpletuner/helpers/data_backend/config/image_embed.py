"""Image embed backend configuration class."""

from dataclasses import dataclass
from typing import Any, Dict

from . import validators
from .base import BaseBackendConfig


@dataclass
class ImageEmbedBackendConfig(BaseBackendConfig):

    def __post_init__(self):
        self.dataset_type = "image_embeds"
        super().__post_init__()

    @classmethod
    def from_dict(cls, backend_dict: Dict[str, Any], args: Dict[str, Any]) -> "ImageEmbedBackendConfig":
        config = cls(
            id=backend_dict["id"],
            backend_type=backend_dict.get("type", "local"),
            dataset_type="image_embeds",
            disabled=backend_dict.get("disabled", backend_dict.get("disable", False)),
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

        config.apply_defaults(args)

        return config

    def apply_defaults(self, args: Dict[str, Any]) -> None:
        pass

    def validate(self, args: Dict[str, Any]) -> None:
        validators.validate_backend_id(self.id)

        validators.validate_dataset_type(self.dataset_type, ["image_embeds"], self.id)

        validators.check_for_caption_filter_list_misuse(self.dataset_type, False, self.id)

    def to_dict(self) -> Dict[str, Any]:
        result = {"id": self.id, "dataset_type": "image_embeds", "config": {}}

        return result
