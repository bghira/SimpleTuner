"""Base configuration class for data backends."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, List
import os

from simpletuner.helpers.training.state_tracker import StateTracker


@dataclass
class BaseBackendConfig(ABC):

    id: str = ""
    backend_type: str = ""

    disabled: bool = False
    dataset_type: str = "image"

    resolution: Optional[Union[int, float]] = None
    resolution_type: Optional[str] = None

    caption_strategy: Optional[str] = None

    metadata_backend: Optional[str] = None

    compress_cache: Optional[bool] = None

    maximum_image_size: Optional[Union[int, float]] = None
    target_downsample_size: Optional[Union[int, float]] = None

    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    @abstractmethod
    def from_dict(cls, backend_dict: Dict[str, Any], args: Dict[str, Any]) -> "BaseBackendConfig":
        pass

    @abstractmethod
    def validate(self, args: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def apply_defaults(self, args: Dict[str, Any]) -> None:
        pass

    def _validate_common_fields(self, args: Dict[str, Any]) -> None:
        valid_dataset_types = ["image", "conditioning", "eval", "video", "text_embeds", "image_embeds"]
        if self.dataset_type not in valid_dataset_types:
            raise ValueError(f"(id={self.id}) dataset_type must be one of {valid_dataset_types}.")

        self._validate_image_size_settings(args)

        self._validate_caption_strategy(args)

    def _validate_image_size_settings(self, args: Dict[str, Any]) -> None:
        if self.maximum_image_size and not self.target_downsample_size:
            raise ValueError(
                "When a data backend is configured to use `maximum_image_size`, you must also provide a value for `target_downsample_size`."
            )

        if (
            self.maximum_image_size
            and self.resolution_type == "area"
            and self.maximum_image_size > 10
            and not os.environ.get("SIMPLETUNER_MAXIMUM_IMAGE_SIZE_OVERRIDE", False)
        ):
            raise ValueError(
                f"When a data backend is configured to use `'resolution_type':area`, `maximum_image_size` must be less than 10 megapixels. You may have accidentally entered {self.maximum_image_size} pixels, instead of megapixels."
            )
        elif (
            self.maximum_image_size
            and self.resolution_type == "pixel"
            and self.maximum_image_size < 512
            and "deepfloyd" not in args.get("model_type", "")
        ):
            raise ValueError(
                f"When a data backend is configured to use `'resolution_type':pixel`, `maximum_image_size` must be at least 512 pixels. You may have accidentally entered {self.maximum_image_size} megapixels, instead of pixels."
            )

        if (
            self.target_downsample_size
            and self.resolution_type == "area"
            and self.target_downsample_size > 10
            and not os.environ.get("SIMPLETUNER_MAXIMUM_IMAGE_SIZE_OVERRIDE", False)
        ):
            raise ValueError(
                f"When a data backend is configured to use `'resolution_type':area`, `target_downsample_size` must be less than 10 megapixels. You may have accidentally entered {self.target_downsample_size} pixels, instead of megapixels."
            )
        elif (
            self.target_downsample_size
            and self.resolution_type == "pixel"
            and self.target_downsample_size < 512
            and "deepfloyd" not in args.get("model_type", "")
        ):
            raise ValueError(
                f"When a data backend is configured to use `'resolution_type':pixel`, `target_downsample_size` must be at least 512 pixels. You may have accidentally entered {self.target_downsample_size} megapixels, instead of pixels."
            )

    def _validate_caption_strategy(self, args: Dict[str, Any]) -> None:
        if self.caption_strategy == "parquet" and (
            self.metadata_backend == "json" or self.metadata_backend == "discovery"
        ):
            raise ValueError(
                f"(id={self.id}) Cannot use caption_strategy=parquet with metadata_backend={self.metadata_backend}. Instead, it is recommended to use the textfile strategy and extract your captions into txt files."
            )

        if self.caption_strategy == "huggingface":
            if self.backend_type != "huggingface":
                raise ValueError(
                    f"(id={self.id}) caption_strategy='huggingface' can only be used with type='huggingface' backends"
                )

    def _apply_common_defaults(self, args: Dict[str, Any]) -> None:
        if self.resolution is None:
            self.resolution = args.get("resolution")

        if self.resolution_type is None:
            self.resolution_type = args.get("resolution_type")

        if self.caption_strategy is None:
            self.caption_strategy = args.get("caption_strategy")

        if self.maximum_image_size is None:
            self.maximum_image_size = args.get("maximum_image_size")

        if self.target_downsample_size is None:
            self.target_downsample_size = args.get("target_downsample_size")

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "dataset_type": self.dataset_type,
            "config": self.config.copy()
        }

        if self.resolution is not None:
            result["config"]["resolution"] = self.resolution
        if self.resolution_type is not None:
            result["config"]["resolution_type"] = self.resolution_type
        if self.caption_strategy is not None:
            result["config"]["caption_strategy"] = self.caption_strategy
        if self.metadata_backend is not None:
            result["config"]["metadata_backend"] = self.metadata_backend
        if self.maximum_image_size is not None:
            result["config"]["maximum_image_size"] = self.maximum_image_size
        if self.target_downsample_size is not None:
            result["config"]["target_downsample_size"] = self.target_downsample_size
        if self.dataset_type is not None:
            result["config"]["dataset_type"] = self.dataset_type
        if self.compress_cache is not None:
            result["config"]["compress_cache"] = self.compress_cache

        return result

    def __post_init__(self):
        if not self.id:
            raise ValueError("Backend configuration must have an 'id' field.")

        if not isinstance(self.config, dict):
            self.config = {}
