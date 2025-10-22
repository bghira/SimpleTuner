"""Image backend configuration class."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from simpletuner.helpers.training.state_tracker import StateTracker

from . import validators
from .base import BaseBackendConfig

logger = logging.getLogger("ImageBackendConfig")


@dataclass
class ImageBackendConfig(BaseBackendConfig):

    instance_data_dir: str = ""
    aws_data_prefix: str = ""

    crop: bool = False
    crop_aspect: str = "square"
    crop_aspect_buckets: Optional[List[Union[float, Dict[str, Any]]]] = None
    crop_style: str = "random"

    csv_cache_dir: Optional[str] = None
    csv_file: Optional[str] = None
    csv_caption_column: Optional[str] = None
    csv_url_column: Optional[str] = None

    aws_bucket_name: Optional[str] = None
    aws_region_name: Optional[str] = None
    aws_endpoint_url: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_max_pool_connections: Optional[int] = None

    dataset_name: Optional[str] = None
    split: Optional[str] = "train"
    revision: Optional[str] = None
    image_column: Optional[str] = "image"
    video_column: Optional[str] = "video"
    huggingface_cache_dir: Optional[str] = None
    huggingface_streaming: Optional[bool] = None
    huggingface_num_proc: Optional[int] = None
    huggingface_composite_config: Optional[Dict[str, Any]] = None
    has_huggingface_block: bool = False
    huggingface_auto_load: Optional[bool] = None

    vae_cache_clear_each_epoch: Optional[bool] = None
    probability: float = 1.0
    repeats: int = 0
    disable_validation: bool = False
    hash_filenames: Optional[bool] = None
    shorten_filenames: Optional[bool] = None
    source_dataset_id: Optional[str] = None

    minimum_image_size: Optional[Union[int, float]] = None
    minimum_aspect_ratio: Optional[Union[int, float]] = None
    maximum_aspect_ratio: Optional[Union[int, float]] = None

    conditioning: Optional[Dict[str, Any]] = None
    conditioning_config: Optional[Dict[str, Any]] = None
    conditioning_data: Optional[str] = None
    conditioning_type: Optional[str] = None

    parquet: Optional[Dict[str, Any]] = None

    video: Optional[Dict[str, Any]] = None

    disable_vae_cache: bool = False

    is_regularisation_data: bool = False
    is_regularization_data: bool = False

    @classmethod
    def from_dict(cls, backend_dict: Dict[str, Any], args: Dict[str, Any]) -> "ImageBackendConfig":
        def _get_arg(key: str, default: Any = None) -> Any:
            if isinstance(args, dict):
                return args.get(key, default)
            return getattr(args, key, default)

        config = cls(
            id=backend_dict["id"],
            backend_type=backend_dict.get("type", "local"),
            dataset_type=backend_dict.get("dataset_type", "image"),
        )

        config.disabled = backend_dict.get("disabled", backend_dict.get("disable", False))

        config.instance_data_dir = backend_dict.get("instance_data_dir", backend_dict.get("aws_data_prefix", ""))
        config.aws_data_prefix = backend_dict.get("aws_data_prefix", "")

        config.crop = backend_dict.get("crop", False)
        config.crop_aspect = backend_dict.get("crop_aspect", "square")
        config.crop_aspect_buckets = backend_dict.get("crop_aspect_buckets")
        config.crop_style = backend_dict.get("crop_style", "random")

        config.metadata_backend = backend_dict.get("metadata_backend")
        config.caption_strategy = backend_dict.get("caption_strategy")

        config.minimum_image_size = backend_dict.get("minimum_image_size", _get_arg("minimum_image_size"))
        config.maximum_image_size = backend_dict.get("maximum_image_size", _get_arg("maximum_image_size"))
        config.target_downsample_size = backend_dict.get("target_downsample_size", _get_arg("target_downsample_size"))
        config.minimum_aspect_ratio = backend_dict.get("minimum_aspect_ratio")
        config.maximum_aspect_ratio = backend_dict.get("maximum_aspect_ratio")

        config.resolution = backend_dict.get("resolution", _get_arg("resolution"))
        config.resolution_type = backend_dict.get("resolution_type", _get_arg("resolution_type"))

        # CSV specific settings
        if config.backend_type == "csv":
            config.csv_cache_dir = backend_dict.get("csv_cache_dir")
            config.csv_file = backend_dict.get("csv_file")
            config.csv_caption_column = backend_dict.get("csv_caption_column")
            config.csv_url_column = backend_dict.get("csv_url_column")
            config.hash_filenames = backend_dict.get(
                "csv_hash_filenames",
                backend_dict.get("hash_filenames", config.hash_filenames),
            )
            config.shorten_filenames = backend_dict.get(
                "csv_shorten_filenames",
                backend_dict.get("shorten_filenames", config.shorten_filenames),
            )

        # AWS specific settings
        if config.backend_type == "aws":
            config.aws_bucket_name = backend_dict.get("aws_bucket_name")
            config.aws_region_name = backend_dict.get("aws_region_name")
            config.aws_endpoint_url = backend_dict.get("aws_endpoint_url")
            config.aws_access_key_id = backend_dict.get("aws_access_key_id")
            config.aws_secret_access_key = backend_dict.get("aws_secret_access_key")
            config.aws_max_pool_connections = backend_dict.get(
                "aws_max_pool_connections", _get_arg("aws_max_pool_connections")
            )

        # HuggingFace specific settings
        if config.backend_type == "huggingface":
            config.has_huggingface_block = "huggingface" in backend_dict
            hf_block = backend_dict.get("huggingface", {})
            config.dataset_name = backend_dict.get("dataset_name", hf_block.get("dataset_name"))
            config.split = backend_dict.get("split", hf_block.get("split", "train"))
            config.revision = backend_dict.get("revision", hf_block.get("revision"))
            config.image_column = backend_dict.get("image_column", hf_block.get("image_column", "image"))
            config.video_column = backend_dict.get("video_column", hf_block.get("video_column", "video"))
            config.huggingface_cache_dir = hf_block.get("cache_dir", backend_dict.get("cache_dir"))
            config.huggingface_streaming = hf_block.get("streaming", backend_dict.get("streaming"))
            config.huggingface_num_proc = hf_block.get("num_proc", backend_dict.get("num_proc"))
            config.huggingface_composite_config = hf_block.get("composite_image_config")
            config.huggingface_auto_load = hf_block.get("auto_load", backend_dict.get("auto_load"))

        config.vae_cache_clear_each_epoch = backend_dict.get("vae_cache_clear_each_epoch")
        config.probability = float(backend_dict.get("probability", 1.0)) if backend_dict.get("probability") else 1.0
        config.repeats = int(backend_dict.get("repeats", 0)) if backend_dict.get("repeats") else 0
        config.disable_validation = backend_dict.get("disable_validation", False)
        if "hash_filenames" in backend_dict and config.backend_type != "csv":
            config.hash_filenames = backend_dict.get("hash_filenames")
        config.source_dataset_id = backend_dict.get("source_dataset_id")

        config.conditioning = backend_dict.get("conditioning")
        config.conditioning_config = backend_dict.get("conditioning_config")
        config.conditioning_data = backend_dict.get("conditioning_data")
        config.conditioning_type = backend_dict.get("conditioning_type")

        config.parquet = backend_dict.get("parquet")
        config.video = backend_dict.get("video")

        config.disable_vae_cache = bool(backend_dict.get("disable_vae_cache", False))

        config.is_regularisation_data = backend_dict.get(
            "is_regularisation_data", backend_dict.get("is_regularization_data", False)
        )
        config.is_regularization_data = backend_dict.get(
            "is_regularization_data", backend_dict.get("is_regularisation_data", False)
        )

        compress_arg = backend_dict.get("compress_cache", None)
        if compress_arg is None:
            compress_arg = _get_arg("compress_disk_cache")
        if compress_arg is not None:
            config.compress_cache = bool(compress_arg)
            config.config["compress_cache"] = config.compress_cache

        config._convert_pixel_area_resolution(_get_arg("resolution", 1.0))

        config.apply_defaults(args)

        return config

    # compatibility helpers
    @property
    def type(self) -> str:
        return self.backend_type

    @type.setter
    def type(self, value: str) -> None:
        self.backend_type = value

    def _convert_pixel_area_resolution(self, base_resolution: Optional[Union[int, float]]) -> None:
        # convert pixel_area resolution to area units

        if self.resolution_type != "pixel_area":
            return

        pixel_edge = self.resolution if self.resolution is not None else base_resolution
        if pixel_edge is None:
            raise ValueError("Resolution type 'pixel_area' requires a numeric 'resolution' value to be provided.")

        try:
            pixel_edge_value = float(pixel_edge)
        except (TypeError, ValueError):
            raise ValueError("Resolution type 'pixel_area' requires the resolution value to be numeric.")

        baseline = float(base_resolution or 1.0)
        if baseline == 0:
            baseline = 1.0

        self.resolution = (pixel_edge_value * pixel_edge_value) / baseline

        def _convert_pixel_edge_to_megapixels(value: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
            if value in (None, ""):
                return value
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return value
            if numeric_value <= 0 or numeric_value <= 10:
                return value
            return (numeric_value * numeric_value) / 1_000_000.0

        for field_name in ("maximum_image_size", "minimum_image_size", "target_downsample_size"):
            current_value = getattr(self, field_name, None)
            converted_value = _convert_pixel_edge_to_megapixels(current_value)
            setattr(self, field_name, converted_value)

        self.resolution_type = "area"

    def apply_defaults(self, args: Dict[str, Any]) -> None:
        self._apply_common_defaults(args)

        if self.backend_type == "huggingface":
            if self.metadata_backend is None:
                self.metadata_backend = "huggingface"
            if self.caption_strategy is None:
                self.caption_strategy = "huggingface"

        if self.disable_vae_cache:
            self.config["disable_vae_cache"] = True

    def validate(self, args: Dict[str, Any]) -> None:
        validators.validate_backend_id(self.id)

        valid_types = ["image", "conditioning", "eval", "video"]
        validators.validate_dataset_type(self.dataset_type, valid_types, self.id)

        self._validate_controlnet_requirements(args)

        validators.validate_crop_aspect(self.crop_aspect, self.crop_aspect_buckets, self.id)
        validators.validate_crop_aspect_buckets(self.crop_aspect, self.crop_aspect_buckets, self.id)
        validators.validate_crop_style(self.crop_style, self.id)

        validators.validate_image_size_constraints(
            self.maximum_image_size, self.target_downsample_size, self.resolution_type, args.get("model_type", ""), self.id
        )

        if self.resolution_type:
            validators.validate_resolution_type(self.resolution_type, self.id)

        validators.validate_caption_strategy_compatibility(
            self.caption_strategy or "", self.metadata_backend or "", self.backend_type, self.id
        )

        self._validate_backend_specific_settings()

        if self.dataset_type == "video":
            self._validate_video_settings(args)

        validators.check_for_caption_filter_list_misuse(self.dataset_type, False, self.id)

    def _validate_controlnet_requirements(self, args: Dict[str, Any]) -> None:
        state_args = validators.StateTracker.get_args()
        if (
            state_args is not None
            and hasattr(state_args, "controlnet")
            and state_args.controlnet
            and self.dataset_type == "image"
            and (self.conditioning_data is None and self.conditioning is None)
        ):
            raise ValueError(
                f"When training ControlNet, a conditioning block or conditioning_data string should be configured in your dataloader. See this link for more information: https://github.com/bghira/SimpleTuner/blob/main/documentation/CONTROLNET.md"
            )

    def _validate_backend_specific_settings(self) -> None:
        validators.validate_huggingface_backend_settings(
            self.backend_type, self.metadata_backend, self.caption_strategy, self.id
        )

    def _validate_video_settings(self, args: Dict[str, Any]) -> None:
        if not self.video:
            self.video = {}

        if "num_frames" not in self.video:
            default_num_seconds = 5
            framerate = args.get("framerate", 30)
            video_duration_in_frames = framerate * default_num_seconds
            logger.warning(
                f"No `num_frames` was provided for video backend. Defaulting to {video_duration_in_frames} ({default_num_seconds} seconds @ {framerate}fps) to avoid memory implosion/explosion. Reduce value further for lower memory use."
            )
            self.video["num_frames"] = video_duration_in_frames

        if "min_frames" not in self.video:
            logger.warning(
                f"No `min_frames` was provided for video backend. Defaulting to {self.video['num_frames']} frames (num_frames). Reduce num_frames further for lower memory use."
            )
            self.video["min_frames"] = self.video["num_frames"]

        if "max_frames" not in self.video:
            logger.warning(
                f"No `max_frames` was provided for video backend. Set this value to avoid scanning huge video files."
            )

        if "is_i2v" not in self.video:
            model_family = args.get("model_family", "")
            if model_family in ["ltxvideo"]:
                logger.warning(
                    f"Setting is_i2v to True for model_family={model_family}. Set this manually to false to override."
                )
                self.video["is_i2v"] = True
            else:
                logger.warning(f"No value for is_i2v was supplied for your dataset. Assuming it is disabled.")
                self.video["is_i2v"] = False

        min_frames = self.video["min_frames"]
        num_frames = self.video["num_frames"]
        validators.validate_video_frame_settings(min_frames, num_frames, self.id)

    def _check_deprecated_settings(self) -> None:
        pass

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()

        config = result["config"]

        result["type"] = self.backend_type
        config["dataset_type"] = self.dataset_type

        config["crop"] = self.crop
        config["crop_aspect"] = self.crop_aspect
        config["crop_style"] = self.crop_style
        config["disable_validation"] = self.disable_validation
        config["probability"] = self.probability
        config["repeats"] = self.repeats
        config["instance_data_dir"] = self.instance_data_dir

        if self.crop_aspect_buckets is not None:
            config["crop_aspect_buckets"] = self.crop_aspect_buckets
        if self.vae_cache_clear_each_epoch is not None:
            config["vae_cache_clear_each_epoch"] = self.vae_cache_clear_each_epoch
        if self.hash_filenames is not None:
            config["hash_filenames"] = self.hash_filenames
        if self.shorten_filenames is not None:
            config["shorten_filenames"] = self.shorten_filenames
        if self.source_dataset_id is not None:
            config["source_dataset_id"] = self.source_dataset_id

        if self.minimum_image_size is not None:
            config["minimum_image_size"] = self.minimum_image_size
        if self.minimum_aspect_ratio is not None:
            config["minimum_aspect_ratio"] = self.minimum_aspect_ratio
        if self.maximum_aspect_ratio is not None:
            config["maximum_aspect_ratio"] = self.maximum_aspect_ratio

        if self.conditioning is not None:
            config["conditioning"] = self.conditioning
        if self.conditioning_config is not None:
            config["conditioning_config"] = self.conditioning_config

        if self.csv_cache_dir is not None:
            config["csv_cache_dir"] = self.csv_cache_dir
        if self.csv_file is not None:
            config["csv_file"] = self.csv_file
        if self.csv_caption_column is not None:
            config["csv_caption_column"] = self.csv_caption_column
        if self.csv_url_column is not None:
            config["csv_url_column"] = self.csv_url_column

        if self.backend_type == "aws":
            config["aws_bucket_name"] = self.aws_bucket_name
            config["aws_region_name"] = self.aws_region_name
            config["aws_endpoint_url"] = self.aws_endpoint_url
            config["aws_access_key_id"] = self.aws_access_key_id
            config["aws_secret_access_key"] = self.aws_secret_access_key
            if self.aws_max_pool_connections is not None:
                config["aws_max_pool_connections"] = self.aws_max_pool_connections

        if self.backend_type == "huggingface":
            config["dataset_name"] = self.dataset_name
            config["split"] = self.split
            config["revision"] = self.revision
            config["image_column"] = self.image_column
            config["video_column"] = self.video_column
            hf_config = config.setdefault("huggingface", {})
            if self.huggingface_cache_dir is not None:
                hf_config["cache_dir"] = self.huggingface_cache_dir
            if self.huggingface_streaming is not None:
                hf_config["streaming"] = self.huggingface_streaming
            if self.huggingface_num_proc is not None:
                hf_config["num_proc"] = self.huggingface_num_proc
            if self.huggingface_composite_config is not None:
                hf_config["composite_image_config"] = self.huggingface_composite_config
            if self.huggingface_auto_load is not None:
                hf_config["auto_load"] = self.huggingface_auto_load

        if self.video is not None:
            config["video"] = self.video

        return result
