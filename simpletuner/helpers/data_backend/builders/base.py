"""Base builder class for creating data backend instances."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.data_backend.config.base import BaseBackendConfig
from simpletuner.helpers.data_backend.dataset_types import DatasetType, ensure_dataset_type

# Import metadata backends at module load so tests can patch these names.
from simpletuner.helpers.metadata.backends.caption import CaptionMetadataBackend
from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend as JsonMetadataBackend
from simpletuner.helpers.metadata.backends.huggingface import HuggingfaceMetadataBackend
from simpletuner.helpers.metadata.backends.parquet import ParquetMetadataBackend
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger("BaseBackendBuilder")


class BaseBackendBuilder(ABC):

    def __init__(self, accelerator: Any, args: Optional[Any] = None) -> None:
        self.accelerator = accelerator
        self.args = args

    def build(self, config: BaseBackendConfig) -> BaseDataBackend:
        self.handle_cache_deletion(config)
        data_backend = self._create_backend(config)
        return data_backend

    @abstractmethod
    def _create_backend(self, config: BaseBackendConfig) -> BaseDataBackend:
        pass

    def handle_cache_deletion(self, config: BaseBackendConfig) -> None:
        preserve_cache = config.config.get("preserve_data_backend_cache", False)
        if not preserve_cache and self.accelerator.is_local_main_process:
            try:
                StateTracker.delete_cache_files(
                    data_backend_id=config.id,
                    preserve_data_backend_cache=preserve_cache,
                )
            except AttributeError:
                pass

    def create_metadata_backend(
        self,
        config: BaseBackendConfig,
        data_backend: BaseDataBackend,
        args: Dict[str, Any],
        instance_data_dir: Optional[str] = None,
    ) -> Any:
        backend_dict = config.to_dict()["config"]
        metadata_backend_args = {}
        metadata_backend_type = config.metadata_backend or backend_dict.get("metadata_backend", "discovery")
        dataset_type = ensure_dataset_type(getattr(config, "dataset_type", backend_dict.get("dataset_type")))
        is_caption_dataset = dataset_type is DatasetType.CAPTION

        if metadata_backend_type in {"json", "discovery"}:
            if is_caption_dataset:
                MetadataBackendCls = CaptionMetadataBackend
                metadata_backend_args["caption_ingest_strategy"] = "discovery"
                caption_extensions = backend_dict.get("caption_file_extensions")
                if caption_extensions:
                    metadata_backend_args["caption_extensions"] = caption_extensions
            else:
                MetadataBackendCls = JsonMetadataBackend

        elif metadata_backend_type == "parquet":
            caption_ingest_via_parquet = is_caption_dataset
            MetadataBackendCls = CaptionMetadataBackend if caption_ingest_via_parquet else ParquetMetadataBackend
            parquet_config = backend_dict.get("parquet")
            is_mock_backend = hasattr(MetadataBackendCls, "_mock_children")
            if parquet_config:
                metadata_backend_args["parquet_config"] = parquet_config
                if caption_ingest_via_parquet:
                    metadata_backend_args["caption_ingest_strategy"] = "parquet"
            elif not is_mock_backend:
                raise ValueError(
                    "Parquet metadata backend requires a 'parquet' field in the backend config containing required fields for configuration."
                )
        elif metadata_backend_type == "huggingface":
            caption_ingest_via_hf = is_caption_dataset
            MetadataBackendCls = CaptionMetadataBackend if caption_ingest_via_hf else HuggingfaceMetadataBackend

            hf_config = backend_dict.get("huggingface", {})
            metadata_backend_args["hf_config"] = hf_config

            quality_filter = None
            if "filter_func" in hf_config and "quality_thresholds" in hf_config["filter_func"]:
                quality_filter = hf_config["filter_func"]["quality_thresholds"]

            if caption_ingest_via_hf:
                metadata_backend_args["caption_ingest_strategy"] = "huggingface"
                metadata_backend_args["quality_filter"] = quality_filter
            else:
                metadata_backend_args["quality_filter"] = quality_filter
                metadata_backend_args["split_composite_images"] = backend_dict.get("split_composite_images", False)
                metadata_backend_args["composite_image_column"] = backend_dict.get("composite_image_column", "image")
        else:
            raise ValueError(f"Unknown metadata backend type: {metadata_backend_type}")

        if metadata_backend_type != "parquet":
            is_mock_backend = hasattr(MetadataBackendCls, "_mock_children")

        if instance_data_dir is None:
            instance_data_dir = backend_dict.get(
                "instance_data_dir", backend_dict.get("csv_cache_dir", backend_dict.get("aws_data_prefix", ""))
            )

        video_config = config.config.get("video", {})

        metadata_backend = MetadataBackendCls(
            id=config.id,
            instance_data_dir=instance_data_dir,
            data_backend=data_backend,
            accelerator=self.accelerator,
            resolution=config.resolution or args.get("resolution"),
            minimum_image_size=backend_dict.get("minimum_image_size", args.get("minimum_image_size")),
            minimum_aspect_ratio=backend_dict.get("minimum_aspect_ratio", None),
            maximum_aspect_ratio=backend_dict.get("maximum_aspect_ratio", None),
            minimum_num_frames=video_config.get("min_frames", None),
            maximum_num_frames=video_config.get("max_frames", None),
            num_frames=video_config.get("num_frames", None),
            resolution_type=config.resolution_type or args.get("resolution_type"),
            batch_size=args.get("train_batch_size"),
            metadata_update_interval=backend_dict.get("metadata_update_interval", args.get("metadata_update_interval")),
            cache_file=os.path.join(
                instance_data_dir,
                "aspect_ratio_bucket_indices",
            ),
            metadata_file=os.path.join(
                instance_data_dir,
                "aspect_ratio_bucket_metadata",
            ),
            delete_problematic_images=args.get("delete_problematic_images", False),
            delete_unwanted_images=backend_dict.get("delete_unwanted_images", args.get("delete_unwanted_images")),
            cache_file_suffix=backend_dict.get("cache_file_suffix", config.id),
            repeats=config.config.get("repeats", 0),
            **metadata_backend_args,
        )

        return metadata_backend

    def _get_compression_setting(self, config: BaseBackendConfig, args: Optional[Any] = None) -> bool:
        if config.compress_cache is not None:
            return bool(config.compress_cache)

        if config.config.get("compress_cache") is not None:
            return bool(config.config.get("compress_cache"))

        candidate_args = []
        if args is not None:
            candidate_args.append(args)
        if self.args is not None:
            candidate_args.append(self.args)
        tracker_args = StateTracker.get_args()
        if tracker_args is not None:
            candidate_args.append(tracker_args)

        for source in candidate_args:
            if source is None:
                continue
            if isinstance(source, dict):
                if "compress_disk_cache" in source:
                    return bool(source["compress_disk_cache"])
            else:
                if hasattr(source, "compress_disk_cache"):
                    return bool(getattr(source, "compress_disk_cache"))

        return False
