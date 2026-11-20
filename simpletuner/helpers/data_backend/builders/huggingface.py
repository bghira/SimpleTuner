"""Hugging Face backend builder for creating HuggingfaceDatasetsBackend instances."""

import logging
from typing import Any, Callable, Dict, Optional

from simpletuner.helpers.data_backend.config.base import BaseBackendConfig
from simpletuner.helpers.data_backend.filters import build_dataset_filter
from simpletuner.helpers.data_backend.huggingface import HuggingfaceDatasetsBackend

from .base import BaseBackendBuilder

logger = logging.getLogger("HuggingfaceBackendBuilder")


class HuggingfaceBackendBuilder(BaseBackendBuilder):

    def _create_backend(self, config: BaseBackendConfig) -> HuggingfaceDatasetsBackend:
        self._validate_huggingface_config(config)

        dataset_name = getattr(config, "dataset_name", None)
        dataset_config = getattr(config, "dataset_config", None)
        data_files = getattr(config, "data_files", None)
        dataset_type = getattr(config, "dataset_type", "image")
        split = getattr(config, "split", "train")
        revision = getattr(config, "revision", None)
        image_column = getattr(config, "image_column", "image")
        video_column = getattr(config, "video_column", "video")
        audio_column = getattr(config, "audio_column", "audio")
        cache_dir = getattr(config, "huggingface_cache_dir", None)
        if not cache_dir:
            cache_dir = self._default_cache_dir(config)
            if cache_dir is not None:
                config.huggingface_cache_dir = str(cache_dir)
                config.config.setdefault("huggingface", {})["cache_dir"] = str(cache_dir)
        cache_dir = str(cache_dir) if cache_dir is not None else None
        streaming = getattr(config, "huggingface_streaming", False)
        num_proc = getattr(config, "huggingface_num_proc", 16)

        compress_cache = self._get_compression_setting(config)

        backend_config = config.to_dict()["config"]
        filter_func = self._create_filter_function(backend_config)

        composite_config = getattr(config, "huggingface_composite_config", None)

        backend_kwargs = {
            "accelerator": self.accelerator,
            "id": config.id,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "data_files": data_files,
            "dataset_type": dataset_type,
            "split": split,
            "revision": revision,
            "image_column": image_column,
            "video_column": video_column,
            "audio_column": audio_column,
            "cache_dir": cache_dir,
            "compress_cache": compress_cache,
            "streaming": streaming,
            "filter_func": filter_func,
            "num_proc": num_proc,
            "composite_config": composite_config,
        }

        auto_load = getattr(config, "huggingface_auto_load", None)
        if auto_load is None:
            auto_load = backend_config.get("huggingface", {}).get("auto_load")
        if isinstance(auto_load, str):
            auto_load = auto_load.lower() in {"1", "true", "yes", "y", "on"}
        backend_kwargs["auto_load"] = bool(auto_load) if auto_load is not None else False

        data_backend = HuggingfaceDatasetsBackend(**backend_kwargs)

        return data_backend

    def _validate_huggingface_config(self, config: BaseBackendConfig) -> None:

        dataset_name = getattr(config, "dataset_name", None)
        if not dataset_name:
            raise ValueError("dataset_name is required for HuggingFace backends.")

        if not getattr(config, "has_huggingface_block", False):
            logger.debug(
                "(id=%s) HuggingFace backend config missing explicit 'huggingface' block; assuming defaults for backwards compatibility.",
                getattr(config, "id", "unknown"),
            )

        metadata_backend = getattr(config, "metadata_backend", None) or "huggingface"
        if metadata_backend not in {"huggingface"}:
            raise ValueError("metadata_backend must be 'huggingface' for HuggingFace backends.")

        caption_strategy = getattr(config, "caption_strategy", "huggingface")
        if caption_strategy not in {"huggingface", "instanceprompt"}:
            raise ValueError("caption_strategy must be 'huggingface' or 'instanceprompt' for HuggingFace backends.")

    def _create_filter_function(self, backend_config: Dict[str, Any]) -> Optional[Callable]:
        dataset_filter = build_dataset_filter(backend_config)
        if dataset_filter is None:
            return None

        def filter_func(item):
            return dataset_filter.matches_item(item)

        return filter_func

    def build_with_metadata(self, config: BaseBackendConfig, args: Dict[str, Any]) -> Dict[str, Any]:
        # huggingface datasets don't have local instance_data_dir
        logger.info(f"(id={config.id}) Loading HuggingFace dataset.")

        data_backend = self.build(config)

        instance_data_dir = ""

        metadata_backend = self.create_metadata_backend(
            config=config, data_backend=data_backend, args=args, instance_data_dir=instance_data_dir
        )

        return {
            "id": config.id,
            "data_backend": data_backend,
            "metadata_backend": metadata_backend,
            "instance_data_dir": instance_data_dir,
            "config": config.to_dict()["config"],
        }
