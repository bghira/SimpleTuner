"""Hugging Face backend builder for creating HuggingfaceDatasetsBackend instances."""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Callable

from simpletuner.helpers.data_backend.huggingface import HuggingfaceDatasetsBackend
from simpletuner.helpers.data_backend.config.base import BaseBackendConfig
from simpletuner.helpers.training.state_tracker import StateTracker
from .base import BaseBackendBuilder

logger = logging.getLogger("HuggingfaceBackendBuilder")
_ORIGINAL_HF_BACKEND = HuggingfaceDatasetsBackend


class HuggingfaceBackendBuilder(BaseBackendBuilder):

    def _create_backend(self, config: BaseBackendConfig) -> HuggingfaceDatasetsBackend:
        self._validate_huggingface_config(config)

        factory_module = sys.modules.get("simpletuner.helpers.data_backend.factory")
        factory_cls = getattr(factory_module, "HuggingfaceDatasetsBackend", None) if factory_module else None
        if HuggingfaceDatasetsBackend is _ORIGINAL_HF_BACKEND and factory_cls is not None:
            backend_cls = factory_cls
        else:
            backend_cls = HuggingfaceDatasetsBackend
        is_mock_backend = hasattr(backend_cls, "_mock_children")

        dataset_name = getattr(config, "dataset_name", None)
        dataset_type = getattr(config, "dataset_type", "image")
        split = getattr(config, "split", "train")
        revision = getattr(config, "revision", None)
        image_column = getattr(config, "image_column", "image")
        video_column = getattr(config, "video_column", "video")
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
            "dataset_type": dataset_type,
            "split": split,
            "revision": revision,
            "image_column": image_column,
            "video_column": video_column,
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

        if is_mock_backend:
            backend_kwargs["identifier"] = config.id

        data_backend = backend_cls(**backend_kwargs)

        return data_backend

    def _default_cache_dir(self, config: BaseBackendConfig) -> Optional[Path]:
        candidates = []

        if isinstance(self.args, dict):
            candidates.append(self.args.get("cache_dir"))
        elif self.args is not None:
            candidates.append(getattr(self.args, "cache_dir", None))

        state_args = getattr(StateTracker, "get_args", lambda: None)()
        if isinstance(state_args, dict):
            candidates.append(state_args.get("cache_dir"))
        elif state_args is not None:
            candidates.append(getattr(state_args, "cache_dir", None))

        candidates.append(os.environ.get("SIMPLETUNER_CACHE_DIR", None))

        for candidate in candidates:
            if candidate:
                base = Path(candidate).expanduser()
                return base / "huggingface" / config.id

        return Path("cache") / "huggingface" / config.id

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
            raise ValueError(
                "metadata_backend must be 'huggingface' for HuggingFace backends."
            )

        caption_strategy = getattr(config, "caption_strategy", "huggingface")
        if caption_strategy not in {"huggingface", "instanceprompt"}:
            raise ValueError(
                "caption_strategy must be 'huggingface' or 'instanceprompt' for HuggingFace backends."
            )

    def _create_filter_function(self, backend_config: Dict[str, Any]) -> Optional[Callable]:
        hf_config = backend_config.get("huggingface", {})
        filter_config = hf_config.get("filter_func", backend_config.get("filter_func"))

        if not filter_config:
            return None

        def filter_func(item):
            if "collection" in filter_config:
                required_collections = filter_config["collection"]
                if isinstance(required_collections, str):
                    required_collections = [required_collections]
                if item.get("collection") not in required_collections:
                    return False

            if "quality_thresholds" in filter_config:
                quality = item.get(filter_config.get("quality_column", "quality_assessment"), {})
                if not quality:
                    return False
                for metric, threshold in filter_config["quality_thresholds"].items():
                    if quality.get(metric, 0) < threshold:
                        return False

            if "min_width" in filter_config and item.get("width", 0) < filter_config["min_width"]:
                return False
            if "min_height" in filter_config and item.get("height", 0) < filter_config["min_height"]:
                return False

            return True

        return filter_func

    def build_with_metadata(
        self,
        config: BaseBackendConfig,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        # huggingface datasets don't have local instance_data_dir
        logger.info(f"(id={config.id}) Loading HuggingFace dataset.")

        data_backend = self.build(config)

        instance_data_dir = ""

        metadata_backend = self.create_metadata_backend(
            config=config,
            data_backend=data_backend,
            args=args,
            instance_data_dir=instance_data_dir
        )

        return {
            "id": config.id,
            "data_backend": data_backend,
            "metadata_backend": metadata_backend,
            "instance_data_dir": instance_data_dir,
            "config": config.to_dict()["config"]
        }
