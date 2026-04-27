"""Webshart backend builder for creating WebshartDataBackend instances."""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from simpletuner.helpers.data_backend.config.base import BaseBackendConfig
from simpletuner.helpers.data_backend.webshart import WebshartDataBackend
from simpletuner.helpers.training.state_tracker import StateTracker

from .base import BaseBackendBuilder

logger = logging.getLogger("WebshartBackendBuilder")
_ORIGINAL_WEBSHART_BACKEND = WebshartDataBackend


class WebshartBackendBuilder(BaseBackendBuilder):
    def _create_backend(self, config: BaseBackendConfig) -> WebshartDataBackend:
        self._validate_webshart_config(config)

        factory_module = sys.modules.get("simpletuner.helpers.data_backend.factory")
        factory_cls = getattr(factory_module, "WebshartDataBackend", None) if factory_module else None
        if WebshartDataBackend is _ORIGINAL_WEBSHART_BACKEND and factory_cls is not None:
            backend_cls = factory_cls
        else:
            backend_cls = WebshartDataBackend
        is_mock_backend = hasattr(backend_cls, "_mock_children")

        webshart_config = getattr(config, "webshart", None) or {}
        cache_dir = getattr(config, "webshart_cache_dir", None)
        if not cache_dir:
            cache_dir = self._default_cache_dir(config)
            config.webshart_cache_dir = str(cache_dir)
            config.config.setdefault("webshart", {})["cache_dir"] = str(cache_dir)

        backend_kwargs = {
            "accelerator": self.accelerator,
            "id": config.id,
            "source": getattr(config, "webshart_source", None),
            "metadata": getattr(config, "webshart_metadata", None),
            "hf_token": getattr(config, "webshart_hf_token", None),
            "subfolder": getattr(config, "webshart_subfolder", None),
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "metadata_cache_dir": webshart_config.get("metadata_cache_dir"),
            "shard_cache_dir": webshart_config.get("shard_cache_dir"),
            "shard_cache_gb": getattr(config, "webshart_shard_cache_gb", 25.0),
            "parallel_downloads": getattr(config, "webshart_parallel_downloads", 4),
            "buffer_size": getattr(config, "webshart_buffer_size", 100),
            "max_file_size": getattr(config, "webshart_max_file_size", 500 * 1024 * 1024),
            "compress_cache": self._get_compression_setting(config),
            "dataset_type": getattr(config, "dataset_type", "image"),
        }
        if is_mock_backend:
            backend_kwargs["identifier"] = config.id

        return backend_cls(**backend_kwargs)

    def _default_cache_dir(self, config: BaseBackendConfig) -> Path:
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
                return Path(candidate).expanduser() / "webshart" / config.id
        return Path("cache") / "webshart" / config.id

    def _validate_webshart_config(self, config: BaseBackendConfig) -> None:
        if not getattr(config, "webshart_source", None):
            raise ValueError("source is required for Webshart backends.")

        metadata_backend = getattr(config, "metadata_backend", None) or "webshart"
        if metadata_backend != "webshart":
            raise ValueError("metadata_backend must be 'webshart' for Webshart backends.")

        caption_strategy = getattr(config, "caption_strategy", None) or "webshart"
        if caption_strategy not in {"webshart", "instanceprompt"}:
            raise ValueError("caption_strategy must be 'webshart' or 'instanceprompt' for Webshart backends.")

    def build_with_metadata(self, config: BaseBackendConfig, args: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("(id=%s) Loading Webshart dataset.", config.id)
        data_backend = self.build(config)
        instance_data_dir = str(getattr(config, "webshart_cache_dir", None) or self._default_cache_dir(config))
        metadata_backend = self.create_metadata_backend(
            config=config,
            data_backend=data_backend,
            args=args,
            instance_data_dir=instance_data_dir,
        )
        return {
            "id": config.id,
            "data_backend": data_backend,
            "metadata_backend": metadata_backend,
            "instance_data_dir": instance_data_dir,
            "config": config.to_dict()["config"],
        }
