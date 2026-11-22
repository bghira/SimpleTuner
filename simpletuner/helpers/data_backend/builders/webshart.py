"""Webshart backend builder for creating WebshartDataBackend instances."""

import logging
from typing import Any, Dict

from simpletuner.helpers.data_backend.config.base import BaseBackendConfig
from simpletuner.helpers.data_backend.webshart import WebshartDataBackend

from .base import BaseBackendBuilder

logger = logging.getLogger("WebshartBackendBuilder")


class WebshartBackendBuilder(BaseBackendBuilder):
    def _create_backend(self, config: BaseBackendConfig) -> WebshartDataBackend:
        self._validate_webshart_config(config)

        webshart_config = getattr(config, "webshart", None) or {}
        shard_cache_gb = getattr(config, "webshart_shard_cache_gb", None)
        if shard_cache_gb is None:
            shard_cache_gb = 25.0
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
            "shard_cache_gb": shard_cache_gb,
            "parallel_downloads": getattr(config, "webshart_parallel_downloads", None) or 4,
            "buffer_size": getattr(config, "webshart_buffer_size", None) or 100,
            "max_file_size": getattr(config, "webshart_max_file_size", None) or 500 * 1024 * 1024,
            "compress_cache": self._get_compression_setting(config),
            "dataset_type": getattr(config, "dataset_type", "image"),
            "optimize_captions": bool(getattr(config, "webshart_optimize_captions", None) or False),
            "caption_key": config.webshart_caption_key,
            "format": webshart_config.get("format", "tar"),
            "manifest": webshart_config.get("manifest"),
            "shuffle_seed": webshart_config.get("shuffle_seed"),
            "columns": webshart_config.get("columns"),
            **{
                key: getattr(config, key, None)
                for key in (
                    "aws_region_name",
                    "aws_endpoint_url",
                    "aws_access_key_id",
                    "aws_secret_access_key",
                    "aws_session_token",
                )
            },
            "aws_max_pool_connections": getattr(config, "aws_max_pool_connections", None) or 32,
        }
        return WebshartDataBackend(**backend_kwargs)

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
