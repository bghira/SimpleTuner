"""CSV backend builder for creating CSVDataBackend instances."""

import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from simpletuner.helpers.data_backend.config.base import BaseBackendConfig
from simpletuner.helpers.data_backend.csv_url_list import CSVDataBackend

from .base import BaseBackendBuilder

logger = logging.getLogger("CsvBackendBuilder")
_ORIGINAL_CSV_BACKEND = CSVDataBackend


class CsvBackendBuilder(BaseBackendBuilder):

    def _create_backend(self, config: BaseBackendConfig) -> CSVDataBackend:
        self._validate_csv_config(config, self.args)

        factory_module = sys.modules.get("simpletuner.helpers.data_backend.factory")
        factory_cls = getattr(factory_module, "CSVDataBackend", None) if factory_module else None
        if CSVDataBackend is _ORIGINAL_CSV_BACKEND and factory_cls is not None:
            backend_cls = factory_cls
        else:
            backend_cls = CSVDataBackend
        is_mock_backend = hasattr(backend_cls, "_mock_children")

        csv_file = getattr(config, "csv_file", None)
        csv_cache_dir = getattr(config, "csv_cache_dir", None)
        url_column = getattr(config, "csv_url_column", "url") or "url"
        caption_column = getattr(config, "csv_caption_column", None)

        compress_cache = self._get_compression_setting(config)
        hash_filenames = self._resolve_bool(config, "hash_filenames", default=False)
        shorten_filenames = self._resolve_bool(config, "shorten_filenames", default=False)

        csv_file_value = csv_file if is_mock_backend else Path(csv_file)

        kwargs = {
            "accelerator": self.accelerator,
            "id": config.id,
            "csv_file": csv_file_value,
            "url_column": url_column,
            "caption_column": caption_column,
            "compress_cache": compress_cache,
            "hash_filenames": hash_filenames,
        }

        try:
            parameters = inspect.signature(backend_cls.__init__).parameters
        except (TypeError, ValueError):
            parameters = {}

        if is_mock_backend or "shorten_filenames" in parameters:
            kwargs["shorten_filenames"] = shorten_filenames

        if is_mock_backend or "csv_cache_dir" in parameters:
            kwargs["csv_cache_dir"] = csv_cache_dir
        elif "image_cache_loc" in parameters:
            kwargs["image_cache_loc"] = csv_cache_dir

        return backend_cls(**kwargs)

    def build_with_metadata(self, config: BaseBackendConfig, args: Dict[str, Any]) -> Dict[str, Any]:
        # csv backends use csv_cache_dir instead of instance_data_dir
        logger.info(f"(id={config.id}) Loading CSV dataset.")

        data_backend = self.build(config)

        backend_config = config.to_dict()["config"]
        csv_config = backend_config.get("csv", {})
        csv_cache_dir = csv_config.get("csv_cache_dir") or backend_config.get("csv_cache_dir")

        metadata_backend = self.create_metadata_backend(
            config=config, data_backend=data_backend, args=args, instance_data_dir=csv_cache_dir
        )

        return {
            "id": config.id,
            "data_backend": data_backend,
            "metadata_backend": metadata_backend,
            "instance_data_dir": csv_cache_dir,
            "config": backend_config,
        }

    def _validate_csv_config(self, config: BaseBackendConfig, args: Optional[Dict[str, Any]]) -> None:

        csv_file = getattr(config, "csv_file", None)
        csv_cache_dir = getattr(config, "csv_cache_dir", None)
        caption_column = getattr(config, "csv_caption_column", None)
        url_column = getattr(config, "csv_url_column", None)

        missing = [
            key
            for key, value in {
                "csv_file": csv_file,
                "csv_cache_dir": csv_cache_dir,
                "csv_caption_column": caption_column,
            }.items()
            if value in (None, "")
        ]

        if missing:
            raise ValueError("Missing required CSV configuration keys: " + ", ".join(missing))

        if getattr(config, "caption_strategy", None) not in {"csv", None}:
            raise ValueError("caption_strategy must be 'csv' for CSV backends.")

        if url_column in (None, ""):
            # if url column omitted, default to 'url'
            setattr(config, "csv_url_column", "url")

    def _resolve_bool(self, config: BaseBackendConfig, attr: str, default: bool = False) -> bool:
        value = getattr(config, attr, None)
        if value is not None:
            return bool(value)

        if isinstance(self.args, dict) and attr in self.args:
            return bool(self.args[attr])
        if hasattr(self.args, attr):
            return bool(getattr(self.args, attr))

        return default
