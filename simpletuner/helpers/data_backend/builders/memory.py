"""Memory backend builder for tmpfs-backed cache storage."""

from typing import Any, Optional

from simpletuner.helpers.data_backend.config.base import BaseBackendConfig
from simpletuner.helpers.data_backend.dataset_types import DatasetType, ensure_dataset_type
from simpletuner.helpers.data_backend.memory import MemoryDataBackend

from .base import BaseBackendBuilder


class MemoryBackendBuilder(BaseBackendBuilder):
    def _get_arg(self, key: str, default: Any = None) -> Any:
        if isinstance(self.args, dict):
            return self.args.get(key, default)
        return getattr(self.args, key, default)

    def _source_path(self, config: BaseBackendConfig) -> Optional[str]:
        dataset_type = ensure_dataset_type(config.dataset_type, default=DatasetType.IMAGE)
        if dataset_type is DatasetType.TEXT_EMBEDS:
            return config.config.get("cache_dir") or self._get_arg("cache_dir_text")
        if dataset_type is DatasetType.IMAGE_EMBEDS:
            return config.config.get("cache_dir") or self._get_arg("cache_dir_vae")
        raise ValueError("Memory backends only support dataset_type='text_embeds' or 'image_embeds'.")

    def _create_backend(self, config: BaseBackendConfig) -> MemoryDataBackend:
        return MemoryDataBackend(
            accelerator=self.accelerator,
            id=config.id,
            source_path=self._source_path(config),
            mount_path=config.config.get("memory_filesystem_path"),
            filesystem_size=config.config.get("memory_filesystem_size"),
            filesystem_sudo=bool(config.config.get("memory_filesystem_sudo", False)),
            compress_cache=self._get_compression_setting(config),
        )
