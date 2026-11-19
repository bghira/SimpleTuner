"""Local backend builder for creating LocalDataBackend instances."""

import logging
from typing import Any, Dict, Optional

from simpletuner.helpers.data_backend.config.base import BaseBackendConfig
from simpletuner.helpers.data_backend.local import LocalDataBackend

from .base import BaseBackendBuilder

logger = logging.getLogger("LocalBackendBuilder")


class LocalBackendBuilder(BaseBackendBuilder):

    def _create_backend(self, config: BaseBackendConfig) -> LocalDataBackend:
        compress_cache = self._get_compression_setting(config)

        return LocalDataBackend(accelerator=self.accelerator, id=config.id, compress_cache=compress_cache)

    def build_with_metadata(
        self, config: BaseBackendConfig, args: Dict[str, Any], instance_data_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info(f"(id={config.id}) Loading local dataset.")

        if instance_data_dir is None:
            instance_data_dir = getattr(config, "instance_data_dir", "") or config.config.get("instance_data_dir", "")

        data_backend = self.build(config)

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
