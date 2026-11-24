"""AWS backend builder for creating S3DataBackend instances."""

import logging
from collections.abc import Sequence
from typing import Any, Dict, Optional, Union

from simpletuner.helpers.data_backend.aws import S3DataBackend
from simpletuner.helpers.data_backend.config.base import BaseBackendConfig

from .base import BaseBackendBuilder

logger = logging.getLogger("AwsBackendBuilder")


class AwsBackendBuilder(BaseBackendBuilder):

    def _create_backend(self, config: BaseBackendConfig) -> S3DataBackend:
        self._validate_aws_config(config)

        aws_bucket_name = getattr(config, "aws_bucket_name", None)
        aws_region_name = getattr(config, "aws_region_name", None)
        aws_endpoint_url = getattr(config, "aws_endpoint_url", None)
        aws_access_key_id = getattr(config, "aws_access_key_id", None)
        aws_secret_access_key = getattr(config, "aws_secret_access_key", None)
        aws_session_token = getattr(config, "aws_session_token", None)

        compress_cache = self._get_compression_setting(config)
        max_pool_connections = self._resolve_max_pool_connections(config)

        data_backend = S3DataBackend(
            id=config.id,
            bucket_name=aws_bucket_name,
            accelerator=self.accelerator,
            region_name=aws_region_name,
            endpoint_url=aws_endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            compress_cache=compress_cache,
            max_pool_connections=max_pool_connections,
            data_prefix=getattr(config, "aws_data_prefix", None),
        )

        return data_backend

    def _validate_aws_config(self, config: BaseBackendConfig) -> None:
        if not getattr(config, "aws_bucket_name", None):
            raise ValueError("Missing required AWS configuration key: aws_bucket_name")

    def _resolve_max_pool_connections(self, config: BaseBackendConfig) -> int:

        config_value = getattr(config, "aws_max_pool_connections", None)
        if config_value is not None:
            return int(config_value)

        args_value = self._get_from_args("aws_max_pool_connections")
        if args_value is not None:
            return int(args_value)

        return 128

    def _get_from_args(self, key: str, source: Optional[Any] = None) -> Optional[Any]:

        for candidate in filter(None, (source, self.args, self._state_tracker_args())):
            if isinstance(candidate, dict) and key in candidate:
                return candidate[key]
            if hasattr(candidate, key):
                return getattr(candidate, key)
        return None

    def _state_tracker_args(self) -> Optional[Any]:
        from simpletuner.helpers.training.state_tracker import StateTracker

        return StateTracker.get_args()

    def build_with_metadata(
        self, config: BaseBackendConfig, args: Dict[str, Any], aws_data_prefix: Union[str, Sequence[str], None] = None
    ) -> Dict[str, Any]:
        logger.info(f"(id={config.id}) Loading AWS S3 dataset.")

        data_backend = self.build(config)

        backend_config = config.to_dict()["config"]
        if aws_data_prefix is None:
            aws_data_prefix = backend_config.get("aws_data_prefix", "")

        metadata_backend = self.create_metadata_backend(
            config=config, data_backend=data_backend, args=args, instance_data_dir=aws_data_prefix
        )

        return {
            "id": config.id,
            "data_backend": data_backend,
            "metadata_backend": metadata_backend,
            "instance_data_dir": aws_data_prefix,
            "config": backend_config,
        }
