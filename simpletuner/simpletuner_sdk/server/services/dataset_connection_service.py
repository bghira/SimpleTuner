"""Connection testing helpers for dataset backends."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import status

from simpletuner.helpers.data_backend.aws import test_s3_connection
from simpletuner.helpers.data_backend.csv_url_list import test_csv_manifest
from simpletuner.helpers.data_backend.huggingface import test_huggingface_dataset
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore
from simpletuner.simpletuner_sdk.server.utils.paths import resolve_config_path


class DatasetConnectionError(Exception):
    """Raised when a dataset connection test fails."""

    def __init__(self, message: str, *, status_code: int = status.HTTP_400_BAD_REQUEST, backend: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.backend = backend


class DatasetConnectionService:
    """Service for validating dataset connectivity before saving configurations."""

    def __init__(self, configs_dir: Optional[str | Path] = None) -> None:
        self._explicit_configs_dir = Path(configs_dir).expanduser() if configs_dir else None

    def _default_configs_dir(self) -> Optional[Path]:
        if self._explicit_configs_dir is not None:
            return self._explicit_configs_dir
        try:
            defaults = WebUIStateStore().load_defaults()
            if defaults.configs_dir:
                return Path(defaults.configs_dir).expanduser()
        except Exception:
            return None
        return None

    def _resolve_path(self, value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        configs_dir = self._default_configs_dir()
        resolved: Optional[Path] = None
        try:
            resolved = resolve_config_path(value, config_dir=configs_dir, check_cwd_first=True)
        except Exception:
            resolved = None
        if resolved:
            return resolved
        candidate = Path(os.path.expanduser(value))
        if candidate.exists():
            return candidate
        return None

    def test_connection(self, dataset: Dict[str, Any], configs_dir: Optional[str | Path] = None) -> Dict[str, Any]:
        if configs_dir:
            self._explicit_configs_dir = Path(configs_dir).expanduser()

        dataset_type = (dataset or {}).get("type") or "local"
        dataset_type = str(dataset_type).lower()

        if dataset_type == "csv":
            return self._test_csv(dataset)
        if dataset_type == "huggingface":
            return self._test_huggingface(dataset)
        if dataset_type == "aws":
            return self._test_aws(dataset)

        raise DatasetConnectionError(f"Connection tests are not supported for dataset type '{dataset_type}'.", backend=dataset_type)

    def _test_csv(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        csv_file = dataset.get("csv_file")
        caption_column = dataset.get("csv_caption_column") or dataset.get("caption_column")
        url_column = dataset.get("csv_url_column") or dataset.get("url_column") or dataset.get("url")

        resolved = self._resolve_path(csv_file)
        target = str(resolved) if resolved else csv_file
        if not target:
            raise DatasetConnectionError("CSV file path is required", backend="csv")

        try:
            details = test_csv_manifest(target, caption_column=caption_column, url_column=url_column)
        except ValueError as exc:
            raise DatasetConnectionError(str(exc), backend="csv") from exc

        message = "CSV manifest looks good"
        if details.get("warnings"):
            message = "CSV manifest accessible with warnings"

        details.update({
            "resolved_path": str(resolved) if resolved else None,
        })

        return {
            "status": "ok",
            "backend": "csv",
            "message": message,
            "details": details,
        }

    def _test_huggingface(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name = dataset.get("dataset_name")
        dataset_config = dataset.get("dataset_config") or dataset.get("config")
        split = dataset.get("split")
        revision = dataset.get("revision")
        streaming = bool(dataset.get("streaming"))
        auth_token = dataset.get("auth_token") or dataset.get("use_auth_token")

        try:
            details = test_huggingface_dataset(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                revision=revision,
                streaming=streaming,
                use_auth_token=auth_token,
                sample_count=1,
            )
        except ImportError as exc:
            raise DatasetConnectionError(str(exc), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, backend="huggingface") from exc
        except ValueError as exc:
            raise DatasetConnectionError(str(exc), backend="huggingface") from exc

        message = f"Hugging Face dataset '{dataset_name}' is reachable"
        if details.get("available_splits") and split and split not in details["available_splits"]:
            message = f"Dataset reachable, but split '{split}' not found"

        return {
            "status": "ok",
            "backend": "huggingface",
            "message": message,
            "details": details,
        }

    def _test_aws(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        bucket = dataset.get("aws_bucket_name")
        prefix = dataset.get("aws_data_prefix")
        region = dataset.get("aws_region_name")
        endpoint = dataset.get("aws_endpoint_url")
        access_key = dataset.get("aws_access_key_id")
        secret_key = dataset.get("aws_secret_access_key")

        try:
            details = test_s3_connection(
                bucket_name=bucket,
                prefix=prefix,
                region_name=region,
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
        except ValueError as exc:
            raise DatasetConnectionError(str(exc), backend="aws") from exc

        message = f"S3 bucket '{bucket}' reachable"
        if not details.get("sample_keys"):
            message = f"S3 bucket '{bucket}' reachable but no objects found for prefix"

        return {
            "status": "ok",
            "backend": "aws",
            "message": message,
            "details": details,
        }
