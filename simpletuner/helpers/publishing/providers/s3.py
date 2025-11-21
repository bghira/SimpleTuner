from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .base import PublishingProvider, PublishingResult

logger = logging.getLogger(__name__)


class S3PublishingProvider(PublishingProvider):
    """Publish artifacts to any S3-compatible storage backend."""

    def __init__(self, config: dict[str, Any], *, provider_type: str = "s3"):
        super().__init__(provider_type=provider_type, config=config, display_name=config.get("name"))
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - exercised only when dependency missing
            raise ImportError(
                "boto3 is required for S3 publishing. Install with `pip install boto3` inside your .venv."
            ) from exc

        session_kwargs: dict[str, Any] = {
            "aws_access_key_id": config.get("access_key") or config.get("aws_access_key_id"),
            "aws_secret_access_key": config.get("secret_key") or config.get("aws_secret_access_key"),
            "region_name": config.get("region") or config.get("region_name"),
        }
        profile = config.get("profile")
        if profile:
            session_kwargs["profile_name"] = profile

        self.bucket = config.get("bucket")
        if not self.bucket:
            raise ValueError("S3 publishing requires a bucket name in publishing_config.")
        self.endpoint_url = config.get("endpoint_url")
        self.public_base_url = config.get("public_base_url")
        self.use_ssl = bool(config.get("use_ssl", True))
        self._session = boto3.session.Session(**{k: v for k, v in session_kwargs.items() if v is not None})
        self._client = self._session.client("s3", endpoint_url=self.endpoint_url, use_ssl=self.use_ssl)

    def _build_uri(self, key_prefix: str) -> str:
        if self.public_base_url:
            return f"{self.public_base_url.rstrip('/')}/{key_prefix.lstrip('/')}"
        if self.endpoint_url:
            return f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{key_prefix.lstrip('/')}"
        return f"s3://{self.bucket}/{key_prefix.lstrip('/')}"

    def publish(
        self,
        artifact_path: str | Path,
        *,
        artifact_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PublishingResult:
        path = Path(artifact_path)
        metadata = dict(metadata) if metadata else {}
        files = self._iter_files(path)
        if not files:
            raise ValueError(f"No files found to publish under {path}")

        destination_root = self._build_destination_root(artifact_name or path.name) or path.name

        last_key: Optional[str] = None
        for file_path in files:
            relative_key = file_path.relative_to(path).as_posix() if path.is_dir() else file_path.name
            destination_key = "/".join([part for part in (destination_root, relative_key) if part])
            self._client.upload_file(str(file_path), self.bucket, destination_key)
            last_key = destination_key

        assert last_key is not None  # for mypy/pylint; guarded by files check
        uri = self._build_uri("/".join(part for part in destination_root.split("/") if part))
        logger.info("Published %s file(s) to %s", len(files), uri)
        return self._record_result(artifact_path=path, uri=uri, metadata=metadata)


class BackblazeB2PublishingProvider(S3PublishingProvider):
    """Backblaze B2 via its S3-compatible API."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config, provider_type="backblaze_b2")
