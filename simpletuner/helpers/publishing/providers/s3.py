from __future__ import annotations

import json
import logging
from urllib.parse import urlparse
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

    def resolve_key_prefix(self, remote_path: str) -> str:
        if not remote_path:
            raise ValueError("Resume checkpoint path is required.")
        candidate = remote_path.strip()
        if candidate.startswith(("s3://", "r2://")):
            parsed = urlparse(candidate)
            bucket = parsed.netloc
            if bucket and bucket != self.bucket:
                raise ValueError(f"Resume checkpoint bucket '{bucket}' does not match configured bucket '{self.bucket}'.")
            return parsed.path.lstrip("/").rstrip("/")
        if candidate.startswith(("http://", "https://")):
            if self.public_base_url and candidate.startswith(self.public_base_url.rstrip("/")):
                return candidate[len(self.public_base_url.rstrip("/")) :].lstrip("/").rstrip("/")
            if self.endpoint_url and candidate.startswith(self.endpoint_url.rstrip("/")):
                remainder = candidate[len(self.endpoint_url.rstrip("/")) :].lstrip("/")
                if remainder.startswith(f"{self.bucket}/"):
                    return remainder[len(f"{self.bucket}/") :].rstrip("/")
            raise ValueError("Resume checkpoint URL does not match configured S3 endpoints.")
        if candidate.startswith("/"):
            return candidate.lstrip("/").rstrip("/")
        return candidate.rstrip("/")

    def download_checkpoint(
        self,
        *,
        key_prefix: str,
        local_dir: str | Path,
        manifest_filename: str,
    ) -> dict[str, Any]:
        key_prefix = key_prefix.strip("/").rstrip("/")
        manifest_key = f"{key_prefix}/{manifest_filename}"
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=manifest_key)
        except self._client.exceptions.NoSuchKey as exc:
            raise FileNotFoundError("Remote checkpoint manifest not found.") from exc

        manifest_bytes = response["Body"].read()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if not isinstance(manifest, dict) or "files" not in manifest:
            raise ValueError("Remote checkpoint manifest is invalid or missing file list.")
        files = manifest.get("files")
        if not isinstance(files, list):
            raise ValueError("Remote checkpoint manifest files must be a list.")

        for entry in files:
            if not isinstance(entry, str):
                raise ValueError("Remote checkpoint manifest entries must be strings.")
            if ".." in entry or entry.startswith(("/", "\\")):
                raise ValueError("Remote checkpoint manifest includes unsafe paths.")
            remote_key = f"{key_prefix}/{entry.lstrip('/')}"
            target_path = local_dir / entry
            target_path.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self.bucket, remote_key, str(target_path))
        manifest_path = local_dir / manifest_filename
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return manifest

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
