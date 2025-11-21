from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any, Optional

from .base import PublishingProvider, PublishingResult

logger = logging.getLogger(__name__)


class AzureBlobPublishingProvider(PublishingProvider):
    """Publish artifacts to Azure Blob Storage."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(provider_type="azure_blob", config=config, display_name=config.get("name"))
        try:
            from azure.core.exceptions import ResourceExistsError
            from azure.storage.blob import BlobServiceClient, ContentSettings
        except ImportError as exc:  # pragma: no cover - only hit when dependency missing
            raise ImportError(
                "azure-storage-blob is required for Azure publishing. "
                "Install with `pip install azure-storage-blob` inside your .venv."
            ) from exc

        container = config.get("container") or config.get("container_name")
        if not container:
            raise ValueError("Azure publishing requires a container value in publishing_config.")
        connection_string = config.get("connection_string")
        account_url = config.get("account_url")
        account_name = config.get("account_name")
        account_key = config.get("account_key")

        if connection_string:
            self._service_client = BlobServiceClient.from_connection_string(connection_string)
        elif account_url and account_key:
            self._service_client = BlobServiceClient(account_url=account_url, credential=account_key)
        elif account_name and account_key:
            self._service_client = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key
            )
        else:
            raise ValueError(
                "Azure publishing requires either connection_string, or account_url/account_name with account_key."
            )

        self._container_client = self._service_client.get_container_client(container)
        try:
            self._container_client.create_container()
        except Exception as exc:  # pragma: no cover - dependency guard
            if "ResourceExists" not in exc.__class__.__name__:
                logger.debug("Container creation raised %s; proceeding with existing container.", exc)

        self._content_settings_cls = ContentSettings

    def _content_settings(self, file_path: Path):
        mime_guess, _ = mimetypes.guess_type(str(file_path))
        if not mime_guess:
            return None
        return self._content_settings_cls(content_type=mime_guess)

    def _build_uri(self, destination_root: str) -> str:
        return f"{self._container_client.url.rstrip('/')}/{destination_root.lstrip('/')}"

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
        last_blob: Optional[str] = None

        for file_path in files:
            blob_name = "/".join(
                [
                    part
                    for part in (
                        destination_root,
                        file_path.relative_to(path).as_posix() if path.is_dir() else file_path.name,
                    )
                    if part
                ]
            )
            with open(file_path, "rb") as handle:
                self._container_client.upload_blob(
                    name=blob_name,
                    data=handle,
                    overwrite=True,
                    content_settings=self._content_settings(file_path),
                )
            last_blob = blob_name

        assert last_blob is not None
        uri = self._build_uri(destination_root)
        logger.info("Published %s file(s) to Azure Blob Storage at %s", len(files), uri)
        return self._record_result(artifact_path=path, uri=uri, metadata=metadata)
