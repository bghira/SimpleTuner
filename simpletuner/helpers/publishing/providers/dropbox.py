from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base import PublishingProvider, PublishingResult

logger = logging.getLogger(__name__)


class DropboxPublishingProvider(PublishingProvider):
    """Publish artifacts to Dropbox using the official SDK."""

    CHUNK_SIZE = 4 * 1024 * 1024

    def __init__(self, config: dict[str, Any]):
        super().__init__(provider_type="dropbox", config=config, display_name=config.get("name"))
        try:
            import dropbox
            from dropbox.files import CommitInfo, UploadSessionCursor, WriteMode
        except ImportError as exc:  # pragma: no cover - only hit when dependency missing
            raise ImportError(
                "dropbox is required for Dropbox publishing. Install with `pip install dropbox` inside your .venv."
            ) from exc

        token = config.get("token") or config.get("access_token")
        if not token:
            raise ValueError("Dropbox publishing requires an access token in publishing_config.")

        self._dbx = dropbox.Dropbox(token)
        self._write_mode = WriteMode
        self._commit_info = CommitInfo
        self._upload_cursor = UploadSessionCursor

    def _destination_root(self, artifact_name: str | None) -> str:
        base_path = self.config.get("base_path") or "/"
        if not base_path.startswith("/"):
            base_path = f"/{base_path}"
        base_path = base_path.rstrip("/") or "/"
        clean_artifact = artifact_name.strip("/") if artifact_name else ""
        if clean_artifact:
            return f"{base_path}/{clean_artifact}"
        return base_path

    def _ensure_shared_link(self, dropbox_path: str) -> str | None:
        try:
            link = self._dbx.sharing_create_shared_link_with_settings(dropbox_path)
            return link.url
        except Exception as exc:
            logger.debug("Unable to create Dropbox shared link for %s: %s", dropbox_path, exc)
            return None

    def _upload_file(self, source: Path, destination: str):
        file_size = source.stat().st_size
        with open(source, "rb") as handle:
            if file_size <= self.CHUNK_SIZE:
                self._dbx.files_upload(handle.read(), destination, mode=self._write_mode.overwrite)
                return

            start_result = self._dbx.files_upload_session_start(handle.read(self.CHUNK_SIZE))
            cursor = self._upload_cursor(session_id=start_result.session_id, offset=handle.tell())
            commit = self._commit_info(path=destination, mode=self._write_mode.overwrite)

            while handle.tell() < file_size:
                if (file_size - handle.tell()) <= self.CHUNK_SIZE:
                    self._dbx.files_upload_session_finish(handle.read(), cursor, commit)
                else:
                    self._dbx.files_upload_session_append_v2(handle.read(self.CHUNK_SIZE), cursor)
                    cursor.offset = handle.tell()

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

        destination_root = self._destination_root(artifact_name or path.name)
        last_path = destination_root

        for file_path in files:
            remote_path = "/".join(
                segment
                for segment in (
                    destination_root.rstrip("/"),
                    file_path.relative_to(path).as_posix() if path.is_dir() else file_path.name,
                )
                if segment
            )
            if not remote_path.startswith("/"):
                remote_path = f"/{remote_path}"
            self._upload_file(file_path, remote_path)
            last_path = remote_path

        uri = self._ensure_shared_link(destination_root) or f"dropbox://{destination_root.lstrip('/')}"
        logger.info("Published %s file(s) to Dropbox at %s", len(files), uri)
        return self._record_result(artifact_path=path, uri=uri, metadata=metadata)
