"""Cloud upload service for packaging and uploading datasets."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import httpx

from .base import CloudUploadBackend
from .secrets import get_secrets_manager

logger = logging.getLogger(__name__)

# Directories that should NEVER be uploaded (absolute paths after resolution)
FORBIDDEN_PATHS: Set[str] = {
    "/etc",
    "/var",
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
    "/root",
    "/run",
    "/tmp",
}

# Sensitive file patterns to exclude
SENSITIVE_PATTERNS: Set[str] = {
    ".git",
    ".env",
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".config",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
}


def _is_safe_path(path: Path, allowed_base: Optional[Path] = None) -> bool:
    """
    Check if a path is safe to include in uploads.

    Args:
        path: The path to check (should be resolved/absolute)
        allowed_base: Optional base directory that paths must be within

    Returns:
        True if the path is safe, False otherwise
    """
    try:
        # Resolve the path to handle symlinks and ..
        resolved = path.resolve()
        resolved_str = str(resolved)

        # Check against forbidden system paths
        for forbidden in FORBIDDEN_PATHS:
            if resolved_str == forbidden or resolved_str.startswith(forbidden + "/"):
                logger.warning("Path %s is in forbidden area %s", path, forbidden)
                return False

        # If an allowed base is specified, ensure path is within it
        if allowed_base:
            allowed_resolved = allowed_base.resolve()
            try:
                resolved.relative_to(allowed_resolved)
            except ValueError:
                logger.warning("Path %s is outside allowed base %s", path, allowed_base)
                return False

        # Check for sensitive directory names in path components
        for part in resolved.parts:
            if part in SENSITIVE_PATTERNS:
                logger.warning("Path %s contains sensitive pattern %s", path, part)
                return False

        return True
    except (OSError, ValueError) as exc:
        logger.warning("Error checking path safety for %s: %s", path, exc)
        return False


def _is_safe_file(file_path: Path, base_path: Path) -> bool:
    """
    Check if a file within a dataset is safe to include.

    Args:
        file_path: The file path to check
        base_path: The base dataset directory

    Returns:
        True if the file is safe, False otherwise
    """
    try:
        resolved = file_path.resolve()

        # Verify file is actually within the base path (symlink protection)
        base_resolved = base_path.resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            logger.warning(
                "File %s resolves outside base directory %s (possible symlink attack)",
                file_path,
                base_path,
            )
            return False

        # Check file name for sensitive patterns
        if file_path.name.startswith("."):
            # Skip hidden files
            return False

        # Skip common sensitive file types
        sensitive_extensions = {".env", ".pem", ".key", ".crt", ".p12", ".pfx"}
        if file_path.suffix.lower() in sensitive_extensions:
            logger.warning("Skipping sensitive file type: %s", file_path)
            return False

        return True
    except (OSError, ValueError) as exc:
        logger.warning("Error checking file safety for %s: %s", file_path, exc)
        return False


class ReplicateUploadBackend(CloudUploadBackend):
    """Upload backend using Replicate's file upload API."""

    REPLICATE_API_BASE = "https://api.replicate.com/v1"

    def __init__(self):
        self._secrets = get_secrets_manager()
        self._http_client: Optional[httpx.AsyncClient] = None

    @property
    def _token(self) -> Optional[str]:
        """Get the API token from secrets manager."""
        return self._secrets.get_replicate_token()

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create an async HTTP client with extended timeout for uploads."""
        if self._http_client is None or self._http_client.is_closed:
            # Use a longer timeout for file uploads
            self._http_client = httpx.AsyncClient(timeout=600.0)
        return self._http_client

    async def upload_archive(
        self,
        local_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Upload an archive to Replicate and return the URL.

        Uses Replicate's file upload API which provides a temporary URL
        that can be used as input to predictions.
        """
        if not self._token:
            raise ValueError("REPLICATE_API_TOKEN not set")

        try:
            client = await self._get_http_client()

            file_path = Path(local_path)
            total_size = file_path.stat().st_size

            # Notify start
            if progress_callback:
                progress_callback(0, total_size)

            # Upload file using Replicate's files API
            with open(local_path, "rb") as f:
                file_content = f.read()

            # Determine content type based on file extension
            filename = file_path.name
            if filename.endswith(".zip"):
                content_type = "application/zip"
            elif filename.endswith(".tar.gz") or filename.endswith(".tgz"):
                content_type = "application/gzip"
            elif filename.endswith(".tar"):
                content_type = "application/x-tar"
            else:
                content_type = "application/octet-stream"

            # Upload as multipart form data
            files = {"file": (filename, file_content, content_type)}
            response = await client.post(
                f"{self.REPLICATE_API_BASE}/files",
                headers={"Authorization": f"Bearer {self._token}"},
                files=files,
            )
            response.raise_for_status()
            file_output = response.json()

            # Get the URL from the file output
            urls = file_output.get("urls", {})
            url = urls.get("get")
            if not url:
                # Fall back to direct URL if available
                url = file_output.get("url")

            if not url:
                raise ValueError("Failed to get URL from Replicate file upload")

            # Notify completion
            if progress_callback:
                progress_callback(total_size, total_size)

            return url

        except httpx.HTTPStatusError as exc:
            logger.error("Failed to upload to Replicate: HTTP %s: %s", exc.response.status_code, exc.response.text)
            raise ValueError(f"Replicate upload error: {exc.response.text}")
        except Exception as exc:
            logger.error("Failed to upload to Replicate: %s", exc)
            raise


class CloudUploadService:
    """
    Service for packaging local datasets and uploading to cloud providers.

    Handles:
    - Detecting local data backends in dataloader config
    - Creating zip archives of dataset directories
    - Uploading via pluggable backends
    - Progress tracking
    """

    def __init__(self, backend: Optional[CloudUploadBackend] = None):
        """
        Initialize the upload service.

        Args:
            backend: Upload backend to use. Defaults to ReplicateUploadBackend.
        """
        self._backend = backend or ReplicateUploadBackend()

    async def package_and_upload(
        self,
        dataloader_config: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        detailed_progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> Optional[str]:
        """
        Package local datasets from dataloader config and upload.

        Args:
            dataloader_config: List of dataset configuration dicts
            progress_callback: Optional callback(stage, current, total)
                             stage is "packaging" or "uploading"
            detailed_progress_callback: Optional callback(stage, current, total, message)
                             Provides human-readable status messages

        Returns:
            URL to uploaded archive, or None if no local data to upload
        """
        # Find local datasets
        local_paths = self._extract_local_paths(dataloader_config)

        if not local_paths:
            logger.info("No local datasets found in dataloader config")
            return None

        logger.info("Found %d local dataset path(s) to package", len(local_paths))

        # Create temporary archive with detailed progress
        archive_path = await self._create_archive(
            local_paths,
            progress_callback=lambda c, t: progress_callback("packaging", c, t) if progress_callback else None,
            detailed_callback=detailed_progress_callback,
        )

        try:
            # Get archive size for upload progress
            archive_size = Path(archive_path).stat().st_size
            archive_size_str = self._format_bytes(archive_size)

            if detailed_progress_callback:
                detailed_progress_callback("uploading", 0, archive_size, f"Uploading {archive_size_str}...")

            # Upload archive with progress
            def upload_progress(current: int, total: int) -> None:
                if progress_callback:
                    progress_callback("uploading", current, total)
                if detailed_progress_callback:
                    pct = int((current / total) * 100) if total > 0 else 0
                    detailed_progress_callback(
                        "uploading",
                        current,
                        total,
                        f"Uploading... {pct}% ({self._format_bytes(current)} / {archive_size_str})",
                    )

            url = await self._backend.upload_archive(
                archive_path,
                progress_callback=upload_progress,
            )

            if detailed_progress_callback:
                detailed_progress_callback("uploading", archive_size, archive_size, f"Upload complete ({archive_size_str})")

            return url
        finally:
            # Clean up temporary file
            try:
                os.unlink(archive_path)
            except OSError:
                pass

    def _extract_local_paths(self, dataloader_config: List[Dict[str, Any]]) -> List[Path]:
        """
        Extract local dataset paths from dataloader configuration.

        Performs security validation to prevent path traversal attacks.
        """
        paths = []

        for dataset in dataloader_config:
            # Check if this is a local backend
            backend_type = dataset.get("type", "local")
            if backend_type not in ("local", None, ""):
                continue

            # Get instance_data_dir
            data_dir = dataset.get("instance_data_dir")
            if not data_dir:
                continue

            # Expand user home directory
            data_path = Path(data_dir).expanduser()

            # Security: Validate the path is safe
            if not _is_safe_path(data_path):
                logger.error(
                    "SECURITY: Rejecting unsafe dataset path: %s (resolved: %s)",
                    data_dir,
                    data_path.resolve() if data_path.exists() else "N/A",
                )
                continue

            if data_path.exists() and data_path.is_dir():
                # Use resolved path to prevent symlink attacks
                paths.append(data_path.resolve())
            else:
                logger.warning("Dataset path does not exist: %s", data_dir)

        return paths

    async def _create_archive(
        self,
        paths: List[Path],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        detailed_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> str:
        """
        Create a zip archive of the given paths.

        Validates each file before adding to prevent symlink attacks.

        Args:
            paths: List of directories to archive
            progress_callback: Simple callback(current, total) for file count
            detailed_callback: Detailed callback(phase, current, total, message)
                              phase: "scanning", "packaging"
                              message: Human-readable status

        Returns path to the temporary archive file.
        """
        loop = asyncio.get_running_loop()

        # Phase 1: Scan files (show progress during scanning)
        if detailed_callback:
            detailed_callback("scanning", 0, 0, "Scanning files...")

        def _scan_files() -> tuple[list[tuple[Path, Path, int]], int]:
            """Scan and collect safe files with their sizes."""
            files_to_add: list[tuple[Path, Path, int]] = []  # (file_path, base_path, size)
            total_bytes = 0
            scanned = 0

            for base_path in paths:
                for file_path in base_path.rglob("*"):
                    if not file_path.is_file():
                        continue
                    scanned += 1
                    if detailed_callback and scanned % 100 == 0:
                        detailed_callback("scanning", scanned, 0, f"Scanned {scanned} files...")

                    if _is_safe_file(file_path, base_path):
                        try:
                            size = file_path.stat().st_size
                            files_to_add.append((file_path, base_path, size))
                            total_bytes += size
                        except OSError:
                            pass

            return files_to_add, total_bytes

        files_to_add, total_bytes = await loop.run_in_executor(None, _scan_files)

        if not files_to_add:
            raise ValueError("No safe files found to upload")

        total_files = len(files_to_add)
        logger.info("Found %d files (%s) to package", total_files, self._format_bytes(total_bytes))

        if progress_callback:
            progress_callback(0, total_files)
        if detailed_callback:
            detailed_callback(
                "packaging", 0, total_bytes, f"Packaging {total_files} files ({self._format_bytes(total_bytes)})..."
            )

        # Create archive in temp directory
        temp_fd, temp_path = tempfile.mkstemp(suffix=".zip", prefix="simpletuner_data_")
        os.close(temp_fd)

        # Phase 2: Create archive with byte-level progress
        def _create_zip():
            current_file = 0
            current_bytes = 0
            skipped_files = 0

            with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path, base_path, file_size in files_to_add:
                    try:
                        rel_path = file_path.resolve().relative_to(base_path.resolve())
                        archive_path = Path(base_path.name) / rel_path

                        zf.write(file_path.resolve(), archive_path)

                        current_file += 1
                        current_bytes += file_size

                        # Report progress for every file (not just every 10)
                        if progress_callback:
                            progress_callback(current_file, total_files)

                        if detailed_callback:
                            # Include current file name for visibility
                            file_name = file_path.name
                            if len(file_name) > 30:
                                file_name = file_name[:27] + "..."
                            pct = int((current_bytes / total_bytes) * 100) if total_bytes > 0 else 0
                            detailed_callback(
                                "packaging",
                                current_bytes,
                                total_bytes,
                                f"[{pct}%] {current_file}/{total_files}: {file_name}",
                            )
                    except ValueError as exc:
                        logger.warning(
                            "Skipping file outside base directory: %s (%s)",
                            file_path,
                            exc,
                        )
                        skipped_files += 1
                    except OSError as exc:
                        logger.warning("Error adding file %s: %s", file_path, exc)
                        skipped_files += 1

            if skipped_files > 0:
                logger.info(
                    "Archive created: %d files included, %d files skipped",
                    current_file,
                    skipped_files,
                )

            if progress_callback:
                progress_callback(total_files, total_files)
            if detailed_callback:
                detailed_callback("packaging", total_bytes, total_bytes, "Packaging complete")

        await loop.run_in_executor(None, _create_zip)

        # Log final archive size
        archive_size = Path(temp_path).stat().st_size
        logger.info(
            "Archive created: %s (compressed from %s)", self._format_bytes(archive_size), self._format_bytes(total_bytes)
        )

        return temp_path

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ["B", "KB", "MB", "GB"]:
            if num_bytes < 1024:
                return f"{num_bytes:.1f} {unit}"
            num_bytes /= 1024
        return f"{num_bytes:.1f} TB"

    def estimate_upload_size(self, dataloader_config: List[Dict[str, Any]]) -> int:
        """
        Estimate the upload size in bytes.

        Args:
            dataloader_config: List of dataset configuration dicts

        Returns:
            Estimated size in bytes (uncompressed)
        """
        local_paths = self._extract_local_paths(dataloader_config)
        total_size = 0

        for path in local_paths:
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    try:
                        total_size += file_path.stat().st_size
                    except OSError:
                        pass

        return total_size

    def has_local_data(self, dataloader_config: List[Dict[str, Any]]) -> bool:
        """Check if dataloader config contains local data that needs uploading."""
        return len(self._extract_local_paths(dataloader_config)) > 0
