"""Service for handling dataset file uploads, folder creation, and zip extraction."""

from __future__ import annotations

import io
import logging
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Import image extensions from the training helpers
try:
    from simpletuner.helpers.training import image_file_extensions

    ALLOWED_IMAGE_EXTENSIONS: Set[str] = set(image_file_extensions)
except ImportError:
    # Fallback if import fails
    ALLOWED_IMAGE_EXTENSIONS = {
        "png",
        "jpg",
        "jpeg",
        "webp",
        "jxl",
        "bmp",
        "gif",
        "tiff",
        "tif",
    }

ALLOWED_METADATA_EXTENSIONS: Set[str] = {"txt", "parquet", "jsonl", "csv", "json"}
ALLOWED_ALL_EXTENSIONS: Set[str] = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_METADATA_EXTENSIONS

# Security limits
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500MB per file
MAX_ZIP_EXTRACTED_SIZE = 10 * 1024 * 1024 * 1024  # 10GB total extracted
MAX_ZIP_FILE_COUNT = 100_000  # Max files in a zip


class UploadResult(BaseModel):
    """Result of an upload operation."""

    success: bool
    files_uploaded: int = 0
    files_skipped: int = 0
    errors: List[str] = Field(default_factory=list)
    target_path: str = ""


class FolderCreateResult(BaseModel):
    """Result of a folder creation operation."""

    success: bool
    path: str = ""
    name: str = ""
    error: Optional[str] = None


class DatasetUploadService:
    """Handles file uploads, zip extraction, and folder creation for datasets."""

    # Characters forbidden in folder names for security
    FORBIDDEN_FOLDER_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
    # Pattern to detect path traversal attempts
    PATH_TRAVERSAL_PATTERN = re.compile(r"(^|[\\/])\.\.($|[\\/])")

    def __init__(self, datasets_dir: Optional[Path] = None, allow_outside: bool = False):
        """Initialize the upload service.

        Args:
            datasets_dir: Root directory for datasets. If None, paths are not restricted.
            allow_outside: If True, allow operations outside datasets_dir.
        """
        self.datasets_dir = datasets_dir.resolve() if datasets_dir else None
        self.allow_outside = allow_outside

    def _validate_path(self, path: Path) -> None:
        """Validate that a path is within the allowed directory.

        Raises:
            ValueError: If path is outside allowed boundaries.
        """
        if self.allow_outside or self.datasets_dir is None:
            return

        resolved = path.resolve()
        try:
            resolved.relative_to(self.datasets_dir)
        except ValueError:
            raise ValueError(f"Path '{path}' is outside the allowed datasets directory")

    def _sanitize_folder_name(self, name: str) -> str:
        """Sanitize a folder name to prevent security issues.

        Args:
            name: The raw folder name.

        Returns:
            Sanitized folder name.

        Raises:
            ValueError: If the name is invalid or contains forbidden characters.
        """
        if not name or not name.strip():
            raise ValueError("Folder name cannot be empty")

        name = name.strip()

        # Check for path traversal
        if self.PATH_TRAVERSAL_PATTERN.search(name):
            raise ValueError("Folder name contains path traversal sequences")

        # Check for forbidden characters
        if self.FORBIDDEN_FOLDER_CHARS.search(name):
            raise ValueError("Folder name contains forbidden characters")

        # Reject names that are just dots
        if name in (".", ".."):
            raise ValueError("Invalid folder name")

        # Reject names starting with dot (hidden files)
        if name.startswith("."):
            raise ValueError("Folder names cannot start with a dot")

        # Check for path separators
        if "/" in name or "\\" in name:
            raise ValueError("Folder name cannot contain path separators")

        # Limit length
        if len(name) > 255:
            raise ValueError("Folder name is too long (max 255 characters)")

        return name

    def _is_allowed_extension(self, filename: str) -> bool:
        """Check if a file has an allowed extension.

        Args:
            filename: The filename to check.

        Returns:
            True if the extension is allowed, False otherwise.
        """
        ext = Path(filename).suffix.lower().lstrip(".")
        return ext in ALLOWED_ALL_EXTENSIONS

    def _is_image_file(self, filename: str) -> bool:
        """Check if a file is an image based on extension.

        Args:
            filename: The filename to check.

        Returns:
            True if the file is an image, False otherwise.
        """
        ext = Path(filename).suffix.lower().lstrip(".")
        return ext in ALLOWED_IMAGE_EXTENSIONS

    def create_folder(self, parent_path: Path, folder_name: str) -> FolderCreateResult:
        """Create a new subfolder in the given parent directory.

        Args:
            parent_path: The parent directory where the folder will be created.
            folder_name: The name for the new folder.

        Returns:
            FolderCreateResult with the operation status.
        """
        try:
            # Validate parent path is within bounds
            self._validate_path(parent_path)

            # Sanitize the folder name
            clean_name = self._sanitize_folder_name(folder_name)

            # Create the full path
            new_folder = parent_path / clean_name

            # Validate the new path is still within bounds
            self._validate_path(new_folder)

            # Check if already exists
            if new_folder.exists():
                return FolderCreateResult(success=False, error=f"Folder '{clean_name}' already exists")

            # Create the folder
            new_folder.mkdir(parents=False, exist_ok=False)

            logger.info(f"Created folder: {new_folder}")
            return FolderCreateResult(success=True, path=str(new_folder), name=clean_name)

        except ValueError as e:
            return FolderCreateResult(success=False, error=str(e))
        except PermissionError:
            return FolderCreateResult(success=False, error="Permission denied creating folder")
        except Exception as e:
            logger.exception(f"Error creating folder: {e}")
            return FolderCreateResult(success=False, error=f"Failed to create folder: {e}")

    async def handle_file_uploads(
        self,
        files: List[Any],  # FastAPI UploadFile objects
        target_dir: Path,
    ) -> UploadResult:
        """Save uploaded files to the target directory, validating extensions.

        Args:
            files: List of UploadFile objects from FastAPI.
            target_dir: Directory where files should be saved.

        Returns:
            UploadResult with counts and any errors.
        """
        result = UploadResult(success=True, target_path=str(target_dir))

        try:
            # Validate target directory
            self._validate_path(target_dir)

            if not target_dir.exists():
                return UploadResult(
                    success=False,
                    errors=[f"Target directory does not exist: {target_dir}"],
                    target_path=str(target_dir),
                )

            for upload_file in files:
                filename = upload_file.filename
                if not filename:
                    result.files_skipped += 1
                    result.errors.append("File with no name skipped")
                    continue

                # Security: sanitize filename
                safe_filename = Path(filename).name
                if safe_filename != filename or ".." in filename:
                    result.files_skipped += 1
                    result.errors.append(f"Invalid filename: {filename}")
                    continue

                # Check extension
                if not self._is_allowed_extension(safe_filename):
                    result.files_skipped += 1
                    result.errors.append(f"Unsupported file type: {safe_filename}")
                    continue

                target_file = target_dir / safe_filename

                # Validate target path
                try:
                    self._validate_path(target_file)
                except ValueError:
                    result.files_skipped += 1
                    result.errors.append(f"Invalid target path for: {safe_filename}")
                    continue

                # Check file size by reading in chunks
                try:
                    content = await upload_file.read()
                    if len(content) > MAX_FILE_SIZE_BYTES:
                        result.files_skipped += 1
                        result.errors.append(f"File too large (max {MAX_FILE_SIZE_BYTES // (1024*1024)}MB): {safe_filename}")
                        continue

                    # Write file
                    target_file.write_bytes(content)
                    result.files_uploaded += 1
                    logger.info(f"Uploaded file: {target_file}")

                except Exception as e:
                    result.files_skipped += 1
                    result.errors.append(f"Failed to save {safe_filename}: {e}")

        except ValueError as e:
            result.success = False
            result.errors.append(str(e))
        except Exception as e:
            result.success = False
            result.errors.append(f"Upload failed: {e}")
            logger.exception(f"Upload error: {e}")

        # Set overall success based on whether any files were uploaded
        if result.files_uploaded == 0 and len(files) > 0:
            result.success = False

        return result

    async def handle_zip_upload(
        self,
        upload_file: Any,  # FastAPI UploadFile
        target_dir: Path,
    ) -> UploadResult:
        """Extract a zip file to the target directory.

        Args:
            upload_file: UploadFile object containing the zip.
            target_dir: Directory where contents should be extracted.

        Returns:
            UploadResult with counts and any errors.
        """
        result = UploadResult(success=True, target_path=str(target_dir))

        try:
            # Validate target directory
            self._validate_path(target_dir)

            if not target_dir.exists():
                return UploadResult(
                    success=False,
                    errors=[f"Target directory does not exist: {target_dir}"],
                    target_path=str(target_dir),
                )

            # Read zip content
            content = await upload_file.read()

            # Extract to temp directory first, then move (for atomicity)
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                try:
                    with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
                        # Security checks
                        total_size = 0
                        file_count = 0

                        for info in zf.infolist():
                            # Check for path traversal in zip
                            if info.filename.startswith("/") or ".." in info.filename:
                                result.errors.append(f"Skipping suspicious path: {info.filename}")
                                result.files_skipped += 1
                                continue

                            # Skip directories
                            if info.is_dir():
                                continue

                            file_count += 1
                            total_size += info.file_size

                            # Check limits
                            if file_count > MAX_ZIP_FILE_COUNT:
                                raise ValueError(f"Too many files in archive (max {MAX_ZIP_FILE_COUNT})")
                            if total_size > MAX_ZIP_EXTRACTED_SIZE:
                                raise ValueError(
                                    f"Extracted size too large (max {MAX_ZIP_EXTRACTED_SIZE // (1024*1024*1024)}GB)"
                                )

                        # Extract files one by one with validation
                        for info in zf.infolist():
                            if info.is_dir():
                                continue

                            if info.filename.startswith("/") or ".." in info.filename:
                                continue

                            # Preserve the relative path from the zip
                            relative_path = Path(info.filename)
                            filename = relative_path.name
                            if not filename:
                                continue

                            # Check extension
                            if not self._is_allowed_extension(filename):
                                result.files_skipped += 1
                                result.errors.append(f"Unsupported file type in zip: {filename}")
                                continue

                            # Determine the final path preserving directory structure
                            final_path = target_dir / relative_path

                            # Validate the final path is still within bounds
                            try:
                                self._validate_path(final_path)
                            except ValueError:
                                result.files_skipped += 1
                                result.errors.append(f"Invalid path in zip: {info.filename}")
                                continue

                            # Create parent directories if needed
                            final_path.parent.mkdir(parents=True, exist_ok=True)

                            # Extract to temp dir first
                            extracted_path = temp_path / filename

                            with zf.open(info) as src, open(extracted_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)

                            # Handle duplicate names
                            if final_path.exists():
                                stem = final_path.stem
                                suffix = final_path.suffix
                                counter = 1
                                while final_path.exists():
                                    final_path = final_path.parent / f"{stem}_{counter}{suffix}"
                                    counter += 1

                            shutil.move(str(extracted_path), str(final_path))
                            result.files_uploaded += 1

                except zipfile.BadZipFile:
                    result.success = False
                    result.errors.append("Invalid or corrupted zip file")

        except ValueError as e:
            result.success = False
            result.errors.append(str(e))
        except Exception as e:
            result.success = False
            result.errors.append(f"Zip extraction failed: {e}")
            logger.exception(f"Zip extraction error: {e}")

        if result.files_uploaded == 0:
            result.success = False

        return result

    def get_allowed_extensions_for_accept(self) -> str:
        """Get a comma-separated string of allowed extensions for HTML accept attribute.

        Returns:
            String like ".png,.jpg,.jpeg,.webp,.txt,.parquet,.jsonl,.csv"
        """
        return ",".join(f".{ext}" for ext in sorted(ALLOWED_ALL_EXTENSIONS))


# Convenience function to get a service instance with common setup
def get_upload_service(datasets_dir: Optional[Path] = None, allow_outside: bool = False) -> DatasetUploadService:
    """Get a DatasetUploadService instance.

    Args:
        datasets_dir: Root directory for datasets.
        allow_outside: If True, allow operations outside datasets_dir.

    Returns:
        Configured DatasetUploadService instance.
    """
    return DatasetUploadService(datasets_dir=datasets_dir, allow_outside=allow_outside)
