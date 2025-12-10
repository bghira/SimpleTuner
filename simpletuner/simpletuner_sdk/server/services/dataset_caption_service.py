"""Service for managing caption file detection and creation for datasets."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Import image extensions from the training helpers
try:
    from simpletuner.helpers.training import image_file_extensions

    ALLOWED_IMAGE_EXTENSIONS: Set[str] = set(image_file_extensions)
except ImportError:
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


class ImageInfo(BaseModel):
    """Information about an image file."""

    path: str
    filename: str
    has_caption: bool = False


class CaptionStatus(BaseModel):
    """Status of captions in a directory."""

    total_images: int = 0
    with_caption: int = 0
    without_caption: int = 0
    coverage_ratio: float = 0.0
    images_without_captions: List[ImageInfo] = Field(default_factory=list)


class ThumbnailInfo(BaseModel):
    """Thumbnail data for an image."""

    path: str
    filename: str
    thumbnail: str  # base64 data URL


class CaptionWriteResult(BaseModel):
    """Result of writing caption files."""

    success: bool
    files_written: int = 0
    errors: List[str] = Field(default_factory=list)


class DatasetCaptionService:
    """Manages caption file detection and creation for datasets."""

    DEFAULT_THUMBNAIL_SIZE = 256
    MAX_THUMBNAIL_SIZE = 512

    def __init__(self, datasets_dir: Optional[Path] = None, allow_outside: bool = False):
        """Initialize the caption service.

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

    def _is_image_file(self, path: Path) -> bool:
        """Check if a path points to an image file.

        Args:
            path: Path to check.

        Returns:
            True if it's an image file, False otherwise.
        """
        if not path.is_file():
            return False
        ext = path.suffix.lower().lstrip(".")
        return ext in ALLOWED_IMAGE_EXTENSIONS

    def _get_caption_path(self, image_path: Path) -> Path:
        """Get the expected caption file path for an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Path to the corresponding .txt caption file.
        """
        return image_path.with_suffix(".txt")

    def _has_caption(self, image_path: Path) -> bool:
        """Check if an image has a corresponding caption file.

        Args:
            image_path: Path to the image file.

        Returns:
            True if a .txt caption file exists, False otherwise.
        """
        caption_path = self._get_caption_path(image_path)
        return caption_path.exists() and caption_path.is_file()

    def get_caption_status(self, path: Path) -> CaptionStatus:
        """Analyze caption coverage in a directory.

        Args:
            path: Directory to analyze.

        Returns:
            CaptionStatus with coverage information.
        """
        try:
            self._validate_path(path)
        except ValueError as e:
            logger.warning(f"Path validation failed: {e}")
            return CaptionStatus()

        if not path.exists() or not path.is_dir():
            logger.warning(f"Path is not a valid directory: {path}")
            return CaptionStatus()

        images_without = []
        total = 0
        with_caption = 0

        try:
            for item in path.iterdir():
                if not self._is_image_file(item):
                    continue

                total += 1
                if self._has_caption(item):
                    with_caption += 1
                else:
                    images_without.append(
                        ImageInfo(
                            path=str(item),
                            filename=item.name,
                            has_caption=False,
                        )
                    )

        except PermissionError as e:
            logger.warning(f"Permission error scanning directory: {e}")
            return CaptionStatus()
        except Exception as e:
            logger.exception(f"Error scanning directory: {e}")
            return CaptionStatus()

        coverage = with_caption / total if total > 0 else 0.0

        return CaptionStatus(
            total_images=total,
            with_caption=with_caption,
            without_caption=len(images_without),
            coverage_ratio=coverage,
            images_without_captions=images_without,
        )

    def get_thumbnails(
        self,
        image_paths: List[Path],
        max_size: int = DEFAULT_THUMBNAIL_SIZE,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ThumbnailInfo]:
        """Generate base64 thumbnails for the given image paths.

        Args:
            image_paths: List of paths to image files.
            max_size: Maximum dimension for thumbnails (width or height).
            limit: Maximum number of thumbnails to return.
            offset: Number of images to skip (for pagination).

        Returns:
            List of ThumbnailInfo with base64 encoded image data.
        """
        # Clamp max_size
        max_size = min(max(max_size, 64), self.MAX_THUMBNAIL_SIZE)

        thumbnails = []
        # Apply pagination
        paths_slice = image_paths[offset : offset + limit]

        for path in paths_slice:
            if isinstance(path, str):
                path = Path(path)

            try:
                self._validate_path(path)
            except ValueError:
                continue

            if not path.exists() or not self._is_image_file(path):
                continue

            try:
                thumbnail_data = self._generate_thumbnail(path, max_size)
                if thumbnail_data:
                    thumbnails.append(
                        ThumbnailInfo(
                            path=str(path),
                            filename=path.name,
                            thumbnail=thumbnail_data,
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to generate thumbnail for {path}: {e}")
                continue

        return thumbnails

    def _generate_thumbnail(self, image_path: Path, max_size: int) -> Optional[str]:
        """Generate a base64 encoded thumbnail for an image.

        Args:
            image_path: Path to the image file.
            max_size: Maximum dimension (width or height).

        Returns:
            Base64 data URL string or None on failure.
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (handles RGBA, P mode, etc.)
                if img.mode in ("RGBA", "P", "LA"):
                    # Create white background for transparency
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    if img.mode in ("RGBA", "LA"):
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    else:
                        img = img.convert("RGB")
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # Calculate thumbnail size maintaining aspect ratio
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # Save to bytes
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                buffer.seek(0)

                # Encode as base64 data URL
                b64 = base64.b64encode(buffer.read()).decode("utf-8")
                return f"data:image/jpeg;base64,{b64}"

        except Exception as e:
            logger.warning(f"Error generating thumbnail for {image_path}: {e}")
            return None

    def create_captions(
        self,
        captions: Dict[str, str],
    ) -> CaptionWriteResult:
        """Write .txt caption files for the given images.

        Args:
            captions: Dictionary mapping image paths to caption text.

        Returns:
            CaptionWriteResult with count of files written.
        """
        result = CaptionWriteResult(success=True)

        for image_path_str, caption_text in captions.items():
            image_path = Path(image_path_str)

            try:
                self._validate_path(image_path)
            except ValueError as e:
                result.errors.append(f"Invalid path {image_path}: {e}")
                continue

            # Verify the image exists
            if not image_path.exists():
                result.errors.append(f"Image not found: {image_path}")
                continue

            if not self._is_image_file(image_path):
                result.errors.append(f"Not an image file: {image_path}")
                continue

            # Write the caption file
            caption_path = self._get_caption_path(image_path)

            try:
                self._validate_path(caption_path)
            except ValueError as e:
                result.errors.append(f"Invalid caption path: {e}")
                continue

            try:
                # Strip and normalize caption text
                clean_caption = caption_text.strip() if caption_text else ""
                caption_path.write_text(clean_caption, encoding="utf-8")
                result.files_written += 1
                logger.info(f"Wrote caption file: {caption_path}")

            except PermissionError:
                result.errors.append(f"Permission denied writing: {caption_path}")
            except Exception as e:
                result.errors.append(f"Failed to write {caption_path}: {e}")

        if result.files_written == 0 and len(captions) > 0:
            result.success = False

        return result


# Convenience function to get a service instance
def get_caption_service(datasets_dir: Optional[Path] = None, allow_outside: bool = False) -> DatasetCaptionService:
    """Get a DatasetCaptionService instance.

    Args:
        datasets_dir: Root directory for datasets.
        allow_outside: If True, allow operations outside datasets_dir.

    Returns:
        Configured DatasetCaptionService instance.
    """
    return DatasetCaptionService(datasets_dir=datasets_dir, allow_outside=allow_outside)
