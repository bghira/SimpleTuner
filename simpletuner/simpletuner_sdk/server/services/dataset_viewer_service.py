"""Service for browsing dataset contents using existing cache files."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import subprocess
from contextlib import ExitStack, closing, contextmanager
from glob import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import ijson
import numpy as np
import trainingsample as tsr
from PIL import Image
from pydantic import BaseModel, Field

from simpletuner.helpers.metadata.backends.webshart_exclusions import read_exclusion_page
from simpletuner.helpers.metadata.backends.webshart_geometry import (
    _read_index,
    accepted_ref_lookup,
    geometry_lookup,
    prepared_geometry_path,
    read_snapshot,
)
from simpletuner.helpers.metadata.backends.webshart_membership import IncompleteWebshartMembership, MembershipPages
from simpletuner.helpers.metadata.preparation import read_bucket_groups, read_bucket_page

logger = logging.getLogger(__name__)

# Lazy-loaded at first use; patchable in tests
GENERATOR_REGISTRY: Optional[Dict] = None
GENERATOR_REGISTRY_AVAILABLE = False


class BucketSummary(BaseModel):
    """Summary of a single aspect ratio bucket."""

    key: str
    file_count: int


class DatasetSummary(BaseModel):
    """Summary of a dataset's cached state."""

    dataset_id: str
    has_cache: bool = False
    scan_incomplete: bool = False
    accepted_samples: Optional[int] = None
    physical_rows: Optional[int] = None
    source_captions_required: Optional[bool] = None
    source_revision: Optional[str] = None
    view_revision: Optional[str] = None
    total_files: int = 0
    bucket_count: int = 0
    buckets: List[BucketSummary] = Field(default_factory=list)
    filtering_statistics: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    cache_file_path: Optional[str] = None
    # Conditioning relationship fields (from plan config, not cache)
    conditioning_data: Optional[List[str]] = None
    source_dataset_id: Optional[str] = None
    conditioning_type: Optional[str] = None
    conditioning_config: Optional[Dict[str, Any]] = None
    conditioning_generators: Optional[List[Dict[str, Any]]] = None
    auto_generated: bool = False


class DatasetViewChanged(ValueError):
    """The indexed dataset was replaced between page requests."""


class DatasetFilePage(BaseModel):
    """Paginated file listing from a dataset."""

    files: List[str]
    total: int
    limit: int
    offset: int
    bucket_key: Optional[str] = None
    view_revision: Optional[str] = None


class ViewerThumbnailInfo(BaseModel):
    """Thumbnail data for a dataset file."""

    path: str
    filename: str
    thumbnail: Optional[str] = None
    error: Optional[str] = None


class FileMetadata(BaseModel):
    """Per-file metadata from the metadata cache."""

    path: str
    display_name: Optional[str] = None
    original_size: Optional[List[int]] = None
    target_size: Optional[List[int]] = None
    intermediary_size: Optional[List[int]] = None
    aspect_ratio: Optional[float] = None
    crop_coordinates: Optional[List[int]] = None
    bucket_key: Optional[str] = None
    caption: Optional[str] = None
    view_revision: Optional[str] = None
    bbox_entities: Optional[List[Dict[str, Any]]] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


# Stage 3: Caption validation models


class CaptionValidationResult(BaseModel):
    """Result of caption validation for a dataset."""

    dataset_id: str
    strategy: str
    coverage_ratio: Optional[float] = 0.0
    total_files: int = 0
    captioned_files: Optional[int] = 0
    uncaptioned_files: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# Stage 4: Size filter models


class FilteredFileEntry(BaseModel):
    """A file that was filtered out during bucketing."""

    path: str
    reason: str
    original_size: Optional[List[int]] = None
    aspect_ratio: Optional[float] = None
    threshold: Optional[str] = None


class FilteredFilesReport(BaseModel):
    """Report of files filtered by size/aspect ratio settings."""

    dataset_id: str
    total_filtered: int = 0
    scope: str = "size_and_aspect"
    by_reason: Dict[str, int] = Field(default_factory=dict)
    files: List[FilteredFileEntry] = Field(default_factory=list)


# Stage 5: Conditioning models


class DatasetNode(BaseModel):
    """A node in the dataset dependency graph."""

    id: str
    dataset_type: str
    backend_type: str


class DatasetEdge(BaseModel):
    """An edge in the dataset dependency graph."""

    source_id: str
    target_id: str
    relationship: str


class DatasetGraph(BaseModel):
    """Dependency graph of datasets."""

    nodes: List[DatasetNode] = Field(default_factory=list)
    edges: List[DatasetEdge] = Field(default_factory=list)


class ConditioningPair(BaseModel):
    """A matched pair of source and conditioning files."""

    source_path: str
    conditioning_path: str


class OrphanReport(BaseModel):
    """Report of orphaned files between source and conditioning datasets."""

    source_id: str
    conditioning_id: str
    source_orphans: List[str] = Field(default_factory=list)
    conditioning_orphans: List[str] = Field(default_factory=list)
    scope: str = "catalogue"
    offset: int = 0
    limit: int = 500


# Stage 6: Audio/Video preview models


class VideoPreview(BaseModel):
    """Video preview with extracted frames."""

    path: str
    frames: List[str] = Field(default_factory=list)
    duration: Optional[float] = None
    fps: Optional[float] = None
    resolution: Optional[List[int]] = None


class AudioPreview(BaseModel):
    """Audio preview with waveform image."""

    path: str
    waveform: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None


def _pil_to_thumbnail(img: Image.Image, max_size: int = 256) -> str:
    """Convert a PIL Image to a base64-encoded JPEG thumbnail data URL."""
    if img.mode in ("RGBA", "P", "LA"):
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

    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# Extensions that require video frame extraction instead of Image.open()
_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".3gp",
    ".ogv",
}


def _extract_first_frame(video_path: Path) -> Optional[np.ndarray]:
    """Extract the first frame from a video file as an RGB numpy array."""
    cap = tsr.PyVideoCapture(str(video_path))
    if not cap.is_opened():
        return None
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return tsr.cvt_color_py(frame, 4)  # BGR→RGB
    finally:
        cap.release()


def generate_thumbnail(image_path: Path, max_size: int = 256) -> Optional[str]:
    """Generate a base64 encoded JPEG thumbnail from a file path.

    Handles both image files (via PIL) and video files (extracts first frame
    via trainingsample's VideoCapture).
    """
    try:
        if image_path.suffix.lower() in _VIDEO_EXTENSIONS:
            frame = _extract_first_frame(image_path)
            if frame is None:
                return None
            return _array_to_thumbnail(frame, max_size)
        with Image.open(image_path) as img:
            return _pil_to_thumbnail(img, max_size)
    except Exception as e:
        logger.warning("Error generating thumbnail for %s: %s", image_path, e)
        return None


def generate_thumbnail_from_pil(img: Image.Image, max_size: int = 256) -> Optional[str]:
    """Generate a base64 encoded JPEG thumbnail from a PIL Image."""
    try:
        return _pil_to_thumbnail(img, max_size)
    except Exception as e:
        logger.warning("Error generating thumbnail from PIL image: %s", e)
        return None


def _array_to_thumbnail(array: np.ndarray, max_size: int = 256) -> str:
    """Convert an RGB numpy array to a base64-encoded JPEG thumbnail data URL.

    Creates a fresh PIL Image from the array so thumbnail() cannot
    mutate any caller-owned data.
    """
    img = Image.fromarray(array)
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"


# Module-level cache for data backends used by on-demand thumbnail generation.
# Keyed by dataset ID. Backends are expensive to create (HuggingFace loads the
# full dataset on first access) but subsequent uses are fast.
_data_backend_cache: Dict[str, Any] = {}


def _webshart_source_config(config):
    native = config.get("webshart") or {}
    return (
        *(config.get(key, native.get(key)) for key in ("source", "subfolder", "metadata")),
        *(native.get(key) for key in ("manifest", "caption_key", "columns")),
        config.get("aws_endpoint_url"),
    )


class DatasetViewerService:
    """Reads existing cache files to provide dataset browsing capabilities."""

    THUMBNAIL_CACHE_DIR = "cache/viewer_thumbnails"
    DEFAULT_THUMBNAIL_SIZE = 256
    MAX_THUMBNAIL_SIZE = 512

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()

    def _find_cache_file(self, dataset_config: Dict[str, Any], file_type: str = "indices") -> Optional[Path]:
        """Locate an aspect ratio bucket cache file for a dataset.

        Searches instance_data_dir, cache directories, and HuggingFace
        metadata paths (cache/huggingface/{id}/huggingface_metadata/{id}/).
        """
        dataset_id = dataset_config.get("id", "")
        instance_data_dir = dataset_config.get("instance_data_dir", "")

        if not dataset_id:
            return None

        if dataset_config.get("type") == "webshart":
            native = dataset_config.get("webshart") or {}
            suffix = dataset_config.get("cache_file_suffix", dataset_id)
            cache_dir = native.get("cache_dir") or dataset_config.get("cache_dir")
            root = Path(cache_dir) if cache_dir else self.project_root / "cache" / "webshart" / dataset_id
            stem = f"aspect_ratio_bucket_{file_type}"
            if root.is_dir() and native.get("format", "tar") == "parquet":
                candidates = root.glob(f"{stem}_source_*_{escape(suffix)}.json")
                matches, incomplete = [], []
                configured_source = _webshart_source_config(dataset_config)
                for candidate in candidates:
                    indices = candidate.with_name(candidate.name.replace(stem, "aspect_ratio_bucket_indices", 1))
                    try:
                        with indices.open("rb") as source:
                            modified = os.fstat(source.fileno()).st_mtime_ns
                            try:
                                index = MembershipPages.open(source)
                            except IncompleteWebshartMembership as scan:
                                if configured_source == _webshart_source_config(scan.header.get("config") or {}):
                                    incomplete.append((modified, candidate))
                            else:
                                if index and configured_source == _webshart_source_config(index.header.get("config") or {}):
                                    matches.append((modified, candidate))
                    except (OSError, json.JSONDecodeError, KeyError) as error:
                        logger.debug("Cannot open Webshart publication for %s: %s", dataset_id, error)
                available = matches or incomplete
                return max(available)[1] if available else None
            if cache_dir:
                instance_data_dir = str(root)
        filename = f"aspect_ratio_bucket_{file_type}_{dataset_config.get('cache_file_suffix', dataset_id)}.json"

        # Search paths in order of likelihood
        search_dirs: list[Path] = []

        if instance_data_dir:
            search_dirs.append(Path(instance_data_dir))

        # Also check cache_dir if specified
        cache_dir = dataset_config.get("cache_dir") or dataset_config.get("cache_dir_vae")
        if cache_dir:
            cache_path = Path(cache_dir)
            search_dirs.append(cache_path)

        # HuggingFace backends store metadata at:
        #   {cache_base}/huggingface/{id}/huggingface_metadata/{id}/
        # where cache_base comes from args.cache_dir, env, or defaults to "cache"
        hf_metadata_dir = Path("cache") / "huggingface" / dataset_id / "huggingface_metadata" / dataset_id
        search_dirs.append(hf_metadata_dir)

        for search_dir in search_dirs:
            candidate = search_dir / filename
            if candidate.is_file():
                return candidate

        # Fallback: glob under instance_data_dir and cache/
        fallback_dirs = [self.project_root / "cache"]
        if instance_data_dir:
            fallback_dirs.insert(0, Path(instance_data_dir))
        for base_dir in fallback_dirs:
            if base_dir.is_dir():
                matches = list(base_dir.rglob(filename))
                if matches:
                    return matches[0]

        return None

    def _load_cache(self, cache_path: Path) -> Dict[str, Any]:
        """Load and parse a cache JSON file."""
        with open(cache_path, "r") as f:
            return json.load(f)

    def _coerce_display_name(self, value: Any) -> Optional[str]:
        """Normalise a candidate dataset label to a readable display name."""
        if value is None:
            return None

        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except Exception:
                return None

        if isinstance(value, Path):
            value = str(value)

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)

        if not isinstance(value, str):
            return None

        candidate = value.strip()
        if not candidate:
            return None

        parsed = urlparse(candidate)
        if parsed.scheme and parsed.path:
            basename = Path(parsed.path).name
            return basename or candidate

        basename = Path(candidate).name
        return basename or candidate

    def _extract_display_name_from_item(
        self,
        dataset_config: Dict[str, Any],
        item: Dict[str, Any],
    ) -> Optional[str]:
        """Extract a human-readable label from a Hugging Face dataset row."""
        if not isinstance(item, dict):
            return None

        candidate_keys = [
            "file_name",
            "filename",
            "image_filename",
            "image_path",
            "path",
            "url",
            "uri",
            "name",
            "id",
        ]

        for key in candidate_keys:
            display_name = self._coerce_display_name(item.get(key))
            if display_name:
                return display_name

        media_columns = [
            dataset_config.get("image_column"),
            dataset_config.get("video_column"),
            dataset_config.get("audio_column"),
            "image",
            "video",
            "audio",
        ]
        for column in media_columns:
            if not column or column not in item:
                continue
            sample = item.get(column)
            if isinstance(sample, dict):
                for nested_key in ("path", "filename", "file_name", "url", "uri", "name"):
                    display_name = self._coerce_display_name(sample.get(nested_key))
                    if display_name:
                        return display_name
            else:
                for attr in ("path", "filename", "name", "url"):
                    display_name = self._coerce_display_name(getattr(sample, attr, None))
                    if display_name:
                        return display_name

        return None

    def _get_display_name(self, dataset_config: Dict[str, Any], file_path: str) -> str:
        """Return the label the viewer should show for a cached file entry."""
        default_name = Path(file_path).name
        if dataset_config.get("type") != "huggingface":
            return default_name

        try:
            backend = self._get_or_create_data_backend(dataset_config)
            index = backend._get_index_from_path(file_path)
            if index is None:
                return default_name
            item = backend.get_dataset_item(index)
            display_name = self._extract_display_name_from_item(dataset_config, item)
            if display_name:
                return display_name
        except Exception as exc:
            logger.debug("Falling back to virtual Hugging Face path for %s: %s", file_path, exc)

        return default_name

    def get_dataset_summary(self, dataset_config: Dict[str, Any]) -> DatasetSummary:
        """Read bucket indices cache and return a summary."""
        dataset_id = dataset_config.get("id", "")

        # Extract conditioning relationship fields from the plan config
        cond_data_raw = dataset_config.get("conditioning_data")
        if isinstance(cond_data_raw, str):
            cond_data = [cond_data_raw]
        elif isinstance(cond_data_raw, list) and cond_data_raw:
            cond_data = cond_data_raw
        else:
            cond_data = None

        cond_generators_raw = dataset_config.get("conditioning")
        cond_generators = cond_generators_raw if isinstance(cond_generators_raw, list) and cond_generators_raw else None

        cond_fields = dict(
            conditioning_data=cond_data,
            source_dataset_id=dataset_config.get("source_dataset_id"),
            conditioning_type=dataset_config.get("conditioning_type"),
            conditioning_config=dataset_config.get("conditioning_config"),
            conditioning_generators=cond_generators,
            auto_generated=bool(dataset_config.get("auto_generated", False)),
        )

        cache_path = self._find_cache_file(dataset_config, "indices")

        if cache_path is None:
            return DatasetSummary(dataset_id=dataset_id, has_cache=False, **cond_fields)

        try:
            with cache_path.open("rb") as source:
                index = MembershipPages.open(source)
            if index is None:
                data = self._load_cache(cache_path)
                if data.get("scan_in_progress") and dataset_config.get("type") == "webshart":
                    raise IncompleteWebshartMembership(data)
                counts = {key: len(files) for key, files in data.get("aspect_ratio_bucket_indices", {}).items()}
            else:
                data, counts = index.header, index.counts
        except IncompleteWebshartMembership:
            return DatasetSummary(dataset_id=dataset_id, scan_incomplete=True, **cond_fields)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read cache for %s: %s", dataset_id, e)
            return DatasetSummary(dataset_id=dataset_id, has_cache=False, **cond_fields)

        buckets = [BucketSummary(key=key, file_count=count) for key, count in counts.items()]
        total_files = sum(b.file_count for b in buckets)

        return DatasetSummary(
            dataset_id=dataset_id,
            has_cache=True,
            total_files=total_files,
            accepted_samples=total_files,
            physical_rows=data.get("physical_rows"),
            source_captions_required=data.get("source_captions_required"),
            source_revision=data.get("source_revision"),
            view_revision=data.get("view_revision"),
            bucket_count=len(buckets),
            buckets=sorted(buckets, key=lambda b: b.file_count, reverse=True),
            filtering_statistics=data.get("filtering_statistics"),
            config=data.get("config"),
            cache_file_path=str(cache_path),
            **cond_fields,
        )

    def get_dataset_files(
        self,
        dataset_config: Dict[str, Any],
        bucket_key: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        view_revision: Optional[str] = None,
    ) -> DatasetFilePage:
        """Return a paginated file listing from the bucket cache."""
        cache_path = self._find_cache_file(dataset_config, "indices")
        if cache_path is None:
            if view_revision is not None:
                raise DatasetViewChanged("The prepared dataset is no longer available.")
            return DatasetFilePage(files=[], total=0, limit=limit, offset=offset)

        try:
            with cache_path.open("rb") as source:
                index = MembershipPages.open(source)
                revision = index.header["view_revision"] if index is not None else None
                if view_revision is not None and revision != view_revision:
                    raise DatasetViewChanged("The prepared dataset changed. Reopen its bucket to browse the current view.")
                if index is None:
                    files, total = read_bucket_page(read_bucket_groups(source), bucket_key, offset, limit)
                else:
                    files, total = index.read_page(bucket_key, offset, limit)
        except IncompleteWebshartMembership:
            if view_revision is not None:
                raise DatasetViewChanged("The prepared dataset is no longer available.") from None
            return DatasetFilePage(files=[], total=0, limit=limit, offset=offset)
        except DatasetViewChanged:
            raise
        except (ijson.JSONError, ValueError, OSError):
            return DatasetFilePage(files=[], total=0, limit=limit, offset=offset)

        return DatasetFilePage(
            files=files, total=total, limit=limit, offset=offset, bucket_key=bucket_key, view_revision=revision
        )

    def _webshart_view(self, dataset_config, view_revision):
        indices = self._find_cache_file(dataset_config, "indices") if dataset_config.get("type") == "webshart" else None
        index = None
        if indices is not None:
            with indices.open("rb") as source:
                try:
                    index = MembershipPages.open(source)
                except IncompleteWebshartMembership:
                    indices = None
        current_revision = index.header["view_revision"] if index is not None else None
        if view_revision is not None and current_revision != view_revision:
            raise DatasetViewChanged("The prepared dataset changed. Reopen its bucket to inspect the current view.")
        return indices, index

    def get_file_metadata(
        self, dataset_config: Dict[str, Any], file_path: str, *, view_revision: Optional[str] = None
    ) -> Optional[FileMetadata]:
        """Read per-file metadata from the metadata cache."""
        indices, index = self._webshart_view(dataset_config, view_revision)
        current_revision = index.header["view_revision"] if index is not None else None
        if index is not None and index.header.get("geometry_format") == "sqlite-v1":
            if index.header.get("source_pins"):
                with accepted_ref_lookup(indices, index) as contains:
                    if not contains(file_path):
                        return None
            name = index.header["geometry_name"]
            if Path(name).name != name:
                raise ValueError("Invalid Webshart geometry filename.")
            with closing(read_snapshot(indices.parent / name, index.header["geometry_sha256"])) as db:
                lookup = geometry_lookup(indices, index.header, db)
                if settings := index.header.get("transform_cache_recipe"):
                    from simpletuner.helpers.metadata.backends.webshart_transforms import WebshartTransforms

                    with (
                        self._open_view_data_backend(dataset_config, current_revision) as source,
                        closing(WebshartTransforms(source, settings, lookup.shard_revision)) as transforms,
                    ):
                        meta = transforms.read_batch([file_path])[0]
                else:
                    meta = lookup.get_metadata_by_filepath(file_path)
        else:
            cache_path = self._find_cache_file(dataset_config, "metadata")
            if cache_path is None:
                return None
            try:
                meta = self._load_cache(cache_path).get(file_path)
            except (json.JSONDecodeError, OSError):
                return None
        if meta is None:
            return None

        known_keys = {
            "original_size",
            "target_size",
            "intermediary_size",
            "aspect_ratio",
            "crop_coordinates",
            "bucket_key",
            "bbox_entities",
        }
        extra = {k: v for k, v in meta.items() if k not in known_keys}

        result = FileMetadata(
            path=file_path,
            display_name=self._get_display_name(dataset_config, file_path),
            original_size=meta.get("original_size"),
            target_size=meta.get("target_size"),
            intermediary_size=meta.get("intermediary_size"),
            aspect_ratio=meta.get("aspect_ratio"),
            crop_coordinates=meta.get("crop_coordinates"),
            bucket_key=meta.get("bucket_key"),
            bbox_entities=meta.get("bbox_entities"),
            view_revision=current_revision,
            extra=extra,
        )
        result.caption = self._get_file_caption(dataset_config, file_path, view_revision=current_revision)
        return result

    def update_crop_coordinates(
        self,
        dataset_config: Dict[str, Any],
        file_path: str,
        crop_coordinates: List[int],
    ) -> bool:
        """Update crop_coordinates for a file in the metadata cache.

        Returns True if the update succeeded.
        """
        cache_path = self._find_cache_file(dataset_config, "metadata")
        if cache_path is None:
            return False

        try:
            data = self._load_cache(cache_path)
        except (json.JSONDecodeError, OSError):
            return False

        if file_path not in data:
            return False

        data[file_path]["crop_coordinates"] = crop_coordinates
        with open(cache_path, "w") as f:
            json.dump(data, f)
        return True

    def update_bbox_entities(
        self,
        dataset_config: Dict[str, Any],
        file_path: str,
        bbox_entities: Optional[List[Dict[str, Any]]],
    ) -> bool:
        """Update bbox_entities for a file in the metadata cache.

        Returns True if the update succeeded.
        """
        cache_path = self._find_cache_file(dataset_config, "metadata")
        if cache_path is None:
            return False

        try:
            data = self._load_cache(cache_path)
        except (json.JSONDecodeError, OSError):
            return False

        if file_path not in data:
            return False

        if bbox_entities:
            data[file_path]["bbox_entities"] = bbox_entities
        else:
            data[file_path].pop("bbox_entities", None)

        with open(cache_path, "w") as f:
            json.dump(data, f)
        return True

    def rebuild_file_metadata(
        self,
        dataset_config: Dict[str, Any],
        global_config: Dict[str, Any],
        file_path: str,
    ) -> Optional[Dict[str, Any]]:
        """Recalculate metadata for a single file using TrainingSample.

        Recompute target size, intermediary size, crop and aspect ratio
        from the original size and the resolved dataset settings.

        Returns the updated metadata dict on success, None on failure.
        """
        from simpletuner.helpers.data_backend.config import create_backend_config
        from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
        from simpletuner.helpers.multiaspect.image import ImageGeometry, MultiaspectImage
        from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import DatasetScanService

        # Load existing metadata
        cache_path = self._find_cache_file(dataset_config, "metadata")
        if cache_path is None:
            return None
        try:
            all_metadata = self._load_cache(cache_path)
        except (json.JSONDecodeError, OSError):
            return None
        if file_path not in all_metadata:
            return None

        existing = all_metadata[file_path]
        original_size = existing.get("original_size")
        if not original_size:
            return None

        scan_args = DatasetScanService._build_scan_args(dataset_config, global_config)

        # Apply pixel_area conversion
        dataset_config = DatasetScanService._convert_pixel_area(dataset_config, scan_args)

        try:
            config = create_backend_config(dataset_config, vars(scan_args))
            geometry = ImageGeometry(
                MultiaspectImage._get_alignment(args=scan_args),
                MultiaspectImage._get_rounding(2, args=scan_args),
            )

            # Build image_metadata with only original_size (forces recalculation)
            rebuild_meta = {"original_size": tuple(original_size)}
            # Preserve video-specific fields
            for vkey in ("num_frames", "original_num_frames", "fps", "video_duration"):
                if vkey in existing:
                    rebuild_meta[vkey] = existing[vkey]

            sample = TrainingSample(
                image=None,
                data_backend_id=config.id,
                image_metadata=rebuild_meta,
                image_path=file_path,
                data_backend_config=config.to_dict()["config"],
                geometry=geometry,
            )
            prepared = sample.prepare()

            # Merge recalculated fields into existing metadata
            existing["target_size"] = list(prepared.target_size)
            existing["intermediary_size"] = list(prepared.intermediary_size)
            existing["crop_coordinates"] = list(prepared.crop_coordinates)
            existing["aspect_ratio"] = prepared.aspect_ratio

            with open(cache_path, "w") as f:
                json.dump(all_metadata, f)

            return existing

        except Exception as e:
            logger.warning("Failed to rebuild metadata for %s: %s", file_path, e)
            return None

    def delete_vae_cache_file(
        self,
        dataset_config: Dict[str, Any],
        file_path: str,
    ) -> Dict[str, Any]:
        """Delete the VAE cache .pt file for a single image/video.

        Returns a dict with 'deleted' (bool) and 'cache_path' (str or None).
        """
        cache_dir_vae = dataset_config.get("cache_dir_vae", "")
        if not cache_dir_vae:
            return {"deleted": False, "cache_path": None, "error": "No cache_dir_vae configured"}

        instance_data_dir = dataset_config.get("instance_data_dir", "")

        # Replicate VAECache.generate_vae_cache_filename logic (hash_filenames=True always)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        hashed = hashlib.sha256(base_filename.encode()).hexdigest()
        cache_filename = hashed + ".pt"

        # Compute subfolder (relative path within instance_data_dir)
        subfolders = ""
        if instance_data_dir:
            subfolders = os.path.dirname(file_path).replace(instance_data_dir, "")
            subfolders = subfolders.lstrip(os.sep)

        if subfolders:
            cache_path = os.path.join(cache_dir_vae, subfolders, cache_filename)
        else:
            cache_path = os.path.join(cache_dir_vae, cache_filename)

        cache_path = os.path.abspath(cache_path)

        if not os.path.isfile(cache_path):
            return {"deleted": False, "cache_path": cache_path, "error": "File not found"}

        try:
            os.remove(cache_path)
            return {"deleted": True, "cache_path": cache_path}
        except OSError as e:
            return {"deleted": False, "cache_path": cache_path, "error": str(e)}

    def _get_file_caption(
        self, dataset_config: Dict[str, Any], file_path: str, *, view_revision: Optional[str] = None
    ) -> Optional[str]:
        """Resolve the caption for a file based on the dataset's caption strategy."""
        strategy = dataset_config.get("caption_strategy", "filename")
        instance_prompt = dataset_config.get("instance_prompt", "")

        if strategy == "instanceprompt":
            return instance_prompt or None

        if strategy == "filename":
            return Path(file_path).stem.replace("_", " ")

        if strategy == "textfile":
            return self._read_textfile_caption(dataset_config, file_path)

        if strategy == "webshart":
            from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
            from simpletuner.helpers.prompts import CaptionNotFoundError, PromptHandler

            with self._open_view_data_backend(dataset_config, view_revision) as pinned:
                backend = pinned if pinned is not None else self._get_or_create_data_backend(dataset_config)
                settings = {**dataset_config, **DatasetRuntime.caption_config(dataset_config)}
                dataset = DatasetRuntime(dataset_config["id"], backend, None, settings)
                try:
                    caption = dataset.resolve_caption(file_path, caption_lookup=backend.get_caption)
                except CaptionNotFoundError:
                    return None
                return "\n".join(PromptHandler._normalize_caption_payload(caption)) or None

        # For parquet/huggingface/csv, try the textfile sidecar as a best-effort fallback
        caption = self._read_textfile_caption(dataset_config, file_path)
        if caption:
            return caption

        return None

    def _read_textfile_caption(self, dataset_config: Dict[str, Any], file_path: str) -> Optional[str]:
        """Read a .txt sidecar caption file."""
        caption_filename = os.path.splitext(file_path)[0] + ".txt"

        # Try local filesystem first
        resolved = self._resolve_image_path(dataset_config, caption_filename)
        if resolved is not None:
            try:
                return resolved.read_text(encoding="utf-8").strip()
            except OSError:
                return None

        # Try remote backend
        try:
            backend = self._get_or_create_data_backend(dataset_config)
            if backend is None:
                return None
            data = backend.read(caption_filename)
            if data is None:
                return None
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return data.strip()
        except Exception:
            return None

    def _resolve_image_path(self, dataset_config: Dict[str, Any], file_path: str) -> Optional[Path]:
        """Resolve a relative file path from the cache to an absolute path."""
        instance_data_dir = dataset_config.get("instance_data_dir", "")
        if not instance_data_dir:
            return None

        candidate = Path(instance_data_dir) / file_path
        if candidate.is_file():
            return candidate

        # The file_path might already be absolute
        abs_candidate = Path(file_path)
        if abs_candidate.is_file():
            return abs_candidate

        return None

    def _get_thumbnail_cache_path(self, dataset_id: str, file_path: str, view_revision: Optional[str] = None) -> Path:
        """Get the local cache path for a thumbnail."""
        cache_key = file_path if view_revision is None else f"{view_revision}:{file_path}"
        path_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_dir = self.project_root / self.THUMBNAIL_CACHE_DIR / dataset_id
        return cache_dir / f"{path_hash}.jpg"

    @staticmethod
    def _ensure_pil_image(data) -> Optional[Image.Image]:
        """Convert backend read_image() result to a PIL Image.

        Some backends (e.g. HuggingFace video) return numpy arrays instead of
        PIL Images.  For multi-frame arrays the first frame is extracted.
        """
        if data is None:
            return None
        if isinstance(data, Image.Image):
            return data
        if isinstance(data, np.ndarray):
            try:
                if data.ndim == 4:
                    data = data[0]
                return Image.fromarray(data)
            except Exception as e:
                logger.warning("Failed to convert numpy array to PIL Image: %s", e)
        return None

    def _save_thumbnail_to_cache(self, cache_path: Path, img: Image.Image, max_size: int = 256) -> None:
        """Save a PIL Image as a JPEG thumbnail to the cache directory."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        thumb = img.copy()
        if thumb.mode != "RGB":
            thumb = thumb.convert("RGB")
        thumb.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        thumb.save(cache_path, format="JPEG", quality=85)

    def _get_or_create_data_backend(self, dataset_config: Dict[str, Any]):
        """Create or retrieve a cached data backend for on-demand image reading."""
        dataset_id = dataset_config.get("id", "")
        if dataset_id in _data_backend_cache:
            return _data_backend_cache[dataset_id]

        data_backend = self._create_data_backend(dataset_config)
        _data_backend_cache[dataset_id] = data_backend
        return data_backend

    @staticmethod
    def _create_data_backend(dataset_config):
        from simpletuner.helpers.data_backend.builders import create_backend_builder
        from simpletuner.helpers.data_backend.config import create_backend_config
        from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import _ScanAcceleratorStub, _ScanArgsNamespace

        scan_args = _ScanArgsNamespace()
        accelerator = _ScanAcceleratorStub()

        args_dict = {k: getattr(scan_args, k, None) for k in vars(scan_args)}
        config = create_backend_config(dataset_config, args_dict)

        builder = create_backend_builder(config.backend_type, accelerator, scan_args)
        return builder.build(config)

    @contextmanager
    def _open_view_data_backend(self, dataset_config, view_revision):
        if view_revision is None:
            yield None
            return
        indices, index = self._webshart_view(dataset_config, view_revision)
        if not index.header.get("source_pins"):
            yield None
            return
        backend = self._create_data_backend(dataset_config)
        if backend.dataset.fingerprint != index.header["source_revision"]:
            raise DatasetViewChanged("The source catalogue changed. Prepare the dataset again to browse the new source.")
        geometry_path = prepared_geometry_path(indices, index.header)
        with closing(_read_index(geometry_path)) as db:
            lookup = geometry_lookup(indices, index.header, db)
            backend.dataset.bind_source_revisions(lookup.shard_revision)
            try:
                yield backend
            finally:
                backend.dataset.bind_source_revisions(None)

    def _read_remote_thumbnail(
        self,
        dataset_config: Dict[str, Any],
        file_path: str,
        max_size: int,
    ) -> Optional[str]:
        """Read an image from a remote backend, generate and cache a thumbnail."""
        dataset_id = dataset_config.get("id", "")
        cache_path = self._get_thumbnail_cache_path(dataset_id, file_path)

        try:
            data_backend = self._get_or_create_data_backend(dataset_config)
        except Exception as e:
            logger.warning("Failed to create data backend for %s: %s", dataset_id, e)
            return None

        try:
            image_data = data_backend.read_image(file_path)
        except Exception as e:
            logger.warning("Failed to read image %s from %s: %s", file_path, dataset_id, e)
            return None

        if image_data is None:
            return None

        pil_image = self._ensure_pil_image(image_data)
        if pil_image is None:
            return None

        # Cache the thumbnail to disk for future requests
        try:
            self._save_thumbnail_to_cache(cache_path, pil_image, max_size)
        except Exception as e:
            logger.warning("Failed to cache thumbnail for %s: %s", file_path, e)

        return generate_thumbnail_from_pil(pil_image, max_size)

    def _load_pil_image(
        self, dataset_config: Dict[str, Any], file_path: str, *, view_revision: Optional[str] = None
    ) -> Optional[Image.Image]:
        """Load the full PIL Image for a file, local or remote.

        For video files, extracts the first frame via trainingsample's VideoCapture.
        """
        backend_type = dataset_config.get("type", "local")

        if backend_type == "local":
            resolved = self._resolve_image_path(dataset_config, file_path)
            if resolved is None:
                return None
            try:
                if resolved.suffix.lower() in _VIDEO_EXTENSIONS:
                    frame = _extract_first_frame(resolved)
                    if frame is None:
                        return None
                    return Image.fromarray(frame)
                with Image.open(resolved) as img:
                    return img.copy()
            except Exception as e:
                logger.warning("Failed to open %s: %s", resolved, e)
                return None

        if view_revision is not None:
            try:
                with self._open_view_data_backend(dataset_config, view_revision) as pinned:
                    if pinned is not None:
                        _, images = pinned.read_image_batch([file_path])
                        return self._ensure_pil_image(images[0]) if images else None
            except DatasetViewChanged:
                raise
            except (ValueError, OSError) as exc:
                logger.warning("Failed to load the pinned image %s: %s", file_path, exc)
                return None

        # Remote: try thumbnail disk cache first for speed, then fetch full image
        dataset_id = dataset_config.get("id", "")
        cached_path = self._get_thumbnail_cache_path(dataset_id, file_path, view_revision)
        if cached_path.is_file():
            try:
                with Image.open(cached_path) as img:
                    return img.copy()
            except Exception:
                pass

        try:
            data_backend = self._get_or_create_data_backend(dataset_config)
            image_data = data_backend.read_image(file_path)
            pil_image = self._ensure_pil_image(image_data)
            if pil_image is not None:
                self._save_thumbnail_to_cache(cached_path, pil_image)
            return pil_image
        except DatasetViewChanged:
            raise
        except Exception as e:
            logger.warning("Failed to load remote image %s: %s", file_path, e)
            return None

    def get_image_preview(
        self,
        dataset_config: Dict[str, Any],
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        max_size: int = 1024,
        *,
        view_revision: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get preview images for the detail modal.

        Returns a dict with:
          - 'intermediary': the image resized to intermediary_size (base64)
          - 'cropped': the crop result from intermediary (base64), or None
          - 'original': the original image (base64)
        The crop simulation mirrors TrainingSample.prepare():
          original -> resize to intermediary_size -> crop at crop_coordinates
        Uses the trainingsample Rust library for resize (matching the training
        pipeline) and numpy slicing for the crop (matching cropping.py).
        """
        if view_revision is None and metadata is not None:
            view_revision = metadata.get("view_revision")
        img = self._load_pil_image(dataset_config, file_path, view_revision=view_revision)
        if img is None:
            return None

        original_b64 = generate_thumbnail_from_pil(img, max_size)

        intermediary_size = metadata.get("intermediary_size") if metadata else None
        crop_coordinates = metadata.get("crop_coordinates") if metadata else None
        target_size = metadata.get("target_size") if metadata else None

        if not intermediary_size or not crop_coordinates or not target_size:
            return {"original": original_b64, "intermediary": None, "cropped": None}

        try:
            inter_w, inter_h = int(intermediary_size[0]), int(intermediary_size[1])
            # crop_coordinates are (top, left) matching cropping.py convention
            top, left = int(crop_coordinates[0]), int(crop_coordinates[1])
            tw, th = int(target_size[0]), int(target_size[1])

            # Convert to numpy RGB for trainingsample operations
            img_array = np.array(img.convert("RGB"))

            # Step 1: Resize to intermediary via trainingsample (mirrors _downsample_before_crop)
            resized = tsr.batch_resize_images([img_array], [(inter_w, inter_h)])
            if not resized:
                raise RuntimeError("batch_resize_images returned empty result")
            intermediary_array = resized[0]

            # Step 2: Crop via numpy slicing (mirrors cropping.py)
            left = max(0, min(left, inter_w))
            top = max(0, min(top, inter_h))
            right = min(left + tw, inter_w)
            bottom = min(top + th, inter_h)
            cropped_array = intermediary_array[top:bottom, left:right, :]

            intermediary_b64 = _array_to_thumbnail(intermediary_array, max_size)
            cropped_b64 = _array_to_thumbnail(cropped_array, max_size)
        except Exception as e:
            logger.warning("Failed to generate crop preview: %s", e)
            return {"original": original_b64, "intermediary": None, "cropped": None}

        return {
            "original": original_b64,
            "intermediary": intermediary_b64,
            "cropped": cropped_b64,
        }

    def get_dataset_thumbnails(
        self,
        dataset_config: Dict[str, Any],
        bucket_key: Optional[str] = None,
        max_size: int = DEFAULT_THUMBNAIL_SIZE,
        limit: int = 24,
        offset: int = 0,
        view_revision: Optional[str] = None,
    ) -> List[ViewerThumbnailInfo]:
        """Generate thumbnails for files in a dataset bucket."""
        max_size = min(max(max_size, 64), self.MAX_THUMBNAIL_SIZE)
        dataset_id = dataset_config.get("id", "")
        backend_type = dataset_config.get("type", "local")

        # Get files from cache
        file_page = self.get_dataset_files(
            dataset_config, bucket_key=bucket_key, limit=limit, offset=offset, view_revision=view_revision
        )

        if file_page.view_revision and file_page.files:
            with self._open_view_data_backend(dataset_config, file_page.view_revision) as pinned:
                if pinned is not None:
                    return self._prepared_thumbnails(dataset_config, pinned, file_page, max_size)

        thumbnails: List[ViewerThumbnailInfo] = []
        for file_path in file_page.files:
            display_name = self._get_display_name(dataset_config, file_path)
            if backend_type == "local":
                resolved = self._resolve_image_path(dataset_config, file_path)
                if resolved is None:
                    thumbnails.append(
                        ViewerThumbnailInfo(
                            path=file_path,
                            filename=display_name,
                            error="File not found",
                        )
                    )
                    continue

                thumb_data = generate_thumbnail(resolved, max_size)
                thumbnails.append(
                    ViewerThumbnailInfo(
                        path=file_path,
                        filename=display_name,
                        thumbnail=thumb_data,
                        error=None if thumb_data else "Failed to generate thumbnail",
                    )
                )
            else:
                # For remote backends (aws, huggingface), check local cache first,
                # then fetch on-demand from the data backend
                cached_path = self._get_thumbnail_cache_path(dataset_id, file_path)
                if cached_path.is_file():
                    thumb_data = generate_thumbnail(cached_path, max_size)
                else:
                    thumb_data = self._read_remote_thumbnail(dataset_config, file_path, max_size)

                thumbnails.append(
                    ViewerThumbnailInfo(
                        path=file_path,
                        filename=display_name,
                        thumbnail=thumb_data,
                        error=None if thumb_data else "Failed to load remote image",
                    )
                )

        return thumbnails

    def _prepared_thumbnails(self, dataset_config, backend, page, max_size):
        dataset_id = dataset_config["id"]
        cached = {ref: self._get_thumbnail_cache_path(dataset_id, ref, page.view_revision) for ref in page.files}
        missing = [ref for ref, path in cached.items() if not path.is_file()]
        loaded = {}
        if missing:
            refs, images = backend.read_image_batch(missing)
            loaded.update(zip(refs, images, strict=True))
        thumbnails = []
        for ref in page.files:
            path = cached[ref]
            image = self._ensure_pil_image(loaded.get(ref))
            if image is not None:
                self._save_thumbnail_to_cache(path, image, self.MAX_THUMBNAIL_SIZE)
                thumbnail = generate_thumbnail_from_pil(image, max_size)
            else:
                thumbnail = generate_thumbnail(path, max_size) if path.is_file() else None
            thumbnails.append(
                ViewerThumbnailInfo(
                    path=ref,
                    filename=self._get_display_name(dataset_config, ref),
                    thumbnail=thumbnail,
                    error=None if thumbnail else "Failed to load remote image",
                )
            )
        return thumbnails

    # --- Stage 3: Caption Validation ---

    def get_caption_status(self, dataset_config: Dict[str, Any]) -> CaptionValidationResult:
        """Validate caption coverage for a dataset based on its caption strategy."""
        dataset_id = dataset_config.get("id", "")
        strategy = dataset_config.get("caption_strategy", "textfile")

        if dataset_config.get("type") == "webshart":
            summary = self.get_dataset_summary(dataset_config)
            if summary.scan_incomplete:
                return CaptionValidationResult(
                    dataset_id=dataset_id,
                    strategy=strategy,
                    captioned_files=None,
                    coverage_ratio=None,
                    warnings=["Caption coverage is unavailable until the scan completes."],
                )
            if summary.view_revision is not None:
                total = summary.accepted_samples
                if strategy == "filename" or (strategy == "webshart" and summary.source_captions_required is True):
                    captioned = total
                elif strategy == "instanceprompt":
                    captioned = total if dataset_config.get("instance_prompt") else 0
                else:
                    return CaptionValidationResult(
                        dataset_id=dataset_id,
                        strategy=strategy,
                        total_files=total,
                        captioned_files=None,
                        coverage_ratio=None,
                        warnings=["Caption coverage for this strategy has not been computed."],
                    )
                return CaptionValidationResult(
                    dataset_id=dataset_id,
                    strategy=strategy,
                    total_files=total,
                    captioned_files=captioned,
                    coverage_ratio=captioned / total if total else 0.0,
                )

        # Get all files from bucket cache
        file_page = self.get_dataset_files(dataset_config, limit=1_000_000, offset=0)
        all_files = file_page.files
        total = len(all_files)

        if total == 0:
            return CaptionValidationResult(dataset_id=dataset_id, strategy=strategy, total_files=0)

        if strategy == "filename":
            return CaptionValidationResult(
                dataset_id=dataset_id,
                strategy=strategy,
                coverage_ratio=1.0,
                total_files=total,
                captioned_files=total,
            )

        if strategy == "instanceprompt":
            instance_prompt = dataset_config.get("instance_prompt", "")
            if instance_prompt:
                return CaptionValidationResult(
                    dataset_id=dataset_id,
                    strategy=strategy,
                    coverage_ratio=1.0,
                    total_files=total,
                    captioned_files=total,
                )
            return CaptionValidationResult(
                dataset_id=dataset_id,
                strategy=strategy,
                coverage_ratio=0.0,
                total_files=total,
                captioned_files=0,
                warnings=["instance_prompt is not set in dataset config"],
            )

        if strategy == "textfile":
            return self._validate_textfile_captions(dataset_config, all_files)

        if strategy == "parquet":
            return self._validate_parquet_captions(dataset_config, all_files)

        if strategy == "csv":
            return self._validate_csv_captions(dataset_config, all_files)

        if strategy == "huggingface":
            return CaptionValidationResult(
                dataset_id=dataset_id,
                strategy=strategy,
                total_files=total,
                warnings=["Cannot validate HuggingFace captions locally"],
            )

        return CaptionValidationResult(
            dataset_id=dataset_id,
            strategy=strategy,
            total_files=total,
            warnings=[f"Unknown caption strategy: {strategy}"],
        )

    def _validate_textfile_captions(self, dataset_config: Dict[str, Any], files: List[str]) -> CaptionValidationResult:
        """Check for .txt sidecar caption files."""
        dataset_id = dataset_config.get("id", "")
        instance_data_dir = dataset_config.get("instance_data_dir", "")
        captioned = 0
        uncaptioned = []

        for f in files:
            resolved = self._resolve_image_path(dataset_config, f)
            if resolved is None:
                uncaptioned.append(f)
                continue
            caption_path = resolved.with_suffix(".txt")
            if caption_path.is_file():
                captioned += 1
            else:
                uncaptioned.append(f)

        total = len(files)
        return CaptionValidationResult(
            dataset_id=dataset_id,
            strategy="textfile",
            coverage_ratio=captioned / total if total > 0 else 0.0,
            total_files=total,
            captioned_files=captioned,
            uncaptioned_files=uncaptioned[:500],
        )

    def _validate_parquet_captions(self, dataset_config: Dict[str, Any], files: List[str]) -> CaptionValidationResult:
        """Check parquet file for caption column coverage."""
        dataset_id = dataset_config.get("id", "")
        total = len(files)
        warnings = []

        parquet_path = dataset_config.get("parquet_file") or dataset_config.get("csv_file")
        caption_column = dataset_config.get("parquet_caption_column") or dataset_config.get("csv_caption_column", "caption")
        filename_column = dataset_config.get("parquet_filename_column") or dataset_config.get("csv_url_column", "filename")

        if not parquet_path:
            return CaptionValidationResult(
                dataset_id=dataset_id,
                strategy="parquet",
                total_files=total,
                warnings=["No parquet_file path configured"],
            )

        try:
            import pyarrow.parquet as pq

            table = pq.read_table(parquet_path, columns=[filename_column, caption_column])
            parquet_filenames = set(table.column(filename_column).to_pylist())
            file_basenames = {Path(f).name for f in files}
            matched = file_basenames & parquet_filenames
            unmatched = [f for f in files if Path(f).name not in parquet_filenames]

            return CaptionValidationResult(
                dataset_id=dataset_id,
                strategy="parquet",
                coverage_ratio=len(matched) / total if total > 0 else 0.0,
                total_files=total,
                captioned_files=len(matched),
                uncaptioned_files=unmatched[:500],
            )
        except ImportError:
            warnings.append("pyarrow not installed; cannot validate parquet captions")
        except Exception as e:
            warnings.append(f"Error reading parquet file: {e}")

        return CaptionValidationResult(
            dataset_id=dataset_id,
            strategy="parquet",
            total_files=total,
            warnings=warnings,
        )

    def _validate_csv_captions(self, dataset_config: Dict[str, Any], files: List[str]) -> CaptionValidationResult:
        """Check CSV file for caption column coverage."""
        import csv as csv_module

        dataset_id = dataset_config.get("id", "")
        total = len(files)

        csv_path = dataset_config.get("csv_file")
        caption_column = dataset_config.get("csv_caption_column", "caption")
        filename_column = dataset_config.get("csv_url_column", "filename")

        if not csv_path or not Path(csv_path).is_file():
            return CaptionValidationResult(
                dataset_id=dataset_id,
                strategy="csv",
                total_files=total,
                warnings=["No csv_file path configured or file not found"],
            )

        try:
            csv_filenames = set()
            with open(csv_path, "r", newline="") as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    fname = row.get(filename_column, "")
                    if fname and row.get(caption_column):
                        csv_filenames.add(Path(fname).name)

            file_basenames = {Path(f).name for f in files}
            matched = file_basenames & csv_filenames
            unmatched = [f for f in files if Path(f).name not in csv_filenames]

            return CaptionValidationResult(
                dataset_id=dataset_id,
                strategy="csv",
                coverage_ratio=len(matched) / total if total > 0 else 0.0,
                total_files=total,
                captioned_files=len(matched),
                uncaptioned_files=unmatched[:500],
            )
        except Exception as e:
            return CaptionValidationResult(
                dataset_id=dataset_id,
                strategy="csv",
                total_files=total,
                warnings=[f"Error reading CSV file: {e}"],
            )

    # --- Stage 4: Size Filter Viewer ---

    def get_filtering_summary(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Return filtering_statistics from the bucket cache."""
        summary = self.get_dataset_summary(dataset_config)
        return {
            "dataset_id": summary.dataset_id,
            "has_cache": summary.has_cache,
            "filtering_statistics": summary.filtering_statistics,
        }

    def get_filtered_files(
        self,
        dataset_config: Dict[str, Any],
        reason: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> FilteredFilesReport:
        """Re-apply filter rules from config to identify which files were excluded."""
        dataset_id = dataset_config.get("id", "")

        if dataset_config.get("type") == "webshart":
            publication = self._find_cache_file(dataset_config, "indices")
            if publication is not None:
                with publication.open("rb") as source:
                    index = MembershipPages.open(source)
                if index is not None and index.header.get("geometry_format") == "sqlite-v1":
                    counts, rows = read_exclusion_page(publication, index.header, reason, limit, offset)
                    return FilteredFilesReport(
                        dataset_id=dataset_id,
                        total_filtered=sum(counts.values()),
                        by_reason=counts,
                        files=[FilteredFileEntry(path=ref, reason=why) for ref, why in rows],
                        scope="recorded_exclusions",
                    )

        # Load metadata cache
        cache_path = self._find_cache_file(dataset_config, "metadata")
        if cache_path is None:
            return FilteredFilesReport(dataset_id=dataset_id)

        try:
            all_metadata = self._load_cache(cache_path)
        except (json.JSONDecodeError, OSError):
            return FilteredFilesReport(dataset_id=dataset_id)

        # Get the set of files that ARE in buckets (accepted)
        indices_path = self._find_cache_file(dataset_config, "indices")
        accepted_files: set = set()
        if indices_path:
            try:
                indices_data = self._load_cache(indices_path)
                for bucket_files in indices_data.get("aspect_ratio_bucket_indices", {}).values():
                    accepted_files.update(bucket_files)
            except (json.JSONDecodeError, OSError):
                pass

        # Filter rules from config
        min_size = dataset_config.get("minimum_image_size")
        max_size = dataset_config.get("maximum_image_size")
        min_aspect = dataset_config.get("minimum_aspect_ratio")
        max_aspect = dataset_config.get("maximum_aspect_ratio")

        entries: List[FilteredFileEntry] = []
        by_reason: Dict[str, int] = {}

        for filepath, meta in all_metadata.items():
            if filepath in accepted_files:
                continue
            if not isinstance(meta, dict):
                continue

            original_size = meta.get("original_size")
            aspect_ratio = meta.get("aspect_ratio")

            entry_reason = "unknown"
            threshold = None

            if original_size and min_size:
                w, h = original_size[0], original_size[1]
                if w < min_size or h < min_size:
                    entry_reason = "too_small"
                    threshold = f"minimum_image_size={min_size}"

            if original_size and max_size and entry_reason == "unknown":
                w, h = original_size[0], original_size[1]
                if w > max_size or h > max_size:
                    entry_reason = "too_large"
                    threshold = f"maximum_image_size={max_size}"

            if aspect_ratio is not None and entry_reason == "unknown":
                if min_aspect and aspect_ratio < min_aspect:
                    entry_reason = "aspect_out_of_range"
                    threshold = f"minimum_aspect_ratio={min_aspect}"
                elif max_aspect and aspect_ratio > max_aspect:
                    entry_reason = "aspect_out_of_range"
                    threshold = f"maximum_aspect_ratio={max_aspect}"

            if reason and entry_reason != reason:
                continue

            by_reason[entry_reason] = by_reason.get(entry_reason, 0) + 1
            entries.append(
                FilteredFileEntry(
                    path=filepath,
                    reason=entry_reason,
                    original_size=original_size,
                    aspect_ratio=aspect_ratio,
                    threshold=threshold,
                )
            )

        total_filtered = len(entries)
        page = entries[offset : offset + limit]

        return FilteredFilesReport(
            dataset_id=dataset_id,
            total_filtered=total_filtered,
            by_reason=by_reason,
            files=page,
        )

    # --- Stage 5: Conditioning Dataset Visualization ---

    def get_dataset_graph(self, datasets: List[Dict[str, Any]]) -> DatasetGraph:
        """Build a dependency graph from multidatabackend config."""
        nodes = []
        edges = []
        dataset_ids = {ds.get("id", "") for ds in datasets}

        for ds in datasets:
            ds_id = ds.get("id", "")
            nodes.append(
                DatasetNode(
                    id=ds_id,
                    dataset_type=str(ds.get("dataset_type", "image")),
                    backend_type=str(ds.get("type", "local")),
                )
            )

            # conditioning_data references
            for cond in ds.get("conditioning_data", []) or []:
                cond_id = cond if isinstance(cond, str) else cond.get("id", "")
                if cond_id and cond_id in dataset_ids:
                    edges.append(
                        DatasetEdge(
                            source_id=ds_id,
                            target_id=cond_id,
                            relationship="conditioning_data",
                        )
                    )

            # source_dataset_id
            source_id = ds.get("source_dataset_id")
            if source_id and source_id in dataset_ids:
                edges.append(
                    DatasetEdge(
                        source_id=source_id,
                        target_id=ds_id,
                        relationship="source_dataset",
                    )
                )

        return DatasetGraph(nodes=nodes, edges=edges)

    @contextmanager
    def _indexed_conditioning_view(self, config):
        if config.get("type") == "webshart":
            publication = self._find_cache_file(config, "indices")
            if publication is not None:
                with publication.open("rb") as source:
                    pages = MembershipPages.open(source)
                    if pages is not None and pages.header.get("geometry_format") == "sqlite-v1":
                        with accepted_ref_lookup(publication, pages) as contains:
                            yield pages, contains
                        return
        yield None

    def get_conditioning_pairs(
        self,
        source_config: Dict[str, Any],
        conditioning_config: Dict[str, Any],
        limit: int = 12,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Match files by basename between source and conditioning datasets."""
        with ExitStack() as stack:
            source, target = [
                stack.enter_context(self._indexed_conditioning_view(config))
                for config in (source_config, conditioning_config)
            ]
            if source or target:
                can_pair = source is not None and target is not None
                page = source[0].iter_refs(None, offset, limit) if can_pair else ()
                return {
                    "pairs": [{"source_path": ref, "conditioning_path": ref} for ref in page if target[1](ref)],
                    "scope": "source_page" if can_pair else "unavailable",
                    "source_total": source[0].count() if can_pair else None,
                    "total_pairs": None,
                    "source_orphan_count": None,
                    "conditioning_orphan_count": None,
                    "limit": limit,
                    "offset": offset,
                }
        source_files = self.get_dataset_files(source_config, limit=1_000_000).files
        cond_files = self.get_dataset_files(conditioning_config, limit=1_000_000).files

        source_by_name = {Path(f).stem: f for f in source_files}
        cond_by_name = {Path(f).stem: f for f in cond_files}

        pairs = []
        for name, source_path in source_by_name.items():
            if name in cond_by_name:
                pairs.append(
                    ConditioningPair(
                        source_path=source_path,
                        conditioning_path=cond_by_name[name],
                    )
                )

        total = len(pairs)
        page = pairs[offset : offset + limit]

        source_orphans = [f for name, f in source_by_name.items() if name not in cond_by_name]
        cond_orphans = [f for name, f in cond_by_name.items() if name not in source_by_name]

        return {
            "pairs": [p.model_dump() for p in page],
            "total_pairs": total,
            "source_orphan_count": len(source_orphans),
            "conditioning_orphan_count": len(cond_orphans),
            "limit": limit,
            "offset": offset,
        }

    def get_orphaned_files(
        self,
        source_config: Dict[str, Any],
        conditioning_config: Dict[str, Any],
        limit: int = 500,
        offset: int = 0,
    ) -> OrphanReport:
        """Find files without matching counterparts between source/conditioning datasets."""
        with ExitStack() as stack:
            source, target = [
                stack.enter_context(self._indexed_conditioning_view(config))
                for config in (source_config, conditioning_config)
            ]
            if source or target:
                result = OrphanReport(
                    source_id=source_config["id"],
                    conditioning_id=conditioning_config["id"],
                    limit=limit,
                    offset=offset,
                    scope="unavailable",
                )
                if source and target:
                    result.scope = "sample_pages"
                    result.source_orphans = [ref for ref in source[0].iter_refs(None, offset, limit) if not target[1](ref)]
                    result.conditioning_orphans = [
                        ref for ref in target[0].iter_refs(None, offset, limit) if not source[1](ref)
                    ]
                return result
        source_files = self.get_dataset_files(source_config, limit=1_000_000).files
        cond_files = self.get_dataset_files(conditioning_config, limit=1_000_000).files

        source_by_name = {Path(f).stem: f for f in source_files}
        cond_by_name = {Path(f).stem: f for f in cond_files}

        return OrphanReport(
            offset=offset,
            limit=limit,
            source_id=source_config.get("id", ""),
            conditioning_id=conditioning_config.get("id", ""),
            source_orphans=[f for name, f in source_by_name.items() if name not in cond_by_name][offset : offset + limit],
            conditioning_orphans=[f for name, f in cond_by_name.items() if name not in source_by_name][
                offset : offset + limit
            ],
        )

    def get_conditioning_file_match(
        self,
        source_config: Dict[str, Any],
        all_datasets: List[Dict[str, Any]],
        source_file_path: str,
    ) -> Optional[Dict[str, Any]]:
        """Find the conditioning image matching a source file by stem.

        Checks linked conditioning datasets in the plan and also attempts
        to discover auto-generated conditioning images on disk.
        """
        from simpletuner.helpers.data_backend.base import conditioning_file_path
        from simpletuner.helpers.data_backend.webshart import WebshartDataBackend

        native_ref = source_config.get("type") == "webshart" and WebshartDataBackend.is_sample_id(source_file_path)
        if native_ref:
            with self._indexed_conditioning_view(source_config) as source:
                if source is not None and not source[1](source_file_path):
                    return None
        source_stem = Path(source_file_path).stem

        # 1. Check conditioning_data references (datasets in the plan)
        cond_ids = source_config.get("conditioning_data") or []
        if isinstance(cond_ids, str):
            cond_ids = [cond_ids]

        for cond_id in cond_ids:
            cond_config = next((d for d in all_datasets if d.get("id") == cond_id), None)
            if cond_config is None:
                continue
            with self._indexed_conditioning_view(cond_config) as target:
                if target is not None:
                    if target[1](source_file_path):
                        return {
                            "conditioning_id": cond_id,
                            "conditioning_type": cond_config.get("conditioning_type"),
                            "file_path": source_file_path,
                            "thumbnail": self._read_remote_thumbnail(
                                cond_config, source_file_path, self.DEFAULT_THUMBNAIL_SIZE
                            ),
                        }
                    continue
            if native_ref:
                root = cond_config.get("instance_data_dir")
                if cond_config.get("type", "local") != "local" or not root:
                    continue
                path = Path(conditioning_file_path(source_file_path, root, cond_config)).resolve()
                if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
                    continue
                match = {"path": path}
            else:
                match = self._find_conditioning_file_by_stem(cond_config, source_stem)
            if match:
                thumbnail = generate_thumbnail(match["path"])
                return {
                    "conditioning_id": cond_id,
                    "conditioning_type": cond_config.get("conditioning_type"),
                    "file_path": str(match["path"]),
                    "thumbnail": thumbnail,
                }

        # 2. Try auto-generated conditioning from 'conditioning' generators
        generators = source_config.get("conditioning") or []
        if isinstance(generators, dict):
            generators = [generators]

        source_id = source_config.get("id", "")
        for gen in generators:
            gen_type = gen.get("type", "")
            # Derive the ID the same way the duplicator does at runtime
            gen_id = gen.get("id") or f"{source_id}_conditioning_{gen_type}"
            # Skip if already checked via conditioning_data
            if gen_id in cond_ids:
                continue

            # Try to find the conditioning images on disk
            instance_dir = gen.get("instance_data_dir")
            if not instance_dir:
                instance_dir = self._derive_conditioning_dir(source_config, gen_id)

            if instance_dir and Path(instance_dir).is_dir():
                match = self._scan_directory_for_stem(instance_dir, source_stem)
                if match:
                    thumbnail = generate_thumbnail(str(match))
                    return {
                        "conditioning_id": gen_id,
                        "conditioning_type": gen.get("conditioning_type", gen.get("type")),
                        "file_path": str(match),
                        "thumbnail": thumbnail,
                    }

            # 3. Images don't exist on disk — generate on-the-fly preview
            result = self._generate_conditioning_preview(source_config, source_file_path, gen)
            if result is not None:
                result["conditioning_id"] = gen_id
                return result

        return None

    def _find_conditioning_file_by_stem(self, cond_config: Dict[str, Any], stem: str) -> Optional[Dict[str, Any]]:
        """Look up a file matching *stem* in a conditioning dataset's cache."""
        files = self.get_dataset_files(cond_config, limit=1_000_000).files
        for f in files:
            if Path(f).stem == stem:
                resolved = self._resolve_image_path(cond_config, f)
                if resolved:
                    return {"path": resolved}
        return None

    @staticmethod
    def _derive_conditioning_dir(source_config: Dict[str, Any], gen_id: str) -> Optional[str]:
        """Best-effort derivation of where auto-generated conditioning images live."""
        # Try cache_dir_vae parent as an approximation of cache_root
        cache_dir_vae = source_config.get("cache_dir_vae", "")
        if cache_dir_vae:
            # cache_dir_vae is typically {cache_root}/vae/{model_family}/{id}
            # We need {cache_root}/conditioning_data/{gen_id}
            parts = Path(cache_dir_vae).parts
            try:
                vae_idx = parts.index("vae")
                cache_root = str(Path(*parts[:vae_idx]))
                return os.path.join(cache_root, "conditioning_data", gen_id)
            except (ValueError, TypeError):
                pass
        return None

    @staticmethod
    def _scan_directory_for_stem(directory: str, stem: str) -> Optional[Path]:
        """Find the first image file in *directory* whose stem matches."""
        dir_path = Path(directory)
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
            candidate = dir_path / (stem + ext)
            if candidate.is_file():
                return candidate
        return None

    def _generate_conditioning_preview(
        self,
        source_config: Dict[str, Any],
        source_file_path: str,
        generator_config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Generate a conditioning image on-the-fly for preview.

        Uses the same SampleGenerator pipeline as the trainer to produce
        a preview when conditioning images haven't been generated yet.
        """
        gen_type = generator_config.get("type", "")
        if not gen_type:
            return None

        global GENERATOR_REGISTRY, GENERATOR_REGISTRY_AVAILABLE
        if GENERATOR_REGISTRY is None and not GENERATOR_REGISTRY_AVAILABLE:
            try:
                from simpletuner.helpers.data_generation.sample_generator import GENERATOR_REGISTRY as _reg

                GENERATOR_REGISTRY = _reg
                GENERATOR_REGISTRY_AVAILABLE = True
            except ImportError:
                pass

        if GENERATOR_REGISTRY is None:
            logger.warning("sample_generator not available for on-the-fly conditioning")
            return None

        generator_class = GENERATOR_REGISTRY.get(gen_type)
        if generator_class is None:
            logger.warning("Unknown conditioning generator type: %s", gen_type)
            return None

        # Load the source image
        source_image = self._load_pil_image(source_config, source_file_path)
        if source_image is None:
            return None

        try:
            generator = generator_class(generator_config)

            # Check if this generator needs GPU — still attempt it, the
            # generator itself handles device selection gracefully
            result_images = generator.transform_batch([source_image], [source_file_path], [{}], None)

            if not result_images or result_images[0] is None:
                return None

            thumbnail = generate_thumbnail_from_pil(result_images[0], max_size=512)
            return {
                "conditioning_type": generator_config.get("conditioning_type", gen_type),
                "file_path": source_file_path,
                "thumbnail": thumbnail,
                "generated_on_the_fly": True,
            }
        except Exception as e:
            logger.warning(
                "Failed to generate %s conditioning preview for %s: %s",
                gen_type,
                source_file_path,
                e,
            )
            return None

    # --- Stage 6: Audio/Video Preview ---

    def get_video_preview(self, dataset_config: Dict[str, Any], file_path: str, frame_count: int = 4) -> VideoPreview:
        """Extract evenly-spaced frames from a video file using ffmpeg."""
        resolved = self._resolve_image_path(dataset_config, file_path)
        if resolved is None:
            return VideoPreview(path=file_path)

        frames = []
        duration = None
        fps = None
        resolution = None

        try:
            # Get video info via ffprobe
            probe_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(resolved)]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            if probe_result.returncode == 0:
                probe_data = json.loads(probe_result.stdout)
                fmt = probe_data.get("format", {})
                duration = float(fmt.get("duration", 0))

                for stream in probe_data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        resolution = [
                            int(stream.get("width", 0)),
                            int(stream.get("height", 0)),
                        ]
                        r_frame_rate = stream.get("r_frame_rate", "0/1")
                        num, den = r_frame_rate.split("/")
                        fps = float(num) / float(den) if float(den) > 0 else 0
                        break

            # Extract frames
            if duration and duration > 0:
                for i in range(frame_count):
                    seek_time = duration * i / max(frame_count, 1)
                    frame_cmd = [
                        "ffmpeg",
                        "-ss",
                        str(seek_time),
                        "-i",
                        str(resolved),
                        "-vframes",
                        "1",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "mjpeg",
                        "-q:v",
                        "5",
                        "pipe:1",
                    ]
                    result = subprocess.run(frame_cmd, capture_output=True, timeout=10)
                    if result.returncode == 0 and result.stdout:
                        b64 = base64.b64encode(result.stdout).decode("utf-8")
                        frames.append(f"data:image/jpeg;base64,{b64}")
        except FileNotFoundError:
            logger.warning("ffmpeg/ffprobe not found; cannot generate video preview")
        except subprocess.TimeoutExpired:
            logger.warning("Video preview timed out for %s", file_path)
        except Exception as e:
            logger.warning("Error generating video preview for %s: %s", file_path, e)

        return VideoPreview(
            path=file_path,
            frames=frames,
            duration=duration,
            fps=fps,
            resolution=resolution,
        )

    def get_audio_preview(self, dataset_config: Dict[str, Any], file_path: str) -> AudioPreview:
        """Generate a waveform PNG for an audio file."""
        resolved = self._resolve_image_path(dataset_config, file_path)
        if resolved is None:
            return AudioPreview(path=file_path)

        duration = None
        sample_rate = None
        channels = None
        waveform = None

        try:
            import numpy as np

            # Try torchaudio first, fall back to soundfile
            audio_data = None
            sr = None

            try:
                import torchaudio

                waveform_tensor, sr = torchaudio.load(str(resolved))
                audio_data = waveform_tensor.numpy()
                channels = audio_data.shape[0]
            except (ImportError, Exception):
                try:
                    import soundfile as sf

                    data, sr = sf.read(str(resolved))
                    if data.ndim == 1:
                        audio_data = data.reshape(1, -1)
                    else:
                        audio_data = data.T
                    channels = audio_data.shape[0]
                except ImportError:
                    pass

            if audio_data is not None and sr:
                sample_rate = sr
                duration = audio_data.shape[1] / sr

                # Generate waveform image
                mono = audio_data.mean(axis=0) if channels > 1 else audio_data[0]

                # Downsample for visualization
                target_width = 800
                chunk_size = max(1, len(mono) // target_width)
                chunks = mono[: chunk_size * target_width].reshape(-1, chunk_size)
                envelope_max = chunks.max(axis=1)
                envelope_min = chunks.min(axis=1)

                height = 120
                img = Image.new("RGB", (target_width, height), (15, 23, 42))
                pixels = img.load()

                for x in range(target_width):
                    y_max = int((1 - envelope_max[x]) * height / 2)
                    y_min = int((1 - envelope_min[x]) * height / 2)
                    y_max = max(0, min(height - 1, y_max))
                    y_min = max(0, min(height - 1, y_min))
                    for y in range(min(y_max, y_min), max(y_max, y_min) + 1):
                        pixels[x, y] = (56, 189, 248)

                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)
                b64 = base64.b64encode(buffer.read()).decode("utf-8")
                waveform = f"data:image/png;base64,{b64}"

        except Exception as e:
            logger.warning("Error generating audio preview for %s: %s", file_path, e)

        return AudioPreview(
            path=file_path,
            waveform=waveform,
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
        )
