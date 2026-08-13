import json
import logging
import os
import random
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import requests
import torch

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.data_backend.dataset_types import DatasetType, ensure_dataset_type
from simpletuner.helpers.image_manipulation.load import load_image, load_video
from simpletuner.helpers.training import video_file_extensions
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger("WebshartDataBackend")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


@dataclass(frozen=True)
class WebshartSampleRef:
    shard_idx: int
    sample_idx: int
    filename: str


class WebshartDataBackend(BaseDataBackend):
    SAMPLE_PREFIX = "webshart://"
    PATH_NORMALIZED_SAMPLE_PREFIX = "webshart:/"
    CACHE_EXTENSIONS = {".json", ".pt", ".msgpack", ".safetensors"}

    def __init__(
        self,
        accelerator,
        id: str,
        source: str,
        metadata: Optional[str] = None,
        hf_token: Optional[str] = None,
        subfolder: Optional[str] = None,
        cache_dir: Optional[str] = None,
        metadata_cache_dir: Optional[str] = None,
        shard_cache_dir: Optional[str] = None,
        shard_cache_gb: float = 25.0,
        parallel_downloads: int = 4,
        buffer_size: int = 100,
        max_file_size: int = 500 * 1024 * 1024,
        compress_cache: bool = False,
        dataset_type: Union[str, DatasetType] = DatasetType.IMAGE,
        optimize_captions: bool = False,
    ):
        if not source:
            raise ValueError("source is required for Webshart data backends.")

        try:
            import webshart
        except ImportError as exc:
            raise ImportError("Webshart data backends require the 'webshart' package to be installed.") from exc

        self.webshart = webshart
        self.accelerator = accelerator
        self.id = id
        self.type = "webshart"
        self.source = str(source)
        self.metadata = str(metadata) if metadata else None
        self.hf_token = hf_token
        self.subfolder = subfolder
        self.cache_dir = str(cache_dir) if cache_dir else str(Path("cache") / "webshart" / id)
        self.metadata_cache_dir = (
            str(metadata_cache_dir) if metadata_cache_dir else str(Path(self.cache_dir) / "metadata_cache")
        )
        self.shard_cache_gb = float(shard_cache_gb)
        if self.shard_cache_gb < 0:
            raise ValueError("shard_cache_gb must be non-negative; use 0 to disable whole-shard caching.")
        self.shard_cache_dir = str(shard_cache_dir) if shard_cache_dir else str(Path(self.cache_dir) / "shard_cache")
        if self.shard_cache_gb == 0:
            self.shard_cache_dir = None
        self.parallel_downloads = int(parallel_downloads)
        self.buffer_size = int(buffer_size)
        self.max_file_size = int(max_file_size)
        self.compress_cache = compress_cache
        self.dataset_type = ensure_dataset_type(dataset_type, default=DatasetType.IMAGE)
        self.optimize_captions = bool(optimize_captions)
        self._shard_sample_index_cache: dict[int, dict[str, int]] = {}

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.metadata_cache_dir).mkdir(parents=True, exist_ok=True)

        self.dataset = self.webshart.discover_dataset(
            source=self.source,
            hf_token=self.hf_token,
            subfolder=self.subfolder,
            metadata=self.metadata,
        )
        self.dataset.enable_metadata_cache(location=self.metadata_cache_dir)
        if self.shard_cache_dir is not None:
            Path(self.shard_cache_dir).mkdir(parents=True, exist_ok=True)
            self.dataset.enable_shard_cache(
                location=self.shard_cache_dir,
                cache_limit_gb=self.shard_cache_gb,
                parallel_downloads=self.parallel_downloads,
            )
        self.loader = self.webshart.TarDataLoader(
            self.dataset,
            buffer_size=self.buffer_size,
            max_file_size=self.max_file_size,
            load_file_data=True,
        )
        if not hasattr(self.loader, "list_shard_sample_aspect_buckets"):
            raise ImportError(
                "SimpleTuner's Webshart backend requires a webshart build that provides "
                "TarDataLoader.list_shard_sample_aspect_buckets()."
            )
        if self.optimize_captions:
            self._optimize_caption_metadata()

    def _optimize_caption_metadata(self) -> None:
        """Fold sidecar captions into the local metadata cache via webshart's coalescer.

        Sidecar-caption datasets (e.g. plain webdataset ``.txt`` members) otherwise cost one
        range-read per sample every time captions are enumerated; after coalescing, the local
        metadata cache serves them as embedded captions.
        """
        probe = getattr(self.dataset, "probe_caption_layout", None)
        coalesce = getattr(self.loader, "coalesce_caption_metadata", None)
        if not callable(probe) or not callable(coalesce):
            raise ImportError(
                "webshart_optimize_captions requires a webshart build that provides "
                "DiscoveredDataset.probe_caption_layout() and TarDataLoader.coalesce_caption_metadata()."
            )

        is_main_process = self.accelerator is None or getattr(self.accelerator, "is_main_process", True)
        if is_main_process:
            layout = str(probe(max_shards=8).get("layout"))
            if layout in ("json_sidecar", "txt_sidecar", "mixed"):
                logger.info(
                    "(id=%s) Coalescing %s captions into the webshart metadata cache...",
                    self.id,
                    layout,
                )
                result = coalesce()
                logger.info(
                    "(id=%s) Coalesced %s captions across %s shards.",
                    self.id,
                    result.get("coalesced_samples"),
                    result.get("shards"),
                )
            else:
                logger.info("(id=%s) Caption layout is '%s'; no coalescing needed.", self.id, layout)
        if self.accelerator is not None and hasattr(self.accelerator, "wait_for_everyone"):
            self.accelerator.wait_for_everyone()

    @classmethod
    def sample_id(cls, shard_idx: int, sample_idx: int, filename: str) -> str:
        return f"{cls.SAMPLE_PREFIX}{int(shard_idx)}/{int(sample_idx)}/{filename}"

    @classmethod
    def normalize_sample_id(cls, identifier: Union[str, Path]) -> str:
        value = str(identifier)
        marker = value.find(cls.SAMPLE_PREFIX)
        if marker >= 0:
            return value[marker:]
        marker = value.find(cls.PATH_NORMALIZED_SAMPLE_PREFIX)
        if marker >= 0:
            remainder = value[marker + len(cls.PATH_NORMALIZED_SAMPLE_PREFIX) :]
            return f"{cls.SAMPLE_PREFIX}{remainder}"
        return value

    @classmethod
    def parse_sample_id(cls, identifier: Union[str, Path]) -> WebshartSampleRef:
        value = cls.normalize_sample_id(identifier)
        if not value.startswith(cls.SAMPLE_PREFIX):
            raise ValueError(f"Invalid webshart sample id: {identifier}")
        remainder = value[len(cls.SAMPLE_PREFIX) :]
        parts = remainder.split("/", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid webshart sample id: {identifier}")
        try:
            shard_idx = int(parts[0])
            sample_idx = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid webshart sample id: {identifier}") from exc
        return WebshartSampleRef(shard_idx=shard_idx, sample_idx=sample_idx, filename=parts[2])

    @classmethod
    def is_sample_id(cls, identifier: Union[str, Path]) -> bool:
        sample_id = cls.normalize_sample_id(identifier)
        if not sample_id.startswith(cls.SAMPLE_PREFIX):
            return False
        filename = sample_id.split("/", 2)[-1] if "/" in sample_id else sample_id
        return Path(filename).suffix.lower() not in {".pt", ".safetensors"}

    @classmethod
    def _normalise_file_extensions(cls, file_extensions: Optional[list]) -> set[str]:
        return {
            extension.lower() if str(extension).startswith(".") else f".{str(extension).lower()}"
            for extension in file_extensions or []
        }

    def _list_cache_files(
        self,
        file_extensions: Optional[list] = None,
        instance_data_dir: str = None,
    ) -> List[Tuple[str, List, List[str]]]:
        root = Path(instance_data_dir) if instance_data_dir else Path(self.cache_dir)
        if not root.exists():
            return []

        wanted_extensions = self._normalise_file_extensions(file_extensions)
        results = []
        for current_root, dirs, files in os.walk(root):
            matched_files = []
            for filename in files:
                if wanted_extensions and Path(filename).suffix.lower() not in wanted_extensions:
                    continue
                matched_files.append(filename)
            results.append((current_root, dirs, matched_files))
        return results

    def _cache_path(self, identifier: Union[str, Path]) -> Path:
        path = Path(identifier)
        if path.is_absolute() or str(path.parent) not in ("", "."):
            return path

        suffix = path.suffix.lower()
        if suffix == ".json":
            return Path(self.cache_dir) / "webshart_metadata" / self.id / path.name
        if suffix in {".pt", ".safetensors"}:
            return Path(self.cache_dir) / "vae" / self.id / path.name
        return Path(self.cache_dir) / "cache" / self.id / path.name

    def _is_cache_identifier(self, identifier: Union[str, Path]) -> bool:
        return Path(identifier).suffix.lower() in self.CACHE_EXTENSIONS

    def _read_sample_bytes(self, identifier: Union[str, Path]) -> bytes:
        sample_ref = self.parse_sample_id(identifier)
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                entry = self.loader.load_sample(sample_ref.shard_idx, sample_ref.sample_idx)
                return bytes(entry.data)
            except Exception as exc:
                message = str(exc).lower()
                retryable = any(
                    marker in message
                    for marker in (
                        "rate limit",
                        "http 429",
                        "status code 429",
                        "connection reset",
                        "temporarily unavailable",
                        "timed out",
                        "timeout",
                    )
                )
                if not retryable or attempt + 1 >= max_attempts:
                    raise
                delay = min(30.0, 2.0**attempt) + random.uniform(0.0, 1.0)
                logger.warning(
                    "Transient error reading Webshart sample %s; retrying in %.1fs (%d/%d): %s",
                    identifier,
                    delay,
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                time.sleep(delay)

    def read_sample_head_tail(
        self,
        identifier: Union[str, Path],
        *,
        file_metadata: Optional[dict] = None,
        head_bytes: int = 4096,
        tail_bytes: int = 131072,
    ) -> tuple[bytes, bytes, int]:
        """Range-read the beginning and end of a sample without loading its full TAR member."""
        sample_ref = self.parse_sample_id(identifier)
        metadata = file_metadata or self.get_shard_metadata(sample_ref.shard_idx).get(sample_ref.filename, {})
        offset = metadata.get("offset")
        length = metadata.get("length", metadata.get("size"))
        shard_info = self.dataset.get_shard_info(sample_ref.shard_idx)
        tar_path = shard_info.get("tar_path") if isinstance(shard_info, dict) else None
        if offset is None or length is None or not str(tar_path or "").startswith(("http://", "https://")):
            raise ValueError(f"Range metadata is unavailable for Webshart sample {identifier}.")

        offset = int(offset)
        length = int(length)
        if length <= 0:
            raise ValueError(f"Invalid Webshart sample length for {identifier}: {length}")

        token = self.hf_token
        if token is True:
            from huggingface_hub import get_token

            token = get_token()
        base_headers = {"Authorization": f"Bearer {token}"} if token else {}

        def _read_range(relative_start: int, relative_end: int) -> bytes:
            absolute_start = offset + relative_start
            absolute_end = offset + relative_end
            headers = {**base_headers, "Range": f"bytes={absolute_start}-{absolute_end}"}
            response = requests.get(str(tar_path), headers=headers, stream=True, timeout=(10, 60))
            try:
                if response.status_code != 206:
                    raise IOError(f"Range request for {identifier} returned HTTP {response.status_code} instead of 206.")
                payload = response.content
            finally:
                response.close()
            expected_size = relative_end - relative_start + 1
            if len(payload) != expected_size:
                raise IOError(f"Range request for {identifier} returned {len(payload)} bytes; expected {expected_size}.")
            return payload

        head_size = min(max(1, int(head_bytes)), length)
        tail_size = min(max(1, int(tail_bytes)), length)
        head = _read_range(0, head_size - 1)
        if tail_size == length:
            tail = head if head_size == length else _read_range(0, length - 1)
        else:
            tail = _read_range(length - tail_size, length - 1)
        return head, tail, length

    def _sample_index_for_filename(self, shard_idx: int, filename: str) -> Optional[int]:
        shard_idx = int(shard_idx)
        if shard_idx not in self._shard_sample_index_cache:
            self._shard_sample_index_cache[shard_idx] = {
                str(sample_filename): sample_idx
                for sample_idx, sample_filename in enumerate(self.dataset.list_samples_in_shard(shard_idx))
            }
        return self._shard_sample_index_cache[shard_idx].get(str(filename))

    def get_caption(self, image_path: str) -> Optional[str]:
        if not self.is_sample_id(image_path):
            return None

        sample_ref = self.parse_sample_id(image_path)
        sample_metadata = self.get_shard_metadata(sample_ref.shard_idx).get(sample_ref.filename, {}) or {}
        caption = sample_metadata.get("captions")
        if caption:
            return str(caption).strip()

        caption_filename = Path(sample_ref.filename).with_suffix(".txt").name
        caption_sample_idx = self._sample_index_for_filename(sample_ref.shard_idx, caption_filename)
        if caption_sample_idx is None:
            return None

        caption_sample_id = self.sample_id(sample_ref.shard_idx, caption_sample_idx, caption_filename)
        caption = self.read(caption_sample_id)
        if isinstance(caption, bytes):
            caption = caption.decode("utf-8")
        return str(caption).strip()

    def read(self, identifier: Union[str, Path], as_byteIO: bool = False) -> Any:
        if self.is_sample_id(identifier):
            identifier = self.normalize_sample_id(identifier)
            data = self._read_sample_bytes(identifier)
            return BytesIO(data) if as_byteIO else data

        cache_path = self._cache_path(identifier)
        with cache_path.open("rb") as handle:
            data = handle.read()
        return BytesIO(data) if as_byteIO else data

    def write(self, identifier: Union[str, Path], data: Any) -> None:
        if self.is_sample_id(identifier):
            raise NotImplementedError("Webshart datasets are read-only.")

        cache_path = self._cache_path(identifier)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.suffix.lower() == ".json":
            if isinstance(data, (dict, list)):
                payload = json.dumps(data)
            elif isinstance(data, bytes):
                payload = data.decode("utf-8")
            else:
                payload = str(data)
            cache_path.write_text(payload, encoding="utf-8")
            return

        with cache_path.open("wb") as handle:
            if isinstance(data, bytes):
                handle.write(data)
            else:
                torch.save(data, handle)

    def delete(self, identifier: Union[str, Path]) -> None:
        if self.is_sample_id(identifier):
            raise NotImplementedError("Webshart datasets are read-only.")
        self._cache_path(identifier).unlink()

    def exists(self, identifier: Union[str, Path]) -> bool:
        if self.is_sample_id(identifier):
            try:
                identifier = self.normalize_sample_id(identifier)
                sample_ref = self.parse_sample_id(identifier)
                shard_info = self.dataset.get_shard_info(sample_ref.shard_idx)
                num_samples = shard_info.get("num_samples")
                return num_samples is None or 0 <= sample_ref.sample_idx < int(num_samples)
            except Exception:
                return False
        return self._cache_path(identifier).exists()

    def open_file(self, identifier: Union[str, Path], mode: str):
        if "w" in mode or "a" in mode:
            raise NotImplementedError("Webshart data backend does not support open_file writes.")
        return BytesIO(self.read(identifier))

    def list_files(self, file_extensions: list = None, instance_data_dir: str = None) -> List[Tuple[str, List, List[str]]]:
        requested_extensions = self._normalise_file_extensions(file_extensions)
        if requested_extensions and requested_extensions.issubset(self.CACHE_EXTENSIONS):
            return self._list_cache_files(file_extensions=file_extensions, instance_data_dir=instance_data_dir)

        files = []
        for shard_idx in range(self.num_shards()):
            for sample_idx, filename in enumerate(self.dataset.list_samples_in_shard(shard_idx)):
                if file_extensions:
                    ext = os.path.splitext(filename)[1].lower().strip(".")
                    if ext not in file_extensions:
                        continue
                files.append(self.sample_id(shard_idx, sample_idx, filename))
        return [("", [], files)]

    def get_abs_path(self, sample_path: str = None) -> Optional[str]:
        if sample_path is None:
            return None
        if self.is_sample_id(sample_path) and self.exists(sample_path):
            return self.normalize_sample_id(sample_path)
        cache_path = self._cache_path(sample_path)
        return str(cache_path) if cache_path.exists() else None

    def read_image(self, filepath: str, delete_problematic_images: bool = False):
        try:
            file_extension = Path(self.normalize_sample_id(filepath)).suffix.lower().strip(".")
            loader = load_video if file_extension in video_file_extensions else load_image
            return loader(self.read(filepath, as_byteIO=True))
        except Exception as exc:
            logger.error("Error opening webshart sample %s: %s", filepath, exc)
            if delete_problematic_images:
                logger.warning("Cannot delete from webshart dataset - skipping problematic image")
            return None

    def read_image_batch(self, filepaths: list, delete_problematic_images: bool = False):
        available_keys = []
        output_images = []
        for filepath in filepaths:
            image = self.read_image(filepath, delete_problematic_images=delete_problematic_images)
            if image is None:
                logger.warning("Unable to load webshart sample '%s', skipping.", filepath)
                continue
            available_keys.append(filepath)
            output_images.append(image)
        return available_keys, output_images

    def create_directory(self, directory_path):
        Path(directory_path).mkdir(parents=True, exist_ok=True)

    def torch_load(self, filename):
        data = self.read(filename, as_byteIO=True)
        if self.compress_cache:
            data = self._decompress_torch(data)
        data.seek(0)
        return torch.load(data, map_location="cpu")

    def torch_save(self, data, filename):
        if self.compress_cache:
            data = self._compress_torch(data)
        self.write(filename, data)

    def write_batch(self, identifiers, files):
        for identifier, data in zip(identifiers, files):
            self.write(identifier, data)

    def get_instance_representation(self) -> dict:
        return {
            "backend_type": "webshart",
            "id": self.id,
            "source": self.source,
            "metadata": self.metadata,
            "hf_token": self.hf_token,
            "subfolder": self.subfolder,
            "cache_dir": self.cache_dir,
            "metadata_cache_dir": self.metadata_cache_dir,
            "shard_cache_dir": self.shard_cache_dir,
            "shard_cache_gb": self.shard_cache_gb,
            "parallel_downloads": self.parallel_downloads,
            "buffer_size": self.buffer_size,
            "max_file_size": self.max_file_size,
            "compress_cache": self.compress_cache,
            "dataset_type": self.dataset_type.value,
        }

    @staticmethod
    def from_instance_representation(representation: dict) -> "WebshartDataBackend":
        if representation.get("backend_type") != "webshart":
            raise ValueError(f"Expected backend_type 'webshart', got {representation.get('backend_type')}")
        return WebshartDataBackend(
            accelerator=None,
            id=representation["id"],
            source=representation["source"],
            metadata=representation.get("metadata"),
            hf_token=representation.get("hf_token"),
            subfolder=representation.get("subfolder"),
            cache_dir=representation.get("cache_dir"),
            metadata_cache_dir=representation.get("metadata_cache_dir"),
            shard_cache_dir=representation.get("shard_cache_dir"),
            shard_cache_gb=representation.get("shard_cache_gb", 25.0),
            parallel_downloads=representation.get("parallel_downloads", 4),
            buffer_size=representation.get("buffer_size", 100),
            max_file_size=representation.get("max_file_size", 500 * 1024 * 1024),
            compress_cache=representation.get("compress_cache", False),
            dataset_type=representation.get("dataset_type", DatasetType.IMAGE),
        )

    def num_shards(self) -> int:
        value = getattr(self.dataset, "num_shards", None)
        if callable(value):
            return int(value())
        return int(value)

    def get_shard_metadata(self, shard_idx: int) -> dict:
        return dict(self.loader.get_metadata(shard_idx))

    def list_shard_sample_aspect_buckets(
        self,
        shard_indices: list[int],
        key: str = "aspect",
        target_pixel_area: Optional[int] = None,
        target_resolution_multiple: int = 64,
        round_to: Optional[int] = 2,
    ) -> list[dict]:
        return self.loader.list_shard_sample_aspect_buckets(
            shard_indices,
            key=key,
            target_pixel_area=target_pixel_area,
            target_resolution_multiple=target_resolution_multiple,
            round_to=round_to,
        )
