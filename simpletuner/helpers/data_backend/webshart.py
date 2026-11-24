import json
import logging
import os
import random
import time
from copy import copy
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import requests
import torch
from atomicwrites import atomic_write
from PIL import Image

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.data_backend.config.validators import validate_webshart_caption_key
from simpletuner.helpers.data_backend.dataset_types import DatasetType, ensure_dataset_type
from simpletuner.helpers.data_backend.filters import DatasetFilter
from simpletuner.helpers.data_backend.s3_settings import resolve_s3_options
from simpletuner.helpers.image_manipulation.load import load_image, load_video
from simpletuner.helpers.prompts import PromptHandler
from simpletuner.helpers.training import video_file_extensions
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger("WebshartDataBackend")
CaptionValue = Union[str, dict, List[Union[str, dict]]]
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
    prepared_partition_shards = 64
    format = "tar"
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
        caption_key: Optional[Union[str, List[str]]] = None,
        format: str = "tar",
        manifest: Optional[str] = None,
        columns: Optional[dict] = None,
        aws_region_name: Optional[str] = None,
        aws_endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_max_pool_connections: int = 32,
        shuffle_seed: Optional[int] = None,
    ):
        if not source:
            raise ValueError("source is required for Webshart data backends.")
        validate_webshart_caption_key(caption_key)
        self.caption_key = caption_key
        if format not in {"tar", "parquet"}:
            raise ValueError("Webshart format must be 'tar' or 'parquet'.")
        if str(source).startswith("s3://") and format != "parquet":
            raise ValueError("S3 Webshart sources require format='parquet'.")
        if format == "tar" and (manifest or any(v not in (None, "") for v in (columns or {}).values())):
            raise ValueError("Webshart manifest and columns require format='parquet'.")
        self.shuffle_seed = shuffle_seed
        self.format = format
        self.manifest = manifest or None
        self.columns = {k: v for k, v in (columns or {}).items() if v not in (None, "")}
        allowed_columns = {"image", "captions", "width", "height"}
        if set(self.columns) - allowed_columns or any(not isinstance(v, str) or not v for v in self.columns.values()):
            raise ValueError("Webshart columns must map image, captions, width, or height to nonempty column names.")
        self.aws_region_name = aws_region_name
        self.aws_endpoint_url = aws_endpoint_url
        self.aws_max_pool_connections = aws_max_pool_connections

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
        elif (
            shuffle_seed is not None
            and accelerator is not None
            and accelerator.num_processes > 1
            and not Path(source).is_dir()
        ):
            # Webshart permits one active prefill order per cache directory.
            self.shard_cache_dir = str(Path(self.shard_cache_dir) / f"rank-{accelerator.process_index}")
        self.parallel_downloads = int(parallel_downloads)
        self.buffer_size = int(buffer_size)
        self.max_file_size = int(max_file_size)
        self.compress_cache = compress_cache
        self.dataset_type = ensure_dataset_type(dataset_type, default=DatasetType.IMAGE)
        self.optimize_captions = bool(optimize_captions)
        self._shard_metadata_cache: dict[int, dict] = {}
        self._shard_metadata_cache_limit = 8

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.metadata_cache_dir).mkdir(parents=True, exist_ok=True)

        discovery = {}
        if self.format == "parquet":
            # Each retained payload reader locks its shard against shared-cache eviction.
            cache_remote_shards = self.shard_cache_dir is not None and not Path(self.source).is_dir()
            discovery = {
                "format": "parquet",
                "manifest": self.manifest,
                "metadata_cache_size": 1 if cache_remote_shards else self._shard_metadata_cache_limit,
            }
            if (
                self.manifest is not None
                and getattr(self.webshart.ParquetDataset, "supports_catalogue_cache", False) is True
            ):
                discovery["catalogue_cache_dir"] = self.metadata_cache_dir
            discovery.update({("caption_column" if k == "captions" else f"{k}_column"): v for k, v in self.columns.items()})
            if self.source.startswith("s3://"):
                s3_options = {
                    "region_name": aws_region_name,
                    "endpoint_url": aws_endpoint_url,
                    "aws_access_key_id": aws_access_key_id,
                    "aws_secret_access_key": aws_secret_access_key,
                    "aws_session_token": aws_session_token,
                    "max_connections": aws_max_pool_connections,
                }
                discovery["s3_options"] = resolve_s3_options(
                    {key: value for key, value in s3_options.items() if value is not None}
                )
        self._discovery_options = discovery
        self.dataset = self._open_source()
        if self.format == "parquet":
            self.loader = self.webshart.ParquetDataLoader(self.dataset)
        else:
            self.loader = self.webshart.TarDataLoader(
                self.dataset,
                buffer_size=self.buffer_size,
                max_file_size=self.max_file_size,
                load_file_data=True,
            )
        if self.caption_key is not None and self.format == "tar":
            self.caption_loader = self.webshart.TarDataLoader(
                self.dataset,
                buffer_size=self.buffer_size,
                max_file_size=self.max_file_size,
                load_file_data=False,
            )
        if not hasattr(self.loader, "list_shard_sample_aspect_buckets"):
            loader_name = "ParquetDataLoader" if self.format == "parquet" else "TarDataLoader"
            raise ImportError(
                "SimpleTuner's Webshart backend requires a webshart build that provides "
                f"{loader_name}.list_shard_sample_aspect_buckets()."
            )
        if self.optimize_captions and self.format == "tar":
            self._optimize_caption_metadata()

    def _open_source(self):
        dataset = self.webshart.discover_dataset(
            source=self.source,
            hf_token=self.hf_token,
            subfolder=self.subfolder,
            metadata=self.metadata,
            **self._discovery_options,
        )
        if self.format == "tar":
            dataset.enable_metadata_cache(location=self.metadata_cache_dir)
        if self.shard_cache_dir is not None:
            Path(self.shard_cache_dir).mkdir(parents=True, exist_ok=True)
            dataset.enable_shard_cache(
                location=self.shard_cache_dir,
                cache_limit_gb=self.shard_cache_gb,
                parallel_downloads=self.parallel_downloads,
            )
        return dataset

    def seeded_shards(self, seed):
        try:
            from webshart.parquet import _shuffled_shard_order
        except ImportError as exc:
            raise ImportError("webshart.shuffle_seed requires Webshart 0.9.3 or newer.") from exc
        prefill = self.sequential_prefill(1, shuffle_seed=seed)
        if prefill is not None:
            return prefill.shard_at, prefill
        return _shuffled_shard_order(self.num_shards(), seed).__getitem__, None

    def sequential_prefill(self, window_shards, *, shuffle_seed=None):
        # Webshart releases before the byte-bounded cursor use on-demand caching.
        if not hasattr(self.dataset, "sequential_prefill") or self.shard_cache_dir is None or Path(self.source).is_dir():
            return None
        options = {"active_shards": window_shards}
        if shuffle_seed is not None:
            options["shuffle_seed"] = shuffle_seed
        downloads = os.environ.get("SIMPLETUNER_WEBSHART_PREFILL_DOWNLOADS")
        if downloads is None:
            return self.dataset.sequential_prefill(**options)
        downloads = int(downloads)
        if downloads < 1:
            raise ValueError("SIMPLETUNER_WEBSHART_PREFILL_DOWNLOADS must be positive.")
        # Keep foreground I/O and the checkpointed dataset configuration independent.
        reader = self.open_refresh_source().dataset
        reader.enable_shard_cache(self.shard_cache_dir, self.shard_cache_gb, downloads)
        return reader.sequential_prefill(**options)

    def open_refresh_source(self):
        """An unpinned Parquet reader using the same transport and cache settings."""
        fresh = copy(self)
        fresh._shard_metadata_cache = {}
        fresh.dataset = fresh._open_source()
        if fresh.dataset.fingerprint != self.dataset.fingerprint:
            raise ValueError("The Webshart catalogue changed. Reopen the dataset before preparing the new catalogue.")
        fresh.loader = self.webshart.ParquetDataLoader(fresh.dataset)
        return fresh

    def adopt_refresh_source(self, fresh):
        self.dataset, self.loader = fresh.dataset, fresh.loader
        self._shard_metadata_cache.clear()

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

    def iter_cache_groups(self, bucket_files: dict):
        """Finish each shard's aspect groups before loading another shard."""
        shard_buckets = {}
        for bucket, files in bucket_files.items():
            for filepath in files:
                shard = self._resolve_sample_ref(filepath).shard_idx
                shard_buckets.setdefault(shard, {}).setdefault(bucket, []).append(filepath)
        for shard in sorted(shard_buckets):
            yield from shard_buckets[shard].items()

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

    def shard_of(self, identifier):
        return self.parse_sample_id(identifier).shard_idx

    def parquet_sample_refs_for_filename(self, filename, *, first_frame=False):
        """Resolve the Parquet source identities represented by a paired filename."""
        shard, separator, basename = filename.rpartition("/")
        row, dot, extension = basename.partition(".")
        if not separator or not dot or len(row) != 12 or not row.isascii() or not row.isdigit():
            return ()
        position = self.dataset.get_shard_position(shard)
        if position is None:
            return ()
        extensions = ("jpg", "jxl") if first_frame and extension == "png" else (extension,)
        return tuple(self.sample_id(position, int(row), f"{shard}/{row}.{suffix}") for suffix in extensions)

    def conditioning_ref(self, pairing_key, *, admits):
        if self.format == "parquet":
            filename = self.parse_sample_id(pairing_key).filename if self.is_sample_id(pairing_key) else pairing_key
            return next((ref for ref in self.parquet_sample_refs_for_filename(filename) if admits(ref)), None)
        self._resolve_sample_ref(pairing_key)
        ref = self.normalize_sample_id(pairing_key)
        return ref if admits(ref) else None

    def _resolve_sample_ref(self, identifier: Union[str, Path]) -> WebshartSampleRef:
        ref = self.parse_sample_id(identifier)
        if self.format == "parquet":
            shard = self.dataset.get_shard_info(ref.shard_idx)
            prefix, _, basename = ref.filename.rpartition("/")
            if prefix != shard["path"] or basename.rsplit(".", 1)[0] != f"{ref.sample_idx:012d}":
                raise ValueError("Parquet sample identity does not match the catalog; rebuild metadata for this snapshot.")
        return ref

    @classmethod
    def is_sample_id(cls, identifier: Union[str, Path]) -> bool:
        sample_id = cls.normalize_sample_id(identifier)
        if not sample_id.startswith(cls.SAMPLE_PREFIX):
            return False
        return cls.filename_suffix(sample_id) not in {".pt", ".safetensors"}

    @staticmethod
    def filename_suffix(filename: str) -> str:
        name = next((part for part in reversed(filename.split("/")) if part and part != "."), "")
        suffix_start = name.rfind(".")
        return name[suffix_start:].lower() if 0 < suffix_start < len(name) - 1 else ""

    @staticmethod
    def caption_sidecar_filename(filename: str) -> str:
        basename_start = filename.rfind("/") + 1
        suffix_start = filename.rfind(".")
        stem = filename[:suffix_start] if basename_start < suffix_start < len(filename) - 1 else filename
        return f"{stem}.txt"

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
        return self.filename_suffix(str(identifier)) in self.CACHE_EXTENSIONS

    def _read_sample_bytes(self, identifier: Union[str, Path]) -> bytes:
        sample_ref = self._resolve_sample_ref(identifier)
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
        sample_ref = self._resolve_sample_ref(identifier)
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

    def get_sample_metadata(self, image_path: str) -> dict:
        sample_ref = self._resolve_sample_ref(image_path)
        sample_metadata = self.get_shard_metadata(sample_ref.shard_idx).get(sample_ref.filename, {}) or {}
        if (
            self.caption_key is not None
            and sample_metadata.get("json_metadata") is None
            and sample_metadata.get("json_path")
        ):
            payload = self.caption_loader.load_sample_json(sample_ref.shard_idx, sample_ref.sample_idx)
            if payload is not None:
                sample_metadata["json_metadata"] = json.loads(payload)
        return sample_metadata

    @property
    def supports_caption_projection(self):
        return self.format == "parquet" and hasattr(self.dataset, "read_metadata_rows")

    @property
    def caption_projection_strategy(self):
        return "webshart" if self.supports_caption_projection else None

    def caption_fields(self):
        fields = {"captions", "url_status", "stored_bytes", "stored_format"}
        if self.caption_key is not None:
            fields.add("json_metadata")
            fields.update([self.caption_key] if isinstance(self.caption_key, str) else self.caption_key)
        return fields

    def read_caption_batch(self, paths):
        """Read only the fields needed by the configured Parquet caption strategy."""
        groups = {}
        for position, path in enumerate(paths):
            ref = self._resolve_sample_ref(path)
            groups.setdefault(ref.shard_idx, []).append((position, ref))
        result = [None] * len(paths)
        fields = self.caption_fields()
        for shard_idx, selected in groups.items():
            rows = self.dataset.read_metadata_rows(shard_idx, [ref.sample_idx for _, ref in selected], fields=fields)
            for (position, ref), row in zip(selected, rows, strict=True):
                if row.get("url_status", "ok") != "ok" or ("stored_bytes" in row and not row["stored_bytes"]):
                    continue
                suffix = ".jxl" if row.get("stored_format") == "jxl" else ".jpg"
                if self.filename_suffix(ref.filename) == suffix:
                    row.update(sample_idx=ref.sample_idx, shard_idx=shard_idx, filename=ref.filename)
                    result[position] = self._caption_from_metadata(row)
        return result

    def _caption_from_metadata(self, metadata):
        if self.caption_key is not None:
            return self._select_caption_keys(metadata)
        caption = metadata.get("captions")
        if caption:
            values = PromptHandler._caption_payload_values(caption)
            return values if isinstance(caption, list) else values[0] if values else None
        return None

    def get_caption(self, image_path: str) -> Optional[CaptionValue]:
        if not self.is_sample_id(image_path):
            return None
        if self.supports_caption_projection:
            return self.read_caption_batch([image_path])[0]
        caption = self._caption_from_metadata(self.get_sample_metadata(image_path))
        if caption is not None or self.caption_key is not None or self.format == "parquet":
            return caption
        sample_ref = self._resolve_sample_ref(image_path)
        caption_filename = self.caption_sidecar_filename(sample_ref.filename)
        if caption_filename not in self.get_shard_metadata(sample_ref.shard_idx):
            return None
        caption = self.loader.load_caption(sample_ref.shard_idx, sample_ref.sample_idx)
        return str(caption).strip() if caption is not None else None

    def _select_caption_keys(self, sample_metadata: dict) -> Optional[CaptionValue]:
        json_metadata = sample_metadata.get("json_metadata") or {}
        sources = [json_metadata, sample_metadata]
        sources.extend(source.get("captions") for source in list(sources) if isinstance(source, dict))
        keys = [self.caption_key] if isinstance(self.caption_key, str) else self.caption_key
        captions = []
        for key in keys:
            for source in sources:
                if isinstance(source, dict) and key in source:
                    captions.extend(PromptHandler._caption_payload_values(source[key]))
                    break
        if not captions:
            return None
        if isinstance(self.caption_key, str) and len(captions) == 1:
            return captions[0]
        return captions

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
            with atomic_write(cache_path, mode="w", encoding="utf-8", overwrite=True) as handle:
                if isinstance(data, (dict, list)):
                    json.dump(data, handle, default=os.fspath)
                else:
                    handle.write(data.decode("utf-8") if isinstance(data, bytes) else str(data))
            return

        with atomic_write(cache_path, mode="wb", encoding=None, overwrite=True) as handle:
            if isinstance(data, bytes):
                handle.write(data)
            else:
                torch.save(data, handle)

    def write_membership(self, identifier, buckets, properties, *, shard_of=None):
        from simpletuner.helpers.metadata.backends.webshart_membership import write_membership

        write_membership(self._cache_path(identifier), buckets, properties, shard_of=shard_of)

    def write_geometry(self, identifier, records):
        path = self._cache_path(identifier)
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256()
        with atomic_write(path, mode="wb", overwrite=True) as handle:

            def write(data):
                digest.update(data)
                handle.write(data)

            write(b"{")
            for index, (ref, metadata) in enumerate(records.items()):
                if index:
                    write(b",")
                write(json.dumps(str(ref)).encode())
                write(b":")
                write(json.dumps(metadata).encode())
            write(b"}")
        return digest.hexdigest()

    def delete(self, identifier: Union[str, Path]) -> None:
        if self.is_sample_id(identifier):
            raise NotImplementedError("Webshart datasets are read-only.")
        self._cache_path(identifier).unlink()

    def exists(self, identifier: Union[str, Path]) -> bool:
        if self.is_sample_id(identifier):
            try:
                identifier = self.normalize_sample_id(identifier)
                sample_ref = self._resolve_sample_ref(identifier)
                shard_info = self.dataset.get_shard_info(sample_ref.shard_idx)
                num_samples = shard_info.get("num_samples")
                return num_samples is None or 0 <= sample_ref.sample_idx < int(num_samples)
            except Exception:
                return False
        return self._cache_path(identifier).exists()

    def open_file(self, identifier: Union[str, Path], mode: str):
        if mode not in {"r", "rt", "rb"}:
            raise NotImplementedError("Webshart data backend does not support open_file writes.")
        if self.is_sample_id(identifier):
            return BytesIO(self.read(identifier))
        return self._cache_path(identifier).open(mode)

    def list_files(self, file_extensions: list = None, instance_data_dir: str = None) -> List[Tuple[str, List, List[str]]]:
        requested_extensions = self._normalise_file_extensions(file_extensions)
        if requested_extensions and requested_extensions.issubset(self.CACHE_EXTENSIONS):
            return self._list_cache_files(file_extensions=file_extensions, instance_data_dir=instance_data_dir)

        files = []
        for shard_idx in range(self.num_shards()):
            samples = (
                self.loader.list_samples_in_shard(shard_idx)
                if self.format == "parquet"
                else [
                    {"sample_idx": i, "filename": name}
                    for i, name in enumerate(self.dataset.list_samples_in_shard(shard_idx))
                ]
            )
            for sample in samples:
                sample_idx, filename = sample["sample_idx"], sample["filename"]
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

    def _require_image_decoders(self, filepaths):
        if any(self.filename_suffix(self.normalize_sample_id(path)) == ".jxl" for path in filepaths):
            if ".jxl" not in Image.registered_extensions():
                raise ImportError(
                    "Webshart JPEG XL samples require pillow-jxl-plugin. Install SimpleTuner with the jxl extra."
                )

    def read_image(self, filepath: str, delete_problematic_images: bool = False):
        self._require_image_decoders((filepath,))
        try:
            file_extension = self.filename_suffix(self.normalize_sample_id(filepath)).strip(".")
            loader = load_video if file_extension in video_file_extensions else load_image
            return loader(self.read(filepath, as_byteIO=True))
        except Exception as exc:
            logger.error("Error opening webshart sample %s: %s", filepath, exc)
            if delete_problematic_images:
                logger.warning("Cannot delete from webshart dataset - skipping problematic image")
            return None

    def read_image_batch(self, filepaths: list, delete_problematic_images: bool = False):
        if self.format == "parquet" and all(self.is_sample_id(path) for path in filepaths):
            self._require_image_decoders(filepaths)
            refs = [self._resolve_sample_ref(path) for path in filepaths]
            samples = self.loader.load_samples(
                [(ref.shard_idx, ref.sample_idx) for ref in refs],
                parallel_reads=self.parallel_downloads,
                **({"metadata_fields": ()} if self.supports_caption_projection else {}),
            )
            available_keys, output_images = [], []
            for filepath, sample in zip(filepaths, samples, strict=True):
                if sample.data is None:
                    continue
                try:
                    image = load_image(BytesIO(sample.data))
                except Exception as exc:
                    logger.warning("Unable to decode webshart sample %s: %s", filepath, exc)
                    continue
                if image is not None:
                    available_keys.append(filepath)
                    output_images.append(image)
            return available_keys, output_images
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
            "format": self.format,
            "dataset_fingerprint": self.dataset.fingerprint if self.format == "parquet" else None,
            "manifest": self.manifest,
            "shuffle_seed": self.shuffle_seed,
            "columns": self.columns,
            "aws_region_name": self.aws_region_name,
            "aws_endpoint_url": self.aws_endpoint_url,
            "aws_max_pool_connections": self.aws_max_pool_connections,
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
            "caption_key": self.caption_key,
        }

    @staticmethod
    def from_instance_representation(representation: dict) -> "WebshartDataBackend":
        if representation.get("backend_type") != "webshart":
            raise ValueError(f"Expected backend_type 'webshart', got {representation.get('backend_type')}")
        backend = WebshartDataBackend(
            accelerator=None,
            id=representation["id"],
            source=representation["source"],
            format=representation.get("format", "tar"),
            manifest=representation.get("manifest"),
            shuffle_seed=representation.get("shuffle_seed"),
            columns=representation.get("columns"),
            aws_region_name=representation.get("aws_region_name"),
            aws_endpoint_url=representation.get("aws_endpoint_url"),
            aws_max_pool_connections=representation.get("aws_max_pool_connections", 32),
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
            caption_key=representation.get("caption_key"),
        )
        expected = representation.get("dataset_fingerprint")
        if expected is not None and expected != backend.dataset.fingerprint:
            raise ValueError("Parquet catalog changed since worker serialization; reopen the dataset with a fixed snapshot.")
        return backend

    def count_source_rows(self, *, exclude_null_images=False):
        if self.format != "parquet":
            return super().count_source_rows(exclude_null_images=exclude_null_images)
        if not exclude_null_images:
            return len(self.dataset)
        count_rows = getattr(self.dataset, "count_rows", None)
        if count_rows is None:
            raise NotImplementedError("The installed Webshart reader does not support counting non-null image rows.")
        return count_rows(exclude_null_images=True)

    def prepared_partition_identity(self, indices):
        """Identify capture shards from the manifest without opening their objects."""
        if self.format != "parquet" or getattr(self.dataset, "physical_row_count", None) is None:
            return None
        shards = []
        for index in indices:
            shard = self.dataset.get_shard_info(index)
            if not shard.get("sha256"):
                return None
            shards.append((index, shard["path"], shard["sha256"], shard["size"], shard["num_rows"]))
        endpoint = None
        if self.source.startswith("s3://"):
            endpoint = (
                self.aws_endpoint_url
                or os.environ.get("AWS_ENDPOINT_URL_S3")
                or os.environ.get("AWS_ENDPOINT_URL")
                or os.environ.get("AWS_ENDPOINT")
            )
        return (self.source, endpoint, self.columns, shards)

    def num_shards(self) -> int:
        value = getattr(self.dataset, "num_shards", None)
        if callable(value):
            return int(value())
        return int(value)

    def get_shard_metadata(self, shard_idx: int) -> dict:
        # Caption scans call this once per sample; without memoization each call
        # re-fetches and copies the full shard metadata (~200ms x dataset size).
        shard_idx = int(shard_idx)
        cached = self._shard_metadata_cache.get(shard_idx)
        if cached is None:
            cached = dict(self.loader.get_metadata(shard_idx))
            if len(self._shard_metadata_cache) >= self._shard_metadata_cache_limit:
                self._shard_metadata_cache.pop(next(iter(self._shard_metadata_cache)))
            self._shard_metadata_cache[shard_idx] = cached
        return cached

    def list_shard_sample_aspect_buckets(
        self,
        shard_indices: list[int],
        key: str = "aspect",
        target_pixel_area: Optional[int] = None,
        target_resolution_multiple: int = 64,
        round_to: Optional[int] = 2,
        dataset_filter: Optional[DatasetFilter] = None,
        metadata_fields=None,
    ) -> list[dict]:
        if self.format == "parquet":
            results = self.loader.list_shard_sample_aspect_buckets(
                shard_indices,
                key=key,
                target_pixel_area=target_pixel_area,
                target_resolution_multiple=target_resolution_multiple,
                round_to=round_to,
                **({"metadata_fields": metadata_fields} if metadata_fields is not None else {}),
            )
            if dataset_filter is not None and dataset_filter.path_active():
                for result in results:
                    result["buckets"] = {
                        bucket: [entry for entry in entries if dataset_filter.matches_path(entry["filename"])]
                        for bucket, entries in result["buckets"].items()
                    }
            return results
        if dataset_filter is not None and dataset_filter.path_active():
            if not hasattr(self.loader, "list_shard_sample_aspect_buckets_filtered"):
                raise ImportError(
                    "Webshart path filtering requires a webshart build that provides "
                    "TarDataLoader.list_shard_sample_aspect_buckets_filtered()."
                )
            return self.loader.list_shard_sample_aspect_buckets_filtered(
                shard_indices,
                key=key,
                target_pixel_area=target_pixel_area,
                target_resolution_multiple=target_resolution_multiple,
                round_to=round_to,
                **dataset_filter.path.to_webshart_kwargs(),
            )
        return self.loader.list_shard_sample_aspect_buckets(
            shard_indices,
            key=key,
            target_pixel_area=target_pixel_area,
            target_resolution_multiple=target_resolution_multiple,
            round_to=round_to,
        )

    def list_samples_in_shard(
        self,
        shard_idx: int,
        dataset_filter: Optional[DatasetFilter] = None,
    ) -> list[dict]:
        if self.format == "parquet":
            samples = self.loader.list_samples_in_shard(shard_idx)
            return [entry for entry in samples if dataset_filter is None or dataset_filter.matches_path(entry["filename"])]
        if dataset_filter is not None and dataset_filter.path_active():
            if not hasattr(self.loader, "list_samples_in_shard_filtered"):
                raise ImportError(
                    "Webshart path filtering requires a webshart build that provides "
                    "TarDataLoader.list_samples_in_shard_filtered()."
                )
            return self.loader.list_samples_in_shard_filtered(
                shard_idx,
                **dataset_filter.path.to_webshart_kwargs(),
            )
        return list(self.loader.list_samples_in_shard(shard_idx))
