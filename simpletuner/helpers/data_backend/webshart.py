import json
import logging
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.data_backend.dataset_types import DatasetType, ensure_dataset_type
from simpletuner.helpers.image_manipulation.load import load_image
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
        self.shard_cache_dir = str(shard_cache_dir) if shard_cache_dir else str(Path(self.cache_dir) / "shard_cache")
        self.shard_cache_gb = float(shard_cache_gb)
        self.parallel_downloads = int(parallel_downloads)
        self.buffer_size = int(buffer_size)
        self.max_file_size = int(max_file_size)
        self.compress_cache = compress_cache
        self.dataset_type = ensure_dataset_type(dataset_type, default=DatasetType.IMAGE)

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.metadata_cache_dir).mkdir(parents=True, exist_ok=True)

        self.dataset = self.webshart.discover_dataset(
            source=self.source,
            hf_token=self.hf_token,
            subfolder=self.subfolder,
            metadata=self.metadata,
        )
        self.dataset.enable_metadata_cache(location=self.metadata_cache_dir)
        if self.shard_cache_dir:
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

    @classmethod
    def sample_id(cls, shard_idx: int, sample_idx: int, filename: str) -> str:
        return f"{cls.SAMPLE_PREFIX}{int(shard_idx)}/{int(sample_idx)}/{filename}"

    @classmethod
    def parse_sample_id(cls, identifier: Union[str, Path]) -> WebshartSampleRef:
        value = str(identifier)
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
        return str(identifier).startswith(cls.SAMPLE_PREFIX)

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
        entry = self.loader.load_sample(sample_ref.shard_idx, sample_ref.sample_idx)
        return bytes(entry.data)

    def read(self, identifier: Union[str, Path], as_byteIO: bool = False) -> Any:
        if self.is_sample_id(identifier):
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
            return sample_path
        cache_path = self._cache_path(sample_path)
        return str(cache_path) if cache_path.exists() else None

    def read_image(self, filepath: str, delete_problematic_images: bool = False):
        try:
            return load_image(self.read(filepath, as_byteIO=True))
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
