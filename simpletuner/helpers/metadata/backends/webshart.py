import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from hashlib import sha256
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.data_backend.webshart import CaptionValue, WebshartDataBackend
from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.metadata.transform_records import (
    _TRANSFORM_READ_ROWS,
    DECIDED,
    decode_transform,
    encode_transform,
    settings_digest,
    transform_recipe,
)
from simpletuner.helpers.prompts import CaptionNotFoundError
from simpletuner.helpers.training import video_file_extensions
from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger("WebshartMetadataBackend")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")

# Under webshart's limit, which counts 32 bytes a row; ours counts 64.
_TRANSFORM_BATCH_BYTES = 1 << 20
_SKIP_REASONS = ("metadata_missing", "caption_missing", "too_small", "aspect_ratio", "error")


class WebshartMetadataBackend(MetadataBackend):
    shuffle_window = None
    _digest = None
    _transforms = None
    # The recipe the bucket index was built under. Without a window, the index names it and every sample it
    # lists has its geometry recorded under it.
    _recipe = None

    def __init__(
        self,
        id: str,
        instance_data_dir: str,
        cache_file: str,
        metadata_file: str,
        data_backend: WebshartDataBackend,
        accelerator,
        batch_size: int,
        resolution: float,
        resolution_type: str,
        delete_problematic_images: bool = False,
        delete_unwanted_images: bool = False,
        metadata_update_interval: int = 3600,
        minimum_image_size: int = None,
        minimum_aspect_ratio: int = None,
        maximum_aspect_ratio: int = None,
        num_frames: int = None,
        minimum_num_frames: int = None,
        maximum_num_frames: int = None,
        cache_file_suffix: str = None,
        repeats: int = 0,
        max_num_samples: int = None,
    ):
        if not isinstance(data_backend, WebshartDataBackend):
            raise ValueError("WebshartMetadataBackend requires WebshartDataBackend")
        # With a window the dataset is indexed a shard at a time, as the cursor opens each one, and
        # has no files that describe all of it.
        self.shuffle_window = data_backend.shuffle_window
        self._open_units: dict[int, list[str]] = {}
        if data_backend.format == "parquet":
            fingerprint = data_backend.dataset.fingerprint[:16]
            cache_file = f"{cache_file}_source_{fingerprint}"
            metadata_file = f"{metadata_file}_source_{fingerprint}"
        if data_backend.caption_key is not None:
            caption_digest = sha256(json.dumps(data_backend.caption_key).encode("utf-8")).hexdigest()[:16]
            cache_file = f"{cache_file}_captions_{caption_digest}"
            metadata_file = f"{metadata_file}_captions_{caption_digest}"
        super().__init__(
            id=id,
            instance_data_dir=instance_data_dir,
            cache_file=cache_file,
            metadata_file=metadata_file,
            data_backend=data_backend,
            accelerator=accelerator,
            batch_size=batch_size,
            resolution=resolution,
            resolution_type=resolution_type,
            delete_problematic_images=delete_problematic_images,
            delete_unwanted_images=delete_unwanted_images,
            metadata_update_interval=metadata_update_interval,
            minimum_image_size=minimum_image_size,
            minimum_aspect_ratio=minimum_aspect_ratio,
            maximum_aspect_ratio=maximum_aspect_ratio,
            maximum_num_frames=maximum_num_frames,
            minimum_num_frames=minimum_num_frames,
            num_frames=num_frames,
            cache_file_suffix=cache_file_suffix,
            repeats=repeats,
            max_num_samples=max_num_samples,
        )
        if self.dataset_type not in {DatasetType.IMAGE, DatasetType.VIDEO, DatasetType.CONDITIONING, DatasetType.EVAL}:
            raise ValueError("WebshartMetadataBackend supports image, video, conditioning, and eval datasets only.")

        context = accelerator.main_process_first() if hasattr(accelerator, "main_process_first") else nullcontext()
        with context:
            self.reload_cache()
        if hasattr(accelerator, "wait_for_everyone"):
            accelerator.wait_for_everyone()

    def _bucketed_sample_ids(self) -> list[str]:
        sample_ids = []
        seen = set()
        for bucket in self.aspect_ratio_bucket_indices.values():
            for sample_path in bucket:
                sample_path = str(sample_path)
                if sample_path in seen:
                    continue
                sample_ids.append(sample_path)
                seen.add(sample_path)
        return sample_ids

    def _sync_image_files_with_buckets(self) -> None:
        sample_ids = self._bucketed_sample_ids()
        if not sample_ids:
            return
        StateTracker.set_image_files([("", [], sample_ids)], data_backend_id=self.data_backend.id)

    def caption_cache_entry(self, index: str) -> Optional[CaptionValue]:
        return self.data_backend.get_caption(self.data_backend.normalize_sample_id(index))

    def reload_cache(self, set_config: bool = True):
        if self.shuffle_window is not None:
            return
        if self.data_backend.exists(self.cache_file):
            try:
                cache_data = json.loads(self.data_backend.read(self.cache_file))
            except Exception as exc:
                logger.warning("Error loading webshart aspect bucket cache, creating new one: %s", exc)
                cache_data = {}
            self.aspect_ratio_bucket_indices = cache_data.get("aspect_ratio_bucket_indices", {})
            self._recipe = cache_data.get("transform_recipe")
            self._sync_image_files_with_buckets()
            if set_config:
                self.config = cache_data.get("config", {})
            self.filtering_statistics = cache_data.get("filtering_statistics")
        else:
            logger.debug("No webshart cache file found, starting fresh.")
        self.load_image_metadata()

    def save_cache(self, enforce_constraints: bool = False):
        if self.shuffle_window is not None:
            return
        if enforce_constraints:
            self._enforce_min_bucket_size()
        self._enforce_min_aspect_ratio()
        self._enforce_max_aspect_ratio()

        if self.read_only:
            logger.debug("Metadata backend is read-only. Skipping save.")
            return

        cache_data = {
            "config": StateTracker.get_data_backend_config(data_backend_id=self.data_backend.id),
            "aspect_ratio_bucket_indices": {
                key: [str(path) for path in value] for key, value in self.aspect_ratio_bucket_indices.items()
            },
            "transform_recipe": self._recipe,
        }
        if self.filtering_statistics is not None:
            cache_data["filtering_statistics"] = self.filtering_statistics
        self.data_backend.write(self.cache_file, json.dumps(cache_data))

    def load_image_metadata(self):
        """What was measured of each sample, from the metadata file, and the geometry recorded for it under the index's recipe."""
        if self.shuffle_window is not None:
            return
        self.image_metadata = {}
        self.image_metadata_loaded = False
        if not self.data_backend.exists(self.metadata_file):
            return
        measured = json.loads(self.data_backend.read(self.metadata_file))
        if self._recipe is not None:
            cache = self._open_transform_cache(self._recipe)
            paths = list(measured)
            for start in range(0, len(paths), _TRANSFORM_READ_ROWS):
                chunk = paths[start : start + _TRANSFORM_READ_ROWS]
                keys = [
                    (
                        self.data_backend.shard_key(measured[path]["webshart"]["shard_idx"]),
                        int(measured[path]["webshart"]["sample_idx"]),
                    )
                    for path in chunk
                ]
                for path, decision in zip(chunk, cache.read_batch(keys)):
                    # A sample with no geometry recorded under the recipe was not decided under it.
                    if decision is not None:
                        metadata = measured[path]
                        metadata.pop("captions", None)
                        self.image_metadata[path] = {**metadata, **decode_transform(decision)}
        self.image_metadata_loaded = True

    def save_image_metadata(self):
        """The metadata file keeps what was measured; the geometry decided for each sample is in the transform cache."""
        if self.shuffle_window is not None:
            return
        measured = {
            path: {key: value for key, value in metadata.items() if key not in DECIDED}
            for path, metadata in self.image_metadata.items()
        }
        self.data_backend.write(self.metadata_file, json.dumps(measured))
        self.image_metadata_loaded = True

    def _all_shard_indices(self) -> list[int]:
        return list(range(self.data_backend.num_shards()))

    def unit_count(self) -> int:
        return self.data_backend.num_shards()

    def open_unit(self, unit: int, epoch: int) -> dict[str, list[str]]:
        """
        Index one shard. Its usable samples are returned by target size and described until close_unit.

        What is decided for a shard is recorded, and a later opening gives every sample the size and the
        crop it was given before: its latent may already be written. A dataset that re-crops each epoch
        records each epoch's decisions apart. The first decisions recorded for a shard are the ones kept,
        so the shard is read back after it is decided.
        """
        samples = self._shard_samples(self._transform_cache(epoch), unit)
        pools: dict[str, list[str]] = {}
        for sample_path, sample_metadata in samples.items():
            width, height = sample_metadata["target_size"]
            pools.setdefault(f"{width}x{height}", []).append(sample_path)
        self.image_metadata.update(samples)
        self._open_units[unit] = list(samples)
        return pools

    def _shard_samples(self, cache, unit: int, skipped: dict | None = None, executor=None) -> dict[str, dict]:
        """
        A shard's usable samples with the geometry recorded for them, deciding the shard first if it was not.
        The first decisions recorded for a shard are the ones kept, so the shard is read back after it is decided.
        """
        shard, entries = self.data_backend.shard_key(unit), self._entries_for_shard(unit)
        # The row after the shard's last is written once all of its decisions are: without it the shard was not decided.
        decided = int(self.data_backend.dataset.get_shard_info(unit)["num_samples"])
        if cache.read_batch([(shard, decided)])[0] is None:
            self._record_decisions(cache, shard, decided, self._decide_samples(unit, entries, skipped, executor))
        return self._recorded_samples(unit, entries, cache, shard)

    @staticmethod
    def _record_decisions(cache, shard: str, decided: int, samples: dict[str, dict]) -> None:
        rows = [
            (int(metadata["webshart"]["sample_idx"]), encode_transform({key: metadata[key] for key in DECIDED}))
            for metadata in samples.values()
        ]
        rows.append((decided, b"1"))
        batch, size = [], len(shard.encode("utf-8"))
        for row in rows:
            cost = 64 + len(row[1])
            if batch and size + cost > _TRANSFORM_BATCH_BYTES:
                cache.enqueue(shard, batch)
                batch, size = [], len(shard.encode("utf-8"))
            batch.append(row)
            size += cost
        cache.enqueue(shard, batch)
        cache.flush()

    def _samples_per_unit_path(self) -> Path:
        return Path(self.data_backend.cache_dir) / "decisions" / self._settings_digest() / "samples_per_unit.json"

    def recorded_samples_per_unit(self) -> float | None:
        """How many samples a unit kept on average, as measured when the first units were decided."""
        path = self._samples_per_unit_path()
        return json.loads(path.read_text())["samples_per_unit"] if path.exists() else None

    def record_samples_per_unit(self, value: float) -> None:
        path = self._samples_per_unit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w") as file:
                json.dump({"samples_per_unit": value}, file)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def _decide_samples(self, unit: int, entries: list[dict], skipped: dict | None, executor) -> dict[str, dict]:
        skipped = dict.fromkeys(_SKIP_REASONS, 0) if skipped is None else skipped
        samples = {}
        for _, sample_path, sample_metadata in self._prepare_shard(
            unit, entries, self._requires_captions(), skipped, executor=executor
        ):
            aspect_ratio = round(sample_metadata["aspect_ratio"], 2)
            if (self.minimum_aspect_ratio and aspect_ratio < self.minimum_aspect_ratio) or (
                self.maximum_aspect_ratio and aspect_ratio > self.maximum_aspect_ratio
            ):
                skipped["aspect_ratio"] += 1
                continue
            samples[sample_path] = sample_metadata
        logger.debug("(id=%s) Shard %s holds %s usable samples; skipped %s.", self.id, unit, len(samples), skipped)
        return samples

    def _recorded_samples(self, unit: int, entries: list[dict], cache, shard: str) -> dict[str, dict]:
        """The shard's samples as they were decided. A sample without a decision was left out then."""
        decisions = cache.read_batch([(shard, int(entry["sample_idx"])) for entry in entries])
        if not any(decisions):
            return {}
        shard_metadata = self.data_backend.get_shard_metadata(unit)
        samples = {}
        for entry, decision in zip(entries, decisions):
            if decision is None:
                continue
            entry = {**entry, "shard_idx": unit}
            sample_path = self._sample_id_from_entry(unit, entry)
            samples[sample_path] = {
                **self._metadata_for_entry(shard_metadata, str(entry["filename"]), entry, sample_path),
                **decode_transform(decision),
            }
        return samples

    def _settings_digest(self) -> str:
        if self._digest is None:
            backend = self.data_backend
            self._digest = settings_digest(
                self,
                fingerprint=backend.dataset.fingerprint if backend.format == "parquet" else None,
                units=backend.num_shards(),
            )
        return self._digest

    def _transform_cache(self, epoch: int):
        """The store of this dataset's decisions for the recipe they follow in this epoch."""
        return self._open_transform_cache(transform_recipe(self._settings_digest(), epoch, self.id))

    def _open_transform_cache(self, recipe: str):
        if self._transforms is None or self._transforms[0] != recipe:
            if self._transforms is not None:
                self._transforms[1].close()
            directory = Path(self.data_backend.cache_dir) / "transforms"
            directory.mkdir(parents=True, exist_ok=True)
            cache = self.data_backend.webshart.TransformCache(str(directory), recipe, max_batch_bytes=_TRANSFORM_BATCH_BYTES)
            self._transforms = (recipe, cache)
        return self._transforms[1]

    def close_unit(self, unit: int) -> None:
        for sample_path in self._open_units.pop(unit):
            del self.image_metadata[sample_path]

    def check_window(self, units: Sequence[int], *, ranks: int) -> None:
        """Refuse a window whose shards the shard cache cannot hold for every rank at once."""
        backend = self.data_backend
        if backend.shard_cache_dir is None or os.path.isdir(backend.source):
            # Nothing is cached: reads go to the source, which for a local dataset is the disk.
            return
        largest = max(backend.dataset.get_shard_info(unit)["size"] for unit in units)
        # Every rank's cache is the same directory, and the limit applies to the directory. A rank holds its
        # window's shards and downloads the next one before it lets the oldest go.
        needed = ranks * (self.shuffle_window + 1) * largest
        limit = backend.shard_cache_gb * 2**30
        if needed <= limit:
            return
        needed_gb = ceil(needed / 2**30)
        message = (
            f"(id={self.id}) webshart.shuffle_window={self.shuffle_window} keeps that many shards open on each of"
            f" {ranks} rank(s), and one more while the next downloads, about {needed / 2**30:.1f} GiB with shards"
            f" of up to {largest / 2**20:.0f} MiB, and shard_cache_gb={backend.shard_cache_gb:g} cannot hold them."
        )
        fits = int(limit // (ranks * largest)) - 1
        if fits >= 1:
            message += f" Set shuffle_window to {fits} or less, or shard_cache_gb to at least {needed_gb}."
        else:
            message += f" Set shard_cache_gb to at least {needed_gb}."
        free = shutil.disk_usage(backend.shard_cache_dir).free
        if free < needed:
            message += (
                f" The volume holding {backend.shard_cache_dir} has {free / 2**30:.1f} GiB free,"
                " which is not enough for that."
            )
        raise ValueError(message)

    def _sample_id_from_entry(self, shard_idx: int, entry: dict) -> str:
        if "sample_idx" not in entry:
            raise ValueError("Webshart sample bucket entries must include sample_idx.")
        return self.data_backend.sample_id(shard_idx, int(entry["sample_idx"]), str(entry["filename"]))

    @staticmethod
    def _coerce_positive_number(value: Any, value_type):
        try:
            result = value_type(value)
        except (TypeError, ValueError):
            return None
        return result if result > 0 else None

    @staticmethod
    def _ffprobe_video_path(probe_path: str) -> dict:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                probe_path,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        probe = json.loads(result.stdout) if result.stdout else {}
        streams = probe.get("streams") or []
        stream = streams[0] if streams else {}
        width = WebshartMetadataBackend._coerce_positive_number(stream.get("width"), int)
        height = WebshartMetadataBackend._coerce_positive_number(stream.get("height"), int)
        metadata = {"original_size": (width, height)} if width and height else {}

        num_frames = WebshartMetadataBackend._coerce_positive_number(stream.get("nb_frames"), int)
        if num_frames:
            metadata["num_frames"] = num_frames
        duration = WebshartMetadataBackend._coerce_positive_number(
            stream.get("duration") or (probe.get("format") or {}).get("duration"),
            float,
        )
        if duration:
            metadata["video_duration"] = duration
        return metadata

    def _probe_video_metadata(self, sample_path: str, file_metadata: Optional[dict] = None) -> dict:
        if shutil.which("ffprobe") is None:
            return {}

        suffix = Path(self.data_backend.parse_sample_id(sample_path).filename).suffix or ".mp4"
        probe_path = None
        try:
            range_reader = getattr(self.data_backend, "read_sample_head_tail", None)
            if callable(range_reader):
                try:
                    head, tail, total_size = range_reader(sample_path, file_metadata=file_metadata)
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                        probe_path = handle.name
                        handle.truncate(total_size)
                        handle.seek(0)
                        handle.write(head)
                        handle.seek(total_size - len(tail))
                        handle.write(tail)
                    metadata = self._ffprobe_video_path(probe_path)
                    if metadata.get("original_size"):
                        return metadata
                except Exception as exc:
                    logger.debug("Unable to range-probe Webshart video %s: %s", sample_path, exc)
                finally:
                    if probe_path:
                        Path(probe_path).unlink(missing_ok=True)
                        probe_path = None

            payload = self.data_backend.read(sample_path)
            if not payload:
                return {}
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(payload)
                probe_path = handle.name
            return self._ffprobe_video_path(probe_path)
        except Exception as exc:
            logger.debug("Unable to probe Webshart video %s: %s", sample_path, exc)
            return {}
        finally:
            if probe_path:
                Path(probe_path).unlink(missing_ok=True)

    def _metadata_for_entry(self, shard_metadata: dict, filename: str, entry: dict, sample_path: str) -> dict:
        file_metadata = shard_metadata.get(filename, {}) or {}
        if self.data_backend.caption_key is not None:
            file_metadata = self.data_backend.get_sample_metadata(sample_path)
        width = entry.get("width", file_metadata.get("width"))
        height = entry.get("height", file_metadata.get("height"))
        metadata = {
            "webshart": {
                "shard_idx": entry.get("shard_idx"),
                "sample_idx": entry.get("sample_idx"),
                "filename": filename,
                "offset": entry.get("offset"),
                "size": entry.get("size"),
                "json_path": file_metadata.get("json_path"),
            },
        }
        if width is not None and height is not None:
            metadata["original_size"] = (int(width), int(height))
        json_metadata = file_metadata.get("json_metadata") or {}
        if json_metadata:
            metadata["json_metadata"] = json_metadata
        if "json_path" in file_metadata:
            metadata["json_path"] = file_metadata["json_path"]

        if self.dataset_type is DatasetType.VIDEO:
            fps = self._coerce_positive_number(json_metadata.get("fps"), float)
            num_frames = self._coerce_positive_number(
                json_metadata.get("frame", json_metadata.get("num_frames")),
                int,
            )
            duration = self._coerce_positive_number(
                json_metadata.get("seconds", json_metadata.get("duration")),
                float,
            )
            if fps:
                metadata["fps"] = fps
            if num_frames:
                metadata["num_frames"] = num_frames
            if duration:
                metadata["video_duration"] = duration
            if "original_size" not in metadata:
                probed = self._probe_video_metadata(sample_path, file_metadata=file_metadata)
                for key, value in probed.items():
                    metadata.setdefault(key, value)
        return metadata

    def _prepare_bucket_entry(
        self,
        shard_metadata: dict,
        entry: dict,
        sample_path: str,
        require_captions: bool,
    ) -> tuple[dict, Optional[tuple[str, dict]], Optional[Exception]]:
        try:
            filename = str(entry["filename"])
            sample_metadata = self._metadata_for_entry(shard_metadata, filename, entry, sample_path)
            prepared = self._prepare_metadata(sample_path, sample_metadata)
            if prepared is not None and require_captions and not self._entry_has_caption(shard_metadata, entry, sample_path):
                raise CaptionNotFoundError(f"Could not find caption for {sample_path}")
            return sample_metadata, prepared, None
        except Exception as exc:
            return {}, None, exc

    def _prepare_metadata(self, sample_path: str, sample_metadata: dict) -> Optional[tuple[str, dict]]:
        if not sample_metadata or "original_size" not in sample_metadata:
            return None
        if not self.meets_resolution_requirements(image_metadata=sample_metadata):
            return None
        training_sample = TrainingSample(
            image=None,
            data_backend_id=self.id,
            image_metadata=sample_metadata,
            image_path=sample_path,
        )
        prepared_sample = training_sample.prepare()
        aspect_ratio = float(prepared_sample.aspect_ratio)
        sample_metadata.update(
            {
                "aspect_ratio": aspect_ratio,
                "intermediary_size": prepared_sample.intermediary_size,
                "crop_coordinates": prepared_sample.crop_coordinates,
                "target_size": prepared_sample.target_size,
            }
        )
        return self._bucket_key(sample_metadata), sample_metadata

    def _bucket_key(self, sample_metadata: dict) -> str:
        """The bucket of a sample, from the geometry decided for it."""
        if self.dataset_type is DatasetType.VIDEO and self.bucket_strategy == "resolution_frames":
            target_width, target_height = sample_metadata["target_size"]
            bucket_key, rounded_frames = self._compute_video_bucket(
                target_width, target_height, sample_metadata["num_frames"]
            )
            sample_metadata["bucket_frames"] = rounded_frames
            return bucket_key
        return str(round(sample_metadata["aspect_ratio"], 2))

    def _entries_for_shard(self, shard_idx: int) -> list[dict]:
        if self.dataset_type is not DatasetType.VIDEO:
            shard_bucket_results = self.data_backend.list_shard_sample_aspect_buckets(
                [shard_idx],
                dataset_filter=self.dataset_filter,
            )
            if not shard_bucket_results:
                return []
            shard_bucket_data = shard_bucket_results[0]
            return [entry for entries in shard_bucket_data.get("buckets", {}).values() for entry in entries]

        entries = []
        for sample in self.data_backend.list_samples_in_shard(shard_idx, dataset_filter=self.dataset_filter):
            sample_idx = int(sample["sample_idx"])
            filename = str(sample["filename"])
            if WebshartDataBackend.filename_suffix(filename).strip(".") not in video_file_extensions:
                continue
            entries.append({"sample_idx": sample_idx, "filename": filename})
        return entries

    def _entry_has_caption(self, shard_metadata: dict, entry: dict, sample_path: str) -> bool:
        if self.data_backend.caption_key is not None:
            return bool(self.data_backend.get_caption(sample_path))
        file_metadata = shard_metadata.get(str(entry["filename"]), {}) or {}
        if file_metadata.get("captions"):
            return True
        caption_member = self.data_backend.caption_sidecar_filename(str(entry["filename"]))
        return caption_member in shard_metadata

    def _requires_captions(self) -> bool:
        backend_config = StateTracker.get_data_backend_config(self.id) or {}
        return str(backend_config.get("caption_strategy", "")).lower() == "webshart"

    def _prepare_shard(
        self, shard_idx: int, shard_entries: list[dict], require_captions: bool, skipped: dict, *, executor=None
    ) -> Iterator[tuple[str, str, dict]]:
        """
        Decide the geometry of a shard's samples, one at a time. Yields (bucket, sample id, metadata)
        for each usable sample and counts the others in `skipped` by reason.
        """
        if not shard_entries:
            return
        shard_metadata = self.data_backend.get_shard_metadata(shard_idx)
        candidates = []
        for entry in shard_entries:
            entry = {**entry, "shard_idx": shard_idx}
            candidates.append((entry, self._sample_id_from_entry(shard_idx, entry)))

        def prepare(candidate):
            return self._prepare_bucket_entry(shard_metadata, candidate[0], candidate[1], require_captions)

        # Probing videos is slow, so they are prepared a few at a time on the executor.
        chunk_size = 1 if executor is None else max(1, int(self.data_backend.parallel_downloads) * 2)
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            results = map(prepare, chunk) if executor is None else executor.map(prepare, chunk)
            for (entry, sample_path), (sample_metadata, prepared, error) in zip(chunk, results):
                if isinstance(error, CaptionNotFoundError):
                    skipped["caption_missing"] += 1
                elif error is not None:
                    logger.error("Error processing webshart bucket entry %s: %s", entry, error)
                    skipped["error"] += 1
                elif prepared is None:
                    skipped["too_small" if sample_metadata else "metadata_missing"] += 1
                else:
                    yield prepared[0], sample_path, prepared[1]

    def compute_aspect_ratio_bucket_indices(self, ignore_existing_cache: bool = False, progress_callback=None):
        if self.shuffle_window is not None:
            raise RuntimeError(
                f"(id={self.id}) This dataset sets webshart.shuffle_window, so it is indexed a shard at a time"
                " while training and has no whole-dataset bucket lists to build."
            )
        logger.info("Building aspect ratio buckets from webshart metadata...")
        statistics = {"total_processed": 0, "skipped": {"already_exists": 0, **dict.fromkeys(_SKIP_REASONS, 0)}}

        if not ignore_existing_cache:
            self.reload_cache()
            if self._recipe is None and any(self.aspect_ratio_bucket_indices.values()):
                logger.info("(id=%s) Its bucket index names no recipe of recorded geometry; bucketing it again.", self.id)
                ignore_existing_cache = True
        if ignore_existing_cache:
            self.aspect_ratio_bucket_indices = {}
            self.image_metadata = {}
            existing_files = set()
            # The recipe follows from the settings and the epoch, so bucketing again under the same ones reads
            # back the geometry recorded before: latents already written stay valid.
            self._recipe = None
        else:
            existing_files = (
                set().union(*self.aspect_ratio_bucket_indices.values()) if self.aspect_ratio_bucket_indices else set()
            )
            statistics["skipped"]["already_exists"] = len(existing_files)
        if self._recipe is None:
            self._recipe = transform_recipe(self._settings_digest(), StateTracker.get_epoch(), self.id)
        cache = self._open_transform_cache(self._recipe)

        last_save_time = time.time()
        aspect_ratio_bucket_updates: Dict[float, list[str]] = {}
        metadata_updates: Dict[str, dict] = {}
        shard_indices = self._all_shard_indices()
        worker_count = max(1, int(getattr(self.data_backend, "parallel_downloads", 1)))
        executor = (
            ThreadPoolExecutor(max_workers=worker_count)
            if self.dataset_type is DatasetType.VIDEO and worker_count > 1
            else None
        )

        try:
            for shard_idx in tqdm(
                shard_indices,
                desc="Processing webshart metadata",
                total=len(shard_indices),
                leave=False,
                ncols=100,
            ):
                if (
                    self.max_num_samples is not None
                    and statistics["total_processed"] + len(existing_files) >= self.max_num_samples
                ):
                    break

                samples = self._shard_samples(cache, shard_idx, statistics["skipped"], executor)
                for sample_path, sample_metadata in samples.items():
                    if sample_path in existing_files:
                        continue
                    aspect_ratio_bucket_updates.setdefault(self._bucket_key(sample_metadata), []).append(sample_path)
                    metadata_updates[sample_path] = sample_metadata
                    statistics["total_processed"] += 1
                    if (
                        self.max_num_samples is not None
                        and statistics["total_processed"] + len(existing_files) >= self.max_num_samples
                    ):
                        break

                    current_time = time.time()
                    if (current_time - last_save_time) >= self.metadata_update_interval:
                        for key, value in aspect_ratio_bucket_updates.items():
                            self.aspect_ratio_bucket_indices.setdefault(key, []).extend(value)
                        aspect_ratio_bucket_updates = {}
                        for path, metadata in metadata_updates.items():
                            self.set_metadata_by_filepath(path, metadata, update_json=False)
                        metadata_updates = {}
                        self.save_cache(enforce_constraints=False)
                        self.save_image_metadata()
                        last_save_time = current_time
                if progress_callback is not None:
                    progress_callback(shard_idx + 1, len(shard_indices))
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        for key, value in aspect_ratio_bucket_updates.items():
            self.aspect_ratio_bucket_indices.setdefault(key, []).extend(value)
        for path, metadata in metadata_updates.items():
            self.set_metadata_by_filepath(path, metadata, update_json=False)

        self.filtering_statistics = statistics
        self.save_image_metadata()
        self.save_cache(enforce_constraints=True)
        self._sync_image_files_with_buckets()
        if self.bucket_report:
            self.bucket_report.update_statistics(statistics)
            self.bucket_report.record_bucket_snapshot("post_refresh", self.aspect_ratio_bucket_indices)

    def refresh_buckets(self, rank: int = None):
        self.compute_aspect_ratio_bucket_indices()
        logger.debug("Refreshing webshart buckets for rank %s via data_backend id %s.", rank, self.id)
        return
