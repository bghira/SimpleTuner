import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.data_backend.webshart import WebshartDataBackend
from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.training import video_file_extensions
from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger("WebshartMetadataBackend")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


def _coerce_bucket_keys_to_float(indices: dict) -> dict:
    coerced = {}
    for key, values in (indices or {}).items():
        try:
            coerced_key = float(key)
        except (TypeError, ValueError):
            coerced_key = key
        coerced[coerced_key] = list(values) if not isinstance(values, list) else values
    return coerced


class WebshartMetadataBackend(MetadataBackend):
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
        if not isinstance(data_backend, WebshartDataBackend):
            raise ValueError("WebshartMetadataBackend requires WebshartDataBackend")
        if self.dataset_type not in {DatasetType.IMAGE, DatasetType.VIDEO, DatasetType.CONDITIONING, DatasetType.EVAL}:
            raise ValueError("WebshartMetadataBackend supports image, video, conditioning, and eval datasets only.")

        self.caption_cache: Dict[str, Union[str, List[str], dict]] = {}

        context = accelerator.main_process_first() if hasattr(accelerator, "main_process_first") else nullcontext()
        with context:
            self.reload_cache()
            self.load_image_metadata()
            self._load_caption_cache()
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

    def caption_cache_entry(self, index: str) -> Optional[Union[str, List[str], dict]]:
        index = self.data_backend.normalize_sample_id(index)
        caption = self.caption_cache.get(index, None)
        if caption is not None:
            return caption
        caption = self.data_backend.get_caption(index)
        if caption is not None:
            self.caption_cache[index] = caption
        return caption

    def _caption_cache_path(self):
        return f"{self.cache_file}_captions.json"

    def _load_caption_cache(self) -> None:
        path = self._caption_cache_path()
        if self.data_backend.exists(path):
            try:
                raw = self.data_backend.read(path)
                loaded = json.loads(raw)
                # An empty cache file (written before captions were indexed) must not
                # short-circuit the rebuild below, or it wedges every startup into
                # per-sample caption lookups.
                if loaded:
                    self.caption_cache = loaded
                    return
            except Exception as exc:
                logger.warning("Error loading webshart caption cache, regenerating when buckets refresh: %s", exc)
        self.caption_cache = {}
        for sample_path, metadata in self.image_metadata.items():
            captions = metadata.get("captions") if isinstance(metadata, dict) else None
            if captions:
                self.caption_cache[str(sample_path)] = captions

    def _save_caption_cache(self) -> None:
        self.data_backend.write(self._caption_cache_path(), json.dumps(self.caption_cache))

    def reload_cache(self, set_config: bool = True):
        if self.data_backend.exists(self.cache_file):
            try:
                cache_data = json.loads(self.data_backend.read(self.cache_file))
            except Exception as exc:
                logger.warning("Error loading webshart aspect bucket cache, creating new one: %s", exc)
                cache_data = {}
            self.aspect_ratio_bucket_indices = _coerce_bucket_keys_to_float(
                cache_data.get("aspect_ratio_bucket_indices", {})
            )
            self._sync_image_files_with_buckets()
            if set_config:
                self.config = cache_data.get("config", {})
                if self.config:
                    StateTracker.set_data_backend_config(data_backend_id=self.id, config=self.config)
            self.filtering_statistics = cache_data.get("filtering_statistics")
        else:
            logger.debug("No webshart cache file found, starting fresh.")

    def save_cache(self, enforce_constraints: bool = False):
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
        }
        if self.filtering_statistics is not None:
            cache_data["filtering_statistics"] = self.filtering_statistics
        self.data_backend.write(self.cache_file, json.dumps(cache_data))

    def load_image_metadata(self):
        self.image_metadata = {}
        self.image_metadata_loaded = False
        if self.data_backend.exists(self.metadata_file):
            self.image_metadata = json.loads(self.data_backend.read(self.metadata_file))
            self.image_metadata_loaded = True

    def save_image_metadata(self):
        self.data_backend.write(self.metadata_file, json.dumps(self.image_metadata))
        self.image_metadata_loaded = True

    def _all_shard_indices(self) -> list[int]:
        return list(range(self.data_backend.num_shards()))

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
        if "captions" in file_metadata:
            metadata["captions"] = file_metadata["captions"]
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
    ) -> tuple[dict, Optional[tuple[float, dict]], Optional[Exception]]:
        try:
            filename = str(entry["filename"])
            sample_metadata = self._metadata_for_entry(shard_metadata, filename, entry, sample_path)
            return sample_metadata, self._prepare_metadata(sample_path, sample_metadata), None
        except Exception as exc:
            return {}, None, exc

    def _prepare_metadata(self, sample_path: str, sample_metadata: dict) -> Optional[tuple[float, dict]]:
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
        if self.dataset_type is DatasetType.VIDEO and self.bucket_strategy == "resolution_frames":
            target_width, target_height = prepared_sample.target_size
            bucket_key, rounded_frames = self._compute_video_bucket(
                target_width,
                target_height,
                sample_metadata["num_frames"],
            )
            sample_metadata["bucket_frames"] = rounded_frames
        else:
            bucket_key = round(aspect_ratio, 2)
        return bucket_key, sample_metadata

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
            if Path(filename).suffix.lower().strip(".") not in video_file_extensions:
                continue
            entries.append({"sample_idx": sample_idx, "filename": filename})
        return entries

    def compute_aspect_ratio_bucket_indices(self, ignore_existing_cache: bool = False, progress_callback=None):
        logger.info("Building aspect ratio buckets from webshart metadata...")
        statistics = {
            "total_processed": 0,
            "skipped": {
                "already_exists": 0,
                "metadata_missing": 0,
                "caption_missing": 0,
                "too_small": 0,
                "error": 0,
            },
        }
        backend_config = StateTracker.get_data_backend_config(self.id) or {}
        require_captions = str(backend_config.get("caption_strategy", "")).lower() == "webshart"

        if not ignore_existing_cache:
            self.reload_cache()
            self.load_image_metadata()
            existing_files = (
                set().union(*self.aspect_ratio_bucket_indices.values()) if self.aspect_ratio_bucket_indices else set()
            )
            statistics["skipped"]["already_exists"] = len(existing_files)
        else:
            self.aspect_ratio_bucket_indices = {}
            self.image_metadata = {}
            self.caption_cache = {}
            existing_files = set()

        processed_entries = 0
        last_save_time = time.time()
        aspect_ratio_bucket_updates: Dict[float, list[str]] = {}
        metadata_updates: Dict[str, dict] = {}
        shard_metadata_cache: Dict[int, dict] = {}
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

                shard_entries = self._entries_for_shard(shard_idx)
                if not shard_entries:
                    continue
                shard_metadata = shard_metadata_cache.setdefault(shard_idx, self.data_backend.get_shard_metadata(shard_idx))
                candidates = []
                for entry in shard_entries:
                    entry = {**entry, "shard_idx": shard_idx}
                    sample_path = self._sample_id_from_entry(shard_idx, entry)
                    if sample_path in existing_files:
                        continue
                    candidates.append((entry, sample_path))

                chunk_size = max(1, worker_count * 2)
                candidate_idx = 0
                while candidate_idx < len(candidates):
                    if (
                        self.max_num_samples is not None
                        and statistics["total_processed"] + len(existing_files) >= self.max_num_samples
                    ):
                        break
                    remaining = (
                        self.max_num_samples - statistics["total_processed"] - len(existing_files)
                        if self.max_num_samples is not None
                        else chunk_size
                    )
                    current_chunk_size = min(chunk_size, remaining)
                    chunk = candidates[candidate_idx : candidate_idx + current_chunk_size]
                    candidate_idx += len(chunk)
                    if executor is None:
                        results = [
                            self._prepare_bucket_entry(shard_metadata, entry, sample_path) for entry, sample_path in chunk
                        ]
                    else:
                        results = executor.map(
                            lambda item: self._prepare_bucket_entry(shard_metadata, item[0], item[1]),
                            chunk,
                        )

                    for (entry, sample_path), (sample_metadata, prepared, error) in zip(chunk, results):
                        processed_entries += 1
                        if error is not None:
                            logger.error("Error processing webshart bucket entry %s: %s", entry, error)
                            statistics["skipped"]["error"] += 1
                            continue
                        if prepared is None:
                            if sample_metadata:
                                statistics["skipped"]["too_small"] += 1
                            else:
                                statistics["skipped"]["metadata_missing"] += 1
                            continue
                        bucket_key, sample_metadata = prepared
                        if require_captions and not sample_metadata.get("captions"):
                            # Captions may live as sibling .txt tar members instead of embedded
                            # metadata (e.g. cc12m); get_caption() range-reads those at runtime.
                            # get_shard_metadata returns a flat mapping keyed by member filename.
                            caption_member = Path(str(entry["filename"])).with_suffix(".txt").name
                            if caption_member not in shard_metadata:
                                statistics["skipped"]["caption_missing"] += 1
                                continue
                        aspect_ratio_bucket_updates.setdefault(bucket_key, []).append(sample_path)
                        metadata_updates[sample_path] = sample_metadata
                        if sample_metadata.get("captions"):
                            self.caption_cache[sample_path] = sample_metadata["captions"]
                        statistics["total_processed"] += 1

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
                        self._save_caption_cache()
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
        self._save_caption_cache()
        self.save_cache(enforce_constraints=True)
        self._sync_image_files_with_buckets()
        if self.bucket_report:
            self.bucket_report.update_statistics(statistics)
            self.bucket_report.record_bucket_snapshot("post_refresh", self.aspect_ratio_bucket_indices)

    def __len__(self):
        """
        Returns:
            int: The number of complete batches available across aspect ratio buckets.
        """

        def repeat_len(bucket):
            return len(bucket) * (self.repeats + 1)

        return sum(
            (repeat_len(bucket) + (self.batch_size - 1)) // self.batch_size
            for bucket in self.aspect_ratio_bucket_indices.values()
            if repeat_len(bucket) >= self.batch_size
        )

    def refresh_buckets(self, rank: int = None):
        self.compute_aspect_ratio_bucket_indices()
        logger.debug("Refreshing webshart buckets for rank %s via data_backend id %s.", rank, self.id)
        return
