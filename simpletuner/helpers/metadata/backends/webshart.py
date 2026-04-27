import json
import logging
import os
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.data_backend.webshart import WebshartDataBackend
from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.metadata.backends.base import MetadataBackend
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
        if self.dataset_type not in {DatasetType.IMAGE, DatasetType.CONDITIONING, DatasetType.EVAL}:
            raise ValueError("WebshartMetadataBackend currently supports image-like datasets only.")

        self.caption_cache: Dict[str, Union[str, List[str]]] = {}

        context = accelerator.main_process_first() if hasattr(accelerator, "main_process_first") else nullcontext()
        with context:
            self.reload_cache()
            self.load_image_metadata()
            self._load_caption_cache()
        if hasattr(accelerator, "wait_for_everyone"):
            accelerator.wait_for_everyone()

    def caption_cache_entry(self, index: str) -> Optional[Union[str, List[str]]]:
        return self.caption_cache.get(str(index), None)

    def _caption_cache_path(self):
        return f"{self.cache_file}_captions.json"

    def _load_caption_cache(self) -> None:
        path = self._caption_cache_path()
        if self.data_backend.exists(path):
            try:
                raw = self.data_backend.read(path)
                self.caption_cache = json.loads(raw)
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

    def _metadata_for_entry(self, shard_metadata: dict, filename: str, entry: dict) -> dict:
        file_metadata = shard_metadata.get(filename, {}) or {}
        width = entry.get("width", file_metadata.get("width"))
        height = entry.get("height", file_metadata.get("height"))
        if width is None or height is None:
            return {}

        metadata = {
            "original_size": (int(width), int(height)),
            "webshart": {
                "shard_idx": entry.get("shard_idx"),
                "sample_idx": entry.get("sample_idx"),
                "filename": filename,
                "offset": entry.get("offset"),
                "size": entry.get("size"),
                "json_path": file_metadata.get("json_path"),
            },
        }
        if "captions" in file_metadata:
            metadata["captions"] = file_metadata["captions"]
        if "json_metadata" in file_metadata:
            metadata["json_metadata"] = file_metadata["json_metadata"]
        if "json_path" in file_metadata:
            metadata["json_path"] = file_metadata["json_path"]
        return metadata

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
        return str(round(aspect_ratio, 2)), sample_metadata

    def compute_aspect_ratio_bucket_indices(self, ignore_existing_cache: bool = False, progress_callback=None):
        logger.info("Building aspect ratio buckets from webshart metadata...")
        statistics = {
            "total_processed": 0,
            "skipped": {
                "already_exists": 0,
                "metadata_missing": 0,
                "too_small": 0,
                "error": 0,
            },
        }

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
        aspect_ratio_bucket_updates: Dict[str, list[str]] = {}
        metadata_updates: Dict[str, dict] = {}
        shard_metadata_cache: Dict[int, dict] = {}
        shard_indices = self._all_shard_indices()

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

            shard_bucket_results = self.data_backend.list_shard_sample_aspect_buckets([shard_idx])
            if not shard_bucket_results:
                continue
            shard_bucket_data = shard_bucket_results[0]
            shard_metadata = shard_metadata_cache.setdefault(shard_idx, self.data_backend.get_shard_metadata(shard_idx))
            for entries in shard_bucket_data.get("buckets", {}).values():
                for entry in entries:
                    if (
                        self.max_num_samples is not None
                        and statistics["total_processed"] + len(existing_files) >= self.max_num_samples
                    ):
                        break
                    processed_entries += 1
                    try:
                        filename = str(entry["filename"])
                        entry = {**entry, "shard_idx": shard_idx}
                        sample_path = self._sample_id_from_entry(shard_idx, entry)
                        if sample_path in existing_files:
                            continue

                        sample_metadata = self._metadata_for_entry(shard_metadata, filename, entry)
                        prepared = self._prepare_metadata(sample_path, sample_metadata)
                        if prepared is None:
                            if sample_metadata:
                                statistics["skipped"]["too_small"] += 1
                            else:
                                statistics["skipped"]["metadata_missing"] += 1
                            continue
                        bucket_key, sample_metadata = prepared
                        aspect_ratio_bucket_updates.setdefault(bucket_key, []).append(sample_path)
                        metadata_updates[sample_path] = sample_metadata
                        if sample_metadata.get("captions"):
                            self.caption_cache[sample_path] = sample_metadata["captions"]
                        statistics["total_processed"] += 1
                    except Exception as exc:
                        logger.error("Error processing webshart bucket entry %s: %s", entry, exc)
                        statistics["skipped"]["error"] += 1

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
                if (
                    self.max_num_samples is not None
                    and statistics["total_processed"] + len(existing_files) >= self.max_num_samples
                ):
                    break
            if progress_callback is not None:
                progress_callback(shard_idx + 1, len(shard_indices))

        for key, value in aspect_ratio_bucket_updates.items():
            self.aspect_ratio_bucket_indices.setdefault(key, []).extend(value)
        for path, metadata in metadata_updates.items():
            self.set_metadata_by_filepath(path, metadata, update_json=False)

        self.filtering_statistics = statistics
        self.save_image_metadata()
        self._save_caption_cache()
        self.save_cache(enforce_constraints=True)
        if self.bucket_report:
            self.bucket_report.update_statistics(statistics)
            self.bucket_report.record_bucket_snapshot("post_refresh", self.aspect_ratio_bucket_indices)

    def refresh_buckets(self, rank: int = None):
        self.compute_aspect_ratio_bucket_indices()
        logger.debug("Refreshing webshart buckets for rank %s via data_backend id %s.", rank, self.id)
        return
