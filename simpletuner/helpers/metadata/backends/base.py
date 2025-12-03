import logging
import os
import threading
import time
from math import ceil, floor
from multiprocessing import Process, Queue
from pathlib import Path
from random import shuffle

# For semaphore
from threading import Semaphore, Thread
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.data_backend.dataset_types import DatasetType, ensure_dataset_type
from simpletuner.helpers.multiaspect.image import MultiaspectImage
from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger("BaseMetadataBackend")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")

DEFAULT_AUDIO_BUCKET_INTERVAL = 3.0


class MetadataBackend:
    def __init__(
        self,
        id: str,
        instance_data_dir: str,
        cache_file: str,
        metadata_file: str,
        data_backend: BaseDataBackend,
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
    ):
        self.id = id
        if self.id != data_backend.id:
            raise ValueError(f"BucketManager ID ({self.id}) must match the DataBackend ID ({data_backend.id}).")
        self.accelerator = accelerator
        self.should_abort = False
        self.data_backend = data_backend
        self.batch_size = int(batch_size)
        self.repeats = int(repeats)
        self.instance_data_dir = instance_data_dir
        if cache_file_suffix is not None:
            cache_file = f"{cache_file}_{cache_file_suffix}"
            metadata_file = f"{metadata_file}_{cache_file_suffix}"
        self.cache_file = Path(f"{cache_file}.json")
        self.metadata_file = Path(f"{metadata_file}.json")
        self._aspect_ratio_bucket_indices = {}
        self.image_metadata = {}  # Store image metadata
        self.seen_images = {}
        self.config = {}
        self.dataset_config = StateTracker.get_data_backend_config(self.id) or {}
        dataset_type_value = self.dataset_config.get("dataset_type")
        try:
            self.dataset_type = ensure_dataset_type(dataset_type_value, default=DatasetType.IMAGE)
        except ValueError:
            self.dataset_type = DatasetType.IMAGE
        self.audio_config = self._extract_audio_config()
        self.bucket_strategy = self._resolve_bucket_strategy()
        self.audio_duration_interval = self._resolve_audio_duration_interval()
        self.audio_max_duration_seconds = self._resolve_audio_max_duration()
        self.audio_truncation_mode = self._resolve_audio_truncation_mode()
        self.audio_lyrics_format = self._resolve_audio_lyrics_format()
        self.audio_lyrics_extension = self._resolve_audio_lyrics_extension()
        self.audio_lyrics_suffix = self._resolve_audio_lyrics_suffix()
        self.reload_cache()
        self.resolution = float(resolution)
        self.resolution_type = resolution_type
        self.delete_problematic_images = delete_problematic_images
        self.delete_unwanted_images = delete_unwanted_images
        self.metadata_update_interval = metadata_update_interval
        self.minimum_image_size = float(minimum_image_size) if minimum_image_size else None
        self.minimum_aspect_ratio = float(minimum_aspect_ratio) if minimum_aspect_ratio else None
        self.maximum_aspect_ratio = float(maximum_aspect_ratio) if maximum_aspect_ratio else None
        self.maximum_num_frames = float(maximum_num_frames) if maximum_num_frames else None
        self.minimum_num_frames = float(minimum_num_frames) if minimum_num_frames else None
        self.num_frames = float(num_frames) if num_frames else None
        self.image_metadata_loaded = False
        self.vae_output_scaling_factor = 8
        self.metadata_semaphor = Semaphore()
        # When a multi-gpu system splits the buckets, we no longer update.
        self.read_only = False
        self.bucket_report = None

    def load_metadata(self):
        raise NotImplementedError

    def save_metadata(self):
        raise NotImplementedError

    def clear_metadata(self):
        """clear metadata cache"""
        self.image_metadata = {}
        self.image_metadata_loaded = False
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        logger.info(f"({self.id}) Cleared metadata cache.")

    def attach_bucket_report(self, bucket_report) -> None:
        """Attach a BucketReport instance for lightweight telemetry."""
        self.bucket_report = bucket_report
        if bucket_report:
            bucket_report.set_constraints(
                resolution_type=self.resolution_type,
                minimum_image_size=self.minimum_image_size,
                minimum_aspect_ratio=self.minimum_aspect_ratio,
                maximum_aspect_ratio=self.maximum_aspect_ratio,
                bucket_strategy=self.bucket_strategy,
                duration_interval=self.audio_duration_interval if self.dataset_type is DatasetType.AUDIO else None,
                max_duration_seconds=self.audio_max_duration_seconds if self.dataset_type is DatasetType.AUDIO else None,
                truncation_mode=self.audio_truncation_mode if self.dataset_type is DatasetType.AUDIO else None,
            )

    @property
    def aspect_ratio_bucket_indices(self):
        """Get aspect ratio bucket indices."""
        return self._aspect_ratio_bucket_indices

    @aspect_ratio_bucket_indices.setter
    def aspect_ratio_bucket_indices(self, value):
        """Set aspect ratio bucket indices with debug tracking."""
        if hasattr(self, "_aspect_ratio_bucket_indices"):
            old_count = sum(len(v) for v in self._aspect_ratio_bucket_indices.values())
            new_count = sum(len(v) for v in value.values()) if value else 0
            if old_count != new_count:
                logger.debug(
                    f"aspect_ratio_bucket_indices changed for {self.id}: "
                    f"{old_count} -> {new_count} images. "
                    f"Old buckets: {list(self._aspect_ratio_bucket_indices.keys())}, "
                    f"New buckets: {list(value.keys()) if value else []}"
                )
        self._aspect_ratio_bucket_indices = value

    def _extract_audio_config(self) -> Dict[str, Any]:
        if self.dataset_config is None:
            return {}
        config = self.dataset_config.get("audio")
        return config if isinstance(config, dict) else {}

    def _resolve_bucket_strategy(self) -> str:
        default_strategy = "duration" if self.dataset_type is DatasetType.AUDIO else "aspect_ratio"
        strategy = self.audio_config.get("bucket_strategy") if isinstance(self.audio_config, dict) else None
        if not strategy:
            strategy = self.dataset_config.get("bucket_strategy")
        if isinstance(strategy, str):
            return strategy.lower()
        return default_strategy

    def _resolve_audio_duration_interval(self) -> Optional[float]:
        if self.dataset_type is not DatasetType.AUDIO:
            return None
        interval = (self.audio_config or {}).get("duration_interval")
        try:
            interval_value = float(interval)
        except (TypeError, ValueError):
            interval_value = DEFAULT_AUDIO_BUCKET_INTERVAL
        if interval_value <= 0:
            interval_value = DEFAULT_AUDIO_BUCKET_INTERVAL
        return interval_value

    def _resolve_audio_max_duration(self) -> Optional[float]:
        if self.dataset_type is not DatasetType.AUDIO:
            return None
        max_duration = (self.audio_config or {}).get("max_duration_seconds")
        if max_duration is None:
            return None
        try:
            value = float(max_duration)
        except (TypeError, ValueError):
            logger.warning(
                f"(id={self.id}) Ignoring audio.max_duration_seconds value '{max_duration}' because it is not numeric."
            )
            return None
        if value <= 0:
            logger.warning(f"(id={self.id}) Ignoring non-positive audio.max_duration_seconds value '{max_duration}'.")
            return None
        return value

    @staticmethod
    def _format_duration_value(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")

    def _build_audio_bucket_key(self, duration_seconds: Optional[float]) -> str:
        bucket_key, _ = self._compute_audio_bucket(duration_seconds)
        return bucket_key

    def _compute_audio_bucket(self, duration_seconds: Optional[float]) -> tuple[str, Optional[float]]:
        if duration_seconds is None:
            return "audio", None
        interval = self.audio_duration_interval or DEFAULT_AUDIO_BUCKET_INTERVAL
        if interval <= 0:
            interval = DEFAULT_AUDIO_BUCKET_INTERVAL
        safe_duration = max(float(duration_seconds), 0.0)
        multiples = floor(safe_duration / interval)
        if multiples <= 0:
            truncated = safe_duration
        else:
            truncated = multiples * interval
        truncated = min(safe_duration, truncated)
        if truncated == 0.0 and safe_duration > 0.0:
            truncated = safe_duration
        bucket_key = f"{self._format_duration_value(truncated)}s"
        return bucket_key, truncated

    def set_metadata(self, metadata_backend, update_json: bool = True):
        if not isinstance(metadata_backend, MetadataBackend):
            raise TypeError(f"Expected MetadataBackend instance, got {type(metadata_backend)}.")
        self.image_metadata = metadata_backend.get_metadata()
        self.aspect_ratio_bucket_indices = metadata_backend.aspect_ratio_bucket_indices.copy()

        self.image_metadata_loaded = True
        if update_json:
            self.save_image_metadata()

    def get_metadata(self):
        if not self.image_metadata_loaded:
            self.load_image_metadata()
        return self.image_metadata

    def print_debug_info(self):
        logger.info(
            f"\n-> MetadataBackend ID: {self.id}, "
            f"\n-> Instance Data Dir: {self.instance_data_dir}, "
            f"\n-> Cache File: {self.cache_file}, "
            f"\n-> Metadata File: {self.metadata_file}, "
            f"\n-> Aspect Ratio Buckets: {len(self.aspect_ratio_bucket_indices)}"
        )

    def set_readonly(self):
        self.read_only = True
        logger.info(f"MetadataBackend {self.id} is now read-only.")

    def _bucket_worker(
        self,
        tqdm_queue,
        files,
        aspect_ratio_bucket_indices_queue,
        metadata_updates_queue,
        written_files_queue,
        existing_files_set,
    ):
        """worker process to bucket files and extract metadata"""
        local_aspect_ratio_bucket_indices = {}
        local_metadata_updates = {}
        processed_file_list = set()
        processed_file_count = 0
        statistics = {
            "total_processed": 0,
            "skipped": {
                "already_exists": 0,
                "metadata_missing": 0,
                "not_found": 0,
                "too_small": 0,
                "too_long": 0,
                "other": 0,
            },
        }

        for file in files:
            if str(file) not in existing_files_set:
                logger.debug(f"Processing file {file}.")
                try:
                    local_aspect_ratio_bucket_indices = self._process_for_bucket(
                        file,
                        local_aspect_ratio_bucket_indices,
                        metadata_updates=local_metadata_updates,
                        delete_problematic_images=self.delete_problematic_images,
                        statistics=statistics,
                    )
                except Exception as e:
                    logger.error(f"Error processing file {file}. Reason: {e}. Skipping.")
                    statistics["skipped"]["error"] += 1
                logger.debug(
                    f"Statistics: {statistics}, total: {sum([len(bucket) for bucket in local_aspect_ratio_bucket_indices.values()])}"
                )
                processed_file_count += 1
                statistics["total_processed"] = processed_file_count
                processed_file_list.add(file)
            else:
                statistics["skipped"]["already_exists"] += 1
            tqdm_queue.put(1)
            if processed_file_count % 500 == 0:
                # periodic queue updates to avoid memory buildup
                if aspect_ratio_bucket_indices_queue is not None:
                    aspect_ratio_bucket_indices_queue.put(local_aspect_ratio_bucket_indices)
                if written_files_queue is not None:
                    written_files_queue.put(processed_file_list)
                metadata_updates_queue.put(local_metadata_updates)
                local_aspect_ratio_bucket_indices = {}
                local_metadata_updates = {}
                processed_file_list = set()
        if aspect_ratio_bucket_indices_queue is not None and local_aspect_ratio_bucket_indices:
            aspect_ratio_bucket_indices_queue.put(local_aspect_ratio_bucket_indices)
        if local_metadata_updates:
            metadata_updates_queue.put(local_metadata_updates)
            metadata_updates_queue.put(("statistics", statistics))
        time.sleep(0.001)
        logger.debug("Bucket worker completed processing. Returning to main thread.")

    def compute_aspect_ratio_bucket_indices(self, ignore_existing_cache: bool = False):
        """compute aspect ratio buckets - main processing function"""
        logger.info("Discovering new files...")
        new_files = self._discover_new_files(ignore_existing_cache=ignore_existing_cache)

        # Audio datasets use duration-based bucketing, not spatial aspect ratios.
        if self.dataset_type is DatasetType.AUDIO:
            logger.info("Audio dataset detected; rebuilding duration buckets.")
            self.load_image_metadata()
            self.aspect_ratio_bucket_indices = {}
            for sample_path, meta in self.image_metadata.items():
                duration_seconds = None
                if isinstance(meta, dict):
                    duration_seconds = (
                        meta.get("bucket_duration_seconds")
                        or meta.get("duration_seconds")
                        or meta.get("original_duration_seconds")
                    )
                bucket_key, truncated = self._compute_audio_bucket(duration_seconds)
                self.aspect_ratio_bucket_indices.setdefault(bucket_key, []).append(sample_path)
                if truncated is not None and isinstance(meta, dict):
                    meta["bucket_duration_seconds"] = truncated
                    meta["duration_seconds"] = truncated
            if self.bucket_report:
                self.bucket_report.record_bucket_snapshot("post_refresh", self.aspect_ratio_bucket_indices)
            logger.info("Audio bucket rebuild complete.")
            return

        existing_files_set = set().union(*self.aspect_ratio_bucket_indices.values())
        logger.info(
            f"Compressed {len(existing_files_set)} existing files from {len(self.aspect_ratio_bucket_indices.values())}."
        )
        if self.bucket_report:
            self.bucket_report.record_stage(
                "existing_cache",
                sample_count=len(existing_files_set),
                bucket_count=len(self.aspect_ratio_bucket_indices),
            )
        aggregated_statistics = {
            "total_processed": 0,
            "skipped": {
                "already_exists": len(existing_files_set),
                "metadata_missing": 0,
                "not_found": 0,
                "too_small": 0,
                "too_long": 0,
                "other": 0,
            },
        }
        if self.bucket_report:
            self.bucket_report.record_stage(
                "new_files_to_process",
                sample_count=len(new_files),
                ignore_existing_cache=ignore_existing_cache,
            )
        if not new_files:
            logger.info("No new files discovered. Doing nothing.")
            logger.info(f"Statistics: {aggregated_statistics}")
            if self.bucket_report:
                self.bucket_report.update_statistics(aggregated_statistics)
                self.bucket_report.record_bucket_snapshot("post_refresh", self.aspect_ratio_bucket_indices)
            return
        num_cpus = StateTracker.get_args().aspect_bucket_worker_count
        files_split = np.array_split(new_files, num_cpus)

        metadata_updates_queue = Queue()
        written_files_queue = Queue()
        tqdm_queue = Queue()
        aspect_ratio_bucket_indices_queue = Queue()
        try:
            self.load_image_metadata()
        except Exception as e:
            if ignore_existing_cache:
                logger.warning(f"Error loading image metadata, creating new metadata cache: {e}")
                self.image_metadata = {}
            else:
                raise Exception(
                    f"Error loading image metadata. You may have to remove the metadata json file '{self.metadata_file}' and VAE cache manually: {e}"
                )
        worker_cls = Process if StateTracker.get_args().enable_multiprocessing else Thread
        workers = [
            worker_cls(
                target=self._bucket_worker,
                args=(
                    tqdm_queue,
                    file_shard,
                    aspect_ratio_bucket_indices_queue,
                    metadata_updates_queue,
                    written_files_queue,
                    existing_files_set,
                ),
            )
            for file_shard in files_split
        ]

        for worker in workers:
            worker.start()
        last_write_time = time.time()
        written_files = set()
        with tqdm(
            desc="Generating aspect bucket cache",
            total=len(new_files),
            leave=False,
            ncols=125,
            miniters=int(len(new_files) / 100),
        ) as pbar:
            if self.should_abort:
                logger.info("Aborting aspect bucket update.")
                return
            while (
                any(worker.is_alive() for worker in workers)
                or not tqdm_queue.empty()
                or not aspect_ratio_bucket_indices_queue.empty()
                or not metadata_updates_queue.empty()
                or not written_files_queue.empty()
            ):
                current_time = time.time()
                while not tqdm_queue.empty():
                    pbar.update(tqdm_queue.get())
                while not aspect_ratio_bucket_indices_queue.empty():
                    aspect_ratio_bucket_indices_update = aspect_ratio_bucket_indices_queue.get()
                    for key, value in aspect_ratio_bucket_indices_update.items():
                        self.aspect_ratio_bucket_indices.setdefault(key, []).extend(value)
                while not metadata_updates_queue.empty():
                    metadata_update = metadata_updates_queue.get()
                    if type(metadata_update) is tuple and metadata_update[0] == "statistics":
                        logger.debug(f"Received statistics update: {metadata_update[1]}")
                        for reason, count in metadata_update[1]["skipped"].items():
                            aggregated_statistics["skipped"][reason] += count
                        aggregated_statistics["total_processed"] += metadata_update[1]["total_processed"]
                        continue
                    for filepath, meta in metadata_update.items():
                        self.set_metadata_by_filepath(filepath=filepath, metadata=meta, update_json=False)
                while not written_files_queue.empty():
                    written_files_batch = written_files_queue.get()
                    written_files.update(written_files_batch)  # Use update for sets

                processing_duration = current_time - last_write_time
                if processing_duration >= self.metadata_update_interval:
                    logger.debug(
                        f"In-flight metadata update after {processing_duration} seconds. Saving {len(self.image_metadata)} metadata entries and {len(self.aspect_ratio_bucket_indices)} aspect bucket lists."
                    )
                    self.save_cache(enforce_constraints=False)
                    self.save_image_metadata()
                    last_write_time = current_time

                time.sleep(0.001)

        for worker in workers:
            worker.join()
        logger.info(f"Sample processing statistics: {aggregated_statistics}")
        self.save_image_metadata()
        self.save_cache(enforce_constraints=True)
        logger.info("Completed aspect bucket update.")
        if self.bucket_report:
            self.bucket_report.update_statistics(aggregated_statistics)
            self.bucket_report.record_bucket_snapshot("post_refresh", self.aspect_ratio_bucket_indices)

    def split_buckets_between_processes(self, gradient_accumulation_steps=1, apply_padding=False):
        """split bucket contents across processes for distributed training"""
        new_aspect_ratio_bucket_indices = {}
        total_samples = sum([len(bucket) for bucket in self.aspect_ratio_bucket_indices.values()])
        logger.debug(f"Count of items before split: {total_samples}")

        num_processes = self.accelerator.num_processes
        # Evaluation datasets are not trained, so gradient accumulation does not apply.
        dataset_type = getattr(self, "dataset_type", None)
        try:
            dataset_type_enum = ensure_dataset_type(dataset_type) if dataset_type is not None else DatasetType.IMAGE
        except Exception:
            dataset_type_enum = DatasetType.IMAGE

        grad_accumulation_for_bucketing = 1 if dataset_type_enum is DatasetType.EVAL else gradient_accumulation_steps
        if dataset_type_enum is DatasetType.EVAL and gradient_accumulation_steps != 1:
            logger.debug(
                f"(id={self.id}) Ignoring gradient accumulation for eval dataset; using 1 instead of {gradient_accumulation_steps}."
            )

        effective_batch_size = self.batch_size * num_processes * grad_accumulation_for_bucketing
        if self.bucket_report:
            self.bucket_report.set_constraints(effective_batch_size=effective_batch_size)

        # Early validation: check if configuration is mathematically impossible
        buckets_that_will_fail = []
        for bucket, images in self.aspect_ratio_bucket_indices.items():
            total_img_count_incl_repeats = len(images) * (self.repeats + 1)
            if total_img_count_incl_repeats < effective_batch_size:
                buckets_that_will_fail.append(
                    {
                        "bucket": bucket,
                        "images": len(images),
                        "samples_with_repeats": total_img_count_incl_repeats,
                    }
                )

        if buckets_that_will_fail:
            # Calculate what repeats would be needed
            min_repeats_needed = {}
            for bucket_info in buckets_that_will_fail:
                images = bucket_info["images"]
                needed_repeats = ceil(effective_batch_size / images) - 1
                min_repeats_needed[bucket_info["bucket"]] = needed_repeats

            max_needed_repeats = max(min_repeats_needed.values())

            # Check if dataset oversubscription is allowed
            args = StateTracker.get_args()
            allow_oversubscription = args.allow_dataset_oversubscription

            # Check if user manually configured repeats in their backend config
            backend_config = StateTracker.get_data_backend_config(self.id) or {}
            user_set_repeats = "repeats" in backend_config

            if allow_oversubscription and not user_set_repeats:
                # Automatically adjust repeats to make training possible
                original_repeats = self.repeats
                self.repeats = max_needed_repeats
                logger.warning(
                    f"(id={self.id}) Dataset oversubscription enabled: automatically increasing repeats from {original_repeats} to {self.repeats}\n"
                    f"  - This allows training with {total_samples} samples across {num_processes} GPUs\n"
                    f"  - Effective batch size: {effective_batch_size}\n"
                    f"  - Each sample will be seen {self.repeats + 1} times per epoch"
                )
                # Validation passed with adjustment, continue
            else:
                # Build error message
                error_msg = (
                    f"Dataset configuration will produce zero usable batches.\n"
                    f"\nCurrent configuration:\n"
                    f"  - Dataset ID: {self.id}\n"
                    f"  - Total samples: {total_samples}\n"
                    f"  - Repeats: {self.repeats}\n"
                    f"  - Batch size: {self.batch_size}\n"
                    f"  - Number of GPUs: {num_processes}\n"
                    f"  - Gradient accumulation steps: {grad_accumulation_for_bucketing}\n"
                    f"  - Effective batch size: {effective_batch_size}\n"
                    f"\nProblem: {len(buckets_that_will_fail)} bucket(s) have insufficient samples:\n"
                )

                for bucket_info in buckets_that_will_fail:
                    bucket = bucket_info["bucket"]
                    images = bucket_info["images"]
                    samples = bucket_info["samples_with_repeats"]
                    needed = min_repeats_needed[bucket]
                    error_msg += (
                        f"  - Bucket {bucket}: {images} samples × {self.repeats + 1} (with repeats) = {samples} samples\n"
                        f"    Need at least {effective_batch_size} samples to form one batch\n"
                        f"    Minimum repeats required: {needed}\n"
                    )

                error_msg += (
                    f"\nSolutions:\n"
                    f"  1. Reduce batch_size (current: {self.batch_size})\n"
                    f"  2. Reduce gradient_accumulation_steps (current: {grad_accumulation_for_bucketing})\n"
                    f"  3. Use fewer GPUs (current: {num_processes})\n"
                    f"  4. Increase repeats to at least {max_needed_repeats} (current: {self.repeats})\n"
                    f"  5. Add more samples to the dataset\n"
                    f"  6. Enable --allow_dataset_oversubscription to automatically adjust repeats\n"
                )

                if user_set_repeats:
                    error_msg += (
                        f"\nNote: You have manually set repeats={self.repeats} in your dataset config.\n"
                        f"--allow_dataset_oversubscription will not override manual repeats settings.\n"
                        f"Either increase your repeats value or remove it from the config to allow auto-adjustment.\n"
                    )

                raise ValueError(error_msg)

        should_shuffle_contents = os.environ.get("SIMPLETUNER_SHUFFLE_BUCKETS", "1") == "1"
        for bucket, images in self.aspect_ratio_bucket_indices.items():
            if should_shuffle_contents:
                logger.debug(f"Shuffling bucket {bucket} contents.")
                shuffle(images)
            total_img_count_incl_repeats = len(images) * (self.repeats + 1)
            num_batches = ceil(total_img_count_incl_repeats / effective_batch_size)
            trimmed_images = images[: num_batches * effective_batch_size]
            removed_for_trim = len(images) - len(trimmed_images)
            if removed_for_trim > 0 and self.bucket_report:
                self.bucket_report.record_bucket_event(
                    bucket=bucket,
                    reason="trimmed_for_effective_batch",
                    removed=removed_for_trim,
                    effective_batch_size=effective_batch_size,
                )
            if len(trimmed_images) == 0 and should_log():
                logger.error(
                    f"Bucket {bucket} has no samples after trimming because {len(images)} samples are not enough to satisfy an effective batch size of {effective_batch_size}."
                    " Lower your batch size, increase repeat count, or increase data pool size."
                )
                if self.bucket_report:
                    self.bucket_report.record_bucket_event(
                        bucket=bucket,
                        reason="insufficient_for_batch_after_trim",
                        removed=len(images),
                        effective_batch_size=effective_batch_size,
                    )

            with self.accelerator.split_between_processes(trimmed_images, apply_padding=apply_padding) as images_split:
                new_aspect_ratio_bucket_indices[bucket] = images_split

        self.aspect_ratio_bucket_indices = new_aspect_ratio_bucket_indices
        post_total = sum([len(bucket) for bucket in self.aspect_ratio_bucket_indices.values()])
        if self.bucket_report:
            self.bucket_report.record_bucket_snapshot("post_split", self.aspect_ratio_bucket_indices)
        if total_samples != post_total:
            self.read_only = True

        # Check if this backend has no samples after splitting (can happen with multi-GPU setups)
        if post_total == 0 and total_samples > 0:
            if should_log():
                logger.warning(
                    f"Backend {self.id} has no samples after splitting between processes. "
                    f"This can happen when using multiple GPUs with small datasets. "
                    f"Consider using a larger dataset or fewer GPUs. "
                    f"Original sample count: {total_samples}"
                )

        logger.debug(f"Count of items after split: {post_total}")

    def mark_as_seen(self, image_path):
        self.seen_images[image_path] = True

    def mark_batch_as_seen(self, image_paths):
        self.seen_images.update({image_path: True for image_path in image_paths})

    def is_seen(self, image_path):
        return self.seen_images.get(image_path, False)

    def reset_seen_images(self):
        self.seen_images.clear()

    def remove_image(self, image_path, bucket: str = None):
        """remove image from bucket(s)"""
        if not bucket:
            for bucket, images in self.aspect_ratio_bucket_indices.items():
                if image_path in images:
                    self.aspect_ratio_bucket_indices[bucket].remove(image_path)
                    break
        if image_path in self.aspect_ratio_bucket_indices[bucket]:
            self.aspect_ratio_bucket_indices[bucket].remove(image_path)

    def update_buckets_with_existing_files(self, existing_files: set):
        """remove non-existent files and duplicates from buckets"""
        logger.debug(
            f"Before updating, in all buckets, we had {sum([len(bucket) for bucket in self.aspect_ratio_bucket_indices.values()])}."
        )
        for bucket, images in self.aspect_ratio_bucket_indices.items():
            # dedupe while preserving order
            filtered_images = list(dict.fromkeys(img for img in images if img in existing_files))
            self.aspect_ratio_bucket_indices[bucket] = filtered_images
            removed = len(images) - len(filtered_images)
            if removed > 0 and self.bucket_report:
                self.bucket_report.record_bucket_event(
                    bucket=bucket,
                    reason="missing_or_duplicate",
                    removed=removed,
                )
        logger.debug(
            f"After updating, in all buckets, we had {sum([len(bucket) for bucket in self.aspect_ratio_bucket_indices.values()])}."
        )
        self.save_cache()

    def refresh_buckets(self, rank: int = None):
        """discover new files and clean up missing ones"""
        self.compute_aspect_ratio_bucket_indices()
        logger.debug(f"Refreshing buckets for rank {rank} via data_backend id {self.id}.")
        existing_files = StateTracker.get_image_files(data_backend_id=self.id)

        if not StateTracker.get_args().ignore_missing_files:
            self.update_buckets_with_existing_files(existing_files)
        return

    def _enforce_min_bucket_size(self):
        """remove undersized buckets and enforce resolution constraints"""
        if self.minimum_image_size is None:
            return

        logger.info(
            f"Enforcing minimum image size of {self.minimum_image_size}." " This could take a while for very-large datasets."
        )
        for bucket in self._iterate_buckets_with_progress("Enforcing minimum bucket size"):
            self._prune_small_buckets(bucket)
            if self.minimum_image_size is not None:
                self._enforce_resolution_constraints(bucket)
                # prune again after resolution filtering
                self._prune_small_buckets(bucket)

    def _enforce_min_aspect_ratio(self):
        """remove buckets below minimum aspect ratio"""
        if self.minimum_aspect_ratio is None or self.minimum_aspect_ratio == 0.0:
            return

        logger.info(
            f"Enforcing minimum aspect ratio of {self.minimum_aspect_ratio}."
            " This could take a while for very-large datasets."
        )
        for bucket in self._iterate_buckets_with_progress("Enforcing minimum aspect ratio"):
            if float(bucket) < self.minimum_aspect_ratio:
                removed = len(self.aspect_ratio_bucket_indices.get(bucket, []))
                if self.bucket_report and removed > 0:
                    self.bucket_report.record_bucket_event(
                        bucket=bucket,
                        reason="below_min_aspect_ratio",
                        removed=removed,
                        minimum_aspect_ratio=self.minimum_aspect_ratio,
                    )
                logger.info(f"Removing bucket {bucket} due to aspect ratio being less than {self.minimum_aspect_ratio}.")
                del self.aspect_ratio_bucket_indices[bucket]

    def _enforce_max_aspect_ratio(self):
        """remove buckets above maximum aspect ratio"""
        if self.maximum_aspect_ratio is None or self.maximum_aspect_ratio == 0.0:
            return

        logger.info(
            f"Enforcing maximum aspect ratio of {self.maximum_aspect_ratio}."
            " This could take a while for very-large datasets."
        )
        for bucket in self._iterate_buckets_with_progress("Enforcing maximum aspect ratio"):
            if float(bucket) > self.maximum_aspect_ratio:
                removed = len(self.aspect_ratio_bucket_indices.get(bucket, []))
                if self.bucket_report and removed > 0:
                    self.bucket_report.record_bucket_event(
                        bucket=bucket,
                        reason="above_max_aspect_ratio",
                        removed=removed,
                        maximum_aspect_ratio=self.maximum_aspect_ratio,
                    )
                logger.info(f"Removing bucket {bucket} due to aspect ratio being greater than {self.maximum_aspect_ratio}.")
                del self.aspect_ratio_bucket_indices[bucket]

    def _prune_small_buckets(self, bucket):
        """remove buckets smaller than batch size"""
        if StateTracker.get_args().disable_bucket_pruning:
            logger.warning("Not pruning small buckets, as --disable_bucket_pruning is provided.")
            return
        if (
            bucket in self.aspect_ratio_bucket_indices
            and (len(self.aspect_ratio_bucket_indices[bucket]) * (int(self.repeats) + 1)) < self.batch_size
        ):
            bucket_sample_count = len(self.aspect_ratio_bucket_indices[bucket])
            if self.bucket_report:
                self.bucket_report.record_bucket_event(
                    bucket=bucket,
                    reason="insufficient_for_batch",
                    removed=bucket_sample_count,
                    batch_size=self.batch_size,
                    repeats=int(self.repeats),
                )
            del self.aspect_ratio_bucket_indices[bucket]
            logger.warning(
                f"Removing bucket {bucket} due to insufficient samples; your batch size may be too large for the small quantity of data (batch_size={self.batch_size} > sample_count={bucket_sample_count})."
            )

    def _iterate_buckets_with_progress(self, desc: str):
        buckets = list(self.aspect_ratio_bucket_indices.keys())
        progress = tqdm(total=len(buckets), leave=False, desc=desc, disable=True if len(buckets) < 100 else False, ncols=125)
        try:
            for bucket in buckets:
                yield bucket
                try:
                    progress.update(1)
                except TypeError:
                    pass
        finally:
            close_fn = getattr(progress, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except TypeError:
                    pass

    def _enforce_resolution_constraints(self, bucket):
        """filter bucket images by resolution requirements"""
        if self.minimum_image_size is not None:
            if bucket not in self.aspect_ratio_bucket_indices:
                logger.debug(f"Bucket {bucket} was already removed due to insufficient samples.")
                return
            images = self.aspect_ratio_bucket_indices[bucket]
            total_before = len(images)
            self.aspect_ratio_bucket_indices[bucket] = [
                img
                for img in images
                if self.meets_resolution_requirements(
                    image_path=img,
                    image=None,
                )
            ]
            total_after = len(self.aspect_ratio_bucket_indices[bucket])
            total_lost = total_before - total_after
            if total_lost > 0:
                if self.bucket_report:
                    self.bucket_report.record_bucket_event(
                        bucket=bucket,
                        reason="resolution_filter",
                        removed=total_lost,
                        minimum_image_size=self.minimum_image_size,
                        resolution_type=self.resolution_type,
                    )
                logger.info(
                    f"Had {total_before} samples before and {total_lost} that did not meet the minimum image size requirement ({self.minimum_image_size})."
                )

    def meets_resolution_requirements(
        self,
        image_path: str = None,
        image: Image = None,
        image_metadata: dict = None,
    ):
        """check if image meets resolution and frame count requirements"""
        if self.dataset_config.get("dataset_type", None) in ["conditioning"]:
            return True
        if self.dataset_type is DatasetType.AUDIO:
            return True
        if image is None and (image_path is not None and image_metadata is None):
            metadata = self.get_metadata_by_filepath(image_path)
            if metadata is None:
                logger.warning(f"Metadata not found for image {image_path}.")
                return False
            width, height = metadata["original_size"]
        elif isinstance(image, np.ndarray):
            # video processing
            width, height = image.shape[2], image.shape[1]
            logger.debug(f"Checking resolution: {width}x{height}")
            if self.minimum_num_frames is not None:
                num_frames = image.shape[0]
                if num_frames < self.minimum_num_frames:
                    logger.debug(
                        f"Video has {num_frames} frames, which is less than the minimum required {self.minimum_num_frames}."
                    )
                    return False
            if self.maximum_num_frames is not None:
                num_frames = image.shape[0]
                if num_frames > self.maximum_num_frames:
                    logger.debug(
                        f"Video has {num_frames} frames, which is more than the maximum configured {self.maximum_num_frames}."
                    )
                    return False
        elif image is not None:
            width, height = image.size
        elif image_metadata is not None:
            width, height = image_metadata["original_size"]
        else:
            raise ValueError(
                f"meets_resolution_requirements expects an image_path"
                f" ({image_path}) or Image object ({image}), but received neither."
            )

        if self.minimum_image_size is None:
            return True

        if self.resolution_type == "pixel":
            return self.minimum_image_size <= width and self.minimum_image_size <= height
        elif self.resolution_type == "area":
            # convert megapixel value to pixels for comparison
            if self.minimum_image_size > 5:
                raise ValueError(
                    f"--minimum_image_size was given with a value of {self.minimum_image_size} but resolution_type is area, which means this value is most likely too large. Please use a value less than 5."
                )
            minimum_image_size = self.minimum_image_size * 1_000_000
            if (
                StateTracker.get_data_backend_config(self.id).get("crop", False)
                and StateTracker.get_data_backend_config(self.id).get("crop_aspect", "square") == "square"
            ):
                # square crop needs minimum edge length check
                pixel_edge_len = floor(np.sqrt(minimum_image_size))
                if not (pixel_edge_len <= width and pixel_edge_len <= height):
                    return False
            return minimum_image_size <= width * height
        else:
            raise ValueError(
                f"BucketManager.meets_resolution_requirements received unexpected value for resolution_type: {self.resolution_type}"
            )

    def handle_incorrect_bucket(self, image_path: str, bucket: str, actual_bucket: str, save_cache: bool = True):
        """move incorrectly bucketed image to proper bucket"""
        logger.debug(f"Found an image in bucket {bucket} it doesn't belong in, when actually it is: {actual_bucket}")
        self.remove_image(image_path, bucket)
        if actual_bucket in self.aspect_ratio_bucket_indices:
            logger.debug("Moved image to bucket, it already existed.")
            self.aspect_ratio_bucket_indices[actual_bucket].append(image_path)
        else:
            logger.debug("Created new bucket for that pesky image.")
            self.aspect_ratio_bucket_indices[actual_bucket] = [image_path]
        if save_cache:
            self.save_cache()

    def handle_small_image(self, image_path: str, bucket: str, delete_unwanted_images: bool):
        """remove or delete undersized image"""
        if delete_unwanted_images:
            try:
                logger.warning(f"Image {image_path} too small: DELETING image and continuing search.")
                self.data_backend.delete(image_path)
            except Exception:
                logger.debug(f"Image {image_path} was already deleted. Another GPU must have gotten to it.")
        else:
            logger.warning(
                f"Image {image_path} too small, but --delete_unwanted_images is not provided, so we simply ignore and remove from bucket."
            )
        self.remove_image(image_path, bucket)

    def has_single_underfilled_bucket(self):
        """check for single bucket with insufficient samples"""
        if len(self.aspect_ratio_bucket_indices) != 1:
            return False

        bucket = list(self.aspect_ratio_bucket_indices.keys())[0]
        if (len(self.aspect_ratio_bucket_indices[bucket]) * (int(self.repeats) + 1)) < self.batch_size:
            return True

        return False

    def read_cache(self):
        return self.aspect_ratio_bucket_indices

    def get_metadata_attribute_by_filepath(self, filepath: str, attribute: str):
        metadata = self.get_metadata_by_filepath(filepath)
        if metadata:
            return metadata.get(attribute, None)
        else:
            return None

    def set_metadata_attribute_by_filepath(self, filepath: str, attribute: str, value: any, update_json: bool = True):
        metadata = self.get_metadata_by_filepath(filepath) or {}
        metadata[attribute] = value
        return self.set_metadata_by_filepath(filepath, metadata, update_json)

    def update_metadata_attribute(self, filepath: str, attribute: str, value: any, update_json: bool = False) -> bool:
        """
        Lightweight helper used by batch utilities to amend cached metadata without
        requiring every backend to implement its own setter.
        """
        if getattr(self, "read_only", False):
            logger.debug(f"Skipping metadata update for {filepath}:{attribute} because backend is read-only.")
            return False
        self.set_metadata_attribute_by_filepath(filepath, attribute, value, update_json=update_json)
        return True

    def set_metadata_by_filepath(self, filepath: str, metadata: dict, update_json: bool = True):
        with self.metadata_semaphor:
            logger.debug(f"Setting metadata for {filepath} to {metadata}.")
            self.image_metadata[filepath] = metadata
            if update_json:
                self.save_image_metadata()

    def get_metadata_by_filepath(self, filepath: str):
        """retrieve metadata by filepath, trying both relative and absolute paths"""
        if type(filepath) not in [tuple, list]:
            filepath = [filepath]
        if type(filepath) is tuple or type(filepath) is list:
            for path in filepath:
                abs_path = self.data_backend.get_abs_path(path)
                if path in self.image_metadata:
                    result = self.image_metadata.get(path, None)
                    if result is not None:
                        return result
                elif abs_path in self.image_metadata:
                    filepath = abs_path
                    result = self.image_metadata.get(filepath, None)
                    if result is not None:
                        return result
            return None

        meta = self.image_metadata.get(filepath, None)
        return meta

    def scan_for_metadata(self):
        """scan for new files and update metadata only (no bucketing)"""
        logger.info(f"Loading metadata from {self.metadata_file}")
        self.load_image_metadata()
        logger.debug(f"A subset of the available metadata: {list(self.image_metadata.keys())[:5]}")
        logger.info("Discovering new samples for metadata scan...")
        new_files = self._discover_new_files(for_metadata=True)
        if not new_files:
            logger.info("No new files discovered. Exiting.")
            return

        existing_files_set = {existing_file for existing_file in self.image_metadata.keys()}

        num_cpus = 8
        files_split = np.array_split(new_files, num_cpus)

        metadata_updates_queue = Queue()
        tqdm_queue = Queue()
        worker_cls = Process if StateTracker.get_args().enable_multiprocessing else Thread
        workers = [
            worker_cls(
                target=self._bucket_worker,
                args=(
                    tqdm_queue,
                    file_shard,
                    None,  # no bucket updates
                    metadata_updates_queue,
                    None,  # no written files tracking
                    existing_files_set,
                ),
            )
            for file_shard in files_split
        ]

        for worker in workers:
            worker.start()

        with tqdm(
            desc="Scanning image metadata",
            total=len(new_files),
            leave=False,
            ncols=125,
        ) as pbar:
            while any(worker.is_alive() for worker in workers):
                while not tqdm_queue.empty():
                    pbar.update(tqdm_queue.get())

                while not metadata_updates_queue.empty():
                    metadata_update = metadata_updates_queue.get()
                    logger.debug(f"Received type of metadata update: {type(metadata_update)}, contents: {metadata_update}")
                    if type(metadata_update) == dict:
                        for filepath, meta in metadata_update.items():
                            self.set_metadata_by_filepath(filepath=filepath, metadata=meta, update_json=False)

        for worker in workers:
            worker.join()

        self.save_image_metadata()
        self.save_cache(enforce_constraints=True)
        logger.info("Completed metadata update.")

    def handle_vae_cache_inconsistencies(self, vae_cache, vae_cache_behavior: str):
        """handle vae cache inconsistencies with bucket indices"""
        if "deepfloyd" in StateTracker.get_args().model_type:
            return
        if vae_cache_behavior not in ["sync", "recreate"]:
            raise ValueError("Invalid VAE cache behavior specified.")
        logger.info("Scanning VAE cache for inconsistencies with aspect buckets...")
        try:
            for cache_file, cache_content in vae_cache.scan_cache_contents():
                if cache_content is None:
                    continue
                if vae_cache_behavior == "sync":
                    expected_bucket = str(self._get_aspect_ratio_from_tensor(cache_content))
                    self._modify_cache_entry_bucket(cache_file, expected_bucket)
                elif vae_cache_behavior == "recreate":
                    if self.is_cache_inconsistent(vae_cache, cache_file, cache_content):
                        threading.Thread(
                            target=self.data_backend.delete,
                            args=(cache_file,),
                            daemon=True,
                        ).start()
        except Exception as e:
            logger.debug(f"Error running VAE cache scan: {e}")
            return

        self.save_cache()

    def _recalculate_target_resolution(self, original_aspect_ratio: float) -> tuple:
        """recalculate target resolution based on backend config"""
        resolution_type = StateTracker.get_data_backend_config(self.id)["resolution_type"]
        resolution = StateTracker.get_data_backend_config(self.id)["resolution"]
        if resolution_type == "pixel":
            return MultiaspectImage.calculate_new_size_by_pixel_edge(original_aspect_ratio, int(resolution))
        elif resolution_type == "area":
            if original_aspect_ratio is None:
                raise ValueError("Original aspect ratio must be provided for area-based resolution.")
            return MultiaspectImage.calculate_new_size_by_pixel_area(original_aspect_ratio, resolution)

    def is_cache_inconsistent(self, vae_cache, cache_file, cache_content):
        """check if cache file tensor doesn't match expected bucket"""
        if cache_content is None:
            return True
        # check for corrupt tensor values
        if torch.isnan(cache_content).any() or torch.isinf(cache_content).any():
            logger.warning(f"Cache file {cache_file} contains NaN or Inf values.")
            return True
        image_filename = vae_cache._image_filename_from_vaecache_filename(cache_file)
        logger.debug(f"Checking cache file {cache_file} for inconsistencies. Image filename: {image_filename}")
        actual_resolution = self._get_image_size_from_tensor(cache_content)
        original_resolution = self.get_metadata_attribute_by_filepath(image_filename, "original_size")
        metadata_target_size = self.get_metadata_attribute_by_filepath(image_filename, "target_size")
        if metadata_target_size is None:
            logger.error(f"Received sample with no metadata: {self.get_metadata_by_filepath(image_filename)}")
            return True
        target_resolution = tuple(metadata_target_size)
        recalculated_target_resolution, intermediary_size, recalculated_aspect_ratio = self._recalculate_target_resolution(
            original_aspect_ratio=MultiaspectImage.calculate_image_aspect_ratio(original_resolution)
        )
        logger.debug(
            f"Original resolution: {original_resolution}, Target resolution: {target_resolution}, Recalculated target resolution: {recalculated_target_resolution}"
        )
        if (
            original_resolution is not None
            and target_resolution is not None
            and (actual_resolution != target_resolution or actual_resolution != recalculated_target_resolution)
        ):
            logger.debug(
                f"Actual resolution {actual_resolution} does not match target resolution {target_resolution}, recalculated as {recalculated_target_resolution}."
            )
            return True
        else:
            logger.debug(f"Actual resolution {actual_resolution} matches target resolution {target_resolution}.")

        actual_aspect_ratio = self._get_aspect_ratio_from_tensor(cache_content)
        expected_bucket = str(recalculated_aspect_ratio)
        logger.debug(f"Expected bucket for {cache_file}: {expected_bucket} vs actual {actual_aspect_ratio}")

        base_filename = os.path.splitext(os.path.basename(cache_file))[0]
        base_filename_png = os.path.join(self.instance_data_dir, f"{base_filename}.png")
        base_filename_jpg = os.path.join(self.instance_data_dir, f"{base_filename}.jpg")
        if any(
            base_filename_png in self.aspect_ratio_bucket_indices.get(bucket, set())
            for bucket in [expected_bucket, str(expected_bucket)]
        ):
            logger.debug(f"File {base_filename} is in the correct bucket.")
            return False
        if any(
            base_filename_jpg in self.aspect_ratio_bucket_indices.get(bucket, set())
            for bucket in [expected_bucket, str(expected_bucket)]
        ):
            logger.debug(f"File {base_filename} is in the correct bucket.")
            return False
        logger.debug(f"File {base_filename} was not found in the correct place.")
        return True

    def _get_aspect_ratio_from_tensor(self, tensor):
        """calculate aspect ratio from tensor dimensions"""
        if tensor.dim() < 3:
            raise ValueError("Tensor does not have enough dimensions to determine aspect ratio.")
        _, height, width = tensor.size()
        return width / height

    def _get_image_size_from_tensor(self, tensor):
        """calculate image size from tensor, accounting for vae scaling"""
        if tensor.dim() < 3:
            raise ValueError(
                f"Tensor does not have enough dimensions to determine an image resolution. Its shape is: {tensor.size}"
            )
        _, height, width = tensor.size()
        return (
            width * self.vae_output_scaling_factor,
            height * self.vae_output_scaling_factor,
        )

    def _modify_cache_entry_bucket(self, cache_file, expected_bucket):
        """move cache file to correct bucket based on actual aspect ratio"""
        for bucket, files in self.aspect_ratio_bucket_indices.items():
            if cache_file in files and str(bucket) != str(expected_bucket):
                files.remove(cache_file)
                self.aspect_ratio_bucket_indices[expected_bucket].append(cache_file)
                break

    def _resolve_audio_truncation_mode(self) -> str:
        truncation_mode = (self.audio_config or {}).get("truncation_mode", "beginning")
        if isinstance(truncation_mode, str):
            mode = truncation_mode.lower()
        else:
            mode = "beginning"
        if mode not in {"beginning", "end", "random"}:
            logger.warning(
                f"(id={self.id}) Unsupported audio truncation_mode '{truncation_mode}'. " f"Falling back to 'beginning'."
            )
            mode = "beginning"
        return mode

    def _resolve_audio_lyrics_format(self) -> Optional[str]:
        if self.dataset_type is not DatasetType.AUDIO:
            return None
        lyrics_format = None
        if isinstance(self.audio_config, dict):
            lyrics_format = self.audio_config.get("lyrics_filename_format")
        if lyrics_format is None and isinstance(self.dataset_config, dict):
            lyrics_format = self.dataset_config.get("lyrics_filename_format")
        if lyrics_format is None:
            return None
        try:
            lyrics_format = str(lyrics_format).strip()
        except Exception:
            return None
        return lyrics_format or None

    def _resolve_audio_lyrics_extension(self) -> Optional[str]:
        if self.dataset_type is not DatasetType.AUDIO:
            return None
        extension = (self.audio_config or {}).get("lyrics_extension")
        if extension is None:
            return None
        extension = str(extension).strip()
        if not extension:
            return None
        return extension.lstrip(".")

    def _resolve_audio_lyrics_suffix(self) -> Optional[str]:
        if self.dataset_type is not DatasetType.AUDIO:
            return None
        suffix = (self.audio_config or {}).get("lyrics_suffix")
        if suffix is None:
            return None
        suffix = str(suffix)
        return suffix if suffix else None

    def _discover_audio_lyrics(self, sample_path: str) -> Optional[str]:
        if not (self.audio_lyrics_extension or self.audio_lyrics_suffix or self.audio_lyrics_format):
            return None
        try:
            path = Path(sample_path)
        except Exception:
            return None
        candidates = []
        if self.audio_lyrics_format:
            replacements = {
                "{filename}": path.stem,
                "{stem}": path.stem,
                "{name}": path.name,
                "{suffix}": "".join(path.suffixes) or path.suffix,
                "{extension}": path.suffix.lstrip("."),
                "{ext}": path.suffix.lstrip("."),
                "{parent}": str(path.parent),
                "{dirname}": path.parent.name,
            }
            rendered = self.audio_lyrics_format
            for token, value in replacements.items():
                rendered = rendered.replace(token, value)
            try:
                format_path = Path(rendered)
            except Exception:
                format_path = None
            if format_path is not None:
                if not format_path.is_absolute():
                    format_path = path.parent / format_path
                candidates.append(format_path)

        if self.audio_lyrics_extension or self.audio_lyrics_suffix:
            suffix_candidate = path
            if self.audio_lyrics_suffix:
                suffix_candidate = suffix_candidate.with_name(suffix_candidate.stem + self.audio_lyrics_suffix)
            if self.audio_lyrics_extension:
                extension = self.audio_lyrics_extension
                if not extension.startswith("."):
                    extension = f".{extension}"
                suffix_candidate = suffix_candidate.with_suffix(extension)
            candidates.append(suffix_candidate)

        seen_paths = set()
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                candidate_key = candidate.resolve()
            except Exception:
                candidate_key = candidate
            if candidate_key in seen_paths:
                continue
            seen_paths.add(candidate_key)
            if not candidate.exists():
                continue
            try:
                content = candidate.read_text(encoding="utf-8").strip()
            except Exception as exc:
                logger.debug(f"(id={self.id}) Failed reading lyrics for {sample_path}: {exc}")
                continue
            if content:
                return content
        return None

    def _build_audio_metadata_entry(
        self,
        sample_path: str,
        sample_rate: Optional[float],
        num_channels: Optional[int],
        num_samples: Optional[int],
        duration_seconds: Optional[float],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a canonical audio metadata payload so downstream components can reuse the same schema
        regardless of the discovery backend.
        """
        safe_samples = int(num_samples) if num_samples is not None else 0
        raw_channels = int(num_channels) if num_channels is not None else 0
        safe_channels = max(raw_channels, 1)
        placeholder_size = (safe_samples, safe_channels)
        payload = {
            "audio_path": sample_path,
            "image_path": sample_path,
            "sample_rate": sample_rate,
            "num_channels": num_channels,
            "num_samples": num_samples,
            "duration_seconds": duration_seconds,
            "truncation_mode": self.audio_truncation_mode,
            "original_size": placeholder_size,
            "intermediary_size": placeholder_size,
            "target_size": placeholder_size,
            "crop_coordinates": (0, 0),
            "aspect_ratio": 1.0,
        }
        lyrics_text = self._discover_audio_lyrics(sample_path)
        if lyrics_text:
            payload["lyrics"] = lyrics_text
        if overrides:
            payload.update(overrides)
        return payload
