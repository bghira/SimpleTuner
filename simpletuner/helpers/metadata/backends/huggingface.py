import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.multiaspect.image import MultiaspectImage
from simpletuner.helpers.training import image_file_extensions
from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.state_tracker import StateTracker


def _coerce_bucket_keys_to_float(indices: dict) -> dict:
    """Coerce bucket keys from strings to floats (fixes JSON serialization issue)."""
    coerced = {}
    for key, values in (indices or {}).items():
        try:
            coerced_key = float(key)
        except (TypeError, ValueError):
            coerced_key = key
        coerced[coerced_key] = list(values) if not isinstance(values, list) else values
    return coerced


logger = logging.getLogger("HuggingfaceMetadataBackend")
import trainingsample as tsr

from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class HuggingfaceMetadataBackend(MetadataBackend):
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
        hf_config: dict,
        delete_problematic_images: bool = False,
        delete_unwanted_images: bool = False,
        metadata_update_interval: int = 3600,
        minimum_image_size: int = None,
        minimum_aspect_ratio: float = None,
        maximum_aspect_ratio: float = None,
        # Video-related parameters
        minimum_num_frames: int = None,
        maximum_num_frames: int = None,
        num_frames: int = None,
        cache_file_suffix: str = None,
        repeats: int = 0,
        # HF-specific parameters
        split_composite_images: bool = False,
        composite_image_column: str = "image",
        quality_filter: Optional[Dict[str, Any]] = None,
        dataset_type: str = "image",
    ):
        self.hf_config = hf_config
        self.dataset_type = dataset_type
        self.split_composite_images = split_composite_images
        self.composite_image_column = composite_image_column
        self.quality_filter = quality_filter

        self.video_column = hf_config.get("video_column", "video")
        self.caption_column = hf_config.get("caption_column", "caption")
        self.fallback_caption_column = hf_config.get("fallback_caption_column", None)
        self.width_column = hf_config.get("width_column", None)
        self.height_column = hf_config.get("height_column", None)
        self.quality_column = hf_config.get("quality_column", "quality_assessment")
        self.description_column = hf_config.get("description_column", "description")
        self.lyrics_column = hf_config.get("lyrics_column", "lyrics")
        self.audio_caption_fields = hf_config.get("audio_caption_fields", ["prompt", "tags"])
        try:
            from simpletuner.helpers.training.state_tracker import StateTracker

            model = StateTracker.get_model()
            if model and hasattr(model, "caption_field_preferences"):
                preferred = model.caption_field_preferences(dataset_type=self.dataset_type)
                if preferred:
                    merged = list(dict.fromkeys(list(preferred) + list(self.audio_caption_fields or [])))
                    self.audio_caption_fields = merged
        except Exception:
            pass
        dataset_type_normalized = str(dataset_type).lower() if dataset_type is not None else "image"
        if dataset_type_normalized == "video":
            default_extension = "mp4"
        elif dataset_type_normalized == "audio":
            default_extension = "wav"
        else:
            default_extension = "jpg"
        self.file_extension = hf_config.get("file_extension", default_extension)
        self.num_frames_column = hf_config.get("num_frames_column", None)
        self.fps_column = hf_config.get("fps_column", None)
        self.composite_config = hf_config.get("composite_image_config", {})
        if should_log():
            logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
        else:
            logger.setLevel("ERROR")

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
            minimum_num_frames=minimum_num_frames,
            maximum_num_frames=maximum_num_frames,
            num_frames=num_frames,
            cache_file_suffix=cache_file_suffix,
            repeats=repeats,
        )

        if not hasattr(data_backend, "dataset"):
            raise ValueError("HuggingfaceMetadataBackend requires HuggingfaceDatasetsBackend")
        with accelerator.main_process_first():
            self.caption_cache = self._extract_captions_to_dict()
        accelerator.wait_for_everyone()
        with accelerator.main_process_first():
            self.reload_cache()
            self.load_image_metadata()
        accelerator.wait_for_everyone()

    def _extract_captions_to_dict(self) -> Dict[str, Union[str, List[str]]]:
        import concurrent.futures

        captions_cache_file = f"{self.cache_file}_captions.json"
        if self.data_backend.exists(captions_cache_file):
            try:
                raw = self.data_backend.read(captions_cache_file)
                if raw:
                    captions = json.loads(raw)
                    logger.info(f"Loaded {len(captions)} captions from cache file {captions_cache_file}")
                    return captions
            except Exception as e:
                logger.warning(f"Error loading caption cache, will regenerate: {e}")
        logger.info("Extracting captions from Hugging Face dataset...")

        def process_item(idx):
            item = self.data_backend.dataset[idx]
            virtual_path = f"{idx}.{self.file_extension}"
            if self.quality_filter and self.quality_column in item:
                quality = item[self.quality_column]
                if not self._passes_quality_filter(quality):
                    return None
            caption = self._extract_caption_from_item(item)
            if caption:
                return (virtual_path, caption)
            return None

        captions = {}
        total_items = len(self.data_backend.dataset)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(process_item, range(total_items)),
                    desc="Extracting captions",
                    total=total_items,
                    ncols=100,
                    mininterval=0.5,
                    ascii=True,
                    position=int(os.environ.get("RANK", 0)),
                )
            )
        for result in results:
            if result:
                virtual_path, caption = result
                captions[virtual_path] = caption

        try:
            self.data_backend.write(captions_cache_file, json.dumps(captions))
            logger.info(f"Saved {len(captions)} captions to cache file {captions_cache_file}")
        except Exception as e:
            logger.warning(f"Error saving caption cache: {e}")

        logger.info(f"Extracted {len(captions)} captions from dataset")
        return captions

    def _extract_caption_from_item(self, item: Dict) -> Optional[Union[str, List[str]]]:
        caption = None
        # handle list of caption columns
        if isinstance(self.caption_column, list):
            caption_values = []
            for col in self.caption_column:
                value = self._get_nested_value(item, col)
                if value:
                    caption_values.append(str(value))
            if caption_values:
                caption = caption_values
        else:
            # single caption column (might be nested)
            caption = self._get_nested_value(item, self.caption_column)
            if caption:
                caption = str(caption)

        # try fallback if no caption found
        if not caption and self.fallback_caption_column:
            fallback = self._get_nested_value(item, self.fallback_caption_column)
            if fallback:
                caption = str(fallback)

        # Audio models often store prompts/lyrics in multiple fields; combine them if no caption yet.
        if not caption and self.dataset_type == "audio":
            parts = []
            for field in self.audio_caption_fields or []:
                value = self._get_nested_value(item, field)
                if value:
                    parts.append(str(value).strip())
            if parts:
                caption = " ".join([p for p in parts if p])

        # handle description column if specified
        if not caption and self.description_column in item:
            desc = item.get(self.description_column)
            if isinstance(desc, dict):
                # extract relevant parts from description dict
                caption_parts = []
                for key in ["main_subject", "description", "text"]:
                    if key in desc and desc[key]:
                        caption_parts.append(str(desc[key]))
                if caption_parts:
                    caption = " ".join(caption_parts)
            else:
                caption = str(desc)

        # clean up caption
        if caption:
            if isinstance(caption, (list, tuple)):
                caption = [str(c).strip() for c in caption if c]
            elif isinstance(caption, str):
                caption = caption.strip()

        return caption if caption else None

    def _get_nested_value(self, item: Dict, key_path: str) -> Any:
        if "." in key_path:
            keys = key_path.split(".")
            current_value = item
            for key in keys:
                if isinstance(current_value, dict) and key in current_value:
                    current_value = current_value[key]
                else:
                    return None
            return current_value
        else:
            return item.get(key_path)

    def reload_cache(self, set_config: bool = True):
        if self.data_backend.exists(self.cache_file):
            try:
                cache_data_raw = self.data_backend.read(self.cache_file)
                cache_data = json.loads(cache_data_raw)
                logger.debug("Loaded existing aspect ratio cache.")
            except Exception as e:
                logger.warning(f"Error loading aspect ratio bucket cache, creating new one: {e}")
                cache_data = {}
            # Coerce bucket keys from strings to floats (JSON serialization converts float keys to strings)
            loaded_indices = cache_data.get("aspect_ratio_bucket_indices", {})
            self.aspect_ratio_bucket_indices = _coerce_bucket_keys_to_float(loaded_indices)
            if set_config:
                self.config = cache_data.get("config", {})
                if self.config != {}:
                    logger.debug(f"Setting config to {self.config}")
                    StateTracker.set_data_backend_config(
                        data_backend_id=self.id,
                        config=self.config,
                    )
        else:
            logger.debug("No cache file found, starting fresh.")

    def save_cache(self, enforce_constraints: bool = False):
        if enforce_constraints:
            self._enforce_min_bucket_size()
        self._enforce_min_aspect_ratio()
        self._enforce_max_aspect_ratio()

        if self.read_only:
            logger.debug("Metadata backend is read-only. Skipping save.")
            return

        aspect_ratio_bucket_indices_str = {
            key: [str(path) for path in value] for key, value in self.aspect_ratio_bucket_indices.items()
        }
        cache_data = {
            "config": StateTracker.get_data_backend_config(data_backend_id=self.data_backend.id),
            "aspect_ratio_bucket_indices": aspect_ratio_bucket_indices_str,
        }
        cache_data_str = json.dumps(cache_data)
        self.data_backend.write(self.cache_file, cache_data_str)
        logger.debug("Aspect ratio cache saved.")

    def save_image_metadata(self):
        full_metadata_path = self.metadata_file
        if not str(self.metadata_file).endswith(".json"):
            full_metadata_path = f"{self.metadata_file}_{self.id}.json"

        self.data_backend.write(full_metadata_path, json.dumps(self.image_metadata))
        logger.debug(f"Metadata file saved to {full_metadata_path}")

    def load_image_metadata(self):
        full_metadata_path = self.metadata_file
        if not str(self.metadata_file).endswith(".json"):
            full_metadata_path = f"{self.metadata_file}_{self.id}.json"
        logger.debug(f"Loading metadata from {full_metadata_path}")
        self.image_metadata = {}
        self.image_metadata_loaded = False

        if self.data_backend.exists(full_metadata_path):
            try:
                raw = self.data_backend.read(full_metadata_path)
                if raw:
                    self.image_metadata = json.loads(raw)
                    self.image_metadata_loaded = True
                    logger.info(f"Loaded {len(self.image_metadata)} metadata entries from {full_metadata_path}")
                else:
                    logger.warning(f"Metadata file exists but is empty: {full_metadata_path}")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        else:
            logger.debug(f"Metadata file does not exist: {full_metadata_path}")

    def _passes_quality_filter(self, quality_assessment: Dict) -> bool:
        if not self.quality_filter or not quality_assessment:
            return True

        for key, min_value in self.quality_filter.items():
            if quality_assessment.get(key, 0) < min_value:
                return False
        return True

    def caption_cache_entry(self, index: str) -> Optional[Union[str, List[str]]]:
        return self.caption_cache.get(str(index), None)

    def _discover_new_files(self, for_metadata: bool = False, ignore_existing_cache: bool = False):
        # for HF datasets, we process all items
        if hasattr(self.data_backend, "streaming") and self.data_backend.streaming:
            logger.warning("Cannot discover files in streaming mode")
            return []

        total_items = len(self.data_backend.dataset)
        is_composite = self.composite_config.get("enabled", False)
        select_index = self.composite_config.get("select_index", None)
        all_files = []
        for idx in range(total_items):
            virtual_path = f"{idx}.{self.file_extension}"
            all_files.append(virtual_path)

        if ignore_existing_cache:
            logger.debug("Ignoring existing cache, returning all files")
            return all_files

        processed_files = set()
        for paths in self.aspect_ratio_bucket_indices.values():
            processed_files.update(paths)
        new_files = [f for f in all_files if f not in processed_files]
        logger.debug(f"Found {len(new_files)} new files out of {len(all_files)} total")

        return new_files

    def _get_video_metadata_from_item(self, item: Dict) -> Dict[str, Any]:
        metadata = {}

        # prefer explicit aspect_ratio column if present
        aspect_ratio = None
        if "aspect_ratio" in item:
            try:
                aspect_ratio = float(item["aspect_ratio"])
            except Exception:
                aspect_ratio = None

        # try width/height columns
        width, height = None, None
        if self.width_column and self.height_column:
            if self.width_column in item and self.height_column in item:
                try:
                    width = int(item[self.width_column])
                    height = int(item[self.height_column])
                except Exception:
                    width, height = None, None

        # if we have width and height, use them
        if width and height:
            metadata["original_size"] = (width, height)
            if aspect_ratio is None and height != 0:
                aspect_ratio = width / height
        # if we have aspect_ratio but not width/height, set dummy size
        elif aspect_ratio is not None:
            # use a default height (e.g., 256) to compute width
            height = 256
            width = int(round(aspect_ratio * height))
            metadata["original_size"] = (width, height)

        if self.num_frames_column and self.num_frames_column in item:
            try:
                metadata["num_frames"] = int(item[self.num_frames_column])
            except Exception:
                pass

        if self.fps_column and self.fps_column in item:
            try:
                metadata["fps"] = float(item[self.fps_column])
            except Exception:
                pass

        if self.video_column in item and hasattr(self.data_backend, "_extract_video_sample_metadata"):
            sample = item[self.video_column]
            video_metadata = self.data_backend._extract_video_sample_metadata(sample)  # type: ignore[attr-defined]

            if "original_size" not in metadata and "original_size" in video_metadata:
                metadata["original_size"] = video_metadata["original_size"]

            if "fps" not in metadata and "fps" in video_metadata:
                metadata["fps"] = video_metadata["fps"]

            if "num_frames" not in metadata and "num_frames" in video_metadata:
                metadata["num_frames"] = video_metadata["num_frames"]

        return metadata

    def _get_image_metadata_from_item(self, item: Dict) -> Dict[str, Any]:
        metadata = {}
        image = item.get(self.data_backend.image_column)
        if image is not None:
            is_composite = self.composite_config.get("enabled", False)
            image_count = self.composite_config.get("image_count", 1)
            select_index = self.composite_config.get("select_index", None)

            if isinstance(image, Image.Image):
                width, height = image.size

                # handle composite image dimensions
                if is_composite and image_count > 1:
                    # calculate dimensions for the selected segment
                    segment_width = width // image_count
                    metadata["original_size"] = (segment_width, height)
                else:
                    metadata["original_size"] = (width, height)

            elif isinstance(image, np.ndarray):
                h, w = image.shape[:2]

                # handle composite image dimensions
                if is_composite and image_count > 1:
                    segment_width = w // image_count
                    metadata["original_size"] = (segment_width, h)
                else:
                    metadata["original_size"] = (w, h)

        if self.width_column and self.height_column:
            if self.width_column in item and self.height_column in item:
                w = item[self.width_column]
                h = item[self.height_column]

                is_composite = self.composite_config.get("enabled", False)
                image_count = self.composite_config.get("image_count", 1)
                if is_composite and image_count > 1:
                    w = int(w) // image_count

                metadata["original_size"] = (int(w), int(h))

        if self.num_frames_column and self.num_frames_column in item:
            metadata["num_frames"] = int(item[self.num_frames_column])

        if self.fps_column and self.fps_column in item:
            metadata["fps"] = float(item[self.fps_column])

        return metadata

    def _process_for_bucket(
        self,
        image_path_str: str,
        aspect_ratio_bucket_indices: Dict,
        metadata_updates: Optional[Dict] = None,
        delete_problematic_images: bool = False,
        statistics: dict = {},
        aspect_ratio_rounding: int = 2,
    ) -> Dict:
        try:
            index = self.data_backend._get_index_from_path(image_path_str)
            if index is None:
                logger.warning(f"Could not get index for path: {image_path_str}")
                statistics.setdefault("skipped", {}).setdefault("not_found", 0)
                statistics["skipped"]["not_found"] += 1
                return aspect_ratio_bucket_indices

            item = self.data_backend.get_dataset_item(index)
            if item is None:
                logger.warning(f"Could not get dataset item for index: {index}")
                statistics.setdefault("skipped", {}).setdefault("not_found", 0)
                statistics["skipped"]["not_found"] += 1
                return aspect_ratio_bucket_indices

            if self.quality_filter and self.quality_column in item:
                if not self._passes_quality_filter(item[self.quality_column]):
                    logger.debug(f"Item {image_path_str} failed quality filter")
                    statistics.setdefault("skipped", {}).setdefault("quality", 0)
                    statistics["skipped"]["quality"] += 1
                    return aspect_ratio_bucket_indices
            if self.dataset_type == "audio":
                sample_metadata = dict(self.image_metadata.get(image_path_str, {}))
                # Ensure audio fields are present in metadata
                if self.audio_caption_fields:
                    for field in self.audio_caption_fields:
                        val = self._get_nested_value(item, field)
                        if val:
                            sample_metadata[field] = str(val)

                # Specific handling for lyrics column
                if "lyrics" not in sample_metadata:
                    lyrics_val = self._get_nested_value(item, self.lyrics_column)
                    if lyrics_val:
                        sample_metadata["lyrics"] = str(lyrics_val)
                    # Fallback for norm_lyrics if not found via lyrics_column
                    elif self.lyrics_column != "norm_lyrics":
                        norm_lyrics = self._get_nested_value(item, "norm_lyrics")
                        if norm_lyrics:
                            sample_metadata["lyrics"] = str(norm_lyrics)

                duration_seconds = sample_metadata.get("duration_seconds") or sample_metadata.get("bucket_duration_seconds")
                if duration_seconds is None:
                    # Prefer explicit duration columns if provided by the dataset.
                    duration_seconds = item.get("audio_duration") or item.get("duration") or item.get("duration_seconds")
                if duration_seconds is None:
                    audio_value = item.get("audio")
                    try:
                        if isinstance(audio_value, dict):
                            array = audio_value.get("array")
                            sample_rate = audio_value.get("sampling_rate")
                        else:
                            array = None
                            sample_rate = None
                        if array is not None and sample_rate:
                            duration_seconds = float(len(array)) / float(sample_rate)
                            sample_metadata["sample_rate"] = sample_rate
                            sample_metadata["num_samples"] = len(array)
                    except Exception:
                        duration_seconds = duration_seconds
                bucket_key, truncated_duration = self._compute_audio_bucket(duration_seconds)
                if truncated_duration is not None:
                    sample_metadata["bucket_duration_seconds"] = truncated_duration
                    sample_metadata["duration_seconds"] = truncated_duration
                elif duration_seconds is not None:
                    sample_metadata["duration_seconds"] = float(duration_seconds)
                sample_metadata.setdefault("dataset_type", "audio")
                aspect_ratio_bucket_indices.setdefault(bucket_key, []).append(image_path_str)
                if metadata_updates is not None:
                    metadata_updates[image_path_str] = sample_metadata
                statistics["total_processed"] += 1
                return aspect_ratio_bucket_indices
            if self.dataset_type == "image":
                sample_metadata = self._get_image_metadata_from_item(item)

                if "original_size" not in sample_metadata:
                    logger.warning(f"Could not determine image size for {image_path_str}")
                    statistics.setdefault("skipped", {}).setdefault("metadata_missing", 0)
                    statistics["skipped"]["metadata_missing"] += 1
                    return aspect_ratio_bucket_indices
                if not self.meets_resolution_requirements(image_metadata=sample_metadata):
                    if self.delete_unwanted_images:
                        logger.debug(f"{image_path_str} does not meet resolution requirements.")
                    statistics.setdefault("skipped", {}).setdefault("too_small", 0)
                    statistics["skipped"]["too_small"] += 1
                    return aspect_ratio_bucket_indices
                training_sample = TrainingSample(
                    image=None,  # We don't load actual image data here
                    data_backend_id=self.id,
                    image_metadata=sample_metadata,
                    image_path=image_path_str,
                )
            elif self.dataset_type == "video":
                if self.video_column not in item:
                    logger.warning(f"Video column '{self.video_column}' not found in item {image_path_str}")
                    statistics.setdefault("skipped", {}).setdefault("metadata_missing", 0)
                    statistics["skipped"]["metadata_missing"] += 1
                    return aspect_ratio_bucket_indices
                sample_metadata = self._get_video_metadata_from_item(item)
                if "num_frames" in sample_metadata:
                    num_frames = sample_metadata["num_frames"]

                    if self.minimum_num_frames and num_frames < self.minimum_num_frames:
                        logger.debug(
                            f"Video {image_path_str} has {num_frames} frames, below minimum {self.minimum_num_frames}"
                        )
                        statistics.setdefault("skipped", {}).setdefault("too_few_frames", 0)
                        statistics["skipped"]["too_few_frames"] += 1
                        return aspect_ratio_bucket_indices

                    if self.maximum_num_frames and num_frames > self.maximum_num_frames:
                        logger.debug(
                            f"Video {image_path_str} has {num_frames} frames, above maximum {self.maximum_num_frames}"
                        )
                    statistics.setdefault("skipped", {}).setdefault("too_many_frames", 0)
                    statistics["skipped"]["too_many_frames"] += 1
                    return aspect_ratio_bucket_indices
                training_sample = TrainingSample(
                    image=None,  # We don't load actual data here
                    data_backend_id=self.id,
                    image_metadata=sample_metadata,
                    image_path=image_path_str,
                )
            else:
                training_sample = None

            if training_sample is None:
                statistics.setdefault("skipped", {}).setdefault("metadata_missing", 0)
                statistics["skipped"]["metadata_missing"] += 1
                return aspect_ratio_bucket_indices

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

            aspect_ratio_key = str(round(aspect_ratio, aspect_ratio_rounding))
            if aspect_ratio_key not in aspect_ratio_bucket_indices:
                aspect_ratio_bucket_indices[aspect_ratio_key] = []
            aspect_ratio_bucket_indices[aspect_ratio_key].append(image_path_str)
            if metadata_updates is not None:
                metadata_updates[image_path_str] = sample_metadata

        except Exception as e:
            logger.error(f"Error processing file {image_path_str}: {e}")
            logger.error(traceback.format_exc())
            statistics.setdefault("skipped", {}).setdefault("error", 0)
            statistics["skipped"]["error"] += 1

        return aspect_ratio_bucket_indices

    def compute_aspect_ratio_bucket_indices(self, ignore_existing_cache: bool = False):
        logger.info("Building aspect ratio buckets from Hugging Face dataset...")

        # for HF datasets, we process all items
        if hasattr(self.data_backend, "streaming") and self.data_backend.streaming:
            logger.warning("Cannot build aspect ratio buckets in streaming mode")
            return

        is_composite = self.composite_config.get("enabled", False)
        select_index = self.composite_config.get("select_index", None)
        statistics = {
            "total_processed": 0,
            "skipped": {
                "already_exists": 0,
                "quality": 0,
                "metadata_missing": 0,
                "not_found": 0,
                "too_small": 0,
                "too_few_frames": 0,
                "too_many_frames": 0,
                "error": 0,
            },
        }

        if not ignore_existing_cache:
            self.reload_cache()
            self.load_image_metadata()
            existing_files = set().union(*self.aspect_ratio_bucket_indices.values())
            statistics["skipped"]["already_exists"] = len(existing_files)
        else:
            self.aspect_ratio_bucket_indices = {}
            existing_files = set()
        if self.bucket_report:
            self.bucket_report.record_stage(
                "existing_cache",
                sample_count=len(existing_files),
                bucket_count=len(self.aspect_ratio_bucket_indices),
            )
        last_save_time = time.time()
        aspect_ratio_bucket_updates = {}
        metadata_updates = {}

        total_items = len(self.data_backend.dataset)
        if self.bucket_report:
            pending_items = max(total_items - len(existing_files), 0)
            self.bucket_report.record_stage(
                "new_files_to_process",
                sample_count=pending_items,
                ignore_existing_cache=ignore_existing_cache,
            )
        for idx in tqdm(
            range(total_items),
            desc="Processing HF dataset items",
            total=total_items,
            leave=False,
            ncols=100,
        ):
            virtual_path = f"{idx}.{self.file_extension}"

            if virtual_path in existing_files:
                continue

            aspect_ratio_bucket_updates = self._process_for_bucket(
                virtual_path,
                aspect_ratio_bucket_updates,
                metadata_updates=metadata_updates,
                delete_problematic_images=self.delete_problematic_images,
                statistics=statistics,
            )
            statistics["total_processed"] += 1

            current_time = time.time()
            if (current_time - last_save_time) >= self.metadata_update_interval:
                logger.debug(f"Saving intermediate results...")
                for key, value in aspect_ratio_bucket_updates.items():
                    self.aspect_ratio_bucket_indices.setdefault(key, []).extend(value)
                aspect_ratio_bucket_updates = {}

                for path, metadata in metadata_updates.items():
                    self.set_metadata_by_filepath(path, metadata, update_json=False)
                metadata_updates = {}

                self.save_cache(enforce_constraints=False)
                self.save_image_metadata()
                last_save_time = current_time
        for key, value in aspect_ratio_bucket_updates.items():
            self.aspect_ratio_bucket_indices.setdefault(key, []).extend(value)

        for path, metadata in metadata_updates.items():
            self.set_metadata_by_filepath(path, metadata, update_json=False)

        logger.info(f"Processing complete. Statistics: {statistics}")
        self.save_image_metadata()
        self.save_cache(enforce_constraints=True)
        if self.bucket_report:
            self.bucket_report.update_statistics(statistics)
            self.bucket_report.record_bucket_snapshot("post_refresh", self.aspect_ratio_bucket_indices)

    def __len__(self):

        def repeat_len(bucket):
            return len(bucket) * (self.repeats + 1)

        return sum(
            (repeat_len(bucket) + (self.batch_size - 1)) // self.batch_size
            for bucket in self.aspect_ratio_bucket_indices.values()
            if repeat_len(bucket) >= self.batch_size
        )

    def remove_images(self, image_paths: List[str]):
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        for image_path in image_paths:
            if image_path in self.caption_cache:
                del self.caption_cache[image_path]
            for bucket, images in self.aspect_ratio_bucket_indices.items():
                if image_path in images:
                    self.aspect_ratio_bucket_indices[bucket].remove(image_path)
                    logger.debug(f"Removed {image_path} from bucket {bucket}")

    def refresh_buckets(self, rank: int = None):
        # override refresh_buckets for HuggingFace datasets
        # since HF datasets use virtual paths, we don't need to check for missing files
        logger.debug(f"Refreshing buckets for HuggingFace backend {self.id}")
        self.compute_aspect_ratio_bucket_indices()
        return

    def update_buckets_with_existing_files(self, existing_files: set):
        # override for HuggingFace - all virtual files always "exist"
        # for HF datasets, we don't need to remove non-existing files
        # since all files are virtual references to dataset indices
        logger.debug(f"Skipping file existence check for HuggingFace backend {self.id}")
        return
