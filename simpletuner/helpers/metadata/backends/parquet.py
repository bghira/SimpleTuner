import json
import logging
import os
import time
import traceback
from io import BytesIO
from typing import Optional

import numpy
from tqdm import tqdm

from simpletuner.helpers.audio import load_audio
from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.multiaspect.image import MultiaspectImage
from simpletuner.helpers.training import audio_file_extensions, image_file_extensions, video_file_extensions
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


logger = logging.getLogger("ParquetMetadataBackend")
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")

try:
    import pandas as pd
except ImportError:
    raise ImportError("Pandas is required for the ParquetMetadataBackend.")


class ParquetMetadataBackend(MetadataBackend):
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
        parquet_config: dict,
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
        self.parquet_config = parquet_config
        self.parquet_path = parquet_config.get("path", None)
        self.is_json_lines = self.parquet_path.endswith(".jsonl")
        self.is_json_file = self.parquet_path.endswith(".json")

        self.num_frames_column = parquet_config.get("num_frames_column", None)
        self.fps_column = parquet_config.get("fps_column", None)

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
        )
        self.load_parquet_database()
        self.caption_cache = self._extract_captions_to_fast_list()
        self.missing_captions = self._locate_missing_caption_from_fast_list()
        if self.missing_captions:
            logger.warning(f"Missing captions for {len(self.missing_captions)} images: {self.missing_captions}")
            self._remove_images_with_missing_captions()

    def _remove_images_with_missing_captions(self):
        # remove items with missing captions from buckets
        for key in self.aspect_ratio_bucket_indices.keys():
            len_before = len(self.aspect_ratio_bucket_indices[key])
            self.aspect_ratio_bucket_indices[key] = [
                path for path in self.aspect_ratio_bucket_indices[key] if path not in self.missing_captions
            ]
            len_after = len(self.aspect_ratio_bucket_indices[key])
            if len_before != len_after:
                logger.warning(
                    f"Removed {len_before - len_after} items from aspect ratio bucket {key} due to missing captions."
                )
        self.save_cache(enforce_constraints=True)
        self.missing_captions = []

    def load_parquet_database(self):
        if self.data_backend.exists(self.parquet_path):
            try:
                bytes_string = self.data_backend.read(self.parquet_path)
                import io

                pq = io.BytesIO(bytes_string)
            except Exception as e:
                raise e

            if self.is_json_lines or self.is_json_file:
                self.parquet_database = pd.read_json(pq, lines=self.is_json_lines)
            else:
                self.parquet_database = pd.read_parquet(pq, engine="pyarrow")

            self.parquet_database.set_index(self.parquet_config.get("filename_column"), inplace=True)
        else:
            raise FileNotFoundError(f"Parquet could not be loaded from {self.parquet_path}: file does not exist.")

    def _locate_missing_caption_from_fast_list(self):
        missing_captions = []
        identifier_includes_extension = self.parquet_config.get("identifier_includes_extension", False)
        # currently we do not do path-based identifiers
        identifier_includes_path = False

        for key in self.aspect_ratio_bucket_indices.keys():
            for filename in self.aspect_ratio_bucket_indices[key]:
                # Chop extension if parquet expects no extension
                if not identifier_includes_extension:
                    filename = os.path.splitext(filename)[0]
                if not identifier_includes_path:
                    filename = filename.replace(self.instance_data_dir, "")
                    if filename.startswith("/"):
                        filename = filename[1:]
                if filename not in self.caption_cache:
                    missing_captions.append(filename)
        return missing_captions

    def _extract_captions_to_fast_list(self):
        # parquet is columnar - dict lookups are faster than repeated column access
        if self.parquet_database is None:
            raise ValueError("Parquet database is not loaded.")

        filename_column = self.parquet_config.get("filename_column")
        caption_column = self.parquet_config.get("caption_column")
        fallback_caption_column = self.parquet_config.get("fallback_caption_column")
        identifier_includes_extension = self.parquet_config.get("identifier_includes_extension", False)
        captions = {}

        for index, row in self.parquet_database.iterrows():
            if filename_column in row:
                file_id = str(row[filename_column])
            else:
                file_id = str(index)

            if not identifier_includes_extension:
                file_id = os.path.splitext(file_id)[0]

            # Could be list or single str
            if isinstance(caption_column, list):
                if len(caption_column) > 0:
                    # Gather all columns in a list
                    cvals = [row.get(c, "") for c in caption_column]
                    cvals = [c for c in cvals if c]  # remove empties
                    caption = cvals if cvals else None
                else:
                    caption = None
            else:
                caption = row.get(caption_column, None)

            if caption is None and fallback_caption_column:
                caption = row.get(fallback_caption_column, None)

            if isinstance(caption, (numpy.ndarray, pd.Series)):
                caption = [str(item) for item in caption if item is not None]
            elif isinstance(caption, bytes):
                caption = caption.decode("utf-8")
            elif isinstance(caption, str):
                caption = caption.strip()

            if not caption:
                continue

            captions[file_id] = caption
        return captions

    def caption_cache_entry(self, index: str):
        return self.caption_cache.get(str(index), None)

    def _discover_new_files(self, for_metadata: bool = False, ignore_existing_cache: bool = False):
        all_image_files = StateTracker.get_image_files(data_backend_id=self.data_backend.id)
        if all_image_files is None:
            extension_pool = audio_file_extensions if self.dataset_type is DatasetType.AUDIO else image_file_extensions
            all_image_files = self.data_backend.list_files(
                instance_data_dir=self.instance_data_dir,
                file_extensions=extension_pool,
            )
            # Flatten nested lists
            if any(isinstance(i, list) for i in all_image_files):
                all_image_files = [item for sublist in all_image_files for item in sublist]
            all_image_files = StateTracker.set_image_files(all_image_files, data_backend_id=self.data_backend.id)
        else:
            # flatten if necessary
            if any(isinstance(i, list) for i in all_image_files):
                all_image_files = [item for sublist in all_image_files for item in sublist]

        if ignore_existing_cache:
            self.aspect_ratio_bucket_indices = {}
            return list(all_image_files)

        all_image_files_set = set(all_image_files)

        if for_metadata:
            result = [file for file in all_image_files if self.get_metadata_by_filepath(file) is None]
        else:
            processed_files = set(path for paths in self.aspect_ratio_bucket_indices.values() for path in paths)
            result = [file for file in all_image_files_set if file not in processed_files]

        return result

    def reload_cache(self, set_config: bool = True):
        if self.data_backend.exists(self.cache_file):
            try:
                cache_data_raw = self.data_backend.read(self.cache_file)
                cache_data = json.loads(cache_data_raw)
            except Exception as e:
                logger.warning(f"Error loading aspect ratio bucket cache, creating new one: {e}")
                cache_data = {}
            # Coerce bucket keys from strings to floats (JSON serialization converts float keys to strings)
            loaded_indices = cache_data.get("aspect_ratio_bucket_indices", {})
            self.aspect_ratio_bucket_indices = _coerce_bucket_keys_to_float(loaded_indices)
            if set_config:
                self.config = cache_data.get("config", {})
                if self.config != {}:
                    StateTracker.set_data_backend_config(
                        data_backend_id=self.id,
                        config=self.config,
                    )
        else:
            logger.warning("No cache file found, starting a fresh one.")

    def save_cache(self, enforce_constraints: bool = False):
        if enforce_constraints:
            self._enforce_min_bucket_size()
        self._enforce_min_aspect_ratio()
        self._enforce_max_aspect_ratio()

        if self.read_only:
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

    def load_image_metadata(self):
        self.image_metadata = {}
        self.image_metadata_loaded = False
        if self.data_backend.exists(self.metadata_file):
            raw = self.data_backend.read(self.metadata_file)
            self.image_metadata = json.loads(raw)
            self.image_metadata_loaded = True

    def save_image_metadata(self):
        self.data_backend.write(self.metadata_file, json.dumps(self.image_metadata))

    def compute_aspect_ratio_bucket_indices(self, ignore_existing_cache: bool = False):
        # build buckets from parquet metadata without loading actual files
        new_files = self._discover_new_files(ignore_existing_cache=ignore_existing_cache)
        existing_files_set = set().union(*self.aspect_ratio_bucket_indices.values())
        if self.bucket_report:
            self.bucket_report.record_stage(
                "existing_cache",
                sample_count=len(existing_files_set),
                bucket_count=len(self.aspect_ratio_bucket_indices),
            )
        statistics = {
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
            if self.bucket_report:
                self.bucket_report.update_statistics(statistics)
                self.bucket_report.record_bucket_snapshot("post_refresh", self.aspect_ratio_bucket_indices)
            return

        try:
            self.load_image_metadata()
        except Exception as e:
            if ignore_existing_cache:
                self.image_metadata = {}
            else:
                raise Exception(f"Error loading image metadata. Consider removing the metadata file manually: {e}")

        last_write_time = time.time()
        aspect_ratio_bucket_updates = {}

        for file in tqdm(
            new_files,
            desc="Generating aspect bucket cache",
            total=len(new_files),
            leave=False,
            ncols=100,
            miniters=max(1, len(new_files) // 100),
        ):
            current_time = time.time()
            if file not in existing_files_set:
                metadata_updates = {}
                if self.should_abort:
                    return

                aspect_ratio_bucket_updates = self._process_for_bucket(
                    file,
                    aspect_ratio_bucket_updates,
                    metadata_updates=metadata_updates,
                    delete_problematic_images=self.delete_problematic_images,
                    statistics=statistics,
                )
                statistics["total_processed"] += 1
            else:
                statistics["skipped"]["already_exists"] += 1
                continue

            # If we have metadata updates, store them
            if metadata_updates and file in metadata_updates:
                self.set_metadata_by_filepath(filepath=file, metadata=metadata_updates[file], update_json=False)

            # periodic save to avoid losing progress
            if (current_time - last_write_time) >= self.metadata_update_interval:
                self.save_cache(enforce_constraints=False)
                self.save_image_metadata()
                last_write_time = current_time

        # Extend final results
        for key, value in aspect_ratio_bucket_updates.items():
            self.aspect_ratio_bucket_indices.setdefault(key, []).extend(value)

        logger.info(f"Sample processing statistics: {statistics}")
        self.save_image_metadata()
        self.save_cache(enforce_constraints=True)
        if self.bucket_report:
            self.bucket_report.update_statistics(statistics)
            self.bucket_report.record_bucket_snapshot("post_refresh", self.aspect_ratio_bucket_indices)

    def _get_first_value(self, series_or_scalar):
        import numpy as np
        import pandas as pd

        if isinstance(series_or_scalar, pd.Series):
            series_or_scalar = series_or_scalar.iloc[0]  # Just unwrap the first value
        elif isinstance(series_or_scalar, str):
            series_or_scalar = float(series_or_scalar) if "." in series_or_scalar else int(series_or_scalar)

        # After unwrapping, if it's an np.int* or np.float*, cast to python int/float
        if isinstance(series_or_scalar, np.integer):
            return int(series_or_scalar)
        elif isinstance(series_or_scalar, np.floating):
            return float(series_or_scalar)
        elif isinstance(series_or_scalar, (int, float)):
            return series_or_scalar
        elif series_or_scalar is None:
            return None
        else:
            raise ValueError(f"Unsupported data type: {type(series_or_scalar)}")

    def _process_for_bucket(
        self,
        image_path_str,
        aspect_ratio_bucket_indices,
        aspect_ratio_rounding: int = 3,
        metadata_updates=None,
        delete_problematic_images: bool = False,
        statistics: dict = {},
    ):
        # process file using parquet metadata only - no actual file loading
        try:
            # 1. Identify the row in parquet
            image_path_filtered = image_path_str
            if not self.parquet_config.get("identifier_includes_extension", False):
                image_path_filtered = os.path.splitext(os.path.split(image_path_str)[-1])[0]
            if self.instance_data_dir in image_path_filtered:
                image_path_filtered = image_path_filtered.replace(self.instance_data_dir, "")
                if image_path_filtered.startswith("/"):
                    image_path_filtered = image_path_filtered[1:]

            try:
                database_row = self.parquet_database.loc[image_path_filtered]
            except KeyError:
                database_row = None

            if database_row is None:
                statistics.setdefault("skipped", {}).setdefault("metadata_missing", 0)
                statistics["skipped"]["metadata_missing"] += 1
                return aspect_ratio_bucket_indices

            if self.dataset_type is DatasetType.AUDIO:
                return self._process_audio_bucket(
                    image_path_str=image_path_str,
                    database_row=database_row,
                    aspect_ratio_bucket_indices=aspect_ratio_bucket_indices,
                    metadata_updates=metadata_updates,
                    delete_problematic_images=delete_problematic_images,
                    statistics=statistics,
                )

            # 2. Check if it's image or video by extension
            extension = os.path.splitext(image_path_str)[1].lower().strip(".")
            is_video = extension in video_file_extensions
            # We'll store image/video metadata in the same place
            image_metadata = {}

            # 3. If it's an image, get width/height from parquet columns
            width_column = self.parquet_config.get("width_column", "width")
            height_column = self.parquet_config.get("height_column", "height")

            if is_video:
                if self.num_frames_column:
                    num_frames_val = database_row.get(self.num_frames_column, None)
                    if num_frames_val is not None:
                        num_frames_val = self._get_first_value(num_frames_val)
                        image_metadata["num_frames"] = num_frames_val
                        # check frame count constraints
                        if self.minimum_num_frames and num_frames_val < self.minimum_num_frames:
                            statistics.setdefault("skipped", {}).setdefault("too_small", 0)
                            statistics["skipped"]["too_small"] += 1
                            return aspect_ratio_bucket_indices

                        if self.maximum_num_frames and num_frames_val > self.maximum_num_frames:
                            if self.delete_unwanted_images:
                                try:
                                    self.data_backend.delete(image_path_str)
                                except:
                                    pass
                            statistics.setdefault("skipped", {}).setdefault("too_small", 0)
                            statistics["skipped"]["too_small"] += 1
                            return aspect_ratio_bucket_indices
                else:
                    # no frame count column configured for video
                    logger.warning(f"Video {image_path_str} found but no num_frames_column in parquet_config. Skipping.")
                    return aspect_ratio_bucket_indices

                # For videos, also store fallback width/height if provided in parquet
                w = database_row.get(width_column, None)
                h = database_row.get(height_column, None)
                if w is not None and h is not None:
                    w = self._get_first_value(w)
                    h = self._get_first_value(h)
                    image_metadata["original_size"] = (w, h)
                else:
                    # no dimensions available for video
                    logger.warning(f"Video {image_path_str} found but no width/height columns in parquet_config. Skipping.")
                    return aspect_ratio_bucket_indices

            else:
                # It's an image, use width/height for min size checks
                if width_column is None or height_column is None:
                    raise ValueError("ParquetMetadataBackend requires width and height columns for images.")
                w = self._get_first_value(database_row[width_column])
                h = self._get_first_value(database_row[height_column])
                image_metadata["original_size"] = (w, h)

            # check size constraints
            if not self.meets_resolution_requirements(image_metadata=image_metadata):
                if self.delete_unwanted_images:
                    try:
                        self.data_backend.delete(image_path_str)
                    except:
                        pass
                statistics.setdefault("skipped", {}).setdefault("too_small", 0)
                statistics["skipped"]["too_small"] += 1
                return aspect_ratio_bucket_indices

            # 5. Create a TrainingSample with minimal metadata
            training_sample = TrainingSample(
                image=None,  # no actual image data
                data_backend_id=self.id,
                image_metadata={"original_size": image_metadata.get("original_size")},
                image_path=image_path_str,
            )
            prepared_sample = training_sample.prepare()

            # We'll store final aspect ratio from the TrainingSample or parquet
            aspect_ratio_column = self.parquet_config.get("aspect_ratio_column")
            if aspect_ratio_column and aspect_ratio_column in database_row:
                raw_ar = database_row[aspect_ratio_column]
                raw_ar = float(self._get_first_value(raw_ar))
                aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(raw_ar)
            else:
                aspect_ratio = float(prepared_sample.aspect_ratio)

            # Build the final metadata
            image_metadata.update(
                {
                    "intermediary_size": prepared_sample.intermediary_size,
                    "crop_coordinates": prepared_sample.crop_coordinates,
                    "target_size": prepared_sample.target_size,
                    "aspect_ratio": aspect_ratio,
                }
            )
            # if is_video and you want to store e.g. fps
            if is_video and self.fps_column:
                fps_val = database_row.get(self.fps_column, None)
                if fps_val is not None:
                    fps_val = self._get_first_value(fps_val)
                    image_metadata["fps"] = fps_val

            # Insert into bucket
            aspect_ratio_key = str(aspect_ratio)
            if aspect_ratio_key not in aspect_ratio_bucket_indices:
                aspect_ratio_bucket_indices[aspect_ratio_key] = []
            aspect_ratio_bucket_indices[aspect_ratio_key].append(image_path_str)

            # If we are accumulating metadata updates
            if metadata_updates is not None:
                metadata_updates[image_path_str] = image_metadata

        except Exception as e:
            logger.error(f"Error processing file {image_path_str}: {e}")
            logger.error(traceback.format_exc())
            if delete_problematic_images:
                logger.error(f"Deleting file {image_path_str} after error.")
                self.data_backend.delete(image_path_str)

        return aspect_ratio_bucket_indices

    def _extract_audio_value(self, database_row, column_name: Optional[str]):
        if not column_name:
            return None
        value = database_row.get(column_name, None)
        if value is None:
            return None
        return self._get_first_value(value)

    def _process_audio_bucket(
        self,
        image_path_str: str,
        database_row,
        aspect_ratio_bucket_indices: dict,
        metadata_updates=None,
        delete_problematic_images: bool = False,
        statistics: Optional[dict] = None,
    ):
        if statistics is None:
            statistics = {}
        if database_row is None:
            statistics.setdefault("skipped", {}).setdefault("metadata_missing", 0)
            statistics["skipped"]["metadata_missing"] += 1
            return aspect_ratio_bucket_indices

        try:
            sample_rate = self._extract_audio_value(database_row, self.parquet_config.get("audio_sample_rate_column"))
            num_samples = self._extract_audio_value(database_row, self.parquet_config.get("audio_num_samples_column"))
            duration_seconds = self._extract_audio_value(database_row, self.parquet_config.get("audio_duration_column"))
            num_channels = self._extract_audio_value(database_row, self.parquet_config.get("audio_channels_column"))

            if (sample_rate is None or num_samples is None) and self.data_backend.exists(image_path_str):
                audio_payload = self.data_backend.read(image_path_str)
                if audio_payload is None:
                    statistics.setdefault("skipped", {}).setdefault("not_found", 0)
                    statistics["skipped"]["not_found"] += 1
                    return aspect_ratio_bucket_indices
                buffer = BytesIO(audio_payload) if not isinstance(audio_payload, BytesIO) else audio_payload
                buffer.seek(0)
                waveform, inferred_sample_rate = load_audio(buffer)
                if sample_rate is None:
                    sample_rate = inferred_sample_rate
                if waveform is not None and hasattr(waveform, "shape") and len(waveform.shape) >= 2:
                    inferred_channels = waveform.shape[0]
                    if num_samples is None:
                        num_samples = waveform.shape[1]
                    if num_channels is None:
                        num_channels = inferred_channels

            if duration_seconds is None and sample_rate and num_samples:
                duration_seconds = float(num_samples) / float(sample_rate)

            overrides = {}
            lyrics_column = self.parquet_config.get("lyrics_column")
            if lyrics_column:
                lyrics_value = self._extract_audio_value(database_row, lyrics_column)
                if lyrics_value:
                    overrides["lyrics"] = lyrics_value

            audio_metadata = self._build_audio_metadata_entry(
                sample_path=image_path_str,
                sample_rate=sample_rate,
                num_channels=num_channels,
                num_samples=num_samples,
                duration_seconds=duration_seconds,
                overrides=overrides,
            )

            max_duration = self.audio_max_duration_seconds
            if max_duration is not None and duration_seconds and duration_seconds > max_duration:
                logger.debug(
                    f"Audio sample {image_path_str} duration {duration_seconds:.2f}s exceeds "
                    f"limit {max_duration:.2f}s. Skipping."
                )
                skipped = statistics.setdefault("skipped", {})
                skipped["too_long"] = skipped.get("too_long", 0) + 1
                return aspect_ratio_bucket_indices

            bucket_key, truncated_duration = self._compute_audio_bucket(duration_seconds)
            audio_metadata["original_duration_seconds"] = duration_seconds
            if truncated_duration is not None:
                audio_metadata["duration_seconds"] = truncated_duration
                audio_metadata["bucket_duration_seconds"] = truncated_duration
            aspect_ratio_bucket_indices.setdefault(bucket_key, []).append(image_path_str)

            if metadata_updates is not None:
                metadata_updates[image_path_str] = audio_metadata
        except Exception as exc:
            logger.error(f"Error processing audio metadata for {image_path_str}: {exc}", exc_info=True)
            if delete_problematic_images:
                logger.error(f"Deleting audio sample {image_path_str}.")
                self.data_backend.delete(image_path_str)

        return aspect_ratio_bucket_indices

    def __len__(self):

        def repeat_len(bucket):
            return len(bucket) * (self.repeats + 1)

        return sum(
            (repeat_len(bucket) + (self.batch_size - 1)) // self.batch_size
            for bucket in self.aspect_ratio_bucket_indices.values()
            if repeat_len(bucket) >= self.batch_size
        )
