import json
import logging
import os
import shutil
import traceback
from io import BytesIO
from typing import Optional

from simpletuner.helpers.audio import generate_zero_audio, load_audio, load_audio_from_video
from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.image_manipulation.brightness import calculate_luminance
from simpletuner.helpers.image_manipulation.load import load_image, load_video
from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.training import audio_file_extensions, image_file_extensions, video_file_extensions
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


logger = logging.getLogger("DiscoveryMetadataBackend")
if should_log():
    target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
else:
    target_level = "ERROR"
logger.setLevel(target_level)

_ffprobe_available: Optional[bool] = None


def _is_ffprobe_available() -> bool:
    global _ffprobe_available
    if _ffprobe_available is None:
        _ffprobe_available = shutil.which("ffprobe") is not None
    return _ffprobe_available


class DiscoveryMetadataBackend(MetadataBackend):
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

    def _should_use_metadata_only_for_video(self) -> bool:
        if not _is_ffprobe_available():
            return False
        if self.dataset_type is not DatasetType.VIDEO:
            return False
        crop_enabled = bool(self.dataset_config.get("crop", False))
        crop_style = str(self.dataset_config.get("crop_style") or "random").lower()
        if crop_enabled and crop_style == "face":
            return False
        return True

    def _needs_video_frame_count(self) -> bool:
        if self.bucket_strategy == "resolution_frames":
            return True
        return self.minimum_num_frames is not None or self.maximum_num_frames is not None

    @staticmethod
    def _parse_frame_rate(value: object) -> Optional[float]:
        if value in (None, "", "N/A"):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text or text.upper() == "N/A":
            return None
        if "/" in text:
            num_text, denom_text = text.split("/", 1)
            try:
                denom = float(denom_text)
                if denom == 0:
                    return None
                return float(num_text) / denom
            except (TypeError, ValueError):
                return None
        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: object) -> Optional[int]:
        if value in (None, "", "N/A"):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        if value in (None, "", "N/A"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _probe_video_metadata(self, video_path: str, payload: Optional[bytes]) -> Optional[dict]:
        if not _is_ffprobe_available():
            return None

        import subprocess
        import tempfile
        from pathlib import Path

        cleanup_temp = False
        probe_path = video_path
        suffix = Path(video_path).suffix or ".mp4"

        if payload is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(payload)
                    probe_path = tmp.name
                cleanup_temp = True
            except Exception as exc:
                logger.debug("Failed to write temp video for ffprobe: %s", exc)
                return None

        try:
            cmd = [
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
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            raw_payload = json.loads(result.stdout) if result.stdout else {}
            streams = raw_payload.get("streams") or []
            stream = streams[0] if streams else {}

            width = self._safe_int(stream.get("width"))
            height = self._safe_int(stream.get("height"))
            if not width or not height:
                return None

            metadata: dict = {"original_size": (width, height)}

            fps = self._parse_frame_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
            if fps:
                metadata["fps"] = fps

            num_frames = self._safe_int(stream.get("nb_frames"))
            duration = self._safe_float(stream.get("duration"))
            if duration is None:
                duration = self._safe_float((raw_payload.get("format") or {}).get("duration"))
            if num_frames is None and fps and duration:
                num_frames = int(round(fps * duration))
            if num_frames:
                metadata["num_frames"] = num_frames
            if duration:
                metadata["video_duration"] = duration

            return metadata
        except Exception as exc:
            logger.debug("ffprobe metadata extraction failed for %s: %s", video_path, exc)
            return None
        finally:
            if cleanup_temp:
                try:
                    Path(probe_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _discover_new_files(self, for_metadata: bool = False, ignore_existing_cache: bool = False):
        """
        Discover new files that have not been processed yet.

        Returns:
            list: A list of new files.
        """
        all_image_files = StateTracker.get_image_files(data_backend_id=self.data_backend.id)
        if ignore_existing_cache:
            # Return all files and remove the existing buckets.
            logger.debug("Resetting the entire aspect bucket cache as we've received the signal to ignore existing cache.")
            self.aspect_ratio_bucket_indices = {}
            return list(all_image_files.keys())
        if all_image_files is None:
            logger.debug("No image file cache available, retrieving fresh")
            if self.dataset_type is DatasetType.AUDIO:
                # Check if audio is sourced from video files
                if self.audio_config.get("source_from_video", False):
                    extension_pool = video_file_extensions
                else:
                    extension_pool = audio_file_extensions
            else:
                extension_pool = image_file_extensions
            all_image_files = self.data_backend.list_files(
                instance_data_dir=self.instance_data_dir,
                file_extensions=extension_pool,
            )
            all_image_files = StateTracker.set_image_files(all_image_files, data_backend_id=self.data_backend.id)
        else:
            logger.debug("Using cached image file list")

        # Flatten the list if it contains nested lists
        if any(isinstance(i, list) for i in all_image_files):
            all_image_files = [item for sublist in all_image_files for item in sublist]

        # logger.debug(f"All image files: {json.dumps(all_image_files, indent=4)}")

        all_image_files_set = set(all_image_files)

        if for_metadata:
            result = [file for file in all_image_files if self.get_metadata_by_filepath(file) is None]
        else:
            processed_files = set(path for paths in self.aspect_ratio_bucket_indices.values() for path in paths)
            result = [file for file in all_image_files_set if file not in processed_files]

        return result

    def reload_cache(self, set_config: bool = True):
        """
        Load cache data from a JSON file.

        Returns:
            dict: The cache data.
        """
        # Query our DataBackend to see whether the cache file exists.
        logger.debug(f"Checking for cache file: {self.cache_file}")
        if self.data_backend.exists(self.cache_file):
            try:
                # Use our DataBackend to actually read the cache file.
                logger.debug("Pulling cache file from storage")
                cache_data_raw = self.data_backend.read(self.cache_file)
                cache_data = json.loads(cache_data_raw)
            except Exception as e:
                logger.warning(f"Error loading aspect bucket cache, creating new one: {e}")
                cache_data = {}
            # Coerce bucket keys from strings to floats (JSON serialization converts float keys to strings)
            loaded_indices = cache_data.get("aspect_ratio_bucket_indices", {})
            self.aspect_ratio_bucket_indices = _coerce_bucket_keys_to_float(loaded_indices)
            if set_config:
                self.config = cache_data.get("config", {})
                if self.config != {}:
                    logger.debug(f"Setting config to {self.config}")
                    logger.debug(f"Loaded previous data backend config: {self.config}")
                    StateTracker.set_data_backend_config(
                        data_backend_id=self.id,
                        config=self.config,
                    )
            logger.debug(f"(id={self.id}) Loaded {len(self.aspect_ratio_bucket_indices)} aspect ratio buckets")
        else:
            logger.warning("No cache file found, creating new one.")

    def save_cache(self, enforce_constraints: bool = False):
        """
        Save cache data to file.
        """
        # Prune any buckets that have fewer samples than batch_size
        if enforce_constraints:
            self._enforce_min_bucket_size()
        self._enforce_min_aspect_ratio()
        self._enforce_max_aspect_ratio()
        if self.read_only:
            logger.debug("Skipping cache update on storage backend, read-only mode.")
            return
        # Convert any non-strings into strings as we save the index.
        aspect_ratio_bucket_indices_str = {
            key: [str(path) for path in value] for key, value in self.aspect_ratio_bucket_indices.items()
        }
        # Encode the cache as JSON.
        cache_data = {
            "config": StateTracker.get_data_backend_config(data_backend_id=self.data_backend.id),
            "aspect_ratio_bucket_indices": aspect_ratio_bucket_indices_str,
        }
        logger.debug(f"save_cache has config to write: {cache_data['config']}")
        cache_data_str = json.dumps(cache_data)
        # Use our DataBackend to write the cache file.
        self.data_backend.write(self.cache_file, cache_data_str)

    def load_image_metadata(self):
        """Load image metadata from a JSON file."""
        self.image_metadata = {}
        self.image_metadata_loaded = False
        if self.data_backend.exists(self.metadata_file):
            cache_data_raw = self.data_backend.read(self.metadata_file)
            self.image_metadata = json.loads(cache_data_raw)
            self.image_metadata_loaded = True

    def save_image_metadata(self):
        """Save image metadata to a JSON file."""
        self.data_backend.write(self.metadata_file, json.dumps(self.image_metadata))

    def _process_for_bucket(
        self,
        image_path_str,
        aspect_ratio_bucket_indices,
        aspect_ratio_rounding: int = 3,
        metadata_updates=None,
        delete_problematic_images: bool = False,
        statistics: dict = {},
    ):
        if self.dataset_type is DatasetType.AUDIO:
            return self._process_audio_sample(
                image_path_str=image_path_str,
                aspect_ratio_bucket_indices=aspect_ratio_bucket_indices,
                metadata_updates=metadata_updates,
                delete_problematic_images=delete_problematic_images,
                statistics=statistics,
            )

        try:
            image_metadata = {}
            image_data = None
            image = None
            file_extension = os.path.splitext(image_path_str)[1].lower()
            is_video_file = file_extension.strip(".") in video_file_extensions

            use_metadata_only = False
            if is_video_file and self._should_use_metadata_only_for_video():
                if getattr(self.data_backend, "type", None) == "local":
                    video_metadata = self._probe_video_metadata(image_path_str, None)
                else:
                    image_data = self.data_backend.read(image_path_str)
                    if image_data is None:
                        logger.debug(f"Image {image_path_str} was not found on the backend. Skipping image.")
                        statistics.setdefault("skipped", {}).setdefault("not_found", 0)
                        statistics["skipped"]["not_found"] += 1
                        return aspect_ratio_bucket_indices
                    video_metadata = self._probe_video_metadata(image_path_str, image_data)

                if video_metadata and self._needs_video_frame_count() and "num_frames" not in video_metadata:
                    logger.debug(
                        "(id=%s) ffprobe metadata missing num_frames for %s; falling back to full decode.",
                        self.id,
                        image_path_str,
                    )
                    video_metadata = None

                if video_metadata:
                    use_metadata_only = True
                    image_metadata.update(video_metadata)
                    logger.debug("(id=%s) Using ffprobe metadata-only scan for %s", self.id, image_path_str)

            if use_metadata_only:
                if not self.meets_resolution_requirements(image_metadata=image_metadata):
                    if not self.delete_unwanted_images:
                        logger.debug(f"Image {image_path_str} does not meet minimum size requirements. Skipping image.")
                    else:
                        logger.debug(f"Image {image_path_str} does not meet minimum size requirements. Deleting image.")
                        self.data_backend.delete(image_path_str)
                    statistics.setdefault("skipped", {}).setdefault("too_small", 0)
                    statistics["skipped"]["too_small"] += 1
                    return aspect_ratio_bucket_indices
            else:
                if image_data is None:
                    image_data = self.data_backend.read(image_path_str)
                if image_data is None:
                    logger.debug(f"Image {image_path_str} was not found on the backend. Skipping image.")
                    statistics.setdefault("skipped", {}).setdefault("not_found", 0)
                    statistics["skipped"]["not_found"] += 1
                    return aspect_ratio_bucket_indices

                file_loader = load_image
                if is_video_file:
                    file_loader = load_video
                image = file_loader(BytesIO(image_data))
                if not self.meets_resolution_requirements(image=image):
                    if not self.delete_unwanted_images:
                        logger.debug(f"Image {image_path_str} does not meet minimum size requirements. Skipping image.")
                    else:
                        logger.debug(f"Image {image_path_str} does not meet minimum size requirements. Deleting image.")
                        self.data_backend.delete(image_path_str)
                    statistics.setdefault("skipped", {}).setdefault("too_small", 0)
                    statistics["skipped"]["too_small"] += 1
                    return aspect_ratio_bucket_indices

            if hasattr(image, "shape"):
                image_metadata["original_size"] = (image.shape[2], image.shape[1])
                image_metadata["num_frames"] = image.shape[0]
            elif hasattr(image, "resize"):
                image_metadata["original_size"] = image.size
            training_sample = TrainingSample(
                image=image,
                data_backend_id=self.id,
                image_metadata=image_metadata,
                image_path=image_path_str,
                model=StateTracker.get_model(),
            )
            prepared_sample = training_sample.prepare()
            cur_image_metadata = {
                "crop_coordinates": prepared_sample.crop_coordinates,
                "target_size": prepared_sample.target_size,
                "intermediary_size": prepared_sample.intermediary_size,
                "aspect_ratio": prepared_sample.aspect_ratio,
            }
            if image is not None:
                cur_image_metadata["luminance"] = calculate_luminance(image)
            image_metadata.update(cur_image_metadata)
            logger.debug(f"Image {image_path_str} has metadata: {cur_image_metadata}")

            # Determine bucket key based on strategy
            is_video = "num_frames" in image_metadata
            if is_video and self.bucket_strategy == "resolution_frames":
                target_w, target_h = prepared_sample.target_size
                num_frames = image_metadata["num_frames"]
                bucket_key, rounded_frames = self._compute_video_bucket(target_w, target_h, num_frames)
                image_metadata["bucket_frames"] = rounded_frames
            else:
                bucket_key = str(prepared_sample.aspect_ratio)
            if bucket_key not in aspect_ratio_bucket_indices:
                aspect_ratio_bucket_indices[bucket_key] = []
            aspect_ratio_bucket_indices[bucket_key].append(image_path_str)

            if metadata_updates is not None:
                metadata_updates[image_path_str] = image_metadata

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            if delete_problematic_images:
                logger.error(f"Deleting image {image_path_str}.")
                self.data_backend.delete(image_path_str)

        return aspect_ratio_bucket_indices

    def _process_audio_sample(
        self,
        image_path_str: str,
        aspect_ratio_bucket_indices: dict,
        metadata_updates=None,
        delete_problematic_images: bool = False,
        statistics: Optional[dict] = None,
    ):
        if statistics is None:
            statistics = {}
        try:
            audio_payload = self.data_backend.read(image_path_str)
            if audio_payload is None:
                logger.debug(f"Audio sample {image_path_str} was not found on the backend. Skipping.")
                statistics.setdefault("skipped", {}).setdefault("not_found", 0)
                statistics["skipped"]["not_found"] += 1
                return aspect_ratio_bucket_indices

            # Check if audio is sourced from video files
            source_from_video = self.audio_config.get("source_from_video", False)
            allow_zero_audio = self.audio_config.get("allow_zero_audio", False)
            file_ext = os.path.splitext(image_path_str)[1].lower().lstrip(".")
            is_video_file = file_ext in video_file_extensions

            if source_from_video and is_video_file:
                # Extract audio from video file
                target_sr = self.audio_config.get("sample_rate", 16000)
                target_channels = self.audio_config.get("channels", 1)
                try:
                    waveform, sample_rate = load_audio_from_video(audio_payload, target_sr, target_channels)
                except ValueError:
                    # Video has no audio stream
                    if allow_zero_audio:
                        video_duration = self._get_video_duration(image_path_str, audio_payload)
                        if video_duration and video_duration > 0:
                            waveform, sample_rate = generate_zero_audio(video_duration, target_sr, target_channels)
                            logger.debug(f"Generated zero audio ({video_duration:.2f}s) for {image_path_str}")
                        else:
                            logger.debug(f"Skipping video without audio (no duration): {image_path_str}")
                            statistics.setdefault("skipped", {}).setdefault("no_duration", 0)
                            statistics["skipped"]["no_duration"] += 1
                            return aspect_ratio_bucket_indices
                    else:
                        logger.debug(f"Skipping video without audio: {image_path_str}")
                        statistics.setdefault("skipped", {}).setdefault("no_audio", 0)
                        statistics["skipped"]["no_audio"] += 1
                        return aspect_ratio_bucket_indices
            else:
                buffer = BytesIO(audio_payload) if not isinstance(audio_payload, BytesIO) else audio_payload
                buffer.seek(0)
                waveform, sample_rate = load_audio(buffer)
            if waveform is None or waveform.numel() == 0:
                logger.debug(f"Audio sample {image_path_str} is empty. Skipping.")
                statistics.setdefault("skipped", {}).setdefault("other", 0)
                statistics["skipped"]["other"] += 1
                return aspect_ratio_bucket_indices

            if not hasattr(waveform, "shape") or len(waveform.shape) < 2:
                logger.debug(
                    f"Audio sample {image_path_str} has malformed shape {getattr(waveform, 'shape', None)}. Skipping."
                )
                statistics.setdefault("skipped", {}).setdefault("malformed_shape", 0)
                statistics["skipped"]["malformed_shape"] += 1
                return aspect_ratio_bucket_indices

            num_channels, num_samples = waveform.shape[0], waveform.shape[1]
            duration_seconds = float(num_samples) / float(sample_rate) if sample_rate else None
            audio_metadata = self._build_audio_metadata_entry(
                sample_path=image_path_str,
                sample_rate=sample_rate,
                num_channels=num_channels,
                num_samples=num_samples,
                duration_seconds=duration_seconds,
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
            logger.error(f"Error processing audio sample {image_path_str}: {exc}", exc_info=True)
            if delete_problematic_images:
                logger.error(f"Deleting audio sample {image_path_str}.")
                self.data_backend.delete(image_path_str)

        return aspect_ratio_bucket_indices

    def _get_video_duration(self, video_path: str, video_bytes: bytes = None) -> Optional[float]:
        """
        Get video duration for zero-audio generation.

        Args:
            video_path: Path to the video file
            video_bytes: Raw video bytes (optional, used if file not directly accessible)

        Returns:
            Duration in seconds, or None if unavailable.
        """
        import subprocess
        import tempfile
        from pathlib import Path

        # Try to get from existing video metadata if this is a linked dataset
        source_dataset_id = self.dataset_config.get("source_dataset_id")
        if source_dataset_id:
            video_meta = StateTracker.get_metadata_by_filepath(video_path, source_dataset_id)
            if video_meta:
                # Check for video_duration or compute from num_frames/fps if available
                if "video_duration" in video_meta:
                    return video_meta["video_duration"]
                if "num_frames" in video_meta:
                    # Assume 24fps if not specified for duration estimate
                    fps = video_meta.get("fps", 24)
                    return video_meta["num_frames"] / fps

        # Fallback: probe video file with ffprobe
        cleanup_temp = False
        probe_path = video_path
        try:
            if video_bytes:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(video_bytes)
                    probe_path = tmp.name
                cleanup_temp = True

            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                probe_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            return float(result.stdout.strip())
        except Exception as e:
            logger.debug(f"Failed to get video duration for {video_path}: {e}")
            return None
        finally:
            if cleanup_temp:
                try:
                    Path(probe_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def __len__(self):
        """
        Returns:
            int: The number of batches in the dataset, accounting for images that can't form a complete batch and are discarded.
        """

        def repeat_len(bucket):
            return len(bucket) * (self.repeats + 1)

        return sum(
            (repeat_len(bucket) + (self.batch_size - 1)) // self.batch_size
            for bucket in self.aspect_ratio_bucket_indices.values()
            if repeat_len(bucket) >= self.batch_size
        )
