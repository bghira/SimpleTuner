from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.image_manipulation.load import load_image, load_video
from simpletuner.helpers.training.multi_process import should_log
import json
import logging
import os
import traceback
from io import BytesIO
from simpletuner.helpers.image_manipulation.brightness import calculate_luminance
from simpletuner.helpers.training import image_file_extensions, video_file_extensions

logger = logging.getLogger("DiscoveryMetadataBackend")
if should_log():
    target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
else:
    target_level = "ERROR"
logger.setLevel(target_level)


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

    def _discover_new_files(
        self, for_metadata: bool = False, ignore_existing_cache: bool = False
    ):
        """
        Discover new files that have not been processed yet.

        Returns:
            list: A list of new files.
        """
        all_image_files = StateTracker.get_image_files(
            data_backend_id=self.data_backend.id
        )
        if ignore_existing_cache:
            # Return all files and remove the existing buckets.
            logger.debug(
                "Resetting the entire aspect bucket cache as we've received the signal to ignore existing cache."
            )
            self.aspect_ratio_bucket_indices = {}
            return list(all_image_files.keys())
        if all_image_files is None:
            logger.debug("No image file cache available, retrieving fresh")
            all_image_files = self.data_backend.list_files(
                instance_data_dir=self.instance_data_dir,
                file_extensions=image_file_extensions,
            )
            all_image_files = StateTracker.set_image_files(
                all_image_files, data_backend_id=self.data_backend.id
            )
        else:
            logger.debug("Using cached image file list")

        # Flatten the list if it contains nested lists
        if any(isinstance(i, list) for i in all_image_files):
            all_image_files = [item for sublist in all_image_files for item in sublist]

        # logger.debug(f"All image files: {json.dumps(all_image_files, indent=4)}")

        all_image_files_set = set(all_image_files)

        if for_metadata:
            result = [
                file
                for file in all_image_files
                if self.get_metadata_by_filepath(file) is None
            ]
        else:
            processed_files = set(
                path
                for paths in self.aspect_ratio_bucket_indices.values()
                for path in paths
            )
            result = [
                file for file in all_image_files_set if file not in processed_files
            ]

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
                logger.warning(
                    f"Error loading aspect bucket cache, creating new one: {e}"
                )
                cache_data = {}
            self.aspect_ratio_bucket_indices = cache_data.get(
                "aspect_ratio_bucket_indices", {}
            )
            if set_config:
                self.config = cache_data.get("config", {})
                if self.config != {}:
                    logger.debug(f"Setting config to {self.config}")
                    logger.debug(f"Loaded previous data backend config: {self.config}")
                    StateTracker.set_data_backend_config(
                        data_backend_id=self.id,
                        config=self.config,
                    )
            logger.debug(
                f"(id={self.id}) Loaded {len(self.aspect_ratio_bucket_indices)} aspect ratio buckets"
            )
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
            key: [str(path) for path in value]
            for key, value in self.aspect_ratio_bucket_indices.items()
        }
        # Encode the cache as JSON.
        cache_data = {
            "config": StateTracker.get_data_backend_config(
                data_backend_id=self.data_backend.id
            ),
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
        try:
            image_metadata = {}
            image_data = self.data_backend.read(image_path_str)
            if image_data is None:
                logger.debug(
                    f"Image {image_path_str} was not found on the backend. Skipping image."
                )
                statistics.setdefault("skipped", {}).setdefault("not_found", 0)
                statistics["skipped"]["not_found"] += 1
                return aspect_ratio_bucket_indices

            file_extension = os.path.splitext(image_path_str)[1].lower()
            file_loader = load_image
            if file_extension.strip(".") in video_file_extensions:
                file_loader = load_video
            image = file_loader(BytesIO(image_data))
            if not self.meets_resolution_requirements(image=image):
                if not self.delete_unwanted_images:
                    logger.debug(
                        f"Image {image_path_str} does not meet minimum size requirements. Skipping image."
                    )
                else:
                    logger.debug(
                        f"Image {image_path_str} does not meet minimum size requirements. Deleting image."
                    )
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
                "luminance": calculate_luminance(image),
            }
            image_metadata.update(cur_image_metadata)
            logger.debug(f"Image {image_path_str} has metadata: {cur_image_metadata}")

            aspect_ratio_key = str(prepared_sample.aspect_ratio)
            if aspect_ratio_key not in aspect_ratio_bucket_indices:
                aspect_ratio_bucket_indices[aspect_ratio_key] = []
            aspect_ratio_bucket_indices[aspect_ratio_key].append(image_path_str)

            if metadata_updates is not None:
                metadata_updates[image_path_str] = image_metadata

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            if delete_problematic_images:
                logger.error(f"Deleting image {image_path_str}.")
                self.data_backend.delete(image_path_str)

        return aspect_ratio_bucket_indices

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
