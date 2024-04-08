from helpers.training.state_tracker import StateTracker
from helpers.multiaspect.image import MultiaspectImage
from helpers.data_backend.base import BaseDataBackend
from helpers.metadata.backends.base import MetadataBackend
from pathlib import Path
import json, logging, os, time, re
from multiprocessing import Manager
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process, Queue
import numpy as np
from math import floor
from io import BytesIO
from helpers.image_manipulation.brightness import calculate_luminance

logger = logging.getLogger("JsonMetadataBackend")
target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
logger.setLevel(target_level)


class JsonMetadataBackend(MetadataBackend):
    def __init__(
        self,
        id: str,
        instance_data_root: str,
        cache_file: str,
        metadata_file: str,
        data_backend: BaseDataBackend,
        accelerator,
        batch_size: int,
        resolution: float,
        resolution_type: str,
        delete_problematic_images: bool = False,
        metadata_update_interval: int = 3600,
        minimum_image_size: int = None,
    ):
        super().__init__(
            id=id,
            instance_data_root=instance_data_root,
            cache_file=cache_file,
            metadata_file=metadata_file,
            data_backend=data_backend,
            accelerator=accelerator,
            batch_size=batch_size,
            resolution=resolution,
            resolution_type=resolution_type,
            delete_problematic_images=delete_problematic_images,
            metadata_update_interval=metadata_update_interval,
            minimum_image_size=minimum_image_size,
        )

    def __len__(self):
        """
        Returns:
            int: The number of batches in the dataset, accounting for images that can't form a complete batch and are discarded.
        """
        return sum(
            len(bucket) // self.batch_size
            for bucket in self.aspect_ratio_bucket_indices.values()
            if len(bucket) >= self.batch_size
        )

    def _discover_new_files(self, for_metadata: bool = False):
        """
        Discover new files that have not been processed yet.

        Returns:
            list: A list of new files.
        """
        listed_image_files = StateTracker.get_image_files(
            data_backend_id=self.data_backend.id
        )
        if listed_image_files is None:
            logger.debug("No image file cache available, retrieving fresh")
            listed_image_files = self.data_backend.list_files(
                instance_data_root=self.instance_data_root,
                str_pattern="*.[jJpP][pPnN][gG]",
            )
            # flatten the os.path.walk results into a dictionary
            all_image_files = []
            for sublist in all_image_files:
                for root, dirs, files in sublist:
                    for file in files:
                        if re.match(r".*\.(jpg|jpeg|png)$", file, re.IGNORECASE):
                            all_image_files.append(os.path.join(root, file))

            StateTracker.set_image_files(
                all_image_files, data_backend_id=self.data_backend.id
            )
        else:
            logger.debug("Using cached image file list")
            all_image_files = listed_image_files
        del listed_image_files

        logger.debug(
            f"Before flattening, all image files: {json.dumps(all_image_files, indent=4)}"
        )

        # Flatten the list if it contains nested lists
        if any(isinstance(i, list) for i in all_image_files):
            all_image_files = [item for sublist in all_image_files for item in sublist]

        logger.debug(f"All image files: {json.dumps(all_image_files, indent=4)}")

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

    def reload_cache(self):
        """
        Load cache data from a JSON file.

        Returns:
            dict: The cache data.
        """
        # Query our DataBackend to see whether the cache file exists.
        logger.info(f"Checking for cache file: {self.cache_file}")
        if self.data_backend.exists(self.cache_file):
            try:
                # Use our DataBackend to actually read the cache file.
                logger.info(f"Pulling cache file from storage")
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
                del image_data
                statistics["skipped"]["not_found"] += 1
                return aspect_ratio_bucket_indices
            with Image.open(BytesIO(image_data)) as image:
                # Apply EXIF transforms
                if not self.meets_resolution_requirements(
                    image=image,
                ):
                    logger.debug(
                        f"Image {image_path_str} does not meet minimum image size requirements. Skipping image."
                    )
                    statistics["skipped"]["too_small"] += 1
                    return aspect_ratio_bucket_indices
                image_metadata["original_size"] = image.size
                original_aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(
                    image, aspect_ratio_rounding
                )
                image, crop_coordinates, new_aspect_ratio = (
                    MultiaspectImage.prepare_image(
                        image=image,
                        resolution=self.resolution,
                        resolution_type=self.resolution_type,
                        id=self.data_backend.id,
                    )
                )
                image_metadata["crop_coordinates"] = crop_coordinates
                image_metadata["target_size"] = image.size
                # Round to avoid excessive unique buckets
                aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(
                    image, aspect_ratio_rounding
                )
                image_metadata["aspect_ratio"] = aspect_ratio
                image_metadata["luminance"] = calculate_luminance(image)
                logger.debug(
                    f"Image {image_path_str} has aspect ratio {aspect_ratio} and size {image.size}."
                )

            # Create a new bucket if it doesn't exist
            if str(aspect_ratio) not in aspect_ratio_bucket_indices:
                aspect_ratio_bucket_indices[str(aspect_ratio)] = []
            aspect_ratio_bucket_indices[str(aspect_ratio)].append(image_path_str)
            # Instead of directly updating, just fill the provided dictionary
            if metadata_updates is not None:
                metadata_updates[image_path_str] = image_metadata
        except Exception as e:
            import traceback

            logger.error(f"Error processing image: {e}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            logger.error(e)
            if delete_problematic_images:
                logger.error(f"Deleting image.")
                self.data_backend.delete(image_path_str)
        return aspect_ratio_bucket_indices
