from helpers.training.state_tracker import StateTracker
from helpers.multiaspect.image import MultiaspectImage
from helpers.data_backend.base import BaseDataBackend
from helpers.metadata.backends.base import MetadataBackend
import json, logging, os
from io import BytesIO
from PIL import Image

logger = logging.getLogger("ParquetMetadataBackend")
target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING")
logger.setLevel(target_level)

try:
    import pandas as pd
except ImportError:
    raise ImportError("Pandas is required for the ParquetMetadataBackend.")


class ParquetMetadataBackend(MetadataBackend):
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
        parquet_config: dict,
        delete_problematic_images: bool = False,
        metadata_update_interval: int = 3600,
        minimum_image_size: int = None,
    ):
        self.parquet_config = parquet_config
        self.parquet_path = parquet_config.get("path", None)
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
        self.load_parquet_database()

    def load_parquet_database(self):
        """
        Load the parquet database from file.
        """
        if self.data_backend.exists(self.parquet_path):
            try:
                bytes_string = self.data_backend.read(self.parquet_path)
                import io

                pq = io.BytesIO(bytes_string)
            except Exception as e:
                raise e
            self.parquet_database = pd.read_parquet(pq, engine="pyarrow")
        else:
            raise FileNotFoundError(
                f"Parquet could not be loaded from {self.parquet_path}: database file does not exist (path={self.parquet_path})."
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
        all_image_files = StateTracker.get_image_files(
            data_backend_id=self.data_backend.id
        ) or StateTracker.set_image_files(
            self.data_backend.list_files(
                instance_data_root=self.instance_data_root,
                str_pattern="*.[jJpP][pPnN][gG]",
            ),
            data_backend_id=self.data_backend.id,
        )
        # Log an excerpt of the all_image_files:
        # logger.debug(
        #     f"Found {len(all_image_files)} images in the instance data root (truncated): {list(all_image_files)[:5]}"
        # )
        # Extract only the files from the data
        if for_metadata:
            result = [
                file
                for file in all_image_files
                if self.get_metadata_by_filepath(file) is None
            ]
            # logger.debug(
            #     f"Found {len(result)} new images for metadata scan (truncated): {list(result)[:5]}"
            # )
            return result
        return [
            file
            for file in all_image_files
            if str(file) not in self.instance_images_path
        ]

    def reload_cache(self):
        """
        Load cache data from a parquet file.

        Returns:
            dict: The cache data.
        """
        # Query our DataBackend to see whether the cache file exists.
        if self.data_backend.exists(self.cache_file):
            try:
                # Use our DataBackend to actually read the cache file.
                logger.debug("Pulling cache file from storage.")
                cache_data_raw = self.data_backend.read(self.cache_file)
                cache_data = json.loads(cache_data_raw)
                logger.debug("Completed loading cache data.")
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
            self.instance_images_path = set(cache_data.get("instance_images_path", []))

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
            "instance_images_path": [str(path) for path in self.instance_images_path],
        }
        logger.debug(f"save_cache has config to write: {cache_data['config']}")
        cache_data_str = json.dumps(cache_data)
        # Use our DataBackend to write the cache file.
        self.data_backend.write(self.cache_file, cache_data_str)

    def load_image_metadata(self):
        """Load image metadata from a JSON file."""
        logger.debug(f"Loading metadata: {self.metadata_file}")
        self.image_metadata = {}
        self.image_metadata_loaded = False
        if self.data_backend.exists(self.metadata_file):
            cache_data_raw = self.data_backend.read(self.metadata_file)
            self.image_metadata = json.loads(cache_data_raw)
            self.image_metadata_loaded = True
        logger.debug("Metadata loaded.")

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
            image_path_filtered = os.path.splitext(os.path.split(image_path_str)[-1])[0]
            if image_path_filtered.isdigit():
                image_path_filtered = int(image_path_filtered)
            logger.debug(
                f"Reading image {image_path_str} metadata from parquet backend column {self.parquet_config.get('filename_column')} without instance root dir prefix {self.instance_data_root}: {image_path_filtered}."
            )
            database_image_metadata = self.parquet_database.loc[
                self.parquet_database[self.parquet_config.get("filename_column")]
                == image_path_filtered
            ]
            logger.debug(f"Found image metadata: {database_image_metadata}")
            if database_image_metadata is None:
                logger.debug(
                    f"Image {image_path_str} was not found on the backend. Skipping image."
                )
                statistics["skipped"]["not_found"] += 1
                return aspect_ratio_bucket_indices
            width_column = self.parquet_config.get("width_column", None)
            height_column = self.parquet_config.get("height_column", None)
            if width_column is None or height_column is None:
                raise ValueError(
                    "ParquetMetadataBackend requires width and height columns to be defined."
                )
            image_metadata = {
                "original_size": (
                    int(database_image_metadata[width_column].values[0]),
                    int(database_image_metadata[height_column].values[0]),
                )
            }

            if not self.meets_resolution_requirements(image_metadata=image_metadata):
                logger.debug(
                    f"Image {image_path_str} does not meet minimum image size requirements. Skipping image."
                )
                statistics["skipped"]["too_small"] += 1
                return aspect_ratio_bucket_indices
            aspect_ratio_column = self.parquet_config.get("aspect_ratio_column", None)
            if aspect_ratio_column:
                original_aspect_ratio = database_image_metadata[
                    aspect_ratio_column
                ].values[0]
            else:
                original_aspect_ratio = (
                    image_metadata["original_size"][0]
                    / image_metadata["original_size"][1]
                )
            final_image_size, crop_coordinates, new_aspect_ratio = (
                MultiaspectImage.prepare_image(
                    image_metadata=image_metadata,
                    resolution=self.resolution,
                    resolution_type=self.resolution_type,
                    id=self.data_backend.id,
                )
            )
            image_metadata["crop_coordinates"] = crop_coordinates
            image_metadata["target_size"] = final_image_size
            # Round to avoid excessive unique buckets
            aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(
                final_image_size, aspect_ratio_rounding
            )
            image_metadata["aspect_ratio"] = aspect_ratio
            luminance_column = self.parquet_config.get("luminance_column", None)
            if luminance_column:
                image_metadata["luminance"] = database_image_metadata[
                    luminance_column
                ].values[0]
            else:
                image_metadata["luminance"] = 0
            logger.debug(
                f"Image {image_path_str} has aspect ratio {aspect_ratio} and size {final_image_size}."
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
