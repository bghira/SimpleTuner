from helpers.training.state_tracker import StateTracker
from helpers.multiaspect.image import MultiaspectImage
from helpers.data_backend.base import BaseDataBackend
from helpers.metadata.backends.base import MetadataBackend
from tqdm import tqdm
import json, logging, os, time
from io import BytesIO
from PIL import Image

logger = logging.getLogger("ParquetMetadataBackend")
target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
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
            self.parquet_database.set_index(
                self.parquet_config.get("filename_column"), inplace=True
            )
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
        )
        if all_image_files is None:
            logger.debug("No image file cache available, retrieving fresh")
            all_image_files = self.data_backend.list_files(
                instance_data_root=self.instance_data_root,
                str_pattern="*.[jJpP][pPnN][gG]",
            )
            all_image_files = StateTracker.set_image_files(
                all_image_files, data_backend_id=self.data_backend.id
            )
        else:
            logger.debug("Using cached image file list")

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

    def save_cache(self, enforce_constraints: bool = False):
        """
        Save cache data to file.
        """
        # Prune any buckets that have fewer samples than batch_size
        if enforce_constraints:
            self._enforce_min_bucket_size()
        if self.read_only:
            logger.debug("Metadata backend is read-only, skipping cache save.")
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

    def compute_aspect_ratio_bucket_indices(self):
        """
        Compute the aspect ratio bucket indices without any threads or queues.

        Parquet backend behaves very differently to JSON backend.

        Returns:
            dict: The aspect ratio bucket indices.
        """
        logger.info("Discovering new files...")
        new_files = self._discover_new_files()

        existing_files_set = set().union(*self.aspect_ratio_bucket_indices.values())
        # Initialize aggregated statistics
        statistics = {
            "total_processed": 0,
            "skipped": {
                "already_exists": len(existing_files_set),
                "metadata_missing": 0,
                "not_found": 0,
                "too_small": 0,
                "other": 0,
            },
        }
        if not new_files:
            logger.debug("No new files discovered. Doing nothing.")
            return

        self.load_image_metadata()
        last_write_time = time.time()
        aspect_ratio_bucket_updates = {}
        for file in tqdm(
            new_files,
            desc="Generating aspect bucket cache",
            total=len(new_files),
            leave=False,
            ncols=100,
            miniters=int(len(new_files) / 100),
        ):
            current_time = time.time()
            if str(file) not in existing_files_set:
                logger.debug(f"Processing file {file}.")
                metadata_updates = {}
                aspect_ratio_bucket_updates = self._process_for_bucket(
                    file,
                    aspect_ratio_bucket_updates,
                    metadata_updates=metadata_updates,
                    delete_problematic_images=self.delete_problematic_images,
                    statistics=statistics,
                )
                statistics["total_processed"] += 1
                logger.debug(f"Statistics: {statistics}")
                logger.debug(f"Metadata updates: {metadata_updates}")
            else:
                statistics["skipped"]["already_exists"] += 1
                continue

            # Now, pull metadata updates from the queue
            if len(metadata_updates) > 0 and file in metadata_updates:
                metadata_update = metadata_updates[file]
                self.set_metadata_by_filepath(
                    filepath=file, metadata=metadata_updates[file], update_json=False
                )

                continue
            processing_duration = current_time - last_write_time
            if processing_duration >= self.metadata_update_interval:
                logger.debug(
                    f"In-flight metadata update after {processing_duration} seconds. Saving {len(self.image_metadata)} metadata entries and {len(self.aspect_ratio_bucket_indices)} aspect bucket lists."
                )
                self.save_cache(enforce_constraints=False)
                self.save_image_metadata()
                last_write_time = current_time

            time.sleep(0.001)

        for key, value in aspect_ratio_bucket_updates.items():
            self.aspect_ratio_bucket_indices.setdefault(key, []).extend(value)

        logger.debug(f"Bucket worker completed processing. Returning to main thread.")
        logger.info(f"Image processing statistics: {statistics}")
        self.save_image_metadata()
        self.save_cache(enforce_constraints=True)
        logger.info("Completed aspect bucket update.")

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
            try:
                database_image_metadata = self.parquet_database.loc[image_path_filtered]
            except KeyError:
                database_image_metadata = None
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
                    int(database_image_metadata[width_column]),
                    int(database_image_metadata[height_column]),
                )
            }
            if (
                image_metadata["original_size"][0] == 0
                or image_metadata["original_size"][1] == 0
            ):
                logger.debug(
                    f"Image {image_path_str} has a zero dimension. Skipping image."
                )
                return aspect_ratio_bucket_indices

            if not self.meets_resolution_requirements(image_metadata=image_metadata):
                logger.debug(
                    f"Image {image_path_str} does not meet minimum image size requirements. Skipping image."
                )
                statistics["skipped"]["too_small"] += 1
                return aspect_ratio_bucket_indices
            aspect_ratio_column = self.parquet_config.get("aspect_ratio_column", None)
            if aspect_ratio_column:
                aspect_ratio = database_image_metadata[aspect_ratio_column]
            else:
                aspect_ratio = (
                    image_metadata["original_size"][0]
                    / image_metadata["original_size"][1]
                )
            aspect_ratio = round(aspect_ratio, aspect_ratio_rounding)
            target_size, crop_coordinates, new_aspect_ratio = (
                MultiaspectImage.prepare_image(
                    image_metadata=image_metadata,
                    resolution=self.resolution,
                    resolution_type=self.resolution_type,
                    id=self.data_backend.id,
                )
            )
            image_metadata["crop_coordinates"] = crop_coordinates
            image_metadata["target_size"] = target_size
            image_metadata["aspect_ratio"] = new_aspect_ratio
            luminance_column = self.parquet_config.get("luminance_column", None)
            if luminance_column:
                image_metadata["luminance"] = database_image_metadata[
                    luminance_column
                ].values[0]
            else:
                image_metadata["luminance"] = 0
            logger.debug(
                f"Image {image_path_str} has aspect ratio {aspect_ratio} and size {image_metadata['target_size']}."
            )

            # Create a new bucket if it doesn't exist
            if str(new_aspect_ratio) not in aspect_ratio_bucket_indices:
                aspect_ratio_bucket_indices[str(new_aspect_ratio)] = []
            aspect_ratio_bucket_indices[str(new_aspect_ratio)].append(image_path_str)
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
