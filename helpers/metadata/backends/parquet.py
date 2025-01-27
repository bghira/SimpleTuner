from helpers.training.state_tracker import StateTracker
from helpers.training import image_file_extensions
from helpers.multiaspect.image import MultiaspectImage
from helpers.data_backend.base import BaseDataBackend
from helpers.image_manipulation.training_sample import TrainingSample
from helpers.metadata.backends.base import MetadataBackend
from tqdm import tqdm
import json
import logging
import os
import time
import traceback
import numpy

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
        cache_file_suffix: str = None,
        repeats: int = 0,
    ):
        self.parquet_config = parquet_config
        self.parquet_path = parquet_config.get("path", None)
        self.is_json_lines = self.parquet_path.endswith(".jsonl")
        self.is_json_file = self.parquet_path.endswith(".json")
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
            cache_file_suffix=cache_file_suffix,
            repeats=repeats,
        )
        self.load_parquet_database()
        self.caption_cache = self._extract_captions_to_fast_list()
        self.missing_captions = self._locate_missing_caption_from_fast_list()
        if self.missing_captions:
            logger.warning(
                f"Missing captions for {len(self.missing_captions)} images: {self.missing_captions}"
            )
            self._remove_images_with_missing_captions()

    def _remove_images_with_missing_captions(self):
        """
        Remove images from the aspect ratio bucket indices that have missing captions.
        """
        for key in self.aspect_ratio_bucket_indices.keys():
            len_before = len(self.aspect_ratio_bucket_indices[key])
            self.aspect_ratio_bucket_indices[key] = [
                path
                for path in self.aspect_ratio_bucket_indices[key]
                if path not in self.missing_captions
            ]
            len_after = len(self.aspect_ratio_bucket_indices[key])
            if len_before != len_after:
                logger.warning(
                    f"Removed {len_before - len_after} images from aspect ratio bucket {key} due to missing captions."
                )
        self.save_cache(enforce_constraints=True)
        self.missing_captions = []

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
            if self.is_json_lines or self.is_json_file:
                self.parquet_database = pd.read_json(pq, lines=self.is_json_lines)
            else:
                self.parquet_database = pd.read_parquet(pq, engine="pyarrow")
            self.parquet_database.set_index(
                self.parquet_config.get("filename_column"), inplace=True
            )
        else:
            raise FileNotFoundError(
                f"Parquet could not be loaded from {self.parquet_path}: database file does not exist (path={self.parquet_path})."
            )

    def _locate_missing_caption_from_fast_list(self):
        """
        Check the fast list keys vs the filenames in our aspect ratio bucket indices.
        """
        missing_captions = []
        identifier_includes_extension = self.parquet_config.get(
            "identifier_includes_extension", False
        )
        # currently we just don't do this.
        identifier_includes_path = False
        for key in self.aspect_ratio_bucket_indices.keys():
            for filename in self.aspect_ratio_bucket_indices[key]:
                if not identifier_includes_extension:
                    filename = os.path.splitext(filename)[0]
                if not identifier_includes_path:
                    # strip out self.instance_data_dir
                    filename = filename.replace(self.instance_data_dir, "")
                    # any leading /
                    if filename.startswith("/"):
                        filename = filename[1:]
                if filename not in self.caption_cache:
                    missing_captions.append(filename)
        return missing_captions

    def _extract_captions_to_fast_list(self):
        """
        Pull the captions from the parquet table into a dict with the format {filename: caption}.

        This helps because parquet's columnar format sucks for searching.

        Returns:
            dict: A dictionary of captions.
        """
        if self.parquet_database is None:
            raise ValueError("Parquet database is not loaded.")
        filename_column = self.parquet_config.get("filename_column")
        caption_column = self.parquet_config.get("caption_column")
        fallback_caption_column = self.parquet_config.get("fallback_caption_column")
        identifier_includes_extension = self.parquet_config.get(
            "identifier_includes_extension", False
        )
        captions = {}
        for index, row in self.parquet_database.iterrows():
            if filename_column in row:
                filename = str(row[filename_column])
            else:
                filename = str(index)
            if not identifier_includes_extension:
                filename = os.path.splitext(filename)[0]

            if type(caption_column) == list:
                caption = None
                if len(caption_column) > 0:
                    caption = [row[c] for c in caption_column]
            else:
                caption = row.get(caption_column)
                if isinstance(caption, (numpy.ndarray, pd.Series)):
                    caption = [str(item) for item in caption if item is not None]

            if caption is None and fallback_caption_column:
                caption = row.get(fallback_caption_column, None)
            if caption is None or caption == "" or caption == []:
                raise ValueError(
                    f"Could not locate caption for image {filename} in sampler_backend {self.id} with filename column {filename_column}, caption column {caption_column}, and a parquet database with {len(self.parquet_database)} entries."
                )
            if type(caption) == bytes:
                caption = caption.decode("utf-8")
            elif type(caption) == list:
                caption = [c.strip() for c in caption if c.strip()]
            elif type(caption) == str:
                caption = caption.strip()
            captions[filename] = caption
        return captions

    def caption_cache_entry(self, index: str):
        result = self.caption_cache.get(str(index), None)

        logger.debug(f"Caption cache entry for idx {str(index)}: {result}")
        return result

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
        if ignore_existing_cache:
            # Return all files and remove the existing buckets.
            logger.debug(
                "Resetting the entire aspect bucket cache as we've received the signal to ignore existing cache."
            )
            self.aspect_ratio_bucket_indices = {}
            return list(all_image_files.keys())
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
        elif ignore_existing_cache:
            # Remove existing aspect bucket indices and return all image files.
            result = all_image_files
            self.aspect_ratio_bucket_indices = {}
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
            if set_config:
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
        self._enforce_min_aspect_ratio()
        self._enforce_max_aspect_ratio()

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

    def compute_aspect_ratio_bucket_indices(self, ignore_existing_cache: bool = False):
        """
        Compute the aspect ratio bucket indices without any threads or queues.

        Parquet backend behaves very differently to JSON backend.

        Returns:
            dict: The aspect ratio bucket indices.
        """
        logger.info("Discovering new files...")
        new_files = self._discover_new_files(
            ignore_existing_cache=ignore_existing_cache
        )

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

        try:
            self.load_image_metadata()
        except Exception as e:
            if ignore_existing_cache:
                logger.warning(
                    f"Error loading image metadata, creating new metadata cache: {e}"
                )
                self.image_metadata = {}
            else:
                raise Exception(
                    f"Error loading image metadata. You may have to remove the metadata json file '{self.metadata_file}' and VAE cache manually: {e}"
                )
        last_write_time = time.time()
        aspect_ratio_bucket_updates = {}
        # log a truncated set of the parquet table
        logger.debug(f"Parquet table head: {self.parquet_database.head().to_string()}")
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
                if self.should_abort:
                    logger.info("Aborting aspect bucket update.")
                    return
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

        for key, value in aspect_ratio_bucket_updates.items():
            self.aspect_ratio_bucket_indices.setdefault(key, []).extend(value)

        logger.debug("Bucket worker completed processing. Returning to main thread.")
        logger.info(f"Image processing statistics: {statistics}")
        self.save_image_metadata()
        self.save_cache(enforce_constraints=True)
        logger.info("Completed aspect bucket update.")

    def _get_first_value(self, series_or_scalar):
        """Extract the first value if the input is a Series, else return the value itself."""
        if isinstance(series_or_scalar, pd.Series):
            return int(series_or_scalar.iloc[0])
        elif isinstance(series_or_scalar, str):
            # Convert to int if the input is a string representing a number
            return int(series_or_scalar)
        elif isinstance(series_or_scalar, (int, float)):
            return series_or_scalar
        elif isinstance(series_or_scalar, numpy.int64):
            new_type = int(series_or_scalar)
            if type(new_type) != int:
                raise ValueError(f"Unsupported data type: {type(series_or_scalar)}.")
            return new_type
        else:
            raise ValueError(f"Unsupported data type: {type(series_or_scalar)}.")

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
            # Adjust image path if the identifier does not include extension
            image_path_filtered = image_path_str
            if not self.parquet_config.get("identifier_includes_extension", False):
                image_path_filtered = os.path.splitext(
                    os.path.split(image_path_str)[-1]
                )[0]
            if self.instance_data_dir in image_path_filtered:
                image_path_filtered = image_path_filtered.replace(
                    self.instance_data_dir, ""
                )
                # remove leading /
                if image_path_filtered.startswith("/"):
                    image_path_filtered = image_path_filtered[1:]
            if image_path_filtered.isdigit():
                image_path_filtered = int(image_path_filtered)

            logger.debug(
                f"Reading image {image_path_str} metadata from parquet backend column {self.parquet_config.get('filename_column')} without instance root dir prefix {self.instance_data_dir}: {image_path_filtered}."
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
                statistics.setdefault("skipped", {}).setdefault("metadata_missing", 0)
                statistics["skipped"]["metadata_missing"] += 1
                return aspect_ratio_bucket_indices

            width_column = self.parquet_config.get("width_column", "width")
            height_column = self.parquet_config.get("height_column", "height")
            if width_column is None or height_column is None:
                raise ValueError(
                    "ParquetMetadataBackend requires width and height columns to be defined."
                )
            w = self._get_first_value(database_image_metadata[width_column])
            h = self._get_first_value(database_image_metadata[height_column])
            logger.debug(
                f"Image {image_path_str} has dimensions {w}x{h} types {type(w)}."
            )
            original_size = (w, h)
            if (
                original_size[0] < StateTracker.get_args().aspect_bucket_alignment
                or original_size[1] < StateTracker.get_args().aspect_bucket_alignment
            ):
                logger.debug(
                    f"Image {image_path_str} is smaller than the aspect bucket index. Skipping image."
                )
                return aspect_ratio_bucket_indices

            training_sample = TrainingSample(
                image=None,
                data_backend_id=self.id,
                image_metadata={"original_size": original_size},
                image_path=image_path_str,
            )
            prepared_sample = training_sample.prepare()
            image_metadata = {"original_size": training_sample.original_size}

            logger.debug("Prepared sample: %s", str(prepared_sample))

            logger.debug("Checking minimum resolution size vs image size...")
            if not self.meets_resolution_requirements(image_metadata=image_metadata):
                if not self.delete_unwanted_images:
                    logger.debug(
                        f"Image {image_path_str} does not meet minimum image size requirements. Skipping image."
                    )
                else:
                    logger.debug(
                        f"Image {image_path_str} does not meet minimum image size requirements. Deleting image."
                    )
                    try:
                        self.data_backend.delete(image_path_str)
                    except:
                        pass
                statistics.setdefault("skipped", {}).setdefault("too_small", 0)
                statistics["skipped"]["too_small"] += 1

                return aspect_ratio_bucket_indices

            logger.debug("Collecting aspect ratio data...")
            aspect_ratio_column = self.parquet_config.get("aspect_ratio_column")
            aspect_ratio = (
                database_image_metadata[aspect_ratio_column]
                if aspect_ratio_column
                else training_sample.aspect_ratio
            )
            aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(
                float(aspect_ratio)
            )

            logger.debug("Image metadata has been generated and collected.")
            image_metadata.update(
                {
                    "intermediary_size": prepared_sample.intermediary_size,
                    "crop_coordinates": prepared_sample.crop_coordinates,
                    "target_size": prepared_sample.target_size,
                    "aspect_ratio": float(prepared_sample.aspect_ratio),
                    "luminance": int(
                        database_image_metadata.get(
                            self.parquet_config.get("luminance_column"), 0
                        )
                    ),
                }
            )
            # logger.debug(
            #     f"Data types for metadata: {[type(v) for v in image_metadata.values()]}"
            # )
            # print the types of any iterable values
            # for key, value in image_metadata.items():
            # if hasattr(value, "__iter__"):
            # logger.debug(f"Key {key} has type {type(value)}: {value}")
            # for v in value:
            # logger.debug(f"Value has type {type(v)}: {v}")

            # logger.debug(
            #     f"Image {image_path_str} has aspect ratio {prepared_sample.aspect_ratio}, intermediary size {image_metadata['intermediary_size']}, target size {image_metadata['target_size']}."
            # )

            # Create a new bucket if it doesn't exist
            aspect_ratio_key = str(prepared_sample.aspect_ratio)
            if aspect_ratio_key not in aspect_ratio_bucket_indices:
                aspect_ratio_bucket_indices[aspect_ratio_key] = []
            logger.debug("Adding to list...")
            aspect_ratio_bucket_indices[aspect_ratio_key].append(image_path_str)
            logger.debug("Added to list.")

            # Instead of directly updating, just fill the provided dictionary
            if metadata_updates is not None:
                logger.debug("Adding to metadata list...")
                metadata_updates[image_path_str] = image_metadata
                logger.debug("Added to metadata list.")

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
