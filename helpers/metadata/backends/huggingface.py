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
from typing import Optional, Dict, Any, List, Union
from PIL import Image
import numpy as np

logger = logging.getLogger("HuggingfaceMetadataBackend")
target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
logger.setLevel(target_level)


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
    ):
        """
        Initialize the Hugging Face metadata backend.

        Args:
            hf_config: Configuration dict containing:
                - caption_column: Column name(s) for captions
                - fallback_caption_column: Fallback if caption_column is empty
                - width_column: Column for image width (optional)
                - height_column: Column for image height (optional)
                - quality_column: Column containing quality assessments
                - description_column: Column for descriptions
                - composite_image_config: Configuration for handling composite images
            split_composite_images: Whether to split composite images
            composite_image_column: Column containing composite images
            quality_filter: Dict of quality thresholds to filter by
        """
        self.hf_config = hf_config
        self.split_composite_images = split_composite_images
        self.composite_image_column = composite_image_column
        self.quality_filter = quality_filter

        # Get column names from config
        self.caption_column = hf_config.get("caption_column", "caption")
        self.fallback_caption_column = hf_config.get("fallback_caption_column", None)
        self.width_column = hf_config.get("width_column", None)
        self.height_column = hf_config.get("height_column", None)
        self.quality_column = hf_config.get("quality_column", "quality_assessment")
        self.description_column = hf_config.get("description_column", "description")
        # Video-specific columns
        self.num_frames_column = hf_config.get("num_frames_column", None)
        self.fps_column = hf_config.get("fps_column", None)

        # Composite image configuration
        self.composite_config = hf_config.get("composite_image_config", {})

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

        # Ensure data backend is HuggingfaceDatasetsBackend
        if not hasattr(data_backend, "dataset"):
            raise ValueError(
                "HuggingfaceMetadataBackend requires HuggingfaceDatasetsBackend"
            )

        self.caption_cache = self._extract_captions_to_dict()

    def _extract_captions_to_dict(self) -> Dict[str, Union[str, List[str]]]:
        """Extract captions from the dataset into a fast lookup dict."""
        captions = {}

        logger.info("Extracting captions from Hugging Face dataset...")

        # Check composite configuration
        is_composite = self.composite_config.get("enabled", False)
        select_index = self.composite_config.get("select_index", None)

        # Iterate through dataset
        for idx in range(len(self.data_backend.dataset)):
            item = self.data_backend.dataset[idx]

            # Generate appropriate virtual path
            virtual_path = f"{idx}.jpg"

            # Apply quality filter if specified
            if self.quality_filter and self.quality_column in item:
                quality = item[self.quality_column]
                if not self._passes_quality_filter(quality):
                    continue

            # Extract caption
            caption = self._extract_caption_from_item(item)
            if caption:
                captions[virtual_path] = caption

        logger.info(f"Extracted {len(captions)} captions from dataset")
        return captions

    def _extract_caption_from_item(self, item: Dict) -> Optional[Union[str, List[str]]]:
        """Extract caption from a dataset item."""
        caption = None

        # Handle list of caption columns
        if isinstance(self.caption_column, list):
            caption_values = []
            for col in self.caption_column:
                value = self._get_nested_value(item, col)
                if value:
                    caption_values.append(str(value))
            if caption_values:
                caption = caption_values
        else:
            # Single caption column (might be nested)
            caption = self._get_nested_value(item, self.caption_column)
            if caption:
                caption = str(caption)

        # Try fallback if no caption found
        if not caption and self.fallback_caption_column:
            fallback = self._get_nested_value(item, self.fallback_caption_column)
            if fallback:
                caption = str(fallback)

        # Handle description column if specified
        if not caption and self.description_column in item:
            desc = item.get(self.description_column)
            if isinstance(desc, dict):
                # Extract relevant parts from description dict
                caption_parts = []
                for key in ["main_subject", "description", "text"]:
                    if key in desc and desc[key]:
                        caption_parts.append(str(desc[key]))
                if caption_parts:
                    caption = " ".join(caption_parts)
            else:
                caption = str(desc)

        # Clean up caption
        if caption:
            if isinstance(caption, (list, tuple)):
                caption = [str(c).strip() for c in caption if c]
            elif isinstance(caption, str):
                caption = caption.strip()

        return caption if caption else None

    def _get_nested_value(self, item: Dict, key_path: str) -> Any:
        """Get a value from a nested dictionary using dot notation."""
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
        """
        Load cache data from a JSON file on the data_backend.
        """
        if self.data_backend.exists(self.cache_file):
            try:
                cache_data_raw = self.data_backend.read(self.cache_file)
                cache_data = json.loads(cache_data_raw)
                logger.debug("Loaded existing aspect ratio cache.")
            except Exception as e:
                logger.warning(
                    f"Error loading aspect ratio bucket cache, creating new one: {e}"
                )
                cache_data = {}
            self.aspect_ratio_bucket_indices = cache_data.get(
                "aspect_ratio_bucket_indices", {}
            )
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
        """
        Save cache data as JSON to the data_backend.
        """
        if enforce_constraints:
            self._enforce_min_bucket_size()
        self._enforce_min_aspect_ratio()
        self._enforce_max_aspect_ratio()

        if self.read_only:
            logger.debug("Metadata backend is read-only. Skipping save.")
            return

        aspect_ratio_bucket_indices_str = {
            key: [str(path) for path in value]
            for key, value in self.aspect_ratio_bucket_indices.items()
        }
        cache_data = {
            "config": StateTracker.get_data_backend_config(
                data_backend_id=self.data_backend.id
            ),
            "aspect_ratio_bucket_indices": aspect_ratio_bucket_indices_str,
        }
        cache_data_str = json.dumps(cache_data)
        self.data_backend.write(self.cache_file, cache_data_str)
        logger.debug("Aspect ratio cache saved.")

    def save_image_metadata(self):
        """Save metadata to file."""
        # Make sure we're using the full path
        full_metadata_path = self.metadata_file
        if not str(self.metadata_file).endswith(".json"):
            full_metadata_path = f"{self.metadata_file}_{self.id}.json"

        self.data_backend.write(full_metadata_path, json.dumps(self.image_metadata))
        logger.debug(f"Metadata file saved to {full_metadata_path}")

    def load_image_metadata(self):
        """Load metadata from file."""
        # Use the same full path as save
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
                    logger.info(
                        f"Loaded {len(self.image_metadata)} metadata entries from {full_metadata_path}"
                    )
                else:
                    logger.warning(
                        f"Metadata file exists but is empty: {full_metadata_path}"
                    )
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        else:
            logger.debug(f"Metadata file does not exist: {full_metadata_path}")

    def _passes_quality_filter(self, quality_assessment: Dict) -> bool:
        """Check if an item passes the quality filter."""
        if not self.quality_filter or not quality_assessment:
            return True

        for key, min_value in self.quality_filter.items():
            if quality_assessment.get(key, 0) < min_value:
                return False
        return True

    def caption_cache_entry(self, index: str) -> Optional[Union[str, List[str]]]:
        """Get caption for a virtual path."""
        return self.caption_cache.get(str(index), None)

    def _discover_new_files(
        self, for_metadata: bool = False, ignore_existing_cache: bool = False
    ):
        """
        Discover new files that have not been processed yet.
        For HuggingFace datasets, this returns virtual paths.
        """
        # For HF datasets, we process all items
        if hasattr(self.data_backend, "streaming") and self.data_backend.streaming:
            logger.warning("Cannot discover files in streaming mode")
            return []

        total_items = len(self.data_backend.dataset)
        # Check composite configuration
        is_composite = self.composite_config.get("enabled", False)
        select_index = self.composite_config.get("select_index", None)

        # Generate virtual paths based on configuration
        all_files = []
        for idx in range(total_items):
            virtual_path = f"{idx}.jpg"
            all_files.append(virtual_path)

        if ignore_existing_cache:
            logger.debug("Ignoring existing cache, returning all files")
            return all_files

        # Get already processed files
        processed_files = set()
        for paths in self.aspect_ratio_bucket_indices.values():
            processed_files.update(paths)

        # Return only unprocessed files
        new_files = [f for f in all_files if f not in processed_files]
        logger.debug(f"Found {len(new_files)} new files out of {len(all_files)} total")

        return new_files

    def _get_image_metadata_from_item(self, item: Dict) -> Dict[str, Any]:
        """Extract image metadata from dataset item."""
        metadata = {}

        # Get image to extract dimensions
        image = item.get(self.data_backend.image_column)
        if image is not None:
            # Check if we need to handle composite image
            is_composite = self.composite_config.get("enabled", False)
            image_count = self.composite_config.get("image_count", 1)
            select_index = self.composite_config.get("select_index", None)

            if isinstance(image, Image.Image):
                width, height = image.size

                # Handle composite image dimensions
                if is_composite and image_count > 1:
                    # Calculate dimensions for the selected segment
                    segment_width = width // image_count
                    metadata["original_size"] = (segment_width, height)
                else:
                    metadata["original_size"] = (width, height)

            elif isinstance(image, np.ndarray):
                h, w = image.shape[:2]

                # Handle composite image dimensions
                if is_composite and image_count > 1:
                    segment_width = w // image_count
                    metadata["original_size"] = (segment_width, h)
                else:
                    metadata["original_size"] = (w, h)

        # Override with explicit width/height columns if available
        if self.width_column and self.height_column:
            if self.width_column in item and self.height_column in item:
                w = item[self.width_column]
                h = item[self.height_column]

                # Handle composite dimensions if needed
                is_composite = self.composite_config.get("enabled", False)
                image_count = self.composite_config.get("image_count", 1)
                if is_composite and image_count > 1:
                    w = int(w) // image_count

                metadata["original_size"] = (int(w), int(h))

        # Extract video metadata if available
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
        """Process an image for aspect ratio bucketing."""
        try:
            # Get index from virtual path
            index = self.data_backend._get_index_from_path(image_path_str)
            if index is None:
                logger.warning(f"Could not get index for path: {image_path_str}")
                statistics.setdefault("skipped", {}).setdefault("not_found", 0)
                statistics["skipped"]["not_found"] += 1
                return aspect_ratio_bucket_indices

            # Get the dataset item
            item = self.data_backend.get_dataset_item(index)
            if item is None:
                logger.warning(f"Could not get dataset item for index: {index}")
                statistics.setdefault("skipped", {}).setdefault("not_found", 0)
                statistics["skipped"]["not_found"] += 1
                return aspect_ratio_bucket_indices

            # Apply quality filter
            if self.quality_filter and self.quality_column in item:
                if not self._passes_quality_filter(item[self.quality_column]):
                    logger.debug(f"Item {image_path_str} failed quality filter")
                    statistics.setdefault("skipped", {}).setdefault("quality", 0)
                    statistics["skipped"]["quality"] += 1
                    return aspect_ratio_bucket_indices

            # Get image metadata (will handle composite images)
            image_metadata = self._get_image_metadata_from_item(item)

            if "original_size" not in image_metadata:
                logger.warning(f"Could not determine image size for {image_path_str}")
                statistics.setdefault("skipped", {}).setdefault("metadata_missing", 0)
                statistics["skipped"]["metadata_missing"] += 1
                return aspect_ratio_bucket_indices

            # Check video frame constraints if this is a video dataset
            if "num_frames" in image_metadata:
                num_frames = image_metadata["num_frames"]

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
                    statistics.setdefault("skipped", {}).setdefault(
                        "too_many_frames", 0
                    )
                    statistics["skipped"]["too_many_frames"] += 1
                    return aspect_ratio_bucket_indices

            # Check resolution requirements
            if not self.meets_resolution_requirements(image_metadata=image_metadata):
                if self.delete_unwanted_images:
                    logger.debug(
                        f"{image_path_str} does not meet resolution requirements."
                    )
                statistics.setdefault("skipped", {}).setdefault("too_small", 0)
                statistics["skipped"]["too_small"] += 1
                return aspect_ratio_bucket_indices

            # Create training sample
            training_sample = TrainingSample(
                image=None,  # We don't load actual image data here
                data_backend_id=self.id,
                image_metadata=image_metadata,
                image_path=image_path_str,
            )
            prepared_sample = training_sample.prepare()

            # Calculate aspect ratio
            aspect_ratio = float(prepared_sample.aspect_ratio)

            # Update metadata
            image_metadata.update(
                {
                    "aspect_ratio": aspect_ratio,
                    "intermediary_size": prepared_sample.intermediary_size,
                    "crop_coordinates": prepared_sample.crop_coordinates,
                    "target_size": prepared_sample.target_size,
                }
            )

            # Add to bucket
            aspect_ratio_key = str(round(aspect_ratio, aspect_ratio_rounding))
            if aspect_ratio_key not in aspect_ratio_bucket_indices:
                aspect_ratio_bucket_indices[aspect_ratio_key] = []
            aspect_ratio_bucket_indices[aspect_ratio_key].append(image_path_str)

            # Store metadata updates
            if metadata_updates is not None:
                metadata_updates[image_path_str] = image_metadata

        except Exception as e:
            logger.error(f"Error processing file {image_path_str}: {e}")
            logger.error(traceback.format_exc())
            statistics.setdefault("skipped", {}).setdefault("error", 0)
            statistics["skipped"]["error"] += 1

        return aspect_ratio_bucket_indices

    def compute_aspect_ratio_bucket_indices(self, ignore_existing_cache: bool = False):
        """Build aspect ratio buckets from the HF dataset."""
        logger.info("Building aspect ratio buckets from Hugging Face dataset...")

        # For HF datasets, we process all items
        if hasattr(self.data_backend, "streaming") and self.data_backend.streaming:
            logger.warning("Cannot build aspect ratio buckets in streaming mode")
            return

        # Check if we have composite image configuration
        is_composite = self.composite_config.get("enabled", False)
        select_index = self.composite_config.get("select_index", None)
        # Use the filtered dataset length, otherwise we'll try accessing elements we've filtered
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

        # Load existing cache unless we're ignoring it
        if not ignore_existing_cache:
            self.reload_cache()
            existing_files = set().union(*self.aspect_ratio_bucket_indices.values())
            statistics["skipped"]["already_exists"] = len(existing_files)
        else:
            self.aspect_ratio_bucket_indices = {}
            existing_files = set()

        # Process each item
        last_save_time = time.time()
        aspect_ratio_bucket_updates = {}
        metadata_updates = {}

        total_items = len(self.data_backend.dataset)
        for idx in tqdm(
            range(total_items),
            desc="Processing HF dataset items",
            total=total_items,
            leave=False,
            ncols=100,
        ):
            # Generate virtual path based on composite configuration
            virtual_path = f"{idx}.jpg"

            # Skip if already processed
            if virtual_path in existing_files:
                continue

            # Process the item
            aspect_ratio_bucket_updates = self._process_for_bucket(
                virtual_path,
                aspect_ratio_bucket_updates,
                metadata_updates=metadata_updates,
                delete_problematic_images=self.delete_problematic_images,
                statistics=statistics,
            )
            statistics["total_processed"] += 1

            # Periodic save
            current_time = time.time()
            if (current_time - last_save_time) >= self.metadata_update_interval:
                logger.debug(f"Saving intermediate results...")
                # Merge updates
                for key, value in aspect_ratio_bucket_updates.items():
                    self.aspect_ratio_bucket_indices.setdefault(key, []).extend(value)
                aspect_ratio_bucket_updates = {}

                # Save metadata
                for path, metadata in metadata_updates.items():
                    self.set_metadata_by_filepath(path, metadata, update_json=False)
                metadata_updates = {}

                self.save_cache(enforce_constraints=False)
                self.save_image_metadata()
                last_save_time = current_time

        # Final merge and save
        for key, value in aspect_ratio_bucket_updates.items():
            self.aspect_ratio_bucket_indices.setdefault(key, []).extend(value)

        for path, metadata in metadata_updates.items():
            self.set_metadata_by_filepath(path, metadata, update_json=False)

        logger.info(f"Processing complete. Statistics: {statistics}")
        self.save_image_metadata()
        self.save_cache(enforce_constraints=True)

    def __len__(self):
        """Count how many full batches we can form."""

        def repeat_len(bucket):
            return len(bucket) * (self.repeats + 1)

        return sum(
            (repeat_len(bucket) + (self.batch_size - 1)) // self.batch_size
            for bucket in self.aspect_ratio_bucket_indices.values()
            if repeat_len(bucket) >= self.batch_size
        )

    def remove_images(self, image_paths: List[str]):
        """Remove images from all aspect ratio buckets."""
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        for image_path in image_paths:
            # Remove from caption cache
            if image_path in self.caption_cache:
                del self.caption_cache[image_path]

            # Remove from all buckets
            for bucket, images in self.aspect_ratio_bucket_indices.items():
                if image_path in images:
                    self.aspect_ratio_bucket_indices[bucket].remove(image_path)
                    logger.debug(f"Removed {image_path} from bucket {bucket}")

    def refresh_buckets(self, rank: int = None):
        """
        Override refresh_buckets for HuggingFace datasets.
        Since HF datasets use virtual paths, we don't need to check for missing files.
        """
        logger.debug(f"Refreshing buckets for HuggingFace backend {self.id}")
        # Just run the bucket computation
        self.compute_aspect_ratio_bucket_indices()
        return

    def update_buckets_with_existing_files(self, existing_files: set):
        """
        Override for HuggingFace - all virtual files always "exist".
        """
        # For HF datasets, we don't need to remove non-existing files
        # since all files are virtual references to dataset indices
        logger.debug(f"Skipping file existence check for HuggingFace backend {self.id}")
        return
