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

        # Ensure data backend is HuggingfaceDatasetsBackend
        if not hasattr(data_backend, "dataset"):
            raise ValueError(
                "HuggingfaceMetadataBackend requires HuggingfaceDatasetsBackend"
            )

        self.dataset = data_backend.dataset
        self.caption_cache = self._extract_captions_to_dict()

    def _extract_captions_to_dict(self) -> Dict[str, Union[str, List[str]]]:
        """Extract captions from the dataset into a fast lookup dict."""
        captions = {}

        logger.info("Extracting captions from Hugging Face dataset...")

        # Iterate through dataset
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
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
                if col in item and item[col]:
                    caption_values.append(str(item[col]))
            if caption_values:
                caption = caption_values
        else:
            # Single caption column
            if self.caption_column in item:
                caption = item.get(self.caption_column)

        # Try fallback if no caption found
        if not caption and self.fallback_caption_column:
            if self.fallback_caption_column in item:
                caption = item.get(self.fallback_caption_column)

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

    def _get_image_metadata_from_item(self, item: Dict) -> Dict[str, Any]:
        """Extract image metadata from dataset item."""
        metadata = {}

        # Get image to extract dimensions
        image = item.get(self.data_backend.image_column)
        if image is not None:
            if isinstance(image, Image.Image):
                metadata["original_size"] = image.size
            elif isinstance(image, np.ndarray):
                h, w = image.shape[:2]
                metadata["original_size"] = (w, h)

        # Override with explicit width/height columns if available
        if self.width_column and self.height_column:
            if self.width_column in item and self.height_column in item:
                w = item[self.width_column]
                h = item[self.height_column]
                metadata["original_size"] = (int(w), int(h))

        return metadata

    def _process_for_bucket(
        self,
        image_path_str: str,
        aspect_ratio_bucket_indices: Dict,
        metadata_updates: Optional[Dict] = None,
        delete_problematic_images: bool = False,
        statistics: dict = {},
        aspect_ratio_rounding: int = 3,
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

            # Get image metadata
            image_metadata = self._get_image_metadata_from_item(item)

            if "original_size" not in image_metadata:
                logger.warning(f"Could not determine image size for {image_path_str}")
                statistics.setdefault("skipped", {}).setdefault("metadata_missing", 0)
                statistics["skipped"]["metadata_missing"] += 1
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

        total_items = len(self.dataset)
        statistics = {
            "total_processed": 0,
            "skipped": {
                "already_exists": 0,
                "quality": 0,
                "metadata_missing": 0,
                "not_found": 0,
                "too_small": 0,
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

        for idx in tqdm(
            range(total_items),
            desc="Processing HF dataset items",
            total=total_items,
            leave=False,
            ncols=100,
        ):
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
