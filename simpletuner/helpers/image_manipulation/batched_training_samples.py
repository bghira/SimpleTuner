"""batch processing using trainingsample Rust library for better performance"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trainingsample as ts
from PIL import Image

from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class BatchedTrainingSamples:
    """batch processing using trainingsample for ~6.8x speedup over PIL"""

    def __init__(self):
        self.debug_enabled = logger.isEnabledFor(logging.DEBUG)

    def batch_resize_images(self, images: List[np.ndarray], target_sizes: List[Tuple[int, int]]) -> List[np.ndarray]:
        try:
            if not images or not target_sizes:
                return []

            # filter valid RGB arrays
            valid_images = []
            valid_sizes = []
            for i, (img, size) in enumerate(zip(images, target_sizes)):
                if isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3:
                    valid_images.append(img)
                    valid_sizes.append(size)
                else:
                    logger.warning(
                        f"Skipping invalid image at index {i}: shape={img.shape if hasattr(img, 'shape') else 'unknown'}"
                    )

            if not valid_images:
                return []

            resized = ts.batch_resize_images(valid_images, valid_sizes)
            if self.debug_enabled:
                logger.debug(f"Batch resized {len(resized)} images")
            return resized

        except Exception as e:
            logger.debug(f"Batch resize failed: {e}")
            return []

    def batch_center_crop_images(self, images: List[np.ndarray], target_sizes: List[Tuple[int, int]]) -> List[np.ndarray]:
        try:
            if not images or not target_sizes:
                return []

            cropped = ts.batch_center_crop_images(images, target_sizes)
            if self.debug_enabled:
                logger.debug(f"Batch center cropped {len(cropped)} images")
            return cropped

        except Exception as e:
            logger.debug(f"Batch center crop failed: {e}")
            return []

    def batch_random_crop_images(self, images: List[np.ndarray], target_sizes: List[Tuple[int, int]]) -> List[np.ndarray]:
        try:
            if not images or not target_sizes:
                return []

            cropped = ts.batch_random_crop_images(images, target_sizes)
            if self.debug_enabled:
                logger.debug(f"Batch random cropped {len(cropped)} images")
            return cropped

        except Exception as e:
            logger.debug(f"Batch random crop failed: {e}")
            return []

    def batch_calculate_luminance(self, images: List[np.ndarray]) -> List[float]:
        try:
            if not images:
                return []

            # only RGB arrays with shape (H, W, 3)
            valid_images = [
                img for img in images if isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3
            ]

            if not valid_images:
                return []

            luminances = ts.batch_calculate_luminance(valid_images)
            if self.debug_enabled:
                logger.debug(f"Calculated luminance for {len(luminances)} images")
            return luminances

        except Exception as e:
            logger.debug(f"Batch luminance calculation failed: {e}")
            return []

    def batch_resize_videos(self, videos: List[np.ndarray], target_sizes: List[Tuple[int, int]]) -> List[np.ndarray]:
        try:
            if not videos or not target_sizes:
                return []

            resized = ts.batch_resize_videos(videos, target_sizes)
            if self.debug_enabled:
                logger.debug(f"Batch resized {len(resized)} videos")
            return resized

        except Exception as e:
            logger.debug(f"Batch video resize failed: {e}")
            return []

    def process_aspect_grouped_images(
        self,
        grouped_data: Dict[str, List[Tuple[str, Any, str]]],
        metadata_backend=None,
    ) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
        processed_results = []

        for aspect_bucket, group_data in grouped_data.items():
            try:
                # PIL -> numpy for batch ops
                batch_images = []
                batch_filepaths = []
                batch_metadata = []

                for filepath, image, _ in group_data:
                    if isinstance(image, Image.Image):
                        img_array = np.array(image)
                        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                            batch_images.append(img_array)
                            batch_filepaths.append(filepath)
                            # fetch metadata
                            metadata = None
                            if metadata_backend:
                                try:
                                    metadata = metadata_backend.get_metadata_by_filepath(filepath)
                                except Exception:
                                    metadata = None
                            batch_metadata.append(metadata or {})
                        else:
                            logger.warning(f"Skipping image {filepath} with unexpected shape: {img_array.shape}")
                    elif isinstance(image, np.ndarray):
                        batch_images.append(image)
                        batch_filepaths.append(filepath)
                        metadata = None
                        if metadata_backend:
                            try:
                                metadata = metadata_backend.get_metadata_by_filepath(filepath)
                            except Exception:
                                metadata = None
                        batch_metadata.append(metadata or {})

                if not batch_images:
                    continue

                if len(batch_images) > 1:
                    try:
                        luminances = self.batch_calculate_luminance(batch_images)
                        if luminances and metadata_backend:
                            for i, (filepath, luminance) in enumerate(zip(batch_filepaths, luminances)):
                                # store luminance if missing
                                if batch_metadata[i] and "luminance" not in batch_metadata[i]:
                                    try:
                                        metadata_backend.update_metadata_attribute(filepath, "luminance", luminance)
                                        batch_metadata[i]["luminance"] = luminance
                                    except Exception as e:
                                        logger.debug(f"Failed to update luminance metadata for {filepath}: {e}")
                    except Exception as e:
                        logger.debug(f"Batch luminance calculation failed: {e}")

                # check resize requirements
                needs_resize = []
                target_sizes = []

                for i, (filepath, metadata) in enumerate(zip(batch_filepaths, batch_metadata)):
                    try:
                        # target size from metadata or fallback
                        if metadata and "target_size" in metadata:
                            target_size = metadata["target_size"]
                        else:
                            raise RuntimeError(
                                f"No target_size in metadata, cannot continue. Filename: {filepath}, Metadata: {metadata}"
                            )

                        current_shape = batch_images[i].shape[:2]  # (H, W)
                        current_size = (current_shape[1], current_shape[0])  # (W, H)

                        if current_size != target_size:
                            needs_resize.append(i)
                            target_sizes.append(target_size)
                    except Exception as e:
                        logger.error(f"Error checking resize for {filepath}: {e}", exc_info=True)

                        raise e

                if needs_resize and len(needs_resize) > 1:
                    try:
                        resize_images = [batch_images[i] for i in needs_resize]
                        resize_targets = target_sizes

                        resized_batch = self.batch_resize_images(resize_images, resize_targets)

                        # replace with resized versions
                        for idx, resized_img in zip(needs_resize, resized_batch):
                            batch_images[idx] = resized_img

                        if self.debug_enabled:
                            logger.debug(f"Batch resized {len(resized_batch)} images for aspect bucket {aspect_bucket}")
                    except Exception as e:
                        logger.debug(f"Batch resize failed, falling back to individual processing: {e}")

                # collect results
                for i, (filepath, image_array) in enumerate(zip(batch_filepaths, batch_images)):
                    processed_results.append((filepath, image_array, batch_metadata[i]))

            except Exception as e:
                logger.error(f"Error in batch processing for aspect {aspect_bucket}: {e}")
                # fallback without batch processing
                for filepath, image, _ in group_data:
                    if isinstance(image, Image.Image):
                        img_array = np.array(image)
                        processed_results.append((filepath, img_array, {}))
                    elif isinstance(image, np.ndarray):
                        processed_results.append((filepath, image, {}))

        return processed_results

    def convert_pil_to_numpy(self, images: List[Image.Image]) -> List[np.ndarray]:
        np_arrays = []
        for img in images:
            if isinstance(img, Image.Image):
                img_array = np.array(img)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    np_arrays.append(img_array)
                else:
                    logger.warning(f"Skipping image with unexpected shape: {img_array.shape}")
        return np_arrays

    def convert_numpy_to_pil(self, arrays: List[np.ndarray]) -> List[Image.Image]:
        pil_images = []
        for arr in arrays:
            if isinstance(arr, np.ndarray) and len(arr.shape) == 3:
                try:
                    pil_img = Image.fromarray(arr)
                    pil_images.append(pil_img)
                except Exception as e:
                    logger.warning(f"Failed to convert array to PIL: {e}")
        return pil_images
