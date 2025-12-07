"""batch processing using trainingsample Rust library for better performance"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trainingsample as ts
from PIL import Image

from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger(__name__)
logger.setLevel(logging._nameToLevel.get(str(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")).upper(), logging.INFO))


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
                    valid_sizes.append(tuple(size))
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
            logger.error(f"Batch resize failed: {e}", exc_info=True)
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
            logger.error(f"Batch center crop failed: {e}", exc_info=True)
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
            logger.error(f"Batch random crop failed: {e}", exc_info=True)
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
            logger.error(f"Batch luminance calculation failed: {e}", exc_info=True)
            return []

    def batch_resize_videos(self, videos: List[np.ndarray], target_sizes: List[Tuple[int, int]]) -> List[np.ndarray]:
        try:
            if not videos or not target_sizes:
                return []

            valid_videos: List[np.ndarray] = []
            valid_sizes: List[Tuple[int, int]] = []
            for i, (video, size) in enumerate(zip(videos, target_sizes)):
                if not isinstance(video, np.ndarray):
                    logger.warning(f"Skipping invalid video at index {i}: type={type(video)}")
                    continue
                if len(video.shape) != 4 or video.shape[3] not in (1, 3, 4):
                    logger.warning(
                        f"Skipping invalid video at index {i}: shape={video.shape if hasattr(video, 'shape') else 'unknown'}"
                    )
                    continue
                if not (isinstance(size, (list, tuple)) and len(size) == 2):
                    logger.warning(f"Skipping video at index {i} due to invalid target_size={size}")
                    continue
                try:
                    valid_sizes.append((int(size[0]), int(size[1])))
                    valid_videos.append(video)
                except Exception as e:
                    logger.warning(f"Skipping video at index {i} due to size conversion error: {e}")

            if not valid_videos:
                return []

            resized = ts.batch_resize_videos(valid_videos, tuple(valid_sizes))
            if self.debug_enabled:
                logger.debug(f"Batch resized {len(resized)} videos")
            return resized

        except Exception as e:
            logger.error(f"Batch video resize failed: {e}", exc_info=True)
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
                                    if self.debug_enabled:
                                        logger.debug(
                                            f"Retrieved metadata for {filepath}: "
                                            f"{'found' if metadata else 'not found'}"
                                            f"{', keys: ' + str(list(metadata.keys())) if metadata else ''}"
                                        )
                                except Exception as e:
                                    logger.debug(f"Exception getting metadata for {filepath}: {e}")
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
                                if self.debug_enabled:
                                    logger.debug(
                                        f"Retrieved metadata for {filepath}: "
                                        f"{'found' if metadata else 'not found'}"
                                        f"{', keys: ' + str(list(metadata.keys())) if metadata else ''}"
                                    )
                            except Exception as e:
                                logger.debug(f"Exception getting metadata for {filepath}: {e}")
                                metadata = None
                        batch_metadata.append(metadata or {})

                if not batch_images:
                    continue

                if len(batch_images) > 1:
                    try:
                        luminances = self.batch_calculate_luminance(batch_images)
                        can_update_metadata = hasattr(metadata_backend, "update_metadata_attribute")
                        if luminances and metadata_backend:
                            for i, (filepath, luminance) in enumerate(zip(batch_filepaths, luminances)):
                                # store luminance if missing
                                if batch_metadata[i] and "luminance" not in batch_metadata[i]:
                                    if can_update_metadata:
                                        try:
                                            metadata_backend.update_metadata_attribute(filepath, "luminance", luminance)
                                            batch_metadata[i]["luminance"] = luminance
                                        except Exception as e:
                                            logger.debug(f"Failed to update luminance metadata for {filepath}: {e}")
                                    else:
                                        batch_metadata[i]["luminance"] = luminance
                    except Exception as e:
                        logger.error(f"Batch luminance calculation failed: {e}", exc_info=True)

                # check resize requirements
                image_resize_indices: List[int] = []
                image_resize_targets: List[Tuple[int, int]] = []
                video_resize_indices: List[int] = []
                video_resize_targets: List[Tuple[int, int]] = []

                for i, (filepath, metadata) in enumerate(zip(batch_filepaths, batch_metadata)):
                    try:
                        if self.debug_enabled:
                            logger.debug(
                                f"Checking metadata for resize: filepath={filepath}, "
                                f"metadata_present={'yes' if metadata else 'no'}, "
                                f"metadata_keys={list(metadata.keys()) if metadata else 'N/A'}"
                            )
                        # target size from metadata or fallback
                        if metadata and "target_size" in metadata:
                            target_value = metadata["target_size"]
                            if isinstance(target_value, (list, tuple)) and len(target_value) == 2:
                                target_size = (int(target_value[0]), int(target_value[1]))
                            elif isinstance(target_value, dict) and {"width", "height"} <= set(target_value.keys()):
                                target_size = (int(target_value["width"]), int(target_value["height"]))
                            else:
                                raise RuntimeError(f"Unsupported target_size format for {filepath}: {target_value}")
                            # keep normalised tuple in metadata copy
                            metadata["target_size"] = target_size
                        else:
                            logger.error(
                                f"No target_size in metadata, cannot continue. "
                                f"Filename: {filepath}, "
                                f"Metadata keys: {list(metadata.keys()) if metadata else 'empty/None'}, "
                                f"Metadata: {metadata}"
                            )
                            raise RuntimeError(
                                f"No target_size in metadata, cannot continue. Filename: {filepath}, Metadata: {metadata}"
                            )

                        arr = batch_images[i]
                        if not isinstance(arr, np.ndarray):
                            logger.warning(f"Skipping resize check for {filepath}: unsupported type {type(arr)}")
                            continue

                        if arr.ndim == 4:  # (frames, H, W, C)
                            current_height, current_width = arr.shape[1], arr.shape[2]
                            current_size = (current_width, current_height)
                            if current_size != target_size:
                                video_resize_indices.append(i)
                                video_resize_targets.append(target_size)
                        elif arr.ndim == 3:  # (H, W, C)
                            current_height, current_width = arr.shape[0], arr.shape[1]
                            current_size = (current_width, current_height)
                            if current_size != target_size:
                                image_resize_indices.append(i)
                                image_resize_targets.append(target_size)
                        else:
                            logger.warning(f"Skipping resize for {filepath}: unexpected array shape {arr.shape}")
                    except Exception as e:
                        logger.error(f"Error checking resize for {filepath}: {e}", exc_info=True)

                        raise e

                if len(image_resize_indices) > 1:
                    try:
                        resize_images = [batch_images[i] for i in image_resize_indices]

                        resized_batch = self.batch_resize_images(resize_images, tuple(image_resize_targets))

                        # replace with resized versions
                        for idx, resized_img in zip(image_resize_indices, resized_batch):
                            batch_images[idx] = resized_img

                        if self.debug_enabled:
                            logger.debug(f"Batch resized {len(resized_batch)} images for aspect bucket {aspect_bucket}")
                    except Exception as e:
                        logger.debug(f"Batch resize failed, falling back to individual processing: {e}")

                if video_resize_indices:
                    try:
                        resize_videos = [batch_images[i] for i in video_resize_indices]
                        resized_videos = self.batch_resize_videos(resize_videos, video_resize_targets)

                        for idx, resized_video in zip(video_resize_indices, resized_videos):
                            batch_images[idx] = resized_video

                        if self.debug_enabled and resized_videos:
                            logger.debug(f"Batch resized {len(resized_videos)} videos for aspect bucket {aspect_bucket}")
                    except Exception as e:
                        logger.debug(f"Batch video resize failed, falling back to individual processing: {e}")

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
