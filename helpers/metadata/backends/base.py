from math import ceil
import os
import time
import logging
import threading
import torch
from helpers.data_backend.base import BaseDataBackend
from helpers.multiaspect.image import MultiaspectImage
from helpers.training.state_tracker import StateTracker
from helpers.training.multi_process import should_log
from multiprocessing import Process, Queue
from threading import Thread
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from math import floor
import numpy as np

# For semaphore
from threading import Semaphore

logger = logging.getLogger("BaseMetadataBackend")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class MetadataBackend:
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
        self.id = id
        if self.id != data_backend.id:
            raise ValueError(
                f"BucketManager ID ({self.id}) must match the DataBackend ID ({data_backend.id})."
            )
        self.accelerator = accelerator
        self.should_abort = False
        self.data_backend = data_backend
        self.batch_size = int(batch_size)
        self.repeats = int(repeats)
        self.instance_data_dir = instance_data_dir
        if cache_file_suffix is not None:
            cache_file = f"{cache_file}_{cache_file_suffix}"
            metadata_file = f"{metadata_file}_{cache_file_suffix}"
        self.cache_file = Path(f"{cache_file}.json")
        self.metadata_file = Path(f"{metadata_file}.json")
        self.aspect_ratio_bucket_indices = {}
        self.image_metadata = {}  # Store image metadata
        self.seen_images = {}
        self.config = {}
        self.reload_cache()
        self.resolution = float(resolution)
        self.resolution_type = resolution_type
        self.delete_problematic_images = delete_problematic_images
        self.delete_unwanted_images = delete_unwanted_images
        self.metadata_update_interval = metadata_update_interval
        self.minimum_image_size = (
            float(minimum_image_size) if minimum_image_size else None
        )
        self.minimum_aspect_ratio = (
            float(minimum_aspect_ratio) if minimum_aspect_ratio else None
        )
        self.maximum_aspect_ratio = (
            float(maximum_aspect_ratio) if maximum_aspect_ratio else None
        )
        self.maximum_num_frames = (
            float(maximum_num_frames) if maximum_num_frames else None
        )
        self.minimum_num_frames = (
            float(minimum_num_frames) if minimum_num_frames else None
        )
        self.num_frames = float(num_frames) if num_frames else None
        self.image_metadata_loaded = False
        self.vae_output_scaling_factor = 8
        self.metadata_semaphor = Semaphore()
        # When a multi-gpu system splits the buckets, we no longer update.
        self.read_only = False

    def load_metadata(self):
        raise NotImplementedError

    def save_metadata(self):
        raise NotImplementedError

    def _bucket_worker(
        self,
        tqdm_queue,
        files,
        aspect_ratio_bucket_indices_queue,
        metadata_updates_queue,
        written_files_queue,
        existing_files_set,
    ):
        """
        A worker function to bucket a list of files.

        Args:
            tqdm_queue (Queue): A queue to report progress to.
            files (list): A list of files to bucket.
            aspect_ratio_bucket_indices_queue (Queue): A queue to report the bucket indices to.
            existing_files_set (set): A set of existing files.

        Returns:
            dict: The bucket indices.
        """
        local_aspect_ratio_bucket_indices = {}
        local_metadata_updates = {}
        processed_file_list = set()
        processed_file_count = 0
        # Initialize statistics dictionary
        statistics = {
            "total_processed": 0,
            "skipped": {
                "already_exists": 0,
                "metadata_missing": 0,
                "not_found": 0,
                "too_small": 0,
                "other": 0,  # Add more specific reasons as needed
            },
        }

        for file in files:
            if str(file) not in existing_files_set:
                logger.debug(f"Processing file {file}.")
                try:
                    local_aspect_ratio_bucket_indices = self._process_for_bucket(
                        file,
                        local_aspect_ratio_bucket_indices,
                        metadata_updates=local_metadata_updates,
                        delete_problematic_images=self.delete_problematic_images,
                        statistics=statistics,
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing file {file}. Reason: {e}. Skipping."
                    )
                    statistics["skipped"]["error"] += 1
                logger.debug(
                    f"Statistics: {statistics}, total: {sum([len(bucket) for bucket in local_aspect_ratio_bucket_indices.values()])}"
                )
                processed_file_count += 1
                # Successfully processed
                statistics["total_processed"] = processed_file_count
                processed_file_list.add(file)
            else:
                statistics["skipped"]["already_exists"] += 1
            tqdm_queue.put(1)
            if processed_file_count % 500 == 0:
                # Send updates to queues and reset the local dictionaries
                if aspect_ratio_bucket_indices_queue is not None:
                    aspect_ratio_bucket_indices_queue.put(
                        local_aspect_ratio_bucket_indices
                    )
                if written_files_queue is not None:
                    written_files_queue.put(processed_file_list)
                metadata_updates_queue.put(local_metadata_updates)
                local_aspect_ratio_bucket_indices = {}
                local_metadata_updates = {}
                processed_file_list = set()
        if (
            aspect_ratio_bucket_indices_queue is not None
            and local_aspect_ratio_bucket_indices
        ):
            aspect_ratio_bucket_indices_queue.put(local_aspect_ratio_bucket_indices)
        if local_metadata_updates:
            metadata_updates_queue.put(local_metadata_updates)
            # At the end of the _bucket_worker method
            metadata_updates_queue.put(("statistics", statistics))
        time.sleep(0.001)
        logger.debug("Bucket worker completed processing. Returning to main thread.")

    def compute_aspect_ratio_bucket_indices(self, ignore_existing_cache: bool = False):
        """
        Compute the aspect ratio bucket indices. The workhorse of this class.

        Arguments:
            ignore_existing_cache (bool): Whether to ignore the existing cache
            and entirely recompute the aspect ratio bucket indices.

        Returns:
            dict: The aspect ratio bucket indices.
        """
        logger.info("Discovering new files...")
        new_files = self._discover_new_files(
            ignore_existing_cache=ignore_existing_cache
        )

        existing_files_set = set().union(*self.aspect_ratio_bucket_indices.values())
        logger.info(
            f"Compressed {len(existing_files_set)} existing files from {len(self.aspect_ratio_bucket_indices.values())}."
        )
        # Initialize aggregated statistics
        aggregated_statistics = {
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
            logger.info("No new files discovered. Doing nothing.")
            logger.info(f"Statistics: {aggregated_statistics}")
            return
        num_cpus = (
            StateTracker.get_args().aspect_bucket_worker_count
        )  # Using a fixed number for better control and predictability
        files_split = np.array_split(new_files, num_cpus)

        metadata_updates_queue = Queue()
        written_files_queue = Queue()
        tqdm_queue = Queue()
        aspect_ratio_bucket_indices_queue = Queue()
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
        worker_cls = (
            Process if StateTracker.get_args().enable_multiprocessing else Thread
        )
        workers = [
            worker_cls(
                target=self._bucket_worker,
                args=(
                    tqdm_queue,
                    file_shard,
                    aspect_ratio_bucket_indices_queue,
                    metadata_updates_queue,
                    written_files_queue,
                    existing_files_set,
                ),
            )
            for file_shard in files_split
        ]

        for worker in workers:
            worker.start()
        last_write_time = time.time()
        written_files = set()
        with tqdm(
            desc="Generating aspect bucket cache",
            total=len(new_files),
            leave=False,
            ncols=125,
            miniters=int(len(new_files) / 100),
        ) as pbar:
            if self.should_abort:
                logger.info("Aborting aspect bucket update.")
                return
            while (
                any(worker.is_alive() for worker in workers)
                or not tqdm_queue.empty()
                or not aspect_ratio_bucket_indices_queue.empty()
                or not metadata_updates_queue.empty()
                or not written_files_queue.empty()
            ):
                current_time = time.time()
                while not tqdm_queue.empty():
                    pbar.update(tqdm_queue.get())
                while not aspect_ratio_bucket_indices_queue.empty():
                    aspect_ratio_bucket_indices_update = (
                        aspect_ratio_bucket_indices_queue.get()
                    )
                    for key, value in aspect_ratio_bucket_indices_update.items():
                        self.aspect_ratio_bucket_indices.setdefault(key, []).extend(
                            value
                        )
                # Now, pull metadata updates from the queue
                while not metadata_updates_queue.empty():
                    metadata_update = metadata_updates_queue.get()
                    if (
                        type(metadata_update) is tuple
                        and metadata_update[0] == "statistics"
                    ):
                        logger.debug(
                            f"Received statistics update: {metadata_update[1]}"
                        )
                        for reason, count in metadata_update[1]["skipped"].items():
                            aggregated_statistics["skipped"][reason] += count
                        aggregated_statistics["total_processed"] += metadata_update[1][
                            "total_processed"
                        ]
                        continue
                    for filepath, meta in metadata_update.items():
                        self.set_metadata_by_filepath(
                            filepath=filepath, metadata=meta, update_json=False
                        )
                # Process the written files queue
                while not written_files_queue.empty():
                    written_files_batch = written_files_queue.get()
                    written_files.update(written_files_batch)  # Use update for sets

                processing_duration = current_time - last_write_time
                if processing_duration >= self.metadata_update_interval:
                    logger.debug(
                        f"In-flight metadata update after {processing_duration} seconds. Saving {len(self.image_metadata)} metadata entries and {len(self.aspect_ratio_bucket_indices)} aspect bucket lists."
                    )
                    self.save_cache(enforce_constraints=False)
                    self.save_image_metadata()
                    last_write_time = current_time

                time.sleep(0.001)

        for worker in workers:
            worker.join()
        logger.info(f"Image processing statistics: {aggregated_statistics}")
        self.save_image_metadata()
        self.save_cache(enforce_constraints=True)
        logger.info("Completed aspect bucket update.")

    def split_buckets_between_processes(self, gradient_accumulation_steps=1):
        """
        Splits the contents of each bucket in aspect_ratio_bucket_indices between the available processes.
        """
        new_aspect_ratio_bucket_indices = {}
        total_images = sum(
            [len(bucket) for bucket in self.aspect_ratio_bucket_indices.values()]
        )
        logger.debug(f"Count of items before split: {total_images}")

        # Determine the effective batch size for all processes considering gradient accumulation
        num_processes = self.accelerator.num_processes
        effective_batch_size = (
            self.batch_size * num_processes * gradient_accumulation_steps
        )

        for bucket, images in self.aspect_ratio_bucket_indices.items():
            # Trim the list to a length that's divisible by the effective batch size
            total_img_count_incl_repeats = len(images) * (self.repeats + 1)
            num_batches = ceil(total_img_count_incl_repeats / effective_batch_size)
            trimmed_images = images[: num_batches * effective_batch_size]
            if len(trimmed_images) == 0 and should_log():
                logger.error(
                    f"Bucket {bucket} has no images after trimming because {len(images)} images are not enough to satisfy an effective batch size of {effective_batch_size}."
                    " Lower your batch size, increase repeat count, or increase data pool size."
                )

            with self.accelerator.split_between_processes(
                trimmed_images, apply_padding=False
            ) as images_split:
                # Now images_split contains only the part of the images list that this process should handle
                new_aspect_ratio_bucket_indices[bucket] = images_split

        # Replace the original aspect_ratio_bucket_indices with the new one containing only this process's share
        self.aspect_ratio_bucket_indices = new_aspect_ratio_bucket_indices
        post_total = sum(
            [len(bucket) for bucket in self.aspect_ratio_bucket_indices.values()]
        )
        if total_images != post_total:
            self.read_only = True

        logger.debug(f"Count of items after split: {post_total}")

    def mark_as_seen(self, image_path):
        """Mark an image as seen."""
        self.seen_images[image_path] = True

    def mark_batch_as_seen(self, image_paths):
        """Efficiently extend the Manager with new contents, image_paths

        Args:
            image_paths (list): A list of image paths to mark as seen.
        """
        self.seen_images.update({image_path: True for image_path in image_paths})

    def is_seen(self, image_path):
        """Check if an image is seen."""
        return self.seen_images.get(image_path, False)

    def reset_seen_images(self):
        """Reset the seen images."""
        self.seen_images.clear()

    def remove_image(self, image_path, bucket: str = None):
        """
        Used by other classes to reliably remove images from a bucket.

        Args:
            image_path (str): The path to the image to remove.
            bucket (str): The bucket to remove the image from.

        Returns:
            dict: The aspect ratio bucket indices.
        """
        if not bucket:
            for bucket, images in self.aspect_ratio_bucket_indices.items():
                if image_path in images:
                    self.aspect_ratio_bucket_indices[bucket].remove(image_path)
                    break
        if image_path in self.aspect_ratio_bucket_indices[bucket]:
            self.aspect_ratio_bucket_indices[bucket].remove(image_path)

    def update_buckets_with_existing_files(self, existing_files: set):
        """
        Update bucket indices to remove entries that no longer exist and remove duplicates.

        Args:
            existing_files (set): A set of existing files.
        """
        logger.debug(
            f"Before updating, in all buckets, we had {sum([len(bucket) for bucket in self.aspect_ratio_bucket_indices.values()])}."
        )
        for bucket, images in self.aspect_ratio_bucket_indices.items():
            # Remove non-existing files and duplicates while preserving order
            filtered_images = list(
                dict.fromkeys(img for img in images if img in existing_files)
            )
            self.aspect_ratio_bucket_indices[bucket] = filtered_images
        logger.debug(
            f"After updating, in all buckets, we had {sum([len(bucket) for bucket in self.aspect_ratio_bucket_indices.values()])}."
        )
        # Save the updated cache
        self.save_cache()

    def refresh_buckets(self, rank: int = None):
        """
        Discover new files and remove images that no longer exist.
        """
        # Discover new files and update bucket indices
        self.compute_aspect_ratio_bucket_indices()

        # Get the list of existing files
        logger.debug(
            f"Refreshing buckets for rank {rank} via data_backend id {self.id}."
        )
        existing_files = StateTracker.get_image_files(data_backend_id=self.id)

        if not StateTracker.get_args().ignore_missing_files:
            # Update bucket indices to remove entries that no longer exist
            self.update_buckets_with_existing_files(existing_files)
        return

    def _enforce_min_bucket_size(self):
        """
        Remove buckets that have fewer samples than batch_size and enforce minimum image size constraints.
        """
        if self.minimum_image_size is None:
            return

        logger.info(
            f"Enforcing minimum image size of {self.minimum_image_size}."
            " This could take a while for very-large datasets."
        )
        for bucket in tqdm(
            list(self.aspect_ratio_bucket_indices.keys()),
            leave=False,
            desc="Enforcing minimum bucket size",
        ):  # Safe iteration over keys
            # Prune the smaller buckets so that we don't enforce resolution constraints on them unnecessarily.
            self._prune_small_buckets(bucket)
            if self.minimum_image_size is not None:
                self._enforce_resolution_constraints(bucket)
                # We do this twice in case there were any new contenders for being too small.
                self._prune_small_buckets(bucket)

    def _enforce_min_aspect_ratio(self):
        """
        Remove buckets that have an aspect ratio outside the specified range.
        """
        if self.minimum_aspect_ratio is None or self.minimum_aspect_ratio == 0.0:
            return

        logger.info(
            f"Enforcing minimum aspect ratio of {self.minimum_aspect_ratio}."
            " This could take a while for very-large datasets."
        )
        for bucket in tqdm(
            list(self.aspect_ratio_bucket_indices.keys()),
            leave=False,
            desc="Enforcing minimum aspect ratio",
        ):  # Safe iteration over keys
            if float(bucket) < self.minimum_aspect_ratio:
                logger.info(
                    f"Removing bucket {bucket} due to aspect ratio being less than {self.minimum_aspect_ratio}."
                )
                del self.aspect_ratio_bucket_indices[bucket]

    def _enforce_max_aspect_ratio(self):
        """
        Remove buckets that have an aspect ratio outside the specified range.
        """
        if self.maximum_aspect_ratio is None or self.maximum_aspect_ratio == 0.0:
            return

        logger.info(
            f"Enforcing maximum aspect ratio of {self.maximum_aspect_ratio}."
            " This could take a while for very-large datasets."
        )
        for bucket in tqdm(
            list(self.aspect_ratio_bucket_indices.keys()),
            leave=False,
            desc="Enforcing maximum aspect ratio",
        ):  # Safe iteration over keys
            if float(bucket) > self.maximum_aspect_ratio:
                logger.info(
                    f"Removing bucket {bucket} due to aspect ratio being greater than {self.maximum_aspect_ratio}."
                )
                del self.aspect_ratio_bucket_indices[bucket]

    def _prune_small_buckets(self, bucket):
        """
        Remove buckets with fewer images than the batch size.
        """
        if StateTracker.get_args().disable_bucket_pruning:
            logger.warning(
                "Not pruning small buckets, as --disable_bucket_pruning is provided."
            )
            return
        if (
            bucket in self.aspect_ratio_bucket_indices
            and (
                len(self.aspect_ratio_bucket_indices[bucket]) * (int(self.repeats) + 1)
            )
            < self.batch_size
        ):
            bucket_sample_count = len(self.aspect_ratio_bucket_indices[bucket])
            del self.aspect_ratio_bucket_indices[bucket]
            logger.warning(
                f"Removing bucket {bucket} due to insufficient samples; your batch size may be too large for the small quantity of data (batch_size={self.batch_size} > sample_count={bucket_sample_count})."
            )

    def _enforce_resolution_constraints(self, bucket):
        """
        Enforce resolution constraints on images in a bucket.
        """
        if self.minimum_image_size is not None:
            if bucket not in self.aspect_ratio_bucket_indices:
                logger.debug(
                    f"Bucket {bucket} was already removed due to insufficient samples."
                )
                return
            images = self.aspect_ratio_bucket_indices[bucket]
            total_before = len(images)
            self.aspect_ratio_bucket_indices[bucket] = [
                img
                for img in images
                if self.meets_resolution_requirements(
                    image_path=img,
                    image=None,
                )
            ]
            total_after = len(self.aspect_ratio_bucket_indices[bucket])
            total_lost = total_before - total_after
            if total_lost > 0:
                logger.info(
                    f"Had {total_before} samples before and {total_lost} that did not meet the minimum image size requirement ({self.minimum_image_size})."
                )

    def meets_resolution_requirements(
        self,
        image_path: str = None,
        image: Image = None,
        image_metadata: dict = None,
    ):
        """
        Check if an image meets the resolution requirements.
        """
        if image is None and (image_path is not None and image_metadata is None):
            metadata = self.get_metadata_by_filepath(image_path)
            if metadata is None:
                logger.warning(f"Metadata not found for image {image_path}.")
                return False
            width, height = metadata["original_size"]
        elif isinstance(image, np.ndarray):
            # we have a video
            width, height = image.shape[2], image.shape[1]
            logger.debug(f"Checking resolution: {width}x{height}")
            if self.minimum_num_frames is not None:
                num_frames = image.shape[0]
                if num_frames < self.minimum_num_frames:
                    logger.debug(
                        f"Video has {num_frames} frames, which is less than the minimum required {self.minimum_num_frames}."
                    )
                    return False
            if self.maximum_num_frames is not None:
                num_frames = image.shape[0]
                if num_frames > self.maximum_num_frames:
                    logger.debug(
                        f"Video has {num_frames} frames, which is more than the maximum configured {self.maximum_num_frames}."
                    )
                    return False
        elif image is not None:
            width, height = image.size
        elif image_metadata is not None:
            width, height = image_metadata["original_size"]
        else:
            # Unexpected condition
            raise ValueError(
                f"meets_resolution_requirements expects an image_path"
                f" ({image_path}) or Image object ({image}), but received neither."
            )

        if self.minimum_image_size is None:
            return True

        if self.resolution_type == "pixel":
            return (
                self.minimum_image_size <= width and self.minimum_image_size <= height
            )
        elif self.resolution_type == "area":
            # We receive megapixel integer value, and then have to compare here by converting minimum_image_size MP to pixels.
            if self.minimum_image_size > 5:
                raise ValueError(
                    f"--minimum_image_size was given with a value of {self.minimum_image_size} but resolution_type is area, which means this value is most likely too large. Please use a value less than 5."
                )
            # We need to find the square image length if crop_style = square.
            minimum_image_size = self.minimum_image_size * 1_000_000
            if (
                StateTracker.get_data_backend_config(self.id).get("crop", False)
                and StateTracker.get_data_backend_config(self.id).get(
                    "crop_aspect", "square"
                )
                == "square"
            ):
                # When comparing the 'area' of an image but cropping to square area, one side might be too small.
                # So we have to convert our megapixel value to a 1.0 aspect square image size.
                # We do this by taking the square root of the megapixel value.
                pixel_edge_len = floor(np.sqrt(minimum_image_size))
                if not (pixel_edge_len <= width and pixel_edge_len <= height):
                    # If the square edge length is too small, then the image is too small.
                    return False
            # Since we've now tested whether a square-cropped image will be adequate, we can calculate the area of the image.
            return minimum_image_size <= width * height
        else:
            raise ValueError(
                f"BucketManager.meets_resolution_requirements received unexpected value for resolution_type: {self.resolution_type}"
            )

    def handle_incorrect_bucket(
        self, image_path: str, bucket: str, actual_bucket: str, save_cache: bool = True
    ):
        """
        Used by other classes to move images between buckets, when mis-detected.

        Args:
            image_path (str): The path to the image to move.
            bucket (str): The bucket to move the image from.
            actual_bucket (str): The bucket to move the image to.
        """
        logger.warning(
            f"Found an image in bucket {bucket} it doesn't belong in, when actually it is: {actual_bucket}"
        )
        self.remove_image(image_path, bucket)
        if actual_bucket in self.aspect_ratio_bucket_indices:
            logger.warning("Moved image to bucket, it already existed.")
            self.aspect_ratio_bucket_indices[actual_bucket].append(image_path)
        else:
            logger.warning("Created new bucket for that pesky image.")
            self.aspect_ratio_bucket_indices[actual_bucket] = [image_path]
        if save_cache:
            self.save_cache()

    def handle_small_image(
        self, image_path: str, bucket: str, delete_unwanted_images: bool
    ):
        """
        Used by other classes to remove an image, or DELETE it from disk, depending on parameters.

        Args:
            image_path (str): The path to the image to remove.
            bucket (str): The bucket to remove the image from.
            delete_unwanted_images (bool): Whether to delete the image from disk.
        """
        if delete_unwanted_images:
            try:
                logger.warning(
                    f"Image {image_path} too small: DELETING image and continuing search."
                )
                self.data_backend.delete(image_path)
            except Exception:
                logger.debug(
                    f"Image {image_path} was already deleted. Another GPU must have gotten to it."
                )
        else:
            logger.warning(
                f"Image {image_path} too small, but --delete_unwanted_images is not provided, so we simply ignore and remove from bucket."
            )
        self.remove_image(image_path, bucket)

    def has_single_underfilled_bucket(self):
        """
        Check if there's only one active bucket and it has fewer images than the batch size.

        Returns:
            bool: True if there's a single underfilled bucket, False otherwise.
        """
        if len(self.aspect_ratio_bucket_indices) != 1:
            return False

        bucket = list(self.aspect_ratio_bucket_indices.keys())[0]
        if (
            len(self.aspect_ratio_bucket_indices[bucket]) * (int(self.repeats) + 1)
        ) < self.batch_size:
            return True

        return False

    def read_cache(self):
        """
        Read the entire bucket cache.
        """
        return self.aspect_ratio_bucket_indices

    def get_metadata_attribute_by_filepath(self, filepath: str, attribute: str):
        """Use get_metadata_by_filepath to return a specific attribute.

        Args:
            filepath (str): The complete path from the aspect bucket list.
            attribute (str): The attribute you are seeking.

        Returns:
            any type: The attribute value, or None.
        """
        metadata = self.get_metadata_by_filepath(filepath)
        if metadata:
            return metadata.get(attribute, None)
        else:
            return None

    def set_metadata_attribute_by_filepath(
        self, filepath: str, attribute: str, value: any, update_json: bool = True
    ):
        """Use set_metadata_by_filepath to update the contents of a specific attribute.

        Args:
            filepath (str): The complete path from the aspect bucket list.
            attribute (str): The attribute you are updating.
            value (any type): The value to set.
        """
        metadata = self.get_metadata_by_filepath(filepath) or {}
        metadata[attribute] = value
        return self.set_metadata_by_filepath(filepath, metadata, update_json)

    def set_metadata_by_filepath(
        self, filepath: str, metadata: dict, update_json: bool = True
    ):
        """Set metadata for a given image file path.

        Args:
            filepath (str): The complete path from the aspect bucket list.
        """
        with self.metadata_semaphor:
            logger.debug(f"Setting metadata for {filepath} to {metadata}.")
            self.image_metadata[filepath] = metadata
            if update_json:
                self.save_image_metadata()

    def get_metadata_by_filepath(self, filepath: str):
        """Retrieve metadata for a given image file path.

        Args:
            filepath (str): The complete or basename path from the aspect bucket list.
                            First, we search for the basename as the key, and we fall
                             back to the

        Returns:
            dict: Metadata for the image. Returns None if not found.
        """
        if type(filepath) is tuple or type(filepath) is list:
            for path in filepath:
                if path in self.image_metadata:
                    result = self.image_metadata.get(path, None)
                    logger.debug(
                        f"Retrieving metadata for path: {filepath}, result: {result}"
                    )
                    if result is not None:
                        return result
            return None

        return self.image_metadata.get(filepath, None)

    def scan_for_metadata(self):
        """
        Update the metadata without modifying the bucket indices.
        """
        logger.info(f"Loading metadata from {self.metadata_file}")
        self.load_image_metadata()
        logger.debug(
            f"A subset of the available metadata: {list(self.image_metadata.keys())[:5]}"
        )
        logger.info("Discovering new images for metadata scan...")
        new_files = self._discover_new_files(for_metadata=True)
        if not new_files:
            logger.info("No new files discovered. Exiting.")
            return

        existing_files_set = {
            existing_file for existing_file in self.image_metadata.keys()
        }

        num_cpus = 8  # Using a fixed number for better control and predictability
        files_split = np.array_split(new_files, num_cpus)

        metadata_updates_queue = Queue()
        tqdm_queue = Queue()
        worker_cls = (
            Process if StateTracker.get_args().enable_multiprocessing else Thread
        )
        workers = [
            worker_cls(
                target=self._bucket_worker,
                args=(
                    tqdm_queue,
                    file_shard,
                    None,  # Passing None to indicate we don't want to update the buckets
                    metadata_updates_queue,
                    None,  # Passing None to indicate we don't want to update the written files list
                    existing_files_set,
                ),
            )
            for file_shard in files_split
        ]

        for worker in workers:
            worker.start()

        with tqdm(
            desc="Scanning image metadata",
            total=len(new_files),
            leave=False,
            ncols=125,
        ) as pbar:
            while any(worker.is_alive() for worker in workers):
                while not tqdm_queue.empty():
                    pbar.update(tqdm_queue.get())

                # Only update the metadata
                while not metadata_updates_queue.empty():
                    metadata_update = metadata_updates_queue.get()
                    logger.debug(
                        f"Received type of metadata update: {type(metadata_update)}, contents: {metadata_update}"
                    )
                    if type(metadata_update) == dict:
                        for filepath, meta in metadata_update.items():
                            self.set_metadata_by_filepath(
                                filepath=filepath, metadata=meta, update_json=False
                            )

        for worker in workers:
            worker.join()

        self.save_image_metadata()
        self.save_cache(enforce_constraints=True)
        logger.info("Completed metadata update.")

    def handle_vae_cache_inconsistencies(self, vae_cache, vae_cache_behavior: str):
        """
        Handles inconsistencies between the aspect buckets and the VAE cache.

        Args:
            vae_cache: The VAECache object.
            vae_cache_behavior (str): Behavior for handling inconsistencies ('sync' or 'recreate').
        """
        if "deepfloyd" in StateTracker.get_args().model_type:
            return
        if vae_cache_behavior not in ["sync", "recreate"]:
            raise ValueError("Invalid VAE cache behavior specified.")
        logger.info("Scanning VAE cache for inconsistencies with aspect buckets...")
        try:
            for cache_file, cache_content in vae_cache.scan_cache_contents():
                if cache_content is None:
                    continue
                if vae_cache_behavior == "sync":
                    # Sync aspect buckets with the cache
                    expected_bucket = str(
                        self._get_aspect_ratio_from_tensor(cache_content)
                    )
                    self._modify_cache_entry_bucket(cache_file, expected_bucket)
                elif vae_cache_behavior == "recreate":
                    # Delete the cache file if it doesn't match the aspect bucket indices
                    if self.is_cache_inconsistent(vae_cache, cache_file, cache_content):
                        threading.Thread(
                            target=self.data_backend.delete,
                            args=(cache_file,),
                            daemon=True,
                        ).start()
        except Exception as e:
            logger.debug(f"Error running VAE cache scan: {e}")
            return

        # Update any state or metadata post-processing
        self.save_cache()

    def _recalculate_target_resolution(self, original_aspect_ratio: float) -> tuple:
        """Given the original resolution, use our backend config to properly recalculate the size."""
        resolution_type = StateTracker.get_data_backend_config(self.id)[
            "resolution_type"
        ]
        resolution = StateTracker.get_data_backend_config(self.id)["resolution"]
        if resolution_type == "pixel":
            return MultiaspectImage.calculate_new_size_by_pixel_edge(
                original_aspect_ratio, int(resolution)
            )
        elif resolution_type == "area":
            if original_aspect_ratio is None:
                raise ValueError(
                    "Original aspect ratio must be provided for area-based resolution."
                )
            return MultiaspectImage.calculate_new_size_by_pixel_area(
                original_aspect_ratio, resolution
            )

    def is_cache_inconsistent(self, vae_cache, cache_file, cache_content):
        """
        Check if a cache file's content is inconsistent with the aspect ratio bucket indices.

        Args:
            cache_file (str): The cache file path.
            cache_content: The content of the cache file (PyTorch Tensor).

        Returns:
            bool: True if the cache file is inconsistent, False otherwise.
        """
        # Get tensor shape and multiply by self.scaling_factor or 8
        if cache_content is None:
            return True
        # is it a tensor with nan or inf values?
        if torch.isnan(cache_content).any() or torch.isinf(cache_content).any():
            logger.warning(f"Cache file {cache_file} contains NaN or Inf values.")
            return True
        image_filename = vae_cache._image_filename_from_vaecache_filename(cache_file)
        logger.debug(
            f"Checking cache file {cache_file} for inconsistencies. Image filename: {image_filename}"
        )
        actual_resolution = self._get_image_size_from_tensor(cache_content)
        original_resolution = self.get_metadata_attribute_by_filepath(
            image_filename, "original_size"
        )
        metadata_target_size = self.get_metadata_attribute_by_filepath(
            image_filename, "target_size"
        )
        if metadata_target_size is None:
            logger.error(
                f"Received sample with no metadata: {self.get_metadata_by_filepath(image_filename)}"
            )
            return True
        target_resolution = tuple(metadata_target_size)
        recalculated_target_resolution, intermediary_size, recalculated_aspect_ratio = (
            self._recalculate_target_resolution(
                original_aspect_ratio=MultiaspectImage.calculate_image_aspect_ratio(
                    original_resolution
                )
            )
        )
        logger.debug(
            f"Original resolution: {original_resolution}, Target resolution: {target_resolution}, Recalculated target resolution: {recalculated_target_resolution}"
        )
        if (
            original_resolution is not None
            and target_resolution is not None
            and (
                actual_resolution != target_resolution
                or actual_resolution != recalculated_target_resolution
            )
        ):
            logger.debug(
                f"Actual resolution {actual_resolution} does not match target resolution {target_resolution}, recalculated as {recalculated_target_resolution}."
            )
            return True
        else:
            logger.debug(
                f"Actual resolution {actual_resolution} matches target resolution {target_resolution}."
            )

        actual_aspect_ratio = self._get_aspect_ratio_from_tensor(cache_content)
        expected_bucket = str(recalculated_aspect_ratio)
        logger.debug(
            f"Expected bucket for {cache_file}: {expected_bucket} vs actual {actual_aspect_ratio}"
        )

        # Extract the base filename without the extension
        base_filename = os.path.splitext(os.path.basename(cache_file))[0]
        base_filename_png = os.path.join(self.instance_data_dir, f"{base_filename}.png")
        base_filename_jpg = os.path.join(self.instance_data_dir, f"{base_filename}.jpg")
        # Check if the base filename is in the correct bucket
        if any(
            base_filename_png in self.aspect_ratio_bucket_indices.get(bucket, set())
            for bucket in [expected_bucket, str(expected_bucket)]
        ):
            logger.debug(f"File {base_filename} is in the correct bucket.")
            return False
        if any(
            base_filename_jpg in self.aspect_ratio_bucket_indices.get(bucket, set())
            for bucket in [expected_bucket, str(expected_bucket)]
        ):
            logger.debug(f"File {base_filename} is in the correct bucket.")
            return False
        logger.debug(f"File {base_filename} was not found in the correct place.")
        return True

    def _get_aspect_ratio_from_tensor(self, tensor):
        """
        Calculate the aspect ratio from a PyTorch Tensor.

        Args:
            tensor (torch.Tensor): The tensor representing the image.

        Returns:
            float: The aspect ratio of the image.
        """
        if tensor.dim() < 3:
            raise ValueError(
                "Tensor does not have enough dimensions to determine aspect ratio."
            )
        # Assuming tensor is in CHW format (channel, height, width)
        _, height, width = tensor.size()
        return width / height

    def _get_image_size_from_tensor(self, tensor):
        """
        Calculate the image size from a PyTorch Tensor.

        Args:
            tensor (torch.Tensor): The tensor representing the image.

        Returns:
            tuple[width, height]: The resolution of the image just before it was encoded.
        """
        if tensor.dim() < 3:
            raise ValueError(
                f"Tensor does not have enough dimensions to determine an image resolution. Its shape is: {tensor.size}"
            )
        # Assuming tensor is in CHW format (channel, height, width)
        _, height, width = tensor.size()
        return (
            width * self.vae_output_scaling_factor,
            height * self.vae_output_scaling_factor,
        )

    def _modify_cache_entry_bucket(self, cache_file, expected_bucket):
        """
        Update the bucket indices based on the cache file's actual aspect ratio.

        Args:
            cache_file (str): The cache file path.
            expected_bucket (str): The bucket that the cache file should belong to.
        """
        for bucket, files in self.aspect_ratio_bucket_indices.items():
            if cache_file in files and str(bucket) != str(expected_bucket):
                files.remove(cache_file)
                self.aspect_ratio_bucket_indices[expected_bucket].append(cache_file)
                break
