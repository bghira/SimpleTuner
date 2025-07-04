import os
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import shuffle
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from numpy import str_ as numpy_str
from queue import Queue
from hashlib import sha256
from typing import List, Dict, Tuple, Any, Optional

logger = logging.getLogger("DataGenerator")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class DataGenerator:
    """
    A data generator that reads from source backend, applies transformations via SampleGenerator,
    and writes to target backend. Mirrors VAECache's efficient multi-GPU batch processing.

    Output files preserve the same extension as input files (e.g., .png -> .png, .jpg -> .jpg).
    """

    def __init__(
        self,
        id: str,
        source_backend: Dict,
        target_backend: Dict,
        sample_generator,  # SampleGenerator object
        accelerator,
        webhook_progress_interval: int = 100,
        write_batch_size: int = 25,
        read_batch_size: int = 25,
        process_queue_size: int = 16,
        transform_batch_size: int = 4,
        max_workers: int = 32,
        delete_problematic_images: bool = False,
        hash_filenames: bool = False,
        conditioning_type: str = None,
    ):
        self.id = id
        self.source_backend = source_backend
        self.target_backend = target_backend
        self.sample_generator = sample_generator
        self.accelerator = accelerator

        # Get data backends
        self.source_data_backend = source_backend.get("data_backend")
        self.target_data_backend = target_backend.get("data_backend")

        # Get metadata backends
        self.source_metadata_backend = source_backend.get("metadata_backend")
        self.target_metadata_backend = target_backend.get("metadata_backend")

        # Processing parameters
        self.webhook_progress_interval = webhook_progress_interval
        self.write_batch_size = write_batch_size
        self.read_batch_size = read_batch_size
        self.process_queue_size = process_queue_size
        self.transform_batch_size = transform_batch_size
        self.max_workers = max_workers
        self.delete_problematic_images = delete_problematic_images
        self.hash_filenames = hash_filenames
        self.conditioning_type = conditioning_type

        # Get directory paths
        self.source_instance_dir = source_backend.get("instance_data_dir")
        self.target_instance_dir = target_backend.get("instance_data_dir")

        # Ensure target directory exists
        if self.target_data_backend.type == "local":
            os.makedirs(self.target_instance_dir, exist_ok=True)

        # Multi-GPU support
        from helpers.training.multi_process import rank_info, _get_rank

        self.rank_info = rank_info()
        self.rank = _get_rank()

        # Processing queues
        self.read_queue = Queue()
        self.process_queue = Queue()
        self.write_queue = Queue()
        self.transform_queue = Queue()

        # File mappings
        self.source_to_target_path = {}
        self.target_to_source_path = {}

        # Load metadata if needed
        if (
            self.source_metadata_backend
            and not self.source_metadata_backend.image_metadata_loaded
        ):
            self.source_metadata_backend.load_image_metadata()
        if (
            self.target_metadata_backend
            and not self.target_metadata_backend.image_metadata_loaded
        ):
            self.target_metadata_backend.load_image_metadata()

    def debug_log(self, msg: str):
        logger.debug(f"{self.rank_info}{msg}")

    def generate_target_filename(self, source_filepath: str) -> Tuple[str, str]:
        """Generate target filename from source filepath."""
        # Extract base filename and extension
        base_filename, extension = os.path.splitext(os.path.basename(source_filepath))
        if self.hash_filenames:
            base_filename = str(sha256(str(base_filename).encode()).hexdigest())

        # Preserve original file extension
        base_filename = f"{base_filename}{extension}"

        # Preserve directory structure
        subfolders = ""
        if self.source_instance_dir is not None:
            subfolders = os.path.dirname(source_filepath).replace(
                self.source_instance_dir, ""
            )
            subfolders = subfolders.lstrip(os.sep)

        if len(subfolders) > 0:
            full_filename = os.path.join(
                self.target_instance_dir, subfolders, base_filename
            )
        else:
            full_filename = os.path.join(self.target_instance_dir, base_filename)

        return full_filename, base_filename

    def build_filename_mappings(self, source_files: List[str]):
        """Build mappings between source and target filepaths."""
        self.source_to_target_path = {}
        self.target_to_source_path = {}

        for source_file in source_files:
            target_file, _ = self.generate_target_filename(source_file)
            if self.target_data_backend.type == "local":
                target_file = os.path.abspath(target_file)
            self.source_to_target_path[source_file] = target_file
            self.target_to_source_path[target_file] = source_file

    def already_processed(self, source_filepath: str) -> bool:
        """Check if a source file has already been processed."""
        target_path = self.source_to_target_path.get(source_filepath, None)
        if target_path and self.target_data_backend.exists(target_path):
            return True
        return False

    def discover_all_files(self) -> List[str]:
        """Discover all source files that need processing."""
        from helpers.training.state_tracker import StateTracker
        from helpers.training import image_file_extensions

        # Get all source files
        source_files = StateTracker.get_image_files(
            data_backend_id=self.source_backend["id"]
        ) or StateTracker.set_image_files(
            self.source_data_backend.list_files(
                instance_data_dir=self.source_instance_dir,
                file_extensions=image_file_extensions,
            ),
            data_backend_id=self.source_backend["id"],
        )

        # Get existing target files (using same extensions as source)
        target_files = self.target_data_backend.list_files(
            instance_data_dir=self.target_instance_dir,
            file_extensions=image_file_extensions,
        )

        self.debug_log(
            f"Found {len(source_files)} source files and {len(target_files)} existing target files"
        )

        # Build mappings
        self.build_filename_mappings(source_files)

        # Find unprocessed files
        unprocessed_files = []
        for source_file in source_files:
            if not self.already_processed(source_file):
                unprocessed_files.append(source_file)

        self.local_unprocessed_files = unprocessed_files
        self.debug_log(f"Found {len(unprocessed_files)} files to process")

        return unprocessed_files

    def _read_from_storage(self, filename: str, hide_errors: bool = False):
        """Read a file from the source storage backend."""
        try:
            return self.source_data_backend.read_image(filename)
        except Exception as e:
            if self.delete_problematic_images:
                self.source_metadata_backend.remove_image(filename)
                self.source_data_backend.delete(filename)
                self.debug_log(f"Deleted {filename} because it was problematic: {e}")
            if hide_errors:
                return None
            raise e

    def _read_from_storage_concurrently(
        self, paths: List[str], hide_errors: bool = False
    ):
        """Read files from storage concurrently."""

        def read_file(path):
            try:
                return path, self._read_from_storage(path, hide_errors=hide_errors)
            except Exception as e:
                logger.error(f"Error reading {path}: {e}")
                if self.delete_problematic_images:
                    self.source_metadata_backend.remove_image(path)
                    self.source_data_backend.delete(path)
                return path, None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {executor.submit(read_file, path): path for path in paths}
            for future in as_completed(future_to_path):
                try:
                    yield future.result()
                except Exception as exc:
                    logger.error(f"Exception during read: {exc}")

    def read_images_in_batch(self):
        """Read a batch of images from the read queue."""
        filepaths = []
        qlen = self.read_queue.qsize()

        for _ in range(qlen):
            filepath, metadata = self.read_queue.get()
            filepaths.append(filepath)

        # Read batch from backend
        available_filepaths, batch_output = self.source_data_backend.read_image_batch(
            filepaths, delete_problematic_images=self.delete_problematic_images
        )

        missing_count = len(filepaths) - len(available_filepaths)
        if missing_count > 0:
            logger.warning(
                f"Failed to read {missing_count} sample{'s' if missing_count > 1 else ''} "
                f"out of {len(filepaths)} total samples requested."
            )

        # Add to process queue
        for filepath, image_data in zip(available_filepaths, batch_output):
            metadata = self.source_metadata_backend.get_metadata_by_filepath(filepath)
            self.process_queue.put((filepath, image_data, metadata))

    def _process_images_in_batch(
        self, batch_data: List = None, disable_queue: bool = False
    ) -> List:
        """Process a batch of images for transformation."""
        try:
            if batch_data is not None:
                qlen = len(batch_data)
            else:
                qlen = self.process_queue.qsize()

            processed_items = []
            for _ in range(qlen):
                if batch_data:
                    filepath, image, metadata = batch_data.pop()
                else:
                    filepath, image, metadata = self.process_queue.get()

                # Skip if doesn't meet requirements
                if hasattr(
                    self.source_metadata_backend, "meets_resolution_requirements"
                ) and not self.source_metadata_backend.meets_resolution_requirements(
                    image_path=filepath
                ):
                    self.debug_log(
                        f"Skipping {filepath} - doesn't meet resolution requirements"
                    )
                    continue

                # Prepare item for transformation
                target_filepath = self.source_to_target_path[filepath]
                item = (filepath, target_filepath, image, metadata)
                processed_items.append(item)

                if not disable_queue:
                    self.transform_queue.put(item)

            return processed_items

        except Exception as e:
            logger.error(f"Error processing images: {e}")
            raise e

    def _transform_images_in_batch(
        self, transform_items: List = None, disable_queue: bool = False
    ) -> List:
        """Apply transformations to a batch of images."""
        try:
            if transform_items is not None:
                qlen = len(transform_items)
            else:
                qlen = self.transform_queue.qsize()

            if qlen == 0:
                return []

            output_items = []
            batch_items = []

            # Collect batch
            for _ in range(min(qlen, self.transform_batch_size)):
                if transform_items:
                    item = transform_items.pop()
                else:
                    item = self.transform_queue.get()
                batch_items.append(item)

            # Extract components
            source_paths = [item[0] for item in batch_items]
            target_paths = [item[1] for item in batch_items]
            images = [item[2] for item in batch_items]
            metadata_list = [item[3] for item in batch_items]

            # Apply transformation
            try:
                transformed_data = self.sample_generator.transform_batch(
                    images=images,
                    source_paths=source_paths,
                    metadata_list=metadata_list,
                    accelerator=self.accelerator,
                )

                # Prepare output
                for idx, (target_path, data) in enumerate(
                    zip(target_paths, transformed_data)
                ):
                    output_item = (target_path, data, metadata_list[idx])
                    output_items.append(output_item)

                    if not disable_queue:
                        self.write_queue.put(output_item)

            except Exception as e:
                logger.error(f"Error during transformation: {e}")
                # Remove problematic images
                for source_path in source_paths:
                    if self.delete_problematic_images:
                        self.source_metadata_backend.remove_image(source_path)
                raise e

            return output_items

        except Exception as e:
            logger.error(f"Error in transform batch: {e}")
            raise e

    def _write_data_in_batch(self, write_items: List = None) -> List:
        """Write transformed data to target backend in batches."""
        if write_items is not None:
            qlen = len(write_items)
        else:
            qlen = self.write_queue.qsize()

        filepaths = []
        data_items = []
        metadata_items = []

        for _ in range(qlen):
            if write_items:
                filepath, data, metadata = write_items.pop()
            else:
                filepath, data, metadata = self.write_queue.get()

            filepaths.append(filepath)
            data_items.append(data)
            metadata_items.append(metadata)

        # Write batch to target backend
        self.target_data_backend.write_batch(filepaths, data_items)

        # Update target metadata if needed
        if self.target_metadata_backend:
            for filepath, metadata in zip(filepaths, metadata_items):
                # Update metadata with target path
                metadata = metadata.copy() if metadata else {}
                metadata["original_source"] = self.target_to_source_path.get(
                    filepath, None
                )
                self.target_metadata_backend.update_metadata_by_filepath(
                    filepath, metadata
                )

        return data_items

    def _process_futures(self, futures: List, executor: ThreadPoolExecutor) -> List:
        """Process completed futures and return remaining ones."""
        completed_futures = []
        for future in as_completed(futures):
            try:
                future.result()
                completed_futures.append(future)
            except Exception as e:
                logger.error(
                    f"Error in future: {e}, traceback: {traceback.format_exc()}"
                )
                completed_futures.append(future)
        return [f for f in futures if f not in completed_futures]

    def process_buckets(self):
        """Main processing loop that handles all unprocessed files."""
        futures = []

        # Get aspect buckets from metadata
        aspect_bucket_cache = self.source_metadata_backend.read_cache().copy()

        # Shuffle if needed
        do_shuffle = (
            os.environ.get("SIMPLETUNER_SHUFFLE_ASPECTS", "true").lower() == "true"
        )
        if do_shuffle:
            shuffled_keys = list(aspect_bucket_cache.keys())
            shuffle(shuffled_keys)
        else:
            shuffled_keys = aspect_bucket_cache.keys()

        # Send initial webhook if configured
        if hasattr(self, "webhook_handler") and self.webhook_handler is not None:
            total_count = len(
                [item for sublist in aspect_bucket_cache.values() for item in sublist]
            )
            processed_count = total_count - len(self.local_unprocessed_files)
            self.send_progress_update(
                type="init_data_generation_started",
                progress=int(processed_count / total_count * 100),
                total=total_count,
                current=processed_count,
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for bucket in shuffled_keys:
                # Get files for this bucket that need processing
                bucket_files = []
                for filepath in aspect_bucket_cache[bucket]:
                    if filepath in self.local_unprocessed_files:
                        bucket_files.append(filepath)

                if not bucket_files:
                    continue

                if do_shuffle:
                    shuffle(bucket_files)

                statistics = {
                    "processed": 0,
                    "skipped": 0,
                    "errors": 0,
                    "total": len(bucket_files),
                }

                last_reported_index = 0

                for filepath in tqdm(
                    bucket_files,
                    desc=f"Processing bucket {bucket}",
                    position=self.rank,
                    ncols=125,
                    leave=False,
                ):
                    try:
                        # Skip if already processed
                        if self.already_processed(filepath):
                            statistics["skipped"] += 1
                            continue

                        # Get metadata
                        metadata = (
                            self.source_metadata_backend.get_metadata_by_filepath(
                                filepath
                            )
                        )

                        # Add to read queue
                        self.read_queue.put((filepath, metadata))

                        # Process read queue when ready
                        if self.read_queue.qsize() >= self.read_batch_size:
                            future = executor.submit(self.read_images_in_batch)
                            futures.append(future)

                        # Process image queue when ready
                        if self.process_queue.qsize() >= self.process_queue_size:
                            future = executor.submit(self._process_images_in_batch)
                            futures.append(future)

                        # Transform when ready
                        if self.transform_queue.qsize() >= self.transform_batch_size:
                            future = executor.submit(self._transform_images_in_batch)
                            futures.append(future)
                            statistics["processed"] += 1

                            # Send progress update
                            if (
                                hasattr(self, "webhook_handler")
                                and self.webhook_handler is not None
                                and int(
                                    statistics["processed"]
                                    // self.webhook_progress_interval
                                )
                                > last_reported_index
                            ):
                                last_reported_index = (
                                    statistics["processed"]
                                    // self.webhook_progress_interval
                                )
                                self.send_progress_update(
                                    type="data_generation",
                                    progress=int(
                                        statistics["processed"]
                                        / statistics["total"]
                                        * 100
                                    ),
                                    total=statistics["total"],
                                    current=statistics["processed"],
                                )

                        # Write when ready
                        if self.write_queue.qsize() >= self.write_batch_size:
                            future = executor.submit(self._write_data_in_batch)
                            futures.append(future)

                        # Process completed futures
                        futures = self._process_futures(futures, executor)

                    except Exception as e:
                        logger.error(f"Error processing {filepath}: {e}")
                        statistics["errors"] += 1
                        if "out of memory" in str(e).lower():
                            import sys

                            sys.exit(1)

                # Process remaining items in queues
                try:
                    # Read remaining
                    if self.read_queue.qsize() > 0:
                        future = executor.submit(self.read_images_in_batch)
                        futures.append(future)

                    futures = self._process_futures(futures, executor)

                    # Process remaining
                    if self.process_queue.qsize() > 0:
                        future = executor.submit(self._process_images_in_batch)
                        futures.append(future)

                    futures = self._process_futures(futures, executor)

                    # Transform remaining
                    if self.transform_queue.qsize() > 0:
                        future = executor.submit(self._transform_images_in_batch)
                        futures.append(future)

                    futures = self._process_futures(futures, executor)

                    # Write remaining
                    if self.write_queue.qsize() > 0:
                        future = executor.submit(self._write_data_in_batch)
                        futures.append(future)

                    futures = self._process_futures(futures, executor)

                    # Log statistics
                    log_msg = f"(id={self.id}) Bucket {bucket} processing results: {statistics}"
                    if self.rank == 0:
                        logger.info(log_msg)
                        tqdm.write(log_msg)

                    # Send final progress update
                    if (
                        hasattr(self, "webhook_handler")
                        and self.webhook_handler is not None
                    ):
                        self.send_progress_update(
                            type="data_generation_bucket_complete",
                            progress=100,
                            total=statistics["total"],
                            current=statistics["processed"],
                        )

                except Exception as e:
                    logger.error(f"Error processing bucket {bucket} remainders: {e}")
                    continue

        self.debug_log("Completed process_buckets, all futures have been returned.")

    def generate_dataset(self):
        """Main entry point to generate the dataset."""
        self.debug_log("Starting dataset generation")

        # Discover files to process
        self.discover_all_files()

        if len(self.local_unprocessed_files) == 0:
            self.debug_log("No files to process")
            return

        # Process all buckets
        self.process_buckets()

        # Update metadata
        if self.accelerator.is_local_main_process and self.target_metadata_backend:
            self.debug_log("Updating target metadata")
            self.target_metadata_backend.save_metadata()

        self.accelerator.wait_for_everyone()
        self.debug_log("Dataset generation complete")

    def send_progress_update(self, **kwargs):
        """Send progress update via webhook if configured."""
        if hasattr(self, "webhook_handler") and self.webhook_handler is not None:
            self.webhook_handler.send_progress_update(**kwargs)
