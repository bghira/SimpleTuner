"""
DataGenerator - Efficient multiprocessing pipeline for dataset transformations.

Architecture Overview:
1. Main process orchestrates the pipeline and manages queues
2. Thread pool handles I/O-bound operations (reading source files)
3. Process pool handles CPU-bound operations (image transformations)
4. Worker processes write directly to target backend to avoid pickling overhead

The key insight is that CPU-intensive image processing cannot use threads in Python
due to the GIL, so we use separate processes. To avoid the overhead of pickling
transformed images back to the main process, workers write directly to the target.
"""

import os
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Manager
from random import shuffle
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from numpy import str_ as numpy_str
from queue import Queue
from hashlib import sha256
from typing import List, Dict, Tuple, Any, Optional
import multiprocessing as mp
import time

logger = logging.getLogger("DataGenerator")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


def transform_worker_process(
    work_queue,
    done_queue,
    source_backend_repr: Dict,
    target_backend_repr: Dict,
    sample_generator_config: Dict,
    accelerator_device: str,
    worker_id: int,
    delete_problematic_images: bool = False,
):
    """
    Worker process that pulls items from queue, transforms them, and writes directly.
    This runs in a separate process to avoid GIL limitations.
    """
    import os
    import io
    import logging
    from PIL import Image
    from helpers.data_backend.factory import from_instance_representation
    from helpers.data_generation.sample_generator import SampleGenerator

    # Set up logging for this worker
    logger = logging.getLogger(f"TransformWorker-{worker_id}")
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

    # Reconstruct backends in the subprocess
    try:
        source_data_backend = from_instance_representation(source_backend_repr)
        target_data_backend = from_instance_representation(target_backend_repr)

        # Create sample generator
        sample_generator = SampleGenerator.from_backend(
            {"conditioning_config": sample_generator_config}
        )

        # Set up minimal accelerator info
        class MinimalAccelerator:
            def __init__(self, device):
                self.device = device

        accelerator = MinimalAccelerator(accelerator_device)

        logger.info(f"Transform worker {worker_id} started")
    except Exception as e:
        logger.error(f"Worker {worker_id} failed to initialize: {e}")
        return

    while True:
        try:
            # Get work item with timeout
            work_item = work_queue.get(timeout=30.0)

            if work_item is None:  # Poison pill
                logger.info(f"Transform worker {worker_id} shutting down")
                break

            batch_items = work_item["batch_items"]
            batch_id = work_item.get("batch_id", 0)

            # Extract components
            source_paths = [item[0] for item in batch_items]
            target_paths = [item[1] for item in batch_items]
            images = [item[2] for item in batch_items]
            metadata_list = [item[3] for item in batch_items]

            # Apply transformation
            try:
                transformed_data = sample_generator.transform_batch(
                    images=images,
                    source_paths=source_paths,
                    metadata_list=metadata_list,
                    accelerator=accelerator,
                )

                # Write directly from the worker process
                filepaths = []
                data_items = []
                successful_paths = []

                for target_path, transformed_img in zip(target_paths, transformed_data):
                    # Convert image to bytes
                    buffer = io.BytesIO()

                    # Determine format from file extension
                    ext = os.path.splitext(target_path)[1].lower()
                    format_map = {
                        ".jpg": "JPEG",
                        ".jpeg": "JPEG",
                        ".png": "PNG",
                        ".webp": "WEBP",
                        ".bmp": "BMP",
                    }
                    save_format = format_map.get(ext, "PNG")

                    # Save to buffer
                    if isinstance(transformed_img, Image.Image):
                        try:
                            transformed_img.save(buffer, format=save_format)
                            data_items.append(buffer.getvalue())
                            filepaths.append(target_path)
                            successful_paths.append(target_path)
                        except Exception as e:
                            logger.error(f"Failed to save image {target_path}: {e}")
                    else:
                        logger.error(
                            f"Unexpected transformed data type: {type(transformed_img)}"
                        )

                # Write batch
                if filepaths:
                    target_data_backend.write_batch(filepaths, data_items)
                    logger.debug(f"Worker {worker_id} wrote {len(filepaths)} files")

                # Report completion
                done_queue.put(
                    {
                        "batch_id": batch_id,
                        "worker_id": worker_id,
                        "successful": len(successful_paths),
                        "total": len(batch_items),
                        "paths": successful_paths,
                    }
                )

            except Exception as e:
                logger.error(f"Worker {worker_id} transform error: {e}")
                # Report failure
                done_queue.put(
                    {
                        "batch_id": batch_id,
                        "worker_id": worker_id,
                        "successful": 0,
                        "total": len(batch_items),
                        "error": str(e),
                    }
                )

        except Exception as e:
            if "Empty" not in str(e) and "timeout" not in str(e).lower():
                logger.error(f"Worker {worker_id} error: {e}")


class DataGenerator:
    """
    A data generator that reads from source backend, applies transformations via SampleGenerator,
    and writes to target backend. Uses multiprocessing for CPU-bound transforms.

    Architecture:
    - Thread pool handles I/O operations (reading source files)
    - Process pool handles CPU-intensive transforms and writes directly to avoid pickling
    - Worker processes reconstruct backends from serialized representations

    Output files preserve the same extension as input files (e.g., .png -> .png, .jpg -> .jpg).
    """

    def __init__(
        self,
        id: str,
        source_backend: Dict,
        target_backend: Dict,
        accelerator,
        webhook_progress_interval: int = 100,
        read_batch_size: int = 25,
        process_queue_size: int = 16,
        transform_batch_size: int = 4,
        max_workers: int = 32,
        num_transform_workers: int = None,
        delete_problematic_images: bool = False,
        hash_filenames: bool = False,
        conditioning_type: str = None,
    ):
        self.id = id
        self.source_backend = source_backend
        self.target_backend = target_backend
        self.accelerator = accelerator

        # Get data backends
        self.source_data_backend = source_backend.get("data_backend")
        self.target_data_backend = target_backend.get("data_backend")

        # Get metadata backends
        self.source_metadata_backend = source_backend.get("metadata_backend")
        self.target_metadata_backend = target_backend.get("metadata_backend")

        # Processing parameters
        self.webhook_progress_interval = webhook_progress_interval
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

        # Processing queues - use regular Queue for thread operations
        self.read_queue = Queue()
        self.process_queue = Queue()

        # Multiprocessing components for transforms
        self.manager = Manager()
        self.transform_queue = self.manager.Queue()  # Work queue
        self.done_queue = self.manager.Queue()  # Completion queue
        self.transform_workers = []

        # Set number of transform workers
        if num_transform_workers is None:
            self.num_transform_workers = max(1, min(mp.cpu_count() - 2, 8))  # Cap at 8
        else:
            self.num_transform_workers = num_transform_workers

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
        metadata_map = {}
        qlen = self.read_queue.qsize()

        for _ in range(qlen):
            filepath, metadata = self.read_queue.get()
            filepaths.append(filepath)
            metadata_map[filepath] = metadata

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
            metadata = metadata_map.get(filepath, None)
            if metadata is None:
                metadata = self.source_metadata_backend.get_metadata_by_filepath(
                    filepath
                )
            self.process_queue.put((filepath, image_data, metadata))

    def _start_transform_workers(self):
        """Start the transform worker processes."""
        # Get serializable representations of backends
        source_backend_repr = self.source_data_backend.get_instance_representation()
        target_backend_repr = self.target_data_backend.get_instance_representation()

        # Get sample generator config
        sample_generator_config = self.target_backend["config"].get(
            "conditioning_config", {}
        )

        # Start worker processes
        for i in range(self.num_transform_workers):
            worker = Process(
                target=transform_worker_process,
                args=(
                    self.transform_queue,
                    self.done_queue,
                    source_backend_repr,
                    target_backend_repr,
                    sample_generator_config,
                    str(self.accelerator.device) if self.accelerator else "cpu",
                    i,
                    self.delete_problematic_images,
                ),
            )
            worker.start()
            self.transform_workers.append(worker)

        self.debug_log(f"Started {self.num_transform_workers} transform workers")

    def _stop_transform_workers(self):
        """Stop all transform worker processes."""
        # Send poison pills
        for _ in self.transform_workers:
            self.transform_queue.put(None)

        # Wait for workers to finish
        for worker in self.transform_workers:
            worker.join(timeout=30)
            if worker.is_alive():
                logger.warning(f"Force terminating worker {worker.pid}")
                worker.terminate()
                worker.join()

        self.transform_workers = []
        self.debug_log("All transform workers stopped")

    def _process_images_in_batch(self, batch_id: int) -> int:
        """Process a batch of images for transformation."""
        try:
            qlen = self.process_queue.qsize()
            if qlen == 0:
                return 0

            batch_items = []
            for _ in range(min(qlen, self.transform_batch_size)):
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
                batch_items.append(item)

            if batch_items:
                # Send to transform workers
                work_item = {"batch_items": batch_items, "batch_id": batch_id}
                self.transform_queue.put(work_item)
                return len(batch_items)

            return 0

        except Exception as e:
            logger.error(f"Error processing images: {e}")
            raise e

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

    def _check_completion_queue(self) -> int:
        """Check the completion queue and return number of successfully processed items."""
        processed = 0
        while not self.done_queue.empty():
            try:
                result = self.done_queue.get_nowait()
                if "error" not in result:
                    processed += result["successful"]
                    self.debug_log(
                        f"Batch {result['batch_id']} completed by worker {result['worker_id']}: {result['successful']}/{result['total']}"
                    )
                else:
                    logger.error(
                        f"Batch {result['batch_id']} failed: {result['error']}"
                    )
            except:
                break
        return processed

    def process_buckets(self):
        """Main processing loop that handles all unprocessed files."""
        # Start transform workers
        self._start_transform_workers()

        try:
            futures = []
            batch_id_counter = 0

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
                    [
                        item
                        for sublist in aspect_bucket_cache.values()
                        for item in sublist
                    ]
                )
                processed_count = total_count - len(self.local_unprocessed_files)
                self.send_progress_update(
                    type="init_data_generation_started",
                    progress=int(processed_count / total_count * 100),
                    total=total_count,
                    current=processed_count,
                )

            # Use thread pool for I/O operations only
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

                            # Process read queue when ready (I/O bound - use threads)
                            if self.read_queue.qsize() >= self.read_batch_size:
                                future = executor.submit(self.read_images_in_batch)
                                futures.append(future)

                            # Process image queue when ready (sends to transform workers)
                            if self.process_queue.qsize() >= self.process_queue_size:
                                batch_id_counter += 1
                                items_sent = self._process_images_in_batch(
                                    batch_id_counter
                                )
                                if items_sent > 0:
                                    self.debug_log(
                                        f"Sent batch {batch_id_counter} with {items_sent} items to workers"
                                    )

                            # Check for completed work
                            completed = self._check_completion_queue()
                            if completed > 0:
                                statistics["processed"] += completed

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
                        while self.process_queue.qsize() > 0:
                            batch_id_counter += 1
                            items_sent = self._process_images_in_batch(batch_id_counter)
                            if items_sent > 0:
                                self.debug_log(
                                    f"Sent final batch {batch_id_counter} with {items_sent} items"
                                )

                        # Wait for all workers to finish processing
                        timeout = 30  # seconds
                        start_time = time.time()
                        while (
                            self.transform_queue.qsize() > 0
                            or not self.done_queue.empty()
                        ):
                            completed = self._check_completion_queue()
                            if completed > 0:
                                statistics["processed"] += completed
                            time.sleep(0.1)
                            if time.time() - start_time > timeout:
                                logger.warning("Timeout waiting for workers to finish")
                                break

                        # Final check for completed work
                        completed = self._check_completion_queue()
                        if completed > 0:
                            statistics["processed"] += completed

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
                        logger.error(
                            f"Error processing bucket {bucket} remainders: {e}"
                        )
                        continue

            self.debug_log("Completed process_buckets, all futures have been returned.")

        finally:
            # Always stop workers
            self._stop_transform_workers()

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

        # Update target metadata with processed files
        if self.accelerator.is_local_main_process and self.target_metadata_backend:
            self.debug_log("Updating target metadata")
            from helpers.training import image_file_extensions

            # Scan target directory and update metadata for new files
            target_files = self.target_data_backend.list_files(
                instance_data_dir=self.target_instance_dir,
                file_extensions=image_file_extensions,
            )
            for target_file in target_files:
                source_file = self.target_to_source_path.get(target_file, None)
                if source_file:
                    source_metadata = (
                        self.source_metadata_backend.get_metadata_by_filepath(
                            source_file
                        )
                    )
                    if source_metadata:
                        # Copy metadata with source reference
                        target_metadata = source_metadata.copy()
                        target_metadata["original_source"] = source_file
                        self.target_metadata_backend.update_metadata_by_filepath(
                            target_file, target_metadata
                        )

            self.target_metadata_backend.save_metadata()

        self.accelerator.wait_for_everyone()
        self.debug_log("Dataset generation complete")

    def send_progress_update(self, **kwargs):
        """Send progress update via webhook if configured."""
        if hasattr(self, "webhook_handler") and self.webhook_handler is not None:
            self.webhook_handler.send_progress_update(**kwargs)


# Key Architecture Points:
#
# 1. **Separation of I/O and CPU work**:
#    - Threads handle file reading (I/O bound)
#    - Processes handle transformations (CPU bound)
#
# 2. **Direct writing from workers**:
#    - Avoids pickling large image data back to main process
#    - Workers reconstruct backends using serialization methods
#
# 3. **Backend serialization**:
#    - Uses backend.get_instance_representation() to serialize
#    - Uses BaseDataBackend.from_instance_representation() to deserialize
#
# 4. **Automatic worker management**:
#    - Number of transform workers based on CPU count
#    - Graceful shutdown with poison pills
#
# 5. **Queue-based coordination**:
#    - Thread-safe Queue for I/O operations
#    - Manager.Queue for inter-process communication

# Example usage:
#
# generator = DataGenerator(
#     id=target_backend["id"],
#     source_backend=source_backend_config,
#     target_backend=target_backend,  # Contains conditioning_config
#     accelerator=accelerator,
#     transform_batch_size=8,  # Images per batch sent to workers
#     read_batch_size=32,      # Files to read at once (I/O bound)
#     max_workers=16,          # Thread pool workers for I/O operations
#     num_transform_workers=4, # Process workers for transforms (optional)
# )
#
# generator.generate_dataset()
