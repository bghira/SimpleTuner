import os
import logging
import traceback
import time
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

# Set up module-level logger
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
    Worker process for CPU-bound image transformations.

    - Reconstructs data backends in the subprocess.
    - Uses SampleGenerator to apply transformations.
    - Writes transformed images directly to target backend to avoid pickling overhead.
    - Reports successes or errors via done_queue.
    """
    import io
    import logging
    from helpers.data_backend.factory import from_instance_representation
    from helpers.data_generation.sample_generator import SampleGenerator

    # Initialize subprocess-specific logger
    logger = logging.getLogger(f"TransformWorker-{worker_id}")
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

    # Reconstruct backends and transformer
    try:
        source_data_backend = from_instance_representation(source_backend_repr)
        target_data_backend = from_instance_representation(target_backend_repr)

        sample_generator = SampleGenerator.from_backend(
            {"conditioning_config": sample_generator_config}
        )

        # Minimal accelerator stub for consistency
        class MinimalAccelerator:
            def __init__(self, device):
                self.device = device

        accelerator = MinimalAccelerator(accelerator_device)
        logger.info(f"Transform worker {worker_id} initialized")
    except Exception as e:
        logger.error(f"Worker {worker_id} init failed: {e}")
        return

    # Main worker loop
    while True:
        try:
            work_item = work_queue.get(timeout=30.0)
            if work_item is None:
                logger.info(f"Worker {worker_id} shutting down")
                break

            batch_items = work_item["batch_items"]
            batch_id = work_item.get("batch_id", 0)

            # Unpack batch elements
            source_paths = [item[0] for item in batch_items]
            target_paths = [item[1] for item in batch_items]
            images = [item[2] for item in batch_items]
            metadata_list = [item[3] for item in batch_items]

            try:
                # Apply batch transformation
                transformed = sample_generator.transform_batch(
                    images=images,
                    source_paths=source_paths,
                    metadata_list=metadata_list,
                    accelerator=accelerator,
                )

                filepaths, data_items, successes = [], [], []
                for target_path, img in zip(target_paths, transformed):
                    buffer = io.BytesIO()
                    ext = os.path.splitext(target_path)[1].lower()
                    format_map = {
                        ".jpg": "JPEG",
                        ".jpeg": "JPEG",
                        ".png": "PNG",
                        ".webp": "WEBP",
                        ".bmp": "BMP",
                    }
                    save_format = format_map.get(ext, "PNG")

                    if isinstance(img, Image.Image):
                        try:
                            img.save(buffer, format=save_format)
                            filepaths.append(target_path)
                            data_items.append(buffer.getvalue())
                            successes.append(target_path)
                        except Exception as e:
                            logger.error(f"Save failed for {target_path}: {e}")
                    else:
                        logger.error(f"Unexpected type in transform: {type(img)}")

                # Write batch if any succeeded
                if filepaths:
                    target_data_backend.write_batch(filepaths, data_items)
                    logger.debug(f"Worker {worker_id} wrote {len(filepaths)} files")

                # Report completion
                done_queue.put(
                    {
                        "batch_id": batch_id,
                        "worker_id": worker_id,
                        "successful": len(successes),
                        "total": len(batch_items),
                        "paths": successes,
                    }
                )

            except Exception as e:
                logger.error(f"Transform error in worker {worker_id}: {e}")
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
            # Ignore timeout; log unexpected errors
            if "Empty" not in str(e) and "timeout" not in str(e).lower():
                logger.error(f"Worker {worker_id} queue error: {e}")


class DataGenerator:
    """
    DataGenerator orchestrates reading, transforming, and writing image datasets.

    Workflow:
    1. discover_all_files: find unprocessed source files
    2. process_buckets: coordinate I/O and CPU-bound transforms
       - ThreadPoolExecutor handles I/O (read_images_in_batch)
       - Pool of subprocess workers handles CPU transforms
    3. generate_dataset: high-level entry point
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
        # Save parameters
        self.id = id
        self.source_backend = source_backend
        self.target_backend = target_backend
        self.accelerator = accelerator

        # Configure data and metadata backends
        self.source_data_backend = source_backend.get("data_backend")
        self.target_data_backend = target_backend.get("data_backend")
        self.source_metadata_backend = source_backend.get("metadata_backend")
        self.target_metadata_backend = target_backend.get("metadata_backend")

        # Pipeline settings
        self.webhook_progress_interval = webhook_progress_interval
        self.read_batch_size = read_batch_size
        self.process_queue_size = process_queue_size
        self.transform_batch_size = transform_batch_size
        self.max_workers = max_workers
        self.delete_problematic_images = delete_problematic_images
        self.hash_filenames = hash_filenames
        self.conditioning_type = conditioning_type

        # Prepare directories
        self.source_instance_dir = source_backend.get("instance_data_dir")
        self.target_instance_dir = target_backend.get("instance_data_dir")
        if self.target_data_backend.type == "local":
            os.makedirs(self.target_instance_dir, exist_ok=True)

        # Multi-GPU rank info
        from helpers.training.multi_process import rank_info, _get_rank

        self.rank_info = rank_info()
        self.rank = _get_rank()

        # Instantiate sample generator to detect GPU requirement
        from helpers.data_generation.sample_generator import SampleGenerator

        cfg = self.target_backend["config"].get("conditioning_config", {})
        self.sample_generator = SampleGenerator.from_backend(
            {"conditioning_config": cfg}
        )
        self.gpu_mode = getattr(self.sample_generator, "requires_gpu", False)

        # Queues for pipeline coordination
        self.read_queue = Queue()
        self.process_queue = Queue()

        if not self.gpu_mode:
            # Multiprocessing setup for CPU-bound transforms
            self.manager = Manager()
            self.transform_queue = self.manager.Queue()
            self.done_queue = self.manager.Queue()
            self.transform_workers = []
            self.num_transform_workers = (
                max(1, min(mp.cpu_count() - 2, 8))
                if num_transform_workers is None
                else num_transform_workers
            )
        else:
            logger.info(
                f"GPU mode enabled ({self.conditioning_type}), transforms in main process"
            )
            self.transform_workers = []
            self.num_transform_workers = 0

        # Path mappings for source -> target
        self.source_to_target_path = {}
        self.target_to_source_path = {}

        # Load metadata caches if needed
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

    def _start_transform_workers(self):
        """
        Launch worker subprocesses for CPU-bound image transforms.

        Each process runs transform_worker_process and communicates via transform_queue and done_queue.
        """
        if self.gpu_mode:
            return

        src_repr = self.source_data_backend.get_instance_representation()
        tgt_repr = self.target_data_backend.get_instance_representation()
        cfg = self.target_backend["config"].get("conditioning_config", {})

        for i in range(self.num_transform_workers):
            p = Process(
                target=transform_worker_process,
                args=(
                    self.transform_queue,
                    self.done_queue,
                    src_repr,
                    tgt_repr,
                    cfg,
                    str(self.accelerator.device) if self.accelerator else "cpu",
                    i,
                    self.delete_problematic_images,
                ),
            )
            p.start()
            self.transform_workers.append(p)

        self.debug_log(f"Launched {self.num_transform_workers} transform workers")

    def _stop_transform_workers(self):
        """
        Send shutdown signals (None) and join all transform workers.
        Force-terminate any that do not exit within 30s.
        """
        if self.gpu_mode:
            return

        for _ in self.transform_workers:
            self.transform_queue.put(None)
        for p in self.transform_workers:
            p.join(timeout=30)
            if p.is_alive():
                logger.warning(f"Force killing worker {p.pid}")
                p.terminate()
                p.join()
        self.transform_workers = []
        self.debug_log("All transform workers stopped")

    def _check_completion_queue(self) -> int:
        """
        Drain done_queue and sum up successfully processed images.
        Returns the count of newly completed items.
        """
        if self.gpu_mode:
            return 0
        completed = 0
        while not self.done_queue.empty():
            res = self.done_queue.get_nowait()
            if "error" not in res:
                completed += res.get("successful", 0)
                self.debug_log(
                    f"Batch {res['batch_id']} done: {res['successful']} of {res['total']}"
                )
            else:
                logger.error(f"Batch {res['batch_id']} failed: {res['error']}")
        return completed

    def _process_images_gpu_mode(self, batch_items: List) -> int:
        """
        Apply transformations in the main process when GPU is available.
        Writes results directly via target_data_backend.
        Returns the count of successful writes.
        """
        import io

        if not batch_items:
            return 0
        src_paths = [it[0] for it in batch_items]
        tgt_paths = [it[1] for it in batch_items]
        imgs = [it[2] for it in batch_items]
        metas = [it[3] for it in batch_items]

        try:
            transformed = self.sample_generator.transform_batch(
                images=imgs,
                source_paths=src_paths,
                metadata_list=metas,
                accelerator=self.accelerator,
            )
            filepaths, data_items, count = [], [], 0
            for tgt, img in zip(tgt_paths, transformed):
                buf = io.BytesIO()
                ext = os.path.splitext(tgt)[1].lower()
                fmt_map = {
                    ".jpg": "JPEG",
                    ".jpeg": "JPEG",
                    ".png": "PNG",
                    ".webp": "WEBP",
                    ".bmp": "BMP",
                }
                fmt = fmt_map.get(ext, "PNG")

                if isinstance(img, Image.Image):
                    try:
                        img.save(buf, format=fmt)
                        filepaths.append(tgt)
                        data_items.append(buf.getvalue())
                        count += 1
                    except Exception as e:
                        logger.error(f"GPU save failed for {tgt}: {e}")
                else:
                    logger.error(f"Unexpected transformed type: {type(img)}")

            if filepaths:
                self.target_data_backend.write_batch(filepaths, data_items)
                logger.debug(f"GPU mode wrote {len(filepaths)} files")
            return count

        except Exception as e:
            logger.error(f"GPU transform error: {e}")
            raise

    def _process_images_in_batch(self, batch_id: int) -> int:
        """
        Prepare and dispatch a batch of images for transformation.
        In CPU mode, sends to worker queue; in GPU mode, calls direct processing.
        Returns the number of items dispatched or processed.
        """
        try:
            qlen = self.process_queue.qsize()
            if qlen == 0:
                return 0

            batch_items = []
            for _ in range(min(qlen, self.transform_batch_size)):
                path, img, meta = self.process_queue.get()
                # Skip images not meeting resolution criteria
                if hasattr(
                    self.source_metadata_backend, "meets_resolution_requirements"
                ) and not self.source_metadata_backend.meets_resolution_requirements(
                    image_path=path
                ):
                    self.debug_log(f"Skipped {path}: resolution requirement")
                    continue

                tgt_path = self.source_to_target_path[path]
                batch_items.append((path, tgt_path, img, meta))

            if not batch_items:
                return 0

            if self.gpu_mode:
                return self._process_images_gpu_mode(batch_items)

            # CPU mode: enqueue work for subprocesses
            self.transform_queue.put({"batch_items": batch_items, "batch_id": batch_id})
            return len(batch_items)

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            raise

    def discover_all_files(self) -> List[str]:
        """
        Identify source files to process and map them to target paths.
        Returns list of unprocessed source file paths.
        """
        from helpers.training.state_tracker import StateTracker
        from helpers.training import image_file_extensions

        # Retrieve or cache list of source image files
        source_files = StateTracker.get_image_files(
            data_backend_id=self.source_backend["id"]
        ) or StateTracker.set_image_files(
            self.source_data_backend.list_files(
                instance_data_dir=self.source_instance_dir,
                file_extensions=image_file_extensions,
            ),
            data_backend_id=self.source_backend["id"],
        )

        # List existing target files
        target_files = self.target_data_backend.list_files(
            instance_data_dir=self.target_instance_dir,
            file_extensions=image_file_extensions,
        )

        self.debug_log(
            f"Found {len(source_files)} source and {len(target_files)} existing target images"
        )

        # Build filename mappings
        self.source_to_target_path.clear()
        self.target_to_source_path.clear()
        for src in source_files:
            tgt, _ = self.generate_target_filename(src)
            if self.target_data_backend.type == "local":
                tgt = os.path.abspath(tgt)
            self.source_to_target_path[src] = tgt
            self.target_to_source_path[tgt] = src

        # Filter out files already processed
        unprocessed = [
            src
            for src in source_files
            if not self.target_data_backend.exists(self.source_to_target_path[src])
        ]
        self.local_unprocessed_files = unprocessed
        self.debug_log(f"{len(unprocessed)} files to process")
        return unprocessed

    def generate_target_filename(self, source_filepath: str) -> Tuple[str, str]:
        """
        Create a corresponding target filename for a given source file.
        Preserves directory structure and file extension, with optional hashing.
        Returns (full_target_path, base_filename).
        """
        base, ext = os.path.splitext(os.path.basename(source_filepath))
        filename = f"{base}{ext}"

        subpath = ""
        if self.source_instance_dir:
            subpath = (
                os.path.dirname(source_filepath)
                .replace(self.source_instance_dir, "")
                .lstrip(os.sep)
            )

        full_path = (
            os.path.join(self.target_instance_dir, subpath, filename)
            if subpath
            else os.path.join(self.target_instance_dir, filename)
        )
        return full_path, filename

    def read_images_in_batch(self):
        """
        Read a batch of images from the source backend (I/O-bound).
        Enqueues the results into process_queue for transformation.
        """
        filepaths, meta_map = [], {}
        qlen = self.read_queue.qsize()
        for _ in range(qlen):
            filepath, metadata = self.read_queue.get()
            filepaths.append(filepath)
            meta_map[filepath] = metadata

        available, data_batch = self.source_data_backend.read_image_batch(
            filepaths, delete_problematic_images=self.delete_problematic_images
        )

        missing = len(filepaths) - len(available)
        if missing:
            logger.warning(f"Read failure: {missing}/{len(filepaths)} images missing")

        for fp, img_data in zip(available, data_batch):
            meta = meta_map.get(
                fp
            ) or self.source_metadata_backend.get_metadata_by_filepath(fp)
            self.process_queue.put((fp, img_data, meta))

    def _process_futures(self, futures: List, executor: ThreadPoolExecutor) -> List:
        """
        Clean up completed ThreadPoolExecutor futures, logging errors if any.
        Returns list of remaining futures.
        """
        done, remain = [], []
        for f in as_completed(futures):
            try:
                f.result()
                done.append(f)
            except Exception as e:
                logger.error(f"Future error: {e}\n{traceback.format_exc()}")
                done.append(f)
        return [f for f in futures if f not in done]

    def process_buckets(self):
        """
        Main loop:
        - Start transform workers
        - For each aspect bucket:
          - Read files in batches via threads
          - Dispatch transform batches
          - Collect completed transforms
          - Send optional webhook updates
        - Shutdown workers
        """
        self._start_transform_workers()
        try:
            futures, batch_id = [], 0
            aspect_cache = self.source_metadata_backend.read_cache().copy()
            buckets = list(aspect_cache.keys())
            if os.environ.get("SIMPLETUNER_SHUFFLE_ASPECTS", "true").lower() == "true":
                shuffle(buckets)

            # Optional initial progress webhook
            if hasattr(self, "webhook_handler") and self.webhook_handler:
                total = sum(len(v) for v in aspect_cache.values())
                self.send_progress_update(
                    type="init_data_generation_started",
                    progress=0,
                    total=total,
                    current=0,
                )

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for bucket in buckets:
                    files = [
                        fp
                        for fp in aspect_cache[bucket]
                        if fp in self.local_unprocessed_files
                    ]
                    if not files:
                        continue
                    if (
                        os.environ.get("SIMPLETUNER_SHUFFLE_ASPECTS", "true").lower()
                        == "true"
                    ):
                        shuffle(files)

                    stats = {
                        "processed": 0,
                        "skipped": 0,
                        "errors": 0,
                        "total": len(files),
                    }
                    last_report = 0

                    for fp in tqdm(
                        files, desc=f"Bucket {bucket}", position=self.rank, leave=False
                    ):
                        try:
                            if self.target_data_backend.exists(
                                self.source_to_target_path[fp]
                            ):
                                stats["skipped"] += 1
                                continue
                            meta = (
                                self.source_metadata_backend.get_metadata_by_filepath(
                                    fp
                                )
                            )
                            self.read_queue.put((fp, meta))

                            # Trigger read batch
                            if self.read_queue.qsize() >= self.read_batch_size:
                                futures.append(
                                    executor.submit(self.read_images_in_batch)
                                )

                            # Trigger transform batch dispatch
                            if self.process_queue.qsize() >= self.process_queue_size:
                                batch_id += 1
                                sent = self._process_images_in_batch(batch_id)
                                if self.gpu_mode and sent:
                                    stats["processed"] += sent
                                elif sent:
                                    self.debug_log(
                                        f"Dispatched batch {batch_id} ({sent} items)"
                                    )

                            # Collect completed work
                            if not self.gpu_mode:
                                done = self._check_completion_queue()
                                stats["processed"] += done

                            # Periodic webhook
                            if (
                                hasattr(self, "webhook_handler")
                                and self.webhook_handler
                                and stats["processed"] // self.webhook_progress_interval
                                > last_report
                            ):
                                last_report = (
                                    stats["processed"] // self.webhook_progress_interval
                                )
                                self.send_progress_update(
                                    type="data_generation",
                                    progress=int(
                                        stats["processed"] / stats["total"] * 100
                                    ),
                                    total=stats["total"],
                                    current=stats["processed"],
                                )

                            # Process finished read tasks
                            futures = self._process_futures(futures, executor)

                        except Exception as e:
                            logger.error(f"Error processing {fp}: {e}")
                            stats["errors"] += 1

                    # Flush remaining reads and transforms
                    if self.read_queue.qsize():
                        futures.append(executor.submit(self.read_images_in_batch))
                    futures = self._process_futures(futures, executor)
                    while self.process_queue.qsize():
                        batch_id += 1
                        self._process_images_in_batch(batch_id)
                    if not self.gpu_mode:
                        start = time.time()
                        while (
                            self.transform_queue.qsize() or not self.done_queue.empty()
                        ) and time.time() - start < 30:
                            stats["processed"] += self._check_completion_queue()
                            time.sleep(0.1)

                    # Final log and webhook for bucket
                    msg = f"(id={self.id}) Bucket {bucket} done: {stats}"
                    if self.rank == 0:
                        logger.info(msg)
                        tqdm.write(msg)
                    if hasattr(self, "webhook_handler") and self.webhook_handler:
                        self.send_progress_update(
                            type="data_generation_bucket_complete",
                            progress=100,
                            total=stats["total"],
                            current=stats["processed"],
                        )

            self.debug_log("All buckets processed")

        finally:
            self._stop_transform_workers()

    def generate_dataset(self):
        """
        High-level entry point to run the full dataset generation pipeline.
        """
        self.debug_log("Starting dataset generation")
        files = self.discover_all_files()
        if not files:
            self.debug_log("No files to process, exiting")
            return
        self.process_buckets()
        self.accelerator.wait_for_everyone()
        self.debug_log("Dataset generation complete")

    def send_progress_update(self, **kwargs):
        """Invoke webhook handler if configured."""
        if hasattr(self, "webhook_handler") and self.webhook_handler:
            self.webhook_handler.send_progress_update(**kwargs)

    def debug_log(self, msg: str):
        """Helper to include rank info in debug logs."""
        logger.debug(f"{self.rank_info}{msg}")
