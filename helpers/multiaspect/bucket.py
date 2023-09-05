from helpers.multiaspect.image import MultiaspectImage
from helpers.data_backend.base import BaseDataBackend
from pathlib import Path
import json, logging, os
from multiprocessing import Manager
from tqdm import tqdm
from multiprocessing import Process, Queue
import numpy as np

logger = logging.getLogger("BucketManager")
target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING")
logger.setLevel(target_level)


class BucketManager:
    def __init__(
        self, instance_data_root: str, cache_file: str, data_backend: BaseDataBackend
    ):
        self.data_backend = data_backend
        self.instance_data_root = Path(instance_data_root)
        self.cache_file = Path(cache_file)
        self.aspect_ratio_bucket_indices = {}
        self.instance_images_path = set()
        # Initialize a multiprocessing.Manager dict for seen_images
        manager = Manager()
        self.seen_images = manager.dict()
        self._load_cache()

    def __len__(self):
        return sum(
            [len(bucket) for bucket in self.aspect_ratio_bucket_indices.values()]
        )

    def _discover_new_files(self):
        """
        Discover new files that have not been processed yet.

        Returns:
            list: A list of new files.
        """
        all_image_files_data = self.data_backend.list_files(
            instance_data_root=self.instance_data_root,
            str_pattern="*.[jJpP][pPnN][gG]",
        )

        # Extract only the files from the data
        all_image_files = [
            file for _, _, files in all_image_files_data for file in files
        ]

        return [
            file
            for file in all_image_files
            if str(file) not in self.instance_images_path
        ]

    def _load_cache(self):
        """
        Load cache data from file.

        Returns:
            dict: The cache data.
        """
        # Query our DataBackend to see whether the cache file exists.
        if self.data_backend.exists(self.cache_file):
            try:
                # Use our DataBackend to actually read the cache file.
                cache_data_raw = self.data_backend.read(self.cache_file)
                cache_data = json.loads(cache_data_raw)
            except Exception as e:
                logger.warning(f'Error loading aspect bucket cache, creating new one: {e}')
                cache_data = {}
            self.aspect_ratio_bucket_indices = cache_data.get(
                "aspect_ratio_bucket_indices", {}
            )
            self.instance_images_path = set(cache_data.get("instance_images_path", []))

    def _save_cache(self):
        """
        Save cache data to file.
        """
        # Convert any non-strings into strings as we save the index.
        aspect_ratio_bucket_indices_str = {
            key: [str(path) for path in value]
            for key, value in self.aspect_ratio_bucket_indices.items()
        }
        # Encode the cache as JSON.
        cache_data = {
            "aspect_ratio_bucket_indices": aspect_ratio_bucket_indices_str,
            "instance_images_path": [str(path) for path in self.instance_images_path],
        }
        cache_data_str = json.dumps(cache_data)
        # Use our DataBackend to write the cache file.
        self.data_backend.write(self.cache_file, cache_data_str)

    def _bucket_worker(
        self,
        tqdm_queue,
        files,
        aspect_ratio_bucket_indices_queue,
        existing_files_set,
        data_backend,
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
        for file in files:
            if str(file) not in existing_files_set:
                local_aspect_ratio_bucket_indices = MultiaspectImage.process_for_bucket(
                    data_backend, file, local_aspect_ratio_bucket_indices
                )
            tqdm_queue.put(1)
        aspect_ratio_bucket_indices_queue.put(local_aspect_ratio_bucket_indices)

    def compute_aspect_ratio_bucket_indices(self):
        """
        Compute the aspect ratio bucket indices. The workhorse of this class.

        Returns:
            dict: The aspect ratio bucket indices.
        """
        logger.info("Discovering new files...")
        new_files = self._discover_new_files()

        if not new_files:
            logger.info("No new files discovered. Exiting.")
            return

        existing_files_set = set().union(*self.aspect_ratio_bucket_indices.values())

        num_cpus = 8  # Using a fixed number for better control and predictability
        files_split = np.array_split(new_files, num_cpus)

        tqdm_queue = Queue()
        aspect_ratio_bucket_indices_queue = Queue()

        workers = [
            Process(
                target=self._bucket_worker,
                args=(
                    tqdm_queue,
                    file_shard,
                    aspect_ratio_bucket_indices_queue,
                    existing_files_set,
                    self.data_backend,
                ),
            )
            for file_shard in files_split
        ]

        for worker in workers:
            worker.start()

        with tqdm(total=len(new_files)) as pbar:
            while any(worker.is_alive() for worker in workers):
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

        for worker in workers:
            worker.join()

        self.instance_images_path.update(new_files)
        self._save_cache()
        logger.info("Completed aspect bucket update.")

    def mark_as_seen(self, image_path):
        """Mark an image as seen."""
        self.seen_images[image_path] = True  # This will be shared across all processes
    
    def is_seen(self, image_path):
        """Check if an image is seen."""
        return self.seen_images.get(image_path, False)

    def remove_image(self, image_path, bucket):
        """
        Used by other classes to reliably remove images from a bucket.

        Args:
            image_path (str): The path to the image to remove.
            bucket (str): The bucket to remove the image from.

        Returns:
            dict: The aspect ratio bucket indices.
        """
        if image_path in self.aspect_ratio_bucket_indices[bucket]:
            self.aspect_ratio_bucket_indices[bucket].remove(image_path)

    def handle_incorrect_bucket(self, image_path: str, bucket: str, actual_bucket: str):
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
            logger.warning(f"Moved image to bucket, it already existed.")
            self.aspect_ratio_bucket_indices[actual_bucket].append(image_path)
        else:
            logger.warning(f"Created new bucket for that pesky image.")
            self.aspect_ratio_bucket_indices[actual_bucket] = [image_path]

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
                self.data_backend.remove(image_path)
            except Exception as e:
                logger.debug(
                    f"Image {image_path} was already deleted. Another GPU must have gotten to it."
                )
        else:
            logger.warning(
                f"Image {image_path} too small, but --delete_unwanted_images is not provided, so we simply ignore and remove from bucket."
            )
        self.remove_image(image_path, bucket)
