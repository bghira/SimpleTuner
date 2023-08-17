from helpers.multiaspect.image import MultiaspectImage
from pathlib import Path
import json, logging, os, multiprocessing
from tqdm import tqdm
from multiprocessing import Process, Queue
import numpy as np

logger = logging.getLogger("BucketManager")


class BucketManager:
    def __init__(self, instance_data_root, cache_file):
        self.instance_data_root = Path(instance_data_root)
        self.cache_file = Path(cache_file)
        self.aspect_ratio_bucket_indices = {}
        self.instance_images_path = set()
        self._load_cache()

    def __len__(self):
        return sum(
            [len(bucket) for bucket in self.aspect_ratio_bucket_indices.values()]
        )

    def _rglob_follow_symlinks(self, path: Path, pattern: str):
        """
        A custom implementation of rglob that efficiently follows symlinks.

        Args:
            path (Path): The path to search.
            pattern (str): The pattern to match.

        Yields:
            Path: The next path that matches the pattern.
        """
        for p in path.glob(pattern):
            yield p
        for p in path.iterdir():
            if p.is_dir() and not p.is_symlink():
                yield from self._rglob_follow_symlinks(p, pattern)
            elif p.is_symlink():
                real_path = Path(os.readlink(p))
                if real_path.is_dir():
                    yield from self._rglob_follow_symlinks(real_path, pattern)

    def _discover_new_files(self):
        """
        Discover new files that have not been processed yet.
        
        Returns:
            list: A list of new files.
        """
        all_image_files = list(
            self._rglob_follow_symlinks(self.instance_data_root, "*.[jJpP][pPnN][gG]")
        )
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
        if self.cache_file.exists():
            with self.cache_file.open("r") as f:
                cache_data = json.load(f)
                self.aspect_ratio_bucket_indices = cache_data.get(
                    "aspect_ratio_bucket_indices", {}
                )
                self.instance_images_path = set(
                    cache_data.get("instance_images_path", [])
                )

    def _save_cache(self):
        """
        Save cache data to file.
        """
        cache_data = {
            "aspect_ratio_bucket_indices": self.aspect_ratio_bucket_indices,
            "instance_images_path": list(self.instance_images_path),
        }
        with self.cache_file.open("w") as f:
            json.dump(cache_data, f)

    def _bucket_worker(
        self, tqdm_queue, files, aspect_ratio_bucket_indices_queue, existing_files_set
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
                    file, local_aspect_ratio_bucket_indices
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

    def handle_small_image(self, image_path: str, bucket: str, delete_unwanted_images: bool):
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
                    f"Image too small: DELETING image and continuing search."
                )
                os.remove(image_path)
            except Exception as e:
                logger.warning(
                    f"The image was already deleted. Another GPU must have gotten to it."
                )
        else:
            logger.warning(
                f"Image too small, but --delete_unwanted_images is not provided, so we simply ignore and remove from bucket."
            )
        self.remove_image(image_path, bucket)
