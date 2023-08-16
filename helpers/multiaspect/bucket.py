from helpers.multiaspect.image import MultiaspectImage
from pathlib import Path
import json, logging, os, multiprocessing
from tqdm import tqdm
from multiprocessing import Process, Queue
import numpy as np

logger = logging.getLogger('BucketManager')

class BucketManager:
    def __init__(self, instance_data_root, cache_file):
        self.instance_data_root = Path(instance_data_root)
        self.cache_file = Path(cache_file)
        self.aspect_ratio_bucket_indices = {}
        self.instance_images_path = set()
        self._load_cache()

    def _discover_new_files(self):
        """Identify files that haven't been processed yet."""
        all_files = {str(f) for f in self.instance_data_root.rglob("*.[jJpP][pPnN][gG]")}
        return list(all_files - self.instance_images_path)

    def _load_cache(self):
        """Load cache data from file."""
        if self.cache_file.exists():
            with self.cache_file.open("r") as f:
                cache_data = json.load(f)
                self.aspect_ratio_bucket_indices = cache_data.get("aspect_ratio_bucket_indices", {})
                self.instance_images_path = set(cache_data.get("instance_images_path", []))

    def _save_cache(self):
        """Save cache data to file."""
        cache_data = {
            "aspect_ratio_bucket_indices": self.aspect_ratio_bucket_indices,
            "instance_images_path": list(self.instance_images_path)
        }
        with self.cache_file.open("w") as f:
            json.dump(cache_data, f)

    def _bucket_worker(self, tqdm_queue, files, aspect_ratio_bucket_indices_queue):
        for file in files:
            aspect_ratio_bucket_indices = MultiaspectImage.process_for_bucket(
                file, self.aspect_ratio_bucket_indices
            )
            tqdm_queue.put(1)
            aspect_ratio_bucket_indices_queue.put(aspect_ratio_bucket_indices)

    def compute_aspect_ratio_bucket_indices(self):
        logger.info("Discovering new files...")
        new_files = self._discover_new_files()

        if not new_files:
            logger.info("No new files discovered. Exiting.")
            return

        num_cpus = multiprocessing.cpu_count()
        files_split = np.array_split(new_files, num_cpus)

        tqdm_queue = Queue()
        aspect_ratio_bucket_indices_queue = Queue()

        logger.info(f"Processing {len(new_files)} new files across {num_cpus} CPUs...")
        workers = [
            Process(
                target=self._bucket_worker,
                args=(tqdm_queue, file_shard, aspect_ratio_bucket_indices_queue)
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
                    aspect_ratio_bucket_indices_update = aspect_ratio_bucket_indices_queue.get()
                    for key, value in aspect_ratio_bucket_indices_update.items():
                        self.aspect_ratio_bucket_indices.setdefault(key, []).extend(value)

        for worker in workers:
            worker.join()

        self.instance_images_path.update(new_files)
        self._save_cache()

        logger.info('Completed aspect bucket update.')

    def remove_image(self, image_path, bucket):
        if image_path in self.aspect_ratio_bucket_indices[bucket]:
            self.aspect_ratio_bucket_indices[bucket].remove(image_path)

    def handle_incorrect_bucket(self, image_path, bucket, actual_bucket):
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

    def handle_small_image(self, image_path, bucket, delete_unwanted_images):
        if delete_unwanted_images:
            try:
                logger.warning(f"Image too small: DELETING image and continuing search.")
                os.remove(image_path)
            except Exception as e:
                logger.warning(
                    f"The image was already deleted. Another GPU must have gotten to it."
                )
        else:
            logger.warning(f"Image too small, but --delete_unwanted_images is not provided, so we simply ignore and remove from bucket.")
        self.remove_image(image_path, bucket)