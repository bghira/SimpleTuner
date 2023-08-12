import os
import json
import random
import logging
from helpers.state_tracker import StateTracker
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger("PIL").setLevel(logging.WARNING)


class BalancedBucketSampler(torch.utils.data.Sampler):
    """
    This sampler is designed to pull samples from a dataset with varying aspect ratios in a balanced manner.
    The dataset is divided into different buckets based on aspect ratios.
    It also keeps track of seen images and can reset after a certain threshold to avoid overfitting.
    """

    def __init__(
        self,
        aspect_ratio_bucket_indices: dict,
        batch_size: int = 15,
        seen_images_path: str = "/notebooks/SimpleTuner/seen_images.json",
        state_path: str = "/notebooks/SimpleTuner/bucket_sampler_state.json",
        reset_threshold: int = 5000,
        debug_aspect_buckets: bool = False,
        delete_unwanted_images: bool = False,
    ):
        """
        Initializes the sampler with provided settings.

        Parameters:
        - aspect_ratio_bucket_indices: Dictionary containing aspect ratios as keys and list of image paths as values.
        - batch_size: Number of samples to draw per batch.
        - seen_images_path: Path to store the seen images.
        - state_path: Path to store the current state of the sampler.
        - reset_threshold: The threshold after which the seen images list should be reset.
        - debug_aspect_buckets: Flag to log state for debugging purposes.
        - delete_unwanted_images: Flag to decide whether to delete unwanted (small) images or just remove from the bucket.
        """
        self.aspect_ratio_bucket_indices = aspect_ratio_bucket_indices
        self.buckets = list(self.aspect_ratio_bucket_indices.keys())
        self.exhausted_buckets = []
        self.batch_size = batch_size
        self.seen_images_path = seen_images_path
        self.state_path = state_path
        self.reset_threshold = reset_threshold
        self.debug_aspect_buckets = debug_aspect_buckets
        self.delete_unwanted_images = delete_unwanted_images
        self.seen_images = self._load_json(self.seen_images_path, default={})
        self.current_bucket = 0

    def __len__(self):
        return sum(
            len(indices) for indices in self.aspect_ratio_bucket_indices.values()
        )

    def save_state(self):
        """
        Saves the current state of the sampler to the specified state path.
        """
        with open(self.state_path, "w") as f:
            json.dump(
                {
                    "aspect_ratio_bucket_indices": self.aspect_ratio_bucket_indices,
                    "buckets": self.buckets,
                    "exhausted_buckets": self.exhausted_buckets,
                    "batch_size": self.batch_size,
                    "current_bucket": self.current_bucket,
                    "seen_images": self.seen_images,
                },
                f,
            )
        self._save_json(self.seen_images_path, self.seen_images)
        self.log_state()

    @staticmethod
    def _load_json(path: str, default=None):
        """
        Utility function to load a JSON file from a given path.
        """
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return default

    @staticmethod
    def _save_json(path: str, data):
        """
        Utility function to save data as a JSON file to a given path.
        """
        with open(path, "w") as f:
            json.dump(data, f)

    def _remove_image(self, image_path: str, bucket: str):
        """
        Removes the specified image from the given bucket.
        """
        if image_path in self.aspect_ratio_bucket_indices[bucket]:
            self.aspect_ratio_bucket_indices[bucket].remove(image_path)
        if image_path in self.seen_images:
            del self.seen_images[image_path]

    def _handle_small_image(self, image_path: str, bucket: str):
        """
        Manages images that are smaller than expected.
        Can either remove them from the bucket or delete them based on settings.
        Deleting images is expected to help with disk space, but we do not yet
         tie that into deleting items from the VAE latent or text embed caches.
        """
        self._remove_image(image_path, bucket)
        if not self.delete_unwanted_images:
            logger.warning(
                f"Image too small: Removing image from bucket and continuing search."
            )
            return
        try:
            logger.warning(
                f"Image too small: !!DELETING!! image: {image_path} due to provided option: --delete_unwanted_images"
            )
            os.remove(image_path)
        except FileNotFoundError:
            logger.warning(
                f"The image was already deleted. Another GPU must have gotten to it."
            )

    def _handle_incorrect_bucket(
        self, image_path: str, bucket: str, actual_bucket: str
    ):
        """
        Re-bucket an image if it was found in the wrong aspect ratio bucket.
        This happens because I write terrible code, and there's no one to stop me.
        """
        logger.warning(
            f"Found an image in a bucket {bucket} it doesn't belong in, when actually it is: {actual_bucket}"
        )
        self._remove_image(image_path, bucket)
        self.aspect_ratio_bucket_indices.setdefault(actual_bucket, []).append(
            image_path
        )

    def _iterate_training_mode(self, bucket: str, available_images: list[str]):
        """
        Iterate through available images in training mode.
        Opportunistically clean up when we find missing data or bad images.
        """
        samples = random.choices(available_images, k=self.batch_size)
        for image_path in samples:
            if not os.path.exists(image_path):
                logger.warning(f"Image path does not exist: {image_path}")
                self._remove_image(image_path, bucket)
                continue
            image = self._load_and_validate_image(image_path, bucket)
            if image:
                aspect_ratio = round(image.width / image.height, 3)
                actual_bucket = str(aspect_ratio)
                if actual_bucket != bucket:
                    self._handle_incorrect_bucket(image_path, bucket, actual_bucket)
                else:
                    yield image_path
                    self.seen_images[image_path] = actual_bucket

    def _load_and_validate_image(self, image_path: str, bucket: str):
        """
        Loads the image and checks if it's valid. Returns the image if valid.
        """
        try:
            image = exif_transpose(Image.open(image_path))
            if (
                image.width < self.minimum_image_size
                or image.height < self.minimum_image_size
            ):
                self._handle_small_image(image_path, bucket)
                return None
            return image
        except Exception:
            logger.warning(f"Image was bad or in-progress: {image_path}")
            return None

    def __iter__(self):
        """
        Iterator function to provide samples from the dataset.
        """
        while True:
            # We might not actually be training yet. This happens when we are working through steps
            #  and we have not yet hit resume_step from a previous checkpoint.
            # We want to keep sampling from the same bucket until we hit resume_step, as it's just
            #  the fastest way to resume training.
            # The VAECache is not impacted by skipping the aspect bucket iteration, as it has its own.
            if not StateTracker.status_training():
                bucket = random.choice(self.buckets)
                yield random.choice(self.aspect_ratio_bucket_indices[bucket])
                continue

            # When all image buckets are exhausted, we technically start a new epoch.
            # We reset the buckets and seen images, and pick a random bucket to start with.
            if not self.buckets:
                logger.warning(f"All buckets are exhausted. Resetting...")
                self.buckets = list(self.aspect_ratio_bucket_indices.keys())

            # Ensure current_bucket is in bounds
            self.current_bucket %= len(self.buckets)

            bucket = self.buckets[self.current_bucket]
            available_images = [
                img
                for img in self.aspect_ratio_bucket_indices[bucket]
                if img not in self.seen_images
            ]

            # If available images are less than batch size, we CANNOT sample from it.
            # We could theoretically find a similar aspect ratio, and crop it to the right size.
            # However, with latent caching, it's really not worth the effort. We might not have a VAE,
            #  or enough VRAM on hand to compute the latents during the training loop.
            # It is easier to add a bunch of regularization images to your data pool instead.
            if len(available_images) < self.batch_size:
                logger.warning(
                    f"Bucket {bucket} is exhausted and sleepy.. It has {len(available_images)} images left,"
                    f" but our batch size is {self.batch_size}. We are going to remove the remaining images."
                )
                self._manage_low_images(bucket, available_images)
                continue

            # Reset if seen images exceed threshold
            if len(self.seen_images) >= self.reset_threshold:
                self.buckets = list(self.aspect_ratio_bucket_indices.keys())
                self.seen_images = {}
                self.current_bucket = random.randint(0, len(self.buckets) - 1)
                self.log_state()
                continue

            for img_path in self._iterate_training_mode(bucket, available_images):
                yield img_path

            # Move to the next bucket
            self.current_bucket = (self.current_bucket + 1) % len(self.buckets)

    def _manage_low_images(self, bucket: str, available_images: list[str]):
        """
        Remove the remaining images from a bucket from our 'seen' list,
         as there is no point in sampling from it.
        """
        if available_images:
            for img in available_images:
                self.seen_images[img] = bucket
            self.buckets.remove(bucket)
        elif bucket not in self.exhausted_buckets:
            self.exhausted_buckets.append(bucket)
        if not self.buckets:
            logger.warning(f"All buckets are exhausted. Resetting...")
            self.buckets = list(self.aspect_ratio_bucket_indices.keys())

    def log_state(self):
        """
        Logs the current state of the sampler. Useful for debugging purposes.
        """
        if not self.debug_aspect_buckets:
            return
        logger.info(
            f"Bucket: {self.current_bucket}, Available Buckets: {len(self.buckets)}, Total Buckets: {len(self.aspect_ratio_bucket_indices.keys())}, Seen Images: {len(self.seen_images)}"
        )
