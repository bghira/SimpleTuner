import torch, logging, json, random, os
from PIL import Image
from PIL.ImageOps import exif_transpose
from helpers.multiaspect.bucket import BucketManager
from helpers.multiaspect.state import BucketStateManager
from helpers.state_tracker import StateTracker

logger = logging.getLogger()
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING"))

pil_logger = logging.getLogger("PIL.Image")
pil_logger.setLevel(logging.WARNING)
pil_logger = logging.getLogger("PIL.PngImagePlugin")
pil_logger.setLevel(logging.WARNING)


class MultiAspectSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        bucket_manager: BucketManager,
        batch_size: int,
        seen_images_path: str,
        state_path: str,
        debug_aspect_buckets: bool = False,
        delete_unwanted_images: bool = False,
        minimum_image_size: int = None,
    ):
        """
        Initializes the sampler with provided settings.
        Parameters:
        - bucket_manager: An initialised instance of BucketManager.
        - batch_size: Number of samples to draw per batch.
        - seen_images_path: Path to store the seen images.
        - state_path: Path to store the current state of the sampler.
        - debug_aspect_buckets: Flag to log state for debugging purposes.
        - delete_unwanted_images: Flag to decide whether to delete unwanted (small) images or just remove from the bucket.
        - minimum_image_size: The minimum pixel length of the smallest side of an image.
        """
        self.bucket_manager = bucket_manager
        self.batch_size = batch_size
        self.seen_images_path = seen_images_path
        self.state_path = state_path
        if debug_aspect_buckets:
            logger.setLevel(logging.DEBUG)
        self.delete_unwanted_images = delete_unwanted_images
        self.minimum_image_size = minimum_image_size
        self.state_manager = BucketStateManager(state_path, seen_images_path)
        self.seen_images = self.state_manager.load_seen_images()
        self.buckets = self.load_buckets()
        self.current_bucket = random.randint(0, len(self.buckets) - 1)
        previous_state = self.state_manager.load_state()
        self.exhausted_buckets = []
        if "exhausted_buckets" in previous_state:
            self.exhausted_buckets = previous_state["exhausted_buckets"]

    def save_state(self):
        state = {
            "aspect_ratio_bucket_indices": self.bucket_manager.aspect_ratio_bucket_indices,
            "buckets": self.buckets,
            "exhausted_buckets": self.exhausted_buckets,
            "batch_size": self.batch_size,
            "current_bucket": self.current_bucket,
            "seen_images": self.seen_images,
        }
        self.state_manager.save_state(state)

    def load_buckets(self):
        return list(
            self.bucket_manager.aspect_ratio_bucket_indices.keys()
        )  # These keys are a float value, eg. 1.78.

    def _yield_random_image(self):
        bucket = random.choice(self.buckets)
        image_path = random.choice(
            self.bucket_manager.aspect_ratio_bucket_indices[bucket]
        )
        return image_path

    def _reset_buckets(self):
        self.buckets = self.load_buckets()
        self.seen_images = {}
        self.current_bucket = random.randint(0, len(self.buckets) - 1)

    def _get_unseen_images(self, bucket=None):
        """
        Get unseen images from the specified bucket.
        If bucket is None, get unseen images from all buckets.
        """
        if bucket:
            return [
                image
                for image in self.bucket_manager.aspect_ratio_bucket_indices[bucket]
                if image not in self.seen_images
            ]
        else:
            unseen_images = []
            for b, images in self.bucket_manager.aspect_ratio_bucket_indices.items():
                unseen_images.extend(
                    [image for image in images if image not in self.seen_images]
                )
            return unseen_images

    def _yield_random_image_if_not_training(self):
        """
        If not in training mode, yield a random image and return True. Otherwise, return False.
        """
        if not StateTracker.status_training():
            return self._yield_random_image()
        return False

    def _handle_bucket_with_insufficient_images(self, bucket):
        """
        Handle buckets with insufficient images. Return True if we changed or reset the bucket.
        """
        if (
            len(self.bucket_manager.aspect_ratio_bucket_indices[bucket])
            < self.batch_size
        ):
            if bucket not in self.exhausted_buckets:
                self.move_to_exhausted()
            self.change_bucket()
            return True
        return False

    def _reset_if_not_enough_unseen_images(self):
        """
        Reset the seen images if there aren't enough unseen images across all buckets to form a batch.
        Return True if we reset the seen images, otherwise return False.
        """
        total_unseen_images = sum(
            len(self._get_unseen_images(bucket)) for bucket in self.buckets
        )

        if total_unseen_images < self.batch_size:
            self.seen_images = {}
            return True
        return False

    def change_bucket(self):
        """
        Change the current bucket. If the current bucket doesn't have enough samples,
        move it to the exhausted list and select a new bucket randomly.
        If all buckets are exhausted, reset the exhausted list and seen images.
        """
        # If we just have a single bucket:
        if len(self.buckets) == 1:
            self.current_bucket = 0
            # If this single bucket is exhausted, reset seen_images and exhausted_buckets
            if not self._get_unseen_images(self.buckets[self.current_bucket]):
                logger.warning(
                    "The only bucket available is exhausted. Resetting seen images."
                )
                self.seen_images = {}
                self.exhausted_buckets = []
            return

        # For multiple buckets:
        old_bucket = self.current_bucket
        while True:  # Keep looking until we find a bucket with unseen images
            self.current_bucket = random.randint(0, len(self.buckets) - 1)
            if self.current_bucket != old_bucket and self._get_unseen_images(
                self.buckets[self.current_bucket]
            ):
                break

        # If all buckets are exhausted
        if not self._get_unseen_images(self.buckets[self.current_bucket]):
            logger.warning("All buckets seem to be exhausted. Resetting...")
            self.exhausted_buckets = []
            self.seen_images = {}
            self.current_bucket = random.randint(0, len(self.buckets) - 1)
        else:
            logger.info(f"Changing bucket to {self.buckets[self.current_bucket]}.")

    def move_to_exhausted(self):
        bucket = self.buckets[self.current_bucket]
        self.exhausted_buckets.append(bucket)
        self.buckets.remove(bucket)
        logger.info(
            f"Bucket {bucket} is empty or doesn't have enough samples for a full batch. Moving to the next bucket."
        )
        self.log_state()

    def log_state(self):
        logger.debug(
            f'Active Buckets: {", ".join(self.convert_to_human_readable(float(b), self.bucket_manager.aspect_ratio_bucket_indices[b]) for b in self.buckets)}'
        )
        logger.debug(
            f'Exhausted Buckets: {", ".join(self.convert_to_human_readable(float(b), self.bucket_manager.aspect_ratio_bucket_indices.get(b, "N/A")) for b in self.exhausted_buckets)}'
        )
        logger.debug(
            "Extended Statistics:\n"
            f"    -> Seen images: {len(self.seen_images)}\n"
            f"    -> Unseen images: {len(self._get_unseen_images())}\n"
            f"    -> Current Bucket: {self.current_bucket}\n"
            f"    -> Buckets: {self.buckets}\n"
            f"    -> Batch size: {self.batch_size}\n"
        )

    def _process_single_image(self, image_path, bucket):
        """
        Validate and process a single image.
        Return the image path if valid, otherwise return None.
        """
        if not os.path.exists(image_path):
            logger.warning(f"Image path does not exist: {image_path}")
            self.bucket_manager.remove_image(image_path, bucket)
            return None

        try:
            logger.debug(f"AspectBucket is loading image: {image_path}")
            with Image.open(image_path) as image:
                if (
                    image.width < self.minimum_image_size
                    or image.height < self.minimum_image_size
                ):
                    image.close()
                    self.bucket_manager.handle_small_image(
                        image_path=image_path,
                        bucket=bucket,
                        delete_unwanted_images=self.delete_unwanted_images,
                    )
                    return None

                image = exif_transpose(image)
                aspect_ratio = round(image.width / image.height, 2)
            actual_bucket = str(aspect_ratio)
            if actual_bucket != bucket:
                self.bucket_manager.handle_incorrect_bucket(
                    image_path, bucket, actual_bucket
                )
                return None

            return image_path
        except:
            logger.warning(f"Image was bad or in-progress: {image_path}")
            return None

    def _validate_and_yield_images_from_samples(self, samples, bucket):
        """
        Validate and yield images from given samples. Return a list of valid image paths.
        """
        to_yield = []
        for image_path in samples:
            processed_image_path = self._process_single_image(image_path, bucket)
            if processed_image_path:
                to_yield.append(processed_image_path)
                if StateTracker.status_training():
                    self.seen_images[processed_image_path] = bucket
        return to_yield

    def __iter__(self):
        """
        Iterate over the sampler to yield image paths.
        - If the system is in training mode, yield batches of unseen images.
        - If not in training mode, yield random images.
        - If the number of unseen images in a bucket is less than the batch size, yield all unseen images.
        - If the number of seen images reaches the reset threshold, reset all buckets and seen images.
        """
        while True:
            logger.debug(f"Running __iter__ for AspectBuckets.")
            early_yield = self._yield_random_image_if_not_training()
            if early_yield:
                yield early_yield
                continue
            if not self.buckets:
                logger.warning(f"All buckets are exhausted. Resetting...")
                self._reset_buckets()

            bucket = self.buckets[self.current_bucket]

            if self._handle_bucket_with_insufficient_images(bucket):
                continue

            available_images = self._get_unseen_images(bucket)

            if len(available_images) < self.batch_size:
                self._reset_if_not_enough_unseen_images()
                self.current_bucket = random.randint(0, len(self.buckets) - 1)
                continue

            samples = random.sample(available_images, k=self.batch_size)
            to_yield = self._validate_and_yield_images_from_samples(samples, bucket)

            if len(to_yield) == self.batch_size:
                # Select a random bucket:
                self.current_bucket = random.randint(0, len(self.buckets) - 1)
                for image_to_yield in to_yield:
                    logger.debug(f"Yielding from __iter__ for AspectBuckets.")
                    yield image_to_yield

    def __len__(self):
        return sum(
            len(indices)
            for indices in self.bucket_manager.aspect_ratio_bucket_indices.values()
        )

    @staticmethod
    def convert_to_human_readable(aspect_ratio_float: float, bucket):
        from math import gcd

        # The smallest side is always 1024. It could be portrait or landscape (eg. under or over 1)
        if aspect_ratio_float < 1:
            ratio_width = 1024
            ratio_height = int(1024 / aspect_ratio_float)
        else:
            ratio_width = int(1024 * aspect_ratio_float)
            ratio_height = 1024

        # Return the aspect ratio as a string in the format "width:height"
        return f"{aspect_ratio_float} (remaining: {len(bucket)})"
        return f"{ratio_width}:{ratio_height}"
