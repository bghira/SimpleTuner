import torch, logging, random, time

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

pil_logger = logging.getLogger("PIL.Image")
pil_logger.setLevel(logging.WARNING)
pil_logger = logging.getLogger("PIL.PngImagePlugin")
pil_logger.setLevel(logging.WARNING)

from PIL import Image
from .state_tracker import StateTracker
import os, json
from PIL.ImageOps import exif_transpose

from concurrent.futures import ThreadPoolExecutor
import threading


class BalancedBucketSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        aspect_ratio_bucket_indices: dict,
        batch_size: int = 15,
        seen_images_path: str = "/notebooks/SimpleTuner/seen_images.json",
        state_path: str = "/notebooks/SimpleTuner/bucket_sampler_state.json",
        reset_threshold: int = 5000,
        debug_aspect_buckets: bool = False,
        delete_unwanted_images: bool = False,
        minimum_image_size: int = None,
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
        self.current_bucket = 0
        self.minimum_image_size = minimum_image_size
        self.seen_images = self.load_seen_images()

    def save_state(self):
        state = {
            "aspect_ratio_bucket_indices": self.aspect_ratio_bucket_indices,
            "buckets": self.buckets,
            "exhausted_buckets": self.exhausted_buckets,
            "batch_size": self.batch_size,
            "current_bucket": self.current_bucket,
            "seen_images": self.seen_images,
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f)
        self.save_seen_images()
        self.log_state()

    def load_buckets(self):
        return list(
            self.aspect_ratio_bucket_indices.keys()
        )  # These keys are a float value, eg. 1.78.

    def load_seen_images(self):
        if os.path.exists(self.seen_images_path):
            with open(self.seen_images_path, "r") as f:
                seen_images = json.load(f)
        else:
            seen_images = {}
        return seen_images

    def save_seen_images(self):
        with open(self.seen_images_path, "w") as f:
            json.dump(self.seen_images, f)

    def remove_image(self, image_path, bucket):
        if image_path in self.aspect_ratio_bucket_indices[bucket]:
            self.aspect_ratio_bucket_indices[bucket].remove(image_path)

    def handle_small_image(self, image_path, bucket):
        if self.delete_unwanted_images:
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

    def handle_incorrect_bucket(self, image_path, bucket, actual_bucket):
        logger.warning(
            f"Found an image in a bucket {bucket} it doesn't belong in, when actually it is: {actual_bucket}"
        )
        self.remove_image(image_path, bucket)
        if actual_bucket in self.aspect_ratio_bucket_indices:
            logger.warning(f"Moved image to bucket, it already existed.")
            self.aspect_ratio_bucket_indices[actual_bucket].append(image_path)
        else:
            # Create a new bucket if it doesn't exist
            logger.warning(f"Created new bucket for that pesky image.")
            self.aspect_ratio_bucket_indices[actual_bucket] = [image_path]

    def __iter__(self):
        """
        Iterate over the sampler to yield image paths. If the system is in training mode, yield batches of unseen images.
        If not in training mode, yield random images. If the number of unseen images in a bucket is less than the batch size,
        yield all unseen images. If the number of seen images reaches the reset threshold, reset all buckets and seen images.
        """
        while True:
            if self.debug_aspect_buckets:
                logger.debug(f"Running __iter__ for AspectBuckets.")
            if not StateTracker.status_training():
                if self.debug_aspect_buckets:
                    logger.debug(f"Skipping Aspect bucket logic, not training yet.")
                # Yield random image:
                bucket = random.choice(self.buckets)
                image_path = random.choice(self.aspect_ratio_bucket_indices[bucket])
                yield image_path
                continue
            if not self.buckets:
                logger.warning(f"All buckets are exhausted. Resetting...")
                self.buckets = self.load_buckets()

            bucket = self.buckets[self.current_bucket]

            if (
                len(self.buckets) > 1
                and len(self.aspect_ratio_bucket_indices[bucket]) < self.batch_size
            ):
                if bucket not in self.exhausted_buckets:
                    self.move_to_exhausted()
                self.change_bucket()
                continue

            if len(self.seen_images) >= self.reset_threshold:
                self.buckets = self.load_buckets()
                self.seen_images = {}
                self.current_bucket = random.randint(0, len(self.buckets) - 1)
                logger.info(
                    "Reset buckets and seen images because the number of seen images reached the reset threshold."
                )

            available_images = [
                image
                for image in self.aspect_ratio_bucket_indices[bucket]
                if image not in self.seen_images
            ]
            # Pad the safety number so that we can ensure we have a large enough bucket to yield samples from.
            if len(self.buckets) > 1 and len(available_images) < self.batch_size:
                logger.warning(
                    f"Not enough unseen images ({len(available_images)}) in the bucket: {bucket}"
                )
                self.move_to_exhausted()
                self.change_bucket()
                continue
            if (len(available_images) < self.batch_size) and (len(self.buckets) == 1):
                # We have to check if we have enough 'seen' images, and bring them back.
                all_bucket_images = self.aspect_ratio_bucket_indices[bucket]
                total = (
                    len(self.seen_images)
                    + len(available_images)
                    + len(all_bucket_images)
                )
                if total < self.batch_size:
                    logger.warning(
                        f"Not enough unseen images ({len(available_images)}) in the bucket: {bucket}! Overly-repeating training images."
                    )
                    self.seen_images = {}
                else:
                    self.log_state()
                    logger.warning(
                        "Cannot continue. There are not enough images to form a single batch:\n"
                        f"    -> Seen images: {len(self.seen_images)}\n"
                        f"    -> Unseen images: {len(available_images)}\n"
                        f"    -> Buckets: {self.buckets}\n"
                        f"    -> Batch size: {self.batch_size}\n"
                        f"    -> Image list: {available_images}\n"
                    )
                    self.buckets = self.load_buckets()
                    self.seen_images = {}
                    continue

            samples = random.choices(available_images, k=self.batch_size)
            to_yield = []
            for image_path in samples:
                if not os.path.exists(image_path):
                    logger.warning(f"Image path does not exist: {image_path}")
                    self.remove_image(image_path, bucket)
                    continue
                try:
                    if self.debug_aspect_buckets:
                        logger.debug(f"AspectBucket is loading image: {image_path}")
                    image = Image.open(image_path)
                except:
                    logger.warning(f"Image was bad or in-progress: {image_path}")
                    continue
                if (
                    image.width < self.minimum_image_size
                    or image.height < self.minimum_image_size
                ):
                    image.close()
                    self.handle_small_image(image_path, bucket)
                    continue
                image = exif_transpose(image)
                aspect_ratio = round(image.width / image.height, 2)
                actual_bucket = str(aspect_ratio)
                if actual_bucket != bucket:
                    self.handle_incorrect_bucket(image_path, bucket, actual_bucket)
                else:
                    if self.debug_aspect_buckets:
                        logger.debug(
                            f"Yielding {image.width}x{image.height} sample from bucket: {bucket} with aspect {actual_bucket}"
                        )
                    to_yield.append(image_path)
                    try:
                        image.close()
                    except:
                        pass
                    if StateTracker.status_training():
                        self.seen_images[image_path] = actual_bucket
                if self.debug_aspect_buckets:
                    logger.debug(
                        f"Completed internal for loop inside __iter__ for AspectBuckets."
                    )

            if len(to_yield) == self.batch_size:
                # Select a random bucket:
                self.current_bucket = random.randint(
                    0, len(self.buckets) - 1
                )  # This is a random integer.
                for image_to_yield in to_yield:
                    if self.debug_aspect_buckets:
                        logger.debug(f"Yielding from __iter__ for AspectBuckets.")
                    yield image_to_yield

    def __len__(self):
        return sum(
            len(indices) for indices in self.aspect_ratio_bucket_indices.values()
        )

    def change_bucket(self):
        """
        Change the current bucket. If the current bucket doesn't have enough samples, move it to the exhausted list
        and select a new bucket randomly. If there's only one bucket left, reset the exhausted list and seen images.
        """
        # Do we just have a single bucket?
        if len(self.buckets) == 1:
            logger.debug(f"Changing bucket to the only one present.")
            self.current_bucket = 0
            return
        if self.buckets:
            old_bucket = self.current_bucket
            self.current_bucket = random.randint(0, len(self.buckets) - 1)
            if old_bucket != self.current_bucket:
                logger.info(f"Changing bucket to {self.buckets[self.current_bucket]}.")
                return
            if len(self.buckets) == 1:
                logger.debug(f"Changing bucket to the only one present.")
                return
            logger.warning(
                f"Only one bucket left, and it doesn't have enough samples. Resetting..."
            )
            logger.warning(
                f'Exhausted buckets: {", ".join(self.convert_to_human_readable(float(b), self.aspect_ratio_bucket_indices[b]) for b in self.exhausted_buckets)}'
            )
            self.exhausted_buckets = []
            self.seen_images = {}
            self.current_bucket = random.randint(0, len(self.buckets) - 1)
            logger.info(
                f"After resetting, changed bucket to {self.buckets[self.current_bucket]}."
            )

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
            f'Active Buckets: {", ".join(self.convert_to_human_readable(float(b), self.aspect_ratio_bucket_indices[b]) for b in self.buckets)}'
        )
        logger.debug(
            f'Exhausted Buckets: {", ".join(self.convert_to_human_readable(float(b), self.aspect_ratio_bucket_indices[b]) for b in self.exhausted_buckets)}'
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
