import torch, logging, random, time

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

pil_logger = logging.getLogger('PIL.Image')
pil_logger.setLevel(logging.WARNING)
pil_logger = logging.getLogger('PIL.PngImagePlugin')
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
        aspect_ratio_bucket_indices,
        batch_size: int = 15,
        seen_images_path: str = "/notebooks/SimpleTuner/seen_images.json",
        state_path: str = "/notebooks/SimpleTuner/bucket_sampler_state.json",
        drop_caption_every_n_percent: int = 5,
        debug_aspect_buckets: bool = False,
    ):
        self.aspect_ratio_bucket_indices = aspect_ratio_bucket_indices  # A dictionary of string float buckets and their image paths.
        self.buckets = self.load_buckets()
        self.exhausted_buckets = (
            []
        )  # Buckets that have been exhausted, eg. all samples have been used.
        self.batch_size = batch_size  # How many images per sample during training. They MUST all be the same aspect.
        self.current_bucket = 0
        self.seen_images_path = seen_images_path
        self.state_path = state_path
        self.seen_images = self.load_seen_images()
        self.drop_caption_every_n_percent = drop_caption_every_n_percent
        self.debug_aspect_buckets = debug_aspect_buckets
        self.caption_loop_count = (
            0  # Store a value and increment on each sample until we hit 100.
        )

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
        logger.warning(f"Image too small: DELETING image and continuing search.")
        try:
            os.remove(image_path)
        except Exception as e:
            logger.warning(
                f"The image was already deleted. Another GPU must have gotten to it."
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

            if len(self.aspect_ratio_bucket_indices[bucket]) < self.batch_size:
                if bucket not in self.exhausted_buckets:
                    self.move_to_exhausted()
                self.change_bucket()
                continue

            available_images = [
                image
                for image in self.aspect_ratio_bucket_indices[bucket]
                if image not in self.seen_images
            ]
            # Pad the safety number so that we can ensure we have a large enough bucket to yield samples from.
            if len(available_images) < (self.batch_size * 2):
                logger.warning(f"Not enough unseen images in the bucket: {bucket}")
                self.move_to_exhausted()
                self.change_bucket()
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
                if image.width < 880 or image.height < 880:
                    image.close()
                    self.handle_small_image(image_path, bucket)
                    continue
                image = exif_transpose(image)
                aspect_ratio = round(image.width / image.height, 3)
                actual_bucket = str(aspect_ratio)
                if actual_bucket != bucket:
                    self.handle_incorrect_bucket(image_path, bucket, actual_bucket)
                else:
                    if self.debug_aspect_buckets:
                        logger.debug(
                            f"Yielding {image.width}x{image.height} sample from bucket: {bucket} with aspect {actual_bucket}"
                        )
                    to_yield.append(image_path)
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
        if self.buckets:
            old_bucket = self.current_bucket
            self.current_bucket %= len(self.buckets)
            if old_bucket != self.current_bucket:
                logger.info(f"Changing bucket to {self.buckets[self.current_bucket]}.")
                return
            if len(self.buckets) == 1:
                logger.debug(f'Changing bucket to the only one present.')
                return
            logger.warning(
                f"Only one bucket left, and it doesn't have enough samples. Resetting..."
            )
            logger.warning(
                f'Exhausted buckets: {", ".join(self.convert_to_human_readable(float(b), self.aspect_ratio_bucket_indices[b]) for b in self.exhausted_buckets)}'
            )
            self.exhausted_buckets = []
            self.seen_images = {}
            self.current_bucket %= len(self.buckets)
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
        # logger.debug(
        #     f'Exhausted Buckets: {", ".join(self.convert_to_human_readable(float(b), self.aspect_ratio_bucket_indices[b]) for b in self.exhausted_buckets)}'
        # )

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
