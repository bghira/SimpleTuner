import torch, logging, json, random, os
from io import BytesIO
from PIL import Image
from PIL.ImageOps import exif_transpose
from helpers.training.multi_process import rank_info
from helpers.multiaspect.bucket import BucketManager
from helpers.multiaspect.state import BucketStateManager
from helpers.data_backend.base import BaseDataBackend
from helpers.training.state_tracker import StateTracker
from accelerate.logging import get_logger

logger = get_logger(
    "MultiAspectSampler", os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING")
)

pil_logger = logging.getLogger("PIL.Image")
pil_logger.setLevel(logging.WARNING)
pil_logger = logging.getLogger("PIL.PngImagePlugin")
pil_logger.setLevel(logging.WARNING)
pil_logger = logging.getLogger("PIL.TiffImagePlugin")
pil_logger.setLevel(logging.WARNING)


class MultiAspectSampler(torch.utils.data.Sampler):
    current_epoch = 1
    vae_cache = None

    def __init__(
        self,
        bucket_manager: BucketManager,
        data_backend: BaseDataBackend,
        accelerator,
        batch_size: int,
        seen_images_path: str,
        state_path: str,
        debug_aspect_buckets: bool = False,
        delete_unwanted_images: bool = False,
        minimum_image_size: int = None,
        resolution: int = 1024,
        resolution_type: str = "pixel",
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
        self.rank_info = rank_info()
        self.accelerator = accelerator
        self.bucket_manager = bucket_manager
        self.data_backend = data_backend
        self.current_bucket = None
        self.batch_size = batch_size
        self.seen_images_path = seen_images_path
        self.state_path = state_path
        if debug_aspect_buckets:
            logger.setLevel(logging.DEBUG)
        self.delete_unwanted_images = delete_unwanted_images
        self.minimum_image_size = minimum_image_size
        self.resolution = resolution
        self.resolution_type = resolution_type
        self.load_states(
            state_path=state_path,
        )
        self.change_bucket()

    def save_state(self, state_path: str = None):
        """
        This method should be called when the accelerator save hook is called,
         so that the state is correctly restored with a given checkpoint.
        """
        state = {
            "aspect_ratio_bucket_indices": self.bucket_manager.aspect_ratio_bucket_indices,
            "buckets": self.buckets,
            "exhausted_buckets": self.exhausted_buckets,
            "batch_size": self.batch_size,
            "current_bucket": self.current_bucket,
            "seen_images": self.bucket_manager.seen_images,
            "current_epoch": self.current_epoch,
        }
        self.state_manager.save_state(state, state_path)

    def load_states(self, state_path: str):
        try:
            self.state_manager = BucketStateManager(state_path, self.seen_images_path)
            self.buckets = self.load_buckets()
            previous_state = self.state_manager.load_state()
        except Exception as e:
            raise e
        self.exhausted_buckets = []
        if "exhausted_buckets" in previous_state:
            self.exhausted_buckets = previous_state["exhausted_buckets"]
        self.current_epoch = 1
        if "current_epoch" in previous_state:
            self.current_epoch = previous_state["current_epoch"]
        # Merge seen_images into self.state_manager.seen_images Manager.dict:
        if "seen_images" in previous_state:
            self.bucket_manager.seen_images.update(previous_state["seen_images"])

    def load_buckets(self):
        return list(
            self.bucket_manager.aspect_ratio_bucket_indices.keys()
        )  # These keys are a float value, eg. 1.78.

    def retrieve_vae_cache(self):
        if self.vae_cache is None:
            self.vae_cache = StateTracker.get_vaecache()
        return self.vae_cache

    def _yield_random_image(self):
        bucket = random.choice(self.buckets)
        image_path = random.choice(
            self.bucket_manager.aspect_ratio_bucket_indices[bucket]
        )
        return image_path

    def _bucket_name_to_id(self, bucket_name: str) -> int:
        """
        Return a bucket array index, by its name.

        Args:
            bucket_name (str): Bucket name, eg. "1.78"
        Returns:
            int: Bucket array index, eg. 0
        """
        if "." not in str(bucket_name):
            self.debug_log(f"Assuming {bucket_name} is already an index.")
            return int(bucket_name)
        return self.buckets.index(str(bucket_name))

    def _reset_buckets(self):
        if (
            len(self.bucket_manager.seen_images) == 0
            and len(self._get_unseen_images()) == 0
        ):
            raise Exception(
                f"No images found in the dataset: {self.bucket_manager.aspect_ratio_bucket_indices}"
                f"\n-> Unseen images: {self._get_unseen_images()}"
                f"\n-> Seen images: {self.bucket_manager.seen_images}"
            )
        logger.info(
            f"Resetting seen image list and refreshing buckets. State before reset:"
        )
        self.log_state()
        # All buckets are exhausted, so we will move onto the next epoch.
        self.current_epoch += 1
        self.exhausted_buckets = []
        self.buckets = self.load_buckets()
        self.bucket_manager.reset_seen_images()
        self.change_bucket()

    def _get_unseen_images(self, bucket=None):
        """
        Get unseen images from the specified bucket.
        If bucket is None, get unseen images from all buckets.
        """
        if bucket and bucket in self.bucket_manager.aspect_ratio_bucket_indices:
            return [
                image
                for image in self.bucket_manager.aspect_ratio_bucket_indices[bucket]
                if not self.bucket_manager.is_seen(image)
            ]
        elif bucket is None:
            unseen_images = []
            for b, images in self.bucket_manager.aspect_ratio_bucket_indices.items():
                unseen_images.extend(
                    [
                        image
                        for image in images
                        if not self.bucket_manager.is_seen(image)
                    ]
                )
            return unseen_images
        else:
            return []

    def _handle_bucket_with_insufficient_images(self, bucket):
        """
        Handle buckets with insufficient images. Return True if we changed or reset the bucket.
        """
        if (
            len(self.bucket_manager.aspect_ratio_bucket_indices[bucket])
            < self.batch_size
        ):
            self.debug_log(
                f"Bucket {bucket} has insufficient ({len(self.bucket_manager.aspect_ratio_bucket_indices[bucket])}) images."
            )
            if bucket not in self.exhausted_buckets:
                self.debug_log(
                    f"Bucket {bucket} is now exhausted and sleepy, and we have to move it to the sleepy list before changing buckets."
                )
                self.move_to_exhausted()
            self.debug_log("Changing bucket to another random selection.")
            self.change_bucket()
            return True
        self.debug_log(
            f"Bucket {bucket} has sufficient ({len(self.bucket_manager.aspect_ratio_bucket_indices[bucket])}) images."
        )
        return False

    def _reset_if_not_enough_unseen_images(self):
        """
        Reset the seen images if there aren't enough unseen images across all buckets to form a batch.
        Return True if we reset the seen images, otherwise return False.
        This is distinctly separate behaviour from change_bucket, which resets based on exhausted buckets.
        """
        total_unseen_images = len(self._get_unseen_images())
        if total_unseen_images < self.batch_size:
            logger.warning(
                f"_reset_if_not_enough_unseen_images: total_unseen_images={total_unseen_images}, batch_size={self.batch_size} triggered reset"
            )
            self._reset_buckets()
            return True
        return False

    def _get_next_bucket(self):
        """
        Get the next bucket excluding the exhausted ones.
        If all buckets are exhausted, first reset the seen images and exhausted buckets.
        """
        available_buckets = [
            bucket for bucket in self.buckets if bucket not in self.exhausted_buckets
        ]
        if not available_buckets:
            logger.warning(
                f"_get_next_bucket: all {len(self.buckets)} buckets are exhausted"
                f" ({len(self.exhausted_buckets)}), resetting"
            )
            self._reset_buckets()
            available_buckets = self.buckets

        self.debug_log(
            f"Selecting next bucket from {len(available_buckets)} possible choices (truncated): {available_buckets[:10]}"
        )
        self.debug_log(f"exhausted buckets: {self.exhausted_buckets}")
        self.debug_log(f"bucket list: {self.buckets}")

        # Sequentially get the next bucket
        if hasattr(self, "current_bucket") and self.current_bucket is not None:
            self.current_bucket = (self.current_bucket + 1) % len(available_buckets)
        else:
            self.current_bucket = 0
        if self.buckets[self.current_bucket] not in available_buckets:
            logger.warning(
                f"Tried to select bucket {self.buckets[self.current_bucket]} from this list of buckets, which do not contain it: {available_buckets}. Setting bucket instead, to {available_buckets[0]}"
            )
            self.current_bucket = 0

        next_bucket = available_buckets[self.current_bucket]
        return next_bucket

    def change_bucket(self):
        """
        Change the current bucket to a new one and exclude exhausted buckets from consideration.
        During _get_next_bucket(), if all buckets are exhausted, reset the exhausted list and seen images.
        """
        next_bucket = self._get_next_bucket()
        self.current_bucket = self._bucket_name_to_id(next_bucket)
        self._clear_batch_accumulator()
        self.debug_log(
            f"Changed bucket to {next_bucket} ({self.buckets[self.current_bucket]})."
        )

    def move_to_exhausted(self):
        bucket = self.buckets[self.current_bucket]
        self.exhausted_buckets.append(bucket)
        self.buckets.remove(bucket)
        self.debug_log(
            f"Bucket {bucket} is empty or doesn't have enough samples for a full batch. Removing from bucket list. {len(self.buckets)} remain."
        )

    def log_state(self):
        self.debug_log(
            f'Active Buckets: {", ".join(self.convert_to_human_readable(float(b), self.bucket_manager.aspect_ratio_bucket_indices[b], self.resolution) for b in self.buckets)}'
        )
        self.debug_log(
            f'Exhausted Buckets: {", ".join(self.convert_to_human_readable(float(b), self.bucket_manager.aspect_ratio_bucket_indices.get(b, "N/A"), self.resolution) for b in self.exhausted_buckets)}'
        )
        logger.info(
            f"{self.rank_info}Multi-aspect sampler statistics:\n"
            f"{self.rank_info}    -> Batch size: {self.batch_size}\n"
            f"{self.rank_info}    -> Seen images: {len(self.bucket_manager.seen_images)}\n"
            f"{self.rank_info}    -> Unseen images: {len(self._get_unseen_images())}\n"
            f"{self.rank_info}    -> Current Bucket: {self.current_bucket}\n"
            f"{self.rank_info}    -> {len(self.buckets)} Buckets: {self.buckets}\n"
            f"{self.rank_info}    -> {len(self.exhausted_buckets)} Exhausted Buckets: {self.exhausted_buckets}\n"
        )

    def _validate_and_yield_images_from_samples(self, samples, bucket):
        """
        Validate and yield images from given samples. Return a list of valid image paths.
        """
        to_yield = []
        for image_path in samples:
            self.debug_log(
                f"Begin analysing sample. We have {len(to_yield)} images to yield."
            )
            crop_coordinates = self.bucket_manager.get_metadata_attribute_by_filepath(
                image_path, "crop_coordinates"
            )
            if crop_coordinates is None:
                raise Exception(
                    f"An image was discovered ({image_path}) that did not have its metadata: {self.bucket_manager.get_metadata_by_filepath(image_path)}"
                )
            self.debug_log(
                f"Image {image_path} is considered valid. Adding to yield list."
            )
            to_yield.append({"image_path": image_path})
            self.debug_log(
                f"Completed analysing sample. We have {len(to_yield)} images to yield."
            )
        return to_yield

    def _clear_batch_accumulator(self):
        self.batch_accumulator = []

    def __iter__(self):
        """
        Iterate over the sampler to yield image paths in batches.
        """
        self._clear_batch_accumulator()  # Initialize an empty list to accumulate images for a batch

        while True:
            all_buckets_exhausted = True  # Initial assumption

            # Loop through all buckets to find one with sufficient images
            self.debug_log(f"Found {len(self.buckets)} buckets.")
            for _ in range(len(self.buckets)):
                self._clear_batch_accumulator()
                self.debug_log(
                    f"Fetching available images for bucket {self.current_bucket} ({self.buckets[self.current_bucket]})."
                )
                available_images = self._get_unseen_images(
                    self.buckets[self.current_bucket]
                )
                self.debug_log(
                    f"We found {len(available_images)} images in bucket {self.current_bucket} ({self.buckets[self.current_bucket]})."
                )

                if len(available_images) >= self.batch_size:
                    all_buckets_exhausted = False  # Found a non-exhausted bucket
                    break
                else:
                    # Current bucket doesn't have enough images, try the next bucket
                    self.debug_log(
                        f"Bucket {self.buckets[self.current_bucket]} is empty or doesn't have enough samples for a full batch. Moving to the next bucket."
                    )
                    self.move_to_exhausted()
                    self.change_bucket()
            while len(available_images) >= self.batch_size:
                self.debug_log(
                    f"Bucket {self.buckets[self.current_bucket]} has {len(available_images)} available images, and our accumulator has {len(self.batch_accumulator)} images ready for yielding."
                )
                all_buckets_exhausted = False  # Found a non-exhausted bucket
                samples = random.sample(available_images, k=self.batch_size)
                to_yield = self._validate_and_yield_images_from_samples(
                    samples, self.buckets[self.current_bucket]
                )
                self.debug_log(
                    f"After validating and yielding, we have {len(to_yield)} images to yield."
                )
                if len(self.batch_accumulator) < self.batch_size:
                    remaining_entries_needed = self.batch_size - len(
                        self.batch_accumulator
                    )
                    # Now we'll add only remaining_entries_needed amount to the accumulator:
                    self.batch_accumulator.extend(to_yield[:remaining_entries_needed])
                # If the batch is full, yield it
                if len(self.batch_accumulator) >= self.batch_size:
                    self.debug_log(
                        f"We have a full batch of {len(self.batch_accumulator)} images ready for yielding. Now we yield them!"
                    )
                    final_yield = self.batch_accumulator[: self.batch_size]
                    self.debug_log(
                        f"Marking {len(final_yield)} images as seen, we have {len(self.bucket_manager.seen_images.values())} unseen images before adding."
                    )
                    self.bucket_manager.mark_batch_as_seen(
                        [instance["image_path"] for instance in final_yield]
                    )
                    self.accelerator.wait_for_everyone()
                    yield tuple(final_yield)
                    # Change bucket after a full batch is yielded
                    self.debug_log(
                        f"Clearing batch accumulator while changing buckets."
                    )
                    self.change_bucket()
                    # Break out of the while loop:
                    break

                # Update available images after yielding
                available_images = self._get_unseen_images(
                    self.buckets[self.current_bucket]
                )
                self.debug_log(
                    f"Bucket {self.buckets[self.current_bucket]} now has {len(available_images)} available images after yielding."
                )

            # Handle exhausted bucket
            if len(available_images) < self.batch_size:
                self.debug_log(
                    f"Bucket {self.buckets[self.current_bucket]} is now exhausted and sleepy, and we have to move it to the sleepy list before changing buckets."
                )
                self.move_to_exhausted()
                self.change_bucket()

            # Check if all buckets are exhausted
            if all_buckets_exhausted:
                # If all buckets are exhausted, reset the seen images and refresh buckets
                logger.warning(
                    f"All buckets exhausted - since this is happening now, most likely you have chronically-underfilled buckets."
                )
                self._reset_buckets()
                # Exit with nothing, and the epoch is over. Magic.
                return

    def __len__(self):
        return sum(
            len(indices)
            for indices in self.bucket_manager.aspect_ratio_bucket_indices.values()
        )

    @staticmethod
    def convert_to_human_readable(
        aspect_ratio_float: float, bucket: iter, resolution: int = 1024
    ):
        from math import gcd

        if aspect_ratio_float < 1:
            ratio_width = resolution
            ratio_height = int(resolution / aspect_ratio_float)
        else:
            ratio_width = int(resolution * aspect_ratio_float)
            ratio_height = resolution

        # Return the aspect ratio as a string in the format "width:height"
        return f"{aspect_ratio_float} ({len(bucket)} samples)"
        return f"{ratio_width}:{ratio_height}"

    def debug_log(self, msg: str):
        logger.debug(f"{self.rank_info}{msg}", main_process_only=False)
