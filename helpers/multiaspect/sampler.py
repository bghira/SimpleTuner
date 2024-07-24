import torch
import logging
import random
import os
from helpers.training.multi_process import rank_info
from helpers.metadata.backends.base import MetadataBackend
from helpers.image_manipulation.training_sample import TrainingSample
from helpers.multiaspect.image import MultiaspectImage
from helpers.multiaspect.state import BucketStateManager
from helpers.data_backend.base import BaseDataBackend
from helpers.training.state_tracker import StateTracker
from helpers.training.exceptions import MultiDatasetExhausted
from helpers.prompts import PromptHandler
from accelerate.logging import get_logger

pil_logger = logging.getLogger("PIL.Image")
pil_logger.setLevel(logging.WARNING)
pil_logger = logging.getLogger("PIL.PngImagePlugin")
pil_logger.setLevel(logging.WARNING)
pil_logger = logging.getLogger("PIL.TiffImagePlugin")
pil_logger.setLevel(logging.WARNING)


class MultiAspectSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        id: str,
        metadata_backend: MetadataBackend,
        data_backend: BaseDataBackend,
        accelerator,
        batch_size: int,
        debug_aspect_buckets: bool = False,
        delete_unwanted_images: bool = False,
        minimum_image_size: int = None,
        resolution: int = 1024,
        resolution_type: str = "pixel",
        caption_strategy: str = "filename",
        use_captions=True,
        prepend_instance_prompt=False,
        instance_prompt: str = None,
    ):
        """
        Initializes the sampler with provided settings.
        Parameters:
        - id: An identifier to link this with its VAECache and DataBackend objects.
        - metadata_backend: An initialised instance of MetadataBackend.
        - batch_size: Number of samples to draw per batch.
        - state_path: Path to store the current state of the sampler.
        - debug_aspect_buckets: Flag to log state for debugging purposes.
        - delete_unwanted_images: Flag to decide whether to delete unwanted (small) images or just remove from the bucket.
        - minimum_image_size: The minimum pixel length of the smallest side of an image.
        """
        self.id = id
        if self.id != data_backend.id or self.id != metadata_backend.id:
            raise ValueError(
                f"Sampler ID ({self.id}) must match DataBackend ID ({data_backend.id}) and MetadataBackend ID ({metadata_backend.id})."
            )
        # Update the logger name with the id:
        self.logger = get_logger(
            f"MultiAspectSampler-{self.id}",
            os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"),
        )
        self.rank_info = rank_info()
        self.accelerator = accelerator
        self.metadata_backend = metadata_backend
        self.data_backend = data_backend
        self.current_bucket = None
        self.current_epoch = 1
        self.batch_size = batch_size
        if debug_aspect_buckets:
            self.logger.setLevel(logging.DEBUG)
        self.delete_unwanted_images = delete_unwanted_images
        self.minimum_image_size = minimum_image_size
        self.resolution = resolution
        self.resolution_type = resolution_type
        self.use_captions = use_captions
        self.caption_strategy = caption_strategy
        self.prepend_instance_prompt = prepend_instance_prompt
        self.instance_prompt = instance_prompt
        self.exhausted_buckets = []
        self.buckets = self.load_buckets()
        self.state_manager = BucketStateManager(self.id)
        self.metadata_backend.refresh_buckets()

    def save_state(self, state_path: str):
        """
        This method should be called when the accelerator save hook is called,
         so that the state is correctly restored with a given checkpoint.
        """
        state = {
            "aspect_ratio_bucket_indices": self.metadata_backend.aspect_ratio_bucket_indices,
            "buckets": self.buckets,
            "exhausted_buckets": self.exhausted_buckets,
            "batch_size": self.batch_size,
            "current_bucket": self.current_bucket,
            "seen_images": self.metadata_backend.seen_images,
            "current_epoch": self.current_epoch,
        }
        self.state_manager.save_state(state, state_path)

    def load_states(self, state_path: str):
        try:
            self.buckets = self.load_buckets()
            previous_state = self.state_manager.load_state(state_path)
        except Exception as e:
            raise e
        self.exhausted_buckets = []
        if "exhausted_buckets" in previous_state:
            self.logger.info(
                f"Previous checkpoint had {len(previous_state['exhausted_buckets'])} exhausted buckets."
            )
            self.exhausted_buckets = previous_state["exhausted_buckets"]
        self.current_epoch = 1
        if "current_epoch" in previous_state:
            self.logger.info(
                f"Previous checkpoint was on epoch {previous_state['current_epoch']}."
            )
            self.current_epoch = previous_state["current_epoch"]
        # Merge seen_images into self.state_manager.seen_images Manager.dict:
        if "seen_images" in previous_state:
            self.logger.info(
                f"Previous checkpoint had {len(previous_state['seen_images'])} seen images."
            )
            self.metadata_backend.seen_images.update(previous_state["seen_images"])

    def load_buckets(self):
        return list(
            self.metadata_backend.aspect_ratio_bucket_indices.keys()
        )  # These keys are a float value, eg. 1.78.

    def retrieve_validation_set(self, batch_size: int):
        """
        Return random images from the set. They should be paired with their caption.

        Args:
            batch_size (int): Number of images to return.
        Returns:
            list: a list of tuples(validation_shortname, validation_prompt, validation_sample)
        """
        results = (
            []
        )  # [tuple(validation_shortname, validation_prompt, validation_sample)]
        for img_idx in range(batch_size):
            image_path = self._yield_random_image()
            image_data = self.data_backend.read_image(image_path)
            image_metadata = self.metadata_backend.get_metadata_by_filepath(image_path)
            training_sample = TrainingSample(
                image=image_data,
                data_backend_id=self.id,
                image_metadata=image_metadata,
                image_path=image_path,
            )
            training_sample.prepare()
            validation_shortname = f"{self.id}_{img_idx}"
            validation_prompt = PromptHandler.magic_prompt(
                sampler_backend_id=self.id,
                data_backend=self.data_backend,
                image_path=image_path,
                caption_strategy=self.caption_strategy,
                use_captions=self.use_captions,
                prepend_instance_prompt=self.prepend_instance_prompt,
                instance_prompt=self.instance_prompt,
            )
            if type(validation_prompt) == list:
                validation_prompt = random.choice(validation_prompt)
                self.debug_log(
                    f"Selecting random prompt from list: {validation_prompt}"
                )
            results.append(
                (validation_shortname, validation_prompt, training_sample.image)
            )

        return results

    def _yield_random_image(self):
        bucket = random.choice(self.buckets)
        image_path = random.choice(
            self.metadata_backend.aspect_ratio_bucket_indices[bucket]
        )
        return image_path

    def yield_single_image(self, filepath: str):
        """
        Yield a single image from the dataset by path.

        If the path prefix isn't in the path, we'll add it.
        """
        if self.metadata_backend.instance_data_root not in filepath and not filepath.startswith("http"):
            filepath = os.path.join(self.metadata_backend.instance_data_root, filepath)
        image_data = self.data_backend.read_image(filepath)
        return image_data

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
            len(self.metadata_backend.seen_images) == 0
            and len(self._get_unseen_images()) == 0
        ):
            raise Exception(
                f"No images found in the dataset: {self.metadata_backend.aspect_ratio_bucket_indices}"
                f"\n-> Unseen images: {self._get_unseen_images()}"
                f"\n-> Seen images: {self.metadata_backend.seen_images}"
            )
        if StateTracker.get_args().print_sampler_statistics:
            self.logger.info(
                "Resetting seen image list and refreshing buckets. State before reset:"
            )
            self.log_state()
        # All buckets are exhausted, so we will move onto the next epoch.
        self.current_epoch += 1
        self.exhausted_buckets = []
        self.buckets = self.load_buckets()
        self.metadata_backend.reset_seen_images()
        self.change_bucket()
        raise MultiDatasetExhausted()

    def _get_unseen_images(self, bucket=None):
        """
        Get unseen images from the specified bucket.
        If bucket is None, get unseen images from all buckets.
        """
        if bucket and bucket in self.metadata_backend.aspect_ratio_bucket_indices:
            return [
                os.path.join(self.metadata_backend.instance_data_root, image) if not image.startswith("http") else image
                for image in self.metadata_backend.aspect_ratio_bucket_indices[bucket]
                if not self.metadata_backend.is_seen(image)
            ]
        elif bucket is None:
            unseen_images = []
            for b, images in self.metadata_backend.aspect_ratio_bucket_indices.items():
                unseen_images.extend(
                    [
                        os.path.join(self.metadata_backend.instance_data_root, image) if not image.startswith("http") else image
                        for image in images
                        if not self.metadata_backend.is_seen(image)
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
            len(self.metadata_backend.aspect_ratio_bucket_indices[bucket])
            < self.batch_size
        ):
            self.debug_log(
                f"Bucket {bucket} has insufficient ({len(self.metadata_backend.aspect_ratio_bucket_indices[bucket])}) images."
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
            f"Bucket {bucket} has sufficient ({len(self.metadata_backend.aspect_ratio_bucket_indices[bucket])}) images."
        )
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
            # Raise MultiDatasetExhausted
            self._reset_buckets()

        if len(self.exhausted_buckets) > 0:
            self.debug_log(f"exhausted buckets: {self.exhausted_buckets}")

        # Sequentially get the next bucket
        if hasattr(self, "current_bucket") and self.current_bucket is not None:
            self.current_bucket = (self.current_bucket + 1) % len(available_buckets)
        else:
            self.current_bucket = 0
        if self.buckets[self.current_bucket] not in available_buckets:
            random_bucket = random.choice(available_buckets)
            self.current_bucket = available_buckets.index(random_bucket)

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

    def move_to_exhausted(self):
        bucket = self.buckets[self.current_bucket]
        self.exhausted_buckets.append(bucket)
        self.buckets.remove(bucket)
        self.debug_log(
            f"Bucket {bucket} is empty or doesn't have enough samples for a full batch. Removing from bucket list. {len(self.buckets)} remain."
        )

    def log_state(self, show_rank: bool = True, alt_stats: bool = False):
        self.debug_log(
            f'Active Buckets: {", ".join(self.convert_to_human_readable(float(b), self.metadata_backend.aspect_ratio_bucket_indices[b], self.resolution) for b in self.buckets)}'
        )
        self.debug_log(
            f'Exhausted Buckets: {", ".join(self.convert_to_human_readable(float(b), self.metadata_backend.aspect_ratio_bucket_indices.get(b, "N/A"), self.resolution) for b in self.exhausted_buckets)}'
        )
        if alt_stats:
            # Return an overview instead of a snapshot.
            # Eg. return totals, and not "as it is now"
            total_image_count = len(self.metadata_backend.seen_images) + len(
                self._get_unseen_images()
            )
            if self.accelerator.num_processes > 1:
                # We don't know the direct count without more work, so we'll estimate it here for multi-GPU training.
                total_image_count *= self.accelerator.num_processes
                total_image_count = f"~{total_image_count}"
            printed_state = (
                f"- Repeats: {StateTracker.get_data_backend_config(self.id).get('repeats', 0)}\n"
                f"- Total number of images: {total_image_count}\n"
                f"- Total number of aspect buckets: {len(self.buckets)}\n"
                f"- Resolution: {self.resolution} {'megapixels' if self.resolution_type == 'area' else 'px'}\n"
                f"- Cropped: {StateTracker.get_data_backend_config(self.id).get('crop')}\n"
                f"- Crop style: {'None' if not StateTracker.get_data_backend_config(self.id).get('crop') else StateTracker.get_data_backend_config(self.id).get('crop_style')}\n"
                f"- Crop aspect: {'None' if not StateTracker.get_data_backend_config(self.id).get('crop') else StateTracker.get_data_backend_config(self.id).get('crop_aspect')}\n"
            )
        else:
            # Return a snapshot of the current state during training.
            printed_state = (
                f"\n{self.rank_info if show_rank else ''}    -> Number of seen images: {len(self.metadata_backend.seen_images)}"
                f"\n{self.rank_info if show_rank else ''}    -> Number of unseen images: {len(self._get_unseen_images())}"
                f"\n{self.rank_info if show_rank else ''}    -> Current Bucket: {self.current_bucket}"
                f"\n{self.rank_info if show_rank else ''}    -> {len(self.buckets)} Buckets: {self.buckets}"
                f"\n{self.rank_info if show_rank else ''}    -> {len(self.exhausted_buckets)} Exhausted Buckets: {self.exhausted_buckets}"
            )
        self.logger.info(printed_state)

        return printed_state

    def _validate_and_yield_images_from_samples(self, samples, bucket):
        """
        Validate and yield images from given samples. Return a list of valid image paths.
        """
        to_yield = []
        for image_path in samples:
            image_metadata = self.metadata_backend.get_metadata_by_filepath(image_path)
            if image_metadata is None:
                image_metadata = {}
            if (
                StateTracker.get_args().model_type
                not in [
                    "legacy",
                    "deepfloyd-full",
                    "deepfloyd-lora",
                    "deepfloyd-stage2",
                    "deepfloyd-stage2-lora",
                ]
                and "crop_coordinates" not in image_metadata
            ):
                raise Exception(
                    f"An image was discovered ({image_path}) that did not have its metadata: {self.metadata_backend.get_metadata_by_filepath(image_path)}"
                )
            image_metadata["data_backend_id"] = self.id
            image_metadata["image_path"] = image_path

            # Use the magic prompt handler to retrieve the captions.
            instance_prompt = PromptHandler.magic_prompt(
                sampler_backend_id=self.id,
                data_backend=self.data_backend,
                image_path=image_metadata["image_path"],
                caption_strategy=self.caption_strategy,
                use_captions=self.use_captions,
                prepend_instance_prompt=self.prepend_instance_prompt,
                instance_prompt=self.instance_prompt,
            )
            if type(instance_prompt) == list:
                instance_prompt = random.choice(instance_prompt)
                self.debug_log(f"Selecting random prompt from list: {instance_prompt}")
            image_metadata["instance_prompt_text"] = instance_prompt

            to_yield.append(image_metadata)
        return to_yield

    def _clear_batch_accumulator(self):
        self.batch_accumulator = []

    def get_conditioning_sample(self, original_sample_path: str) -> str:
        """
        Given an original dataset sample path, return a TrainingSample
        """
        # strip leading /
        original_sample_path = original_sample_path.lstrip("/")
        full_path = os.path.join(
            self.metadata_backend.instance_data_root, original_sample_path
        )
        conditioning_sample = TrainingSample(
            image=self.data_backend.read_image(full_path),
            data_backend_id=self.id,
            image_metadata=self.metadata_backend.get_metadata_by_filepath(full_path),
            image_path=full_path,
        )
        return conditioning_sample

    def connect_conditioning_samples(self, samples: tuple):
        if not StateTracker.get_args().controlnet:
            return samples
        # Locate the conditioning data
        conditioning_dataset = StateTracker.get_conditioning_dataset(self.id)
        sampler = conditioning_dataset["sampler"]
        outputs = list(samples)
        for sample in samples:
            sample_path = sample["image_path"].split(
                self.metadata_backend.instance_data_root
            )[-1]
            conditioning_sample = sampler.get_conditioning_sample(sample_path)
            outputs.append(conditioning_sample)
        return tuple(outputs)

    def __iter__(self):
        """
        Iterate over the sampler to yield image paths in batches.
        """
        self._clear_batch_accumulator()  # Initialize an empty list to accumulate images for a batch
        self.change_bucket()
        while True:
            all_buckets_exhausted = True  # Initial assumption

            # Loop through all buckets to find one with sufficient images
            for _ in range(len(self.buckets)):
                self._clear_batch_accumulator()
                available_images = self._get_unseen_images(
                    self.buckets[self.current_bucket]
                )
                self.debug_log(
                    f"From {len(self.buckets)} buckets, selected {self.buckets[self.current_bucket]} ({self.buckets[self.current_bucket]}) -> {len(available_images)} available images, and our accumulator has {len(self.batch_accumulator)} images ready for yielding."
                )
                if len(available_images) >= self.batch_size:
                    all_buckets_exhausted = False  # Found a non-exhausted bucket
                    break
                else:
                    # Current bucket doesn't have enough images, try the next bucket
                    self.move_to_exhausted()
                    self.change_bucket()
            while len(available_images) >= self.batch_size:
                all_buckets_exhausted = False  # Found a non-exhausted bucket
                samples = random.sample(available_images, k=self.batch_size)
                to_yield = self._validate_and_yield_images_from_samples(
                    samples, self.buckets[self.current_bucket]
                )
                self.debug_log(
                    f"Building batch with {len(self.batch_accumulator)} samples."
                )
                if len(self.batch_accumulator) < self.batch_size:
                    remaining_entries_needed = self.batch_size - len(
                        self.batch_accumulator
                    )
                    # Now we'll add only remaining_entries_needed amount to the accumulator:
                    if "target_size" in to_yield[0]:
                        self.debug_log(
                            f"Current bucket: {self.current_bucket}. Adding samples with aspect ratios: {[MultiaspectImage.calculate_image_aspect_ratio(i['target_size']) for i in to_yield[:remaining_entries_needed]]}"
                        )
                    self.batch_accumulator.extend(to_yield[:remaining_entries_needed])
                # If the batch is full, yield it
                if len(self.batch_accumulator) >= self.batch_size:
                    final_yield = self.batch_accumulator[: self.batch_size]
                    self.debug_log(
                        f"Yielding samples and marking {len(final_yield)} images as seen, we have {len(self.metadata_backend.seen_images.values())} seen images before adding."
                    )
                    self.metadata_backend.mark_batch_as_seen(
                        [instance["image_path"] for instance in final_yield]
                    )
                    self.accelerator.wait_for_everyone()
                    final_yield = self.connect_conditioning_samples(final_yield)
                    yield tuple(final_yield)
                    # Change bucket after a full batch is yielded
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
                self.logger.warning(
                    "All buckets exhausted - since this is happening now, most likely you have chronically-underfilled buckets."
                )
                # Resetting buckets raises MultiDatasetExhausted
                self._reset_buckets()

    def __len__(self):
        backend_config = StateTracker.get_data_backend_config(self.id)
        repeats = backend_config.get("repeats", 0)
        # We need at least a multiplier of 1. Repeats is the number of extra sample steps.
        multiplier = 1
        if repeats > 0:
            multiplier = repeats + 1
        return (
            sum(
                len(indices)
                for indices in self.metadata_backend.aspect_ratio_bucket_indices.values()
            )
            * multiplier
        )

    @staticmethod
    def convert_to_human_readable(
        aspect_ratio_float: float, bucket: iter, resolution: int = 1024
    ):

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
        self.logger.debug(f"{self.rank_info} {msg}", main_process_only=False)
