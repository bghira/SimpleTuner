import torch
import logging
import random
import os
from simpletuner.helpers.training.multi_process import rank_info
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.multiaspect.image import MultiaspectImage
from simpletuner.helpers.multiaspect.state import BucketStateManager
from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from simpletuner.helpers.prompts import PromptHandler
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
        model,
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
        conditioning_type: str = None,
        is_regularisation_data: bool = False,
        dataset_type: str = "image",
        source_dataset_id: str = None,
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
        self.model = model
        # Update the logger name with the id:
        self.dataset_type = dataset_type
        self.sample_type_str = "image"
        self.sample_type_strs = "images"
        if dataset_type == "video":
            self.sample_type_str = "video"
            self.sample_type_strs = "videos"
        self.logger = get_logger(
            f"MultiAspectSampler-{self.id}",
            os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"),
        )

        self._val_cursor = 0
        self._val_master_list = []
        self.conditioning_type = conditioning_type
        self.is_regularisation_data = is_regularisation_data

        self.rank_info = rank_info()
        self.accelerator = accelerator
        self.metadata_backend = metadata_backend
        if conditioning_type is not None:
            if conditioning_type not in [
                "controlnet",
                "mask",
                "segmentation",
                "reference_strict",
                "reference_loose",
            ]:
                raise ValueError(
                    f"Unknown conditioning image type: {conditioning_type}"
                )
        self.data_backend = data_backend
        self.source_dataset_id = source_dataset_id
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
        self._val_master_list = sorted(
            sum(self.metadata_backend.aspect_ratio_bucket_indices.values(), [])
        )

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
                f"Previous checkpoint had {len(previous_state['seen_images'])} seen {self.sample_type_strs}."
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
        results = []
        seen_paths = set()

        # Don't try to get more images than we have
        available_count = len(self._val_master_list)
        actual_batch_size = min(batch_size, available_count)

        if actual_batch_size < batch_size:
            self.logger.warning(
                f"Requested {batch_size} validation images but only {available_count} available. "
                f"Returning {actual_batch_size} unique images."
            )

        for img_idx in range(actual_batch_size):
            image_path = self._yield_sequential_image()

            # Skip if we've already seen this path (in case of wraparound)
            if image_path in seen_paths:
                continue

            seen_paths.add(image_path)

            image_data = self.data_backend.read_image(image_path)
            image_metadata = self.metadata_backend.get_metadata_by_filepath(image_path)
            training_sample = TrainingSample(
                image=image_data,
                data_backend_id=self.id,
                image_metadata=image_metadata,
                image_path=image_path,
                model=self.model,
            )
            training_sample.prepare()
            validation_shortname = f"{self.id}_{img_idx}"
            # Use the magic prompt handler to retrieve the captions.
            prompt_kwargs = {
                "caption_strategy": self.caption_strategy,
                "instance_prompt": self.instance_prompt,
                "data_backend": self.data_backend,
                "sampler_backend_id": self.id,
                "prepend_instance_prompt": self.prepend_instance_prompt,
                "use_captions": self.use_captions,
                "image_path": image_path,
            }
            if self.source_dataset_id is not None:
                # we'll retrieve captions from the source dataset.
                training_sample_path = training_sample.training_sample_path(
                    training_dataset_id=self.source_dataset_id
                )
                source_dataset: "MultiAspectSampler" = StateTracker.get_data_backend(
                    self.source_dataset_id
                )["sampler"]
                prompt_kwargs.update(
                    {
                        "caption_strategy": source_dataset.caption_strategy,
                        "instance_prompt": source_dataset.instance_prompt,
                        "data_backend": source_dataset.data_backend,
                        "sampler_backend_id": source_dataset.id,
                        "prepend_instance_prompt": source_dataset.prepend_instance_prompt,
                        "use_captions": source_dataset.use_captions,
                        "image_path": training_sample_path,
                    }
                )
            self.logger.debug(f"Using prompt kwargs: {prompt_kwargs}")
            validation_prompt = PromptHandler.magic_prompt(**prompt_kwargs)
            if type(validation_prompt) == list:
                self.debug_log(
                    f"Selecting random prompt from list: {validation_prompt}"
                )
                validation_prompt = random.choice(validation_prompt)
            results.append(
                (
                    validation_shortname,
                    validation_prompt,
                    image_path,
                    training_sample.image,
                )
            )

        return results

    def _yield_n_from_exhausted_bucket(self, n: int, bucket: str):
        """
        when a bucket is exhausted, and we have to populate the remainder of the batch,
        we shall use this quick and dirty method to retrieve n samples from the exhausted bucket.
        the thing is we can have a batch size of 4 and 1 image. so we'll have to just return the same image 4 times.
        """
        available_images = self.metadata_backend.aspect_ratio_bucket_indices[bucket]
        if len(available_images) == 0:
            self.debug_log(f"Bucket {bucket} is empty.")
            return []
        samples = []
        while len(samples) < n:
            to_grab = min(n, len(available_images), (n - len(samples)))
            if to_grab == 0:
                break
            samples.extend(random.sample(available_images, k=to_grab))

        to_yield = self._validate_and_yield_images_from_samples(samples, bucket)
        return to_yield

    def _yield_sequential_image(self):
        """
        Always return the *next* image in a fixed list, wrapping around when
        we reach the end.  Nothing is random, so the same N calls → same N
        paths every run.
        """
        if self._val_cursor >= len(self._val_master_list):
            self._val_cursor = 0
            if len(self._val_master_list) == 0:
                raise MultiDatasetExhausted(
                    "No validation images available. Please check your dataset."
                )
        path = self._val_master_list[self._val_cursor]
        self._val_cursor = (self._val_cursor + 1) % len(self._val_master_list)
        return path

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
        if (
            self.metadata_backend.instance_data_dir is not None
            and self.metadata_backend.instance_data_dir not in filepath
            and not filepath.startswith("http")
        ):
            filepath = os.path.join(self.metadata_backend.instance_data_dir, filepath)
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

    def _reset_buckets(self, raise_exhaustion_signal: bool = True):
        if (
            len(self.metadata_backend.seen_images) == 0
            and len(self._get_unseen_images()) == 0
        ):
            raise Exception(
                f"No images found in the dataset: {self.metadata_backend.aspect_ratio_bucket_indices}"
                f"\n-> Unseen {self.sample_type_strs}: {self._get_unseen_images()}"
                f"\n-> Seen {self.sample_type_strs}: {self.metadata_backend.seen_images}"
            )
        if StateTracker.get_args().print_sampler_statistics:
            self.logger.info(
                f"Resetting seen {self.sample_type_str} list and refreshing buckets. State before reset:"
            )
            self.log_state()
        # All buckets are exhausted, so we will move onto the next epoch.
        self.current_epoch += 1
        self.exhausted_buckets = []
        self.buckets = self.load_buckets()
        self.metadata_backend.reset_seen_images()
        self.change_bucket()
        if raise_exhaustion_signal:
            raise MultiDatasetExhausted()

    def _get_unseen_images(self, bucket=None):
        """
        Get unseen {self.sample_type_strs} from the specified bucket.
        If bucket is None, get unseen {self.sample_type_strs} from all buckets.
        """
        if bucket and bucket in self.metadata_backend.aspect_ratio_bucket_indices:
            return [
                (
                    os.path.join(self.metadata_backend.instance_data_dir, image)
                    if not image.startswith("http")
                    else image
                )
                for image in self.metadata_backend.aspect_ratio_bucket_indices[bucket]
                if not self.metadata_backend.is_seen(image)
            ]
        elif bucket is None:
            unseen_images = []
            for b, images in self.metadata_backend.aspect_ratio_bucket_indices.items():
                unseen_images.extend(
                    [
                        (
                            os.path.join(self.metadata_backend.instance_data_dir, image)
                            if not image.startswith("http")
                            else image
                        )
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
        If all buckets are exhausted, first reset the seen {self.sample_type_strs} and exhausted buckets.
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
        During _get_next_bucket(), if all buckets are exhausted, reset the exhausted list and seen {self.sample_type_strs}.
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
            data_backend_config = StateTracker.get_data_backend_config(self.id)
            printed_state = (
                f"- Repeats: {data_backend_config.get('repeats', 0)}\n"
                f"- Total number of images: {total_image_count}\n"
                f"- Total number of aspect buckets: {len(self.buckets)}\n"
                f"- Resolution: {self.resolution} {'megapixels' if self.resolution_type == 'area' else 'px'}\n"
                f"- Cropped: {data_backend_config.get('crop')}\n"
                f"- Crop style: {'None' if not data_backend_config.get('crop') else data_backend_config.get('crop_style')}\n"
                f"- Crop aspect: {'None' if not data_backend_config.get('crop') else data_backend_config.get('crop_aspect')}\n"
                f"- Used for regularisation data: {'Yes' if self.is_regularisation_data else 'No'}\n"
            )
            if self.conditioning_type:
                printed_state += f"- Conditioning type: {self.conditioning_type}\n"
        else:
            # Return a snapshot of the current state during training.
            printed_state = (
                f"\n{self.rank_info if show_rank else ''}    -> Number of seen {self.sample_type_strs}: {len(self.metadata_backend.seen_images)}"
                f"\n{self.rank_info if show_rank else ''}    -> Number of unseen {self.sample_type_strs}: {len(self._get_unseen_images())}"
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
                StateTracker.get_args().model_family
                not in [
                    "sd1x",
                    "sd2x",
                    "deepfloyd",
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

    def get_conditioning_sample(
        self, original_sample_path: str
    ) -> TrainingSample | None:
        """
        Given an original dataset sample path, return a TrainingSample
        """
        # strip leading /
        original_sample_path = original_sample_path.lstrip("/")
        if self.metadata_backend.instance_data_dir not in original_sample_path:
            full_path = os.path.join(
                self.metadata_backend.instance_data_dir, original_sample_path
            )
        else:
            full_path = original_sample_path
        try:
            conditioning_sample_data = self.data_backend.read_image(full_path)
        except Exception as e:
            self.logger.error(f"Could not fetch conditioning sample: {e}")
            return None
        if not conditioning_sample_data:
            self.debug_log(f"Could not fetch conditioning sample from {full_path}.")
            return None

        conditioning_sample_metadata = self.metadata_backend.get_metadata_by_filepath(
            full_path
        )
        conditioning_sample = TrainingSample(
            image=conditioning_sample_data,
            data_backend_id=self.id,
            image_metadata=conditioning_sample_metadata,
            image_path=full_path,
            conditioning_type=self.conditioning_type,
            model=self.model,
        )
        # Use the magic prompt handler to retrieve the captions.
        prompt_kwargs = {
            "caption_strategy": self.caption_strategy,
            "instance_prompt": self.instance_prompt,
            "data_backend": self.data_backend,
            "sampler_backend_id": self.id,
            "prepend_instance_prompt": self.prepend_instance_prompt,
            "use_captions": self.use_captions,
            "image_path": full_path,
        }
        if self.source_dataset_id is not None and self.caption_strategy is None:
            # we'll retrieve captions from the source dataset.
            training_sample_path = conditioning_sample.training_sample_path(
                training_dataset_id=self.source_dataset_id
            )
            source_dataset: "MultiAspectSampler" = StateTracker.get_data_backend(
                self.source_dataset_id
            )["sampler"]
            prompt_kwargs.update(
                {
                    "caption_strategy": source_dataset.caption_strategy,
                    "instance_prompt": source_dataset.instance_prompt,
                    "data_backend": source_dataset.data_backend,
                    "sampler_backend_id": source_dataset.id,
                    "prepend_instance_prompt": source_dataset.prepend_instance_prompt,
                    "use_captions": source_dataset.use_captions,
                    "image_path": training_sample_path,
                }
            )
        self.logger.debug(f"Using prompt kwargs: {prompt_kwargs}")
        instance_prompt = PromptHandler.magic_prompt(**prompt_kwargs)
        if type(instance_prompt) == list:
            instance_prompt = random.choice(instance_prompt)
            self.debug_log(f"Selecting random prompt from list: {instance_prompt}")
        conditioning_sample.set_caption(instance_prompt)

        return conditioning_sample

    def connect_conditioning_samples(self, samples: tuple):
        # Locate the conditioning data
        conditioning_datasets = StateTracker.get_conditioning_datasets(self.id)
        if not len(conditioning_datasets):
            return samples

        # Get sampling mode from args
        sampling_mode = getattr(
            StateTracker.get_args(), "conditioning_multidataset_sampling", "combined"
        )

        # Override to combined mode for validation to ensure consistency
        is_validation = getattr(self, "_is_validation_pass", False)
        if is_validation and sampling_mode == "random":
            self.debug_log("Using 'combined' mode for validation consistency")
            sampling_mode = "combined"

        outputs = list(samples)

        if sampling_mode == "random" and len(conditioning_datasets) > 1:
            # Random mode: select one conditioning dataset per training sample
            for sample in samples:
                sample_path: str = sample["image_path"]
                if (
                    self.metadata_backend.instance_data_dir is not None
                    and self.metadata_backend.instance_data_dir != ""
                ):
                    sample_path = sample_path.split(
                        self.metadata_backend.instance_data_dir
                    )[-1]

                # Deterministic selection based on hash of image path and current epoch
                # This ensures the same image gets the same conditioning dataset within an epoch
                # but different datasets across epochs
                path_hash = hash(f"{sample_path}_{self.current_epoch}")
                selected_idx = path_hash % len(conditioning_datasets)
                selected_dataset = conditioning_datasets[selected_idx]
                sampler = selected_dataset["sampler"]

                self.debug_log(
                    f"Selected conditioning dataset {selected_dataset['id']} "
                    f"for sample {sample_path} (epoch {self.current_epoch})"
                )

                # Get the conditioning sample from the selected dataset
                conditioning_sample = sampler.get_conditioning_sample(sample_path)
                if conditioning_sample is not None:
                    # Tag the sample with which dataset it came from for debugging
                    conditioning_sample._source_dataset_id = selected_dataset["id"]
                    outputs.append(conditioning_sample)
                else:
                    # Try other datasets if the selected one doesn't have this sample
                    found = False
                    for fallback_idx, fallback_dataset in enumerate(
                        conditioning_datasets
                    ):
                        if fallback_idx == selected_idx:
                            continue  # Skip the one we already tried

                        fallback_sampler = fallback_dataset["sampler"]
                        conditioning_sample = fallback_sampler.get_conditioning_sample(
                            sample_path
                        )
                        if conditioning_sample is not None:
                            self.logger.warning(
                                f"Sample {sample_path} not found in {selected_dataset['id']}, "
                                f"using fallback from {fallback_dataset['id']}"
                            )
                            conditioning_sample._source_dataset_id = fallback_dataset[
                                "id"
                            ]
                            outputs.append(conditioning_sample)
                            found = True
                            break

                    if not found:
                        self.logger.error(
                            f"Could not find conditioning sample for {sample_path} "
                            f"in any conditioning dataset"
                        )

        elif sampling_mode == "combined" or len(conditioning_datasets) == 1:
            # Combined mode: append conditioning samples from all datasets (current behavior)
            missing_samples = []

            for dataset in conditioning_datasets:
                sampler = dataset["sampler"]
                dataset_samples_added = 0

                for sample in samples:
                    sample_path: str = sample["image_path"]
                    if (
                        self.metadata_backend.instance_data_dir is not None
                        and self.metadata_backend.instance_data_dir != ""
                    ):
                        sample_path = sample_path.split(
                            self.metadata_backend.instance_data_dir
                        )[-1]

                    conditioning_sample = sampler.get_conditioning_sample(sample_path)
                    if conditioning_sample is not None:
                        # Tag the sample with which dataset it came from
                        conditioning_sample._source_dataset_id = dataset["id"]
                        outputs.append(conditioning_sample)
                        dataset_samples_added += 1
                    else:
                        missing_samples.append((sample_path, dataset["id"]))

                self.debug_log(
                    f"Added {dataset_samples_added}/{len(samples)} conditioning samples "
                    f"from dataset {dataset['id']}"
                )

            # Report missing samples in batch if there are any
            if missing_samples:
                self.logger.warning(
                    f"Missing {len(missing_samples)} conditioning samples: "
                    f"{missing_samples[:3]}{'...' if len(missing_samples) > 3 else ''}"
                )

        else:
            raise ValueError(
                f"Unknown conditioning_multidataset_sampling mode: {sampling_mode}. "
                "Must be 'random' or 'combined'."
            )

        # Validate output count
        expected_conditioning_count = len(samples) * (
            1 if sampling_mode == "random" else len(conditioning_datasets)
        )
        actual_conditioning_count = len(outputs) - len(samples)

        if actual_conditioning_count != expected_conditioning_count:
            self.logger.warning(
                f"Expected {expected_conditioning_count} conditioning samples but got "
                f"{actual_conditioning_count} (mode: {sampling_mode}, "
                f"datasets: {len(conditioning_datasets)})"
            )

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
                if len(available_images) > 0:
                    all_buckets_exhausted = False  # Found a non-exhausted bucket
                    break
                else:
                    # Current bucket doesn't have enough images, try the next bucket
                    self.move_to_exhausted()
                    self.change_bucket()
            while len(available_images) > 0:
                if len(available_images) < self.batch_size:
                    need_image_count = self.batch_size - len(available_images)
                    self.debug_log(
                        f"Bucket {self.buckets[self.current_bucket]} has {len(available_images)} available images, but we need {need_image_count} more."
                    )
                    to_yield = self._yield_n_from_exhausted_bucket(
                        need_image_count, self.buckets[self.current_bucket]
                    )
                    # add the available images
                    to_yield.extend(
                        self._validate_and_yield_images_from_samples(
                            available_images, self.buckets[self.current_bucket]
                        )
                    )
                else:
                    all_buckets_exhausted = False  # Found a non-exhausted bucket
                    samples = random.sample(
                        available_images, k=min(len(available_images), self.batch_size)
                    )
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
                        f"Yielding samples and marking {len(final_yield)} images as seen, we have {len(self.metadata_backend.seen_images.values())} seen {self.sample_type_strs} before adding."
                    )
                    self.metadata_backend.mark_batch_as_seen(
                        [instance["image_path"] for instance in final_yield]
                    )
                    # if applicable, we'll append TrainingSample(s) to the end for conditioning inputs.
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
                # If all buckets are exhausted, reset the seen {self.sample_type_strs} and refresh buckets
                self.logger.warning(
                    "All buckets exhausted - since this is happening now, most likely you have chronically-underfilled buckets."
                )
                # Resetting buckets raises MultiDatasetExhausted
                self._reset_buckets()

    def __len__(self):
        backend_config = StateTracker.get_data_backend_config(self.id)
        repeats = backend_config.get("repeats", 0)
        # We need at least a multiplier of 1. Repeats is the number of extra sample steps.
        multiplier = repeats + 1 if repeats > 0 else 1

        total_samples = (
            sum(
                len(indices)
                for indices in self.metadata_backend.aspect_ratio_bucket_indices.values()
            )
            * multiplier
        )

        # Calculate the total number of full batches
        total_batches = (total_samples + (self.batch_size - 1)) // self.batch_size

        return total_batches

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
