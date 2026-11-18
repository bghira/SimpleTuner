import logging
import os
import random

import numpy as np
import torch
from accelerate.logging import get_logger
from PIL import Image

from simpletuner.helpers.data_backend.cursor import DatasetCursor
from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.metadata.backends.base import get_cp_aware_dp_info
from simpletuner.helpers.multiaspect.state import BucketStateManager
from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from simpletuner.helpers.training.multi_process import broadcast_object_from_main, rank_info
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.utils.pathing import canonicalize_data_uri

pil_logger = logging.getLogger("PIL.Image")
pil_logger.setLevel(logging.WARNING)
pil_logger = logging.getLogger("PIL.PngImagePlugin")
pil_logger.setLevel(logging.WARNING)
pil_logger = logging.getLogger("PIL.TiffImagePlugin")
pil_logger.setLevel(logging.WARNING)


class MultiAspectSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset_runtime: DatasetRuntime,
        model,
        accelerator,
        batch_size: int,
        debug_aspect_buckets: bool = False,
        delete_unwanted_images: bool = False,
        minimum_image_size: int = None,
        resolution: int = 1024,
        resolution_type: str = "pixel",
        conditioning_type: str = None,
        is_regularisation_data: bool = False,
        source_dataset_id: str = None,
    ):
        self.dataset_runtime = dataset_runtime
        self.id = dataset_runtime.id
        self.data_backend = dataset_runtime.data_backend
        self.metadata_backend = dataset_runtime.metadata_backend
        if self.id != self.data_backend.id or self.id != self.metadata_backend.id:
            raise ValueError("Dataset, data backend, and metadata backend IDs must match.")
        dataset_type = dataset_runtime.config["dataset_type"]
        self.model = model
        # Update the logger name with the id:
        self.dataset_type = dataset_type
        self.sample_type_str = "image"
        self.sample_type_strs = "images"
        if dataset_type == "video":
            self.sample_type_str = "video"
            self.sample_type_strs = "videos"
        elif dataset_type == "audio":
            self.sample_type_str = "audio"
            self.sample_type_strs = "audio"
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
        if conditioning_type is not None:
            if conditioning_type not in [
                "controlnet",
                "mask",
                "segmentation",
                "reference_strict",
                "reference_loose",
            ]:
                raise ValueError(f"Unknown conditioning image type: {conditioning_type}")
        self.source_dataset_id = source_dataset_id
        self.batch_size = batch_size
        if debug_aspect_buckets:
            self.logger.setLevel(logging.DEBUG)
        self.delete_unwanted_images = delete_unwanted_images
        self.minimum_image_size = minimum_image_size
        self.resolution = resolution
        self.resolution_type = resolution_type
        self.use_captions = dataset_runtime.config["use_captions"]
        self.caption_strategy = dataset_runtime.config["caption_strategy"]
        self.prepend_instance_prompt = dataset_runtime.config["prepend_instance_prompt"]
        self.instance_prompt = dataset_runtime.config["instance_prompt"]
        self.disable_multiline_split = dataset_runtime.config["disable_multiline_split"]
        self.state_manager = BucketStateManager(self.id)
        self._val_master_list = sorted(sum(self.metadata_backend.aspect_ratio_bucket_indices.values(), []))
        # Every rank must shuffle with the same seed, and drawing one is a collective call,
        # so it happens here, where the ranks construct their samplers in step.
        self._cursor_seed = getattr(StateTracker.get_args(), "seed", None)
        if self._cursor_seed is None:
            is_main_process = self.accelerator.is_main_process
            self._cursor_seed = broadcast_object_from_main(
                random.SystemRandom().getrandbits(64) if is_main_process else None
            )
        self._cursor = None

    @property
    def cursor(self) -> DatasetCursor:
        # Built on first use: the bucket lists are still being filtered when the sampler is constructed.
        if self._cursor is None:
            self._cursor = DatasetCursor(
                self.id,
                self.metadata_backend.aspect_ratio_bucket_indices,
                batch_size=self.batch_size,
                repeats=int(StateTracker.get_data_backend_config(self.id).get("repeats") or 0),
                seed=self._cursor_seed,
                dp_size=1,
                dp_rank=0,
                gradient_accumulation_steps=1,
                allow_oversubscription=False,
                # split_buckets_between_processes has already partitioned and checked these lists.
                is_eval=True,
            )
        return self._cursor

    def rebuild_cursor(self) -> None:
        """Pick up bucket lists that were recomputed between epochs."""
        epoch = self.cursor.epoch
        self._cursor = None
        self.cursor.epoch = epoch

    @property
    def current_epoch(self) -> int:
        return self.cursor.epoch

    def finish_epoch(self) -> None:
        self.cursor.finish_epoch()

    def drop_unfinished_epoch(self) -> None:
        """Start the next epoch if this one was left part-way, as when another rank ended it."""
        if self.cursor.progress()[0] > 0:
            self.cursor.finish_epoch()

    def save_state(self, state_path: str):
        """
        This method should be called when the accelerator save hook is called,
         so that the state is correctly restored with a given checkpoint.
        """
        self.state_manager.save_state(self.cursor.state(), state_path)

    def load_states(self, state_path: str):
        state = self.state_manager.load_state(state_path)
        if not state:
            self.logger.info(f"Dataset '{self.id}' has no sampler state in this checkpoint; it starts a new epoch.")
            return
        allow_batch_size_change = os.environ.get("SIMPLETUNER_ALLOW_MODIFYING_BSZ", "").lower() in (
            "1",
            "true",
            "yes",
        ) or getattr(StateTracker.get_args(), "i_know_what_i_am_doing", False)
        saved_batch_size = state.get("layout", {}).get("batch_size", self.batch_size)
        self.cursor.load_state(state, allow_batch_size_change=bool(allow_batch_size_change))
        if saved_batch_size != self.batch_size:
            self.logger.warning(
                f"Dataset '{self.id}': resuming with batch_size={self.batch_size} but checkpoint recorded "
                f"batch_size={saved_batch_size}. Proceeding because override is set."
            )

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
            image_metadata = self.dataset_runtime.read_samples([image_path])[0]
            training_sample = TrainingSample(
                image=image_data,
                data_backend_id=self.id,
                image_metadata=image_metadata,
                image_path=image_path,
                model=self.model,
            )
            training_sample.prepare()
            validation_shortname = f"{self.id}_{img_idx}"
            caption_dataset = self.dataset_runtime
            caption_path = image_path
            if self.source_dataset_id is not None:
                caption_path = training_sample.training_sample_path(training_dataset_id=self.source_dataset_id)
                caption_dataset = StateTracker.get_data_backend(self.source_dataset_id)["dataset_runtime"]
            validation_prompt = caption_dataset.resolve_caption(caption_path)
            if type(validation_prompt) == list:
                self.debug_log(f"Selecting random prompt from list: {validation_prompt}")
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

    def _yield_sequential_image(self):
        """
        Always return the *next* image in a fixed list, wrapping around when
        we reach the end.  Nothing is random, so the same N calls → same N
        paths every run.
        """
        if self._val_cursor >= len(self._val_master_list):
            self._val_cursor = 0
            if len(self._val_master_list) == 0:
                raise MultiDatasetExhausted("No validation images available. Please check your dataset.")
        path = self._val_master_list[self._val_cursor]
        self._val_cursor = (self._val_cursor + 1) % len(self._val_master_list)
        return path

    def yield_single_image(self, filepath: str):
        """
        Yield a single image from the dataset by path.

        If the path prefix isn't in the path, we'll add it.
        """
        filepath = canonicalize_data_uri(filepath)
        if (
            self.metadata_backend.instance_data_dir is not None
            and self.metadata_backend.instance_data_dir not in filepath
            and not filepath.startswith(("http", "webshart://"))
        ):
            filepath = os.path.join(self.metadata_backend.instance_data_dir, filepath)
        image_data = self.data_backend.read_image(filepath)
        return image_data

    def _empty_dataset_message(self) -> str:
        bucket_report = getattr(self.metadata_backend, "bucket_report", None)
        if bucket_report:
            bucket_report.add_note("Sampler attempted to reset buckets but none were available.")
            bucket_report.record_stage("sampler_batches", sample_count=len(self.metadata_backend))
            return bucket_report.format_empty_dataset_message()
        empty_report = getattr(self.metadata_backend, "empty_dataset_report", None)
        if empty_report:
            return empty_report
        return f"No images found in the dataset: {self.metadata_backend.aspect_ratio_bucket_indices}"

    def _sample_path(self, image):
        image = canonicalize_data_uri(image)
        return (
            image
            if image.startswith(("http", "webshart://"))
            else os.path.join(self.metadata_backend.instance_data_dir, image)
        )

    @staticmethod
    def _positive_int(value):
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _configured_max_samples(self, data_backend_config):
        configured_max_samples = self._positive_int(data_backend_config.get("max_num_samples"))
        if configured_max_samples is not None:
            return configured_max_samples
        return self._positive_int(getattr(self.metadata_backend, "max_num_samples", None))

    def _bucket_report_total(self, stage: str):
        bucket_report = getattr(self.metadata_backend, "bucket_report", None)
        bucket_summaries = getattr(bucket_report, "bucket_summaries", None)
        if not isinstance(bucket_summaries, dict):
            return None
        summary = bucket_summaries.get(stage)
        if not isinstance(summary, dict):
            return None
        return self._positive_int(summary.get("total_samples"))

    def _model_card_sample_count(self, data_backend_config):
        configured_max_samples = self._configured_max_samples(data_backend_config)
        report_total = self._bucket_report_total("post_refresh")
        if report_total is not None:
            total_sample_count = report_total
            approximate = False
        else:
            total_sample_count = sum(len(indices) for indices in self.metadata_backend.aspect_ratio_bucket_indices.values())
            effective_dp_size, _dp_rank, _cp_size = get_cp_aware_dp_info(self.accelerator)
            approximate = effective_dp_size > 1
            if approximate:
                total_sample_count *= effective_dp_size

        if configured_max_samples is not None:
            total_sample_count = min(total_sample_count, configured_max_samples)
            approximate = False
        return f"~{total_sample_count}" if approximate else total_sample_count

    def log_state(self, show_rank: bool = True, alt_stats: bool = False):
        if alt_stats:
            # Return an overview instead of a snapshot.
            # Eg. return totals, and not "as it is now"
            data_backend_config = StateTracker.get_data_backend_config(self.id)
            total_image_count = self._model_card_sample_count(data_backend_config)
            printed_state = [f"- Repeats: {data_backend_config.get('repeats', 0)}"]
            printed_state.append(f"- Total number of {self.sample_type_strs}: {total_image_count}")
            printed_state.append(
                f"- Total number of aspect buckets: {len(self.metadata_backend.aspect_ratio_bucket_indices)}"
            )
            schedule_state = StateTracker.get_dataset_schedule(self.id) or {}
            start_epoch = schedule_state.get("start_epoch", 1)
            start_step = schedule_state.get("start_step", 0)
            reached = schedule_state.get("reached", False)
            reached_epoch = schedule_state.get("reached_at_epoch")
            reached_step = schedule_state.get("reached_at_step")
            if start_epoch > 1 or start_step > 0 or not reached:
                status = "not reached"
                if reached:
                    epoch_str = reached_epoch if reached_epoch is not None else "?"
                    step_str = reached_step if reached_step is not None else "?"
                    status = f"reached at epoch {epoch_str}, step {step_str}"
                printed_state.append(f"- Schedule: start_epoch={start_epoch}, start_step={start_step} ({status})")
            if self.sample_type_strs == "images":
                printed_state.append(
                    f"- Resolution: {self.resolution} {'megapixels' if self.resolution_type == 'area' else 'px'}"
                )
                printed_state.append(f"- Cropped: {data_backend_config.get('crop')}")
                printed_state.append(
                    f"- Crop style: {'None' if not data_backend_config.get('crop') else data_backend_config.get('crop_style')}"
                )
                printed_state.append(
                    f"- Crop aspect: {'None' if not data_backend_config.get('crop') else data_backend_config.get('crop_aspect')}"
                )
            elif self.sample_type_strs == "videos":
                video_config = data_backend_config.get("video") or {}
                target_frame_count = data_backend_config.get("frames_per_video") or video_config.get("num_frames")
                fps = data_backend_config.get("fps") or video_config.get("fps")
                printed_state.append(
                    f"- Target frame count: {target_frame_count if target_frame_count is not None else 'unknown'}"
                )
                printed_state.append(f"- FPS: {fps if fps is not None else 'unknown'}")
            elif self.sample_type_strs == "audio":
                sample_rate = data_backend_config.get("sample_rate") or data_backend_config.get("audio_sample_rate")
                target_len = data_backend_config.get("sample_size") or data_backend_config.get("audio_samples")
                printed_state.append(f"- Sample rate: {sample_rate if sample_rate is not None else 'unknown'}")
                printed_state.append(f"- Target samples: {target_len if target_len is not None else 'unknown'}")
            printed_state.append(f"- Used for regularisation data: {'Yes' if self.is_regularisation_data else 'No'}")
            if self.conditioning_type:
                printed_state.append(f"- Conditioning type: {self.conditioning_type}")
            printed_state = "\n".join(printed_state) + "\n"
        else:
            # Return a snapshot of the current state during training.
            consumed, total = self.cursor.progress()
            prefix = f"\n{self.rank_info if show_rank else ''}    -> "
            printed_state = (
                f"{prefix}Epoch: {self.cursor.epoch}"
                f"{prefix}Number of consumed {self.sample_type_strs}: {consumed}"
                f"{prefix}Number of {self.sample_type_strs} in an epoch: {total}"
                f"{prefix}Batches in an epoch: {len(self.cursor)}"
            )
        self.logger.info(printed_state)

        return printed_state

    def _validate_and_yield_images_from_samples(self, samples, bucket):
        """
        Validate and yield images from given samples. Return a list of valid image paths.
        """
        to_yield = []
        for image_metadata in self.dataset_runtime.read_samples(samples):
            image_path = image_metadata["image_path"]
            requires_crop = (
                StateTracker.get_args().model_family
                not in [
                    "sd1x",
                    "sd2x",
                    "deepfloyd",
                    "ace_step",
                ]
                and self.dataset_type != "audio"
                and image_metadata.get("dataset_type") != "audio"
            )
            if requires_crop and "crop_coordinates" not in image_metadata:
                raise Exception(f"An image was discovered ({image_path}) that did not have its metadata: {image_metadata}")
            instance_prompt = self.dataset_runtime.resolve_caption(image_path)
            if type(instance_prompt) == list:
                instance_prompt = random.choice(instance_prompt)
                self.debug_log(f"Selecting random prompt from list: {instance_prompt}")
            image_metadata["instance_prompt_text"] = instance_prompt

            to_yield.append(image_metadata)
        return to_yield

    def get_conditioning_sample(self, original_sample_path: str) -> TrainingSample | None:
        """
        Given an original dataset sample path, return a TrainingSample
        """
        conditioning_dir = self.metadata_backend.instance_data_dir
        if os.path.isabs(original_sample_path):
            full_path = original_sample_path
            conditioning_dir_abs = os.path.abspath(conditioning_dir) if conditioning_dir is not None else None
            if self.source_dataset_id is not None:
                source_backend = StateTracker.get_data_backend(self.source_dataset_id)
                source_dir = source_backend["config"]["instance_data_dir"]
                source_dir_abs = os.path.abspath(source_dir)
                if full_path.startswith(source_dir_abs):
                    full_path = full_path.replace(source_dir_abs, conditioning_dir_abs, 1)
            if conditioning_dir_abs is not None and not full_path.startswith(conditioning_dir_abs):
                full_path = os.path.join(conditioning_dir_abs, os.path.basename(full_path))
        else:
            sample_path = original_sample_path.lstrip("/")
            if conditioning_dir not in sample_path:
                full_path = os.path.join(conditioning_dir, sample_path)
            else:
                full_path = sample_path
        conditioning_config = StateTracker.get_data_backend_config(self.id).get("conditioning_config") or {}
        if conditioning_config.get("type") == "i2v_first_frame":
            full_path = os.path.splitext(full_path)[0] + ".png"
        try:
            conditioning_sample_data = self.data_backend.read_image(full_path)
        except Exception as e:
            self.logger.error(f"Could not fetch conditioning sample: {e}")
            return None
        if conditioning_sample_data is None:
            self.debug_log(f"Could not fetch conditioning sample from {full_path}.")
            return None

        conditioning_sample_metadata = self.dataset_runtime.read_samples([full_path])[0]
        conditioning_sample = TrainingSample(
            image=conditioning_sample_data,
            data_backend_id=self.id,
            image_metadata=conditioning_sample_metadata,
            image_path=full_path,
            conditioning_type=self.conditioning_type,
            model=self.model,
        )

        sample_image = conditioning_sample.image
        if isinstance(sample_image, np.ndarray) and sample_image.ndim >= 4:
            conditioning_sample.image = sample_image[0]
        elif isinstance(sample_image, list) and len(sample_image) > 0:
            conditioning_sample.image = sample_image[0]
        if isinstance(conditioning_sample.image, np.ndarray) and conditioning_sample.image.ndim == 3:
            conditioning_sample.image = Image.fromarray(conditioning_sample.image.astype(np.uint8))
        # Use the magic prompt handler to retrieve the captions.
        caption_dataset = self.dataset_runtime
        caption_path = full_path
        if self.source_dataset_id is not None and self.caption_strategy is None:
            caption_path = conditioning_sample.training_sample_path(training_dataset_id=self.source_dataset_id)
            caption_dataset = StateTracker.get_data_backend(self.source_dataset_id)["dataset_runtime"]
        instance_prompt = caption_dataset.resolve_caption(caption_path)
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
        sampling_mode = getattr(StateTracker.get_args(), "conditioning_multidataset_sampling", "combined")

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
                if self.metadata_backend.instance_data_dir is not None and self.metadata_backend.instance_data_dir != "":
                    sample_path = sample_path.split(self.metadata_backend.instance_data_dir)[-1].lstrip("/")

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
                    for fallback_idx, fallback_dataset in enumerate(conditioning_datasets):
                        if fallback_idx == selected_idx:
                            continue  # Skip the one we already tried

                        fallback_sampler = fallback_dataset["sampler"]
                        conditioning_sample = fallback_sampler.get_conditioning_sample(sample_path)
                        if conditioning_sample is not None:
                            self.logger.warning(
                                f"Sample {sample_path} not found in {selected_dataset['id']}, "
                                f"using fallback from {fallback_dataset['id']}"
                            )
                            conditioning_sample._source_dataset_id = fallback_dataset["id"]
                            outputs.append(conditioning_sample)
                            found = True
                            break

                    if not found:
                        self.logger.error(
                            f"Could not find conditioning sample for {sample_path} " f"in any conditioning dataset"
                        )

        elif sampling_mode == "combined" or len(conditioning_datasets) == 1:
            # Combined mode: append conditioning samples from all datasets (current behavior)
            missing_samples = []

            for dataset in conditioning_datasets:
                sampler = dataset["sampler"]
                dataset_samples_added = 0

                for sample in samples:
                    sample_path: str = sample["image_path"]
                    if self.metadata_backend.instance_data_dir is not None and self.metadata_backend.instance_data_dir != "":
                        sample_path = sample_path.split(self.metadata_backend.instance_data_dir)[-1].lstrip("/")

                    conditioning_sample = sampler.get_conditioning_sample(sample_path)
                    if conditioning_sample is not None:
                        # Tag the sample with which dataset it came from
                        conditioning_sample._source_dataset_id = dataset["id"]
                        outputs.append(conditioning_sample)
                        dataset_samples_added += 1
                    else:
                        missing_samples.append((sample_path, dataset["id"]))

                self.debug_log(
                    f"Added {dataset_samples_added}/{len(samples)} conditioning samples " f"from dataset {dataset['id']}"
                )

            # Report missing samples in batch if there are any
            if missing_samples:
                self.logger.warning(
                    f"Missing {len(missing_samples)} conditioning samples: "
                    f"{missing_samples[:3]}{'...' if len(missing_samples) > 3 else ''}"
                )

        else:
            raise ValueError(
                f"Unknown conditioning_multidataset_sampling mode: {sampling_mode}. " "Must be 'random' or 'combined'."
            )

        # Validate output count
        expected_conditioning_count = len(samples) * (1 if sampling_mode == "random" else len(conditioning_datasets))
        actual_conditioning_count = len(outputs) - len(samples)

        if actual_conditioning_count != expected_conditioning_count:
            self.logger.warning(
                f"Expected {expected_conditioning_count} conditioning samples but got "
                f"{actual_conditioning_count} (mode: {sampling_mode}, "
                f"datasets: {len(conditioning_datasets)})"
            )

        return tuple(outputs)

    def connect_s2v_samples(self, samples: tuple):
        """
        Connect S2V audio samples to video samples.

        For S2V (Speech-to-Video) models, this method looks up audio files
        from s2v_datasets that match video file paths by base filename
        (without extension), and adds the audio paths to each sample's metadata.

        When an S2V dataset has `source_from_video: True`, the audio is extracted
        from the video file itself, so the video path is used directly as the audio path.
        """
        s2v_datasets = StateTracker.get_s2v_datasets(self.id)
        if not s2v_datasets:
            return samples

        from pathlib import Path

        outputs = []
        for sample in samples:
            if isinstance(sample, dict):
                video_path = sample.get("image_path")
            else:
                image_path_attr = getattr(sample, "image_path", None)
                video_path = image_path_attr() if callable(image_path_attr) else image_path_attr
            if video_path is None:
                outputs.append(sample)
                continue

            # Extract base filename without extension
            video_stem = Path(video_path).stem

            # Look through s2v_datasets for matching audio file
            audio_path = None
            audio_backend_id = None
            for s2v_dataset in s2v_datasets:
                s2v_config = s2v_dataset.get("config", {})
                audio_config = s2v_config.get("audio", {})

                # Check if audio is extracted from video files
                if audio_config.get("source_from_video", False):
                    metadata_backend = s2v_dataset.get("metadata_backend")
                    if metadata_backend is not None:
                        metadata = metadata_backend.get_metadata_by_filepath(video_path)
                        if metadata is None:
                            self.debug_log(f"No matching audio found for video: {video_path}")
                            continue
                    # Audio comes from the video file itself
                    audio_path = video_path
                    audio_backend_id = s2v_dataset.get("id")
                    break

                # Original matching logic - look for separate audio files
                audio_root = s2v_config.get("instance_data_dir")
                if not audio_root:
                    continue

                # Try common audio extensions
                audio_backend = s2v_dataset.get("data_backend")
                for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
                    candidate = os.path.join(audio_root, f"{video_stem}{ext}")
                    exists = audio_backend.exists(candidate) if audio_backend is not None else Path(candidate).exists()
                    if exists:
                        audio_path = candidate
                        audio_backend_id = s2v_dataset.get("id")
                        break
                if audio_path:
                    break

            # Add audio path to sample metadata
            if isinstance(sample, dict):
                sample["s2v_audio_path"] = audio_path
                sample["s2v_audio_backend_id"] = audio_backend_id
            elif hasattr(sample, "image_metadata") and sample.image_metadata is not None:
                sample.image_metadata["s2v_audio_path"] = audio_path
                sample.image_metadata["s2v_audio_backend_id"] = audio_backend_id
            else:
                # For TrainingSample objects, store in a new attribute
                sample._s2v_audio_path = audio_path
                sample._s2v_audio_backend_id = audio_backend_id

            outputs.append(sample)

            if audio_path is None:
                self.debug_log(f"No matching audio found for video: {video_path}")

        return tuple(outputs)

    def __iter__(self):
        """
        Iterate over the sampler to yield image paths in batches.
        """
        if len(self.cursor) == 0:
            raise Exception(self._empty_dataset_message())
        while True:
            try:
                bucket, sample_refs = self.cursor.next_batch()
            except MultiDatasetExhausted:
                if StateTracker.get_args().print_sampler_statistics:
                    self.logger.info(f"Dataset '{self.id}' finished an epoch. State after reset:")
                    self.log_state()
                raise
            batch = self._validate_and_yield_images_from_samples(
                [self._sample_path(sample_ref) for sample_ref in sample_refs], bucket
            )
            # if applicable, we'll append TrainingSample(s) to the end for conditioning inputs.
            batch = self.connect_conditioning_samples(batch)
            # if applicable, connect S2V audio paths for speech-to-video models.
            batch = self.connect_s2v_samples(batch)
            yield tuple(batch)

    def __len__(self):
        return len(self.cursor)

    @staticmethod
    def convert_to_human_readable(aspect_ratio_float: float, bucket: iter, resolution: int = 1024):

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
