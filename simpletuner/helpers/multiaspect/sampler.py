import bisect
import logging
import os
import random
from collections import Counter

import torch
from accelerate.logging import get_logger

from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime, DatasetValidationSample
from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.metadata.backends.base import get_cp_aware_dp_info
from simpletuner.helpers.metadata.preparation import DatasetReadView
from simpletuner.helpers.multiaspect.image import MultiaspectImage
from simpletuner.helpers.multiaspect.progress import LegacySamplerProgress
from simpletuner.helpers.multiaspect.state import BucketStateManager
from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from simpletuner.helpers.training.multi_process import rank_info
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.utils.pathing import canonicalize_data_uri

pil_logger = logging.getLogger("PIL.Image")
pil_logger.setLevel(logging.WARNING)
pil_logger = logging.getLogger("PIL.PngImagePlugin")
pil_logger.setLevel(logging.WARNING)
pil_logger = logging.getLogger("PIL.TiffImagePlugin")
pil_logger.setLevel(logging.WARNING)


class _UnseenOccurrenceIndex:
    """One bucket's unseen occurrences, kept equal to what `_filter_unseen_occurrences` would return.

    Marking a filepath seen raises its occurrence count by one, which retires exactly its earliest
    unseen occurrence, so the index is updated per consumed sample instead of rescanning the bucket.
    """

    def __init__(self, source: list, seen_images: dict, positions: list, sample_paths: list):
        self.snapshot = list(source)
        self.seen_images = seen_images
        self.positions = positions
        self.sample_paths = sample_paths
        self.pending = {}
        for position in positions:
            self.pending.setdefault(source[position], []).append(position)

    def is_current(self, source: list, seen_images: dict) -> bool:
        return self.seen_images is seen_images and self.snapshot == source

    def consume(self, image_path: str) -> None:
        occurrences = self.pending.get(image_path)
        if not occurrences:
            return
        index = bisect.bisect_left(self.positions, occurrences.pop(0))
        del self.positions[index]
        del self.sample_paths[index]


class MultiAspectSampler(torch.utils.data.Sampler):
    epoch_provider = None
    _legacy_progress = None

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
        track_consumed: bool = False,
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

        self.epoch_provider = dataset_runtime.epoch_provider
        self.training_cursor = None
        self.consumed_state = None
        self.validation_cursor = dataset_runtime.open_validation()
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
        self.current_bucket = None
        self.current_epoch = 1
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
        self.exhausted_buckets = []
        self._unseen_indices = {}
        self.buckets = self.load_buckets() if self.epoch_provider is None else []
        self.state_manager = BucketStateManager(self.id)
        if track_consumed and self.epoch_provider is None:
            self._legacy_repeat = 0
            self._legacy_progress = LegacySamplerProgress(self.metadata_backend.seen_images, self._legacy_state(0))

    @property
    def tracks_consumed_batches(self):
        return self._legacy_progress is not None

    def start_training_epoch(self, epoch):
        if self.current_epoch != epoch:
            self.current_epoch = epoch
            self._legacy_repeat = 0
            self.current_bucket = None
            self.buckets = self.load_buckets()
            self.exhausted_buckets = []
            self.metadata_backend.seen_images = {}
            self._unseen_indices.clear()
            self._legacy_progress.reset(self.metadata_backend.seen_images, self._legacy_state(0))

    def _legacy_state(self, position):
        return {
            "position": position,
            "repeat": self._legacy_repeat,
            "current_epoch": self.current_epoch,
            "current_bucket": self.current_bucket,
            "buckets": list(self.buckets),
            "exhausted_buckets": list(self.exhausted_buckets),
        }

    def _legacy_layout(self):
        args = StateTracker.get_args()
        parallel = getattr(self.accelerator, "parallelism_config", None)
        return {
            "batch_size": self.batch_size,
            "seed": getattr(args, "seed", None),
            "accumulation": getattr(args, "gradient_accumulation_steps", 1),
            "data_parallel": list(get_cp_aware_dp_info(self.accelerator)),
            "world_size": self.accelerator.num_processes,
            "parallelism": [getattr(parallel, key, 1) for key in ("dp_replicate_size", "dp_shard_size", "tp_size")],
            "occurrence_revision": self.dataset_runtime.occurrence_revision(args),
        }

    def fetched_state(self):
        return self._legacy_progress.fetched

    def start_native_epoch(self, epoch):
        if self.current_epoch != epoch or (self.consumed_state and self.consumed_state["layout"]["epoch"] != epoch):
            if self.training_cursor is not None:
                self.training_cursor.close()
                self.training_cursor = None
            self.consumed_state = None
            self.current_epoch = epoch

    def _open_training_cursor(self):
        if self.training_cursor is None:
            spec = self.epoch_provider.epoch_spec(
                self.current_epoch,
                self.batch_size,
                occurrence_revision=self.dataset_runtime.occurrence_revision(StateTracker.get_args()),
            )
            self.training_cursor = self.dataset_runtime.open_epoch(spec)
            if self.consumed_state is None:
                self.consumed_state = self.training_cursor.state_dict()
        return self.training_cursor

    def _iterate_native(self):
        cursor = self._open_training_cursor()
        while True:
            try:
                bucket, refs = cursor.next_batch()
            except StopIteration:
                cursor.close()
                self.training_cursor = None
                self.current_epoch += 1
                raise MultiDatasetExhausted()
            state = cursor.state_dict()
            spec = cursor.spec
            rng = random.Random(f"{self.id}:{spec.seed}:{spec.epoch}:{spec.dp_rank}:{cursor.position}")
            view = DatasetReadView(state["view"], spec.epoch, spec.seed, getattr(cursor, "partition", None))
            records = self._validate_and_yield_images_from_samples(refs, bucket, rng=rng, view=view)
            records[0]["dataset_continuation"] = state
            records[0]["dataset_read_view"] = view
            records[0]["caption_dropout_seed"] = rng.getrandbits(64)
            yield tuple(self.connect_s2v_samples(self.connect_conditioning_samples(records, rng=rng, view=view)))

    def close(self):
        self.validation_cursor.close()
        if self.training_cursor is not None:
            self.training_cursor.close()
            self.training_cursor = None

    def reset_evaluation(self):
        if self.epoch_provider is None:
            self._reset_buckets(raise_exhaustion_signal=False)
            return
        if self.training_cursor is not None:
            self.training_cursor.close()
            self.training_cursor = None
        self.current_epoch = 1
        self.consumed_state = None

    def acknowledge_batch(self, continuation):
        if self.tracks_consumed_batches:
            self._legacy_progress.acknowledge(continuation)
        else:
            self.consumed_state = continuation

    def save_state(self, state_path: str):
        """
        This method should be called when the accelerator save hook is called,
         so that the state is correctly restored with a given checkpoint.
        """
        if self.epoch_provider is not None:
            if self.consumed_state is None:
                self._open_training_cursor()
            self.state_manager.save_state(
                {"native_epoch": self.consumed_state, "validation": self.validation_cursor.state_dict()}, state_path
            )
            return
        effective_dp_size, dp_rank, _cp_size = get_cp_aware_dp_info(self.accelerator)
        state = {
            "aspect_ratio_bucket_indices": self.metadata_backend.aspect_ratio_bucket_indices,
            "buckets": self.buckets,
            "exhausted_buckets": self.exhausted_buckets,
            "batch_size": self.batch_size,
            "current_bucket": self.current_bucket,
            "seen_images": self.metadata_backend.seen_images,
            "current_epoch": self.current_epoch,
            "dp_size": effective_dp_size,
            "dp_rank": dp_rank,
        }
        if self.tracks_consumed_batches:
            progress = self._legacy_progress
            with progress.checkpoint_seen() as seen:
                state.update(
                    {
                        key: progress.consumed[key]
                        for key in ("current_epoch", "current_bucket", "buckets", "exhausted_buckets")
                    }
                )
                state["seen_images"] = seen
                state["legacy_cursor"] = {
                    "version": 1,
                    "layout": self._legacy_layout(),
                    "position": progress.consumed["position"],
                    "repeat": progress.consumed["repeat"],
                }
                self.state_manager.save_state(state, state_path)
        else:
            self.state_manager.save_state(state, state_path)

    def _saved_schedule_is_restorable(self, previous_state: dict, state_path: str) -> bool:
        """A saved schedule is one rank's shard of the dataset.

        Restoring it is only correct when the data-parallel layout is unchanged and every rank
        has a shard to restore. Restoring on some ranks while the others re-split leaves the
        ranks disagreeing about who owns which samples.
        """
        effective_dp_size, dp_rank, _cp_size = get_cp_aware_dp_info(self.accelerator)
        if previous_state.get("dp_size") != effective_dp_size or previous_state.get("dp_rank") != dp_rank:
            self.logger.warning(
                f"Checkpoint schedule was written for dp_size={previous_state.get('dp_size')} "
                f"dp_rank={previous_state.get('dp_rank')}, but this run is dp_size={effective_dp_size} "
                f"dp_rank={dp_rank}. Keeping the freshly split schedule."
            )
            return False

        checkpoint_dir = os.path.dirname(state_path)
        own_filename = os.path.basename(state_path)
        stem, extension = os.path.splitext(own_filename)
        stem = stem.split("-rank", 1)[0]
        for rank in range(self.accelerator.num_processes):
            # Rank 0 writes the unsuffixed name, so it has to be built separately rather than
            # scanned as a -rank{N} sibling. This rank's own state is evidently present.
            filename = f"{stem}{extension}" if rank == 0 else f"{stem}-rank{rank}{extension}"
            if filename == own_filename:
                continue
            sibling = self.state_manager.mangle_state_path(os.path.join(checkpoint_dir, filename))
            if not os.path.exists(sibling):
                self.logger.warning(
                    f"Checkpoint holds no sampler state for rank {rank}, so the rank-local schedules "
                    "cannot be restored consistently. Keeping the freshly split schedule."
                )
                return False
        return True

    def load_states(self, state_path: str):
        try:
            previous_state = self.state_manager.load_state(state_path)
        except Exception as e:
            raise e

        if self.epoch_provider is not None:
            saved = previous_state.get("native_epoch")
            if not isinstance(saved, dict):
                raise ValueError("Provider-managed sampling requires a matching native checkpoint.")
            self.current_epoch = saved["layout"]["epoch"]
            if self.training_cursor is not None:
                self.training_cursor.close()
                self.training_cursor = None
            self._open_training_cursor().load_state_dict(saved)
            self.consumed_state = saved
            self.validation_cursor.load_state_dict(previous_state.get("validation"))
            return
        if "native_epoch" in previous_state:
            raise ValueError("Restoring a provider-managed checkpoint requires the same sampling policy.")
        legacy_cursor = previous_state.get("legacy_cursor")
        if legacy_cursor is not None:
            if (
                not self.tracks_consumed_batches
                or not isinstance(legacy_cursor, dict)
                or legacy_cursor.get("version") != 1
                or legacy_cursor.get("layout") != self._legacy_layout()
                or previous_state.get("aspect_ratio_bucket_indices") != self.metadata_backend.aspect_ratio_bucket_indices
            ):
                raise ValueError("Legacy consumed checkpoint requires unchanged prefetch, dataset, schedule and layout.")
            position, repeat = legacy_cursor.get("position"), legacy_cursor.get("repeat")
            if (
                type(position) is not int
                or position < 0
                or type(repeat) is not int
                or not 0 <= repeat <= self._legacy_repeats()
            ):
                raise ValueError("Invalid legacy consumed checkpoint position or repeat.")
        saved_batch_size = previous_state.get("batch_size")
        if "batch_size" in previous_state and saved_batch_size != self.batch_size:
            _allow = os.environ.get("SIMPLETUNER_ALLOW_MODIFYING_BSZ", "").lower() in (
                "1",
                "true",
                "yes",
            ) or getattr(StateTracker.get_args(), "i_know_what_i_am_doing", False)
            if not _allow:
                raise ValueError(
                    f"Dataset '{self.id}' checkpoint batch_size={saved_batch_size} does not match "
                    f"current batch_size={self.batch_size}. Resume with the same per-dataset train_batch_size, "
                    f"or set --i_know_what_i_am_doing / SIMPLETUNER_ALLOW_MODIFYING_BSZ=1 to bypass."
                )
            self.logger.warning(
                f"Dataset '{self.id}': resuming with batch_size={self.batch_size} but checkpoint recorded "
                f"batch_size={saved_batch_size}. Proceeding because override is set."
            )

        # Checkpoints contain the rank-local schedule. Restore it before seen
        # state so legacy boolean flags can be expanded to all occurrences.
        saved_schedule = previous_state.get("aspect_ratio_bucket_indices")
        if (
            legacy_cursor is None
            and isinstance(saved_schedule, dict)
            and self._saved_schedule_is_restorable(previous_state, state_path)
        ):
            self.metadata_backend.aspect_ratio_bucket_indices = saved_schedule
            self.validation_cursor.close()
        self.buckets = [str(bucket) for bucket in previous_state.get("buckets", self.load_buckets())]
        if "current_bucket" in previous_state:
            self.current_bucket = previous_state["current_bucket"]

        self.exhausted_buckets = []
        if "exhausted_buckets" in previous_state:
            self.logger.info(f"Previous checkpoint had {len(previous_state['exhausted_buckets'])} exhausted buckets.")
            self.exhausted_buckets = [str(bucket) for bucket in previous_state["exhausted_buckets"]]
        self.current_epoch = 1
        if "current_epoch" in previous_state:
            self.logger.info(f"Previous checkpoint was on epoch {previous_state['current_epoch']}.")
            self.current_epoch = previous_state["current_epoch"]
        # Merge seen_images into self.state_manager.seen_images Manager.dict:
        if "seen_images" in previous_state:
            seen = previous_state["seen_images"]
            self.logger.info(f"Previous checkpoint had {len(seen)} seen {self.sample_type_strs}.")
            legacy_seen = {ref for ref, value in seen.items() if value is True}
            occurrence_counts = {}
            if legacy_seen:
                occurrence_counts = Counter(
                    ref
                    for images in self.metadata_backend.aspect_ratio_bucket_indices.values()
                    for ref in images
                    if ref in legacy_seen
                )
            self.metadata_backend.seen_images.clear()
            self.metadata_backend.seen_images.update(
                (ref, occurrence_counts.get(ref, True) if value is True else 0 if value is False else value)
                for ref, value in seen.items()
            )
            self._unseen_indices = {}
        if legacy_cursor is None:
            # Older checkpoints record fetched legacy progress and have no consumed frontier.
            self._legacy_progress = None
        else:
            self._legacy_repeat = repeat
            self._legacy_progress.reset(self.metadata_backend.seen_images, self._legacy_state(position))

    def load_buckets(self):
        return list(self.metadata_backend.aspect_ratio_bucket_indices.keys())

    def retrieve_validation_set(self, batch_size: int):
        """
        Return random images from the set. They should be paired with their caption.

        Args:
            batch_size (int): Number of images to return.
        Returns:
            list[DatasetValidationSample]: input images and the dataset view used to prepare them
        """
        results = []
        positions = {}
        for position, ref in enumerate(self.validation_cursor.next_refs(batch_size)):
            positions.setdefault(ref, position)
        if not positions:
            return results
        revision = self.validation_cursor.state_dict()["view"] if self.dataset_runtime.uses_native_schedule else None
        view = DatasetReadView(revision) if revision is not None else self.dataset_runtime.published_read_view()
        records = self.dataset_runtime.read_samples(list(positions), fields=("metadata", "image"), view=view)
        prepared = []
        caption_paths = []
        caption_dataset = self.dataset_runtime.source_dataset or self.dataset_runtime
        for image_metadata in records:
            image_path = image_metadata["image_path"]
            training_sample = TrainingSample(
                image=image_metadata.pop("image"),
                data_backend_id=self.id,
                image_metadata=image_metadata,
                image_path=image_path,
                model=self.model,
                data_backend_config=self.dataset_runtime.config,
            )
            training_sample.prepare()
            caption_paths.append(
                image_path
                if caption_dataset is self.dataset_runtime
                else training_sample.training_sample_path(
                    caption_dataset.id, training_dataset=caption_dataset, conditioning_dataset=self.dataset_runtime
                )
            )
            prepared.append((image_path, training_sample.image))
        for (image_path, image), validation_prompt in zip(
            prepared, caption_dataset.resolve_captions(caption_paths), strict=True
        ):
            validation_shortname = f"{self.id}_{positions[image_path]}"
            if type(validation_prompt) == list:
                self.debug_log(f"Selecting random prompt from list: {validation_prompt}")
                if revision is not None:
                    selector = random.Random(f"{self.id}:{revision}:{image_path}")
                    validation_prompt = selector.choice(validation_prompt)
                else:
                    validation_prompt = random.choice(validation_prompt)
            results.append(
                DatasetValidationSample(
                    shortname=validation_shortname,
                    prompt=validation_prompt,
                    ref=image_path,
                    image=image,
                    dataset_id=self.id,
                    view=view,
                )
            )

        return results

    def _yield_n_from_exhausted_bucket(self, n: int, bucket: str, *, rng=random):
        """
        when a bucket is exhausted, and we have to populate the remainder of the batch,
        we shall use this quick and dirty method to retrieve n samples from the exhausted bucket.
        the thing is we can have a batch size of 4 and 1 image. so we'll have to just return the same image 4 times.
        """
        available_images = self._get_bucket_images(bucket)
        if len(available_images) == 0:
            self.debug_log(f"Bucket {bucket} is empty.")
            return []
        samples = []
        while len(samples) < n:
            to_grab = min(n, len(available_images), (n - len(samples)))
            if to_grab == 0:
                break
            samples.extend(rng.sample(available_images, k=to_grab))

        to_yield = self._validate_and_yield_images_from_samples(samples, bucket, rng=rng)
        return to_yield

    def _yield_random_image(self):
        bucket = random.choice(self.buckets)
        bucket_images = self._get_bucket_images(bucket)
        if not bucket_images:
            raise ValueError(f"Bucket {bucket} is empty")
        image_path = random.choice(bucket_images)
        return image_path

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

    def _bucket_name_to_id(self, bucket_name: str) -> int:
        """
        Return a bucket array index, by its name.

        Args:
            bucket_name (str): Bucket name, eg. "1.78"
        Returns:
            int: Bucket array index, eg. 0
        """
        # Numeric index
        if isinstance(bucket_name, (int, float)):
            # If it's already a valid index, return it.
            if int(bucket_name) == bucket_name and 0 <= int(bucket_name) < len(self.buckets):
                return int(bucket_name)
            # Otherwise try to resolve by value in buckets (for float bucket identifiers).
            if bucket_name in self.buckets:
                return self.buckets.index(bucket_name)
        name = str(bucket_name)
        if name.isdigit():
            self.debug_log(f"Assuming {bucket_name} is already an index.")
            return int(name)
        try:
            numeric_name = float(name)
            for idx, bucket in enumerate(self.buckets):
                if isinstance(bucket, (int, float)) and bucket == numeric_name:
                    return idx
        except (TypeError, ValueError):
            pass
        if bucket_name in self.buckets:
            return self.buckets.index(bucket_name)
        raise ValueError(f"Bucket name {bucket_name} not found in buckets: {self.buckets}")

    def _reset_buckets(self, raise_exhaustion_signal: bool = True, *, rng=random):
        if len(self.metadata_backend.seen_images) == 0 and len(self._get_unseen_images()) == 0:
            bucket_report = getattr(self.metadata_backend, "bucket_report", None)
            if bucket_report:
                bucket_report.add_note("Sampler attempted to reset buckets but none were available.")
                bucket_report.record_stage("sampler_batches", sample_count=len(self.metadata_backend))
                message = bucket_report.format_empty_dataset_message()
            else:
                message = (
                    f"No images found in the dataset: {self.metadata_backend.aspect_ratio_bucket_indices}"
                    f"\n-> Unseen {self.sample_type_strs}: {self._get_unseen_images()}"
                    f"\n-> Seen {self.sample_type_strs}: {self.metadata_backend.seen_images}"
                )
            empty_report = getattr(self.metadata_backend, "empty_dataset_report", None)
            if empty_report and not bucket_report:
                message = empty_report
            raise Exception(message)
        if StateTracker.get_args().print_sampler_statistics:
            self.logger.info(f"Resetting seen {self.sample_type_str} list and refreshing buckets. State before reset:")
            self.log_state()
        # All buckets are exhausted, so we will move onto the next epoch.
        if self.tracks_consumed_batches:
            self.metadata_backend.seen_images = {}
        else:
            self.current_epoch += 1
            self.metadata_backend.reset_seen_images()
        self.exhausted_buckets = []
        self.buckets = self.load_buckets()
        self._unseen_indices = {}
        self.change_bucket(rng=rng)
        if raise_exhaustion_signal:
            raise MultiDatasetExhausted()

    def _get_bucket_images(self, bucket):
        return self.metadata_backend.aspect_ratio_bucket_indices.get(str(bucket), [])

    def _unseen_occurrence_positions(self, images):
        """Positions of unconsumed occurrences, without collapsing duplicate filepaths."""
        occurrence_indices = {}
        positions = []
        for position, image in enumerate(images):
            occurrence_index = occurrence_indices.get(image, 0)
            occurrence_indices[image] = occurrence_index + 1
            if not self.metadata_backend.is_seen(image, occurrence_index):
                positions.append(position)
        return positions

    def _filter_unseen_occurrences(self, images):
        """Filter consumed positions without collapsing duplicate filepaths."""
        return [images[position] for position in self._unseen_occurrence_positions(images)]

    def _sample_path(self, image):
        image = canonicalize_data_uri(image)
        return (
            image
            if image.startswith(("http", "webshart://"))
            else os.path.join(self.metadata_backend.instance_data_dir, image)
        )

    def _warn_if_bucket_empty(self, bucket, bucket_images):
        if len(bucket_images) == 0:
            self.logger.warning(
                f"BUCKET {bucket} IS EMPTY! aspect_ratio_bucket_indices keys: "
                f"{list(self.metadata_backend.aspect_ratio_bucket_indices.keys())}"
            )

    def _get_unseen_images(self, bucket=None):
        """
        Get unseen {self.sample_type_strs} from the specified bucket.
        If bucket is None, get unseen {self.sample_type_strs} from all buckets.
        """
        if bucket:
            bucket_images = self._get_bucket_images(bucket)
            # Debug: Track bucket contents
            self.debug_log(
                f"_get_unseen_images(bucket={bucket}): "
                f"bucket has {len(bucket_images)} total images, "
                f"seen_images has {len(self.metadata_backend.seen_images)} entries"
            )
            self._warn_if_bucket_empty(bucket, bucket_images)
            return [self._sample_path(image) for image in self._filter_unseen_occurrences(bucket_images)]
        elif bucket is None:
            unseen_images = []
            for b, images in self.metadata_backend.aspect_ratio_bucket_indices.items():
                unseen_images.extend(self._sample_path(image) for image in self._filter_unseen_occurrences(images))
            return unseen_images
        else:
            return []

    def _iteration_unseen_images(self, bucket):
        """`_get_unseen_images(bucket)` for the iterator, served from an index that `__iter__` keeps current."""
        bucket_images = self._get_bucket_images(bucket)
        seen_images = self.metadata_backend.seen_images
        index = self._unseen_indices.get(bucket)
        if index is None or not index.is_current(bucket_images, seen_images):
            self._warn_if_bucket_empty(bucket, bucket_images)
            positions = self._unseen_occurrence_positions(bucket_images)
            index = _UnseenOccurrenceIndex(
                bucket_images,
                seen_images,
                positions,
                [self._sample_path(bucket_images[position]) for position in positions],
            )
            self._unseen_indices[bucket] = index
        return list(index.sample_paths)

    def _consume_unseen_occurrences(self, image_paths):
        """Retire the occurrences that marking `image_paths` seen consumes."""
        for image_path in image_paths:
            for index in self._unseen_indices.values():
                index.consume(image_path)

    def _handle_bucket_with_insufficient_images(self, bucket):
        """
        Handle buckets with insufficient images. Return True if we changed or reset the bucket.
        """
        bucket_images = self._get_bucket_images(bucket)
        if len(bucket_images) < self.batch_size:
            self.debug_log(f"Bucket {bucket} has insufficient ({len(bucket_images)}) images.")
            if bucket not in self.exhausted_buckets:
                self.debug_log(
                    f"Bucket {bucket} is now exhausted and sleepy, and we have to move it to the sleepy list before changing buckets."
                )
                self.move_to_exhausted()
            self.debug_log("Changing bucket to another random selection.")
            self.change_bucket()
            return True
        self.debug_log(f"Bucket {bucket} has sufficient ({len(bucket_images)}) images.")
        return False

    def _get_next_bucket(self, *, rng=random):
        """
        Get the next bucket excluding the exhausted ones.
        If all buckets are exhausted, first reset the seen {self.sample_type_strs} and exhausted buckets.
        """
        available_buckets = [bucket for bucket in self.buckets if bucket not in self.exhausted_buckets]
        if not available_buckets:
            # Raise MultiDatasetExhausted
            self._reset_buckets(rng=rng)

        if len(self.exhausted_buckets) > 0:
            self.debug_log(f"exhausted buckets: {self.exhausted_buckets}")

        # Sequentially get the next bucket
        if hasattr(self, "current_bucket") and self.current_bucket is not None:
            self.current_bucket = (self.current_bucket + 1) % len(available_buckets)
        else:
            self.current_bucket = 0
        if self.buckets[self.current_bucket] not in available_buckets:
            random_bucket = rng.choice(available_buckets)
            self.current_bucket = available_buckets.index(random_bucket)

        next_bucket = available_buckets[self.current_bucket]
        return next_bucket

    def change_bucket(self, *, rng=random):
        """
        Change the current bucket to a new one and exclude exhausted buckets from consideration.
        During _get_next_bucket(), if all buckets are exhausted, reset the exhausted list and seen {self.sample_type_strs}.
        """
        next_bucket = self._get_next_bucket(rng=rng)
        # Resolve to an integer index in self.buckets for downstream indexing.
        if next_bucket in self.buckets:
            self.current_bucket = self.buckets.index(next_bucket)
        else:
            # Fallback for stringified indices (e.g., "0", "1") or float/int values.
            try:
                self.current_bucket = self._bucket_name_to_id(next_bucket)
            except Exception:
                # If still unresolved, pick bucket 0 to avoid TypeError and log for debugging.
                self.logger.warning(
                    f"Bucket {next_bucket} not found in buckets; defaulting to index 0 out of {len(self.buckets)} buckets."
                )
                self.current_bucket = 0
        self._clear_batch_accumulator()

    def move_to_exhausted(self):
        bucket = self.buckets[self.current_bucket]
        self.exhausted_buckets.append(bucket)
        self.buckets.remove(bucket)
        self.debug_log(
            f"Bucket {bucket} is empty or doesn't have enough samples for a full batch. Removing from bucket list. {len(self.buckets)} remain."
        )

    def log_state(self, show_rank: bool = True):
        if self.epoch_provider is not None:
            desc = self.dataset_runtime.describe()
            overview = f"Dataset {self.id}: {desc.accepted_samples} accepted samples, {len(self)} batches per rank."
            self.logger.info(overview)
            return overview
        printed_state = (
            f"\n{self.rank_info if show_rank else ''}    -> Number of seen {self.sample_type_strs}: {len(self.metadata_backend.seen_images)}"
            f"\n{self.rank_info if show_rank else ''}    -> Number of unseen {self.sample_type_strs}: {len(self._get_unseen_images())}"
            f"\n{self.rank_info if show_rank else ''}    -> Current Bucket: {self.current_bucket}"
            f"\n{self.rank_info if show_rank else ''}    -> {len(self.buckets)} Buckets: {self.buckets}"
            f"\n{self.rank_info if show_rank else ''}    -> {len(self.exhausted_buckets)} Exhausted Buckets: {self.exhausted_buckets}"
        )
        self.logger.info(printed_state)
        return printed_state

    def _validate_and_yield_images_from_samples(self, samples, bucket, *, rng=random, view=None):
        """
        Validate and yield images from given samples. Return a list of valid image paths.
        """
        records = self.dataset_runtime.read_samples(samples, view=view)
        for image_metadata in records:
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
        for image_metadata, instance_prompt in zip(records, self.dataset_runtime.resolve_captions(samples), strict=True):
            if type(instance_prompt) == list:
                instance_prompt = rng.choice(instance_prompt)
                self.debug_log(f"Selecting random prompt from list: {instance_prompt}")
            image_metadata["instance_prompt_text"] = instance_prompt

        return records

    def _clear_batch_accumulator(self):
        self.batch_accumulator = []

    def connect_conditioning_samples(self, samples: tuple, *, rng=None, view=None):
        # Locate the conditioning data
        conditioning_datasets = self.dataset_runtime.conditioning_datasets
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

        def read_pair(dataset, sample_path, source_path):
            return dataset.read_conditioning_sample(
                sample_path, source=self.dataset_runtime, source_ref=source_path, source_view=view, model=self.model, rng=rng
            )

        if sampling_mode == "random" and len(conditioning_datasets) > 1:
            # Random mode: select one conditioning dataset per training sample
            for sample in samples:
                sample_path = self.dataset_runtime.pairing_key(sample["image_path"])

                # Deterministic selection based on hash of image path and current epoch
                # This ensures the same image gets the same conditioning dataset within an epoch
                # but different datasets across epochs
                if rng is None or self.tracks_consumed_batches:
                    selected_idx = random.Random(f"{self.id}:{self.current_epoch}:{sample_path}").randrange(
                        len(conditioning_datasets)
                    )
                else:
                    selected_idx = rng.randrange(len(conditioning_datasets))
                selected_dataset = conditioning_datasets[selected_idx]

                self.debug_log(
                    f"Selected conditioning dataset {selected_dataset.id} "
                    f"for sample {sample_path} (epoch {self.current_epoch})"
                )

                # Get the conditioning sample from the selected dataset
                conditioning_sample = read_pair(selected_dataset, sample_path, sample["image_path"])
                if conditioning_sample is not None:
                    # Tag the sample with which dataset it came from for debugging
                    conditioning_sample._source_dataset_id = selected_dataset.id
                    outputs.append(conditioning_sample)
                else:
                    # Try other datasets if the selected one doesn't have this sample
                    found = False
                    for fallback_idx, fallback_dataset in enumerate(conditioning_datasets):
                        if fallback_idx == selected_idx:
                            continue  # Skip the one we already tried

                        conditioning_sample = read_pair(fallback_dataset, sample_path, sample["image_path"])
                        if conditioning_sample is not None:
                            self.logger.warning(
                                f"Sample {sample_path} not found in {selected_dataset.id}, "
                                f"using fallback from {fallback_dataset.id}"
                            )
                            conditioning_sample._source_dataset_id = fallback_dataset.id
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
                dataset_samples_added = 0

                for sample in samples:
                    sample_path = self.dataset_runtime.pairing_key(sample["image_path"])

                    conditioning_sample = read_pair(dataset, sample_path, sample["image_path"])
                    if conditioning_sample is not None:
                        # Tag the sample with which dataset it came from
                        conditioning_sample._source_dataset_id = dataset.id
                        outputs.append(conditioning_sample)
                        dataset_samples_added += 1
                    else:
                        missing_samples.append((sample_path, dataset.id))

                self.debug_log(
                    f"Added {dataset_samples_added}/{len(samples)} conditioning samples " f"from dataset {dataset.id}"
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
        """Attach the linked audio reference to each video occurrence."""
        if not self.dataset_runtime.audio_datasets:
            return samples

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

            pairing = self.dataset_runtime.resolve_s2v_audio(video_path)
            audio_dataset, audio_path = pairing if pairing is not None else (None, None)
            audio_backend_id = audio_dataset.id if audio_dataset is not None else None

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
        if self.epoch_provider is not None:
            yield from self._iterate_native()
        elif self.tracks_consumed_batches:
            yield from self._iterate_resumable()
        else:
            yield from self._iterate_legacy()

    def _legacy_repeats(self):
        return max(0, self.dataset_runtime.config.get("repeats", 0) or 0)

    def _iterate_resumable(self):
        progress = self._legacy_progress
        seed = getattr(StateTracker.get_args(), "seed", 0) or 0
        _, rank, _ = get_cp_aware_dp_info(self.accelerator)
        repeats = self._legacy_repeats()
        while True:
            with progress.lock:
                rng = random.Random(
                    f"{self.id}:{seed}:{self.current_epoch}:{rank}:{self._legacy_repeat}:{progress.fetched['position']}"
                )
                try:
                    batch = next(self._iterate_legacy(rng=rng))
                except MultiDatasetExhausted:
                    if self._legacy_repeat >= repeats:
                        raise
                    self._legacy_repeat += 1
                    continue
                refs = [record["image_path"] for record in batch[: self.batch_size]]
                progress.record(self.metadata_backend.seen_images, self._legacy_state(0), refs)
            yield batch

    def _iterate_legacy(self, *, rng=random):
        # Debug: Track bucket state at iteration start
        if self.buckets:
            first_bucket = self.buckets[0]
            bucket_images = self._get_bucket_images(first_bucket)
            all_bucket_keys = list(self.metadata_backend.aspect_ratio_bucket_indices.keys())
            self.debug_log(
                f"__iter__ called. Bucket {first_bucket} (type={type(first_bucket).__name__}) has {len(bucket_images)} images. "
                f"Total buckets in sampler: {len(self.buckets)}. "
                f"aspect_ratio_bucket_indices has {len(all_bucket_keys)} keys: {all_bucket_keys} "
                f"(key types: {[type(k).__name__ for k in all_bucket_keys[:3]]}). "
                f"seen_images count: {len(self.metadata_backend.seen_images)}"
            )
            if len(bucket_images) == 0 and len(all_bucket_keys) > 0:
                self.logger.warning(
                    f"BUCKET KEY MISMATCH! Looking for {first_bucket} ({type(first_bucket).__name__}) "
                    f"but aspect_ratio_bucket_indices has keys: {all_bucket_keys}"
                )
        else:
            self.logger.warning(f"__iter__ called but self.buckets is EMPTY!")

        self._clear_batch_accumulator()  # Initialize an empty list to accumulate images for a batch
        self.change_bucket(rng=rng)
        while True:
            # Loop through all buckets to find one with sufficient images
            for _ in range(len(self.buckets)):
                self._clear_batch_accumulator()
                available_images = self._iteration_unseen_images(self.buckets[self.current_bucket])
                self.debug_log(
                    f"From {len(self.buckets)} buckets, selected {self.buckets[self.current_bucket]} ({self.buckets[self.current_bucket]}) -> {len(available_images)} available images, and our accumulator has {len(self.batch_accumulator)} images ready for yielding."
                )
                if available_images:
                    break
                self.move_to_exhausted()
                self.change_bucket(rng=rng)
            else:
                self._reset_buckets(rng=rng)

            while len(available_images) > 0:
                if len(available_images) < self.batch_size:
                    need_image_count = self.batch_size - len(available_images)
                    self.debug_log(
                        f"Bucket {self.buckets[self.current_bucket]} has {len(available_images)} available images, but we need {need_image_count} more."
                    )
                    to_yield = self._yield_n_from_exhausted_bucket(
                        need_image_count, self.buckets[self.current_bucket], rng=rng
                    )
                    # add the available images
                    to_yield.extend(
                        self._validate_and_yield_images_from_samples(
                            available_images, self.buckets[self.current_bucket], rng=rng
                        )
                    )
                else:
                    samples = rng.sample(available_images, k=min(len(available_images), self.batch_size))
                    to_yield = self._validate_and_yield_images_from_samples(
                        samples, self.buckets[self.current_bucket], rng=rng
                    )
                self.debug_log(f"Building batch with {len(self.batch_accumulator)} samples.")
                if len(self.batch_accumulator) < self.batch_size:
                    remaining_entries_needed = self.batch_size - len(self.batch_accumulator)
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
                    yielded_paths = [instance["image_path"] for instance in final_yield]
                    self.metadata_backend.mark_batch_as_seen(yielded_paths)
                    self._consume_unseen_occurrences(yielded_paths)
                    # if applicable, we'll append TrainingSample(s) to the end for conditioning inputs.
                    if self.tracks_consumed_batches:
                        final_yield[0]["caption_dropout_seed"] = rng.getrandbits(64)
                        final_yield = self.connect_conditioning_samples(final_yield, rng=rng)
                    else:
                        final_yield = self.connect_conditioning_samples(final_yield)
                    # if applicable, connect S2V audio paths for speech-to-video models.
                    final_yield = self.connect_s2v_samples(final_yield)
                    yield tuple(final_yield)
                    # Change bucket after a full batch is yielded
                    self.change_bucket(rng=rng)
                    # Break out of the while loop:
                    break

                # Update available images after yielding
                available_images = self._iteration_unseen_images(self.buckets[self.current_bucket])
                self.debug_log(
                    f"Bucket {self.buckets[self.current_bucket]} now has {len(available_images)} available images after yielding."
                )

    def __len__(self):
        if self.epoch_provider is not None:
            spec = self.epoch_provider.epoch_spec(self.current_epoch, self.batch_size)
            count = self.epoch_provider.scheduled_batches(spec)
            if count is None:
                return self.metadata_backend.runtime_args.max_train_steps * spec.accumulation
            return count
        batches_per_pass = sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.metadata_backend.aspect_ratio_bucket_indices.values()
        )
        return batches_per_pass * (1 + self._legacy_repeats())

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
