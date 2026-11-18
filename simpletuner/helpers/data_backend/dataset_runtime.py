"""Dataset access shared by training and preparation consumers."""

import os
import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.data_backend.cursor import DatasetCursor, WindowedCursor
from simpletuner.helpers.metadata.backends.base import MetadataBackend, get_cp_aware_dp_info
from simpletuner.helpers.prompts import CaptionNotFoundError, PromptHandler
from simpletuner.helpers.training.multi_process import broadcast_object_from_main
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.utils.pathing import normalize_data_path

# Dataset settings a run may change without the dataset's samples being decided again.
RUNTIME_MUTABLE_KEYS = ("train_batch_size", "repeats", "start_epoch", "start_step", "end_epoch", "end_step")


def shuffle_window(config: Mapping[str, Any]) -> int | None:
    """How many shards a rank keeps open, for a dataset that is indexed a shard at a time."""
    return (config.get("webshart") or {}).get("shuffle_window")


def recrops_each_epoch(config: Mapping[str, Any], args) -> bool:
    """Whether a dataset's samples are given a new aspect and crop when an epoch ends."""
    return bool(config.get("crop")) and config.get("crop_aspect") == "random" and not args.aspect_bucket_disable_rebuild


class DatasetRuntime:
    def __init__(
        self,
        id: str,
        data_backend: BaseDataBackend,
        metadata_backend: MetadataBackend,
        config: dict[str, Any],
    ):
        self.id = id
        self.data_backend = data_backend
        self.metadata_backend = metadata_backend
        self.config = {
            "dataset_type": "image",
            "caption_strategy": "filename",
            "use_captions": True,
            "prepend_instance_prompt": False,
            "instance_prompt": None,
            "disable_multiline_split": False,
            **deepcopy(config),
        }
        self._cursor_options: dict[str, Any] | None = None
        self._cursor: DatasetCursor | WindowedCursor | None = None

    def configure_cursor(self, *, accelerator, batch_size: int, shared_by_all_ranks: bool) -> None:
        """Fix how this dataset is traversed. Every rank must call this at the same point."""
        args = StateTracker.get_args()
        seed = getattr(args, "seed", None)
        if seed is None:
            # Drawing a seed is a collective call: every rank has to shuffle with the same one.
            seed = broadcast_object_from_main(random.SystemRandom().getrandbits(64) if accelerator.is_main_process else None)
            if seed is None:
                raise RuntimeError(
                    f"Dataset '{self.id}': this rank did not receive the shuffle seed from the main process,"
                    " so it would divide the dataset differently from the other ranks."
                )
        dp_size, dp_rank, _cp_size = get_cp_aware_dp_info(accelerator)
        self._cursor_options = {
            "batch_size": batch_size,
            "seed": seed,
            "dp_size": dp_size,
            "dp_rank": dp_rank,
            "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", 1),
            "allow_oversubscription": bool(getattr(args, "allow_dataset_oversubscription", False)),
            "shared_by_all_ranks": shared_by_all_ranks,
        }
        self._cursor = None
        window = shuffle_window(self.config)
        if window is not None:
            self._cursor = self._windowed_cursor(accelerator, window)

    def _windowed_cursor(self, accelerator, window: int) -> WindowedCursor:
        options = self._cursor_options
        cursor = WindowedCursor(
            self.id,
            unit_count=self.metadata_backend.unit_count(),
            open_unit=self.metadata_backend.open_unit,
            close_unit=self.metadata_backend.close_unit,
            window=window,
            batch_size=options["batch_size"],
            repeats=int(StateTracker.get_data_backend_config(self.id).get("repeats") or 0),
            seed=options["seed"],
            dp_size=options["dp_size"],
            dp_rank=options["dp_rank"],
        )
        self.metadata_backend.check_window(cursor.units_in_window(), ranks=options["dp_size"])
        # The ranks have to plan the same number of steps for an epoch, so one of them measures
        # its first window and tells the others.
        cursor.samples_per_unit = broadcast_object_from_main(
            cursor.measure_samples_per_unit() if accelerator.is_main_process else None
        )
        if not cursor.samples_per_unit:
            raise ValueError(
                f"(id={self.id}) The first {len(cursor.units_in_window())} shard(s) opened hold no usable sample."
                " Check minimum_image_size, the aspect ratio limits and the caption settings against what the dataset holds."
            )
        return cursor

    def cursor(self) -> DatasetCursor | WindowedCursor:
        """This rank's traversal of the dataset, built from the bucket lists on first use."""
        if self._cursor is None:
            if self._cursor_options is None:
                raise RuntimeError(f"Dataset '{self.id}' has no traversal configured; call configure_cursor first.")
            self._cursor = DatasetCursor(
                self.id,
                self.metadata_backend.aspect_ratio_bucket_indices,
                repeats=int(StateTracker.get_data_backend_config(self.id).get("repeats") or 0),
                **self._cursor_options,
            )
        return self._cursor

    def rebuild_cursor(self) -> None:
        """Read the bucket lists again after they changed. The epoch and the seed are kept."""
        previous = self._cursor
        # A windowed dataset has no bucket lists; it indexes each shard again as it opens.
        if previous is None or isinstance(previous, WindowedCursor):
            return
        self._cursor_options["seed"] = previous.seed
        self._cursor = None
        self.cursor().epoch = previous.epoch

    def load_cursor_state(self, state: dict) -> None:
        """Continue this rank's traversal from the state a checkpoint saved for it."""
        allow_batch_size_change = os.environ.get("SIMPLETUNER_ALLOW_MODIFYING_BSZ", "").lower() in (
            "1",
            "true",
            "yes",
        ) or bool(getattr(StateTracker.get_args(), "i_know_what_i_am_doing", False))
        self.cursor().load_state(state, allow_batch_size_change=allow_batch_size_change)

    def describe(self, sample_ref: str) -> dict[str, Any] | None:
        """The geometry this dataset holds for a sample, or None when it holds none."""
        return self.metadata_backend.get_metadata_by_filepath(sample_ref)

    def read_samples(self, sample_refs: Sequence[str]) -> list[dict[str, Any]]:
        """Return independent occurrence metadata in request order, including duplicates."""
        records = []
        for sample_ref in sample_refs:
            metadata = self.describe(sample_ref)
            record = deepcopy(metadata) if metadata is not None else {}
            record.setdefault("dataset_type", self.config["dataset_type"])
            record["data_backend_id"] = self.id
            record["image_path"] = sample_ref
            records.append(record)
        return records

    def resolve_caption(self, sample_ref: str) -> str | list[str] | None:
        """Resolve caption alternatives without choosing an occurrence's alternative."""
        return PromptHandler.magic_prompt(
            image_path=sample_ref,
            data_backend=self.data_backend,
            sampler_backend_id=self.id,
            metadata_backend=self.metadata_backend,
            backend_config=self.config,
            caption_strategy=self.config["caption_strategy"],
            use_captions=self.config["use_captions"],
            prepend_instance_prompt=self.config["prepend_instance_prompt"],
            instance_prompt=self.config["instance_prompt"],
            disable_multiline_split=self.config["disable_multiline_split"],
        )

    def iter_caption_records(
        self,
        sample_refs: Iterable[str],
        *,
        on_missing_caption: Callable[[str], None] | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Visit the caller's sample selection without retaining captions or selecting alternatives."""
        for sample_ref in sample_refs:
            try:
                caption = self.resolve_caption(sample_ref)
            except CaptionNotFoundError:
                if on_missing_caption is None:
                    raise
                on_missing_caption(sample_ref)
                continue
            for value in PromptHandler._normalize_caption_payload(caption):
                yield value, sample_ref

    def iter_text_embed_records(
        self,
        model,
        sample_refs: Iterable[str],
        *,
        init_backend: dict[str, Any],
        on_missing_caption: Callable[[str], None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Build artifact requests incrementally, preserving model keys and caption variants."""
        from simpletuner.helpers.models.common import TextEmbedCacheKey

        key_type = model.text_embed_cache_key()
        dataset_root = self.config.get("instance_data_dir")
        for caption, sample_ref in self.iter_caption_records(sample_refs, on_missing_caption=on_missing_caption):
            image_path = str(sample_ref)
            relative_path = normalize_data_path(image_path, dataset_root)
            metadata = {
                "image_path": image_path,
                "data_backend_id": self.id,
                "prompt": caption,
                "dataset_relative_path": relative_path,
            }
            metadata_builder = getattr(model, "text_embed_cache_metadata_for_filepath", None)
            if callable(metadata_builder):
                metadata.update(
                    metadata_builder(
                        init_backend=init_backend,
                        image_path=image_path,
                        prompt=caption,
                        data_backend_id=self.id,
                        dataset_relative_path=relative_path,
                    )
                )
            if key_type is TextEmbedCacheKey.DATASET_AND_FILENAME:
                key = f"{self.id}:{relative_path}"
            elif key_type is TextEmbedCacheKey.FILENAME:
                key = normalize_data_path(image_path, None)
            else:
                key = caption
            key_builder = getattr(model, "text_embed_cache_key_value", None)
            if callable(key_builder):
                key = key_builder(prompt=caption, default_key=key, metadata=metadata)
            yield {"prompt": caption, "key": key, "metadata": metadata}

            sample_metadata = self.describe(image_path)
            if sample_metadata is not None:
                for index, entity in enumerate(sample_metadata.get("bbox_entities") or []):
                    label = entity.get("label", "")
                    if label:
                        yield {
                            "prompt": label,
                            "key": f"{relative_path}__bbox_{index}",
                            "metadata": {"grounding_entity": True},
                        }
