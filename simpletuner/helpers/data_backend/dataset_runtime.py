"""Dataset access shared by training and preparation consumers."""

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import deepcopy
from itertools import batched, islice
from typing import Any

from simpletuner.helpers.data_backend.base import BaseDataBackend, ordered_sample_payloads
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.prompts import CaptionNotFoundError, PromptHandler
from simpletuner.helpers.utils.pathing import normalize_data_path


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
        self.text_embed_cache = None
        self.vae_cache = None
        self.sampler = None
        self.training_dataset = None
        self.training_dataloader = None
        self.conditioning_datasets = ()
        self.audio_datasets = ()
        self.conditioning_image_embed_cache = None
        self.grounding_image_embed_cache = None
        self.config = {
            "dataset_type": "image",
            "caption_strategy": "filename",
            "use_captions": True,
            "prepend_instance_prompt": False,
            "instance_prompt": None,
            "disable_multiline_split": False,
            **deepcopy(config),
        }

    @property
    def epoch_provider(self):
        return self.metadata_backend.epoch_provider

    @property
    def uses_native_schedule(self) -> bool:
        return self.epoch_provider is not None

    def start_epoch(self, epoch):
        if self.uses_native_schedule:
            self.sampler.start_native_epoch(epoch)
        elif self.tracks_consumed_batches:
            self.sampler.start_training_epoch(epoch)

    @property
    def tracks_consumed_batches(self) -> bool:
        return getattr(self.sampler, "tracks_consumed_batches", False) is True

    @property
    def supports_consumed_checkpoint(self) -> bool:
        return self.uses_native_schedule or self.tracks_consumed_batches

    def __iter__(self):
        if not self.tracks_consumed_batches:
            yield from self.training_dataloader
            return
        for batch in self.training_dataloader:
            batch["dataset_continuation"] = self.sampler.fetched_state()
            yield batch

    def training_batch_count(self) -> int:
        """Count the per-rank batches scheduled by the training dataloader."""
        return len(self.training_dataloader)

    def reset_evaluation(self):
        self.sampler.reset_evaluation()

    def acknowledge_batch(self, continuation):
        """Commit the provider's continuation for a consumed training batch."""
        self.sampler.acknowledge_batch(continuation)

    @staticmethod
    def caption_config(config, args=None):
        strategy = config.get("caption_strategy", getattr(args, "caption_strategy", "filename"))
        return {
            "caption_strategy": strategy,
            "use_captions": not (
                config.get("only_instance_prompt")
                or getattr(args, "only_instance_prompt", False)
                or strategy == "instanceprompt"
            ),
            "prepend_instance_prompt": config.get(
                "prepend_instance_prompt", getattr(args, "prepend_instance_prompt", False)
            ),
            "instance_prompt": config.get("instance_prompt", getattr(args, "instance_prompt", None)),
            "disable_multiline_split": config.get("disable_multiline_split", False),
        }

    def close(self):
        self.metadata_backend.close()

    def describe(self, spec=None):
        """Report source and accepted counts, plus global occurrences and per-rank batches for a native epoch."""
        if spec is None:
            return self.metadata_backend.describe()
        return self.metadata_backend.describe(spec)

    def count_source_rows(self, *, exclude_null_images=False):
        """Request an exact source count. Providers may need to open shards."""
        return self.data_backend.count_source_rows(exclude_null_images=exclude_null_images)

    def read_page(self, *, bucket=None, offset=0, limit=50):
        return self.metadata_backend.read_page(bucket=bucket, offset=offset, limit=limit)

    def occurrence_revision(self, args):
        """Identify inputs that can change the caption, conditioning or transform for a scheduled sample."""
        from simpletuner.helpers.configuration.sanitization import make_public_config_serializable

        def describe_binding(dataset):
            description = dataset.describe()
            return [dataset.id, description.source_revision, description.view_revision, dataset.config]

        inputs = {
            "dataset": self.config,
            "conditioning": [describe_binding(dataset) for dataset in self.conditioning_datasets],
            "audio": [describe_binding(dataset) for dataset in self.audio_datasets],
            "conditioning_sampling": getattr(args, "conditioning_multidataset_sampling", "combined"),
            "caption_dropout": getattr(args, "caption_dropout_probability", None),
        }
        serialized = json.dumps(make_public_config_serializable(inputs), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    def open_epoch(self, spec):
        return self.epoch_provider.open_epoch(spec)

    def open_validation(self):
        if self.uses_native_schedule:
            return self.epoch_provider.open_validation()
        return ValidationCursor(self.metadata_backend.iter_validation_cycles)

    def read_samples(self, sample_refs: Sequence[str], *, fields=("metadata",)) -> list[dict[str, Any]]:
        """Read a batch of occurrence records with the requested metadata, image or caption."""
        fields = frozenset(fields)
        unknown = fields.difference({"metadata", "image", "caption"})
        if unknown:
            raise ValueError(f"Unknown sample fields: {sorted(unknown)}")
        payloads = None
        if sample_refs and "image" in fields:
            paths, images = self.data_backend.read_image_batch(list(sample_refs))
            payloads = ordered_sample_payloads(self.id, sample_refs, zip(paths, images, strict=True))
        captions = iter(self.resolve_captions(sample_refs)) if "caption" in fields else None
        records = []
        if "metadata" not in fields:
            geometry = (None for _ in sample_refs)
        else:
            geometry = self.metadata_backend.get_metadata_batch(sample_refs)
        for sample_ref, metadata in zip(sample_refs, geometry, strict=True):
            record = deepcopy(metadata) if metadata is not None else {}
            record.setdefault("dataset_type", self.config["dataset_type"])
            record["data_backend_id"] = self.id
            record["image_path"] = sample_ref
            if captions is not None:
                record["caption"] = next(captions)
            if payloads is not None:
                record["image"] = next(payloads)
            records.append(record)
        return records

    def pairing_key(self, sample_ref: str) -> str:
        return normalize_data_path(sample_ref, self.metadata_backend.instance_data_dir)

    def conditioning_ref(self, relative_ref):
        """Resolve a training sample's pairing key inside this conditioning dataset."""
        from simpletuner.helpers.data_backend.webshart import WebshartDataBackend

        if isinstance(self.data_backend, WebshartDataBackend):
            self.data_backend._resolve_sample_ref(relative_ref)
            return self.data_backend.normalize_sample_id(relative_ref)
        root = self.config.get("instance_data_dir") or self.metadata_backend.instance_data_dir
        return self.conditioning_file_path(relative_ref, root, self.config)

    @staticmethod
    def conditioning_file_path(relative_ref, root, config):
        from simpletuner.helpers.data_backend.webshart import WebshartDataBackend

        if WebshartDataBackend.is_sample_id(relative_ref):
            relative_ref = WebshartDataBackend.parse_sample_id(relative_ref).filename
            if os.path.isabs(relative_ref) or ".." in relative_ref.split("/"):
                raise ValueError("Webshart pairing requires a filename relative to the conditioning dataset.")
        path = os.path.join(root, relative_ref) if root else relative_ref
        if (config.get("conditioning_config") or {}).get("type") == "i2v_first_frame":
            path = os.path.splitext(path)[0] + ".png"
        return path

    def resolve_captions(self, sample_refs):
        lookup = self._caption_lookup(sample_refs)
        return [self.resolve_caption(ref, caption_lookup=lookup) for ref in sample_refs]

    def resolve_caption(self, sample_ref: str, *, caption_lookup=None) -> str | list[str] | None:
        """Resolve caption alternatives without choosing an occurrence's alternative."""
        return PromptHandler.magic_prompt(
            image_path=sample_ref,
            data_backend=self.data_backend,
            sampler_backend_id=self.id,
            metadata_backend=self.metadata_backend,
            caption_lookup=caption_lookup,
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
        batch_size: int = 1,
        on_missing_caption: Callable[[str], None] | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Visit the caller's sample selection without retaining captions or selecting alternatives."""
        for refs in batched(sample_refs, batch_size):
            lookup = self._caption_lookup(refs)
            for sample_ref in refs:
                try:
                    caption = self.resolve_caption(sample_ref, caption_lookup=lookup)
                except CaptionNotFoundError:
                    if on_missing_caption is None:
                        raise
                    on_missing_caption(sample_ref)
                    continue
                for value in PromptHandler._normalize_caption_payload(caption):
                    yield value, sample_ref

    def _caption_lookup(self, refs):
        if (
            self.config["caption_strategy"] == "webshart"
            and self.config["use_captions"]
            and getattr(self.data_backend, "supports_caption_projection", False) is True
        ):
            return dict(zip(refs, self.data_backend.read_caption_batch(refs), strict=True)).__getitem__
        return None

    @property
    def requires_complete_preparation(self) -> bool:
        return self.metadata_backend.requires_complete_preparation

    def iter_preparation_batches(self, *, worker=0, workers=1, batch_size):
        """Yield shape-compatible batches from a partition of accepted source samples."""
        if workers < 1 or not 0 <= worker < workers:
            raise ValueError("Preparation worker must be in the range [0, workers).")
        if batch_size < 1:
            raise ValueError("Preparation batch size must be positive.")
        return self.metadata_backend.iter_preparation_batches(worker, workers, batch_size)

    def scan_cache_and_metadata(self, cache_behavior):
        self.metadata_backend.scan_dataset_cache(self, cache_behavior)

    def iter_accepted_captions(self, *, worker=0, workers=1, batch_size=128):
        if self.config["dataset_type"] == "caption":
            for record in self.metadata_backend.iter_records(worker=worker, workers=workers):
                yield record.caption_text
            return
        with self.metadata_backend.open_preparation_view(worker=worker, workers=workers) as view:
            for caption, _ in self.iter_caption_records(view, batch_size=batch_size, on_missing_caption=lambda ref: None):
                yield caption

    def iter_preparation_text_records(
        self,
        model,
        *,
        init_backend: dict[str, Any],
        builder=None,
        worker: int = 0,
        workers: int = 1,
        batch_size: int = 1,
        on_missing_caption: Callable[[str], None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        if workers < 1 or not 0 <= worker < workers:
            raise ValueError("Preparation worker must be in the range [0, workers).")
        if builder is None:
            builder = TextRequestBuilder(model, self.id, self.config.get("instance_data_dir"))
        with self.metadata_backend.open_preparation_view(worker=worker, workers=workers) as view:
            captions = self.iter_caption_records(view, batch_size=batch_size, on_missing_caption=on_missing_caption)
            yield from builder.records(captions, {**init_backend, "metadata_backend": view.metadata})

    def iter_text_embed_records(
        self,
        model,
        sample_refs: Iterable[str],
        *,
        init_backend: dict[str, Any],
        on_missing_caption: Callable[[str], None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        builder = TextRequestBuilder(model, self.id, self.config.get("instance_data_dir"))
        captions = self.iter_caption_records(sample_refs, on_missing_caption=on_missing_caption)
        yield from builder.records(captions, {"metadata_backend": self.metadata_backend, **init_backend})


class ValidationCursor:
    """Keep validation order and wraparound independent of training progress."""

    def __init__(self, iter_cycles):
        self.iter_cycles = iter_cycles
        self.cycles = None
        self.refs = None
        self.position = 0

    def close(self):
        if self.cycles is not None:
            self.cycles.close()
        self.cycles = self.refs = None
        self.position = 0

    def next_refs(self, limit):
        if limit <= 0:
            return []
        if self.cycles is None:
            self.cycles = self.iter_cycles()
            self.refs = next(self.cycles)
        selected = list(islice(self.refs, limit))
        self.position += len(selected)
        if len(selected) < limit:
            remaining = min(self.position, limit) - len(selected)
            self.refs = next(self.cycles)
            selected.extend(islice(self.refs, remaining))
            self.position = remaining
        return selected


def text_embed_request(model, key_type, prompt, metadata, dataset_id, image_path, relative_path):
    from simpletuner.helpers.models.common import TextEmbedCacheKey

    if key_type is TextEmbedCacheKey.DATASET_AND_FILENAME and dataset_id and image_path:
        default_key = f"{dataset_id}:{relative_path}"
    elif key_type is TextEmbedCacheKey.FILENAME and image_path:
        default_key = normalize_data_path(image_path, None)
    else:
        default_key = prompt
    key_builder = getattr(model, "text_embed_cache_key_value", None)
    key = key_builder(prompt=prompt, default_key=default_key, metadata=metadata) if callable(key_builder) else default_key
    return {"prompt": prompt, "key": key, "metadata": metadata}


class TextRequestBuilder:
    def __init__(
        self,
        model,
        dataset_id,
        dataset_root,
        *,
        conditioning_datasets=None,
        args=None,
    ):
        self.model = model
        self.dataset_id = dataset_id
        self.dataset_root = dataset_root
        ordered = sorted(
            conditioning_datasets or (),
            key=lambda entry: entry.config.get("conditioning_type") not in ("reference_strict", "reference_loose"),
        )
        self.references = {
            entry.id: entry for entry in ordered if entry.config.get("conditioning_type") not in ("mask", "segmentation")
        }
        self.conditioning_datasets = ordered
        self.deferred = conditioning_datasets is not None
        self.sampling = getattr(args, "conditioning_multidataset_sampling", None) or "combined"
        self.include_dropout = False
        if args is not None and getattr(model, "uses_image_context_dropout_caption_cache", lambda: False)():
            from simpletuner.helpers.distillation.factory import DistillerFactory

            needs = DistillerFactory.training_batch_requirements(
                getattr(args, "distillation_method", None), {"distillation_config": getattr(args, "distillation_config", {})}
            )
            probability = getattr(args, "caption_dropout_probability", 0.1)
            self.include_dropout = (
                0.1 if probability is None else probability
            ) > 0 or "unconditional_text_embeddings" in needs

    def _contexts(self, metadata, caption, image_path, init_backend):
        if self.sampling != "random" or len(self.conditioning_datasets) <= 1:
            yield metadata, caption
            return
        for dataset in self.conditioning_datasets:
            path = dataset.conditioning_ref(metadata["dataset_relative_path"])
            context = dict(metadata)
            if dataset.id in self.references:
                context.update(image_paths=[path], data_backend_ids=[dataset.id])
            else:
                context.pop("image_paths", None)
                context.pop("data_backend_ids", None)
            if dataset.config.get("source_dataset_id") is not None and dataset.config["caption_strategy"] is None:
                dataset, path = init_backend["dataset_runtime"], image_path
            alternatives = dataset.resolve_caption(path) or [caption]
            for alternative in alternatives if isinstance(alternatives, list) else [alternatives]:
                yield context, alternative or caption

    def records(self, captions, init_backend):
        key_type = self.model.text_embed_cache_key()
        for caption, sample_ref in captions:
            image_path = str(sample_ref)
            relative_path = normalize_data_path(image_path, self.dataset_root)
            metadata = dict(
                image_path=image_path, data_backend_id=self.dataset_id, prompt=caption, dataset_relative_path=relative_path
            )
            if self.references:
                paths = [entry.conditioning_ref(relative_path) for entry in self.references.values()]
                metadata.update(image_paths=paths, data_backend_ids=list(self.references))
            metadata_builder = getattr(self.model, "text_embed_cache_metadata_for_filepath", None)

            def request(context, text):
                details = dict(context, prompt=text)
                if callable(metadata_builder):
                    details.update(
                        metadata_builder(
                            init_backend=init_backend,
                            image_path=image_path,
                            prompt=text,
                            data_backend_id=self.dataset_id,
                            dataset_relative_path=relative_path,
                        )
                    )
                return text_embed_request(self.model, key_type, text, details, self.dataset_id, image_path, relative_path)

            for context, text in self._contexts(metadata, caption, image_path, init_backend):
                yield request(context, text)
                if self.include_dropout and text:
                    yield request(context, "")
            if not self.deferred:
                sample = init_backend["metadata_backend"].get_metadata_by_filepath(image_path)
                if isinstance(sample, dict):
                    for index, entity in enumerate(sample.get("bbox_entities") or ()):
                        if label := entity.get("label"):
                            yield {
                                "prompt": label,
                                "key": f"{relative_path}__bbox_{index}",
                                "metadata": {"grounding_entity": True},
                            }
