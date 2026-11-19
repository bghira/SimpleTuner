"""Dataset access shared by training and preparation consumers."""

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import deepcopy
from itertools import batched, islice
from typing import Any

from simpletuner.helpers.data_backend.base import BaseDataBackend, ordered_sample_payloads
from simpletuner.helpers.data_backend.dataset_types import get_arg_value
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.metadata.preparation import DatasetReadView
from simpletuner.helpers.prompts import CaptionNotFoundError, PromptHandler
from simpletuner.helpers.training import video_file_extensions
from simpletuner.helpers.utils.pathing import normalize_data_path


class DatasetRuntime:
    @classmethod
    def from_config(cls, config, args, accelerator, *, preparation_only=False):
        """Open one standalone dataset with resolved source and metadata providers."""
        from simpletuner.helpers.data_backend.builders import create_backend_builder
        from simpletuner.helpers.data_backend.config import create_backend_config

        args_dict = (
            args if isinstance(args, dict) else {key: value for key, value in vars(args).items() if not key.startswith("_")}
        )
        settings = create_backend_config(config, args_dict)
        builder = create_backend_builder(settings.backend_type, accelerator, args)
        source = builder.build(settings)
        metadata = builder.create_metadata_backend(settings, source, args_dict, preparation_only=preparation_only)
        resolved = {**config, **settings.to_dict()["config"], "instance_data_dir": metadata.instance_data_dir}
        return cls(settings.id, source, metadata, {**resolved, **cls.caption_config(resolved, args)})

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
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config
        if isinstance(self.metadata_backend, MetadataBackend):
            self.metadata_backend.bind_dataset_config(config)

    @property
    def epoch_provider(self):
        return self.metadata_backend.epoch_provider

    @property
    def uses_native_schedule(self) -> bool:
        return self.epoch_provider is not None

    @property
    def recrops_lazily(self) -> bool:
        return getattr(self.metadata_backend, "recrops_lazily", False) is True

    def prepare_epoch(self, epoch, *, rebuild_random_aspect, gradient_accumulation_steps, apply_padding):
        """Refresh prepared data before the sampler and prefetcher start the next epoch."""
        metadata, cache = self.metadata_backend, self.vae_cache
        rebuilt = metadata.refresh_epoch_geometry(
            epoch,
            rebuild_random_aspect=rebuild_random_aspect,
            gradient_accumulation_steps=gradient_accumulation_steps,
            apply_padding=apply_padding,
        )
        if cache is None:
            return
        clear_latents = self.config.get("vae_cache_clear_each_epoch")
        if self.requires_complete_preparation is True:
            accelerator = metadata.accelerator
            if clear_latents and self.uses_native_schedule:
                cache.expire_epoch_latents()
                if not self.recrops_lazily and not cache.vae_cache_ondemand:
                    cache.prepare_dataset(self, worker=accelerator.process_index, workers=accelerator.num_processes)
                    accelerator.wait_for_everyone()
            elif clear_latents:
                cache.rebuild_prepared_cache(self, worker=accelerator.process_index, workers=accelerator.num_processes)
            elif rebuilt and not cache.vae_cache_ondemand:
                cache.prepare_dataset(self, worker=accelerator.process_index, workers=accelerator.num_processes)
        elif rebuilt or clear_latents:
            cache.rebuild_cache()

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

    @property
    def supports_validation_samples(self):
        return self.sampler is not None

    def validation_samples(self, *, batch_size):
        return self.sampler.retrieve_validation_set(batch_size=batch_size)

    def acknowledge_batch(self, continuation):
        """Commit the provider's continuation for a consumed training batch."""
        self.sampler.acknowledge_batch(continuation)

    def save_checkpoint(self, state_path):
        if self.sampler is not None:
            self.sampler.save_state(state_path=state_path)

    def restore_checkpoint(self, state_path):
        if self.sampler is not None:
            self.sampler.load_states(state_path=state_path)

    def log_sampling_state(self):
        if self.sampler is not None:
            self.sampler.log_state()

    def abort(self):
        for component in (self.vae_cache, self.metadata_backend, self.text_embed_cache, self.sampler):
            if component is not None:
                component.should_abort = True

    @staticmethod
    def caption_config(config, args=None):
        strategy = config.get("caption_strategy", get_arg_value(args, "caption_strategy", "filename"))
        return {
            "caption_strategy": strategy,
            "use_captions": not (
                config.get("only_instance_prompt")
                or get_arg_value(args, "only_instance_prompt", False)
                or strategy == "instanceprompt"
            ),
            "prepend_instance_prompt": config.get(
                "prepend_instance_prompt", get_arg_value(args, "prepend_instance_prompt", False)
            ),
            "instance_prompt": config.get("instance_prompt", get_arg_value(args, "instance_prompt")),
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
        if self.recrops_lazily:
            inputs["epoch_recrop"] = self.metadata_backend.epoch_recrop_version
        if not self.uses_native_schedule:
            revision = self.describe().source_revision
            if isinstance(revision, str):
                inputs["source_revision"] = revision
        serialized = json.dumps(make_public_config_serializable(inputs), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    def open_epoch(self, spec):
        return self.epoch_provider.open_epoch(spec)

    def published_read_view(self):
        if not self.uses_native_schedule:
            return None
        revision = self.metadata_backend.view_revision
        if revision is None:
            raise ValueError("Prepare the dataset before opening its published view.")
        return DatasetReadView(revision)

    def open_validation(self):
        if self.uses_native_schedule:
            return self.epoch_provider.open_validation()
        return ValidationCursor(self.metadata_backend.iter_validation_cycles)

    def read_sample_images(self, sample_refs, *, allow_missing=False):
        refs = list(sample_refs)
        if not refs:
            return []
        paths, images = self.data_backend.read_image_batch(refs)
        return list(ordered_sample_payloads(self.id, refs, zip(paths, images, strict=True), allow_missing=allow_missing))

    def read_samples(
        self, sample_refs: Sequence[str], *, fields=("metadata",), view: DatasetReadView | None = None
    ) -> list[dict[str, Any]]:
        """Read fields for the selected view. Unscoped reads use the provider's published geometry."""
        fields = frozenset(fields)
        unknown = fields.difference({"metadata", "image", "caption"})
        if unknown:
            raise ValueError(f"Unknown sample fields: {sorted(unknown)}")
        if "metadata" not in fields:
            if view is not None:
                self.metadata_backend.validate_read_view(view)
            geometry = (None for _ in sample_refs)
        elif view is None:
            geometry = self.metadata_backend.get_metadata_batch(sample_refs)
        else:
            geometry = self.metadata_backend.get_view_metadata(sample_refs, view)
        payloads = None
        if "image" in fields:
            payloads = iter(self.read_sample_images(sample_refs))
        captions = iter(self.resolve_captions(sample_refs)) if "caption" in fields else None
        records = []
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
        return self.metadata_backend.conditioning_ref(relative_ref)

    def read_conditioning_sample(self, relative_ref, *, paired_source, rng=None):
        options = {"paired_source": paired_source}
        if rng is not None:
            options["rng"] = rng
        return self.sampler.get_conditioning_sample(relative_ref, **options)

    def resolve_s2v_audio(self, video_ref):
        """Resolve a video to its attached audio dataset and sample reference."""
        key = self._s2v_sidecar_key(video_ref)
        for audio in self.audio_datasets:
            if audio.config.get("audio", {}).get("source_from_video"):
                if audio.metadata_backend is None or audio.metadata_backend.get_metadata_by_filepath(video_ref) is not None:
                    root = audio.config.get("instance_data_dir")
                    is_local = getattr(audio.data_backend, "type", None) == "local"
                    ref = os.path.join(root, video_ref) if root and is_local else video_ref
                    return audio, ref
                continue
            root = audio.config.get("instance_data_dir")
            if root is None or key is None:
                continue
            roots = (root,) if isinstance(root, str) else root
            for directory in roots:
                for extension in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                    ref = os.path.join(directory, key + extension)
                    if audio.data_backend.exists(ref):
                        return audio, ref
        return None

    def _s2v_sidecar_key(self, video_ref):
        root = self.config.get("instance_data_dir") or getattr(self.metadata_backend, "instance_data_dir", None)
        if getattr(self.data_backend, "type", None) == "aws":
            if isinstance(root, (list, tuple)):
                root = root[0] if len(root) == 1 else None
            prefix = root.rstrip("/") if root else ""
            relative_ref = video_ref.removeprefix(prefix + "/") if prefix else video_ref
        else:
            relative_ref = normalize_data_path(video_ref, root)
        if not relative_ref or os.path.isabs(relative_ref) or "://" in relative_ref or ".." in relative_ref.split("/"):
            return None
        return os.path.splitext(relative_ref)[0]

    def read_audio_sample(self, ref):
        """Read a waveform and sample rate from the bound audio provider."""
        audio = self.config.get("audio") or {}
        extension = os.path.splitext(ref)[1].lower().lstrip(".")
        if audio.get("source_from_video") and extension in video_file_extensions:
            from simpletuner.helpers.audio.load import NoAudioStreamError, generate_zero_audio

            sample_rate = audio.get("sample_rate", 16000)
            channels = audio.get("channels", 1)
            try:
                return self.data_backend.read_embedded_audio(ref, sample_rate=sample_rate, channels=channels)
            except NoAudioStreamError:
                if not audio.get("allow_zero_audio"):
                    raise
                (record,) = self.read_samples([ref])
                duration = record.get("duration_seconds")
                if not duration:
                    raise
                return generate_zero_audio(duration, sample_rate, channels)
        sample = self.data_backend.read_image(ref)
        if sample is None:
            raise FileNotFoundError(f"Could not read audio {ref} from dataset {self.id}.")
        return sample

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
        strategy = self.config["caption_strategy"]
        if strategy and self.config["use_captions"]:
            for provider in (self.metadata_backend, self.data_backend):
                if strategy == getattr(provider, "caption_projection_strategy", None):
                    return dict(zip(refs, provider.read_caption_batch(refs), strict=True)).__getitem__
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

    def prepare_view(self, *, force_rescan=False, progress_callback=None):
        self.metadata_backend.compute_aspect_ratio_bucket_indices(
            ignore_existing_cache=force_rescan, progress_callback=progress_callback
        )

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
        builder=None,
        worker: int = 0,
        workers: int = 1,
        batch_size: int = 1,
        on_missing_caption: Callable[[str], None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        if workers < 1 or not 0 <= worker < workers:
            raise ValueError("Preparation worker must be in the range [0, workers).")
        if builder is None:
            builder = TextRequestBuilder(model, self)
        with self.metadata_backend.open_preparation_view(worker=worker, workers=workers) as view:
            captions = self.iter_caption_records(view, batch_size=batch_size, on_missing_caption=on_missing_caption)
            yield from builder.records(captions, read_metadata=view.metadata.get_metadata_by_filepath)


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
    @staticmethod
    def needs_conditioning_datasets(model, linked_ids, args):
        if not linked_ids:
            return False
        if getattr(model, "requires_text_embed_image_context", lambda: False)():
            return True
        return (
            getattr(args, "conditioning_multidataset_sampling", None) == "random"
            and not isinstance(linked_ids, str)
            and len(linked_ids) > 1
        )

    def __init__(
        self,
        model,
        dataset,
        *,
        conditioning_datasets=None,
        args=None,
    ):
        self.model = model
        self.dataset = dataset
        self.dataset_root = dataset.metadata_backend.instance_data_dir
        ordered = sorted(
            conditioning_datasets or (),
            key=lambda entry: entry.config.get("conditioning_type") not in ("reference_strict", "reference_loose"),
        )
        self.references = {
            entry.id: entry for entry in ordered if entry.config.get("conditioning_type") not in ("mask", "segmentation")
        }
        self.conditioning_datasets = ordered
        self.include_grounding = (
            conditioning_datasets is None if args is None else bool(getattr(args, "max_grounding_entities", 0))
        )
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

    def _contexts(self, metadata, caption, image_path, related_samples):
        if self.sampling != "random" or len(self.conditioning_datasets) <= 1 or not related_samples:
            yield metadata, caption
            return
        for dataset, path in related_samples:
            context = dict(metadata)
            if dataset.id in self.references:
                context.update(image_paths=[path], data_backend_ids=[dataset.id])
            else:
                context.pop("image_paths", None)
                context.pop("data_backend_ids", None)
            if dataset.config.get("source_dataset_id") is not None and dataset.config.get("caption_strategy") is None:
                dataset, path = self.dataset, image_path
            alternatives = dataset.resolve_caption(path) or [caption]
            for alternative in alternatives if isinstance(alternatives, list) else [alternatives]:
                yield context, alternative or caption

    def records(self, captions, *, read_metadata):
        key_type = self.model.text_embed_cache_key()
        dataset_id = self.dataset.id
        metadata_builder = getattr(self.model, "text_embed_cache_metadata_for_preparation", None)
        for caption, sample_ref in captions:
            image_path = str(sample_ref)
            relative_path = normalize_data_path(image_path, self.dataset_root)
            metadata = dict(
                image_path=image_path, data_backend_id=dataset_id, prompt=caption, dataset_relative_path=relative_path
            )
            relation_sources = (
                self.conditioning_datasets
                if self.sampling == "random" and len(self.conditioning_datasets) > 1
                else self.references.values()
            )
            related_samples = [
                (dataset, path)
                for dataset in relation_sources
                if (path := dataset.conditioning_ref(relative_path)) is not None
            ]
            references = [(entry.id, path) for entry, path in related_samples if entry.id in self.references]
            if references:
                metadata.update(
                    image_paths=[path for _, path in references],
                    data_backend_ids=[dataset_id for dataset_id, _ in references],
                )

            def request(context, text):
                details = dict(context, prompt=text)
                if callable(metadata_builder):
                    details.update(
                        metadata_builder(
                            read_metadata=read_metadata,
                            image_path=image_path,
                            prompt=text,
                            data_backend_id=dataset_id,
                            dataset_relative_path=relative_path,
                        )
                    )
                return text_embed_request(self.model, key_type, text, details, dataset_id, image_path, relative_path)

            for context, text in self._contexts(metadata, caption, image_path, related_samples):
                yield request(context, text)
                if self.include_dropout and text:
                    yield request(context, "")
            if self.include_grounding:
                sample = read_metadata(image_path)
                if isinstance(sample, dict):
                    for index, entity in enumerate(sample.get("bbox_entities") or ()):
                        if label := entity.get("label"):
                            yield {
                                "prompt": label,
                                "key": f"{relative_path}__bbox_{index}",
                                "metadata": {"grounding_entity": True},
                            }
