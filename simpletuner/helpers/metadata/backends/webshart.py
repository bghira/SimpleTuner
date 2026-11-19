import json
import logging
import os
import random
import shutil
import sqlite3
import subprocess
import tempfile
import time
from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager, nullcontext
from copy import copy, deepcopy
from hashlib import file_digest, sha256
from itertools import batched, chain, islice
from pathlib import Path
from queue import SimpleQueue
from threading import Event
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from uuid import uuid4

import ijson
from filelock import FileLock
from tqdm import tqdm

from simpletuner.helpers.configuration.sanitization import make_public_config_serializable
from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.data_backend.webshart import CaptionValue, WebshartDataBackend
from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.metadata.backends.base import MetadataBackend, get_cp_aware_dp_info
from simpletuner.helpers.metadata.backends.webshart_epoch import EpochSpec, WebshartEpoch
from simpletuner.helpers.metadata.backends.webshart_exclusions import (
    collect_exclusions,
    exclusion_path,
    retain_source_exclusions,
)
from simpletuner.helpers.metadata.backends.webshart_geometry import (
    _read_index,
    accepted_geometry_predicate,
    geometry_lookup,
    link_snapshot,
    open_geometry,
    read_snapshot,
    snapshot_path,
    stage_geometry,
)
from simpletuner.helpers.metadata.backends.webshart_membership import (
    IndexedValidationCursor,
    MembershipPages,
    publish_indexed_membership,
)
from simpletuner.helpers.metadata.backends.webshart_transforms import (
    PreparationGeometry,
    WebshartTransforms,
    prepared_fields,
    recrop_record,
)
from simpletuner.helpers.metadata.preparation import (
    DatasetDescription,
    DatasetReadView,
    PreparationView,
    bucket_groups_from_events,
)
from simpletuner.helpers.multiaspect.image import ImageGeometry, MultiaspectImage
from simpletuner.helpers.prompts import CaptionNotFoundError
from simpletuner.helpers.training import video_file_extensions
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger("WebshartMetadataBackend")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


def _empty_filtering_statistics():
    return {
        "total_processed": 0,
        "skipped": {"already_exists": 0, "metadata_missing": 0, "caption_missing": 0, "too_small": 0, "error": 0},
    }


class WebshartMetadataBackend(MetadataBackend):
    native_policy = None
    epoch_recrop_version = 1
    maximum_canvas_size = None
    collect_exclusions = collect_exclusions

    def __init__(
        self,
        id: str,
        instance_data_dir: str,
        cache_file: str,
        metadata_file: str,
        data_backend: WebshartDataBackend,
        accelerator,
        batch_size: int,
        resolution: float,
        resolution_type: str,
        delete_problematic_images: bool = False,
        delete_unwanted_images: bool = False,
        metadata_update_interval: int = 3600,
        minimum_image_size: int = None,
        minimum_aspect_ratio: int = None,
        maximum_aspect_ratio: int = None,
        num_frames: int = None,
        minimum_num_frames: int = None,
        maximum_num_frames: int = None,
        cache_file_suffix: str = None,
        repeats: int = 0,
        dataset_config: dict | None = None,
        runtime_args=None,
        max_num_samples: int = None,
        preparation_only: bool = False,
        maximum_canvas_size: int | None = None,
    ):
        if not isinstance(data_backend, WebshartDataBackend):
            raise ValueError("WebshartMetadataBackend requires WebshartDataBackend")
        if data_backend.format == "parquet":
            fingerprint = data_backend.dataset.fingerprint[:16]
            cache_file = f"{cache_file}_source_{fingerprint}"
            metadata_file = f"{metadata_file}_source_{fingerprint}"
        if data_backend.caption_key is not None:
            caption_digest = sha256(json.dumps(data_backend.caption_key).encode("utf-8")).hexdigest()[:16]
            cache_file = f"{cache_file}_captions_{caption_digest}"
            metadata_file = f"{metadata_file}_captions_{caption_digest}"
        super().__init__(
            id=id,
            instance_data_dir=instance_data_dir,
            cache_file=cache_file,
            metadata_file=metadata_file,
            data_backend=data_backend,
            accelerator=accelerator,
            batch_size=batch_size,
            resolution=resolution,
            resolution_type=resolution_type,
            delete_problematic_images=delete_problematic_images,
            delete_unwanted_images=delete_unwanted_images,
            metadata_update_interval=metadata_update_interval,
            minimum_image_size=minimum_image_size,
            minimum_aspect_ratio=minimum_aspect_ratio,
            maximum_aspect_ratio=maximum_aspect_ratio,
            maximum_num_frames=maximum_num_frames,
            minimum_num_frames=minimum_num_frames,
            num_frames=num_frames,
            dataset_config=dataset_config,
            runtime_args=runtime_args,
            cache_file_suffix=cache_file_suffix,
            repeats=repeats,
            max_num_samples=max_num_samples,
            load_cache_on_init=False,
        )
        if self.dataset_type not in {DatasetType.IMAGE, DatasetType.VIDEO, DatasetType.CONDITIONING, DatasetType.EVAL}:
            raise ValueError("WebshartMetadataBackend supports image, video, conditioning, and eval datasets only.")

        context = accelerator.main_process_first() if hasattr(accelerator, "main_process_first") else nullcontext()
        self._geometry_digest = None
        self.maximum_canvas_size = maximum_canvas_size
        self.native_policy = (self.dataset_config.get("webshart") or {}).get("sampling_policy")
        self._artifact_models = {}
        self._snapshot = None
        self._accepted_geometry = None
        self._published = None
        self._schedule_padding = None
        self._stored_geometry_digest = None
        self._transforms = None
        self._writing_transforms = None
        with context:
            if self.native_policy or not preparation_only:
                self.reload_cache(set_config=not preparation_only)
        if hasattr(accelerator, "wait_for_everyone"):
            accelerator.wait_for_everyone()

    def _bucketed_sample_ids(self) -> list[str]:
        sample_ids = []
        seen = set()
        for bucket in self.aspect_ratio_bucket_indices.values():
            for sample_path in bucket:
                sample_path = str(sample_path)
                if sample_path in seen:
                    continue
                sample_ids.append(sample_path)
                seen.add(sample_path)
        return sample_ids

    def _sync_image_files_with_buckets(self) -> None:
        sample_ids = self._bucketed_sample_ids()
        if not sample_ids:
            return
        self.data_backend.source_files.publish([("", [], sample_ids)])

    def caption_cache_entry(self, index: str) -> Optional[CaptionValue]:
        return self.data_backend.get_caption(self.data_backend.normalize_sample_id(index))

    def reload_cache(self, set_config: bool = True):
        if self.native_policy:
            self.close()
            self._published = None
            if self.data_backend.exists(self.cache_file):
                with self._open_membership() as source:
                    pages = MembershipPages.open(source)
                    if pages is not None and pages.header.get("geometry_sha256"):
                        self._published = pages.header
                        self._geometry_digest = pages.header["geometry_sha256"]
                        if pages.header.get("source_pins"):
                            with ExitStack() as stack:
                                self._lookup = self._open_geometry(stack, self._published, self._geometry_digest)
                                if settings := pages.header.get("transform_cache_recipe"):
                                    self._transforms = stack.enter_context(
                                        closing(self._open_transforms(settings, self._lookup.shard_revision))
                                    )
                                self.data_backend.dataset.bind_source_revisions(self._lookup.shard_revision)
                                self._snapshot = stack.pop_all()
                        self.filtering_statistics = pages.header.get("filtering_statistics")
                        if set_config:
                            self.config = pages.header.get("config", {})
            return
        if self.data_backend.exists(self.cache_file):
            try:
                cache_data = json.loads(self.data_backend.read(self.cache_file))
            except Exception as exc:
                logger.warning("Error loading webshart aspect bucket cache, creating new one: %s", exc)
                cache_data = {}
            self._stored_geometry_digest = (
                cache_data.get("geometry_sha256") if cache_data.get("geometry_format") == "sqlite-v1" else None
            )
            self.aspect_ratio_bucket_indices = cache_data.get("aspect_ratio_bucket_indices", {})
            self._sync_image_files_with_buckets()
            if set_config:
                self.config = cache_data.get("config", {})
            self.filtering_statistics = cache_data.get("filtering_statistics")
        else:
            logger.debug("No webshart cache file found, starting fresh.")
        self.load_image_metadata()

    def close(self):
        self._accepted_geometry = None
        self._transforms = None
        if self._snapshot is not None:
            if self._published.get("source_pins"):
                self.data_backend.dataset.bind_source_revisions(None)
            self._snapshot.close()
            self._snapshot = None

    def _open_membership(self):
        if not self.data_backend.exists(self.cache_file):
            raise ValueError("No accepted samples in dataset metadata; scan the dataset before preparing text embeddings.")
        return self.data_backend.open_file(self.cache_file, "rb")

    @property
    def membership_path(self):
        return self.data_backend._cache_path(self.cache_file)

    def describe(self, spec=None):
        if self.data_backend.exists(self.cache_file):
            with self._open_membership() as source:
                pages = MembershipPages.open(source)
                if pages is not None:
                    if spec is not None:
                        self._check_publication(pages)
                    return pages.describe(spec)
        return DatasetDescription(**self._source_description())

    @property
    def view_revision(self):
        return self._published["view_revision"] if self._published else None

    def read_page(self, *, bucket=None, offset=0, limit=50):
        with self._open_membership() as source:
            pages = MembershipPages.open(source)
            if self.native_policy:
                self._check_publication(pages)
            if pages is not None:
                return pages.read_page(bucket, offset, limit)
        return super().read_page(bucket=bucket, offset=offset, limit=limit)

    @property
    def epoch_provider(self):
        return self if self.native_policy else None

    @property
    def requires_complete_preparation(self):
        return self.native_policy is not None

    def refresh_epoch_geometry(self, epoch, *, rebuild_random_aspect, gradient_accumulation_steps, apply_padding):
        if self.native_policy:
            return False
        return super().refresh_epoch_geometry(
            epoch,
            rebuild_random_aspect=rebuild_random_aspect,
            gradient_accumulation_steps=gradient_accumulation_steps,
            apply_padding=apply_padding,
        )

    @property
    def recrops_lazily(self):
        config = self.active_dataset_config
        return (
            bool(self.native_policy)
            and config.get("vae_cache_clear_each_epoch") is True
            and config.get("crop")
            and config.get("crop_style", "random") == "random"
            and config.get("crop_aspect") != "random"
            and self.dataset_type is DatasetType.IMAGE
            and self.data_backend.format == "parquet"
            and not config.get("conditioning_data")
        )

    def epoch_spec(self, epoch, batch_size, *, occurrence_revision=None):
        args = self.runtime_args
        dp_size, dp_rank, cp_size = get_cp_aware_dp_info(self.accelerator)
        parallel = getattr(self.accelerator, "parallelism_config", None)
        topology = (
            self.accelerator.num_processes,
            cp_size,
            str(getattr(self.accelerator, "distributed_type", "NO")),
            *(getattr(parallel, name, 1) for name in ("dp_replicate_size", "dp_shard_size", "tp_size")),
        )
        return EpochSpec(
            self.id,
            self.native_policy,
            int(args.seed or 0),
            epoch,
            batch_size,
            dp_size,
            dp_rank,
            args.gradient_accumulation_steps,
            self.repeats,
            (
                self._schedule_padding
                if self._schedule_padding is not None
                else (not args.max_train_steps or args.allow_dataset_oversubscription)
            ),
            topology,
            occurrence_revision=occurrence_revision,
        )

    def _check_publication(self, pages):
        if pages is None or self._published is None:
            raise ValueError("Scan this Webshart dataset with a native sampling policy to publish its prepared geometry.")
        if pages.header["view_revision"] != self._published["view_revision"]:
            raise ValueError("Webshart membership changed since the prepared geometry was opened. Reload the dataset.")

    def scan_dataset_cache(self, dataset, cache_behavior):
        if not self.native_policy:
            super().scan_dataset_cache(dataset, cache_behavior)
            return
        if dataset.vae_cache is None:
            return
        if cache_behavior != "recreate":
            raise ValueError("Native Webshart cache scans require vae_cache_scan_behaviour='recreate'.")
        dataset.vae_cache.remove_inconsistent_prepared_latents(dataset)

    def iter_preparation_batches(self, worker, workers, batch_size):
        if not self.native_policy:
            yield from super().iter_preparation_batches(worker, workers, batch_size)
            return
        with self._open_membership() as stream:
            pages = MembershipPages.open(stream)
            self._check_publication(pages)
            for bucket, refs in pages.partition_groups(worker, workers):
                for batch in batched(refs, batch_size):
                    yield bucket, batch

    def open_epoch(self, spec):
        source = self._open_membership()
        try:
            if self.native_policy:
                self._check_publication(MembershipPages.open(source))
                source.seek(0)
            return WebshartEpoch(source, spec, self.data_backend.shard_of)
        except BaseException:
            source.close()
            raise

    def scheduled_batches(self, spec):
        with self._open_membership() as source:
            pages = MembershipPages.open(source)
            self._check_publication(pages)
            return pages.describe(spec).batches

    def _source_description(self):
        if self.data_backend.format != "parquet":
            return {}
        dataset = self.data_backend.dataset
        return {"source_revision": dataset.fingerprint, "physical_rows": getattr(dataset, "physical_row_count", None)}

    def _read_membership(self, stream):
        events = iter(ijson.parse(stream))
        expected_digest = None
        for prefix, event, value in events:
            if prefix == "geometry_sha256" and event == "string":
                expected_digest = value
            if prefix == "scan_in_progress" and event == "boolean" and value:
                raise ValueError("Webshart scan is incomplete; finish the dataset scan before preparing text embeddings.")
            if prefix == "aspect_ratio_bucket_indices" and event == "start_map":
                return chain.from_iterable(refs for _, refs in bucket_groups_from_events(events)), expected_digest
        raise ValueError("Missing aspect_ratio_bucket_indices in Webshart bucket metadata.")

    def open_validation(self):
        return IndexedValidationCursor(self._open_membership, self._check_publication)

    def iter_validation_cycles(self):
        with self._open_membership() as stream:
            while True:
                stream.seek(0)
                pages = MembershipPages.open(stream)
                if pages is not None:
                    yield pages.iter_refs(None, 0, pages.count())
                else:
                    refs, _ = self._read_membership(stream)
                    yield refs

    @contextmanager
    def open_preparation_view(self, *, worker=0, workers=1):
        with self._open_membership() as stream, ExitStack() as stack:
            pages = MembershipPages.open(stream)
            if self.native_policy:
                self._check_publication(pages)
            if pages is None:
                refs, expected_digest = self._read_membership(stream)
                refs = islice(refs, worker, None, workers)
            else:
                refs = pages.partition(worker, workers)
                expected_digest = pages.header.get("geometry_sha256")
            lookup = self
            if self._transforms is None and (expected_digest is not None or self.data_backend.exists(self.metadata_file)):
                lookup = self._open_geometry(stack, pages.header if pages else {}, expected_digest)
            elif self.native_policy:
                view = DatasetReadView(pages.header["view_revision"])
                lookup = SimpleNamespace(get_metadata_by_filepath=lambda ref: self.get_view_metadata((ref,), view)[0])
            yield PreparationView(refs, lookup)

    def save_cache(self, enforce_constraints: bool = False, *, scan_in_progress: bool = False):
        if self.native_policy:
            return
        if enforce_constraints:
            self._enforce_min_bucket_size()
        self._enforce_min_aspect_ratio()
        self._enforce_max_aspect_ratio()

        if self.read_only:
            logger.debug("Metadata backend is read-only. Skipping save.")
            return

        cache_data = {
            "scan_in_progress": scan_in_progress,
            **self._source_description(),
            "geometry_sha256": self._geometry_digest,
            "config": make_public_config_serializable(self.active_dataset_config),
        }
        if self.filtering_statistics is not None:
            cache_data["filtering_statistics"] = self.filtering_statistics
        self.data_backend.write_membership(self.cache_file, self.aspect_ratio_bucket_indices, cache_data)

    def _open_geometry(self, stack, header, digest):
        if header.get("geometry_format") == "sqlite-v1":
            path = self.data_backend._cache_path(self.metadata_file)
            db = stack.enter_context(closing(read_snapshot(path, digest)))
            return geometry_lookup(self.data_backend._cache_path(self.cache_file), header, db)
        if not self.data_backend.exists(self.metadata_file):
            raise ValueError("Published Webshart geometry is missing. Run a new scan.")
        source = stack.enter_context(self.data_backend.open_file(self.metadata_file, "rb"))
        return stack.enter_context(open_geometry(source, digest))

    def conditioning_ref(self, pairing_key):
        source = self.data_backend
        if source.format == "parquet":
            filename = source.parse_sample_id(pairing_key).filename if source.is_sample_id(pairing_key) else pairing_key
            return next(
                (ref for ref in source.parquet_sample_refs_for_filename(filename) if self.admits_related_sample(ref)),
                None,
            )
        source._resolve_sample_ref(pairing_key)
        ref = source.normalize_sample_id(pairing_key)
        return ref if self.admits_related_sample(ref) else None

    def admits_related_sample(self, ref):
        """Check the target view's membership before a related sample can be read."""
        if not self.native_policy:
            return True
        if self._published is None:
            raise ValueError("Scan this Webshart dataset before resolving related samples.")
        if self._published.get("geometry_format") != "sqlite-v1":
            return True
        return self._membership_contains()(self.data_backend.normalize_sample_id(ref))

    def _membership_contains(self):
        lookup = self._geometry_lookup()
        if self._accepted_geometry is None:
            publication = self.data_backend._cache_path(self.cache_file)
            self._accepted_geometry = accepted_geometry_predicate(lookup, publication, self._published, self._snapshot)
        return self._accepted_geometry

    def get_metadata_by_filepath(self, filepath):
        if not self.native_policy or self._published is None:
            return super().get_metadata_by_filepath(filepath)
        refs = filepath if isinstance(filepath, (tuple, list)) else (filepath,)
        for record in self.get_metadata_batch(refs):
            if record is not None:
                return record
        return None

    def _geometry_lookup(self):
        if self._snapshot is None:
            with ExitStack() as stack:
                self._lookup = self._open_geometry(stack, self._published, self._geometry_digest)
                self._snapshot = stack.pop_all()
        return self._lookup

    def get_metadata_batch(self, refs):
        if not self.native_policy or self._published is None:
            return super().get_metadata_batch(refs)
        return self._published_metadata_batch([self.data_backend.normalize_sample_id(ref) for ref in refs])

    def get_view_metadata(self, refs, view):
        self.validate_read_view(view)
        refs = [self.data_backend.normalize_sample_id(ref) for ref in refs]
        records = self._published_metadata_batch(refs)
        if view.epoch in (None, 1) or not self.recrops_lazily:
            return records
        return [
            (
                None
                if record is None
                else recrop_record(
                    record, random.Random(f"{view.seed}:recrop-v{self.epoch_recrop_version}:{view.epoch}:{ref}")
                )
            )
            for ref, record in zip(refs, records, strict=True)
        ]

    def _published_metadata_batch(self, refs):
        if self._transforms is None:
            return self._geometry_lookup().get_metadata_batch(refs)
        accepted = self._membership_contains().batch(refs)
        records = iter(self._transforms.read_batch([ref for ref, include in zip(refs, accepted, strict=True) if include]))
        return [next(records) if include else None for include in accepted]

    def _open_transforms(self, settings, shard_revision):
        return WebshartTransforms(self.data_backend, settings, shard_revision)

    def enqueue_prepared_metadata(self, records):
        transforms = self._writing_transforms or self._transforms
        if transforms is None:
            super().enqueue_prepared_metadata(records)
        else:
            transforms.enqueue(records)

    def flush_prepared_metadata(self):
        transforms = self._writing_transforms or self._transforms
        if transforms is None:
            super().flush_prepared_metadata()
        else:
            transforms.flush()

    @contextmanager
    def _prepare_transforms(self, settings):
        if settings is None:
            yield
            return
        self._writing_transforms = self._open_transforms(settings, self.data_backend.dataset.source_revision_for_shard)
        try:
            yield
        finally:
            transforms, self._writing_transforms = self._writing_transforms, None
            transforms.close()

    @staticmethod
    def _component_digest(digest, locator, folders, revision, processor=False):
        if not locator:
            return
        root = Path(locator)
        names = ("preprocessor_config.json", "processor_config.json") if processor else ("config.json",)
        if root.is_file():
            files = [root]
        elif root.is_dir():
            directories = [root / folder for folder in folders if folder and (root / folder).is_dir()]
            directories.append(root)
            files = []
            for directory in directories:
                if processor:
                    files = [directory / name for name in names if (directory / name).is_file()]
                else:
                    files = sorted(
                        file
                        for file in directory.iterdir()
                        if file.is_file()
                        and file.suffix in {".json", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"}
                    )
                if files:
                    break
        else:
            from huggingface_hub import try_to_load_from_cache

            if not root.is_absolute():
                for folder in (*folders, ""):
                    for name in names:
                        snapshot = try_to_load_from_cache(str(locator), str(Path(folder or "") / name), revision=revision)
                        if isinstance(snapshot, str) and "snapshots" in Path(snapshot).parts:
                            parts = Path(snapshot).parts
                            digest.update(parts[parts.index("snapshots") + 1].encode())
                            return
            return
        for file in files:
            with file.open("rb") as data:
                digest.update(json.dumps([file.name, file_digest(data, "sha256").hexdigest()]).encode())

    def _artifact_model_revision(self, kind, config):
        settings = ("model_family", "model_flavour", "pretrained_model_name_or_path", "revision", "variant", "weight_dtype")
        revision = getattr(config, "revision", None)
        base = getattr(config, "pretrained_model_name_or_path", None)
        if kind == "vae":
            settings += ("pretrained_vae_model_name_or_path", "vae_path", "vae_dtype", "vae_tiling", "vae_slicing")
            source = getattr(config, "vae_path", None) or getattr(config, "pretrained_vae_model_name_or_path", None) or base
            components = [(source, ["vae"], revision, False)]
        else:
            settings += (
                "image_encoder_pretrained_model_name_or_path",
                "hunyuan_image_encoder_path",
                "image_encoder_subfolder",
                "image_encoder_revision",
                "image_processor_pretrained_model_name_or_path",
                "image_processor_subfolder",
                "image_processor_revision",
            )
            source = (
                getattr(config, "image_encoder_pretrained_model_name_or_path", None)
                or getattr(config, "hunyuan_image_encoder_path", None)
                or base
            )
            processor = getattr(config, "image_processor_pretrained_model_name_or_path", None) or source
            components = []
            for label, locator, defaults in (
                ("image_encoder", source, ["image_encoder", "vision_encoder"]),
                ("image_processor", processor, ["image_processor", "feature_extractor"]),
            ):
                configured = getattr(config, label + "_subfolder", None)
                folders = [configured] if isinstance(configured, str) else list(configured or ())
                components.append(
                    (
                        locator,
                        folders + defaults,
                        getattr(config, label + "_revision", None) or revision,
                        label == "image_processor",
                    )
                )

        spec = tuple(str(getattr(config, key, None)) for key in settings)
        cached = self._artifact_models.get(kind)
        if cached is not None and cached[0] == spec:
            return cached[1]
        digest = sha256(json.dumps(spec).encode())
        for component in components:
            self._component_digest(digest, *component)
        revision_digest = digest.hexdigest()
        self._artifact_models[kind] = spec, revision_digest
        return revision_digest

    def image_artifact_key(self, ref, kind, model, *, view=None):
        if view is None:
            metadata = self.get_metadata_by_filepath(ref)
        else:
            geometry_view = DatasetReadView(view.revision) if kind == "conditioning" else view
            (metadata,) = self.get_view_metadata((ref,), geometry_view)
        if metadata is None or self._published is None:
            raise ValueError("Image artifacts require published Webshart geometry.")
        shard = metadata["webshart"]["shard_idx"]
        content_revision = self._lookup.shard_revision(shard) if self._published.get("source_pins") else None
        spec = [
            kind,
            ref,
            self._published.get("source_revision"),
            content_revision,
            metadata,
            self._artifact_model_revision(kind, model.config),
            {
                key: self.dataset_config.get(key)
                for key in ("crop", "crop_style", "crop_aspect", "video", "dataset_type", "conditioning_config")
            },
        ]
        if self.dataset_config.get("crop") and self.dataset_config.get("crop_style") != "face":
            spec.append("prepared-crop-v1")
        if kind == "vae" and self.recrops_lazily:
            spec.append(view.epoch if view is not None and view.epoch is not None else 1)
        return sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def load_image_metadata(self):
        if self.native_policy:
            self.reload_cache(set_config=False)
            return
        self._geometry_digest = None
        self.image_metadata = {}
        self.image_metadata_loaded = False
        if getattr(self, "_stored_geometry_digest", None):
            path = self.data_backend._cache_path(self.metadata_file)
            with closing(read_snapshot(path, self._stored_geometry_digest)) as db:
                self.image_metadata = {
                    ref: json.loads(record) for ref, record in db.execute("SELECT ref, record FROM geometry")
                }
            self.image_metadata_loaded = True
            return
        if self.data_backend.exists(self.metadata_file):
            raw = self.data_backend.read(self.metadata_file)
            self._geometry_digest = sha256(raw if isinstance(raw, bytes) else raw.encode()).hexdigest()
            self.image_metadata = json.loads(raw)
            for metadata in self.image_metadata.values():
                metadata.pop("captions", None)
            self.image_metadata_loaded = True

    def save_image_metadata(self):
        if self.native_policy:
            return
        self._geometry_digest = self.data_backend.write_geometry(self.metadata_file, self.image_metadata)
        self.image_metadata_loaded = True

    def _all_shard_indices(self) -> range:
        return range(self.data_backend.num_shards())

    def _sample_id_from_entry(self, shard_idx: int, entry: dict) -> str:
        if "sample_idx" not in entry:
            raise ValueError("Webshart sample bucket entries must include sample_idx.")
        return self.data_backend.sample_id(shard_idx, int(entry["sample_idx"]), str(entry["filename"]))

    @staticmethod
    def _coerce_positive_number(value: Any, value_type):
        try:
            result = value_type(value)
        except (TypeError, ValueError):
            return None
        return result if result > 0 else None

    @staticmethod
    def _ffprobe_video_path(probe_path: str) -> dict:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                probe_path,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        probe = json.loads(result.stdout) if result.stdout else {}
        streams = probe.get("streams") or []
        stream = streams[0] if streams else {}
        width = WebshartMetadataBackend._coerce_positive_number(stream.get("width"), int)
        height = WebshartMetadataBackend._coerce_positive_number(stream.get("height"), int)
        metadata = {"original_size": (width, height)} if width and height else {}

        num_frames = WebshartMetadataBackend._coerce_positive_number(stream.get("nb_frames"), int)
        if num_frames:
            metadata["num_frames"] = num_frames
        duration = WebshartMetadataBackend._coerce_positive_number(
            stream.get("duration") or (probe.get("format") or {}).get("duration"),
            float,
        )
        if duration:
            metadata["video_duration"] = duration
        return metadata

    def _probe_video_metadata(self, sample_path: str, file_metadata: Optional[dict] = None) -> dict:
        if shutil.which("ffprobe") is None:
            return {}

        suffix = Path(self.data_backend.parse_sample_id(sample_path).filename).suffix or ".mp4"
        probe_path = None
        try:
            range_reader = getattr(self.data_backend, "read_sample_head_tail", None)
            if callable(range_reader):
                try:
                    head, tail, total_size = range_reader(sample_path, file_metadata=file_metadata)
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                        probe_path = handle.name
                        handle.truncate(total_size)
                        handle.seek(0)
                        handle.write(head)
                        handle.seek(total_size - len(tail))
                        handle.write(tail)
                    metadata = self._ffprobe_video_path(probe_path)
                    if metadata.get("original_size"):
                        return metadata
                except Exception as exc:
                    logger.debug("Unable to range-probe Webshart video %s: %s", sample_path, exc)
                finally:
                    if probe_path:
                        Path(probe_path).unlink(missing_ok=True)
                        probe_path = None

            payload = self.data_backend.read(sample_path)
            if not payload:
                return {}
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(payload)
                probe_path = handle.name
            return self._ffprobe_video_path(probe_path)
        except Exception as exc:
            logger.debug("Unable to probe Webshart video %s: %s", sample_path, exc)
            return {}
        finally:
            if probe_path:
                Path(probe_path).unlink(missing_ok=True)

    def _metadata_for_entry(self, shard_metadata: dict, filename: str, entry: dict, sample_path: str) -> dict:
        file_metadata = shard_metadata.get(filename, {}) or {}
        if self.data_backend.caption_key is not None and self.data_backend.format != "parquet":
            file_metadata = self.data_backend.get_sample_metadata(sample_path)
        width = entry.get("width", file_metadata.get("width"))
        height = entry.get("height", file_metadata.get("height"))
        metadata = {
            "webshart": {
                "shard_idx": entry.get("shard_idx"),
                "sample_idx": entry.get("sample_idx"),
                "filename": filename,
                "offset": entry.get("offset", file_metadata.get("offset")),
                "size": entry.get("size", file_metadata.get("size")),
                "json_path": file_metadata.get("json_path"),
            },
        }
        if width is not None and height is not None:
            metadata["original_size"] = (int(width), int(height))
        json_metadata = file_metadata.get("json_metadata") or {}
        if json_metadata:
            metadata["json_metadata"] = json_metadata
        if "json_path" in file_metadata:
            metadata["json_path"] = file_metadata["json_path"]

        if self.dataset_type is DatasetType.VIDEO:
            fps = self._coerce_positive_number(json_metadata.get("fps"), float)
            num_frames = self._coerce_positive_number(
                json_metadata.get("frame", json_metadata.get("num_frames")),
                int,
            )
            duration = self._coerce_positive_number(
                json_metadata.get("seconds", json_metadata.get("duration")),
                float,
            )
            if fps:
                metadata["fps"] = fps
            if num_frames:
                metadata["num_frames"] = num_frames
            if duration:
                metadata["video_duration"] = duration
            if "original_size" not in metadata:
                probed = self._probe_video_metadata(sample_path, file_metadata=file_metadata)
                for key, value in probed.items():
                    metadata.setdefault(key, value)
        return metadata

    def _prepare_bucket_entry(
        self,
        shard_metadata: dict,
        entry: dict,
        sample_path: str,
        require_captions: bool,
        recipe: PreparationGeometry | None = None,
    ) -> tuple[dict, Optional[tuple[str, dict]], Optional[Exception]]:
        try:
            filename = str(entry["filename"])
            sample_metadata = self._metadata_for_entry(shard_metadata, filename, entry, sample_path)
            prepared = self._prepare_metadata(sample_path, sample_metadata, recipe)
            if prepared is not None and require_captions and not self._entry_has_caption(shard_metadata, entry, sample_path):
                raise CaptionNotFoundError(f"Could not find caption for {sample_path}")
            return sample_metadata, prepared, None
        except Exception as exc:
            return {}, None, exc

    def _prepare_metadata(
        self, sample_path: str, sample_metadata: dict, recipe: PreparationGeometry | None = None
    ) -> Optional[tuple[str, dict]]:
        if not sample_metadata or "original_size" not in sample_metadata:
            return None
        config = recipe.config if recipe is not None else self.active_dataset_config
        if not self.meets_resolution_requirements(image_metadata=sample_metadata, dataset_config=config):
            return None
        if recipe is not None:
            prepared_sample = recipe.prepare(self.id, sample_path, sample_metadata)
        else:
            prepared_sample = TrainingSample(
                image=None,
                data_backend_id=self.id,
                image_metadata=sample_metadata,
                image_path=sample_path,
                data_backend_config=config,
            ).prepare()
        sample_metadata.update(prepared_fields(prepared_sample))
        aspect_ratio = sample_metadata["aspect_ratio"]
        if self.dataset_type is DatasetType.VIDEO and self.bucket_strategy == "resolution_frames":
            target_width, target_height = prepared_sample.target_size
            bucket_key, rounded_frames = self._compute_video_bucket(
                target_width,
                target_height,
                sample_metadata["num_frames"],
            )
            sample_metadata["bucket_frames"] = rounded_frames
        elif recipe is not None:
            width, height = prepared_sample.target_size
            bucket_key = f"{width}x{height}"
        else:
            bucket_key = str(round(aspect_ratio, 2))
        if recipe is not None:
            sample_metadata["prepared_geometry"] = True
        return bucket_key, sample_metadata

    def _entries_for_shard(self, shard_idx: int) -> list[dict]:
        if self.dataset_type is not DatasetType.VIDEO:
            fields = ("width", "height") if self.data_backend.supports_caption_projection is True else None
            shard_bucket_results = self.data_backend.list_shard_sample_aspect_buckets(
                [shard_idx],
                dataset_filter=self.dataset_filter,
                **({"metadata_fields": fields} if fields is not None else {}),
            )
            if not shard_bucket_results:
                return []
            shard_bucket_data = shard_bucket_results[0]
            return [entry for entries in shard_bucket_data.get("buckets", {}).values() for entry in entries]

        entries = []
        for sample in self.data_backend.list_samples_in_shard(shard_idx, dataset_filter=self.dataset_filter):
            sample_idx = int(sample["sample_idx"])
            filename = str(sample["filename"])
            if WebshartDataBackend.filename_suffix(filename).strip(".") not in video_file_extensions:
                continue
            entries.append({"sample_idx": sample_idx, "filename": filename})
        return entries

    def _entry_has_caption(self, shard_metadata: dict, entry: dict, sample_path: str) -> bool:
        file_metadata = shard_metadata.get(str(entry["filename"]), {}) or {}
        if self.data_backend.format == "parquet":
            return bool(self.data_backend._caption_from_metadata(file_metadata))
        if self.data_backend.caption_key is not None:
            return bool(self.data_backend.get_caption(sample_path))
        if file_metadata.get("captions"):
            return True
        caption_member = self.data_backend.caption_sidecar_filename(str(entry["filename"]))
        return caption_member in shard_metadata

    def compute_aspect_ratio_bucket_indices(self, ignore_existing_cache: bool = False, progress_callback=None):
        logger.info("Building aspect ratio buckets from webshart metadata...")
        statistics = _empty_filtering_statistics()
        backend_config = self.active_dataset_config or {}
        captions = DatasetRuntime.caption_config(backend_config, self.runtime_args)
        require_captions = captions["caption_strategy"] == "webshart" and captions["use_captions"]

        if getattr(self, "native_policy", None):
            if ignore_existing_cache and self._published and self._published.get("source_pins"):
                self._refresh_native_view(backend_config, statistics, require_captions, progress_callback)
            else:
                self._publish_native_view(
                    backend_config, statistics, require_captions, ignore_existing_cache, progress_callback
                )
            return

        if not ignore_existing_cache:
            self.reload_cache()
            existing_files = (
                set().union(*self.aspect_ratio_bucket_indices.values()) if self.aspect_ratio_bucket_indices else set()
            )
            statistics["skipped"]["already_exists"] = len(existing_files)
        else:
            self.aspect_ratio_bucket_indices = {}
            self.image_metadata = {}
            existing_files = set()

        last_save_time = time.time()
        for key, ref, record in self._scan_rows(existing_files, statistics, require_captions, progress_callback):
            self.aspect_ratio_bucket_indices.setdefault(key, []).append(ref)
            self.enqueue_prepared_metadata({ref: record})
            if time.time() - last_save_time >= self.metadata_update_interval:
                self.flush_prepared_metadata()
                self.save_cache(enforce_constraints=False, scan_in_progress=True)
                last_save_time = time.time()

        self.filtering_statistics = statistics
        self.flush_prepared_metadata()
        self.save_cache(enforce_constraints=True)
        self._sync_image_files_with_buckets()
        if self.bucket_report:
            self.bucket_report.update_statistics(statistics)
            self.bucket_report.record_bucket_snapshot("post_refresh", self.aspect_ratio_bucket_indices)

    def _geometry_recipe(self, config):
        sizing = ImageGeometry(
            MultiaspectImage._get_alignment(args=self.runtime_args),
            MultiaspectImage._get_rounding(2, args=self.runtime_args),
        )
        seed = getattr(self.runtime_args, "seed", None) or 0
        model_geometry = SimpleNamespace(MAXIMUM_CANVAS_SIZE=self.maximum_canvas_size)
        return PreparationGeometry(deepcopy(config), sizing, seed, model_geometry)

    def _scan_rows(
        self, existing_files, statistics, require_captions, progress_callback, *, recipe=None, shard_indices=None, stop=None
    ):
        if recipe is None and self.native_policy:
            recipe = self._geometry_recipe(self.dataset_config)
        show_progress = shard_indices is None
        if show_progress:
            shard_indices = self._all_shard_indices()
        total_shards = len(shard_indices)
        if show_progress:
            shard_indices = tqdm(
                shard_indices, total=total_shards, desc="Processing webshart metadata", leave=False, ncols=100
            )
        worker_count = max(1, int(getattr(self.data_backend, "parallel_downloads", 1)))
        executor = (
            ThreadPoolExecutor(max_workers=worker_count)
            if self.dataset_type is DatasetType.VIDEO and worker_count > 1
            else None
        )

        try:
            for shard_idx in shard_indices:
                if (
                    self.max_num_samples is not None
                    and statistics["total_processed"] + len(existing_files) >= self.max_num_samples
                ):
                    break

                projected = (
                    self.data_backend.supports_caption_projection is True and self.dataset_type is not DatasetType.VIDEO
                )
                native_rows = bool(
                    self.native_policy
                    and self.data_backend.format == "parquet"
                    and self.dataset_type is not DatasetType.VIDEO
                )
                fields = {"width", "height", "json_path", "json_metadata", "offset", "size"}
                if require_captions:
                    fields.update(self.data_backend.caption_fields() if projected or native_rows else ())
                if native_rows:
                    projection = getattr(self.data_backend.loader, "iter_samples_in_shard", None)
                    if projection is None:
                        raise ImportError("Native Parquet sampling requires Webshart projected sample iteration.")
                    shard_entries = (
                        row
                        for row in projection(shard_idx, metadata_fields=fields, batch_size=128)
                        if (row.get("width") or 0) > 0
                        and (row.get("height") or 0) > 0
                        and (
                            self.dataset_filter is None
                            or not self.dataset_filter.path_active()
                            or self.dataset_filter.matches_path(row["filename"])
                        )
                    )
                else:
                    shard_entries = self._entries_for_shard(shard_idx)
                shard_metadata = None if projected or native_rows else self.data_backend.get_shard_metadata(shard_idx)
                candidates = (
                    (entry, ref)
                    for entry in shard_entries
                    if (ref := self._sample_id_from_entry(shard_idx, entry)) not in existing_files
                )
                chunk_size = 128 if projected or native_rows else max(1, worker_count * 2)
                while True:
                    if stop is not None and stop.is_set():
                        raise CancelledError()
                    remaining = (
                        chunk_size
                        if self.max_num_samples is None
                        else self.max_num_samples - statistics["total_processed"] - len(existing_files)
                    )
                    if remaining <= 0:
                        break
                    chunk = [
                        ({**entry, "shard_idx": shard_idx}, ref)
                        for entry, ref in islice(candidates, min(chunk_size, remaining))
                    ]
                    if not chunk:
                        break
                    if native_rows:
                        shard_metadata = {entry["filename"]: entry for entry, _ in chunk}
                    elif projected:
                        rows = self.data_backend.dataset.read_metadata_rows(
                            shard_idx, [entry["sample_idx"] for entry, _ in chunk], fields=fields
                        )
                        shard_metadata = {entry["filename"]: row for (entry, _), row in zip(chunk, rows, strict=True)}
                    if executor is None:
                        results = [
                            self._prepare_bucket_entry(shard_metadata, entry, sample_path, require_captions, recipe)
                            for entry, sample_path in chunk
                        ]
                    else:
                        results = executor.map(
                            lambda item: self._prepare_bucket_entry(
                                shard_metadata, item[0], item[1], require_captions, recipe
                            ),
                            chunk,
                        )

                    for (entry, sample_path), (sample_metadata, prepared, error) in zip(chunk, results):
                        if isinstance(error, CaptionNotFoundError):
                            statistics["skipped"]["caption_missing"] += 1
                            continue
                        if error is not None:
                            logger.error("Error processing webshart bucket entry %s: %s", entry, error)
                            statistics["skipped"]["error"] += 1
                            continue
                        if prepared is None:
                            if sample_metadata:
                                statistics["skipped"]["too_small"] += 1
                            else:
                                statistics["skipped"]["metadata_missing"] += 1
                            continue
                        bucket_key, sample_metadata = prepared
                        statistics["total_processed"] += 1
                        yield bucket_key, sample_path, sample_metadata

                if progress_callback is not None:
                    progress_callback(shard_idx + 1, total_shards)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    def _write_native_geometry(self, writer, rows, exclusions):
        pinned_shard = None
        pin_source = self.data_backend.format == "parquet"
        prepared = {}
        for bucket, ref, record in rows:
            shard = record["webshart"]["shard_idx"]
            if pin_source and shard != pinned_shard:
                writer.pin_shard(shard, self.data_backend.dataset.source_revision_for_shard(shard))
                pinned_shard = shard
            if exclusions is None or exclusions.execute("SELECT 1 FROM exclusions WHERE ref=?", (ref,)).fetchone() is None:
                if self._writing_transforms is None:
                    writer.add(bucket, ref, record, shard=shard)
                else:
                    writer.add(bucket, ref, None, shard=shard)
                    prepared[ref] = record
                    if len(prepared) >= 128:
                        self.enqueue_prepared_metadata(prepared)
                        prepared = {}
        if prepared:
            self.enqueue_prepared_metadata(prepared)

    def _partition_cache_path(self, root, indices, rules, exclusions):
        if self.max_num_samples is not None or self.dataset_type is DatasetType.VIDEO:
            return None
        source = self.data_backend.prepared_partition_identity(indices)
        if source is None:
            return None
        digest = sha256(b"webshart-prepared-group-v1")
        digest.update(json.dumps([source, rules], sort_keys=True, separators=(",", ":")).encode())
        if exclusions is not None:
            for shard in indices:
                prefix = f"{self.data_backend.SAMPLE_PREFIX}{shard}/"
                for (ref,) in exclusions.execute(
                    "SELECT ref FROM exclusions WHERE ref>=? AND ref<? ORDER BY ref", (prefix, f"{prefix[:-1]}0")
                ):
                    digest.update(json.dumps(ref).encode())
        return root / f"{digest.hexdigest()}.json"

    def _prepare_native_partitions(
        self, writer, rules, statistics, require_captions, progress_callback, recipe, exclusion_file, refresh
    ):
        shards = self._all_shard_indices()
        group_rules = {key: value for key, value in rules.items() if key != "source_revision"}
        group_rules["shards_per_partition"] = writer.shards_per_partition
        cache_root = writer.staging.parent / ".prepared-webshart-groups"
        workers = 1
        if (
            self.max_num_samples is None
            and self.dataset_type is not DatasetType.VIDEO
            and self.data_backend.manifest
            and getattr(self.data_backend.dataset, "supports_parallel_metadata_reads", False) is True
        ):
            groups = (len(shards) + writer.shards_per_partition - 1) // writer.shards_per_partition
            workers = max(1, min(groups, self.data_backend.parallel_downloads))

        def jobs():
            for partition, indices in enumerate(batched(shards, writer.shards_per_partition)):
                if self.max_num_samples is not None and statistics["total_processed"] >= self.max_num_samples:
                    break
                yield partition, indices

        def prepare(job, report, stop=None):
            partition, indices = job
            with closing(_read_index(exclusion_file)) if exclusion_file is not None else nullcontext() as exclusions:
                cached = self._partition_cache_path(cache_root, indices, group_rules, exclusions)
                lock = nullcontext()
                if cached is not None:
                    cache_root.mkdir(exist_ok=True)
                    lock = FileLock(str(cached) + ".lock")
                with lock:
                    if stop is not None and stop.is_set():
                        raise CancelledError()
                    if cached is not None and not refresh and cached.is_file():
                        entry = json.loads(cached.read_text())
                        snapshot = snapshot_path(cache_root / "geometry", entry["geometry"])
                        for _ in indices:
                            report()
                        return snapshot, entry["filtering_statistics"]

                    group_statistics = statistics if self.max_num_samples is not None else _empty_filtering_statistics()
                    with writer.stage_partition(partition) as (group_writer, staged):
                        rows = self._scan_rows(
                            (),
                            group_statistics,
                            require_captions,
                            lambda *_: report(),
                            recipe=recipe,
                            shard_indices=indices,
                            stop=stop,
                        )
                        self._write_native_geometry(group_writer, rows, exclusions)
                    if cached is None or group_statistics["skipped"]["error"]:
                        return staged, group_statistics

                    digest = group_writer.digest.hexdigest()
                    snapshot = snapshot_path(cache_root / "geometry", digest)
                    link_snapshot(staged, snapshot)
                    staged.unlink()
                    entry = {"geometry": digest, "filtering_statistics": group_statistics}
                    staged_entry = staged.with_suffix(".json")
                    staged_entry.write_text(json.dumps(entry))
                    os.replace(staged_entry, cached)
                    return snapshot, group_statistics

        def adopt(partition, result):
            snapshot, group_statistics = result
            writer.adopt_partition(partition, snapshot)
            if group_statistics is not statistics:
                statistics["total_processed"] += group_statistics["total_processed"]
                for key, count in group_statistics["skipped"].items():
                    statistics["skipped"][key] += count

        with tqdm(total=len(shards), desc="Processing webshart metadata", leave=False, ncols=100) as progress:
            completed = 0

            def report():
                nonlocal completed
                completed += 1
                progress.update(1)
                if progress_callback is not None:
                    progress_callback(completed, len(shards))

            work = jobs()
            if workers == 1:
                for job in work:
                    adopt(job[0], prepare(job, report))
                return

            events = SimpleQueue()
            stop = Event()
            pending = deque()
            executor = ThreadPoolExecutor(max_workers=workers)

            def submit(job):
                future = executor.submit(prepare, job, lambda: events.put(True), stop)
                future.add_done_callback(lambda _: events.put(False))
                pending.append((job[0], future))

            try:
                for job in islice(work, workers):
                    submit(job)
                while pending:
                    partition, future = pending.popleft()
                    while not future.done():
                        if events.get():
                            report()
                    while not events.empty():
                        if events.get():
                            report()
                    adopt(partition, future.result())
                    for job in islice(work, 1):
                        submit(job)
            finally:
                stop.set()
                executor.shutdown(wait=True, cancel_futures=True)

    def _refresh_native_view(self, config, statistics, require_captions, progress_callback):
        fresh = self.data_backend.open_refresh_source()
        with closing(copy(self)) as staged:
            staged.data_backend = fresh
            staged._snapshot = None
            staged._accepted_geometry = None
            staged._transforms = None
            staged._publish_native_view(
                config, statistics, require_captions, True, progress_callback, previous_pins=self._lookup.shard_revision
            )
        self.close()
        self.data_backend.adopt_refresh_source(fresh)
        self.reload_cache()

    def _publish_native_view(self, config, statistics, require_captions, refresh, progress_callback, *, previous_pins=None):
        public_config = make_public_config_serializable(config)
        geometry_keys = (
            "dataset_type",
            "filter_func",
            "video",
            "crop",
            "crop_aspect",
            "crop_style",
            "crop_aspect_buckets",
            "maximum_image_size",
            "max_upscale_threshold",
            "target_downsample_size",
        )
        shards_per_partition = self.data_backend.prepared_partition_shards
        recipe = self._geometry_recipe(config)
        supports_transform_cache = (
            self.data_backend.format == "parquet"
            and self.dataset_type is not DatasetType.VIDEO
            and (not config.get("crop") or config.get("crop_style") != "face")
            and getattr(self.data_backend.webshart, "TransformCache", None) is not None
            and (refresh or self._published is None or self._published.get("transform_cache_recipe"))
        )
        transform_settings = recipe.cache_settings() if supports_transform_cache else None
        rules = {key: public_config.get(key) for key in geometry_keys}
        if transform_settings is not None:
            previous_settings = (self._published or {}).get("transform_cache_recipe") or {}
            generation = uuid4().hex if refresh else previous_settings.get("generation")
            if generation is not None:
                transform_settings["generation"] = generation
            rules["transform_cache_recipe"] = transform_settings
        rules.update(require_captions=require_captions, caption_key=self.data_backend.caption_key)
        args = self.runtime_args
        rules.update(
            preparation_version=6,
            crop_seed=recipe.seed,
            source_revision=self._source_description().get("source_revision"),
            minimum_num_frames=self.minimum_num_frames,
            maximum_num_frames=self.maximum_num_frames,
            num_frames=self.num_frames,
            bucket_strategy=self.bucket_strategy,
            aspect_bucket_alignment=recipe.sizing.alignment,
            aspect_bucket_rounding=recipe.sizing.rounding,
            framerate=getattr(args, "framerate", None),
            model_family=getattr(args, "model_family", None),
            maximum_canvas_size=getattr(recipe.model, "MAXIMUM_CANVAS_SIZE", None),
            resolution=self.resolution,
            resolution_type=self.resolution_type,
            max_num_samples=self.max_num_samples,
            minimum_image_size=self.minimum_image_size,
        )
        pruning = {
            "repeats": self.repeats,
            "scan_batch_size": self.batch_size,
            "maximum_aspect_ratio": self.maximum_aspect_ratio,
            "minimum_aspect_ratio": self.minimum_aspect_ratio,
            "disable_bucket_pruning": bool(args.disable_bucket_pruning),
        }
        if self.dataset_filter is not None:
            rules["filter_func"] = make_public_config_serializable(self.dataset_filter.config)
        if not config.get("crop"):
            rules["alignment_crop"] = "center-v1"
        rules_revision = sha256(json.dumps({**rules, **pruning}, sort_keys=True).encode()).hexdigest()
        if not refresh and self._published and self._published.get("rules_revision") == rules_revision:
            return
        path = self.data_backend._cache_path(self.metadata_file)
        with stage_geometry(path, shards_per_partition=shards_per_partition) as (writer, staging), ExitStack() as stack:
            stack.enter_context(self._prepare_transforms(transform_settings))
            exclusion_file = None
            exclusion_info = {}
            publication = self.membership_path
            if self._published and (excluded := self._published.get("exclusion_revision")):
                counts = self._published["exclusion_counts"]
                if previous_pins is not None:
                    exclusion_file, excluded, counts = retain_source_exclusions(
                        publication,
                        excluded,
                        staging.parent,
                        shard_of=self.data_backend.shard_of,
                        old_pin=previous_pins,
                        new_pin=self.data_backend.dataset.source_revision_for_shard,
                    )
                else:
                    exclusion_file = exclusion_path(publication, excluded)
                if excluded is not None:
                    exclusion_info = {"exclusion_revision": excluded, "exclusion_counts": counts}
            pin_source = self.data_backend.format == "parquet"
            if shards_per_partition is None:
                rows = self._scan_rows((), statistics, require_captions, progress_callback, recipe=recipe)
                exclusions = (
                    stack.enter_context(closing(_read_index(exclusion_file))) if exclusion_file is not None else None
                )
                self._write_native_geometry(writer, rows, exclusions)
            else:
                self._prepare_native_partitions(
                    writer, rules, statistics, require_captions, progress_callback, recipe, exclusion_file, refresh
                )
            if transform_settings is not None:
                self.flush_prepared_metadata()
            counts = writer.finish()
            for key, size in list(counts.items()):
                minimum_batch = 1 if self.dataset_type is DatasetType.EVAL else self.batch_size
                if (
                    self.minimum_image_size is not None
                    and not pruning["disable_bucket_pruning"]
                    and size * (self.repeats + 1) < minimum_batch
                ):
                    del counts[key]
                elif self.minimum_aspect_ratio or self.maximum_aspect_ratio:
                    width, height = (int(side) for side in key.split("@", 1)[0].split("x"))
                    aspect = width / height
                    if "@" not in key:
                        aspect = round(
                            MultiaspectImage.calculate_image_aspect_ratio((width, height), geometry=recipe.sizing), 2
                        )
                    if (self.minimum_aspect_ratio and aspect < self.minimum_aspect_ratio) or (
                        self.maximum_aspect_ratio and aspect > self.maximum_aspect_ratio
                    ):
                        del counts[key]
            digest = writer.digest.hexdigest()
            writer.db.commit()
            destination = snapshot_path(path, digest)
            expected = self._published.get("view_revision") if self._published else None
            with FileLock(str(publication) + ".publish.lock"):
                if publication.is_file():
                    with publication.open("rb") as file:
                        current = MembershipPages.open(file)
                    current_revision = current.header["view_revision"] if current is not None else None
                    if current_revision != expected:
                        raise ValueError("The accepted Webshart view changed during discovery. Reload before rescanning.")
                if shards_per_partition is not None:
                    writer.publish(path)
                os.replace(staging, destination)
                source = self._source_description()
                revision_inputs = [source.get("source_revision"), digest, counts]
                if transform_settings is not None:
                    revision_inputs.append(json.dumps(transform_settings, sort_keys=True, separators=(",", ":")))
                view_revision = sha256(json.dumps(revision_inputs, separators=(",", ":")).encode()).hexdigest()
                publish_indexed_membership(
                    publication,
                    {
                        **exclusion_info,
                        **source,
                        "view_revision": view_revision,
                        "bucket_counts": counts,
                        "geometry_format": "sqlite-v1",
                        "transform_cache_recipe": transform_settings,
                        "geometry_partition_shards": shards_per_partition,
                        "geometry_name": path.name,
                        "source_pins": pin_source,
                        "source_captions_required": require_captions,
                        "geometry_sha256": digest,
                        "rules_revision": rules_revision,
                        "config": public_config,
                        "filtering_statistics": statistics,
                    },
                )
            if self.bucket_report:
                self.bucket_report.record_bucket_counts("post_refresh", counts)
                self.bucket_report.update_statistics(statistics)
        self.reload_cache()

    def split_buckets_between_processes(self, gradient_accumulation_steps=1, apply_padding=False):
        if self.native_policy:
            self._schedule_padding = apply_padding
            self.read_only = True
            return
        return super().split_buckets_between_processes(gradient_accumulation_steps, apply_padding)

    def published_bucket_counts(self):
        return self._published["bucket_counts"] if self._published else {}

    def __len__(self):
        """
        Returns:
            int: The number of complete batches available across aspect ratio buckets.
        """

        if self.native_policy:
            spec = self.epoch_spec(1, self.batch_size)
            return self.scheduled_batches(spec)

        def repeat_len(bucket):
            return len(bucket) * (self.repeats + 1)

        return sum(
            (repeat_len(bucket) + (self.batch_size - 1)) // self.batch_size
            for bucket in self.aspect_ratio_bucket_indices.values()
            if repeat_len(bucket) >= self.batch_size
        )

    def refresh_buckets(self, rank: int = None):
        self.compute_aspect_ratio_bucket_indices()
        logger.debug("Refreshing webshart buckets for rank %s via data_backend id %s.", rank, self.id)
        return
