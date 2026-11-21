"""Publication of Discovery and Parquet sample geometry."""

import json
import shutil
from contextlib import contextmanager, nullcontext
from hashlib import sha256
from itertools import batched, chain
from tempfile import TemporaryDirectory
from uuid import uuid4

from filelock import FileLock

from simpletuner.helpers.data_backend.dataset_types import DatasetType, get_arg_value
from simpletuner.helpers.metadata.transform_store import GEOMETRY_FIELDS, DatasetTransformStore, transform_recipe_directory
from simpletuner.helpers.training.state_tracker import StateTracker


class SQLiteGeometryMetadata:
    _transform_recipe = None
    _rebuilding_transforms = False
    _bucket_version = None

    def __init__(self, *args, **kwargs):
        self._dirty_geometry = set()
        super().__init__(*args, **kwargs)

    def _metadata_changed(self, refs):
        if self._uses_transform_store():
            self._dirty_geometry.update(refs)

    def _uses_transform_store(self):
        return (
            getattr(self.data_backend, "type", None) in ("local", "csv", "aws")
            and self.dataset_type is not DatasetType.AUDIO
            and not self.has_copied_metadata
        )

    @property
    def uses_remote_transform_store(self):
        return getattr(self.data_backend, "type", None) == "aws" and self._uses_transform_store()

    def _new_transform_recipe(self):
        args = self.runtime_args
        model = StateTracker.get_model()
        settings = {
            "config": self.active_dataset_config,
            "resolution": (self.resolution, self.resolution_type),
            "alignment": get_arg_value(args, "aspect_bucket_alignment"),
            "rounding": get_arg_value(args, "aspect_bucket_rounding"),
            "canvas": getattr(model, "MAXIMUM_CANVAS_SIZE", None),
        }
        digest = sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:16]
        if self.recrops_each_epoch(rebuild_random_aspect=True):
            digest = f"{digest}-epoch-{StateTracker.get_epoch() or 0}"
        return f"{digest}-{uuid4().hex}"

    def _read_bucket_document(self):
        if self.uses_remote_transform_store:
            return self.data_backend.read_with_version(self.cache_file)
        return self.data_backend.read(self.cache_file), None

    @contextmanager
    def _open_transform_store(self, recipe, *, write=False):
        directory = transform_recipe_directory(self.metadata_file, recipe)
        if self.data_backend.type != "aws":
            store = DatasetTransformStore(directory)
            if not write and not store.path.is_file():
                raise FileNotFoundError(store.path)
            yield store
            return

        key = directory / DatasetTransformStore.filename
        with TemporaryDirectory() as temporary:
            store = DatasetTransformStore(temporary)
            if not write:
                with self.data_backend.open_read_stream(key) as source, store.path.open("wb") as target:
                    shutil.copyfileobj(source, target)
            yield store
            if write:
                with store.path.open("rb") as source:
                    self.data_backend.write(key, source)

    def _adopt_transform_recipe(self, cache_data, version):
        previous = self._transform_recipe
        recipe = cache_data.get("transform_recipe")
        if recipe is not None:
            transform_recipe_directory(self.metadata_file, recipe)
        if (recipe is not None or previous is not None) and getattr(self, "image_metadata_loaded", False):
            self.image_metadata, self.image_metadata_loaded = self._read_image_metadata(
                recipe, cache_data.get("aspect_ratio_bucket_indices", {})
            )
        self._transform_recipe = recipe
        self._bucket_version = version

    def load_image_metadata(self):
        recipe = self._transform_recipe
        if recipe is not None and not self._uses_transform_store():
            recipe = None
        self.image_metadata, self.image_metadata_loaded = self._read_image_metadata(recipe, self.aspect_ratio_bucket_indices)

    def _measurement_path(self, recipe):
        if recipe is None:
            return self.metadata_file
        return transform_recipe_directory(self.metadata_file, recipe) / DatasetTransformStore.measurements_filename

    def _read_image_metadata(self, recipe, buckets):
        path = self._measurement_path(recipe)
        if recipe is None and not self.data_backend.exists(path):
            return {}, False

        raw = self.data_backend.read(path)
        if raw is None:
            raise FileNotFoundError(path)
        metadata = json.loads(raw)
        if recipe is not None:
            with self._open_transform_store(recipe) as store:
                for paths in batched(metadata, 2048):
                    for path, record in zip(paths, store.read_batch(paths)):
                        if record is not None:
                            metadata[path].update(record)
            self._require_bucket_geometry(metadata, buckets)
        return metadata, True

    def _require_bucket_geometry(self, metadata, buckets):
        missing = self._first_sample_without_geometry(metadata, buckets)
        if missing is not None:
            raise ValueError(f"(id={self.id}) Published bucket sample is missing geometry: {missing!r}. Rescan the dataset.")

    @staticmethod
    def _first_sample_without_geometry(metadata, buckets):
        return next(
            (ref for ref in chain.from_iterable(buckets.values()) if not GEOMETRY_FIELDS.issubset(metadata.get(ref, {}))),
            None,
        )

    def _needs_geometry_rebuild(self, requested):
        if requested or not self._uses_transform_store() or self._transform_recipe is not None:
            return requested
        if not any(self.aspect_ratio_bucket_indices.values()):
            return False
        if not self.image_metadata_loaded:
            self.load_image_metadata()
        return self._first_sample_without_geometry(self.image_metadata, self.aspect_ratio_bucket_indices) is not None

    def _write_transforms(self, recipe, refs):
        with self._open_transform_store(recipe, write=True) as store:
            for paths in batched(refs, 2048):
                records = {}
                for path in paths:
                    metadata = self.image_metadata.get(path, {})
                    geometry = {key: metadata[key] for key in GEOMETRY_FIELDS if key in metadata}
                    if geometry:
                        records[str(path)] = geometry
                store.update(records)

    def _measured_records(self, refs):
        return {
            path: {key: value for key, value in self.image_metadata[path].items() if key not in GEOMETRY_FIELDS}
            for path in refs
        }

    def _write_measured_metadata(self, recipe):
        self.data_backend.write(self._measurement_path(recipe), json.dumps(self._measured_records(self.image_metadata)))

    def save_image_metadata(self):
        if not self._uses_transform_store():
            self.data_backend.write(self.metadata_file, json.dumps(self.image_metadata))
        elif not self._rebuilding_transforms:
            with self._geometry_publication_lock():
                if self._transform_recipe is None:
                    self.data_backend.write(self.metadata_file, json.dumps(self.image_metadata))
                elif self.data_backend.type == "aws":
                    raw, version = self._read_bucket_document()
                    if version != self._bucket_version:
                        raise RuntimeError("Dataset membership changed before saving its metadata. Reload the dataset.")
                    self._publish_geometry_and_buckets(json.loads(raw), replace=True)
                else:
                    self._save_current_geometry()

    def _geometry_publication_lock(self):
        if self.data_backend.type == "aws":
            return nullcontext()
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.metadata_file) + ".lock")

    def _save_current_geometry(self):
        if not self._dirty_geometry:
            return
        self._write_transforms(self._transform_recipe, self._dirty_geometry)
        path = self._measurement_path(self._transform_recipe)
        records = json.loads(self.data_backend.read(path))
        records.update(self._measured_records(self._dirty_geometry))
        self.data_backend.write(path, json.dumps(records))
        self._dirty_geometry.clear()

    def flush_prepared_metadata(self):
        if self.uses_remote_transform_store:
            # The S3 index update publishes the measured and transform objects together.
            return
        with self.metadata_semaphor:
            super().flush_prepared_metadata()

    def _publish_bucket_cache(self, cache_data):
        if self._uses_transform_store():
            with self.metadata_semaphor, self._geometry_publication_lock():
                self._publish_geometry_and_buckets(cache_data)
        else:
            self.data_backend.write(self.cache_file, json.dumps(cache_data))

    def _publish_geometry_and_buckets(self, cache_data, *, replace=False):
        if not self.image_metadata and not self.image_metadata_loaded and not self._rebuilding_transforms:
            self.load_image_metadata()
        previous = recipe = self._transform_recipe
        remote = self.data_backend.type == "aws"
        if previous is None or self._rebuilding_transforms or replace or (remote and self._dirty_geometry):
            if any(GEOMETRY_FIELDS.intersection(record) for record in self.image_metadata.values()):
                recipe = self._new_transform_recipe()
                self._require_bucket_geometry(self.image_metadata, cache_data["aspect_ratio_bucket_indices"])
                self._write_transforms(recipe, self.image_metadata)
                self._write_measured_metadata(recipe)
            else:
                recipe = None
        elif self._dirty_geometry:
            self._save_current_geometry()
        cache_data.pop("transform_recipe", None)
        if recipe is not None:
            cache_data = {"transform_recipe": recipe, **cache_data}
        encoded = json.dumps(cache_data)
        if remote:
            self._bucket_version = self.data_backend.write_if_version(
                self.cache_file, encoded, expected_version=self._bucket_version
            )
        else:
            self.data_backend.write(self.cache_file, encoded)
        self._transform_recipe = recipe
        self._dirty_geometry.clear()
        if recipe is not None and recipe != previous:
            self._write_measured_metadata(None)

    @contextmanager
    def _replace_geometry(self, rebuild):
        if rebuild and self._uses_transform_store():
            published = self._transform_recipe
            original_buckets = self.aspect_ratio_bucket_indices
            original_metadata = self.image_metadata
            original_loading_state = self.image_metadata_loaded
            original_statistics = self.filtering_statistics
            original_dirty = self._dirty_geometry.copy()
            self._rebuilding_transforms = True
            try:
                yield
            except Exception:
                if self._transform_recipe == published:
                    self.aspect_ratio_bucket_indices = original_buckets
                    self.image_metadata = original_metadata
                    self.image_metadata_loaded = original_loading_state
                    self.filtering_statistics = original_statistics
                    self._dirty_geometry = original_dirty
                raise
            finally:
                self._rebuilding_transforms = False
        else:
            yield
