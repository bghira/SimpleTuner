import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import replace

from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.metadata.preparation import PreparationView
from simpletuner.helpers.utils.pathing import canonicalize_data_uri


class AlignedConditioningMetadata(MetadataBackend):
    """Map generated conditioning artifacts onto a pinned source preparation view."""

    requires_complete_preparation = True

    def __init__(self, *, source, **kwargs):
        super().__init__(load_cache_on_init=False, **kwargs)
        self.source = source
        description = source.describe()
        self.source_view_revision = description.view_revision
        conditioning = self.dataset_config.get("conditioning_config") or {}
        inputs = [self.source_view_revision, conditioning]
        revision = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        self.description = replace(description, view_revision=revision, physical_rows=None)
        self.artifact_root = os.path.abspath(os.path.join(self.instance_data_dir, revision))
        self.first_frame = conditioning.get("type") == "i2v_first_frame"
        self.image_metadata_loaded = True
        self.set_readonly()

    def describe(self, spec=None):
        if spec is not None:
            raise ValueError("Generated conditioning uses the source schedule.")
        if self.source.metadata_backend.view_revision != self.source_view_revision:
            raise ValueError("Source membership changed. Reload the generated conditioning view.")
        return self.description

    def __len__(self):
        return self.describe().accepted_samples

    def _target_ref(self, source_ref):
        relative_path = canonicalize_data_uri(source_ref).removeprefix("webshart://")
        if os.path.isabs(relative_path) or ".." in relative_path.split("/"):
            raise ValueError("Generated Webshart conditioning requires a source reference with a safe relative path.")
        return os.path.join(self.artifact_root, relative_path + (".png" if self.first_frame else ""))

    def _source_ref(self, target_ref):
        relative = os.path.relpath(target_ref, self.artifact_root)
        if relative == ".." or relative.startswith(f"..{os.sep}"):
            raise ValueError("Conditioning artifact does not belong to the prepared source revision.")
        return "webshart://" + (relative.removesuffix(".png") if self.first_frame else relative)

    def conditioning_ref(self, pairing_key):
        self.describe()
        if not self.source.metadata_backend.admits_related_sample(pairing_key):
            return None
        return self._target_ref(pairing_key)

    def read_page(self, *, bucket=None, offset=0, limit=50):
        self.describe()
        refs, total = self.source.read_page(bucket=bucket, offset=offset, limit=limit)
        return [self._target_ref(ref) for ref in refs], total

    def iter_preparation_batches(self, worker, workers, batch_size):
        self.describe()
        for bucket, refs in self.source.iter_preparation_batches(worker=worker, workers=workers, batch_size=batch_size):
            yield bucket, tuple(self._target_ref(ref) for ref in refs)

    @contextmanager
    def open_preparation_view(self, *, worker=0, workers=1):
        self.describe()
        with self.source.metadata_backend.open_preparation_view(worker=worker, workers=workers) as view:
            yield PreparationView((self._target_ref(ref) for ref in view), self)

    def scan_dataset_cache(self, dataset, cache_behavior):
        if dataset.vae_cache is None:
            return
        if cache_behavior != "recreate":
            raise ValueError("Generated native conditioning cache scans require vae_cache_scan_behaviour='recreate'.")
        dataset.vae_cache.remove_inconsistent_prepared_latents(dataset)

    def get_metadata_by_filepath(self, filepath):
        return next(self.get_metadata_batch((filepath,)))

    def get_metadata_batch(self, refs):
        self.describe()
        refs = tuple(refs)
        source_refs = [self._source_ref(ref) for ref in refs]
        geometry = self.source.metadata_backend.get_metadata_batch(source_refs)
        for ref, source_ref, metadata in zip(refs, source_refs, geometry, strict=True):
            if metadata is None:
                yield None
            else:
                yield {**metadata, "training_sample_path": source_ref, "image_path": ref}

    def iter_validation_cycles(self):
        self.describe()
        for refs in self.source.metadata_backend.iter_validation_cycles():
            yield (self._target_ref(ref) for ref in refs)

    def load_image_metadata(self):
        return
