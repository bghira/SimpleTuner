import hashlib
import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from itertools import batched, chain
from types import SimpleNamespace

from filelock import FileLock

from simpletuner.helpers.data_backend.base import conditioning_file_path
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.metadata.backends.webshart_exclusions import collect_exclusions, exclusion_path
from simpletuner.helpers.metadata.backends.webshart_geometry import accepted_geometry_predicate, link_snapshot
from simpletuner.helpers.metadata.backends.webshart_membership import (
    IndexedMembership,
    MembershipPages,
    publish_indexed_membership,
)
from simpletuner.helpers.metadata.preparation import PreparationView
from simpletuner.helpers.utils.pathing import canonicalize_data_uri


class AlignedConditioningMetadata(MetadataBackend):
    """Map conditioning files onto a pinned source preparation view."""

    requires_complete_preparation = True
    stable_prepared_geometry = True

    def __init__(self, *, source, generated, **kwargs):
        super().__init__(load_cache_on_init=False, **kwargs)
        self.source = source
        self.source_metadata = source.metadata_backend
        description = source.describe()
        self.source_view_revision = description.view_revision
        conditioning = self.dataset_config.get("conditioning_config") or {}
        inputs = [self.source_view_revision, conditioning]
        if not generated:
            inputs.append("configured")
        revision = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        self.description = replace(description, view_revision=revision, physical_rows=None)
        self.generated = generated
        self.artifact_root = os.path.abspath(
            os.path.join(self.instance_data_dir, revision) if generated else self.instance_data_dir
        )
        self.first_frame = conditioning.get("type") == "i2v_first_frame"
        source_path = self.source_metadata.membership_path
        artifact_scope = hashlib.sha256(os.path.abspath(self.cache_file).encode()).hexdigest()[:20]
        self._publication = source_path.with_name(f"conditioning-{artifact_scope}-{revision}.json")
        self._filtered = None
        self._filter_scope = ExitStack()
        self._contains = None
        self.reload_cache()
        self.image_metadata_loaded = True
        self.set_readonly()

    def describe(self, spec=None):
        if spec is not None:
            raise ValueError("Aligned conditioning uses the source schedule.")
        if self.source_metadata.view_revision != self.source_view_revision:
            raise ValueError("Source membership changed. Reload the aligned conditioning view.")
        if self._filtered is None:
            return self.description
        return replace(
            self.description,
            view_revision=self._filtered["view_revision"],
            accepted_samples=sum(self._filtered["bucket_counts"].values()),
        )

    def close(self):
        self._filter_scope.close()
        self._contains = None

    def reload_cache(self, set_config=False):
        self.close()
        self._filtered = None
        if self._publication.is_file():
            with self._publication.open("rb") as stream:
                self._filtered = MembershipPages.open(stream).header

    @contextmanager
    def _filtered_view(self):
        with self._publication.open("rb") as stream:
            pages = MembershipPages.open(stream)
            if pages.header["view_revision"] != self._filtered["view_revision"]:
                raise ValueError("Conditioning membership changed. Reload the aligned conditioning view.")
            yield pages

    def _publish_initial_membership(self):
        with FileLock(str(self._publication) + ".publish.lock"):
            if self._publication.is_file():
                return
            with self.source_metadata._open_membership() as stream:
                pages = MembershipPages.open(stream)
                self.source_metadata._check_publication(pages)
                if not isinstance(pages, IndexedMembership):
                    raise ValueError("Publish indexed Webshart geometry before filtering aligned conditioning.")
                header = {**pages.header, "view_revision": self.description.view_revision}
                if revision := header.get("exclusion_revision"):
                    link_snapshot(
                        exclusion_path(self.source_metadata.membership_path, revision),
                        exclusion_path(self._publication, revision),
                    )
                publish_indexed_membership(self._publication, header)

    @contextmanager
    def collect_exclusions(self):
        self.describe()
        self._publish_initial_membership()
        self.reload_cache()
        with collect_exclusions(self, publication=self._publication, shard_of=self.source.data_backend.shard_of) as ledger:
            yield SimpleNamespace(add=lambda ref, bucket, reason: ledger.add(self._source_ref(ref), bucket, reason))

    def __len__(self):
        return self.describe().accepted_samples

    def _target_ref(self, source_ref):
        if not self.generated:
            return conditioning_file_path(source_ref, self.artifact_root, self.dataset_config)
        relative_path = canonicalize_data_uri(source_ref).removeprefix("webshart://")
        if os.path.isabs(relative_path) or ".." in relative_path.split("/"):
            raise ValueError("Generated Webshart conditioning requires a source reference with a safe relative path.")
        return os.path.join(self.artifact_root, relative_path + (".png" if self.first_frame else ""))

    def _source_ref(self, target_ref):
        relative = os.path.relpath(target_ref, self.artifact_root)
        if relative == ".." or relative.startswith(f"..{os.sep}"):
            raise ValueError("Conditioning artifact does not belong to the prepared source revision.")
        if self.generated:
            return "webshart://" + (relative.removesuffix(".png") if self.first_frame else relative)
        candidates = self.source.data_backend.parquet_sample_refs_for_filename(relative, first_frame=self.first_frame)
        return next((ref for ref in candidates if self.source_metadata.admits_related_sample(ref)), None)

    def conditioning_ref(self, pairing_key):
        self.describe()
        if self._filtered is None:
            contains = self.source_metadata.admits_related_sample
        else:
            if self._contains is None:
                self._contains = accepted_geometry_predicate(
                    self.source_metadata._geometry_lookup(), self._publication, self._filtered, self._filter_scope
                )
            contains = self._contains
        if contains(canonicalize_data_uri(pairing_key)):
            return self._target_ref(pairing_key)
        return None

    def read_page(self, *, bucket=None, offset=0, limit=50):
        self.describe()
        if self._filtered is None:
            refs, total = self.source.read_page(bucket=bucket, offset=offset, limit=limit)
        else:
            with self._filtered_view() as pages:
                refs, total = pages.read_page(bucket, offset, limit)
        return [self._target_ref(ref) for ref in refs], total

    def iter_preparation_batches(self, worker, workers, batch_size):
        self.describe()
        with self._open_groups(worker, workers) as groups:
            for bucket, refs in groups:
                for batch in batched(refs, batch_size):
                    yield bucket, tuple(map(self._target_ref, batch))

    @contextmanager
    def _open_groups(self, worker, workers):
        if self._filtered is None:
            with self.source_metadata._open_membership() as stream:
                pages = MembershipPages.open(stream)
                self.source_metadata._check_publication(pages)
                yield pages.partition_groups(worker, workers)
        else:
            with self._filtered_view() as pages:
                yield pages.partition_groups(worker, workers)

    @contextmanager
    def open_preparation_view(self, *, worker=0, workers=1):
        self.describe()
        with self._open_groups(worker, workers) as groups:
            refs = chain.from_iterable(refs for _, refs in groups)
            yield PreparationView(map(self._target_ref, refs), self)

    def scan_dataset_cache(self, dataset, cache_behavior):
        if dataset.vae_cache is None:
            return
        if cache_behavior != "recreate":
            raise ValueError("Aligned native conditioning cache scans require vae_cache_scan_behaviour='recreate'.")
        dataset.vae_cache.remove_inconsistent_prepared_latents(dataset)

    def get_metadata_by_filepath(self, filepath):
        return next(self.get_metadata_batch((filepath,)))

    def get_metadata_batch(self, refs):
        self.describe()
        refs = tuple(refs)
        source_refs = [self._source_ref(ref) for ref in refs]
        geometry = self.source_metadata.get_metadata_batch(source_refs)
        for ref, source_ref, metadata in zip(refs, source_refs, geometry, strict=True):
            if source_ref is None or metadata is None:
                yield None
            else:
                yield {**metadata, "training_sample_path": source_ref, "image_path": ref}

    def iter_validation_cycles(self):
        self.describe()
        if self._filtered is None:
            for refs in self.source_metadata.iter_validation_cycles():
                yield map(self._target_ref, refs)
            return
        while True:
            with self._filtered_view() as pages:
                yield map(self._target_ref, pages.iter_refs(None, 0, pages.count()))

    def load_image_metadata(self):
        return
