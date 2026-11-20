"""Metadata access scoped to one traversal of the accepted samples."""

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from heapq import nsmallest
from itertools import batched, chain, groupby, islice
from operator import itemgetter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from atomicwrites import atomic_write
from filelock import FileLock


class BucketValidationCursor:
    """Page through current bucket membership in path order, including after a rescan."""

    def __init__(self, metadata):
        self.metadata = metadata
        self.after = None

    def close(self):
        self.after = None

    def next_refs(self, limit):
        if limit <= 0:
            return []
        buckets = self.metadata.aspect_ratio_bucket_indices
        refs = chain.from_iterable(buckets.values())
        selected = nsmallest(limit, (ref for ref in refs if self.after is None or ref > self.after))
        if len(selected) < limit and self.after is not None:
            refs = chain.from_iterable(buckets.values())
            selected.extend(nsmallest(limit - len(selected), (ref for ref in refs if ref <= self.after)))
        self.after = selected[-1] if selected else None
        return selected


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


@dataclass(frozen=True)
class TrainingPreparation:
    online_text: bool
    ondemand_latents: bool
    requires_complete_view: bool


@dataclass(frozen=True)
class DatasetDescription:
    source_revision: str | None = None
    view_revision: str | None = None
    physical_rows: int | None = None
    accepted_samples: int | None = None
    scheduled_occurrences: int | None = None
    batches: int | None = None


class DatasetViewChangedError(ValueError):
    """A consumer requested a view which is no longer open in its provider."""


@dataclass(frozen=True)
class LegacyMetadataEdit:
    record: dict
    revision: str


class LegacyMetadataPublication:
    """A file-backed metadata view with serialized, atomic file edits."""

    def __init__(self, path):
        self.path = Path(path)

    @staticmethod
    def _revision(stat):
        version = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        return hashlib.sha256(repr(version).encode()).hexdigest()

    def _read_snapshot(self):
        with self.path.open(encoding="utf-8") as source:
            return json.load(source), self._revision(os.fstat(source.fileno()))

    def read(self, sample_ref):
        records, revision = self._read_snapshot()
        return records.get(sample_ref), revision

    def edit(self, sample_ref, update, *, expected_revision=None):
        with FileLock(str(self.path) + ".lock"):
            records, revision = self._read_snapshot()
            if expected_revision is not None and expected_revision != revision:
                raise DatasetViewChangedError("The metadata changed. Reopen the file before editing it.")
            if sample_ref not in records or (record := update(records[sample_ref])) is None:
                return None
            records[sample_ref] = record
            with atomic_write(self.path, overwrite=True, encoding="utf-8") as output:
                json.dump(records, output)
                output.flush()
                published_revision = self._revision(os.fstat(output.fileno()))
                if self._revision(self.path.stat()) != revision:
                    raise DatasetViewChangedError("The metadata changed. Reopen the file before editing it.")
            return LegacyMetadataEdit(record, published_revision)


@dataclass(frozen=True)
class DatasetReadView:
    """A published view and, for training, its deterministic epoch transform."""

    revision: str
    epoch: int | None = None
    seed: int = 0
    partition: int | None = None


class MetadataLookup(Protocol):
    def get_metadata_by_filepath(self, filepath: str) -> dict | None: ...


@dataclass(frozen=True)
class PreparationView:
    refs: Iterable[str]
    metadata: MetadataLookup

    def __iter__(self):
        return iter(self.refs)


@dataclass(frozen=True)
class ShardedPreparationIndex:
    directory: Path
    partitions: int
    accepted_samples: int

    @classmethod
    def open(cls, source, *, source_revision, partition_sizes, partition_for):
        stat = os.fstat(source.fileno())
        version = json.dumps(
            [1, source_revision, partition_sizes, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        )
        revision = hashlib.sha256(version.encode()).hexdigest()
        root = Path(source.name).with_suffix(".preparation")
        directory = root / revision
        if not directory.is_dir():
            root.mkdir(parents=True, exist_ok=True)
            with FileLock(str(root / f"{revision}.lock")):
                if not directory.is_dir():
                    cls._publish(source, directory, partition_for, source_revision)
        return cls(directory, len(partition_sizes), int((directory / "count").read_text(encoding="utf-8")))

    @staticmethod
    def _publish(source, destination, partition_for, source_revision):
        with TemporaryDirectory(dir=destination.parent) as temporary:
            staging = Path(temporary) / "membership"
            staging.mkdir()
            count = 0
            for bucket, refs in read_bucket_groups(source, source_revision=source_revision):
                for batch in batched(refs, 2048):
                    groups = defaultdict(list)
                    for ref in batch:
                        groups[partition_for(ref)].append(ref)
                    for partition, selected in groups.items():
                        with (staging / f"{partition}.jsonl").open("a", encoding="utf-8") as output:
                            output.writelines(json.dumps((bucket, ref)) + "\n" for ref in selected)
                    count += len(batch)
            (staging / "count").write_text(str(count), encoding="utf-8")
            os.replace(staging, destination)

    def groups(self, worker, workers):
        for partition in range(worker, self.partitions, workers):
            path = self.directory / f"{partition}.jsonl"
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as source:
                records = map(json.loads, source)
                for bucket, rows in groupby(records, key=itemgetter(0)):
                    yield bucket, map(itemgetter(1), rows)


def bucket_groups_from_events(events):
    """Yield each JSON bucket and a stream of its refs from inside the bucket map."""
    for _, event, bucket in events:
        if event == "end_map":
            return
        if event != "map_key" or next(events)[1] != "start_array":
            raise ValueError("Expected a list of sample references for each aspect bucket.")

        def read_refs():
            for _, event, ref in events:
                if event == "end_array":
                    return
                if event != "string":
                    raise ValueError("Expected a string sample reference in the bucket index.")
                yield ref

        refs = read_refs()
        yield bucket, refs
        for _ in refs:
            pass


def read_bucket_groups(source, *, source_revision=None):
    import ijson

    events = iter(ijson.parse(source))
    for prefix, event, value in events:
        if prefix == "source_revision" and event == "string" and source_revision is not None and value != source_revision:
            raise ValueError("Cached membership belongs to a different source revision. Scan the dataset again.")
        if prefix == "aspect_ratio_bucket_indices" and event == "start_map":
            yield from bucket_groups_from_events(events)
            return
    raise ValueError("Missing aspect_ratio_bucket_indices in dataset metadata.")


def read_bucket_page(groups, bucket, offset, limit):
    if offset < 0 or limit < 0:
        raise ValueError("Page offset and limit must be nonnegative.")
    stop = offset + limit
    page = []
    total = 0
    for key, refs in groups:
        if bucket is not None and key != bucket:
            continue
        for ref in refs:
            if offset <= total < stop:
                page.append(ref)
            total += 1
    return page, total
