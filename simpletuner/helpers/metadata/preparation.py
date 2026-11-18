"""Metadata access scoped to one traversal of the accepted samples."""

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import batched, groupby
from operator import itemgetter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from filelock import FileLock


@dataclass(frozen=True)
class DatasetDescription:
    source_revision: str | None = None
    view_revision: str | None = None
    physical_rows: int | None = None
    accepted_samples: int | None = None
    scheduled_occurrences: int | None = None
    batches: int | None = None


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
                    cls._publish(source, directory, partition_for)
        return cls(directory, len(partition_sizes), int((directory / "count").read_text(encoding="utf-8")))

    @staticmethod
    def _publish(source, destination, partition_for):
        with TemporaryDirectory(dir=destination.parent) as temporary:
            staging = Path(temporary) / "membership"
            staging.mkdir()
            count = 0
            for bucket, refs in read_bucket_groups(source):
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


def read_bucket_groups(source):
    import ijson

    events = iter(ijson.parse(source))
    for prefix, event, _ in events:
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
