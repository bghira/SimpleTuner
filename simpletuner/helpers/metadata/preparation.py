"""Metadata access scoped to one traversal of the accepted samples."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


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
