"""Seekable membership views backed by prepared geometry or legacy paged JSON."""

import hashlib
import json
import os
import sqlite3
from contextlib import ExitStack, closing, nullcontext
from itertools import batched
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory, TemporaryFile

from atomicwrites import atomic_write

from simpletuner.helpers.metadata.preparation import DatasetDescription

from .webshart_geometry import _read_index, geometry_lookup, prepared_geometry_path, snapshot_path

_PREFIX = b'{"webshart_page_table":"'
_INDEX_PREFIX = b'{"membership_format":"sqlite-v1",'
_OFFSET_WIDTH = 16
_PAGE_SIZE = 256


def shard_runs_path(publication, revision):
    return snapshot_path(publication, revision).with_suffix(".shard-runs.sqlite")


def write_membership(path, buckets, properties, *, shard_of=None):
    """Publish seekable membership and optional shard ranges for the same revision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {str(key): len(refs) for key, refs in buckets.items()}
    header = {**properties, "bucket_counts": counts, "page_size": _PAGE_SIZE}
    with atomic_write(path, mode="wb", overwrite=True) as output, TemporaryFile() as offsets, ExitStack() as stack:
        runs = None
        if shard_of is not None:
            staging = Path(stack.enter_context(TemporaryDirectory(dir=path.parent))) / "runs.sqlite"
            runs = stack.enter_context(closing(sqlite3.connect(staging)))
            runs.execute("CREATE TABLE publication(revision TEXT NOT NULL)")
            runs.execute("CREATE TABLE runs(shard INTEGER, bucket INTEGER, start INTEGER, n INTEGER)")
            header["shard_runs_format"] = "sqlite-v1"
        output.write(_PREFIX + b"0" * _OFFSET_WIDTH + b'","view_revision":"')
        revision_position = output.tell()
        output.write(b"0" * 64 + b'",')
        output.write(json.dumps(header, default=os.fspath).encode()[1:-1] + b",\n")
        revision = hashlib.sha256(
            json.dumps(
                [properties.get("source_revision"), properties.get("geometry_sha256"), counts], separators=(",", ":")
            ).encode()
        )
        output.write(b'"aspect_ratio_bucket_indices":{')
        for number, (key, refs) in enumerate(buckets.items()):
            if number:
                output.write(b",")
            output.write(json.dumps(str(key)).encode() + b":[")
            written = 0
            previous = None
            start = 0
            for page, chunk in enumerate(batched(refs, _PAGE_SIZE)):
                if runs is not None:
                    for index, ref in enumerate(chunk, written):
                        shard = shard_of(ref)
                        if previous is not None and shard != previous:
                            runs.execute("INSERT INTO runs VALUES (?,?,?,?)", (previous, number, start, index - start))
                            start = index
                        previous = shard
                written += len(chunk)
                if page:
                    output.write(b",")
                offsets.write(f"{output.tell():016x}".encode("ascii"))
                encoded = json.dumps(chunk, default=os.fspath).encode()
                revision.update(encoded)
                output.write(encoded[1:-1] + b"\n")
            if written != counts[str(key)]:
                raise ValueError("Published Webshart bucket length differs from its membership.")
            if previous is not None:
                runs.execute("INSERT INTO runs VALUES (?,?,?,?)", (previous, number, start, written - start))
            output.write(b"]")
        output.write(b'},"page_offsets":"')
        table_start = output.tell()
        offsets.seek(0)
        copyfileobj(offsets, output)
        output.write(b'"}')
        output.seek(len(_PREFIX))
        output.write(f"{table_start:016x}".encode("ascii"))
        output.seek(revision_position)
        view_revision = revision.hexdigest()
        output.write(view_revision.encode("ascii"))
        if runs is not None:
            runs.execute("INSERT INTO publication VALUES (?)", (view_revision,))
            runs.commit()
            os.replace(staging, shard_runs_path(path, view_revision))


def publish_indexed_membership(path, header):
    """Commit a view after its immutable index files have been published."""
    with atomic_write(path, mode="w", overwrite=True, encoding="utf-8") as output:
        json.dump({"membership_format": "sqlite-v1", **header}, output, separators=(",", ":"), default=os.fspath)


class MembershipPages:
    def __init__(self, source, header):
        self.source = source
        self.header = header
        self.counts = header["bucket_counts"]
        self.page_size = header.get("page_size", _PAGE_SIZE)

    @classmethod
    def open(cls, source):
        """Return the index for a new publication, or None for legacy unindexed JSON."""
        prefix = source.read(len(_INDEX_PREFIX))
        if prefix == _INDEX_PREFIX:
            source.seek(0)
            return IndexedMembership.open(source)
        source.seek(0)
        if source.read(len(_PREFIX)) != _PREFIX:
            source.seek(0)
            return None
        header = json.loads(_PREFIX + source.readline().removesuffix(b",\n") + b"}")
        if header.get("scan_in_progress"):
            raise ValueError("Webshart scan is incomplete.")
        return cls(source, header)

    def describe(self, spec=None):
        batches = sum(spec.batches_for(count) for count in self.counts.values()) if spec is not None else None
        return DatasetDescription(
            source_revision=self.header.get("source_revision"),
            view_revision=self.header.get("view_revision"),
            physical_rows=self.header.get("physical_rows"),
            accepted_samples=self.count(),
            batches=batches,
            scheduled_occurrences=batches * spec.batch_size * spec.dp_size if spec is not None else None,
        )

    def count(self, bucket=None):
        return sum(self.counts.values()) if bucket is None else self.counts.get(bucket, 0)

    def iter_refs(self, bucket, offset, limit):
        if offset < 0 or limit < 0:
            raise ValueError("Page offset and limit must be nonnegative.")
        table_start = int(self.header["webshart_page_table"], 16)
        table_index = 0
        for key, count in self.counts.items():
            if not limit:
                return
            if bucket is None or key == bucket:
                if offset < count:
                    page, skip = divmod(offset, self.page_size)
                    while page * self.page_size < count and limit:
                        self.source.seek(table_start + (table_index + page) * _OFFSET_WIDTH)
                        position = int(self.source.read(_OFFSET_WIDTH), 16)
                        self.source.seek(position)
                        refs = json.loads(b"[" + self.source.readline() + b"]")[skip : skip + limit]
                        yield from refs
                        limit -= len(refs)
                        page += 1
                        skip = 0
                offset = max(0, offset - count)
            table_index += (count + self.page_size - 1) // self.page_size

    def read_page(self, bucket, offset, limit):
        return list(self.iter_refs(bucket, offset, limit)), self.count(bucket)

    def read_positions(self, bucket, positions):
        pages = {}
        refs = []
        for position in positions:
            page, index = divmod(position, self.page_size)
            if page not in pages:
                pages[page] = list(self.iter_refs(bucket, page * self.page_size, self.page_size))
            refs.append(pages[page][index])
        return refs

    def partition_groups(self, worker, workers):
        if workers < 1 or not 0 <= worker < workers:
            raise ValueError("Preparation worker must be in the range [0, workers).")
        size, extra = divmod(self.count(), workers)
        start = worker * size + min(worker, extra)
        end = start + size + (worker < extra)
        offset = 0
        for bucket, count in self.counts.items():
            first, last = max(offset, start), min(offset + count, end)
            if first < last:
                yield bucket, self.iter_refs(bucket, first - offset, last - first)
            offset += count

    def partition(self, worker, workers):
        for _, refs in self.partition_groups(worker, workers):
            yield from refs


class IndexedMembership(MembershipPages):
    """Read accepted references from the same index that serves prepared geometry."""

    @classmethod
    def open(cls, source):
        return cls(source, json.load(source))

    def open_geometry(self):
        from .webshart_exclusions import exclusion_path

        publication = Path(self.source.name)
        db = _read_index(prepared_geometry_path(publication, self.header))
        revision = self.header.get("membership_exclusion_revision")
        if revision is not None:
            path = exclusion_path(publication, revision).resolve().as_uri()
            try:
                db.execute("ATTACH DATABASE ? AS membership_exclusions", (path + "?mode=ro&immutable=1",))
            except BaseException:
                db.close()
                raise
        return db

    def _has_exclusions(self, bucket):
        base = self.header.get("base_bucket_counts")
        return base is not None and base[bucket] != self.counts[bucket]

    @staticmethod
    def _excluded_through_position(db, bucket, visible):
        row = db.execute(
            "SELECT ordinal-position+1 FROM membership_exclusions.positions "
            "WHERE bucket=? AND position<=? ORDER BY position DESC,ordinal DESC LIMIT 1",
            (bucket, visible),
        ).fetchone()
        return row[0] if row is not None else 0

    @staticmethod
    def _excluded_before_ordinal(db, bucket, physical):
        row = db.execute(
            "SELECT ordinal-position+1 FROM membership_exclusions.positions "
            "WHERE bucket=? AND ordinal<? ORDER BY ordinal DESC LIMIT 1",
            (bucket, physical),
        ).fetchone()
        return row[0] if row is not None else 0

    def _physical_ranges(self, db, bucket, start, stop):
        if not self._has_exclusions(bucket):
            yield start, stop
            return

        shift = self._excluded_through_position(db, bucket, start)
        for position, last in db.execute(
            "SELECT position,max(ordinal) FROM membership_exclusions.positions "
            "WHERE bucket=? AND position>? AND position<? GROUP BY position ORDER BY position",
            (bucket, start, stop),
        ):
            yield start + shift, position + shift
            start, shift = position, last - position + 1
        yield start + shift, stop + shift

    def read_positions(self, bucket, positions, *, db=None):
        with closing(self.open_geometry()) if db is None else nullcontext(db) as connection:
            if self._has_exclusions(bucket):
                positions = (
                    position + self._excluded_through_position(connection, bucket, position) for position in positions
                )
            return geometry_lookup(self.source.name, self.header, connection).read_positions(bucket, positions)

    def shard_runs(self, db):
        db.execute("CREATE TEMP TABLE bucket_order(bucket TEXT PRIMARY KEY, position INTEGER) WITHOUT ROWID")
        db.executemany(
            "INSERT INTO bucket_order VALUES (?,?)", ((bucket, position) for position, bucket in enumerate(self.counts))
        )
        for shard, bucket, position, start, size in db.execute(
            "SELECT shard,bucket,position,start,n FROM runs JOIN bucket_order USING(bucket) "
            "ORDER BY shard_order(shard),shard,position,start"
        ):
            if self._has_exclusions(bucket):
                end = start + size
                visible_start = start - self._excluded_before_ordinal(db, bucket, start)
                size = end - self._excluded_before_ordinal(db, bucket, end) - visible_start
                start = visible_start
            if size:
                yield shard, position, start, size

    def iter_refs(self, bucket, offset, limit):
        if offset < 0 or limit < 0:
            raise ValueError("Page offset and limit must be nonnegative.")
        with closing(self.open_geometry()) as db:
            lookup = geometry_lookup(self.source.name, self.header, db)
            for key, count in self.counts.items():
                if not limit:
                    return
                if bucket is not None and key != bucket:
                    continue
                count_to_read = min(limit, max(0, count - offset))
                if count_to_read:
                    for start, stop in self._physical_ranges(db, key, offset, offset + count_to_read):
                        yield from lookup.iter_refs(key, start, stop)
                    limit -= count_to_read
                offset = max(0, offset - count)


class IndexedValidationCursor:
    """Select a bounded, wrapping validation page from one pinned publication."""

    def __init__(self, open_source, check):
        self.open_source = open_source
        self.check = check
        self.pages = None
        self.position = 0

    def _open(self):
        if self.pages is None:
            source = self.open_source()
            try:
                pages = MembershipPages.open(source)
                self.check(pages)
                self.pages = pages
            except BaseException:
                source.close()
                raise
        return self.pages

    def next_refs(self, limit):
        if limit <= 0:
            return []
        pages = self._open()
        count = pages.count()
        limit = min(count, limit)
        refs = list(pages.iter_refs(None, self.position, limit))
        if len(refs) < limit:
            refs.extend(pages.iter_refs(None, 0, limit - len(refs)))
        if count:
            self.position = (self.position + len(refs)) % count
        return refs

    def state_dict(self):
        if self.pages is None:
            return None
        return {"view": self.pages.header["view_revision"], "position": self.position}

    def load_state_dict(self, state):
        self.close()
        if state is None:
            return
        pages = self._open()
        if state["view"] != pages.header["view_revision"]:
            raise ValueError("Validation resume requires the same accepted Webshart view.")
        if type(state["position"]) is not int or not 0 <= state["position"] < max(1, pages.count()):
            raise ValueError("Invalid validation position.")
        self.position = state["position"]

    def close(self):
        if self.pages is not None:
            self.pages.source.close()
            self.pages = None
        self.position = 0
