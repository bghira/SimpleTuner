"""Publish batched filter results against a pinned Webshart view."""

import hashlib
import json
import os
import sqlite3
from collections import Counter
from contextlib import closing, contextmanager
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory

import torch.distributed as dist
from filelock import FileLock

from .webshart_geometry import _read_index, geometry_lookup, link_snapshot, prepared_geometry_path, snapshot_path
from .webshart_membership import IndexedMembership, MembershipPages, publish_indexed_membership, write_membership


def exclusion_path(publication, digest):
    return snapshot_path(Path(publication).with_suffix(".sqlite"), digest).with_suffix(".excluded.sqlite")


def retain_source_exclusions(publication, revision, staging, *, shard_of, old_pin, new_pin):
    """Retain exclusions only for shards with the same physical content revision."""
    original = exclusion_path(publication, revision)
    retained = Path(staging) / "retained-exclusions.sqlite"
    counts = Counter()
    digest = hashlib.sha256(f"source-refresh:{revision}".encode())
    changed = False
    with closing(_read_index(original)) as source, closing(sqlite3.connect(retained)) as destination:
        writer = Exclusions(destination)
        entries = source.execute("SELECT ref, bucket, reason FROM exclusions ORDER BY ref")
        for shard, group in groupby(entries, key=lambda row: shard_of(row[0])):
            previous = old_pin(shard)
            if previous is None or new_pin(shard) != previous:
                changed = True
                continue
            for ref, bucket, reason in group:
                writer.add(ref, bucket, reason)
                counts[reason] += 1
                digest.update(json.dumps([ref, bucket, reason], separators=(",", ":")).encode())
        if not changed:
            return original, revision, dict(counts)
        if not counts:
            return None, None, {}
        destination.commit()
    updated = digest.hexdigest()
    path = exclusion_path(publication, updated)
    link_snapshot(retained, path)
    return path, updated, dict(counts)


def read_exclusion_page(publication, header, reason, limit, offset):
    counts = header.get("exclusion_counts", {})
    if reason is not None:
        counts = {reason: counts[reason]} if reason in counts else {}
    if limit < 0 or offset < 0:
        raise ValueError("Page offset and limit must be nonnegative.")
    if offset >= sum(counts.values()) or not limit:
        return counts, []
    path = exclusion_path(publication, header["exclusion_revision"])
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        where, params = (" WHERE reason=?", (reason,)) if reason is not None else ("", ())
        rows = db.execute(
            "SELECT ref, reason FROM exclusions" + where + " ORDER BY ref LIMIT ? OFFSET ?", (*params, limit, offset)
        )
        return counts, rows.fetchall()


class Exclusions:
    def __init__(self, db):
        self.db = db
        db.execute("CREATE TABLE exclusions(ref TEXT PRIMARY KEY, bucket TEXT NOT NULL, reason TEXT NOT NULL) WITHOUT ROWID")

    def add(self, ref, bucket, reason):
        self.db.execute("INSERT OR IGNORE INTO exclusions VALUES (?,?,?)", (ref, bucket, reason))


class _Survivors:
    def __init__(self, pages, ledger, bucket, removed):
        self.pages, self.ledger, self.bucket = pages, ledger, bucket
        self.count = pages.count(bucket) - removed

    def __len__(self):
        return self.count

    def __iter__(self):
        count = 0
        for ref in self.pages.iter_refs(self.bucket, 0, self.pages.count(self.bucket)):
            if self.ledger.execute("SELECT 1 FROM exclusions WHERE ref=?", (ref,)).fetchone() is None:
                count += 1
                yield ref
        if count != self.count:
            raise ValueError("Filter results contain references outside their published Webshart bucket.")


def _indexed_exclusions(db, path, header, changes):
    db.execute("CREATE TEMP TABLE rejected_ordinals(bucket TEXT, ordinal INTEGER)")
    if header.get("membership_exclusion_revision") is not None:
        db.execute("INSERT INTO rejected_ordinals SELECT bucket,ordinal FROM previous.positions")

    with closing(_read_index(prepared_geometry_path(path, header))) as prepared:
        lookup = geometry_lookup(path, header, prepared)
        for ref, bucket, reason in changes.execute("SELECT * FROM exclusions"):
            entry = lookup.locate_ref(ref)
            already_rejected = db.execute("SELECT 1 FROM exclusions WHERE ref=?", (ref,)).fetchone()
            if entry is None or entry[0] != bucket or already_rejected is not None:
                raise ValueError("Filter results contain references outside their published Webshart bucket.")
            db.execute("INSERT INTO exclusions VALUES (?,?,?)", (ref, bucket, reason))
            db.execute("INSERT INTO rejected_ordinals VALUES (?,?)", entry)

    db.execute(
        "CREATE TABLE positions(bucket TEXT, ordinal INTEGER, position INTEGER, PRIMARY KEY(bucket, ordinal)) WITHOUT ROWID"
    )
    # Each excluded ordinal records the number of surviving rows that precede it.
    db.execute(
        "INSERT INTO positions SELECT bucket,ordinal,"
        "ordinal-row_number() OVER (PARTITION BY bucket ORDER BY ordinal)+1 FROM rejected_ordinals"
    )
    db.execute("CREATE INDEX positions_visible ON positions(bucket, position, ordinal)")


def _publish(path, expected, changes, *, shard_of=None):
    if not changes.execute("SELECT 1 FROM exclusions LIMIT 1").fetchone():
        return
    change_hash = hashlib.sha256()
    for row in changes.execute("SELECT * FROM exclusions ORDER BY ref"):
        change_hash.update(json.dumps(row, separators=(",", ":")).encode())
    revision = hashlib.sha256((expected + change_hash.hexdigest()).encode()).hexdigest()
    with FileLock(str(path) + ".publish.lock"), path.open("rb") as source:
        pages = MembershipPages.open(source)
        if pages.header.get("exclusion_revision") == revision:
            return
        if pages.header["view_revision"] != expected:
            raise ValueError("The accepted Webshart view changed during filtering.")
        counts = dict(changes.execute("SELECT bucket, count(*) FROM exclusions GROUP BY bucket"))
        if not counts.keys() <= pages.counts.keys() or any(n > pages.count(k) for k, n in counts.items()):
            raise ValueError("Filter results exceed their published Webshart bucket.")
        indexed = isinstance(pages, IndexedMembership)
        properties = {
            k: v
            for k, v in pages.header.items()
            if k not in ("view_revision", "webshart_page_table", "bucket_counts", "page_size", "shard_runs_format")
        }
        with TemporaryDirectory(dir=path.parent) as stage:
            new = Path(stage) / "excluded.sqlite"
            with closing(sqlite3.connect(new, uri=True)) as db, db:
                Exclusions(db)
                previous = properties.get("exclusion_revision")
                if previous:
                    previous_uri = exclusion_path(path, previous).resolve().as_uri()
                    db.execute("ATTACH DATABASE ? AS previous", (previous_uri + "?mode=ro&immutable=1",))
                    db.execute("INSERT INTO exclusions SELECT * FROM previous.exclusions")
                if indexed:
                    _indexed_exclusions(db, path, pages.header, changes)
                else:
                    db.executemany("INSERT INTO exclusions VALUES (?,?,?)", changes.execute("SELECT * FROM exclusions"))
                db.execute("CREATE INDEX exclusion_reason ON exclusions(reason)")
                properties["exclusion_counts"] = dict(db.execute("SELECT reason,count(*) FROM exclusions GROUP BY reason"))
            os.replace(new, exclusion_path(path, revision))
            properties["exclusion_revision"] = revision
            if indexed:
                properties.update(
                    membership_exclusion_revision=revision,
                    view_revision=revision,
                    base_bucket_counts=pages.header.get("base_bucket_counts", pages.counts),
                    bucket_counts={key: count - counts.get(key, 0) for key, count in pages.counts.items()},
                )
                publish_indexed_membership(path, properties)
            else:
                buckets = {key: _Survivors(pages, changes, key, counts.get(key, 0)) for key in pages.counts}
                write_membership(path, buckets, properties, shard_of=shard_of)


def _gather(value, distributed):
    if not distributed:
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


@contextmanager
def collect_exclusions(backend, *, publication=None, shard_of=None):
    expected = backend.describe().view_revision
    path = backend.data_backend._cache_path(backend.cache_file) if publication is None else publication
    shard_of = backend.data_backend.shard_of if shard_of is None else shard_of
    distributed = dist.is_available() and dist.is_initialized()
    publish = not distributed or backend.accelerator.is_local_main_process
    with closing(sqlite3.connect("")) as local, closing(sqlite3.connect("")) as merged:
        ledger = Exclusions(local)
        failed = None
        try:
            yield ledger
        except Exception as exc:
            failed = exc
        failures = _gather(type(failed).__name__ if failed else None, distributed)
        if any(failures):
            if failed:
                raise failed
            raise RuntimeError(f"Webshart filter preparation failed on another rank: {failures}")
        rows = local.execute("SELECT * FROM exclusions")
        if publish:
            Exclusions(merged)
        while any(chunks := _gather(rows.fetchmany(128), distributed)):
            if publish:
                for chunk in chunks:
                    merged.executemany("INSERT OR IGNORE INTO exclusions VALUES (?,?,?)", chunk)
        error = None
        if publish:
            try:
                _publish(path, expected, merged, shard_of=shard_of)
            except Exception as exc:
                error = exc
        if any(_gather(type(error).__name__ if error else None, distributed)):
            if error:
                raise error
            raise RuntimeError("A peer could not publish the Webshart filter results.")
        try:
            backend.reload_cache(set_config=False)
            revision = backend.describe().view_revision
        except Exception as exc:
            error = exc
            revision = None
        revisions = _gather(revision, distributed)
        if error:
            raise error
        if None in revisions or len(set(revisions)) != 1:
            raise ValueError("Webshart workers opened different accepted revisions after filtering.")
