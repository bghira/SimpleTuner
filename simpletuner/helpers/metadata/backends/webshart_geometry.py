"""Point reads of locally cached Webshart geometry for standalone preparation."""

import hashlib
import json
import os
import sqlite3
import zlib
from collections import defaultdict
from contextlib import ExitStack, closing, contextmanager
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory

import ijson
from filelock import FileLock


class GeometryLookup:
    def __init__(self, connection):
        self.connection = connection

    def shard_revision(self, index):
        row = self.connection.execute("SELECT revision FROM shards WHERE position=?", (index,)).fetchone()
        return row[0] if row is not None else None

    def get_metadata_by_filepath(self, filepath):
        row = self.connection.execute("SELECT record FROM geometry WHERE ref = ?", (str(filepath),)).fetchone()
        return self._decode_record(row[0]) if row is not None else None

    def get_metadata_batch(self, refs):
        return [self._decode_record(record) if record is not None else None for record in self._lookup_batch(refs, "record")]

    def bucket_batch(self, refs):
        return self._lookup_batch(refs, "bucket")

    def _lookup_batch(self, refs, column):
        keys = [str(ref) for ref in refs]
        records = {}
        limit = self.connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
        for chunk in batched(dict.fromkeys(keys), limit):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(f"SELECT ref, {column} FROM geometry WHERE ref IN ({placeholders})", chunk)
            records.update(rows)
        return [records.get(ref) for ref in keys]

    @staticmethod
    def _decode_record(record):
        return None if record is None else json.loads(zlib.decompress(record) if isinstance(record, bytes) else record)

    def locate_ref(self, ref):
        return self.connection.execute("SELECT bucket,ordinal FROM geometry WHERE ref=?", (ref,)).fetchone()

    def iter_refs(self, bucket, start, stop):
        rows = self.connection.execute(
            "SELECT ref FROM geometry WHERE bucket=? AND ordinal>=? AND ordinal<? ORDER BY ordinal",
            (bucket, start, stop),
        )
        for (ref,) in rows:
            yield ref

    def read_positions(self, bucket, positions):
        refs = []
        for chunk in batched(positions, self.connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER) - 1):
            requested = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT ordinal,ref FROM geometry WHERE bucket=? AND ordinal IN ({requested})",
                (bucket, *chunk),
            )
            by_position = dict(rows)
            refs.extend(by_position[position] for position in chunk)
        return refs


class PartitionedGeometryLookup(GeometryLookup):
    def __init__(self, connection, metadata_file, shards_per_partition):
        super().__init__(connection)
        self.metadata_file = metadata_file
        self.shards_per_partition = shards_per_partition

    def _partition(self, ref):
        from simpletuner.helpers.data_backend.webshart import WebshartDataBackend

        if not WebshartDataBackend.is_sample_id(ref):
            return None
        return WebshartDataBackend.parse_sample_id(ref).shard_idx // self.shards_per_partition

    def _digest(self, partition):
        row = self.connection.execute("SELECT digest FROM partitions WHERE id=?", (partition,)).fetchone()
        return row[0] if row is not None else None

    def get_metadata_by_filepath(self, filepath):
        return self.get_metadata_batch((filepath,))[0]

    def _lookup_batch(self, refs, column):
        groups = defaultdict(list)
        result = []
        for ref in refs:
            groups[self._partition(ref)].append((len(result), ref))
            result.append(None)
        for partition, group in groups.items():
            digest = self._digest(partition)
            if digest is not None:
                with closing(read_snapshot(self.metadata_file, digest)) as db:
                    records = GeometryLookup(db)._lookup_batch((ref for _, ref in group), column)
                for (position, _), record in zip(group, records, strict=True):
                    result[position] = record
        return result

    def locate_ref(self, ref):
        partition = self._partition(ref)
        digest = self._digest(partition)
        if digest is None:
            return None
        with closing(read_snapshot(self.metadata_file, digest)) as db:
            entry = GeometryLookup(db).locate_ref(ref)
        if entry is None:
            return None
        bucket, ordinal = entry
        (start,) = self.connection.execute(
            "SELECT start FROM bucket_partitions WHERE bucket=? AND partition=?", (bucket, partition)
        ).fetchone()
        return bucket, start + ordinal

    def iter_refs(self, bucket, start, stop):
        first = self.connection.execute(
            "SELECT start FROM bucket_partitions WHERE bucket=? AND start<=? ORDER BY start DESC LIMIT 1",
            (bucket, start),
        ).fetchone()
        if first is None:
            return
        for origin, size, digest in self.connection.execute(
            "SELECT bp.start,bp.n,p.digest FROM bucket_partitions bp JOIN partitions p ON p.id=bp.partition "
            "WHERE bp.bucket=? AND bp.start>=? AND bp.start<? ORDER BY bp.start",
            (bucket, first[0], stop),
        ):
            with closing(read_snapshot(self.metadata_file, digest)) as db:
                yield from GeometryLookup(db).iter_refs(bucket, max(0, start - origin), min(size, stop - origin))

    def read_positions(self, bucket, positions):
        groups = defaultdict(list)
        result = []
        for ordinal in positions:
            start, digest = self.connection.execute(
                "SELECT bp.start,p.digest FROM bucket_partitions bp JOIN partitions p ON p.id=bp.partition "
                "WHERE bp.bucket=? AND bp.start<=? ORDER BY bp.start DESC LIMIT 1",
                (bucket, ordinal),
            ).fetchone()
            groups[digest].append((len(result), ordinal - start))
            result.append(None)
        for digest, group in groups.items():
            with closing(read_snapshot(self.metadata_file, digest)) as db:
                refs = GeometryLookup(db).read_positions(bucket, (ordinal for _, ordinal in group))
            for (position, _), ref in zip(group, refs, strict=True):
                result[position] = ref
        return result


def _read_index(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True, check_same_thread=False)


def _build_index(source, target, stamp):
    # The source descriptor is pinned while the index is built and atomically published.
    digest = hashlib.file_digest(source, "sha256").hexdigest()
    source.seek(0)
    with TemporaryDirectory(dir=target.parent) as temporary:
        staging = Path(temporary) / "index.sqlite"
        with closing(sqlite3.connect(staging)) as db, db:
            db.execute("CREATE TABLE geometry(ref TEXT PRIMARY KEY, record TEXT NOT NULL) WITHOUT ROWID")
            db.execute("CREATE TABLE source(stamp TEXT, digest TEXT)")
            db.execute("INSERT INTO source VALUES (?, ?)", (stamp, digest))
            for ref, record in ijson.kvitems(source, ""):
                record.pop("captions", None)
                db.execute("INSERT INTO geometry VALUES (?, ?)", (ref, json.dumps(record, default=float)))
        os.replace(staging, target)


@contextmanager
def open_geometry(source, expected_digest):
    path = Path(source.name)
    target = path.with_suffix(".geometry-v1.sqlite")
    stat = os.fstat(source.fileno())
    stamp = json.dumps([stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
    with FileLock(str(target) + ".lock"):
        current = False
        if target.exists():
            with closing(_read_index(target)) as db:
                current = db.execute("SELECT stamp FROM source").fetchone() == (stamp,)
        if not current:
            _build_index(source, target, stamp)
        reader = _read_index(target)
        # Open the file and pin its inode before another builder can replace it.
        try:
            (digest,) = reader.execute("SELECT digest FROM source").fetchone()
            if expected_digest is not None and digest != expected_digest:
                raise ValueError("Published Webshart membership and geometry revisions differ. Run a new scan.")
        except BaseException:
            reader.close()
            raise
    with closing(reader):
        yield GeometryLookup(reader)


class PreparedGeometry:
    def __init__(self, db):
        self.db = db
        db.execute(
            "CREATE TABLE geometry(ref TEXT NOT NULL UNIQUE, record BLOB, bucket TEXT NOT NULL, ordinal INTEGER NOT NULL)"
        )
        db.execute(
            "CREATE TABLE runs(shard INTEGER NOT NULL, bucket TEXT NOT NULL, start INTEGER NOT NULL, n INTEGER NOT NULL)"
        )
        db.execute("CREATE TABLE shards(position INTEGER PRIMARY KEY, revision TEXT NOT NULL)")
        self.counts = {}
        self.pending_runs = {}
        self.digest = hashlib.sha256(b"webshart-geometry-ordinals-v2")

    def pin_shard(self, index, revision):
        self.db.execute("INSERT INTO shards VALUES (?, ?)", (index, revision))
        self.digest.update(json.dumps(["shard", index, revision], separators=(",", ":")).encode())

    def add(self, bucket, ref, metadata, *, shard=None):
        ordinal = self.counts.get(bucket, 0)
        record = json.dumps(metadata, separators=(",", ":")) if metadata is not None else None
        packed = zlib.compress(record.encode(), 1) if record is not None else None
        self.db.execute("INSERT INTO geometry VALUES (?,?,?,?)", (ref, packed, bucket, ordinal))
        self.digest.update(json.dumps([bucket, ref, record, shard], separators=(",", ":")).encode())
        self.counts[bucket] = ordinal + 1
        if shard is not None:
            pending = self.pending_runs.get(bucket)
            if pending is None or pending[0] != shard:
                self._flush_run(bucket, pending, ordinal)
                self.pending_runs[bucket] = (shard, ordinal)

    def _flush_run(self, bucket, pending, stop):
        if pending is not None:
            shard, start = pending
            self.db.execute("INSERT INTO runs VALUES (?,?,?,?)", (shard, bucket, start, stop - start))

    def finish(self):
        for bucket, pending in self.pending_runs.items():
            self._flush_run(bucket, pending, self.counts[bucket])
        self.pending_runs.clear()
        self.db.execute("CREATE UNIQUE INDEX geometry_bucket_position ON geometry(bucket, ordinal)")
        self.db.execute("CREATE TABLE geometry_publication(digest TEXT NOT NULL)")
        self.db.execute("INSERT INTO geometry_publication VALUES (?)", (self.digest.hexdigest(),))
        self.db.execute("CREATE TABLE geometry_bucket_counts(bucket TEXT NOT NULL, n INTEGER NOT NULL)")
        self.db.executemany("INSERT INTO geometry_bucket_counts VALUES (?,?)", self.counts.items())
        return self.counts.copy()


class PreparedGeometryPartitions:
    def __init__(self, db, staging, shards_per_partition):
        self.db = db
        self.staging = staging
        self.shards_per_partition = shards_per_partition
        self.counts = {}
        self.digest = hashlib.sha256(b"webshart-geometry-partitions-v1")
        db.execute("CREATE TABLE partitions(id INTEGER PRIMARY KEY, digest TEXT NOT NULL)")
        db.execute("CREATE TEMP TABLE partition_staging(id INTEGER PRIMARY KEY, path TEXT NOT NULL)")
        db.execute(
            "CREATE TABLE bucket_partitions(bucket TEXT NOT NULL, start INTEGER NOT NULL, n INTEGER NOT NULL, "
            "partition INTEGER NOT NULL, PRIMARY KEY(bucket,start), UNIQUE(bucket,partition)) WITHOUT ROWID"
        )
        db.execute(
            "CREATE TABLE runs(shard INTEGER NOT NULL, bucket TEXT NOT NULL, start INTEGER NOT NULL, n INTEGER NOT NULL)"
        )
        db.execute("CREATE TABLE shards(position INTEGER PRIMARY KEY, revision TEXT NOT NULL)")

    @contextmanager
    def stage_partition(self, partition):
        path = self.staging / f"{partition}.sqlite"
        with closing(sqlite3.connect(path)) as db:
            writer = PreparedGeometry(db)
            yield writer, path
            writer.finish()
            db.commit()

    def adopt_partition(self, partition, path):
        """Add a completed group in canonical source order."""
        (previous,) = self.db.execute("SELECT max(id) FROM partitions").fetchone()
        if previous is not None and partition <= previous:
            raise ValueError("Webshart index must adopt source partitions in order.")

        with closing(_read_index(path)) as prepared:
            (digest,) = prepared.execute("SELECT digest FROM geometry_publication").fetchone()
            bounds = (partition * self.shards_per_partition, (partition + 1) * self.shards_per_partition)
            if (
                prepared.execute("SELECT 1 FROM runs WHERE shard<? OR shard>=? LIMIT 1", bounds).fetchone()
                or prepared.execute("SELECT 1 FROM shards WHERE position<? OR position>=? LIMIT 1", bounds).fetchone()
            ):
                raise ValueError("Prepared Webshart group contains shards outside its partition.")

            self.db.execute("INSERT INTO partitions VALUES (?,?)", (partition, digest))
            self.db.execute("INSERT INTO partition_staging VALUES (?,?)", (partition, os.fspath(path)))
            self.digest.update(json.dumps([partition, digest], separators=(",", ":")).encode())
            self.db.executemany("INSERT INTO shards VALUES (?,?)", prepared.execute("SELECT position,revision FROM shards"))
            self.db.executemany(
                "INSERT INTO runs VALUES (?,?,?,?)",
                (
                    (shard, bucket, self.counts.get(bucket, 0) + start, size)
                    for shard, bucket, start, size in prepared.execute("SELECT shard,bucket,start,n FROM runs")
                ),
            )
            for bucket, count in prepared.execute("SELECT bucket,n FROM geometry_bucket_counts ORDER BY rowid"):
                start = self.counts.get(bucket, 0)
                self.db.execute("INSERT INTO bucket_partitions VALUES (?,?,?,?)", (bucket, start, count, partition))
                self.counts[bucket] = start + count

    def finish(self):
        return self.counts.copy()

    def publish(self, metadata_file):
        for digest, staged in self.db.execute("SELECT digest,path FROM partitions JOIN partition_staging USING(id)"):
            link_snapshot(staged, snapshot_path(metadata_file, digest))


def link_snapshot(source, destination):
    try:
        os.link(source, destination)
    except FileExistsError:
        if not destination.is_file():
            raise


def snapshot_path(metadata_file, digest):
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid Webshart geometry revision.")
    return Path(metadata_file).with_suffix(f".{digest}.sqlite")


@contextmanager
def stage_geometry(metadata_file, *, shards_per_partition=None):
    root = Path(metadata_file).parent
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as temporary:
        path = Path(temporary) / "geometry.sqlite"
        with closing(sqlite3.connect(path)) as db:
            if shards_per_partition is None:
                writer = PreparedGeometry(db)
            else:
                writer = PreparedGeometryPartitions(db, path.parent, shards_per_partition)
            yield writer, path


def read_snapshot(metadata_file, digest):
    return _read_index(snapshot_path(metadata_file, digest))


def prepared_geometry_path(publication, header):
    name = header["geometry_name"]
    if Path(name).name != name:
        raise ValueError("Invalid Webshart geometry filename.")
    return snapshot_path(Path(publication).parent / name, header["geometry_sha256"])


def geometry_lookup(publication, header, db):
    shards_per_partition = header.get("geometry_partition_shards")
    if shards_per_partition is None:
        return GeometryLookup(db)
    metadata_file = Path(publication).parent / header["geometry_name"]
    return PartitionedGeometryLookup(db, metadata_file, shards_per_partition)


class AcceptedGeometry:
    def __init__(self, lookup, counts, excluded):
        self.lookup = lookup
        self.counts = counts
        self.excluded = excluded

    def __call__(self, ref):
        location = self.lookup.locate_ref(ref)
        if location is None or not self.counts.get(location[0]):
            return False
        return (
            self.excluded is None or self.excluded.execute("SELECT 1 FROM exclusions WHERE ref=?", (ref,)).fetchone() is None
        )

    def batch(self, refs):
        buckets = self.lookup.bucket_batch(refs)
        rejected = set()
        if self.excluded is not None:
            limit = self.excluded.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
            for chunk in batched(dict.fromkeys(refs), limit):
                placeholders = ",".join("?" for _ in chunk)
                rows = self.excluded.execute(f"SELECT ref FROM exclusions WHERE ref IN ({placeholders})", chunk)
                rejected.update(ref for (ref,) in rows)
        return [bool(self.counts.get(bucket)) and ref not in rejected for ref, bucket in zip(refs, buckets, strict=True)]


def accepted_geometry_predicate(lookup, publication, header, stack):
    from .webshart_exclusions import exclusion_path

    revision = header.get("exclusion_revision")
    excluded = stack.enter_context(closing(_read_index(exclusion_path(publication, revision)))) if revision else None
    return AcceptedGeometry(lookup, header["bucket_counts"], excluded)


@contextmanager
def accepted_ref_lookup(publication, pages):
    """Point membership in a pinned native geometry, including bucket pruning and durable exclusions."""
    with ExitStack() as stack:
        geometry = stack.enter_context(closing(_read_index(prepared_geometry_path(publication, pages.header))))
        lookup = geometry_lookup(publication, pages.header, geometry)
        yield accepted_geometry_predicate(lookup, publication, pages.header, stack)
