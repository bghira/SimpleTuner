"""Reconstructible Webshart training order over an immutable membership publication."""

import hashlib
import json
import os
import sqlite3
from bisect import bisect_right
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock

from .webshart_geometry import _read_index
from .webshart_membership import IndexedMembership, MembershipPages, shard_runs_path


@dataclass(frozen=True)
class EpochSpec:
    dataset_id: str
    policy: str
    seed: int
    epoch: int
    batch_size: int
    dp_size: int
    dp_rank: int
    accumulation: int
    repeats: int
    pad: bool
    topology: tuple
    occurrence_revision: str | None = None
    sampling_order: str = "shuffled"
    shuffle_seed: int | None = None

    def __post_init__(self):
        if self.policy not in {"global", "shard_local"}:
            raise ValueError("Webshart sampling must be global or shard_local.")
        allowed_orders = ("shuffled", "sequential") if self.policy == "shard_local" else ("shuffled",)
        if self.sampling_order not in allowed_orders:
            raise ValueError("Webshart sampling order must be shuffled, or sequential with shard-local sampling.")
        if (
            min(self.batch_size, self.dp_size, self.accumulation) < 1
            or self.repeats < 0
            or not 0 <= self.dp_rank < self.dp_size
        ):
            raise ValueError("Invalid Webshart epoch layout.")

    @property
    def rank_shuffle_seed(self):
        return (self.shuffle_seed + (self.epoch - 1) * self.dp_size + self.dp_rank) % (1 << 64)

    def rank_positions(self, count, batch, key):
        owned = len(range(self.dp_rank, count, self.dp_size))
        for position in range(batch * self.batch_size, (batch + 1) * self.batch_size):
            offset = position % owned
            if self.sampling_order == "shuffled":
                offset = _permute(offset, owned, key)
            yield self.dp_rank + self.dp_size * offset

    def batches_for(self, count):
        if self.shuffle_seed is not None:
            if count < self.dp_size:
                return 0
            # All ranks take the same step count. Repeats and padding stay within each rank's rows.
            owned = (count + (self.dp_size - 1 if self.pad else 0)) // self.dp_size
            count = owned * self.dp_size
        logical = count * (self.repeats + 1)
        unit = self.batch_size * self.dp_size * self.accumulation
        return ((logical + (unit - 1 if self.pad else 0)) // unit) * self.accumulation

    def checkpoint_layout(self):
        layout = {**asdict(self), "topology": list(self.topology)}
        if self.shuffle_seed is None:
            layout.pop("shuffle_seed")
        if self.sampling_order == "shuffled":
            layout.pop("sampling_order")
        return layout


def _key(*parts):
    return hashlib.blake2s(json.dumps(parts, separators=(",", ":")).encode(), digest_size=16).digest()


def _permute(position, count, key):
    """Cycle-walk a balanced Feistel permutation onto [0, count)."""
    if count <= 1:
        return position
    bits = ((count - 1).bit_length() + 1) // 2
    mask = (1 << bits) - 1
    while True:
        left, right = position >> bits, position & mask
        for round_id in range(6):
            value = hashlib.blake2s(right.to_bytes(8, "little") + bytes([round_id]), key=key, digest_size=8).digest()
            left, right = right, left ^ (int.from_bytes(value, "little") & mask)
        position = left << bits | right
        if position < count:
            return position


def _shard_runs(pages, shard_of):
    publication = Path(pages.source.name)
    revision = pages.header["view_revision"]
    runs_format = pages.header.get("shard_runs_format")
    if runs_format == "sqlite-v1":
        path = shard_runs_path(publication, revision)
        if not path.is_file():
            raise FileNotFoundError("Published Webshart shard ranges are missing. Run a new scan.")
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True, check_same_thread=False)
        try:
            if db.execute("SELECT revision FROM publication").fetchone() != (revision,):
                raise ValueError("Published Webshart shard ranges do not match membership.")
        except BaseException:
            db.close()
            raise
        return db
    if runs_format is not None:
        raise ValueError(f"Unsupported Webshart shard range format: {runs_format}")

    # Existing indexed publications can have a checkpoint and no published shard ranges.
    path = publication.with_suffix(".shard-runs.sqlite")
    with FileLock(str(path) + ".lock"):
        if path.exists():
            db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True, check_same_thread=False)
            if db.execute("SELECT revision FROM publication").fetchone() == (revision,):
                return db
            db.close()
        with TemporaryDirectory(dir=path.parent) as staging:
            temp = Path(staging) / "runs.sqlite"
            with closing(sqlite3.connect(temp)) as db, db:
                db.execute("CREATE TABLE publication(revision TEXT)")
                db.execute("INSERT INTO publication VALUES (?)", (revision,))
                db.execute("CREATE TABLE runs(shard INTEGER, bucket INTEGER, start INTEGER, n INTEGER)")
                for bucket, (key, count) in enumerate(pages.counts.items()):
                    start, previous = 0, None
                    for index, ref in enumerate(pages.iter_refs(key, 0, count)):
                        shard = shard_of(ref)
                        if previous is not None and shard != previous:
                            db.execute("INSERT INTO runs VALUES (?,?,?,?)", (previous, bucket, start, index - start))
                            start = index
                        previous = shard
                    if previous is not None:
                        db.execute("INSERT INTO runs VALUES (?,?,?,?)", (previous, bucket, start, count - start))
            os.replace(temp, path)
        return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True, check_same_thread=False)


def _ordered_runs(pages, db, shard_order):
    db.create_function("shard_order", 1, shard_order, deterministic=True)
    if isinstance(pages, IndexedMembership):
        return pages.shard_runs(db)
    return db.execute("SELECT shard,bucket,start,n FROM runs ORDER BY shard_order(shard),shard,bucket,start")


def seeded_published_batches(pages, spec, shard_of):
    runs = pages.open_geometry() if isinstance(pages, IndexedMembership) else _shard_runs(pages, shard_of)
    with closing(runs):
        if not pages.header.get("membership_exclusion_revision"):
            return sum(
                spec.batches_for(n) * frequency for n, frequency in runs.execute("SELECT n,count(*) FROM runs GROUP BY n")
            )
        return sum(spec.batches_for(count) for _, _, _, count in _ordered_runs(pages, runs, int))


class WebshartEpoch:
    version = 1

    def __init__(self, source, spec, shard_of, *, shard_at=None, shard_count=0, prefill=None):
        self.prefill = prefill
        self.prefill_position = None
        self.source = source
        self.spec = spec
        self.db = None
        self.index = None
        try:
            self.pages = MembershipPages.open(source)
            if self.pages is None or not self.pages.header.get("view_revision"):
                raise ValueError("Scan this Webshart dataset to publish indexed membership for native sampling.")
            self.keys = list(self.pages.counts)
            self.counts = list(self.pages.counts.values())
            self.position = 0
            self.global_batch = spec.batch_size * spec.dp_size
            self.order_key = _key(spec.dataset_id, spec.seed, spec.epoch)
            self.bucket_keys = [_key(spec.dataset_id, spec.seed, spec.epoch, key) for key in self.keys]
            self.ends = []
            self.batches = 0
            for count in self.counts:
                self.batches += spec.batches_for(count)
                self.ends.append(self.batches)
            if isinstance(self.pages, IndexedMembership):
                self.index = self.pages.open_geometry()
            if spec.policy == "shard_local":
                self.db = self.index if self.index is not None else _shard_runs(self.pages, shard_of)
                if shard_at is not None:
                    self._order_seeded_shards(shard_at, shard_count)
                elif self.index is None:
                    self._order_shards(self.db, self.db, temporary=True)
                else:
                    self.db = self._open_shared_schedule()
        except BaseException:
            self.close()
            raise

    def _order_seeded_shards(self, shard_at, shard_count):
        positions = {shard_at(i): i for i in range(shard_count)}
        self.db.execute(
            "CREATE TEMP TABLE shard_batches(step INTEGER PRIMARY KEY, slot INTEGER, bucket INTEGER, start INTEGER, n INTEGER)"
        )
        self.batches = 0
        for shard, bucket, start, n in _ordered_runs(self.pages, self.db, positions.__getitem__):
            batches = self.spec.batches_for(n)
            if batches:
                self.db.execute(
                    "INSERT INTO shard_batches VALUES (?,?,?,?,?)",
                    (self.batches, positions[shard], bucket, start, n),
                )
                self.batches += batches

    def _open_shared_schedule(self):
        layout_parts = (
            self.version,
            self.spec.dataset_id,
            self.spec.seed,
            self.global_batch,
            self.spec.accumulation,
            self.spec.repeats,
            self.spec.pad,
        )
        if self.spec.sampling_order == "sequential":
            layout_parts += ("sequential",)
        layout = _key(*layout_parts).hex()
        publication = Path(self.source.name)
        path = publication.with_suffix(f".shard-local-{layout}.sqlite")
        expected = (self.pages.header["view_revision"], self.spec.epoch)
        current = self._open_matching_schedule(path, expected)
        if current is not None:
            return current
        with FileLock(str(path) + ".lock"):
            current = self._open_matching_schedule(path, expected)
            if current is not None:
                return current
            with TemporaryDirectory(dir=path.parent) as temporary:
                staging = Path(temporary) / "schedule.sqlite"
                with closing(sqlite3.connect(staging)) as db:
                    db.execute("CREATE TABLE schedule_revision(view TEXT, epoch INTEGER)")
                    db.execute("INSERT INTO schedule_revision VALUES (?,?)", expected)
                    self._order_shards(self.index, db)
                os.replace(staging, path)
            return _read_index(path)

    @staticmethod
    def _open_matching_schedule(path, expected):
        if not path.is_file():
            return None
        db = _read_index(path)
        try:
            actual = db.execute("SELECT view,epoch FROM schedule_revision").fetchone()
        except BaseException:
            db.close()
            raise
        if actual == expected:
            return db
        db.close()
        return None

    def _order_shards(self, runs_db, schedule_db, *, temporary=False):
        runs_db.execute("PRAGMA temp_store=FILE")
        shard_order = (
            int
            if self.spec.sampling_order == "sequential"
            else (lambda shard: _key(self.spec.seed, self.spec.epoch, self.spec.dataset_id, shard))
        )
        table = "TEMP TABLE" if temporary else "TABLE"
        schedule_db.execute(
            f"CREATE {table} segments(bucket INTEGER, logical INTEGER, source INTEGER, n INTEGER, key BLOB, "
            "PRIMARY KEY(bucket, logical)) WITHOUT ROWID"
        )
        schedule_db.execute(f"CREATE {table} batches(step INTEGER PRIMARY KEY, bucket INTEGER, first_batch INTEGER)")
        consumed = [0] * len(self.keys)
        scheduled = [0] * len(self.keys)
        step = 0
        limits = [end - start for start, end in zip([0] + self.ends, self.ends)]
        for shard, bucket, start, count in _ordered_runs(self.pages, runs_db, shard_order):
            schedule_db.execute(
                "INSERT INTO segments VALUES (?,?,?,?,?)",
                (bucket, consumed[bucket], start, count, _key(self.spec.seed, self.spec.epoch, shard, bucket, start)),
            )
            consumed[bucket] += count * (1 + self.spec.repeats)
            end = min(limits[bucket], consumed[bucket] // self.global_batch)
            if end > scheduled[bucket]:
                schedule_db.execute("INSERT INTO batches VALUES (?,?,?)", (step, bucket, scheduled[bucket]))
                step += end - scheduled[bucket]
                scheduled[bucket] = end
        for bucket, limit in enumerate(limits):
            if limit > scheduled[bucket]:
                schedule_db.execute("INSERT INTO batches VALUES (?,?,?)", (step, bucket, scheduled[bucket]))
                step += limit - scheduled[bucket]
        schedule_db.commit()

    def state_dict(self):
        return {
            "version": self.version,
            "view": self.pages.header["view_revision"],
            "source": self.pages.header.get("source_revision"),
            "layout": self.spec.checkpoint_layout(),
            "position": self.position,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        if any(state.get(key) != expected[key] for key in ("version", "view", "source", "layout")):
            raise ValueError("Webshart resume requires unchanged source, accepted view, sampling policy and layout.")
        position = state.get("position")
        if type(position) is not int or not 0 <= position <= self.batches:
            raise ValueError("Invalid Webshart checkpoint position.")
        self.position = position

    def next_batch(self):
        if self.position == self.batches:
            raise StopIteration
        if self.spec.shuffle_seed is not None:
            step, slot, bucket, start, n = self.db.execute(
                "SELECT step,slot,bucket,start,n FROM shard_batches WHERE step<=? ORDER BY step DESC LIMIT 1",
                (self.position,),
            ).fetchone()
            if self.prefill is not None and slot != self.prefill_position:
                self.prefill.advance(slot)
                self.prefill_position = slot
            key = _key(self.order_key.hex(), bucket, start)
            positions = [start + i for i in self.spec.rank_positions(n, self.position - step, key)]
            return self._read_batch(bucket, positions)
        if self.db is None:
            step = _permute(self.position, self.batches, self.order_key)
            bucket = bisect_right(self.ends, step)
            batch = step - (self.ends[bucket - 1] if bucket else 0)
        else:
            first_step, bucket, first_batch = self.db.execute(
                "SELECT step,bucket,first_batch FROM batches WHERE step<=? ORDER BY step DESC LIMIT 1", (self.position,)
            ).fetchone()
            batch = first_batch + self.position - first_step
        start = batch * self.global_batch + self.spec.dp_rank * self.spec.batch_size
        count = self.counts[bucket]
        positions = []
        for occurrence in range(start, start + self.spec.batch_size):
            logical = occurrence % (count * (self.spec.repeats + 1))
            if self.db is None:
                ordinal = _permute(logical, count * (1 + self.spec.repeats), self.bucket_keys[bucket]) % count
            else:
                origin, member, length, key = self.db.execute(
                    "SELECT logical,source,n,key FROM segments WHERE bucket=? AND logical<=? ORDER BY logical DESC LIMIT 1",
                    (bucket, logical),
                ).fetchone()
                offset = (logical - origin) % length
                if self.spec.sampling_order == "shuffled":
                    offset = _permute(offset, length, key)
                ordinal = member + offset
            positions.append(ordinal)
        return self._read_batch(bucket, positions)

    def _read_batch(self, bucket, positions):
        if self.index is None:
            refs = self.pages.read_positions(self.keys[bucket], positions)
        else:
            refs = self.pages.read_positions(self.keys[bucket], positions, db=self.index)
        self.position += 1
        return self.keys[bucket], refs

    def close(self):
        if self.prefill is not None:
            self.prefill.close()
        if self.db is not None and self.db is not self.index:
            self.db.close()
        if self.index is not None:
            self.index.close()
        self.source.close()


class WindowedWebshartEpoch:
    """Schedule one bounded group of adjacent Parquet shards at a time."""

    version = "shard-window-v1"

    def __init__(
        self, spec, source, view, partition_count, open_partition, *, window_shards=None, prefill=None, shard_at=None
    ):
        self.spec = spec
        self.source = source
        self.view = view
        self.partition_count = partition_count
        self.prefill = prefill
        self.shard_at = shard_at
        self.window_shards = window_shards
        self.open_partition = open_partition
        self.order_key = _key(spec.seed, spec.epoch, spec.dataset_id, self.version)
        self.global_batch = spec.batch_size * spec.dp_size
        self.position = 0
        self.window = 0
        self.batch_in_window = 0
        self.partition = None
        self.revision = None
        self.frontier = (0, 0, None)

    def _open_window(self):
        self.partition = self.window
        if self.shard_at is not None:
            self.partition = self.shard_at(self.window)
        elif self.spec.sampling_order == "shuffled":
            self.partition = _permute(self.window, self.partition_count, self.order_key)
        if self.prefill is not None:
            self.prefill.advance(self.window * self.window_shards)
        self.revision, buckets = self.open_partition(self.partition)
        self.buckets = sorted(buckets.items())
        self.ends = []
        total = 0
        for _, refs in self.buckets:
            total += self.spec.batches_for(len(refs))
            self.ends.append(total)

    def next_batch(self):
        while self.window < self.partition_count:
            if self.partition is None:
                self._open_window()
            if self.ends and self.batch_in_window < self.ends[-1]:
                break
            self.window += 1
            self.batch_in_window = 0
            self.partition = None
        else:
            if self.position == 0:
                raise ValueError(
                    f"Webshart dataset {self.spec.dataset_id!r} has no complete training batches in its shard windows. "
                    "Reduce the global batch size or prepare an indexed view."
                )
            raise StopIteration

        bucket_index = bisect_right(self.ends, self.batch_in_window)
        bucket, refs = self.buckets[bucket_index]
        batch = self.batch_in_window - (self.ends[bucket_index - 1] if bucket_index else 0)
        start = batch * self.global_batch + self.spec.dp_rank * self.spec.batch_size
        count = len(refs)
        logical_count = count * (self.spec.repeats + 1)
        occurrences = range(start, start + self.spec.batch_size)
        if self.spec.shuffle_seed is not None:
            key = _key(self.order_key.hex(), self.partition, bucket)
            result = [refs[i] for i in self.spec.rank_positions(count, batch, key)]
        elif self.spec.sampling_order == "sequential":
            result = [refs[occurrence % count] for occurrence in occurrences]
        else:
            key = _key(self.order_key.hex(), self.partition, bucket)
            result = [refs[_permute(occurrence % logical_count, logical_count, key) % count] for occurrence in occurrences]
        self.position += 1
        self.batch_in_window += 1
        self.frontier = (self.window, self.batch_in_window, self.revision)
        return bucket, result

    def state_dict(self):
        window, batch, revision = self.frontier
        return {
            "version": self.version,
            "view": self.view,
            "source": self.source,
            "layout": self.spec.checkpoint_layout(),
            "position": self.position,
            "window": window,
            "batch_in_window": batch,
            "window_revision": revision,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        if any(state.get(key) != expected[key] for key in ("version", "view", "source", "layout")):
            raise ValueError("Webshart resume requires unchanged source, accepted view, sampling policy and layout.")
        position, window, batch = (state.get(key) for key in ("position", "window", "batch_in_window"))
        if (
            any(not isinstance(value, int) or isinstance(value, bool) for value in (position, window, batch))
            or not 0 <= window < self.partition_count
            or not 0 <= batch <= position
            or (position == 0 and (window or state.get("window_revision") is not None))
            or (position > 0 and (batch == 0 or (position - batch) % self.spec.accumulation))
        ):
            raise ValueError("Invalid Webshart window position.")
        self.window = window
        self.partition = self.revision = None
        if position:
            self._open_window()
            if self.revision != state.get("window_revision"):
                raise ValueError("Webshart shard contents changed since the training checkpoint.")
            if not self.ends or batch > self.ends[-1]:
                raise ValueError("Invalid Webshart position within the prepared shard window.")
        self.position, self.batch_in_window = position, batch
        self.frontier = (window, batch, self.revision)

    def close(self):
        if self.prefill is not None:
            self.prefill.close()
        self.buckets = ()
