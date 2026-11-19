"""The geometry decided for a sample, how it is encoded, the recipe of settings it follows from, and where it is kept."""

import json
import queue
import sqlite3
import threading
from contextlib import closing
from hashlib import sha256
from pathlib import Path

from simpletuner.helpers.data_backend.dataset_runtime import RUNTIME_MUTABLE_KEYS, recrops_each_epoch
from simpletuner.helpers.training.state_tracker import StateTracker

DECIDED = ("aspect_ratio", "target_size", "intermediary_size", "crop_coordinates")
_PAIRS = DECIDED[1:]


def encode_transform(record: dict) -> bytes:
    return json.dumps([record[key] for key in DECIDED], separators=(",", ":")).encode("utf-8")


def decode_transform(value: bytes) -> dict:
    aspect_ratio, *pairs = json.loads(value)
    return {"aspect_ratio": aspect_ratio, **{key: None if pair is None else tuple(pair) for key, pair in zip(_PAIRS, pairs)}}


def settings_digest(metadata_backend, *, fingerprint=None, units=None) -> str:
    """What a dataset's decisions follow from: its settings, the model's canvas and the source they were made over."""
    args, model = StateTracker.get_args(), StateTracker.get_model()
    config = StateTracker.get_data_backend_config(metadata_backend.id)
    decided_by = {
        "config": {key: value for key, value in config.items() if key not in RUNTIME_MUTABLE_KEYS},
        "resolution": [metadata_backend.resolution, metadata_backend.resolution_type],
        "limits": [
            metadata_backend.minimum_image_size,
            metadata_backend.minimum_aspect_ratio,
            metadata_backend.maximum_aspect_ratio,
        ],
        "alignment": getattr(args, "aspect_bucket_alignment", None),
        "rounding": getattr(args, "aspect_bucket_rounding", None),
        "canvas": getattr(model, "MAXIMUM_CANVAS_SIZE", None),
        "shards": units,
        "fingerprint": fingerprint,
    }
    return sha256(json.dumps(decided_by, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def transform_recipe(digest: str, epoch: int, dataset_id: str) -> str:
    """A dataset that re-crops each epoch decides again in each one, so the epoch is part of its recipe."""
    if recrops_each_epoch(StateTracker.get_data_backend_config(dataset_id), StateTracker.get_args()):
        return f"{digest}-epoch-{int(epoch)}"
    return digest


def record_bytes(row: int | str, value: bytes) -> int:
    """What a record counts against a batch's max_batch_bytes: as webshart counts it, plus a path row's length."""
    return 32 + len(value) + (len(row.encode("utf-8")) if isinstance(row, str) else 0)


class PythonTransformCache:
    """
    webshart.TransformCache for datasets without shards: one SQLite file per partition and recipe, a
    bounded queue drained by one writer thread, one transaction per batch, the first committed value
    kept, and reads in the order asked. Rows are integers or strings. Open one in each process.

    replace() is the one write that changes a recorded value; it is for edits a person makes.
    """

    _BUSY_SECONDS = 5

    def __init__(self, directory, recipe: str, *, queue_batches: int = 16, max_batch_bytes: int = 1 << 20):
        if not recipe or queue_batches < 1 or max_batch_bytes < 1:
            raise ValueError("recipe must be nonempty, and queue_batches and max_batch_bytes positive")
        self.directory, self.recipe, self.max_batch_bytes = Path(directory), recipe, max_batch_bytes
        self._queue: queue.Queue = queue.Queue(maxsize=queue_batches)
        self._failure: BaseException | None = None
        self._closed = False
        self._writer = threading.Thread(target=self._write_batches, name="transform-cache-writer", daemon=True)
        self._writer.start()

    def path(self, partition: str) -> Path:
        digest = sha256()
        digest.update(b"simpletuner-transform-cache-v1")
        digest.update(len(self.recipe).to_bytes(8, "little"))
        digest.update(self.recipe.encode("utf-8"))
        digest.update(partition.encode("utf-8"))
        name = digest.hexdigest()
        return self.directory / name[:2] / f"{name}.sqlite3"

    def read_batch(self, requests) -> list[bytes | None]:
        result: list[bytes | None] = [None] * len(requests)
        groups: dict[str, list[tuple[int, int | str]]] = {}
        for position, (partition, row) in enumerate(requests):
            groups.setdefault(partition, []).append((position, row))
        for partition, rows in groups.items():
            path = self.path(partition)
            if not path.exists():
                continue
            uri = f"{path.resolve().as_uri()}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=self._BUSY_SECONDS)) as connection:
                # The file can be visible before its first writer has made the table.
                ready = connection.execute("SELECT 1 FROM sqlite_master WHERE name = 'transforms'").fetchone()
                if ready is None:
                    continue
                for position, row in rows:
                    found = connection.execute("SELECT value FROM transforms WHERE row = ?", (row,)).fetchone()
                    result[position] = None if found is None else bytes(found[0])
        return result

    def enqueue(self, partition: str, records) -> None:
        """Hand a partition's batch to the writer, waiting while the queue is full."""
        records = list(records)
        size = len(partition.encode("utf-8")) + sum(record_bytes(row, value) for row, value in records)
        if size > self.max_batch_bytes:
            raise ValueError("transform batch exceeds max_batch_bytes")
        self._put(("batch", partition, records))

    def flush(self) -> None:
        """Wait for every batch handed over so far to commit."""
        done = threading.Event()
        self._put(("flush", done))
        done.wait()
        self._raise_failure()

    def replace(self, partition: str, records) -> None:
        """Record values over what is recorded. A queued batch for the same rows keeps these."""
        with closing(self._open_writer(partition)) as connection:
            with connection:
                connection.executemany(
                    "INSERT INTO transforms VALUES (?, ?) ON CONFLICT(row) DO UPDATE SET value = excluded.value",
                    list(records),
                )

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._queue.put(None)
        self._writer.join()

    def _put(self, command) -> None:
        if self._closed:
            raise ValueError("transform cache is closed")
        self._raise_failure()
        self._queue.put(command)

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise OSError(f"transform cache write failed: {self._failure}") from self._failure

    def _open_writer(self, partition: str) -> sqlite3.Connection:
        path = self.path(partition)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=self._BUSY_SECONDS)
        connection.execute("CREATE TABLE IF NOT EXISTS transforms (row PRIMARY KEY, value BLOB NOT NULL)")
        return connection

    def _write_batches(self) -> None:
        current: tuple[str, sqlite3.Connection] | None = None
        try:
            while (command := self._queue.get()) is not None:
                if command[0] == "flush":
                    command[1].set()
                    continue
                if self._failure is not None:
                    continue
                _, partition, records = command
                try:
                    if current is None or current[0] != partition:
                        if current is not None:
                            current[1].close()
                        current = (partition, self._open_writer(partition))
                    with current[1]:
                        current[1].executemany("INSERT INTO transforms VALUES (?, ?) ON CONFLICT(row) DO NOTHING", records)
                except BaseException as error:
                    self._failure = error
        finally:
            if current is not None:
                current[1].close()
