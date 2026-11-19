"""The geometry decided for a sample, how it is encoded, the recipe of settings it follows from, and where it is kept."""

import json
import logging
import os
import queue
import sqlite3
import tempfile
import threading
from contextlib import closing
from hashlib import sha256
from pathlib import Path

from simpletuner.helpers.data_backend.dataset_runtime import RUNTIME_MUTABLE_KEYS, recrops_each_epoch
from simpletuner.helpers.data_backend.dataset_types import DatasetType, ensure_dataset_type
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)

DECIDED = ("aspect_ratio", "target_size", "intermediary_size", "crop_coordinates")
_PAIRS = DECIDED[1:]
_TRANSFORM_BATCH_BYTES = 1 << 20
_TRANSFORM_READ_ROWS = 10_000
# Data backends that write metadata files to this machine's disk, so the store can sit beside them.
_LOCAL_METADATA = ("local", "csv", "huggingface")


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


class RecordedGeometry:
    """
    For a metadata backend whose samples are listed in a bucket index: the geometry decided for each sample
    is recorded in a PythonTransformCache under the recipe the index names, and the metadata file keeps what
    was measured. A dataset that re-crops each epoch has the epoch in its recipe. The backend names its recipe
    and calls these where it loads, saves and buckets.
    """

    transform_recipe_name: str
    _recipe: str | None = None
    _transforms: PythonTransformCache | None = None
    _wrote_transforms = False
    image_metadata_loaded = False
    # A copy of another dataset's metadata: the source decided its geometry, so it stays in this metadata file.
    geometry_is_copied = False

    def record_transforms(self, records: dict[str, dict]) -> None:
        if not self._decides_geometry() or not records:
            return
        if self._recipe is None:
            self._recipe = self._new_recipe()
        store = self._transform_store()
        empty = len(self.id.encode("utf-8"))
        batch, size = [], empty
        for path, metadata in records.items():
            row = (str(path), encode_transform(metadata))
            cost = record_bytes(*row)
            if batch and size + cost > _TRANSFORM_BATCH_BYTES:
                store.enqueue(self.id, batch)
                batch, size = [], empty
            batch.append(row)
            size += cost
        store.enqueue(self.id, batch)
        self._wrote_transforms = True

    def _adopt_index_recipe(self, cache_data: dict) -> None:
        """Take the recipe a bucket index names; the geometry of its samples is read again under it."""
        recipe = cache_data.get("transform_recipe")
        if recipe != self._recipe:
            self._recipe = recipe
            if self.image_metadata_loaded:
                self.load_image_metadata()

    def _committed_recipe(self) -> str | None:
        """The recipe for a bucket index to name, once every record it lists is committed."""
        self._flush_transforms()
        return self._recipe

    def _metadata_to_save(self) -> dict:
        """What was measured of each sample, once the geometry decided for it is committed."""
        if not self._decides_geometry():
            return self.image_metadata
        self._flush_transforms()
        return {
            path: {key: value for key, value in metadata.items() if key not in DECIDED}
            for path, metadata in self.image_metadata.items()
        }

    def _begin_bucketing(self, ignore_existing_cache: bool) -> bool:
        """Whether every sample is decided again: a forced rescan, or an index that names no recipe."""
        if not self._decides_geometry():
            return ignore_existing_cache
        if self._recipe is None and any(self.aspect_ratio_bucket_indices.values()):
            logger.info("(id=%s) Its bucket index names no recipe of recorded geometry; bucketing it again.", self.id)
            ignore_existing_cache = True
        if ignore_existing_cache:
            self._recipe = self._new_recipe()
            self._discard_transforms()
        return ignore_existing_cache

    def _end_bucketing(self) -> None:
        if self._wrote_transforms:
            # The first geometry recorded for a sample is kept, so every sample takes what was recorded.
            self._merge_recorded_geometry()

    def _new_recipe(self) -> str:
        return transform_recipe(self.transform_recipe_name, StateTracker.get_epoch(), self.id)

    def _decides_geometry(self) -> bool:
        if self.geometry_is_copied:
            return False
        return ensure_dataset_type(self.dataset_type, default=DatasetType.IMAGE) is not DatasetType.AUDIO

    def _metadata_on_local_disk(self) -> bool:
        return self.data_backend.type in _LOCAL_METADATA

    def _metadata_directory(self) -> Path:
        """The directory the data backend writes the metadata file into."""
        return Path(self.metadata_file).parent

    def _transform_store(self) -> PythonTransformCache:
        """This process's handle on the store of the recipe in force."""
        if self._transforms is None or self._transforms.recipe != self._recipe:
            self.close_transforms()
            if self._metadata_on_local_disk():
                directory = self._metadata_directory() / "transforms"
            else:
                # Each process keeps its own copy of the published store, so a download never replaces a file being written.
                directory = (
                    Path(StateTracker.get_args().cache_dir)
                    / "transforms"
                    / self.id
                    / f"process-{self.accelerator.process_index}"
                )
            self._transforms = PythonTransformCache(directory, self._recipe, max_batch_bytes=_TRANSFORM_BATCH_BYTES)
        return self._transforms

    def close_transforms(self) -> None:
        if self._transforms is not None:
            self._transforms.close()
            self._transforms = None
            self._wrote_transforms = False

    def _discard_transforms(self) -> None:
        """A forced rescan decides every sample again, so what its recipe recorded is dropped first."""
        path = self._transform_store().path(self.id)
        self.close_transforms()
        path.unlink(missing_ok=True)

    def _published_transforms(self, store: PythonTransformCache) -> str:
        """Where the store is published beside the metadata file, for a data backend not on local disk."""
        return str(self._metadata_directory() / "transforms" / store.path(self.id).relative_to(store.directory))

    def _flush_transforms(self) -> None:
        if not self._wrote_transforms:
            return
        self._transforms.flush()
        if not self._metadata_on_local_disk():
            self.data_backend.write(
                self._published_transforms(self._transforms), self._transforms.path(self.id).read_bytes()
            )

    def _download_transforms(self, store: PythonTransformCache) -> None:
        """Replace this process's copy with the published store, which only the writing process changes."""
        published = self._published_transforms(store)
        if not self.data_backend.exists(published):
            return
        local = store.path(self.id)
        local.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=local.parent, prefix=f".{local.name}.", suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(self.data_backend.read(published))
            os.replace(temporary, local)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def _merge_recorded_geometry(self) -> None:
        """Give each sample the geometry recorded for it. A sample with none was not decided under this recipe."""
        if not self._decides_geometry():
            return
        if self._recipe is None:
            self.image_metadata = {}
            return
        store = self._transform_store()
        if self._wrote_transforms:
            store.flush()
        elif not self._metadata_on_local_disk():
            self._download_transforms(store)
        paths = list(self.image_metadata)
        for start in range(0, len(paths), _TRANSFORM_READ_ROWS):
            chunk = paths[start : start + _TRANSFORM_READ_ROWS]
            for path, value in zip(chunk, store.read_batch([(self.id, path) for path in chunk])):
                if value is None:
                    del self.image_metadata[path]
                else:
                    self.image_metadata[path].update(decode_transform(value))
