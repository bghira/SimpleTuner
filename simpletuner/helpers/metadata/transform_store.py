"""SQLite storage for partitioned sample transform records."""

import json
import os
import re
import sqlite3
from collections import defaultdict
from collections.abc import MutableMapping
from contextlib import closing
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock

GEOMETRY_FIELDS = frozenset(("aspect_ratio", "target_size", "intermediary_size", "crop_coordinates"))


class SQLiteTransformStore(MutableMapping):
    def __init__(self, directory, partition_for_ref):
        self.directory = Path(directory).absolute()
        self.partition_for_ref = partition_for_ref

    def _partition_path(self, ref):
        return self.directory / f"{self.partition_for_ref(ref)}.sqlite3"

    def __getitem__(self, ref):
        (record,) = self.read_batch((ref,))
        if record is None:
            raise KeyError(ref)
        return record

    def __setitem__(self, ref, record):
        self.update({ref: record})

    def __delitem__(self, ref):
        path = self._partition_path(ref)
        if not path.is_file():
            raise KeyError(ref)
        with closing(sqlite3.connect(path)) as database, database:
            if not database.execute("DELETE FROM transforms WHERE ref = ?", (ref,)).rowcount:
                raise KeyError(ref)

    def __iter__(self):
        for path in self.directory.glob("*.sqlite3"):
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as database:
                for (ref,) in database.execute("SELECT ref FROM transforms"):
                    yield ref

    def __len__(self):
        count = 0
        for path in self.directory.glob("*.sqlite3"):
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as database:
                count += database.execute("SELECT count(*) FROM transforms").fetchone()[0]
        return count

    def update(self, records):
        groups = defaultdict(list)
        for ref, record in records.items():
            groups[self._partition_path(ref)].append((ref, json.dumps(record, separators=(",", ":"))))
        for path, rows in groups.items():
            if not path.is_file():
                self.directory.mkdir(parents=True, exist_ok=True)
                with FileLock(str(path) + ".lock"):
                    if not path.is_file():
                        with TemporaryDirectory(dir=self.directory) as temporary:
                            staging = Path(temporary) / path.name
                            with closing(sqlite3.connect(staging)) as database, database:
                                database.execute("CREATE TABLE transforms (ref TEXT PRIMARY KEY, record TEXT NOT NULL)")
                            os.replace(staging, path)
            with closing(sqlite3.connect(path, timeout=30)) as database, database:
                database.executemany(
                    "INSERT INTO transforms VALUES (?, ?) "
                    "ON CONFLICT(ref) DO UPDATE SET record = excluded.record WHERE transforms.record != excluded.record",
                    rows,
                )

    def read_batch(self, refs):
        refs = tuple(refs)
        groups = defaultdict(set)
        for ref in refs:
            try:
                path = self._partition_path(ref)
            except KeyError:
                continue
            groups[path].add(ref)
        records = {}
        for path, keys in groups.items():
            if not path.is_file():
                continue
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as database:
                for batch in batched(keys, 512):
                    placeholders = ",".join("?" for _ in batch)
                    values = database.execute(f"SELECT ref, record FROM transforms WHERE ref IN ({placeholders})", batch)
                    records.update((ref, json.loads(record)) for ref, record in values)
        return [records.get(ref) for ref in refs]


def transform_recipe_directory(metadata_file, recipe):
    if not isinstance(recipe, str) or not re.fullmatch(r"[a-f0-9]{16}(?:-epoch-[0-9]+)?-[a-f0-9]{32}", recipe):
        raise ValueError("Invalid dataset transform recipe.")
    return Path(metadata_file).with_suffix(".transforms") / recipe


class DatasetTransformStore(SQLiteTransformStore):
    """A single transform partition for an unsharded dataset."""

    filename = "0.sqlite3"
    measurements_filename = "metadata.json"

    def __init__(self, directory):
        super().__init__(directory, lambda _: 0)

    @property
    def path(self):
        return self.directory / self.filename

    @property
    def measurements_path(self):
        return self.directory / self.measurements_filename
