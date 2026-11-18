"""A bounded, run-scoped presence index for an artifact-store prefix."""

import os
import sqlite3
from contextlib import closing, contextmanager
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from filelock import FileLock

from simpletuner.helpers.training.multi_process import broadcast_object_from_main


class ArtifactListing:
    def __init__(self, store, prefix, output_dir, accelerator):
        self.store = store
        self.prefix = prefix
        token = broadcast_object_from_main(uuid4().hex)
        self.path = Path(output_dir) / ".artifact-listings" / f"{token}.sqlite"
        self.lock = FileLock(f"{self.path}.lock")
        self.owns_file = accelerator.is_local_main_process

    def _build(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            if self.path.is_file():
                return
            with TemporaryDirectory(dir=self.path.parent) as staging:
                path = Path(staging) / self.path.name
                with closing(sqlite3.connect(path)) as db:
                    db.execute("PRAGMA cache_size=-8192")
                    db.execute("CREATE TABLE artifacts(key TEXT PRIMARY KEY) WITHOUT ROWID")
                    for keys in batched(self.store.iter_file_keys(["pt"], self.prefix), 4096):
                        db.executemany("INSERT OR IGNORE INTO artifacts VALUES (?)", ((key,) for key in keys))
                    db.commit()
                os.replace(path, self.path)

    @contextmanager
    def open(self):
        self._build()
        with closing(sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)) as db:
            db.execute("PRAGMA cache_size=-8192")

            def probe(keys):
                requested = tuple(keys)
                present = set()
                for chunk in batched(requested, db.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)):
                    placeholders = ",".join("?" for _ in chunk)
                    present.update(
                        row[0] for row in db.execute(f"SELECT key FROM artifacts WHERE key IN ({placeholders})", chunk)
                    )
                misses = tuple(dict.fromkeys(key for key in requested if key not in present))
                if misses:
                    present.update(key for key, found in zip(misses, self.store.bulk_exists(misses), strict=True) if found)
                return [key in present for key in requested]

            yield probe

    def close(self):
        if self.owns_file:
            self.path.unlink(missing_ok=True)
            Path(self.lock.lock_file).unlink(missing_ok=True)
