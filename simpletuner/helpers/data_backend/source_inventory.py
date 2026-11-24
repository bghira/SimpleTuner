"""File and shard inventories exposed by dataset sources."""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceShard:
    """A source-relative shard path with any published size and row count."""

    path: str
    size_bytes: int | None = None
    rows: int | None = None


class SourceFileInventory:
    def __init__(self, path: Path | None = None, bucket_report=None):
        self.path = path
        self.bucket_report = bucket_report
        self._keys = None

    def read(self, *, retry_limit=0):
        if self._keys is not None or self.path is None:
            return self._keys
        for attempt in range(retry_limit + 1):
            try:
                with self.path.open(encoding="utf-8") as stream:
                    self._keys = json.load(stream)
                return self._keys
            except FileNotFoundError:
                if attempt < retry_limit:
                    time.sleep(1)
            except (OSError, ValueError) as exc:
                logger.warning("Could not read source file inventory %s: %s", self.path, exc)
                return None
        return None

    def publish(self, groups):
        keys = {}
        directories = 0
        for _, _, files in groups:
            directories += 1
            for key in files:
                keys[key] = False

        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            stream = NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False)
            temporary = Path(stream.name)
            try:
                with stream:
                    json.dump(keys, stream)
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
        self._keys = keys
        if self.bucket_report:
            self.bucket_report.record_stage("file_discovery", sample_count=len(keys), directories=directories)
        return keys
