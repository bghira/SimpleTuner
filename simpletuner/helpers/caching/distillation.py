import hashlib
import json
import logging
import os
from bisect import bisect_left
from pathlib import Path
from typing import List, Optional, Tuple

from atomicwrites import atomic_write

from simpletuner.helpers.training.multi_process import rank_info, should_log
from simpletuner.helpers.webhooks.mixin import WebhookMixin

logger = logging.getLogger("DistillationCache")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class DistillationCache(WebhookMixin):
    """Generic storage helper for deterministic ODE pair artifacts."""

    def __init__(self, id: str, data_backend, cache_dir: str, distillation_type: str = "generic", *, config=None):
        self.id = id
        self.data_backend = data_backend
        self.cache_dir = cache_dir or ""
        self.distillation_type = distillation_type or "generic"
        self.config = dict(config or {})
        self.rank_info = rank_info()
        self.webhook_handler = None

        if self.data_backend and self.cache_dir:
            self.data_backend.create_directory(self.cache_dir)
        self._artifact_paths: Optional[List[str]] = None
        self._artifact_cursor: int = 0
        self._membership_revision: Optional[str] = None

    def debug(self, message: str) -> None:
        logger.debug(f"{self.rank_info}(distillation_cache id={self.id}) {message}")

    def set_webhook_handler(self, webhook_handler):
        self.webhook_handler = webhook_handler

    def discover_all_files(self) -> List[str]:
        """Reopen artifact traversal against the store's current membership."""
        self._artifact_paths = sorted(set(self.data_backend.iter_file_keys(["pt"], self.cache_dir)))
        self._membership_revision = None
        self._artifact_cursor = 0
        self.debug(f"discovered {len(self._artifact_paths)} distillation cache entries.")
        return self._artifact_paths

    def _paths(self) -> List[str]:
        return self.discover_all_files() if self._artifact_paths is None else self._artifact_paths

    def list_cached_pairs(self) -> List[str]:
        return self._paths().copy()

    def has_cached_pairs(self) -> bool:
        return bool(self._paths())

    def next_artifact_name(self, prefix: str = "pair") -> str:
        paths = self._paths()
        suffix = len(paths)
        while True:
            filename = f"{prefix}_{suffix:05d}.pt"
            path = self._artifact_path(filename)
            position = bisect_left(paths, path)
            if position == len(paths) or paths[position] != path:
                return filename
            suffix += 1

    def _artifact_path(self, filename):
        path = os.path.join(self.cache_dir, filename)
        return os.path.abspath(path) if self.data_backend.type == "local" else path

    def write_tensor(self, filename: str, payload) -> str:
        if not filename.endswith(".pt"):
            filename = f"{filename}.pt"
        paths = self._paths()
        target = self._artifact_path(filename)
        self.data_backend.torch_save(payload, target)
        position = bisect_left(paths, target)
        if position == len(paths) or paths[position] != target:
            paths.insert(position, target)
            self._membership_revision = None
            if position < self._artifact_cursor:
                self._artifact_cursor += 1
        return target

    def checkpoint_position(self):
        paths = self._paths()
        if self._membership_revision is None:
            digest = hashlib.sha256()
            for path in paths:
                digest.update(json.dumps(path).encode("utf-8"))
                digest.update(b"\n")
            self._membership_revision = digest.hexdigest()
        return {"id": self.id, "membership": self._membership_revision, "position": self._artifact_cursor}

    def validate_position(self, state):
        current = self.checkpoint_position()
        position = state.get("position")
        if (
            state.get("id") != current["id"]
            or state.get("membership") != current["membership"]
            or type(position) is not int
            or not 0 <= position < max(1, len(self._paths()))
        ):
            raise ValueError(f"Distillation cache checkpoint does not match the artifact inventory for {self.id}.")
        return position

    def restore_position(self, position):
        self._artifact_cursor = position

    def load_next_pair(self) -> Tuple[Optional[object], Optional[str]]:
        """
        Retrieve the next cached pair for consumption.

        Returns the payload loaded via torch along with the source path. If no entries
        are available, (None, None) is returned.
        """
        paths = self._paths()
        if not paths:
            return None, None

        path = paths[self._artifact_cursor]
        self._artifact_cursor = (self._artifact_cursor + 1) % len(paths)

        try:
            payload = self.data_backend.torch_load(path)
        except Exception as exc:  # pragma: no cover - storage edge case logging
            logger.error("Failed to load distillation cache artifact '%s': %s", path, exc)
            return None, None
        return payload, path


class DistillationArtifactCycle:
    """Round-robin artifact consumption with independent per-store positions."""

    def __init__(self, caches):
        self.caches = tuple(caches)
        if not self.caches:
            raise ValueError("Distillation requires at least one artifact cache.")
        self.position = 0

    def next_entry(self):
        for _ in self.caches:
            cache = self.caches[self.position]
            self.position = (self.position + 1) % len(self.caches)
            payload, path = cache.load_next_pair()
            if payload is None:
                continue
            entry = dict(payload) if isinstance(payload, dict) else {"value": payload}
            metadata = dict(entry.get("metadata") or {})
            if path is not None:
                metadata.setdefault("artifact_path", path)
            metadata.setdefault("distillation_type", cache.distillation_type)
            entry["metadata"] = metadata
            return entry
        raise ValueError("Distillation caches did not yield any artifacts for training.")

    def save_checkpoint(self, directory, rank, world_size):
        path = self._checkpoint_path(directory, rank)
        state = {
            "version": 1,
            "world_size": world_size,
            "position": self.position,
            "caches": [cache.checkpoint_position() for cache in self.caches],
        }
        with atomic_write(path, overwrite=True, encoding="utf-8") as stream:
            json.dump(state, stream)

    def load_checkpoint(self, directory, rank, world_size):
        path = self._checkpoint_path(directory, rank)
        if not path.is_file():
            return
        state = json.loads(path.read_text(encoding="utf-8"))
        position = state.get("position")
        saved_caches = state.get("caches", ())
        if (
            state.get("version") != 1
            or state.get("world_size") != world_size
            or type(position) is not int
            or not 0 <= position < len(self.caches)
            or len(saved_caches) != len(self.caches)
        ):
            raise ValueError("Distillation cache checkpoint does not match the configured traversal.")
        positions = [cache.validate_position(saved) for cache, saved in zip(self.caches, saved_caches, strict=True)]
        for cache, cache_position in zip(self.caches, positions, strict=True):
            cache.restore_position(cache_position)
        self.position = position

    @staticmethod
    def _checkpoint_path(directory, rank):
        return Path(directory) / f"distillation-artifacts-rank-{rank}.json"
