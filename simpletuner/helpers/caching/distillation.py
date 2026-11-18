import logging
import os
from bisect import bisect_left
from typing import List, Optional, Tuple

from simpletuner.helpers.training.multi_process import rank_info, should_log
from simpletuner.helpers.webhooks.mixin import WebhookMixin

logger = logging.getLogger("DistillationCache")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class DistillationCache(WebhookMixin):
    """Generic storage helper for deterministic ODE pair artifacts."""

    def __init__(self, id: str, data_backend, cache_dir: str, distillation_type: str = "generic"):
        self.id = id
        self.data_backend = data_backend
        self.cache_dir = cache_dir or ""
        self.distillation_type = distillation_type or "generic"
        self.rank_info = rank_info()
        self.webhook_handler = None

        if self.data_backend and self.cache_dir:
            self.data_backend.create_directory(self.cache_dir)
        self._artifact_paths: Optional[List[str]] = None
        self._artifact_cursor: int = 0

    def debug(self, message: str) -> None:
        logger.debug(f"{self.rank_info}(distillation_cache id={self.id}) {message}")

    def set_webhook_handler(self, webhook_handler):
        self.webhook_handler = webhook_handler

    def discover_all_files(self) -> List[str]:
        """Reopen artifact traversal against the store's current membership."""
        self._artifact_paths = sorted(set(self.data_backend.iter_file_keys(["pt"], self.cache_dir)))
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
            if position < self._artifact_cursor:
                self._artifact_cursor += 1
        return target

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
