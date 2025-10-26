import logging
import os
from typing import List

from simpletuner.helpers.training.multi_process import rank_info, should_log
from simpletuner.helpers.training.state_tracker import StateTracker

try:
    from simpletuner.helpers.webhooks.mixin import WebhookMixin
except Exception:  # pragma: no cover

    class WebhookMixin:  # type: ignore
        def set_webhook_handler(self, webhook_handler):
            self.webhook_handler = webhook_handler


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

    def debug(self, message: str) -> None:
        logger.debug(f"{self.rank_info}(distillation_cache id={self.id}) {message}")

    def set_webhook_handler(self, webhook_handler):
        self.webhook_handler = webhook_handler

    def discover_all_files(self) -> List[str]:
        """Refresh StateTracker with the current set of cached artifacts."""
        listing = self.data_backend.list_files(
            file_extensions=["pt"],
            instance_data_dir=self.cache_dir,
        )
        StateTracker.set_distillation_cache_files(listing, data_backend_id=self.id)
        flattened = self._flatten_listing(listing)
        self.debug(f"discovered {len(flattened)} distillation cache entries.")
        return flattened

    def _flatten_listing(self, listing) -> List[str]:
        paths: List[str] = []
        for _, _, files in listing or []:
            paths.extend(files)
        return paths

    def list_cached_pairs(self) -> List[str]:
        return list(StateTracker.get_distillation_cache_files(self.id).keys())

    def has_cached_pairs(self) -> bool:
        return bool(self.list_cached_pairs())

    def next_artifact_name(self, prefix: str = "pair") -> str:
        existing = self.list_cached_pairs()
        suffix = len(existing)
        return f"{prefix}_{suffix:05d}.pt"

    def write_tensor(self, filename: str, payload) -> str:
        if not filename.endswith(".pt"):
            filename = f"{filename}.pt"
        target = os.path.join(self.cache_dir, filename)
        self.data_backend.torch_save(payload, target)
        # Update StateTracker with the new listing so downstream consumers see the fresh entry.
        self.discover_all_files()
        return target
