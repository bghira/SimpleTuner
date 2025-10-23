"""Maintenance utilities for cache management."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .fsdp_service import FSDP_SERVICE
from .training_service import get_config_store

logger = logging.getLogger(__name__)


class MaintenanceServiceError(Exception):
    """Domain error raised when maintenance operations fail."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class MaintenanceService:
    """Service providing maintenance utilities for the WebUI."""

    def clear_fsdp_block_cache(self) -> Dict[str, Any]:
        """Clear cached FSDP block detection results."""
        result = FSDP_SERVICE.clear_cache()
        return {
            "cleared": True,
            "details": result,
        }

    def clear_deepspeed_offload_cache(self, config_name: Optional[str] = None) -> Dict[str, Any]:
        """Remove DeepSpeed NVMe offload directories as configured in the active environment."""

        store = get_config_store()
        active_config = config_name or store.get_active_config()
        if not active_config:
            raise MaintenanceServiceError("No active training configuration is selected.")

        try:
            config_data, _ = store.load_config(active_config)
        except (FileNotFoundError, ValueError) as exc:  # pragma: no cover - defensive
            raise MaintenanceServiceError(f"Unable to load configuration '{active_config}': {exc}") from exc

        offload_path = self._extract_offload_path(config_data)
        if not offload_path:
            raise MaintenanceServiceError("The active configuration does not define --offload_param_path. Nothing to clear.")

        resolved_root = Path(os.path.expanduser(str(offload_path))).resolve()
        if not resolved_root.exists() or not resolved_root.is_dir():
            raise MaintenanceServiceError(f"Offload path '{resolved_root}' does not exist or is not a directory.")

        removed: List[str] = []
        for entry in resolved_root.iterdir():
            if not entry.is_dir():
                continue
            if not entry.name.startswith("zero_stage_"):
                continue
            try:
                shutil.rmtree(entry, ignore_errors=True)
                removed.append(str(entry))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Unable to remove DeepSpeed offload directory %s: %s", entry, exc, exc_info=True)

        return {
            "cleared": True,
            "root": str(resolved_root),
            "removed": removed,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _extract_offload_path(self, config: Dict[str, Any]) -> Optional[str]:
        candidates = [
            config.get("--offload_param_path"),
            config.get("offload_param_path"),
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None


MAINTENANCE_SERVICE = MaintenanceService()
