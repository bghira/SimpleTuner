from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger("SelfForcingODEGenerator")


class SelfForcingODEGenerator:
    """Simple placeholder generator that materialises deterministic ODE pairs into the cache."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def generate(self, cache, backend_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Materialise a placeholder ODE entry when the cache is empty.

        The real implementation should mirror realtime-video/scripts/generate_ode_pairs.py, but we
        keep things simple here to exercise the caching pipeline.
        """
        cache.discover_all_files()
        if cache.has_cached_pairs() and not self.config.get("force_regenerate", False):
            logger.debug("Distillation cache %s already contains ODE pairs; skipping generation.", cache.id)
            return

        filename = cache.next_artifact_name(prefix=self.config.get("artifact_prefix", "self_forcing"))
        payload = {
            "metadata": {
                "generated_at": time.time(),
                "distillation_type": cache.distillation_type,
                "guidance_scale": self.config.get("guidance_scale", 6.0),
            },
            # Placeholder tensor to confirm persistence works end-to-end.
            "latents": torch.zeros((1,), dtype=torch.float32),
        }

        artifact_path = cache.write_tensor(filename, payload)
        logger.info("Wrote placeholder ODE pair to %s", artifact_path)
