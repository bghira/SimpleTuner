"""Hardware detection utilities used by the Web UI and training runtime."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _format_memory_bytes(total_bytes: Optional[int]) -> Optional[float]:
    if not total_bytes:
        return None
    try:
        gib = total_bytes / (1024**3)
    except Exception:
        return None
    return round(gib, 2)


def _resolve_visible_cuda_devices() -> Optional[List[int]]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return None
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    parsed: List[int] = []
    for token in tokens:
        try:
            parsed.append(int(token))
        except ValueError:
            logger.debug("Skipping non-integer CUDA_VISIBLE_DEVICES entry: %s", token)
    return parsed or None


def detect_gpu_inventory() -> Dict[str, Any]:
    """
    Detect the available GPU/accelerator devices.

    Returns:
        Dictionary describing detected devices and recommended process counts.
    """

    devices: List[Dict[str, Any]] = []
    backend: Optional[str] = None
    detected = False
    visible_env = _resolve_visible_cuda_devices()

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            backend = "cuda"
            try:
                count = torch.cuda.device_count()
            except Exception as error:
                logger.debug("Failed to query CUDA device count: %s", error, exc_info=True)
                count = 0

            for index in range(max(count, 0)):
                name = None
                memory_bytes = None
                try:
                    name = torch.cuda.get_device_name(index)
                except Exception:
                    logger.debug("Unable to read CUDA device name for index %s", index, exc_info=True)

                try:
                    props = torch.cuda.get_device_properties(index)
                    memory_bytes = getattr(props, "total_memory", None)
                except Exception:
                    logger.debug("Unable to read CUDA properties for index %s", index, exc_info=True)

                devices.append(
                    {
                        "index": index,
                        "id": index,
                        "name": name or f"GPU {index}",
                        "memory_bytes": memory_bytes,
                        "memory_gb": _format_memory_bytes(memory_bytes),
                    }
                )

            detected = bool(devices)

        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            backend = "mps"
            devices.append(
                {
                    "index": 0,
                    "id": 0,
                    "name": "Apple Metal (MPS)",
                    "memory_bytes": None,
                    "memory_gb": None,
                }
            )
            detected = True

    except Exception:
        logger.debug("GPU detection via torch is unavailable", exc_info=True)

    count = len(devices)
    optimal_processes = max(count, 1)

    if not detected:
        backend = backend or "cpu"

    return {
        "detected": detected,
        "backend": backend,
        "devices": devices,
        "count": count,
        "optimal_processes": optimal_processes,
        "visible_device_ids": visible_env,
    }
