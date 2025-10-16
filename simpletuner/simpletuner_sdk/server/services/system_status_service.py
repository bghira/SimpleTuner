"""System status metrics service used by the Web UI event dock."""

from __future__ import annotations

import logging
import os
import platform
import plistlib
import subprocess
import time
from typing import Any, Dict, List, Optional

from .hardware_service import detect_gpu_inventory

logger = logging.getLogger(__name__)

try:  # psutil is optional at runtime
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

try:  # torch is required for CUDA utilisation but might be unavailable in some environments
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency in CPU-only environments
    torch = None  # type: ignore


class SystemStatusService:
    """Expose basic system statistics for display in the Web UI."""

    def get_status(self) -> Dict[str, Any]:
        """Collect system metrics for the API response."""
        return {
            "timestamp": time.time(),
            "load_avg_5min": self._get_load_average_5min(),
            "memory_percent": self._get_memory_percent(),
            "gpus": self._get_gpu_utilisation(),
        }

    def _get_load_average_5min(self) -> Optional[float]:
        try:
            load_averages = os.getloadavg()
        except (AttributeError, OSError):
            return None
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to read system load averages", exc_info=True)
            return None

        if len(load_averages) < 2:
            return None
        return round(float(load_averages[1]), 2)

    def _get_memory_percent(self) -> Optional[float]:
        if psutil is None:
            return None
        try:
            memory = psutil.virtual_memory()
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to read system memory metrics", exc_info=True)
            return None
        return round(float(memory.percent), 1)

    def _get_gpu_utilisation(self) -> List[Dict[str, Any]]:
        inventory = detect_gpu_inventory()
        backend = (inventory or {}).get("backend")
        devices = (inventory or {}).get("devices") or []
        results: List[Dict[str, Any]] = []
        mac_utilisation: Optional[List[Optional[float]]] = None
        nvidia_fallback: Optional[List[Optional[float]]] = None

        if backend == "mps":
            mac_utilisation = self._get_macos_gpu_utilisation()

        for position, device in enumerate(devices):
            index = device.get("index")
            name = device.get("name") or f"GPU {index if index is not None else '?'}"
            utilisation: Optional[float] = None

            if (
                backend == "cuda"
                and index is not None
                and torch is not None
                and hasattr(torch.cuda, "utilization")
            ):
                try:
                    if torch.cuda.is_available():  # type: ignore[attr-defined]
                        utilisation = float(torch.cuda.utilization(index))  # type: ignore[attr-defined]
                except Exception:
                    logger.debug("Failed to read CUDA utilisation for device %s", index, exc_info=True)
                    utilisation = None
            if utilisation is None and backend == "mps" and mac_utilisation:
                target_idx: Optional[int] = None
                if isinstance(index, int) and 0 <= index < len(mac_utilisation):
                    target_idx = index
                elif 0 <= position < len(mac_utilisation):
                    target_idx = position
                if target_idx is not None:
                    utilisation = mac_utilisation[target_idx]
            if utilisation is None and backend == "cuda":
                if nvidia_fallback is None:
                    nvidia_fallback = self._get_nvidia_gpu_utilisation()
                if nvidia_fallback:
                    target_idx = None
                    if isinstance(index, int) and 0 <= index < len(nvidia_fallback):
                        target_idx = index
                    elif 0 <= position < len(nvidia_fallback):
                        target_idx = position
                    if target_idx is not None:
                        utilisation = nvidia_fallback[target_idx]

            results.append(
                {
                    "index": index,
                    "name": name,
                    "backend": backend,
                    "utilization_percent": round(utilisation, 1) if utilisation is not None else None,
                }
            )

        return results

    def _get_macos_gpu_utilisation(self) -> Optional[List[Optional[float]]]:
        if platform.system() != "Darwin":
            return None

        try:
            completed = subprocess.run(
                ["ioreg", "-r", "-k", "PerformanceStatistics", "-d", "1", "-a"],
                check=True,
                capture_output=True,
                text=False,
                timeout=2,
            )
            if not completed.stdout:
                return None
            data = plistlib.loads(completed.stdout)
        except (FileNotFoundError, subprocess.SubprocessError, plistlib.InvalidFileException, ValueError) as exc:
            logger.debug("Unable to query macOS GPU utilisation via ioreg: %s", exc, exc_info=True)
            return None

        if not isinstance(data, list):
            return None

        utilisation_values: List[Optional[float]] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            perf = entry.get("PerformanceStatistics")
            if not isinstance(perf, dict):
                continue
            value = perf.get("Device Utilization %")
            if isinstance(value, (int, float)):
                utilisation_values.append(round(float(value), 1))
            else:
                utilisation_values.append(None)

        return utilisation_values or None

    def _get_nvidia_gpu_utilisation(self) -> Optional[List[Optional[float]]]:
        if platform.system() == "Darwin":
            return None

        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            logger.debug("Unable to query GPU utilisation via nvidia-smi: %s", exc, exc_info=True)
            return None

        lines = completed.stdout.strip().splitlines()
        if not lines:
            return None

        utilisation_values: List[Optional[float]] = []
        for line in lines:
            text = line.strip()
            if not text:
                utilisation_values.append(None)
                continue
            try:
                utilisation_values.append(round(float(text), 1))
            except ValueError:
                logger.debug("Discarding unexpected nvidia-smi output: %s", text)
                utilisation_values.append(None)

        return utilisation_values or None


__all__ = ["SystemStatusService"]
