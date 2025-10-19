"""System status metrics service used by the Web UI event dock."""

from __future__ import annotations

import ast
import json
import logging
import os
import platform
import plistlib
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

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

try:  # nvidia-ml-py exposes the pynvml module
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore


class SystemStatusService:
    """Expose basic system statistics for display in the Web UI."""

    def __init__(self) -> None:
        self._offload_cache: Dict[str, Any] = {}

    def get_status(self) -> Dict[str, Any]:
        """Collect system metrics for the API response."""
        status = {
            "timestamp": time.time(),
            "load_avg_5min": self._get_load_average_5min(),
            "memory_percent": self._get_memory_percent(),
            "gpus": self._get_gpu_utilisation(),
        }
        offload = self._get_deepspeed_offload_usage()
        if offload:
            status["deepspeed_offload"] = offload
        return status

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
        mac_memory: Optional[List[Optional[float]]] = None
        nvidia_fallback: Optional[List[Dict[str, Optional[float]]]] = None

        if backend == "mps":
            mac_utilisation = self._get_macos_gpu_utilisation()
            mac_memory = self._get_mps_memory_percent()

        for position, device in enumerate(devices):
            index = device.get("index")
            name = device.get("name") or f"GPU {index if index is not None else '?'}"
            utilisation: Optional[float] = None
            memory_percent: Optional[float] = None

            if backend == "cuda" and index is not None and torch is not None and hasattr(torch.cuda, "utilization"):
                try:
                    if torch.cuda.is_available():  # type: ignore[attr-defined]
                        utilisation = float(torch.cuda.utilization(index))  # type: ignore[attr-defined]
                except Exception:
                    logger.debug("Failed to read CUDA utilisation for device %s", index, exc_info=True)
                    utilisation = None
                memory_percent = self._get_cuda_memory_percent(index)
            if utilisation is None and backend == "mps" and mac_utilisation:
                target_idx: Optional[int] = None
                if isinstance(index, int) and 0 <= index < len(mac_utilisation):
                    target_idx = index
                elif 0 <= position < len(mac_utilisation):
                    target_idx = position
                if target_idx is not None:
                    utilisation = mac_utilisation[target_idx]
                if mac_memory:
                    mem_idx: Optional[int] = None
                    if isinstance(index, int) and 0 <= index < len(mac_memory):
                        mem_idx = index
                    elif 0 <= position < len(mac_memory):
                        mem_idx = position
                    if mem_idx is not None:
                        memory_percent = mac_memory[mem_idx]
            if utilisation is None and backend == "cuda":
                if nvidia_fallback is None:
                    nvidia_fallback = self._get_nvidia_gpu_stats()
                if nvidia_fallback:
                    target_idx: Optional[int] = None
                    if isinstance(index, int) and 0 <= index < len(nvidia_fallback):
                        target_idx = index
                    elif 0 <= position < len(nvidia_fallback):
                        target_idx = position
                    if target_idx is not None:
                        fallback_entry = nvidia_fallback[target_idx]
                        if utilisation is None:
                            utilisation = fallback_entry.get("utilization_percent")
                        if memory_percent is None:
                            memory_percent = fallback_entry.get("memory_percent")

            results.append(
                {
                    "index": index,
                    "name": name,
                    "backend": backend,
                    "utilization_percent": round(utilisation, 1) if utilisation is not None else None,
                    "memory_percent": round(memory_percent, 1) if memory_percent is not None else None,
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

    def _get_deepspeed_offload_usage(self) -> Optional[Dict[str, Any]]:
        try:
            from simpletuner.simpletuner_sdk.api_state import APIState
        except Exception:
            return None

        config = APIState.get_state("training_config")
        if not isinstance(config, dict):
            return None

        ds_config_raw = (
            config.get("deepspeed_config")
            or config.get("--deepspeed_config")
            or config.get("hf_deepspeed_config")
            or config.get("--hf_deepspeed_config")
        )

        ds_config = self._coerce_mapping(ds_config_raw)
        if not isinstance(ds_config, dict):
            return None

        zero_cfg = ds_config.get("zero_optimization")
        if not isinstance(zero_cfg, dict):
            return None

        offload_param_cfg = zero_cfg.get("offload_param")
        if not isinstance(offload_param_cfg, dict):
            return None

        device = str(offload_param_cfg.get("device", "")).lower() or None
        stage = self._coerce_int(zero_cfg.get("stage"))

        explicit_path = (
            config.get("offload_param_path")
            or config.get("--offload_param_path")
            or offload_param_cfg.get("nvme_path")
            or offload_param_cfg.get("path")
        )
        offload_path = self._normalize_path(explicit_path)
        if not offload_path:
            return None

        size_bytes, file_count = self._measure_directory(offload_path)

        return {
            "path": offload_path,
            "size_bytes": size_bytes,
            "file_count": file_count,
            "device": device,
            "stage": stage,
        }

    def _normalize_path(self, path_candidate: Any) -> Optional[str]:
        if not path_candidate or isinstance(path_candidate, (bool, int, float)):
            return None
        if isinstance(path_candidate, str):
            candidate = path_candidate.strip()
            if not candidate or candidate.lower() == "none":
                return None
            try:
                return os.path.abspath(os.path.expanduser(candidate))
            except Exception:
                return None
        return None

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_mapping(self, raw: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return None
            try:
                return json.loads(text)
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return None
        return None

    def _measure_directory(self, path: str) -> Tuple[Optional[int], Optional[int]]:
        cache = self._offload_cache
        now = time.time()
        cache_valid = (
            isinstance(cache, dict)
            and cache.get("path") == path
            and isinstance(cache.get("timestamp"), (int, float))
            and now - cache["timestamp"] < 5
        )
        if cache_valid:
            return cache.get("size_bytes"), cache.get("file_count")

        total_size = 0
        file_count = 0
        try:
            if not os.path.isdir(path):
                size = None
                count = None
            else:
                for root, _dirs, files in os.walk(path):
                    for filename in files:
                        full_path = os.path.join(root, filename)
                        try:
                            stat = os.stat(full_path)
                        except OSError:
                            continue
                        total_size += stat.st_size
                        file_count += 1
                size = total_size
                count = file_count
        except Exception:
            size = None
            count = None

        self._offload_cache = {
            "path": path,
            "size_bytes": size,
            "file_count": count,
            "timestamp": now,
        }
        return size, count

    def _get_nvidia_gpu_stats(self) -> Optional[List[Dict[str, Optional[float]]]]:
        if platform.system() == "Darwin":
            return None

        nvml_stats = self._get_nvml_gpu_stats()
        if nvml_stats:
            return nvml_stats

        return self._get_nvidia_smi_stats()

    def _get_nvml_gpu_stats(self) -> Optional[List[Dict[str, Optional[float]]]]:
        if pynvml is None:
            return None

        initialised_here = False
        try:
            pynvml.nvmlInit()  # type: ignore[attr-defined]
            initialised_here = True
        except Exception as exc:  # pragma: no cover - NVML optional
            try:
                already_init_cls = getattr(pynvml, "NVMLError_AlreadyInitialized", None)  # type: ignore[attr-defined]
                if already_init_cls and isinstance(exc, already_init_cls):
                    initialised_here = False
                else:
                    logger.debug("Unable to initialise NVML: %s", exc, exc_info=True)
                    return None
            except Exception:
                logger.debug("Unable to initialise NVML: %s", exc, exc_info=True)
                return None

        try:
            try:
                device_count = pynvml.nvmlDeviceGetCount()  # type: ignore[attr-defined]
            except Exception as exc:
                logger.debug("Failed to query NVML device count: %s", exc, exc_info=True)
                return None

            if device_count <= 0:
                return None

            stats: List[Dict[str, Optional[float]]] = []
            for index in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)  # type: ignore[attr-defined]
                except Exception as exc:
                    logger.debug("Failed to acquire NVML handle for device %s: %s", index, exc, exc_info=True)
                    stats.append({"utilization_percent": None, "memory_percent": None})
                    continue

                utilisation_value: Optional[float] = None
                memory_percent: Optional[float] = None

                try:
                    utilisation = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[attr-defined]
                    gpu_util = getattr(utilisation, "gpu", None)
                    if gpu_util is not None:
                        utilisation_value = float(gpu_util)
                except Exception as exc:
                    logger.debug("Failed to read NVML utilisation for device %s: %s", index, exc, exc_info=True)

                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[attr-defined]
                    total_raw = getattr(mem_info, "total", None)
                    used_raw = getattr(mem_info, "used", None)
                    if total_raw not in (None, 0):
                        total = float(total_raw)
                        used = float(used_raw or 0.0)
                        if total > 0:
                            memory_percent = (used / total) * 100.0
                except Exception as exc:
                    logger.debug("Failed to read NVML memory info for device %s: %s", index, exc, exc_info=True)

                stats.append(
                    {
                        "utilization_percent": round(utilisation_value, 1) if utilisation_value is not None else None,
                        "memory_percent": round(memory_percent, 1) if memory_percent is not None else None,
                    }
                )

            return stats or None
        finally:
            if initialised_here:
                try:
                    pynvml.nvmlShutdown()  # type: ignore[attr-defined]
                except Exception as exc:
                    logger.debug("Failed to shutdown NVML cleanly: %s", exc, exc_info=True)

    def _get_nvidia_smi_stats(self) -> Optional[List[Dict[str, Optional[float]]]]:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
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

        stats: List[Dict[str, Optional[float]]] = []
        for line in lines:
            text = line.strip()
            if not text:
                stats.append({"utilization_percent": None, "memory_percent": None})
                continue
            parts = [part.strip() for part in text.split(",")]
            if len(parts) < 3:
                logger.debug("Discarding unexpected nvidia-smi output: %s", text)
                stats.append({"utilization_percent": None, "memory_percent": None})
                continue
            util_raw, mem_used_raw, mem_total_raw = parts[:3]
            util_val: Optional[float]
            mem_percent: Optional[float]
            try:
                util_val = round(float(util_raw), 1)
            except ValueError:
                util_val = None
            try:
                mem_used = float(mem_used_raw)
                mem_total = float(mem_total_raw)
                if mem_total > 0:
                    mem_percent = round((mem_used / mem_total) * 100.0, 1)
                else:
                    mem_percent = None
            except ValueError:
                mem_percent = None
            stats.append({"utilization_percent": util_val, "memory_percent": mem_percent})

        return stats or None

    def _get_cuda_memory_percent(self, index: int) -> Optional[float]:
        if torch is None or not torch.cuda.is_available():  # type: ignore[attr-defined]
            return None
        if not hasattr(torch.cuda, "mem_get_info"):
            return None
        try:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)  # type: ignore[misc]
            except TypeError:
                with torch.cuda.device(index):
                    free_bytes, total_bytes = torch.cuda.mem_get_info()  # type: ignore[call-arg]
        except Exception:
            logger.debug("Failed to read CUDA memory info for device %s", index, exc_info=True)
            return None
        if not total_bytes:
            return None
        used = total_bytes - free_bytes
        if used < 0:
            used = 0
        try:
            percent = (used / total_bytes) * 100.0
        except Exception:
            return None
        return round(float(percent), 1)

    def _get_mps_memory_percent(self) -> Optional[List[Optional[float]]]:
        if torch is None:
            return None
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            return None
        driver_alloc = getattr(torch.mps, "driver_allocated_memory", None)
        driver_total = getattr(torch.mps, "driver_total_memory", None)
        if not callable(driver_alloc) or not callable(driver_total):
            return None
        try:
            allocated = float(driver_alloc())
            total = float(driver_total())
        except Exception:
            logger.debug("Unable to query MPS memory statistics", exc_info=True)
            return None
        if total <= 0:
            return None
        percent = round((allocated / total) * 100.0, 1)
        return [percent]


__all__ = ["SystemStatusService"]
