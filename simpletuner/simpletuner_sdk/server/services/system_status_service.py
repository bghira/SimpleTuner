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
        nvidia_fallback: Optional[List[Optional[float]]] = None
        rocm_fallback: Optional[List[Optional[float]]] = None

        if backend == "mps":
            mac_utilisation = self._get_macos_gpu_utilisation()

        for position, device in enumerate(devices):
            index = device.get("index")
            name = device.get("name") or f"GPU {index if index is not None else '?'}"
            utilisation: Optional[float] = None

            if backend == "cuda" and index is not None and torch is not None and hasattr(torch.cuda, "utilization"):
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
            if utilisation is None and backend == "rocm":
                if rocm_fallback is None:
                    rocm_fallback = self._get_rocm_gpu_utilisation()
                if rocm_fallback:
                    target_idx = None
                    if isinstance(index, int) and 0 <= index < len(rocm_fallback):
                        target_idx = index
                    elif 0 <= position < len(rocm_fallback):
                        target_idx = position
                    if target_idx is not None:
                        utilisation = rocm_fallback[target_idx]

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

    def _get_rocm_gpu_utilisation(self) -> Optional[List[Optional[float]]]:
        rocm_smi = shutil.which("rocm-smi")
        if not rocm_smi:
            for candidate in ("/opt/rocm/bin/rocm-smi", "/usr/bin/rocm-smi"):
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    rocm_smi = candidate
                    break
        if not rocm_smi:
            return None

        commands = [
            [rocm_smi, "--showuse", "--json"],
            [rocm_smi, "--showuse"],
        ]

        for command in commands:
            try:
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
            except (FileNotFoundError, subprocess.SubprocessError) as exc:
                logger.debug("Unable to query GPU utilisation via rocm-smi (%s): %s", " ".join(command[1:]), exc, exc_info=True)
                continue

            output = (completed.stdout or "").strip()
            if not output:
                continue

            if "--json" in command:
                try:
                    data = json.loads(output)
                except json.JSONDecodeError as exc:
                    logger.debug("Failed to parse rocm-smi JSON output: %s", exc, exc_info=True)
                    continue

                if isinstance(data, dict):
                    values: Dict[int, Optional[float]] = {}
                    for key, entry in data.items():
                        if not isinstance(entry, dict):
                            continue
                        gpu_idx = self._coerce_int(key.lstrip("card")) if isinstance(key, str) else self._coerce_int(key)
                        raw_value = (
                            entry.get("GPU use (%)")
                            or entry.get("GPU use (%) (avg)")
                            or entry.get("GPU use (%) (average)")
                            or entry.get("GPU use (%) (peak)")
                            or entry.get("GPU use (%) (current)")
                        )
                        values[gpu_idx] = self._coerce_percent(raw_value)
                    if values:
                        ordered = [values.get(idx, None) for idx in sorted(values.keys())]
                        return ordered or None
                continue

            parsed_values = self._parse_rocm_smi_text(output)
            if parsed_values:
                return parsed_values

        return None

    def _parse_rocm_smi_text(self, output: str) -> Optional[List[Optional[float]]]:
        pattern = re.compile(r"GPU\s*\[\s*(\d+)\s*\].*?GPU use.*?:\s*([0-9]+(?:\.[0-9]+)?)")
        matches = pattern.findall(output)
        values: Dict[int, Optional[float]] = {}
        if matches:
            for gpu, raw in matches:
                idx = self._coerce_int(gpu)
                if idx is None or idx in values:
                    continue
                values[idx] = self._coerce_percent(raw)
        else:
            lines = output.splitlines()
            header_index = None
            headers = []
            for line in lines:
                if "GPU use" in line and header_index is None:
                    headers = [token for token in line.strip().split() if token]
                    if headers:
                        try:
                            header_index = headers.index("use") - 1 if "use" in headers else None
                        except ValueError:
                            header_index = None
                    continue
                if not line or not line.strip():
                    continue
                tokens = [token for token in line.strip().split() if token]
                if len(tokens) < 2:
                    continue
                gpu_token = tokens[0]
                gpu_idx = self._coerce_int(gpu_token.strip("GPU[]"))
                if gpu_idx is None or gpu_idx in values:
                    continue
                percent_value = None
                for token in tokens[1:]:
                    maybe_percent = self._coerce_percent(token)
                    if maybe_percent is not None:
                        percent_value = maybe_percent
                        break
                values[gpu_idx] = percent_value

        if not values:
            return None

        ordered = [values.get(idx, None) for idx in sorted(values.keys())]
        return ordered or None

    @staticmethod
    def _coerce_percent(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return round(float(value), 1)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("%"):
                text = text[:-1]
            try:
                return round(float(text), 1)
            except ValueError:
                return None
        return None

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
