from __future__ import annotations

import json
import logging
import os
import platform
import plistlib
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import psutil

from simpletuner.helpers.training.reporting import report_to_tokens

logger = logging.getLogger(__name__)

NATIVE_SYSTEM_METRIC_TRACKERS = {"wandb"}


def should_collect_manual_system_metrics(report_to: Any) -> bool:
    return any(token not in NATIVE_SYSTEM_METRIC_TRACKERS and token != "none" for token in report_to_tokens(report_to))


def _coerce_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def _gib(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value) / (1024**3), 3)


def _put_metric(metrics: dict[str, float], name: str, value: Any, *, digits: int = 3) -> None:
    number = _coerce_number(value)
    if number is None:
        return
    metrics[name] = round(number, digits)


def _first_key_value(entry: dict[str, Any], key_fragments: Iterable[tuple[str, ...]]) -> Optional[float]:
    lowered = [(key.lower(), value) for key, value in entry.items()]
    for fragments in key_fragments:
        for key, value in lowered:
            if all(fragment in key for fragment in fragments):
                return _coerce_number(value)
    return None


class SystemMetricsSampler:
    def __init__(
        self,
        *,
        output_dir: str | os.PathLike[str],
        min_interval_seconds: float = 5.0,
        time_source: Any = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.min_interval_seconds = float(min_interval_seconds)
        self._time_source = time_source or time.monotonic
        self._last_sample_time: Optional[float] = None
        self._last_net_counters: Optional[Any] = None
        self._last_net_time: Optional[float] = None

    def sample(self, *, force: bool = False) -> dict[str, float]:
        now = float(self._time_source())
        if not force and self._last_sample_time is not None and now - self._last_sample_time < self.min_interval_seconds:
            return {}
        self._last_sample_time = now

        metrics: dict[str, float] = {}
        self._sample_system(metrics, now)
        self._sample_gpu(metrics)
        return metrics

    def _sample_system(self, metrics: dict[str, float], now: float) -> None:
        cpu_percent = psutil.cpu_percent(interval=None)
        _put_metric(metrics, "system/cpu_percent", cpu_percent, digits=1)

        memory = psutil.virtual_memory()
        _put_metric(metrics, "system/memory_percent", memory.percent, digits=1)
        _put_metric(metrics, "system/memory_available_gb", _gib(float(memory.available)), digits=3)

        disk = shutil.disk_usage(self.output_dir)
        _put_metric(metrics, "system/disk_free_gb", _gib(float(disk.free)), digits=3)
        if disk.total > 0:
            _put_metric(metrics, "system/disk_percent", (disk.used / disk.total) * 100.0, digits=1)

        counters = psutil.net_io_counters()
        if self._last_net_counters is not None and self._last_net_time is not None:
            elapsed = now - self._last_net_time
            if elapsed > 0:
                sent_delta = max(0, counters.bytes_sent - self._last_net_counters.bytes_sent)
                recv_delta = max(0, counters.bytes_recv - self._last_net_counters.bytes_recv)
                _put_metric(metrics, "system/network_sent_mbps", (sent_delta * 8) / elapsed / 1_000_000, digits=3)
                _put_metric(metrics, "system/network_recv_mbps", (recv_delta * 8) / elapsed / 1_000_000, digits=3)
        self._last_net_counters = counters
        self._last_net_time = now

    def _sample_gpu(self, metrics: dict[str, float]) -> None:
        try:
            import torch
        except ImportError:
            return

        if torch.cuda.is_available():
            if bool(getattr(torch.version, "hip", None)):
                self._sample_rocm(metrics)
            else:
                self._sample_cuda(metrics)
            return

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            self._sample_mps(metrics, torch)

    def _sample_cuda(self, metrics: dict[str, float]) -> None:
        if self._sample_nvml(metrics):
            return
        self._sample_nvidia_smi(metrics)

    def _sample_nvml(self, metrics: dict[str, float]) -> bool:
        try:
            import pynvml
        except ImportError:
            return False

        initialized_here = False
        try:
            pynvml.nvmlInit()
            initialized_here = True
        except Exception as exc:
            already_initialized = getattr(pynvml, "NVMLError_AlreadyInitialized", None)
            if already_initialized is None or not isinstance(exc, already_initialized):
                logger.debug("Unable to initialise NVML for system metrics: %s", exc, exc_info=True)
                return False

        initial_metric_count = len(metrics)
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for index in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                prefix = f"system/gpu/{index}"
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    _put_metric(metrics, f"{prefix}/utilization_percent", getattr(util, "gpu", None), digits=1)
                    _put_metric(metrics, f"{prefix}/memory_utilization_percent", getattr(util, "memory", None), digits=1)
                except Exception:
                    logger.debug("Unable to read NVML utilisation for GPU %s", index, exc_info=True)
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    _put_metric(metrics, f"{prefix}/memory_used_gb", _gib(float(mem.used)), digits=3)
                    _put_metric(metrics, f"{prefix}/memory_total_gb", _gib(float(mem.total)), digits=3)
                    if mem.total:
                        _put_metric(metrics, f"{prefix}/memory_percent", (mem.used / mem.total) * 100.0, digits=1)
                except Exception:
                    logger.debug("Unable to read NVML memory for GPU %s", index, exc_info=True)
                try:
                    _put_metric(
                        metrics,
                        f"{prefix}/temperature_celsius",
                        pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                        digits=1,
                    )
                except Exception:
                    logger.debug("Unable to read NVML temperature for GPU %s", index, exc_info=True)
                try:
                    _put_metric(metrics, f"{prefix}/fan_speed_percent", pynvml.nvmlDeviceGetFanSpeed(handle), digits=1)
                except Exception:
                    logger.debug("Unable to read NVML fan speed for GPU %s", index, exc_info=True)
                try:
                    _put_metric(metrics, f"{prefix}/power_usage_watts", pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
                except Exception:
                    logger.debug("Unable to read NVML power usage for GPU %s", index, exc_info=True)
            return len(metrics) > initial_metric_count
        finally:
            if initialized_here:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    logger.debug("Unable to shutdown NVML after system metrics sampling", exc_info=True)

    def _sample_nvidia_smi(self, metrics: dict[str, float]) -> None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,fan.speed,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            logger.debug("Unable to query nvidia-smi for system metrics: %s", exc, exc_info=True)
            return

        for index, line in enumerate(completed.stdout.strip().splitlines()):
            values = [value.strip() for value in line.split(",")]
            if len(values) < 6:
                continue
            util, mem_used_mib, mem_total_mib, temperature, fan_speed, power = values[:6]
            prefix = f"system/gpu/{index}"
            _put_metric(metrics, f"{prefix}/utilization_percent", util, digits=1)
            mem_used = _coerce_number(mem_used_mib)
            mem_total = _coerce_number(mem_total_mib)
            if mem_used is not None:
                _put_metric(metrics, f"{prefix}/memory_used_gb", mem_used / 1024.0, digits=3)
            if mem_total is not None:
                _put_metric(metrics, f"{prefix}/memory_total_gb", mem_total / 1024.0, digits=3)
            if mem_used is not None and mem_total:
                _put_metric(metrics, f"{prefix}/memory_percent", (mem_used / mem_total) * 100.0, digits=1)
            _put_metric(metrics, f"{prefix}/temperature_celsius", temperature, digits=1)
            _put_metric(metrics, f"{prefix}/fan_speed_percent", fan_speed, digits=1)
            _put_metric(metrics, f"{prefix}/power_usage_watts", power)

    def _sample_rocm(self, metrics: dict[str, float]) -> None:
        rocm_smi = shutil.which("rocm-smi")
        if not rocm_smi:
            return
        try:
            completed = subprocess.run(
                [rocm_smi, "--showuse", "--showmemuse", "--showtemp", "--showfan", "--showpower", "--json"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            logger.debug("Unable to query rocm-smi for system metrics: %s", exc, exc_info=True)
            return

        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            logger.debug("Unable to parse rocm-smi JSON for system metrics: %s", exc, exc_info=True)
            return
        if not isinstance(payload, dict):
            return

        for position, entry in enumerate(payload.values()):
            if not isinstance(entry, dict):
                continue
            prefix = f"system/gpu/{position}"
            _put_metric(metrics, f"{prefix}/utilization_percent", _first_key_value(entry, [("gpu", "use")]), digits=1)
            memory_percent = _first_key_value(entry, [("vram", "%"), ("memory", "%"), ("mem", "%")])
            _put_metric(metrics, f"{prefix}/memory_percent", memory_percent, digits=1)
            _put_metric(
                metrics, f"{prefix}/temperature_celsius", _first_key_value(entry, [("temperature",), ("temp",)]), digits=1
            )
            _put_metric(
                metrics, f"{prefix}/fan_speed_percent", _first_key_value(entry, [("fan", "%"), ("fan", "speed")]), digits=1
            )
            _put_metric(metrics, f"{prefix}/power_usage_watts", _first_key_value(entry, [("power",), ("watt",)]))

    def _sample_mps(self, metrics: dict[str, float], torch: Any) -> None:
        prefix = "system/gpu/0"
        utilization = self._mps_utilization()
        _put_metric(metrics, f"{prefix}/utilization_percent", utilization, digits=1)

        driver_alloc = getattr(torch.mps, "driver_allocated_memory", None)
        driver_total = getattr(torch.mps, "driver_total_memory", None)
        if callable(driver_alloc) and callable(driver_total):
            try:
                allocated = float(driver_alloc())
                total = float(driver_total())
            except Exception:
                logger.debug("Unable to query MPS memory statistics for system metrics", exc_info=True)
            else:
                _put_metric(metrics, f"{prefix}/memory_used_gb", _gib(allocated), digits=3)
                _put_metric(metrics, f"{prefix}/memory_total_gb", _gib(total), digits=3)
                if total > 0:
                    _put_metric(metrics, f"{prefix}/memory_percent", (allocated / total) * 100.0, digits=1)

    def _mps_utilization(self) -> Optional[float]:
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
            data = plistlib.loads(completed.stdout)
        except (FileNotFoundError, subprocess.SubprocessError, plistlib.InvalidFileException, ValueError) as exc:
            logger.debug("Unable to query MPS utilisation for system metrics: %s", exc, exc_info=True)
            return None
        if not isinstance(data, list):
            return None
        for entry in data:
            if not isinstance(entry, dict):
                continue
            perf = entry.get("PerformanceStatistics")
            if not isinstance(perf, dict):
                continue
            value = _coerce_number(perf.get("Device Utilization %"))
            if value is not None:
                return value
        return None


def log_system_metrics_to_trackers(trackers: Iterable[Any], metrics: dict[str, float], *, step: int) -> None:
    if not metrics:
        return
    for tracker in trackers:
        name = str(getattr(tracker, "name", "") or "").strip().lower()
        if not name or name in NATIVE_SYSTEM_METRIC_TRACKERS:
            continue
        log = getattr(tracker, "log", None)
        if not callable(log):
            logger.warning("Tracker '%s' cannot receive manual system metrics because it has no log method.", name)
            continue
        try:
            log(metrics, step=step)
        except Exception as exc:
            logger.warning("Failed to log manual system metrics to tracker '%s': %s", name, exc)
