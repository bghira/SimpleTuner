"""Lightweight runtime counters for RamTorch transfer policy profiling."""

from __future__ import annotations

import atexit
import json
import os
import statistics
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import torch

_LOCK = threading.Lock()
_DUMPED = False
_OUTSTANDING: dict[tuple[str, str, str], dict[str, Any]] = {}
_STEP_DURATIONS: list[dict[str, float]] = []


def _initial_stats() -> dict[str, Any]:
    return {
        "created_at_unix": time.time(),
        "counters": {
            "prefetch_attempts": 0,
            "prefetch_enqueued": 0,
            "prefetch_skipped_existing": 0,
            "prefetch_skipped_policy": 0,
            "prefetch_skipped_non_cuda": 0,
            "prefetch_consumed": 0,
            "prefetch_stale": 0,
            "prefetch_unused": 0,
            "fallback_forward_transfers": 0,
            "hook_prefetch_calls": 0,
            "hook_prefetch_successes": 0,
            "sync_hook_calls": 0,
            "bytes_prefetch_attempted": 0,
            "bytes_prefetched": 0,
            "bytes_prefetch_consumed": 0,
            "bytes_prefetch_stale": 0,
            "bytes_prefetch_unused": 0,
            "bytes_fallback_forward_transfer": 0,
        },
        "devices": {},
    }


_STATS = _initial_stats()


def reset_for_new_run() -> None:
    global _DUMPED, _STATS
    with _LOCK:
        _DUMPED = False
        _STATS = _initial_stats()
        _OUTSTANDING.clear()
        _STEP_DURATIONS.clear()


def profile_enabled() -> bool:
    return bool(os.environ.get("SIMPLETUNER_RAMTORCH_PROFILE_PATH")) or (
        os.environ.get("SIMPLETUNER_RAMTORCH_PROFILE", "0") == "1"
    )


def policy() -> str:
    return os.environ.get("SIMPLETUNER_RAMTORCH_PREFETCH_POLICY", "current").strip().lower() or "current"


def prefetch_hooks_allowed() -> bool:
    return policy() not in {"0", "false", "off", "sync", "sync-only", "disabled"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _device_name(device: torch.device | str | int | None) -> str:
    try:
        return str(_as_device(device)) if device is not None else "cuda"
    except Exception:
        return str(device)


def _as_device(device: torch.device | str | int | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int) and torch.cuda.is_available():
        return torch.device("cuda", device)
    if device is None:
        return torch.device("cuda")
    return torch.device(device)


def _token(owner: str, device: torch.device | str | int | None, key: Any) -> tuple[str, str, str]:
    return (owner, _device_name(device), repr(key))


def tensor_bytes(tensors) -> int:
    total = 0
    for tensor in tensors:
        if tensor is not None:
            total += int(tensor.numel()) * int(tensor.element_size())
    return total


def _mem_get_info(device: torch.device | str | int | None) -> tuple[int | None, int | None]:
    if not torch.cuda.is_available():
        return None, None
    device_obj = _as_device(device)
    try:
        with torch.cuda.device(device_obj):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        return int(free_bytes), int(total_bytes)
    except Exception:
        return None, None


def _device_stats(device: torch.device | str | int | None) -> dict[str, Any]:
    return _STATS["devices"].setdefault(_device_name(device), {})


def _record_free_sample(
    device: torch.device | str | int | None,
    event: str,
    free_bytes: int | None,
    total_bytes: int | None,
) -> None:
    if free_bytes is None or total_bytes in (None, 0):
        return
    device_stats = _device_stats(device)
    prefix = f"free_vram_{event}"
    ratio = free_bytes / total_bytes
    count_key = f"{prefix}_sample_count"
    min_key = f"{prefix}_bytes_min"
    max_key = f"{prefix}_bytes_max"
    last_key = f"{prefix}_bytes_last"
    ratio_min_key = f"{prefix}_ratio_min"
    ratio_last_key = f"{prefix}_ratio_last"
    device_stats[count_key] = int(device_stats.get(count_key, 0)) + 1
    device_stats[min_key] = min(int(device_stats.get(min_key, free_bytes)), free_bytes)
    device_stats[max_key] = max(int(device_stats.get(max_key, free_bytes)), free_bytes)
    device_stats[last_key] = free_bytes
    device_stats[ratio_min_key] = min(float(device_stats.get(ratio_min_key, ratio)), ratio)
    device_stats[ratio_last_key] = ratio
    device_stats["total_vram_bytes"] = total_bytes


def should_prefetch(
    owner: str,
    key: Any,
    device: torch.device | str | int | None,
    tensors,
) -> tuple[bool, int, int | None, int | None]:
    bytes_to_prefetch = tensor_bytes(tensors)
    free_bytes, total_bytes = _mem_get_info(device)
    current_policy = policy()

    with _LOCK:
        counters = _STATS["counters"]
        counters["prefetch_attempts"] += 1
        counters["bytes_prefetch_attempted"] += bytes_to_prefetch
        _record_free_sample(device, "before_forward_prefetch", free_bytes, total_bytes)

        device_obj = _as_device(device)
        if device_obj.type != "cuda":
            counters["prefetch_skipped_non_cuda"] += 1
            return False, bytes_to_prefetch, free_bytes, total_bytes

        if current_policy in {"adaptive", "free-vram", "free_vram"}:
            min_free_ratio = _env_float("SIMPLETUNER_RAMTORCH_PREFETCH_MIN_FREE_RATIO", 0.20)
            min_bytes = _env_int("SIMPLETUNER_RAMTORCH_PREFETCH_MIN_BYTES", 0)
            free_ratio = (free_bytes / total_bytes) if free_bytes is not None and total_bytes else None
            if free_ratio is None or free_ratio < min_free_ratio or bytes_to_prefetch < min_bytes:
                counters["prefetch_skipped_policy"] += 1
                return False, bytes_to_prefetch, free_bytes, total_bytes

    return True, bytes_to_prefetch, free_bytes, total_bytes


def record_prefetch_existing(owner: str, key: Any, device: torch.device | str | int | None, bytes_to_prefetch: int) -> None:
    with _LOCK:
        _STATS["counters"]["prefetch_skipped_existing"] += 1


def record_prefetch_enqueued(
    owner: str,
    key: Any,
    device: torch.device | str | int | None,
    bytes_to_prefetch: int,
    *,
    free_before: int | None = None,
    total_before: int | None = None,
) -> None:
    free_after, total_after = _mem_get_info(device)
    with _LOCK:
        counters = _STATS["counters"]
        counters["prefetch_enqueued"] += 1
        counters["bytes_prefetched"] += bytes_to_prefetch
        token = _token(owner, device, key)
        _OUTSTANDING[token] = {
            "bytes": bytes_to_prefetch,
            "device": _device_name(device),
            "owner": owner,
            "created_at_unix": time.time(),
        }
        _record_free_sample(device, "after_forward_prefetch", free_after, total_after)


def record_prefetch_consumed(owner: str, key: Any, device: torch.device | str | int | None) -> None:
    with _LOCK:
        counters = _STATS["counters"]
        counters["prefetch_consumed"] += 1
        entry = _OUTSTANDING.pop(_token(owner, device, key), None)
        if entry is not None:
            counters["bytes_prefetch_consumed"] += int(entry.get("bytes", 0))


def record_prefetch_stale(owner: str, key: Any, device: torch.device | str | int | None) -> None:
    with _LOCK:
        counters = _STATS["counters"]
        counters["prefetch_stale"] += 1
        entry = _OUTSTANDING.pop(_token(owner, device, key), None)
        if entry is not None:
            counters["bytes_prefetch_stale"] += int(entry.get("bytes", 0))


def record_fallback_forward_transfer(device: torch.device | str | int | None, tensors) -> None:
    bytes_transferred = tensor_bytes(tensors)
    with _LOCK:
        counters = _STATS["counters"]
        counters["fallback_forward_transfers"] += 1
        counters["bytes_fallback_forward_transfer"] += bytes_transferred


def record_hook_prefetch(success: bool) -> None:
    with _LOCK:
        counters = _STATS["counters"]
        counters["hook_prefetch_calls"] += 1
        if success:
            counters["hook_prefetch_successes"] += 1


def record_sync_hook() -> None:
    with _LOCK:
        _STATS["counters"]["sync_hook_calls"] += 1


def record_train_step(global_step: float | int, duration_seconds: float | None) -> None:
    if duration_seconds is None:
        return
    with _LOCK:
        _STEP_DURATIONS.append(
            {
                "global_step": float(global_step),
                "duration_seconds": float(duration_seconds),
            }
        )


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def _step_duration_summary() -> dict[str, Any]:
    warmup_steps = max(_env_int("SIMPLETUNER_RAMTORCH_PROFILE_WARMUP_STEPS", 5), 0)
    durations = [entry["duration_seconds"] for entry in _STEP_DURATIONS]
    measured = durations[warmup_steps:] if len(durations) > warmup_steps else []
    measured_sorted = sorted(measured)
    summary: dict[str, Any] = {
        "sample_count": len(durations),
        "warmup_steps_excluded": warmup_steps,
        "measured_sample_count": len(measured),
    }
    if measured:
        summary.update(
            {
                "mean_seconds": statistics.fmean(measured),
                "p50_seconds": _percentile(measured_sorted, 0.50),
                "p95_seconds": _percentile(measured_sorted, 0.95),
                "min_seconds": measured_sorted[0],
                "max_seconds": measured_sorted[-1],
            }
        )
    if os.environ.get("SIMPLETUNER_RAMTORCH_PROFILE_INCLUDE_STEP_SAMPLES", "0") == "1":
        summary["samples"] = list(_STEP_DURATIONS)
    return summary


def _flush_unused_prefetches_locked() -> None:
    if not _OUTSTANDING:
        return
    counters = _STATS["counters"]
    counters["prefetch_unused"] += len(_OUTSTANDING)
    counters["bytes_prefetch_unused"] += sum(int(entry.get("bytes", 0)) for entry in _OUTSTANDING.values())
    _OUTSTANDING.clear()


def _has_activity_locked() -> bool:
    if _STEP_DURATIONS:
        return True
    return any(int(value) for value in _STATS["counters"].values())


def _peak_memory() -> dict[str, dict[str, int]]:
    peaks: dict[str, dict[str, int]] = {}
    if not torch.cuda.is_available():
        return peaks
    try:
        device_count = torch.cuda.device_count()
    except Exception:
        return peaks
    for index in range(device_count):
        device = torch.device("cuda", index)
        try:
            peaks[str(device)] = {
                "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        except Exception:
            continue
    return peaks


def snapshot() -> dict[str, Any]:
    with _LOCK:
        _flush_unused_prefetches_locked()
        data = json.loads(json.dumps(_STATS))
        data["policy"] = {
            "prefetch_policy": policy(),
            "min_free_ratio": _env_float("SIMPLETUNER_RAMTORCH_PREFETCH_MIN_FREE_RATIO", 0.20),
            "min_bytes": _env_int("SIMPLETUNER_RAMTORCH_PREFETCH_MIN_BYTES", 0),
            "warmup_steps": _env_int("SIMPLETUNER_RAMTORCH_PROFILE_WARMUP_STEPS", 5),
        }
        data["train_step_durations"] = _step_duration_summary()
        data["peak_memory"] = _peak_memory()
        data["finished_at_unix"] = time.time()
        return data


def dump_profile_from_env() -> None:
    global _DUMPED
    path = os.environ.get("SIMPLETUNER_RAMTORCH_PROFILE_PATH")
    should_print = os.environ.get("SIMPLETUNER_RAMTORCH_PROFILE_PRINT", "0") == "1"
    if not path and not should_print:
        return

    with _LOCK:
        if _DUMPED:
            return
        if not _has_activity_locked() and os.environ.get("SIMPLETUNER_RAMTORCH_PROFILE_WRITE_EMPTY", "0") != "1":
            return
        _DUMPED = True

    data = snapshot()
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=str(target.parent), delete=False) as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temp_name = handle.name
        Path(temp_name).replace(target)

    if should_print:
        print("RAMTORCH_PROFILE " + json.dumps(data, sort_keys=True), flush=True)


atexit.register(dump_profile_from_env)
