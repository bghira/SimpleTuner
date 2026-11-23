"""Read system snapshots for the Metrics page."""

from datetime import datetime, timedelta, timezone

from simpletuner.worker_protocol import WORKER_HEARTBEAT_SECONDS

from ..models.worker import Worker, WorkerStatus
from .cloud.auth.models import User
from .worker_repository import get_worker_repository

_STALE_AFTER = timedelta(seconds=3 * WORKER_HEARTBEAT_SECONDS)
_GPU_FIELDS = {
    "temperature_celsius": "temperature_celsius",
    "fan_speed_percent": "fan_speed_percent",
    "utilization_percent": "gpu_utilization_percent",
    "memory_utilization_percent": "memory_utilization_percent",
    "memory_percent": "memory_used_percent",
    "power_usage_watts": "power_usage_watts",
}
_GPU_MEMORY_FIELDS = {
    "memory_used_gb": "memory_used_bytes",
    "memory_total_gb": "memory_total_bytes",
}


async def visible_system_workers(user: User) -> list[Worker]:
    user_id = None if user.has_permission("admin.workers") else user.id
    return await get_worker_repository().list_workers(user_id=user_id, limit=-1)


def worker_system_status(worker: Worker) -> dict:
    """Convert the worker sampler's metrics to the System tab response."""
    sample = worker.system_sample or {}
    sampled_at = sample.get("sampled_at")
    metrics = sample.get("metrics", {})
    gpus = {}
    for key, value in metrics.items():
        if not key.startswith("system/gpu/"):
            continue
        index, separator, sensor = key.removeprefix("system/gpu/").partition("/")
        if not separator or not index.isascii() or not index.isdecimal():
            continue
        if sensor not in _GPU_MEMORY_FIELDS and sensor not in _GPU_FIELDS:
            continue
        index = int(index)
        if index >= 512:
            continue
        name = worker.gpu_info.get("name") if index == 0 else None
        gpu = gpus.setdefault(index, {"index": index, "name": name})
        if sensor in _GPU_MEMORY_FIELDS:
            gpu[_GPU_MEMORY_FIELDS[sensor]] = round(value * 1024**3)
        else:
            gpu[_GPU_FIELDS[sensor]] = value

    return {
        "gpus": [gpus[index] for index in sorted(gpus)],
        "total": len(gpus),
        "sampled_at": sampled_at,
        "stale": sampled_at is not None
        and (
            worker.status == WorkerStatus.OFFLINE
            or datetime.now(timezone.utc) - datetime.fromisoformat(sampled_at) > _STALE_AFTER
        ),
        "system": {
            "cpu_percent": metrics.get("system/cpu_percent"),
            "memory_percent": metrics.get("system/memory_percent"),
        },
    }
