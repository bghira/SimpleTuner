"""CUDA staging for batches produced by the background dataloader fetcher."""

import threading
from dataclasses import dataclass
from typing import Any, Callable

import torch


def _map_tensors(value: Any, transform: Callable[[torch.Tensor], torch.Tensor], memo=None) -> Any:
    if memo is None:
        memo = {}
    if torch.is_tensor(value):
        identity = id(value)
        if identity not in memo:
            memo[identity] = transform(value)
        return memo[identity]
    if isinstance(value, dict):
        return {key: _map_tensors(child, transform, memo) for key, child in value.items()}
    if isinstance(value, list):
        return [_map_tensors(child, transform, memo) for child in value]
    if isinstance(value, tuple):
        children = [_map_tensors(child, transform, memo) for child in value]
        if hasattr(value, "_fields"):
            return type(value)(*children)
        return tuple(children)
    return value


def cpu_tensor_bytes(value: Any) -> int:
    seen = set()

    def count(item: Any) -> int:
        if torch.is_tensor(item):
            identity = id(item)
            if identity in seen or item.device.type != "cpu":
                return 0
            seen.add(identity)
            return item.numel() * item.element_size()
        if isinstance(item, dict):
            return sum(count(child) for child in item.values())
        if isinstance(item, (list, tuple)):
            return sum(count(child) for child in item)
        return 0

    return count(value)


def _pin_cpu_tensors(value: Any) -> Any:
    return _map_tensors(
        value,
        lambda tensor: tensor.pin_memory() if tensor.device.type == "cpu" and not tensor.is_pinned() else tensor,
    )


def _move_cpu_tensors(value: Any, device: torch.device) -> Any:
    return _map_tensors(
        value,
        lambda tensor: tensor.to(device=device, non_blocking=True) if tensor.device.type == "cpu" else tensor,
    )


def _record_device_stream(value: Any, stream: torch.cuda.Stream) -> Any:
    def record(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type == "cuda":
            tensor.record_stream(stream)
        return tensor

    return _map_tensors(value, record)


@dataclass
class DevicePrefetchedBatch:
    batch: Any
    ready_event: torch.cuda.Event
    source_batch: Any


class CudaBatchPrefetcher:
    """Run collation on its own CUDA stream and optionally stage CPU tensors there."""

    def __init__(self, device: torch.device, minimum_bytes: int) -> None:
        if device is None:
            raise ValueError("CUDA batch prefetch requires an accelerator device")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA batch prefetch requires a CUDA device")
        if minimum_bytes < 0:
            raise ValueError("CUDA batch prefetch requires a nonnegative payload threshold")
        self.minimum_bytes = minimum_bytes
        self._pending_sources = []
        self._pending_sources_lock = threading.Lock()
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
            self.stream.wait_stream(torch.cuda.current_stream(self.device))

    def _retire_completed_sources(self) -> None:
        with self._pending_sources_lock:
            self._pending_sources = [(event, source) for event, source in self._pending_sources if not event.query()]

    def prefetch(self, batch: Any) -> Any:
        self._retire_completed_sources()
        if not self._should_stage(batch):
            return batch
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            return self._ready_batch(batch, stage=True)

    def fetch(self, retrieve: Callable[[], Any]) -> Any:
        self._retire_completed_sources()
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            batch = retrieve()
            if batch is False:
                return False
            return self._ready_batch(batch, stage=self._should_stage(batch))

    def _should_stage(self, batch: Any) -> bool:
        return self.minimum_bytes > 0 and cpu_tensor_bytes(batch) >= self.minimum_bytes

    def _ready_batch(self, batch: Any, *, stage: bool) -> DevicePrefetchedBatch:
        source_batch = _pin_cpu_tensors(batch) if stage else None
        device_batch = _move_cpu_tensors(source_batch, self.device) if stage else batch
        ready_event = torch.cuda.Event(enable_timing=False)
        ready_event.record(self.stream)
        return DevicePrefetchedBatch(
            batch=device_batch,
            ready_event=ready_event,
            source_batch=source_batch,
        )

    def consume(self, value: Any) -> Any:
        if not isinstance(value, DevicePrefetchedBatch):
            return value

        with torch.cuda.device(self.device):
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_event(value.ready_event)
            batch = _record_device_stream(value.batch, current_stream)
            if value.source_batch is not None and not value.ready_event.query():
                with self._pending_sources_lock:
                    self._pending_sources.append((value.ready_event, value.source_batch))
        return batch
