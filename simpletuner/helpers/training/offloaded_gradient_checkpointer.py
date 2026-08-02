"""CPU saved-tensor offload helpers for activation memory pressure."""

import sys
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from diffusers.utils.torch_utils import is_torch_version
from torch.utils.checkpoint import checkpoint as torch_checkpoint

_DEFAULT_PIN_MEMORY_MAX_BUCKETS = 12
_ACTIVATION_PREFETCH_MIN_OBSERVATIONS = 2
_ACTIVATION_PREFETCH_MIN_CONFIDENCE = 0.80
_ACTIVATION_PREFETCH_AUTOTUNE_MIN_SAMPLES = 8
_ACTIVATION_PREFETCH_AUTOTUNE_MARGIN = 0.98
_DEFAULT_D2H_COPY_STREAMS = 2
_DEFAULT_H2D_PREFETCH_STREAMS = 4


@dataclass(frozen=True)
class _PinnedBucketKey:
    size: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    layout: torch.layout


@dataclass(frozen=True)
class _RestoreView:
    size: tuple[int, ...]
    stride: tuple[int, ...]


@dataclass
class _OffloadedActivationRecord:
    logical_id: str
    predictor_id: str
    generation: int
    tensor: torch.Tensor
    original_device: torch.device
    pool_key: "_PinnedBucketKey | None"
    restore_view: _RestoreView | None
    ready_event: torch.cuda.Event | None
    prefetched_tensor: torch.Tensor | None = None
    prefetch_event: torch.cuda.Event | None = None
    consumed: bool = False
    cpu_released: bool = False


@dataclass
class _PinnedBucketStats:
    accesses: int = 0
    resident_accesses: int = 0
    pinned_checkouts: int = 0
    buffer_reuses: int = 0
    allocations: int = 0
    admissions: int = 0
    evictions: int = 0
    cap_misses: int = 0
    disabled_misses: int = 0
    allocation_failures: int = 0
    releases: int = 0
    dropped_releases: int = 0
    last_access: int = 0
    last_admission: int = 0
    last_eviction: int = 0

    @property
    def pinned_checkout_rate(self) -> float:
        return self.pinned_checkouts / self.accesses if self.accesses else 0.0

    @property
    def buffer_reuse_rate(self) -> float:
        return self.buffer_reuses / self.pinned_checkouts if self.pinned_checkouts else 0.0


class _PinnedMemoryPool:
    def __init__(self, max_buckets: int = _DEFAULT_PIN_MEMORY_MAX_BUCKETS):
        self.max_buckets = max(0, int(max_buckets))
        self.available: OrderedDict[_PinnedBucketKey, list[torch.Tensor]] = OrderedDict()
        self.pending: dict[_PinnedBucketKey, list[tuple[torch.cuda.Event, torch.Tensor]]] = defaultdict(list)
        self.stats: dict[_PinnedBucketKey, _PinnedBucketStats] = {}
        self.total_accesses = 0
        self.total_pinned_checkouts = 0
        self.total_buffer_reuses = 0
        self.total_allocations = 0
        self.total_cap_misses = 0
        self.total_evictions = 0

    def set_max_buckets(self, max_buckets: int) -> None:
        self.max_buckets = max(0, int(max_buckets))
        while len(self.available) > self.max_buckets:
            victim, _ = next(iter(self.available.items()))
            self._evict(victim)

    def key_for(self, tensor: torch.Tensor) -> _PinnedBucketKey | None:
        if tensor.layout != torch.strided:
            return None
        return _PinnedBucketKey(tuple(tensor.size()), tuple(tensor.stride()), tensor.dtype, tensor.layout)

    def checkout(self, key: _PinnedBucketKey) -> torch.Tensor | None:
        stats = self._record_access(key)
        if self.max_buckets <= 0:
            stats.disabled_misses += 1
            return None
        self._drain_completed(key)
        if key not in self.available:
            if len(self.available) >= self.max_buckets:
                if not self._admit_by_eviction(key):
                    stats.cap_misses += 1
                    self.total_cap_misses += 1
                    return None
            self.available[key] = []
            stats.admissions += 1
            stats.last_admission = self.total_accesses
        else:
            stats.resident_accesses += 1
        self.available.move_to_end(key)
        if self.available[key]:
            stats.buffer_reuses += 1
            stats.pinned_checkouts += 1
            self.total_buffer_reuses += 1
            self.total_pinned_checkouts += 1
            return self.available[key].pop()
        try:
            tensor = self._allocate(key)
        except RuntimeError:
            stats.allocation_failures += 1
            self.discard_empty_bucket(key)
            return None
        stats.allocations += 1
        stats.pinned_checkouts += 1
        self.total_allocations += 1
        self.total_pinned_checkouts += 1
        return tensor

    def release_after_cuda_copy(self, key: _PinnedBucketKey, tensor: torch.Tensor, device: torch.device) -> None:
        stats = self._stats_for(key)
        if key not in self.available:
            stats.dropped_releases += 1
            return
        stats.releases += 1
        if device.type == "cuda" and torch.cuda.is_available():
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            self._release_after_event(key, tensor, event)
            return
        self.available[key].append(tensor)

    def release_after_record_event(
        self, key: _PinnedBucketKey, tensor: torch.Tensor, event: torch.cuda.Event | None
    ) -> None:
        stats = self._stats_for(key)
        if key not in self.available:
            stats.dropped_releases += 1
            return
        stats.releases += 1
        self._release_after_event(key, tensor, event)

    def _release_after_event(self, key: _PinnedBucketKey, tensor: torch.Tensor, event: torch.cuda.Event | None) -> None:
        if event is not None and torch.cuda.is_available():
            self.pending[key].append((event, tensor))
            return
        self.available[key].append(tensor)

    def discard_empty_bucket(self, key: _PinnedBucketKey) -> None:
        if not self.available.get(key) and not self.pending.get(key):
            self.available.pop(key, None)

    def _drain_completed(self, key: _PinnedBucketKey) -> None:
        pending = self.pending.get(key)
        if not pending:
            return
        remaining = []
        for event, tensor in pending:
            if event.query():
                self.available.setdefault(key, []).append(tensor)
            else:
                remaining.append((event, tensor))
        if remaining:
            self.pending[key] = remaining
        else:
            self.pending.pop(key, None)

    def _allocate(self, key: _PinnedBucketKey) -> torch.Tensor:
        return torch.empty_strided(key.size, key.stride, dtype=key.dtype, layout=key.layout, device="cpu", pin_memory=True)

    def _record_access(self, key: _PinnedBucketKey) -> _PinnedBucketStats:
        stats = self._stats_for(key)
        self.total_accesses += 1
        stats.accesses += 1
        stats.last_access = self.total_accesses
        return stats

    def _stats_for(self, key: _PinnedBucketKey) -> _PinnedBucketStats:
        stats = self.stats.get(key)
        if stats is None:
            stats = _PinnedBucketStats()
            self.stats[key] = stats
        return stats

    def _admit_by_eviction(self, candidate: _PinnedBucketKey) -> bool:
        evictable = [key for key in self.available if not self.pending.get(key)]
        if not evictable:
            return False
        candidate_score = self._admission_score(candidate)
        victim = min(evictable, key=self._admission_score)
        if candidate_score <= self._admission_score(victim):
            return False
        self._evict(victim)
        return True

    def _admission_score(self, key: _PinnedBucketKey) -> tuple[int, float, int]:
        stats = self._stats_for(key)
        return (stats.accesses, stats.pinned_checkout_rate, stats.last_access)

    def _evict(self, key: _PinnedBucketKey) -> None:
        self.available.pop(key, None)
        stats = self._stats_for(key)
        stats.evictions += 1
        stats.last_eviction = self.total_accesses
        self.total_evictions += 1

    def snapshot(self) -> dict:
        buckets = []
        resident = set(self.available)
        pending_counts = {key: len(value) for key, value in self.pending.items()}
        for key, stats in self.stats.items():
            buckets.append(
                {
                    "size": key.size,
                    "stride": key.stride,
                    "dtype": str(key.dtype),
                    "layout": str(key.layout),
                    "resident": key in resident,
                    "available_buffers": len(self.available.get(key, ())),
                    "pending_buffers": pending_counts.get(key, 0),
                    "accesses": stats.accesses,
                    "resident_accesses": stats.resident_accesses,
                    "pinned_checkouts": stats.pinned_checkouts,
                    "buffer_reuses": stats.buffer_reuses,
                    "allocations": stats.allocations,
                    "admissions": stats.admissions,
                    "evictions": stats.evictions,
                    "cap_misses": stats.cap_misses,
                    "disabled_misses": stats.disabled_misses,
                    "allocation_failures": stats.allocation_failures,
                    "releases": stats.releases,
                    "dropped_releases": stats.dropped_releases,
                    "pinned_checkout_rate": stats.pinned_checkout_rate,
                    "buffer_reuse_rate": stats.buffer_reuse_rate,
                    "last_access": stats.last_access,
                    "last_admission": stats.last_admission,
                    "last_eviction": stats.last_eviction,
                }
            )
        buckets.sort(key=lambda item: (item["accesses"], item["pinned_checkouts"], item["last_access"]), reverse=True)
        return {
            "max_buckets": self.max_buckets,
            "resident_buckets": len(self.available),
            "tracked_buckets": len(self.stats),
            "total_accesses": self.total_accesses,
            "total_pinned_checkouts": self.total_pinned_checkouts,
            "total_buffer_reuses": self.total_buffer_reuses,
            "total_allocations": self.total_allocations,
            "total_cap_misses": self.total_cap_misses,
            "total_evictions": self.total_evictions,
            "pinned_checkout_rate": self.total_pinned_checkouts / self.total_accesses if self.total_accesses else 0.0,
            "buffer_reuse_rate": (
                self.total_buffer_reuses / self.total_pinned_checkouts if self.total_pinned_checkouts else 0.0
            ),
            "buckets": buckets,
        }

    def reset_stats(self) -> None:
        self.stats.clear()
        self.total_accesses = 0
        self.total_pinned_checkouts = 0
        self.total_buffer_reuses = 0
        self.total_allocations = 0
        self.total_cap_misses = 0
        self.total_evictions = 0


_PINNED_MEMORY_POOL = _PinnedMemoryPool()
_ACTIVATION_PREFETCH_ENABLED = False
_ACTIVATION_PREFETCH_AUTOTUNE_ENABLED = True


class _CudaStreamPool:
    def __init__(self, width: int):
        self.width = max(1, int(width))
        self.streams: dict[int, list[torch.cuda.Stream]] = {}
        self.next_index: dict[int, int] = defaultdict(int)
        self.uses: dict[int, list[int]] = {}

    def set_width(self, width: int) -> None:
        self.width = max(1, int(width))
        self.streams.clear()
        self.next_index.clear()
        self.uses.clear()

    def next(self, device: torch.device) -> torch.cuda.Stream:
        device_index = _device_index(device)
        streams = self.streams.get(device_index)
        if streams is None or len(streams) != self.width:
            streams = [torch.cuda.Stream(device=device_index) for _ in range(self.width)]
            self.streams[device_index] = streams
            self.next_index[device_index] = 0
            self.uses[device_index] = [0 for _ in range(self.width)]
        stream_index = self.next_index[device_index]
        self.next_index[device_index] = (stream_index + 1) % self.width
        self.uses[device_index][stream_index] += 1
        return streams[stream_index]

    def snapshot(self) -> dict:
        return {
            "width": self.width,
            "devices": {
                device_index: {
                    "allocated_streams": len(self.streams.get(device_index, ())),
                    "uses": list(self.uses.get(device_index, ())),
                    "total_uses": sum(self.uses.get(device_index, ())),
                }
                for device_index in sorted(set(self.streams) | set(self.uses))
            },
        }

    def reset_stats(self) -> None:
        for device_index, uses in list(self.uses.items()):
            self.uses[device_index] = [0 for _ in uses]


_D2H_COPY_STREAMS = _CudaStreamPool(_DEFAULT_D2H_COPY_STREAMS)
_H2D_PREFETCH_STREAMS = _CudaStreamPool(_DEFAULT_H2D_PREFETCH_STREAMS)


class _ActivationOffloadPrefetchRuntime:
    def __init__(self):
        self.records: dict[tuple[int, str], list[_OffloadedActivationRecord]] = defaultdict(list)
        self.transitions: dict[str, dict[str, int]] = defaultdict(dict)
        self.successors: dict[str, str] = {}
        self.disabled: dict[str, dict] = {}
        self.previous_by_generation: dict[int, str] = {}
        self.current_generation = 0
        self.saw_unpack_since_pack = False
        self.total_packs = 0
        self.total_unpacks = 0
        self.prefetch_attempts = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.prefetch_enqueued = 0
        self.prefetch_skipped = 0
        self.prefetch_stale = 0
        self.transition_updates = 0
        self.jit_restore_ms: list[float] = []
        self.prefetch_wait_ms: list[float] = []
        self.autotune_disabled = False
        self.autotune_decision: str | None = None

    def next_generation_for_pack(self) -> int:
        if self.saw_unpack_since_pack:
            self._retire_generation(self.current_generation)
            self.current_generation += 1
            self.saw_unpack_since_pack = False
            self.previous_by_generation.pop(self.current_generation, None)
        return self.current_generation

    def register(self, record: _OffloadedActivationRecord) -> None:
        self.total_packs += 1
        self.records[(record.generation, record.predictor_id)].append(record)

    def consume(self, record: _OffloadedActivationRecord) -> None:
        self.total_unpacks += 1
        self.saw_unpack_since_pack = True
        self._remove_record(record)
        previous_id = self.previous_by_generation.get(record.generation)
        if previous_id is not None:
            self._record_transition(previous_id, record.predictor_id)
        self.previous_by_generation[record.generation] = record.predictor_id
        successor_id = self.successors.get(record.predictor_id)
        if successor_id and self.prefetch_allowed():
            self.prefetch_attempts += 1
            if not self.prefetch(record.generation, successor_id):
                self.prefetch_misses += 1

    def prefetch(self, generation: int, logical_id: str) -> bool:
        candidates = self.records.get((generation, logical_id), ())
        for candidate in reversed(candidates):
            if candidate.consumed or candidate.prefetched_tensor is not None:
                continue
            if candidate.original_device.type != "cuda" or not torch.cuda.is_available():
                self.prefetch_skipped += 1
                return False
            if candidate.pool_key is None or not candidate.tensor.is_pinned():
                self.prefetch_skipped += 1
                return False
            try:
                copy_stream = _h2d_prefetch_stream_for(candidate.original_device)
                with torch.cuda.stream(copy_stream):
                    if candidate.ready_event is not None:
                        copy_stream.wait_event(candidate.ready_event)
                    restored = candidate.tensor.to(candidate.original_device, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(copy_stream)
                candidate.prefetched_tensor = restored
                candidate.prefetch_event = event
                self.prefetch_enqueued += 1
                return True
            except RuntimeError:
                candidate.prefetched_tensor = None
                candidate.prefetch_event = None
                self.prefetch_stale += 1
                return False
        self.prefetch_skipped += 1
        return False

    def _remove_record(self, record: _OffloadedActivationRecord) -> None:
        key = (record.generation, record.predictor_id)
        entries = self.records.get(key)
        if not entries:
            return
        try:
            entries.remove(record)
        except ValueError:
            return
        if not entries:
            self.records.pop(key, None)

    def _retire_generation(self, generation: int) -> None:
        stale_keys = [key for key in self.records if key[0] == generation]
        for key in stale_keys:
            entries = self.records.pop(key, ())
            for record in entries:
                self._release_unused_record(record)

    def _release_unused_record(self, record: _OffloadedActivationRecord) -> None:
        if record.cpu_released or record.pool_key is None:
            return
        release_event = record.prefetch_event or record.ready_event
        _PINNED_MEMORY_POOL.release_after_record_event(record.pool_key, record.tensor, release_event)
        record.cpu_released = True
        record.prefetched_tensor = None
        record.prefetch_event = None

    def _record_transition(self, previous_id: str, actual_id: str) -> None:
        if self.disabled.get(previous_id):
            return
        counts = self.transitions[previous_id]
        counts[actual_id] = int(counts.get(actual_id, 0)) + 1
        total = sum(int(value) for value in counts.values())
        best_id, best_count = max(counts.items(), key=lambda item: int(item[1]))
        if total < _ACTIVATION_PREFETCH_MIN_OBSERVATIONS:
            return
        confidence = best_count / total if total else 0.0
        old_successor = self.successors.get(previous_id)
        if confidence >= _ACTIVATION_PREFETCH_MIN_CONFIDENCE:
            self.successors[previous_id] = best_id
            if old_successor != best_id:
                self.transition_updates += 1
        else:
            self.successors.pop(previous_id, None)

    def record_hit(self) -> None:
        self.prefetch_hits += 1

    def prefetch_allowed(self) -> bool:
        return _ACTIVATION_PREFETCH_ENABLED and not self.autotune_disabled

    @staticmethod
    def _median(values: list[float]) -> float:
        ordered = sorted(values)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[midpoint]
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0

    def record_jit_restore_ms(self, value: float) -> None:
        if not _ACTIVATION_PREFETCH_AUTOTUNE_ENABLED or self.autotune_decision is not None:
            return
        self.jit_restore_ms.append(float(value))

    def record_prefetch_wait_ms(self, value: float) -> None:
        if not _ACTIVATION_PREFETCH_AUTOTUNE_ENABLED or self.autotune_decision is not None:
            return
        self.prefetch_wait_ms.append(float(value))
        self._maybe_autotune()

    def _maybe_autotune(self) -> None:
        if len(self.jit_restore_ms) < _ACTIVATION_PREFETCH_AUTOTUNE_MIN_SAMPLES:
            return
        if len(self.prefetch_wait_ms) < _ACTIVATION_PREFETCH_AUTOTUNE_MIN_SAMPLES:
            return
        jit_median = self._median(self.jit_restore_ms)
        prefetch_median = self._median(self.prefetch_wait_ms)
        if prefetch_median < jit_median * _ACTIVATION_PREFETCH_AUTOTUNE_MARGIN:
            self.autotune_decision = "prefetch"
            return
        self.autotune_disabled = True
        self.autotune_decision = "jit"

    def snapshot(self) -> dict:
        active_records = sum(len(entries) for entries in self.records.values())
        return {
            "enabled": _ACTIVATION_PREFETCH_ENABLED,
            "active_records": active_records,
            "generations_tracked": len({generation for generation, _logical_id in self.records}),
            "total_packs": self.total_packs,
            "total_unpacks": self.total_unpacks,
            "prefetch_attempts": self.prefetch_attempts,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "prefetch_enqueued": self.prefetch_enqueued,
            "prefetch_skipped": self.prefetch_skipped,
            "prefetch_stale": self.prefetch_stale,
            "transition_updates": self.transition_updates,
            "learned_successors": len(self.successors),
            "hit_rate": self.prefetch_hits / self.prefetch_attempts if self.prefetch_attempts else 0.0,
            "autotune_enabled": _ACTIVATION_PREFETCH_AUTOTUNE_ENABLED,
            "autotune_disabled": self.autotune_disabled,
            "autotune_decision": self.autotune_decision,
            "jit_restore_samples": len(self.jit_restore_ms),
            "prefetch_wait_samples": len(self.prefetch_wait_ms),
            "jit_restore_median_ms": self._median(self.jit_restore_ms) if self.jit_restore_ms else None,
            "prefetch_wait_median_ms": self._median(self.prefetch_wait_ms) if self.prefetch_wait_ms else None,
            "copy_streams": get_activation_offload_copy_stream_stats(),
        }

    def reset(self) -> None:
        for entries in list(self.records.values()):
            for record in entries:
                self._release_unused_record(record)
        self.records.clear()
        self.transitions.clear()
        self.successors.clear()
        self.disabled.clear()
        self.previous_by_generation.clear()
        self.current_generation = 0
        self.saw_unpack_since_pack = False
        self.total_packs = 0
        self.total_unpacks = 0
        self.prefetch_attempts = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.prefetch_enqueued = 0
        self.prefetch_skipped = 0
        self.prefetch_stale = 0
        self.transition_updates = 0
        self.jit_restore_ms.clear()
        self.prefetch_wait_ms.clear()
        self.autotune_disabled = False
        self.autotune_decision = None


_ACTIVATION_PREFETCH_RUNTIME = _ActivationOffloadPrefetchRuntime()


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _d2h_copy_stream_for(device: torch.device) -> torch.cuda.Stream:
    return _D2H_COPY_STREAMS.next(device)


def _h2d_prefetch_stream_for(device: torch.device) -> torch.cuda.Stream:
    return _H2D_PREFETCH_STREAMS.next(device)


def set_activation_offload_d2h_copy_stream_count(count: int) -> None:
    """Set the number of round-robin CUDA streams used for GPU-to-CPU offload copies."""
    _D2H_COPY_STREAMS.set_width(count)


def get_activation_offload_d2h_copy_stream_count() -> int:
    return _D2H_COPY_STREAMS.width


def set_activation_offload_h2d_prefetch_stream_count(count: int) -> None:
    """Set the number of round-robin CUDA streams used for CPU-to-GPU prefetch copies."""
    _H2D_PREFETCH_STREAMS.set_width(count)


def get_activation_offload_h2d_prefetch_stream_count() -> int:
    return _H2D_PREFETCH_STREAMS.width


def get_activation_offload_copy_stream_stats() -> dict:
    return {
        "d2h": _D2H_COPY_STREAMS.snapshot(),
        "h2d_prefetch": _H2D_PREFETCH_STREAMS.snapshot(),
    }


def reset_activation_offload_copy_stream_stats() -> None:
    _D2H_COPY_STREAMS.reset_stats()
    _H2D_PREFETCH_STREAMS.reset_stats()


def set_activation_offload_pin_memory_max_buckets(max_buckets: int) -> None:
    """Set the max number of distinct pinned CPU tensor buckets used for activation offload."""
    _PINNED_MEMORY_POOL.set_max_buckets(max_buckets)


def get_activation_offload_pin_memory_max_buckets() -> int:
    return _PINNED_MEMORY_POOL.max_buckets


def get_activation_offload_pin_memory_stats() -> dict:
    """Return lifetime pinned CPU bucket statistics for activation offload."""
    return _PINNED_MEMORY_POOL.snapshot()


def reset_activation_offload_pin_memory_stats() -> None:
    """Reset pinned CPU bucket statistics without clearing resident pinned buffers."""
    _PINNED_MEMORY_POOL.reset_stats()


def set_activation_offload_prefetch_enabled(enabled: bool) -> None:
    """Enable learned H2D prefetching for labeled activation offload contexts."""
    global _ACTIVATION_PREFETCH_ENABLED
    _ACTIVATION_PREFETCH_ENABLED = bool(enabled)


def get_activation_offload_prefetch_enabled() -> bool:
    return _ACTIVATION_PREFETCH_ENABLED


def set_activation_offload_prefetch_autotune_enabled(enabled: bool) -> None:
    """Enable first-run latency gating for activation prefetch."""
    global _ACTIVATION_PREFETCH_AUTOTUNE_ENABLED
    _ACTIVATION_PREFETCH_AUTOTUNE_ENABLED = bool(enabled)


def get_activation_offload_prefetch_autotune_enabled() -> bool:
    return _ACTIVATION_PREFETCH_AUTOTUNE_ENABLED


def get_activation_offload_prefetch_stats() -> dict:
    return _ACTIVATION_PREFETCH_RUNTIME.snapshot()


def reset_activation_offload_prefetch_stats() -> None:
    _ACTIVATION_PREFETCH_RUNTIME.reset()
    reset_activation_offload_copy_stream_stats()


def set_activation_offload_prefetch_runtime_disabled(disabled: bool, *, decision: str | None = None) -> None:
    """Force the learned prefetch runtime on or off without changing payload labeling."""
    _ACTIVATION_PREFETCH_RUNTIME.autotune_disabled = bool(disabled)
    _ACTIVATION_PREFETCH_RUNTIME.autotune_decision = decision


def mark_activation_offload_prefetch_autotune_decision(decision: str) -> None:
    """Record the end-to-end autotune decision made by the trainer."""
    normalised = str(decision).lower()
    if normalised not in {"prefetch", "jit"}:
        raise ValueError("activation offload prefetch decision must be 'prefetch' or 'jit'")
    set_activation_offload_prefetch_runtime_disabled(normalised == "jit", decision=normalised)


def activation_offload_prefetch_autotune_decision() -> str | None:
    return _ACTIVATION_PREFETCH_RUNTIME.autotune_decision


class CPUOffloadHooks:
    """Context manager hooks that offload saved tensors to CPU during checkpointing."""

    def __init__(
        self,
        *,
        offload_leaf_tensors: bool = False,
        pin_memory: bool = True,
        label: str | None = None,
        prefetch: bool | None = None,
    ):
        self.offload_leaf_tensors = offload_leaf_tensors
        self.pin_memory = pin_memory
        self.label = str(label) if label else None
        self.prefetch = _ACTIVATION_PREFETCH_ENABLED if prefetch is None else bool(prefetch)
        self.pack_index = 0

    @staticmethod
    def _flat_storage_view(tensor: torch.Tensor) -> torch.Tensor | None:
        if tensor.layout != torch.strided:
            return None
        span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
        if span != tensor.numel():
            return None
        return torch.as_strided(tensor, (tensor.numel(),), (1,), tensor.storage_offset())

    def _transfer_view(self, tensor: torch.Tensor) -> tuple[torch.Tensor, _RestoreView | None]:
        flat_view = self._flat_storage_view(tensor)
        if flat_view is None:
            return tensor, None
        return flat_view, _RestoreView(tuple(tensor.size()), tuple(tensor.stride()))

    def _copy_to_cpu(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, _PinnedBucketKey | None, _RestoreView | None, torch.cuda.Event | None]:
        detached = tensor.detach()
        transfer_tensor, restore_view = self._transfer_view(detached)
        if self.pin_memory:
            key = _PINNED_MEMORY_POOL.key_for(transfer_tensor)
            if key is not None:
                cpu_tensor = _PINNED_MEMORY_POOL.checkout(key)
                if cpu_tensor is not None:
                    try:
                        copy_stream = _d2h_copy_stream_for(tensor.device)
                        current_stream = torch.cuda.current_stream(tensor.device)
                        with torch.cuda.stream(copy_stream):
                            copy_stream.wait_stream(current_stream)
                            transfer_tensor.record_stream(copy_stream)
                            cpu_tensor.copy_(transfer_tensor, non_blocking=True)
                            ready_event = torch.cuda.Event()
                            ready_event.record(copy_stream)
                        return cpu_tensor, key, restore_view, ready_event
                    except RuntimeError:
                        _PINNED_MEMORY_POOL.discard_empty_bucket(key)
        return transfer_tensor.to("cpu", non_blocking=True), None, restore_view, None

    def pack(self, tensor: torch.Tensor):
        """Called when a tensor is saved for backward - offload to CPU.

        Returns a payload (cpu_tensor, original_device) so that unpack() can
        restore only tensors that were actually offloaded, and to their
        correct original devices.
        """
        if tensor.device.type == "cuda" and (self.offload_leaf_tensors or not tensor.is_leaf):
            cpu_tensor, pool_key, restore_view, ready_event = self._copy_to_cpu(tensor)
            if self.prefetch and self.label is not None:
                generation = _ACTIVATION_PREFETCH_RUNTIME.next_generation_for_pack()
                predictor_id = f"{self.label}:{self.pack_index}"
                logical_id = f"{generation}:{predictor_id}:{id(cpu_tensor)}"
                self.pack_index += 1
                record = _OffloadedActivationRecord(
                    logical_id=logical_id,
                    predictor_id=predictor_id,
                    generation=generation,
                    tensor=cpu_tensor,
                    original_device=tensor.device,
                    pool_key=pool_key,
                    restore_view=restore_view,
                    ready_event=ready_event,
                )
                _ACTIVATION_PREFETCH_RUNTIME.register(record)
                return record
            return cpu_tensor, tensor.device, pool_key, restore_view, ready_event
        return tensor, None

    def unpack(self, payload) -> torch.Tensor:
        """Called when a tensor is needed for backward - restore to original device if needed."""
        if isinstance(payload, _OffloadedActivationRecord):
            return self._unpack_record(payload)

        # Expect payload of the form (cpu_tensor, original_device[, pool_key, restore_view]).
        try:
            tensor, original_device, *rest = payload
        except (TypeError, ValueError):
            # Fallback: if payload is not in the expected form, return it as-is.
            return payload

        if original_device is not None and tensor.device.type == "cpu":
            current_stream = None
            ready_event = rest[2] if len(rest) > 2 else None
            if ready_event is not None and original_device.type == "cuda":
                current_stream = torch.cuda.current_stream(original_device)
                current_stream.wait_event(ready_event)
            restored = tensor.to(original_device, non_blocking=True)
            if original_device.type == "cuda":
                current_stream = current_stream or torch.cuda.current_stream(original_device)
                restored.record_stream(current_stream)
            pool_key = rest[0] if rest else None
            restore_view = rest[1] if len(rest) > 1 else None
            if pool_key is not None:
                _PINNED_MEMORY_POOL.release_after_cuda_copy(pool_key, tensor, original_device)
            if restore_view is not None:
                restored = torch.as_strided(restored, restore_view.size, restore_view.stride, 0)
            return restored
        return tensor

    def _release_cpu_tensor(self, record: _OffloadedActivationRecord) -> None:
        if record.cpu_released or record.pool_key is None:
            return
        _PINNED_MEMORY_POOL.release_after_cuda_copy(record.pool_key, record.tensor, record.original_device)
        record.cpu_released = True

    def _restore_view_if_needed(self, tensor: torch.Tensor, restore_view: _RestoreView | None) -> torch.Tensor:
        if restore_view is None:
            return tensor
        return torch.as_strided(tensor, restore_view.size, restore_view.stride, 0)

    def _unpack_record(self, record: _OffloadedActivationRecord) -> torch.Tensor:
        record.consumed = True
        if record.prefetched_tensor is not None:
            elapsed_ms = None
            if record.prefetch_event is not None and record.original_device.type == "cuda":
                current_stream = torch.cuda.current_stream(record.original_device)
                if _ACTIVATION_PREFETCH_AUTOTUNE_ENABLED and _ACTIVATION_PREFETCH_RUNTIME.autotune_decision is None:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record(current_stream)
                    current_stream.wait_event(record.prefetch_event)
                    end_event.record(current_stream)
                    end_event.synchronize()
                    elapsed_ms = start_event.elapsed_time(end_event)
                else:
                    current_stream.wait_event(record.prefetch_event)
                record.prefetched_tensor.record_stream(current_stream)
            restored = record.prefetched_tensor
            _ACTIVATION_PREFETCH_RUNTIME.record_hit()
            if elapsed_ms is not None:
                _ACTIVATION_PREFETCH_RUNTIME.record_prefetch_wait_ms(elapsed_ms)
            self._release_cpu_tensor(record)
            _ACTIVATION_PREFETCH_RUNTIME.consume(record)
            return self._restore_view_if_needed(restored, record.restore_view)

        if record.ready_event is not None and record.original_device.type == "cuda":
            torch.cuda.current_stream(record.original_device).wait_event(record.ready_event)
        elapsed_ms = None
        if (
            _ACTIVATION_PREFETCH_AUTOTUNE_ENABLED
            and _ACTIVATION_PREFETCH_RUNTIME.autotune_decision is None
            and record.original_device.type == "cuda"
        ):
            current_stream = torch.cuda.current_stream(record.original_device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(current_stream)
            restored = record.tensor.to(record.original_device, non_blocking=True)
            end_event.record(current_stream)
            end_event.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            restored = record.tensor.to(record.original_device, non_blocking=True)
        if record.original_device.type == "cuda":
            restored.record_stream(torch.cuda.current_stream(record.original_device))
        if elapsed_ms is not None:
            _ACTIVATION_PREFETCH_RUNTIME.record_jit_restore_ms(elapsed_ms)
        self._release_cpu_tensor(record)
        _ACTIVATION_PREFETCH_RUNTIME.consume(record)
        return self._restore_view_if_needed(restored, record.restore_view)


def activation_offload_context(enabled: bool = True, *, label: str | None = None, prefetch: bool | None = None):
    """Offload non-leaf CUDA tensors saved for backward inside the context."""
    if not enabled:
        return nullcontext()
    if label is not None:
        frame = sys._getframe(1)
        label = f"{label}:{frame.f_code.co_name}:{frame.f_lineno}"
    hooks = CPUOffloadHooks(offload_leaf_tensors=False, label=label, prefetch=prefetch)
    return torch.autograd.graph.saved_tensors_hooks(hooks.pack, hooks.unpack)


def activation_offload(function, *args, **kwargs):
    """Run a function normally while offloading saved non-leaf CUDA tensors."""
    kwargs.pop("use_reentrant", None)
    with activation_offload_context():
        return function(*args, **kwargs)


def offloaded_checkpoint(function, *args, use_reentrant: bool = False, **kwargs):
    """
    Drop-in replacement for torch.utils.checkpoint.checkpoint using CPU offload.

    This still uses PyTorch checkpoint rematerialization. Saved tensor hooks
    offload tensors that autograd saves inside the checkpointed region and
    restore them when needed.

    Args:
        function: The forward function to checkpoint
        *args: Positional arguments to pass to function
        use_reentrant: Whether to use reentrant checkpointing (passed to torch.checkpoint)
        **kwargs: Keyword arguments to pass to torch.checkpoint

    Returns:
        Output of the function

    Note:
        This backend is most effective when PCIe bandwidth is high and can hide
        the CPU<->GPU transfer latency during forward/backward computation.
    """
    hooks = CPUOffloadHooks()
    with torch.autograd.graph.saved_tensors_hooks(hooks.pack, hooks.unpack):
        # Only pass use_reentrant on PyTorch >= 1.11.0
        if is_torch_version(">=", "1.11.0"):
            return torch_checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)
        else:
            return torch_checkpoint(function, *args, **kwargs)
