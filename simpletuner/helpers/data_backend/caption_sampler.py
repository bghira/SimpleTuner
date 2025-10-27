"""Sampler for caption-only datasets."""

from __future__ import annotations

import math
import random
from typing import Iterator, List, Sequence, Tuple

from simpletuner.helpers.metadata.backends.caption import CaptionMetadataBackend

try:  # pragma: no cover - allow running without full torch install
    from torch.utils.data import Sampler as TorchSampler
except Exception:  # noqa: BLE001
    class TorchSampler:  # type: ignore[misc]
        pass


class CaptionSampler(TorchSampler):
    """Simple shuffle + repeat sampler that yields caption metadata ids in batches."""

    def __init__(
        self,
        id: str,
        metadata_backend: CaptionMetadataBackend,
        accelerator,
        batch_size: int,
        *,
        repeats: int = 0,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.id = id
        self.metadata_backend = metadata_backend
        self.accelerator = accelerator
        self.batch_size = max(int(batch_size or 1), 1)
        self.shuffle = shuffle
        self.seed = int(seed or 0)
        self.repeats = max(int(repeats or 0), 0)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Mirror DistributedSampler API so Accelerate can drive determinism."""
        self.epoch = int(epoch)

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        metadata_ids = self.metadata_backend.list_metadata_ids()
        if not metadata_ids:
            return
        epoch_entries = self._prepare_epoch_entries(metadata_ids)
        for start in range(0, len(epoch_entries), self.batch_size):
            yield tuple(epoch_entries[start : start + self.batch_size])

    def __len__(self) -> int:
        metadata_ids = self.metadata_backend.list_metadata_ids()
        total_entries = len(metadata_ids) * max(self.repeats + 1, 1)
        if total_entries == 0:
            return 0
        total_size = self._total_size(total_entries)
        per_rank = total_size // self._num_replicas()
        return per_rank // self.batch_size

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_epoch_entries(self, metadata_ids: Sequence[str]) -> List[str]:
        entries = list(metadata_ids) * max(self.repeats + 1, 1)
        if not entries:
            return []

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(entries)

        total_size = self._total_size(len(entries))
        if total_size > len(entries):
            pad = entries[: total_size - len(entries)]
            entries.extend(pad)

        num_replicas = self._num_replicas()
        rank = self._rank()
        if num_replicas <= 1:
            local_entries = entries
        else:
            local_entries = entries[rank:total_size:num_replicas]

        return local_entries

    def _num_replicas(self) -> int:
        accelerator = getattr(self, "accelerator", None)
        candidate = getattr(accelerator, "num_processes", None)
        if candidate is None:
            state = getattr(accelerator, "state", None) if accelerator is not None else None
            candidate = getattr(state, "num_processes", None)
        return int(candidate or 1)

    def _rank(self) -> int:
        accelerator = getattr(self, "accelerator", None)
        candidate = getattr(accelerator, "process_index", None)
        if candidate is None:
            state = getattr(accelerator, "state", None) if accelerator is not None else None
            candidate = getattr(state, "process_index", None)
        return int(candidate or 0)

    def _total_size(self, current_size: int) -> int:
        num_replicas = max(self._num_replicas(), 1)
        world_batch = max(self.batch_size * num_replicas, 1)
        return int(math.ceil(current_size / world_batch) * world_batch)
