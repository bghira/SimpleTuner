"""Sampler for caption-only datasets."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from typing import Iterator, List, Sequence, Tuple

from torch.utils.data import Sampler

from simpletuner.helpers.metadata.backends.caption import CaptionMetadataBackend
from simpletuner.helpers.multiaspect.state import BucketStateManager
from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)


class CaptionSampler(Sampler):
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
        self._cursor = 0
        self._epoch_entries = None
        self.state_manager = BucketStateManager(self.id)

    def set_epoch(self, epoch: int) -> None:
        """Mirror DistributedSampler API so Accelerate can drive determinism."""
        if self.epoch != int(epoch):
            self.epoch = int(epoch)
            self._cursor = 0
            self._epoch_entries = None

    def _checkpoint_layout(self) -> dict:
        records = [
            [metadata_id, self.metadata_backend.get_record(metadata_id).caption_text]
            for metadata_id in self.metadata_backend.list_metadata_ids()
        ]
        parallelism = getattr(self.accelerator, "parallelism_config", None)
        return {
            "batch_size": self.batch_size,
            "repeats": self.repeats,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "num_processes": self._num_replicas(),
            "rank": self._rank(),
            "distributed_type": str(getattr(self.accelerator, "distributed_type", "NO")),
            "gradient_accumulation_steps": getattr(self.accelerator, "gradient_accumulation_steps", 1),
            "parallelism": {
                key: getattr(parallelism, key, 1) for key in ("dp_replicate_size", "dp_shard_size", "cp_size", "tp_size")
            },
            "captions_sha256": hashlib.sha256(json.dumps(records, ensure_ascii=False).encode("utf-8")).hexdigest(),
        }

    def save_state(self, state_path: str) -> None:
        self.state_manager.save_state(
            {"epoch": self.epoch, "cursor": self._cursor, "layout": self._checkpoint_layout()}, state_path
        )

    def load_states(self, state_path: str) -> None:
        state = self.state_manager.load_state(state_path)
        if not state:
            raise ValueError(f"Caption sampler '{self.id}' checkpoint state is missing.")
        layout = self._checkpoint_layout()
        if state["layout"] != layout:
            changed = [key for key, value in layout.items() if state["layout"].get(key) != value]
            raise ValueError(
                f"Caption sampler resume requires unchanged dataset and topology; changed: {', '.join(changed)}."
            )
        epoch, cursor = state["epoch"], state["cursor"]
        if (
            not isinstance(epoch, int)
            or epoch < 0
            or not isinstance(cursor, int)
            or cursor < 0
            or cursor > len(self) * self.batch_size
            or cursor % self.batch_size
        ):
            raise ValueError("Caption sampler checkpoint has an invalid epoch or cursor.")
        self.epoch = epoch
        self._cursor = cursor
        self._epoch_entries = None

    def log_state(self) -> None:
        logger.info(
            "Caption sampler %s: epoch=%s, consumed_batches=%s/%s",
            self.id,
            self.epoch,
            self._cursor // self.batch_size,
            len(self),
        )

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        metadata_ids = self.metadata_backend.list_metadata_ids()
        if not metadata_ids:
            raise MultiDatasetExhausted()
        if self._epoch_entries is None:
            self._epoch_entries = self._prepare_epoch_entries(metadata_ids)
        epoch_entries = self._epoch_entries
        if self._cursor >= len(epoch_entries):
            self.set_epoch(self.epoch + 1)
            StateTracker.set_repeats(data_backend_id=self.id, repeats=self.repeats)
            raise MultiDatasetExhausted()
        while self._cursor < len(epoch_entries):
            start = self._cursor
            self._cursor += self.batch_size
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
            entries = (entries * math.ceil(total_size / len(entries)))[:total_size]

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
