"""Sampler for caption-only datasets."""

from __future__ import annotations

import logging
from typing import Iterator, Tuple

from torch.utils.data import Sampler

from simpletuner.helpers.data_backend.cursor import DatasetCursor
from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.multiaspect.state import BucketStateManager

logger = logging.getLogger(__name__)


class CaptionSampler(Sampler):
    """Yields batches of caption metadata ids in the order of the dataset's cursor."""

    def __init__(self, dataset_runtime: DatasetRuntime):
        self.id = dataset_runtime.id
        self.dataset_runtime = dataset_runtime
        self.state_manager = BucketStateManager(self.id)

    @property
    def cursor(self) -> DatasetCursor:
        return self.dataset_runtime.cursor()

    @property
    def batch_size(self) -> int:
        return self.cursor.batch_size

    def drop_unfinished_epoch(self) -> None:
        """Start the next epoch if this one was left part-way, as when another rank ended it."""
        if self.cursor.progress()[0] > 0:
            self.cursor.finish_epoch()

    def save_state(self, state_path: str) -> None:
        self.state_manager.save_state(self.cursor.state(), state_path)

    def load_states(self, state_path: str) -> None:
        state = self.state_manager.load_state(state_path)
        if not state:
            raise ValueError(f"Caption sampler '{self.id}' checkpoint state is missing.")
        self.dataset_runtime.load_cursor_state(state)

    def log_state(self) -> None:
        consumed, total = self.cursor.progress()
        logger.info("Caption sampler %s: epoch=%s, consumed_captions=%s/%s", self.id, self.cursor.epoch, consumed, total)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        while True:
            _bucket, metadata_ids = self.cursor.next_batch()
            yield tuple(metadata_ids)

    def __len__(self) -> int:
        return len(self.cursor)
