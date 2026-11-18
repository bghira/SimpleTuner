import logging
import os
import random
from collections.abc import Callable, Mapping, Sequence
from math import ceil

from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger("DatasetCursor")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")

STATE_FORMAT = 2


class DatasetCursor:
    """One rank's traversal of a dataset's buckets.

    The rank owns a fixed share of every bucket. Each epoch reshuffles that share and cuts it into
    batches; ranks draw the same number of batches from a bucket, in the same bucket order.
    """

    def __init__(
        self,
        dataset_id: str,
        buckets: Mapping[str, Sequence[str]],
        *,
        batch_size: int,
        repeats: int,
        seed: int,
        dp_size: int,
        dp_rank: int,
        gradient_accumulation_steps: int,
        allow_oversubscription: bool,
        shared_by_all_ranks: bool,
    ):
        self.dataset_id = dataset_id
        self.batch_size = int(batch_size)
        self.repeats = int(repeats)
        self.seed = seed
        # A dataset every rank reads whole (evaluation, conditioning) is not partitioned, and no
        # optimizer step depends on its buckets filling an effective batch.
        self.dp_size = 1 if shared_by_all_ranks else int(dp_size)
        self.dp_rank = 0 if shared_by_all_ranks else int(dp_rank)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.effective_batch_size = (
            1 if shared_by_all_ranks else self.batch_size * self.dp_size * self.gradient_accumulation_steps
        )
        self.epoch = 1

        self._buckets = buckets
        self._counts = {bucket: len(buckets[bucket]) for bucket in sorted(buckets, key=str) if buckets[bucket]}
        self._auto_repeats = self._undersized_bucket_repeats(allow_oversubscription)
        # Samples a rank draws from each bucket per epoch. It is the same on every rank.
        self._lengths = {}
        for bucket, count in self._counts.items():
            if bucket in self._auto_repeats:
                logical = count * (self._auto_repeats[bucket] + 1)
                scheduled = ceil(logical / self.effective_batch_size) * self.effective_batch_size
                self._lengths[bucket] = scheduled // self.dp_size
            else:
                self._lengths[bucket] = ceil(count / self.dp_size) * (self.repeats + 1)
        self._share = self._build_share()
        self._consumed = dict.fromkeys(self._counts, 0)
        self._orders: dict[str, list[str]] | None = None
        self._schedule: list[str] = []
        self._next = 0

    def _undersized_bucket_repeats(self, allow_oversubscription: bool) -> dict[str, int]:
        """Repeats needed by each bucket that cannot fill one effective batch."""
        passes = self.repeats + 1
        needed = {
            bucket: ceil(self.effective_batch_size / count) - 1
            for bucket, count in self._counts.items()
            if count * passes < self.effective_batch_size
        }
        if not needed:
            return {}
        total_samples = sum(self._counts.values())
        if allow_oversubscription and self.repeats == 0:
            logger.warning(
                f"(id={self.dataset_id}) Dataset oversubscription enabled: automatically padding {len(needed)} undersized bucket(s)\n"
                f"  - This allows training with {total_samples} samples across {self.dp_size} GPUs\n"
                f"  - Effective batch size: {self.effective_batch_size}\n"
                f"  - Maximum per-bucket repeat factor: {max(needed.values()) + 1}"
            )
            return needed

        error_msg = (
            f"Dataset configuration will produce zero usable batches.\n"
            f"\nCurrent configuration:\n"
            f"  - Dataset ID: {self.dataset_id}\n"
            f"  - Total samples: {total_samples}\n"
            f"  - Repeats: {self.repeats}\n"
            f"  - Batch size: {self.batch_size}\n"
            f"  - Number of GPUs: {self.dp_size}\n"
            f"  - Gradient accumulation steps: {self.gradient_accumulation_steps}\n"
            f"  - Effective batch size: {self.effective_batch_size}\n"
            f"\nProblem: {len(needed)} bucket(s) have insufficient samples:\n"
        )
        for bucket, repeats in needed.items():
            count = self._counts[bucket]
            error_msg += (
                f"  - Bucket {bucket}: {count} samples × {passes} (with repeats) = {count * passes} samples\n"
                f"    Need at least {self.effective_batch_size} samples to form one batch\n"
                f"    Minimum repeats required: {repeats}\n"
            )
        error_msg += (
            f"\nSolutions:\n"
            f"  1. Reduce batch_size (current: {self.batch_size})\n"
            f"  2. Reduce gradient_accumulation_steps (current: {self.gradient_accumulation_steps})\n"
            f"  3. Use fewer GPUs (current: {self.dp_size})\n"
            f"  4. Increase repeats to at least {max(needed.values())} (current: {self.repeats})\n"
            f"  5. Add more samples to the dataset\n"
            f"  6. Enable --allow_dataset_oversubscription to automatically adjust repeats\n"
        )
        if self.repeats > 0:
            error_msg += (
                f"\nNote: You have manually set repeats={self.repeats} in your dataset config.\n"
                f"--allow_dataset_oversubscription will not override manual repeats settings.\n"
                f"Either increase your repeats value or remove it from the config to allow auto-adjustment.\n"
            )
        raise ValueError(error_msg)

    def _build_share(self) -> dict[str, list[str]]:
        share = {}
        for bucket, count in self._counts.items():
            # Cache builders and cache reloads can hold the same set in different orders.
            order = sorted(self._buckets[bucket], key=str)
            random.Random(f"{self.seed}:{self.dataset_id}:{bucket}").shuffle(order)
            if bucket in self._auto_repeats:
                local = self._lengths[bucket]
                share[bucket] = [order[(self.dp_rank * local + offset) % count] for offset in range(local)]
                continue
            per_rank, extra = divmod(count, self.dp_size)
            start = self.dp_rank * per_rank + min(self.dp_rank, extra)
            # A bucket with fewer samples than ranks leaves this rank without one of its own.
            share[bucket] = order[start : start + per_rank + int(self.dp_rank < extra)] or [order[self.dp_rank % count]]
        return share

    def _start_epoch(self) -> None:
        self._orders = {}
        schedule = []
        for bucket, share in self._share.items():
            order = []
            for pass_index in range(self.repeats + 1):
                one_pass = list(share)
                random.Random(f"{self.seed}:{self.dataset_id}:{bucket}:{self.epoch}:{pass_index}").shuffle(one_pass)
                order.extend(one_pass)
            self._orders[bucket] = order
            remaining = self._lengths[bucket] - self._consumed[bucket]
            schedule.extend([bucket] * ceil(remaining / self.batch_size))
        random.Random(f"{self.seed}:{self.dataset_id}:{self.epoch}").shuffle(schedule)
        self._schedule = schedule
        self._next = 0

    def next_batch(self) -> tuple[str, list[str]]:
        if self._orders is None:
            self._start_epoch()
        if self._next >= len(self._schedule):
            self.finish_epoch()
            raise MultiDatasetExhausted()
        bucket = self._schedule[self._next]
        order = self._orders[bucket]
        start = self._consumed[bucket]
        # Reading past the end wraps to the head: a share shorter than the common length repeats
        # its first samples, and so does a final batch the bucket cannot fill.
        batch = [order[(start + offset) % len(order)] for offset in range(self.batch_size)]
        self._consumed[bucket] = min(start + self.batch_size, self._lengths[bucket])
        self._next += 1
        return bucket, batch

    def __len__(self) -> int:
        return sum(ceil(length / self.batch_size) for length in self._lengths.values())

    def progress(self) -> tuple[int, int]:
        """Samples consumed in this epoch, and samples in an epoch."""
        return sum(self._consumed.values()), sum(self._lengths.values())

    def rank_share(self) -> dict[str, list[str]]:
        """The distinct samples this rank trains on, by bucket. It does not change between epochs."""
        return {bucket: list(dict.fromkeys(share)) for bucket, share in self._share.items()}

    @property
    def counts(self) -> dict[str, int]:
        """The samples in each bucket when the cursor was built."""
        return dict(self._counts)

    def resume_at(self, consumed: Mapping[str, int]) -> None:
        """Continue the epoch with this many samples already drawn from each bucket."""
        # An oversubscribed bucket's length follows the batch size, so a count saved under another
        # batch size can exceed it.
        self._consumed = {bucket: min(consumed[bucket], self._lengths[bucket]) for bucket in self._counts}
        self._orders = None

    def finish_epoch(self) -> None:
        self.epoch += 1
        self._consumed = dict.fromkeys(self._counts, 0)
        self._orders = None

    def _layout(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "repeats": self.repeats,
            "dp_size": self.dp_size,
            "dp_rank": self.dp_rank,
            "counts": dict(self._counts),
        }

    def state(self) -> dict:
        return {
            "format": STATE_FORMAT,
            "epoch": self.epoch,
            "seed": self.seed,
            "consumed": dict(self._consumed),
            "layout": self._layout(),
        }

    def load_state(self, state: dict, *, allow_batch_size_change: bool) -> None:
        _require_current_format(self.dataset_id, state)
        saved, current = state["layout"], self._layout()
        if {**saved, "batch_size": current["batch_size"], "counts": current["counts"]} != current:
            raise _different_layout(self.dataset_id, saved, current)
        _check_batch_size(self.dataset_id, saved["batch_size"], self.batch_size, allow_batch_size_change)
        if state["seed"] != self.seed:
            self.seed = state["seed"]
            self._share = self._build_share()
        self.epoch = state["epoch"]
        if saved["counts"] != current["counts"]:
            logger.warning(
                f"Dataset '{self.dataset_id}' has changed since the checkpoint was written "
                f"({sum(saved['counts'].values())} samples then, {sum(current['counts'].values())} now). "
                f"Epoch {self.epoch} starts over."
            )
            self.resume_at(dict.fromkeys(self._counts, 0))
            return
        self.resume_at(state["consumed"])


class WindowedCursor:
    """One rank's traversal of a dataset that is indexed a unit at a time, such as a shard.

    Each epoch shuffles the units and deals them to the ranks. A rank opens a window of its units,
    trains on every sample in them, closes them and opens the next window.
    """

    def __init__(
        self,
        dataset_id: str,
        *,
        unit_count: int,
        open_unit: Callable[[int, int], Mapping[str, Sequence[str]]],
        close_unit: Callable[[int], None],
        window: int,
        batch_size: int,
        repeats: int,
        seed: int,
        dp_size: int,
        dp_rank: int,
    ):
        self.dataset_id = dataset_id
        self.unit_count = int(unit_count)
        self.window = int(window)
        self.batch_size = int(batch_size)
        self.repeats = int(repeats)
        self.seed = seed
        self.dp_size = int(dp_size)
        self.dp_rank = int(dp_rank)
        if self.window < 1:
            raise ValueError(f"Dataset '{dataset_id}': the window must hold at least one unit, not {window}.")
        if self.unit_count < self.dp_size:
            raise ValueError(
                f"Dataset '{dataset_id}' is divided into {self.unit_count} unit(s), fewer than the {self.dp_size} ranks"
                " reading it. Every rank needs a unit of its own."
            )
        self._open_unit = open_unit
        self._close_unit = close_unit
        # Samples a unit holds on average. Every rank is given the same figure, so they plan the
        # same number of steps for an epoch.
        self.samples_per_unit: float | None = None
        self.epoch = 1
        self._groups = self._deal_units()
        self._group = 0
        self._inner: DatasetCursor | None = None
        self._finished = 0

    def _deal_units(self) -> list[list[int]]:
        order = list(range(self.unit_count))
        random.Random(f"{self.seed}:{self.dataset_id}:units:{self.epoch}").shuffle(order)
        mine = order[self.dp_rank :: self.dp_size]
        return [mine[start : start + self.window] for start in range(0, len(mine), self.window)]

    def _open_group(self) -> None:
        pools: dict[str, list[str]] = {}
        for unit in self._groups[self._group]:
            for pool, refs in self._open_unit(unit, self.epoch).items():
                pools.setdefault(pool, []).extend(refs)
        self._inner = DatasetCursor(
            self.dataset_id,
            pools,
            batch_size=self.batch_size,
            repeats=self.repeats,
            seed=f"{self.seed}:{self.epoch}:{self._group}",
            dp_size=1,
            dp_rank=0,
            gradient_accumulation_steps=1,
            allow_oversubscription=False,
            # The window is this rank's alone, and a pool smaller than a batch is padded, not refused.
            shared_by_all_ranks=True,
        )

    def _close_group(self) -> None:
        for unit in self._groups[self._group]:
            self._close_unit(unit)
        self._inner = None

    def next_batch(self) -> tuple[str, list[str]]:
        while True:
            if self._inner is None:
                if self._group >= len(self._groups):
                    self.finish_epoch()
                    raise MultiDatasetExhausted()
                self._open_group()
            try:
                return self._inner.next_batch()
            except MultiDatasetExhausted:
                self._finished += self._inner.progress()[1]
                self._close_group()
                self._group += 1

    def measure_samples_per_unit(self) -> float:
        """The average over the window this rank reads first, which is opened to find out."""
        if self._inner is None:
            self._open_group()
        return sum(self._inner.counts.values()) / len(self._groups[self._group])

    def _estimated_samples(self) -> int:
        if self.samples_per_unit is None:
            raise RuntimeError(f"Dataset '{self.dataset_id}' has no estimate of its samples per unit yet.")
        units = sum(len(group) for group in self._groups)
        return ceil(units * self.samples_per_unit * (self.repeats + 1))

    def __len__(self) -> int:
        """An estimate: the units are only counted as they are opened."""
        return max(1, ceil(self._estimated_samples() / self.batch_size))

    def progress(self) -> tuple[int, int]:
        """Samples consumed in this epoch, and an estimate of the samples in an epoch."""
        consumed = self._finished + (self._inner.progress()[0] if self._inner is not None else 0)
        return consumed, max(consumed, self._estimated_samples())

    def rank_share(self) -> dict[str, list[str]]:
        raise RuntimeError(
            f"Dataset '{self.dataset_id}' is read a window at a time; the samples a rank trains on cannot be listed."
        )

    def finish_epoch(self) -> None:
        if self._inner is not None:
            self._close_group()
        self.epoch += 1
        self._groups = self._deal_units()
        self._group = 0
        self._finished = 0

    def _layout(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "repeats": self.repeats,
            "dp_size": self.dp_size,
            "dp_rank": self.dp_rank,
            "window": self.window,
            "units": self.unit_count,
        }

    def state(self) -> dict:
        return {
            "format": STATE_FORMAT,
            "epoch": self.epoch,
            "seed": self.seed,
            "group": self._group,
            "finished": self._finished,
            # The open window: how far each pool has been read, and the pool sizes that count refers to.
            "consumed": self._inner.state()["consumed"] if self._inner is not None else {},
            "counts": self._inner.counts if self._inner is not None else None,
            "layout": self._layout(),
        }

    def load_state(self, state: dict, *, allow_batch_size_change: bool) -> None:
        _require_current_format(self.dataset_id, state)
        saved, current = state["layout"], self._layout()
        if {**saved, "batch_size": current["batch_size"]} != current:
            raise _different_layout(self.dataset_id, saved, current)
        _check_batch_size(self.dataset_id, saved["batch_size"], self.batch_size, allow_batch_size_change)
        if self._inner is not None:
            self._close_group()
        self.seed = state["seed"]
        self.epoch = state["epoch"]
        self._groups = self._deal_units()
        self._group = state["group"]
        self._finished = state["finished"]
        if state["counts"] is None:
            return
        self._open_group()
        if self._inner.counts != state["counts"]:
            logger.warning(
                f"Dataset '{self.dataset_id}': the units open at the checkpoint held "
                f"{sum(state['counts'].values())} samples then and hold {sum(self._inner.counts.values())} now. "
                "That window starts over."
            )
            return
        self._inner.resume_at(state["consumed"])


def _require_current_format(dataset_id: str, state: dict) -> None:
    if state.get("format") != STATE_FORMAT:
        raise ValueError(
            f"Dataset '{dataset_id}': the checkpoint's sampler state was written by an older release "
            "and cannot be resumed."
        )


def _different_layout(dataset_id: str, saved: dict, current: dict) -> ValueError:
    return ValueError(
        f"Dataset '{dataset_id}': the checkpoint was written for a different dataset layout "
        f"(saved {_describe_layout(saved)}, current {_describe_layout(current)})."
    )


def _check_batch_size(dataset_id: str, saved: int, current: int, allow_change: bool) -> None:
    if saved == current:
        return
    if not allow_change:
        raise ValueError(
            f"Dataset '{dataset_id}' checkpoint batch_size={saved} does not match "
            f"current batch_size={current}. Resume with the same per-dataset train_batch_size, "
            f"or set --i_know_what_i_am_doing / SIMPLETUNER_ALLOW_MODIFYING_BSZ=1 to bypass."
        )
    logger.warning(
        f"Dataset '{dataset_id}': resuming with batch_size={current} but checkpoint recorded "
        f"batch_size={saved}. Proceeding because override is set."
    )


def _describe_layout(layout: dict) -> str:
    return " ".join(f"{key}={value}" for key, value in layout.items() if key not in ("batch_size", "counts"))
