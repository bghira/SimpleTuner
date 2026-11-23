"""BatchFetcher class for prefetching training batches."""

import logging
import os
import queue
import threading
from functools import partial
from typing import Any, Dict, Optional

from simpletuner.helpers.training.multi_process import should_log

from .dataloader_iterator import _set_epoch_step_for_training_step
from .dataloader_iterator import random_dataloader_iterator as _runtime_random_iterator

DEFAULT_KEEP_RUNNING_PROPERTY = None
DEFAULT_RANDOM_ITERATOR = _runtime_random_iterator

# legacy export for tests that patch this symbol directly
random_dataloader_iterator = _runtime_random_iterator

logger = logging.getLogger("DataBackendFactory")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel(logging.ERROR)

prefetch_log = logging.getLogger("DataBackendPrefetch")
if should_log():
    prefetch_log.setLevel(os.environ.get("SIMPLETUNER_PREFETCH_LOG_LEVEL", "INFO"))
else:
    prefetch_log.setLevel(logging.ERROR)


def prefetch_log_debug(message: str) -> None:
    from simpletuner.helpers.training.multi_process import rank_info

    prefetch_log.debug(f"{rank_info()} {message}")


class BatchFetcher:
    """Prefetch batches using the last consumed training step as the first schedule hint."""

    def __init__(
        self,
        step: int,
        max_size: int = 10,
        datasets: Optional[Dict[str, Any]] = None,
        device=None,
        device_prefetch_threshold_bytes: int = 0,
        mixer=None,
    ) -> None:
        if DEFAULT_KEEP_RUNNING_PROPERTY is not None:
            type(self).keep_running = DEFAULT_KEEP_RUNNING_PROPERTY
        self.queue = queue.Queue(max_size)
        self.datasets = datasets or {}
        self.mixer = mixer
        self._thread = None
        self._keep_running = True
        self._finished = threading.Event()
        self._space_available = threading.Event()
        self._error = None
        self._agreed_batches_remaining = 0
        self.step = step
        self._next_fetch_step = step
        self.device_prefetcher = None
        if getattr(device, "type", device) == "cuda" or device_prefetch_threshold_bytes > 0:
            from .device_prefetch import CudaBatchPrefetcher

            self.device_prefetcher = CudaBatchPrefetcher(
                device=device,
                minimum_bytes=device_prefetch_threshold_bytes,
            )

    def start_fetching(self) -> threading.Thread:
        thread = threading.Thread(target=self.fetch_responses)
        self._thread = thread
        thread.start()
        return thread

    def fetch_responses(self) -> None:
        """main loop for fetching responses in background thread"""
        prefetch_log_debug("Launching retrieval thread.")
        iterator_fn = random_dataloader_iterator
        iterator_options = {}
        if self.mixer is not None:
            iterator_options.update(update_epoch_step=False, mixer=self.mixer)
        elif iterator_fn is DEFAULT_RANDOM_ITERATOR:
            iterator_options["update_epoch_step"] = False
        try:
            while not self._finished.is_set():
                if self.queue.qsize() < self.queue.maxsize:
                    prefetch_log_debug(f"Queue size: {self.queue.qsize()}. Fetching more data.")
                    retrieve = partial(iterator_fn, self._next_fetch_step, self.datasets, **iterator_options)
                    item = self.device_prefetcher.fetch(retrieve) if self.device_prefetcher is not None else retrieve()
                    self.queue.put(item)
                    self._next_fetch_step += 1
                    if item is False:
                        break
                    if self.queue.qsize() >= self.queue.maxsize:
                        prefetch_log_debug("Completed fetching data. Queue is full.")
                        if threading.current_thread() is threading.main_thread():
                            break
                        continue
                else:
                    if threading.current_thread() is threading.main_thread():
                        break
                    self._space_available.clear()
                    # A consumer may have dequeued before the event was cleared.
                    if not self._finished.is_set() and self.keep_running and self.queue.qsize() >= self.queue.maxsize:
                        self._space_available.wait()
                if not self.keep_running:
                    break
        except StopIteration:
            pass
        except Exception as exc:
            self._error = exc
        finally:
            self._finished.set()
            if DEFAULT_KEEP_RUNNING_PROPERTY is not None:
                type(self).keep_running = DEFAULT_KEEP_RUNNING_PROPERTY
            prefetch_log_debug("Exiting retrieval thread.")

    def _dequeue(self):
        while True:
            try:
                item = self.queue.get(timeout=0.1)
                self._space_available.set()
                return item
            except queue.Empty:
                if self._finished.is_set() and self.queue.empty():
                    if self._error is not None:
                        raise self._error
                    return False

    def next_response(self, step: int, *, agree_on_count=None) -> Any:
        self.step = step
        if agree_on_count is not None and self._agreed_batches_remaining == 0:
            batch_error = None
            try:
                item = self._dequeue()
            except Exception as error:
                batch_error = error
                item = False
            count = 0
            if item is not False:
                # The producer only appends, and False can only be the last item.
                with self.queue.mutex:
                    queued = self.queue.queue
                    count = 1 + len(queued) - int(bool(queued) and queued[-1] is False)
            self._agreed_batches_remaining = agree_on_count(count, batch_error=batch_error)
            if self._agreed_batches_remaining == 0:
                return False
        else:
            item = self._dequeue()

        if agree_on_count is not None:
            self._agreed_batches_remaining -= 1
        if self.device_prefetcher is not None:
            item = self.device_prefetcher.consume(item)
        _set_epoch_step_for_training_step(step)
        return item

    def stop_fetching(self) -> None:
        self._keep_running = False
        self._finished.set()
        self._space_available.set()

    def close(self) -> None:
        self.stop_fetching()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()

    @property
    def keep_running(self) -> bool:
        return self._keep_running

    @keep_running.setter
    def keep_running(self, value: bool) -> None:
        self._keep_running = bool(value)
        if not self._keep_running:
            self._space_available.set()


# store default property so tests that monkey-patch keep_running can be restored safely
DEFAULT_KEEP_RUNNING_PROPERTY = BatchFetcher.keep_running
