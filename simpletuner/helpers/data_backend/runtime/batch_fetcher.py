"""BatchFetcher class for prefetching training batches."""

import logging
import os
import queue
import threading
import time
from typing import Dict, Any, Optional

from simpletuner.helpers.training.multi_process import should_log
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


def _resolve_random_iterator():
    if random_dataloader_iterator is not DEFAULT_RANDOM_ITERATOR:
        return random_dataloader_iterator

    try:
        from simpletuner.helpers.data_backend import factory as factory_module

        return getattr(factory_module, "random_dataloader_iterator", DEFAULT_RANDOM_ITERATOR)
    except Exception:
        return DEFAULT_RANDOM_ITERATOR


class BatchFetcher:

    def __init__(self, step: int, max_size: int = 10, datasets: Optional[Dict[str, Any]] = None) -> None:
        global DEFAULT_KEEP_RUNNING_PROPERTY
        if DEFAULT_KEEP_RUNNING_PROPERTY is not None:
            type(self).keep_running = DEFAULT_KEEP_RUNNING_PROPERTY
        self.queue = queue.Queue(max_size)
        self.datasets = datasets or {}
        self._keep_running = True
        self.step = step

    def start_fetching(self) -> threading.Thread:
        thread = threading.Thread(target=self.fetch_responses)
        thread.start()
        return thread

    def fetch_responses(self) -> None:
        """main loop for fetching responses in background thread"""
        prefetch_log_debug("Launching retrieval thread.")
        iterator_fn = _resolve_random_iterator()
        try:
            while True:
                if self.queue.qsize() < self.queue.maxsize:
                    prefetch_log_debug(f"Queue size: {self.queue.qsize()}. Fetching more data.")
                    try:
                        item = iterator_fn(self.step, self.datasets)
                    except ValueError:
                        prefetch_log_debug("No datasets available during prefetch; stopping fetch thread.")
                        self._keep_running = False
                        break
                    except StopIteration:
                        prefetch_log_debug("Dataloader iterator signaled completion.")
                        break
                    except Exception as exc:
                        logger.debug(f"BatchFetcher encountered exception: {exc}")
                        break
                    self.queue.put(item)
                    if self.queue.qsize() >= self.queue.maxsize:
                        prefetch_log_debug("Completed fetching data. Queue is full.")
                        if threading.current_thread() is threading.main_thread():
                            break
                        continue
                else:
                    if threading.current_thread() is threading.main_thread():
                        break
                    time.sleep(0.5)
                if not self.keep_running:
                    break
        finally:
            if DEFAULT_KEEP_RUNNING_PROPERTY is not None:
                type(self).keep_running = DEFAULT_KEEP_RUNNING_PROPERTY
            prefetch_log_debug("Exiting retrieval thread.")

    def next_response(self, step: int) -> Any:
        self.step = step
        if self.queue.empty():
            prefetch_log_debug("Queue is empty. Waiting for data.")
        while self.queue.empty():
            continue
        prefetch_log_debug("Queue has data. Yielding next item.")
        return self.queue.get()

    def stop_fetching(self) -> None:
        self._keep_running = False

    @property
    def keep_running(self) -> bool:
        return self._keep_running

    @keep_running.setter
    def keep_running(self, value: bool) -> None:
        self._keep_running = bool(value)


# store default property so tests that monkey-patch keep_running can be restored safely
DEFAULT_KEEP_RUNNING_PROPERTY = BatchFetcher.keep_running
