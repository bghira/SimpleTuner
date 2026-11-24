import logging
from collections import deque
from threading import Condition, Thread
from time import monotonic

import requests

_BATCH_INTERVAL = 1.0
_RETRY_INTERVAL = 5.0
_REQUEST_TIMEOUT = 5.0
_VALIDATION_QUEUE_SIZE = 128


class WorkerTelemetrySender:
    """Deliver the latest training progress and validation snapshots in the background."""

    def __init__(self, url: str, token: str):
        self._url = url
        job_url = url.removesuffix("/status")
        self._validation_url = job_url + "/validation"
        self._datasets_url = job_url + "/datasets"
        self.callback_url = url.partition("/api/workers/job/")[0] + "/callback"
        self._headers = {"X-Worker-Token": token}
        self._pending: dict[str, dict] = {}
        self._validation_images: deque[dict] = deque(maxlen=_VALIDATION_QUEUE_SIZE)
        self._condition = Condition()
        self._closed = False
        self._thread: Thread | None = None

    def enqueue(self, report: dict, *, kind: str | None = None) -> None:
        kind = kind or ("validation" if "validation" in report else "progress")
        with self._condition:
            if self._closed:
                return
            was_empty = not self._pending
            # Replacing a snapshot moves it behind any older snapshot of the other kind.
            self._pending.pop(kind, None)
            self._pending[kind] = report
            self._start_thread()
            if was_empty:
                self._condition.notify()

    def enqueue_validation_images(self, report: dict) -> None:
        with self._condition:
            if self._closed:
                return
            self._validation_images.append(report)
            self._start_thread()
            self._condition.notify()

    def _start_thread(self) -> None:
        if self._thread is None:
            self._thread = Thread(target=self._send_reports, name="WorkerTelemetry", daemon=True)
            self._thread.start()

    def close(self) -> None:
        """Attempt a final flush with a bounded shutdown wait."""
        with self._condition:
            self._closed = True
            self._condition.notify()
        if self._thread is not None:
            self._thread.join(timeout=_REQUEST_TIMEOUT)

    def _send_reports(self) -> None:
        send_after = monotonic() + _BATCH_INTERVAL
        with requests.Session() as session:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._pending or self._validation_images or self._closed)
                    if not self._pending and not self._validation_images:
                        return
                    self._condition.wait_for(
                        lambda: self._closed or self._validation_images, timeout=max(0, send_after - monotonic())
                    )
                    batch = {}
                    if self._closed or monotonic() >= send_after:
                        batch, self._pending = self._pending, {}
                    images = self._validation_images.popleft() if self._validation_images else None

                if batch:
                    interval = self._send_batch(session, batch)
                    send_after = monotonic() + interval
                if images is not None:
                    try:
                        self._post(session, self._validation_url, images)
                    except requests.RequestException as exc:
                        logging.warning("Could not deliver worker validation images: %s", exc)

    def _post(self, session: requests.Session, url: str, report: dict) -> None:
        response = session.post(url, headers=self._headers, json=report, timeout=_REQUEST_TIMEOUT, allow_redirects=False)
        response.raise_for_status()

    def _send_batch(self, session: requests.Session, batch: dict) -> float:
        while batch:
            kind = next(iter(batch))
            try:
                self._post(session, self._datasets_url if kind == "datasets" else self._url, batch[kind])
            except requests.RequestException as exc:
                if kind == "datasets" and exc.response is not None and exc.response.status_code in {403, 404, 409, 422}:
                    del batch[kind]
                    continue
                logging.warning("Could not report worker %s: %s", kind, exc)
                with self._condition:
                    if not self._closed:
                        unsuperseded = {key: report for key, report in batch.items() if key not in self._pending}
                        self._pending = unsuperseded | self._pending
                return _RETRY_INTERVAL
            del batch[kind]
        return _BATCH_INTERVAL
