import logging
from collections import deque
from threading import Condition, Thread
from time import monotonic, sleep
from uuid import uuid4

import requests

_BATCH_INTERVAL = 1.0
_RETRY_INTERVAL = 5.0
_REQUEST_TIMEOUT = 5.0
_VALIDATION_TIMEOUT = (5.0, 20.0)
_VALIDATION_SHUTDOWN_TIMEOUT = 30.0
_VALIDATION_ATTEMPTS = 4
_VALIDATION_QUEUE_SIZE = 128


class WorkerTelemetrySender:
    """Send progress snapshots and completed validation outputs in the background."""

    def __init__(self, url: str, token: str):
        self._url = url
        self._validation_url = url.removesuffix("/status") + "/validation"
        self.callback_url = url.partition("/api/workers/job/")[0] + "/callback"
        self._headers = {"X-Worker-Token": token}
        self._pending: dict[str, dict] = {}
        self._validation_reports: deque[dict] = deque(maxlen=_VALIDATION_QUEUE_SIZE)
        self._condition = Condition()
        self._closed = False
        self._thread: Thread | None = None
        self._validation_thread: Thread | None = None

    def enqueue(self, report: dict) -> None:
        kind = "validation" if "validation" in report else "progress"
        with self._condition:
            if self._closed:
                return
            was_empty = not self._pending
            # Replacing a snapshot moves it behind any older snapshot of the other kind.
            self._pending.pop(kind, None)
            self._pending[kind] = report
            if self._thread is None:
                self._thread = Thread(target=self._send_reports, name="WorkerTelemetry", daemon=True)
                self._thread.start()
            if was_empty:
                self._condition.notify_all()

    def enqueue_validation(self, report: dict) -> None:
        with self._condition:
            if self._closed:
                return
            if len(self._validation_reports) == self._validation_reports.maxlen:
                logging.warning("Worker validation queue is full; dropping its oldest pending report")
            self._validation_reports.append({**report, "delivery_id": str(uuid4())})
            if self._validation_thread is None:
                self._validation_thread = Thread(target=self._send_validation_reports, name="WorkerValidation", daemon=True)
                self._validation_thread.start()
            self._condition.notify_all()

    def close(self) -> None:
        """Attempt a final flush with a bounded shutdown wait."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=_REQUEST_TIMEOUT)
        if self._validation_thread is not None:
            self._validation_thread.join(timeout=_VALIDATION_SHUTDOWN_TIMEOUT)
            if self._validation_thread.is_alive():
                logging.warning("Worker validation delivery did not finish before shutdown")

    def _send_reports(self) -> None:
        send_after = monotonic() + _BATCH_INTERVAL
        with requests.Session() as session:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._pending or self._closed)
                    if not self._pending:
                        return
                    self._condition.wait_for(lambda: self._closed, timeout=max(0, send_after - monotonic()))
                    batch, self._pending = self._pending, {}

                interval = self._send_batch(session, batch)
                send_after = monotonic() + interval

    def _send_validation_reports(self) -> None:
        with requests.Session() as session:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._validation_reports or self._closed)
                    if not self._validation_reports:
                        return
                    report = self._validation_reports.popleft()

                for attempt in range(_VALIDATION_ATTEMPTS):
                    try:
                        self._post(session, self._validation_url, report, timeout=_VALIDATION_TIMEOUT)
                    except requests.RequestException as exc:
                        response = exc.response
                        retryable = isinstance(
                            exc, (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)
                        ) or (response is not None and (response.status_code in {408, 429} or response.status_code >= 500))
                        if not retryable or attempt + 1 == _VALIDATION_ATTEMPTS:
                            logging.warning("Could not deliver worker validation output: %s", exc)
                            break
                        sleep(_RETRY_INTERVAL)
                    else:
                        break

    def _post(self, session: requests.Session, url: str, report: dict, *, timeout=_REQUEST_TIMEOUT) -> None:
        response = session.post(url, headers=self._headers, json=report, timeout=timeout, allow_redirects=False)
        response.raise_for_status()
        if response.status_code in range(300, 400):
            raise requests.HTTPError("Worker telemetry endpoint redirected the request", response=response)

    def _send_batch(self, session: requests.Session, batch: dict) -> float:
        while batch:
            kind = next(iter(batch))
            try:
                self._post(session, self._url, batch[kind])
            except requests.RequestException as exc:
                logging.warning("Could not report worker training progress: %s", exc)
                with self._condition:
                    if not self._closed:
                        unsuperseded = {key: report for key, report in batch.items() if key not in self._pending}
                        self._pending = unsuperseded | self._pending
                return _RETRY_INTERVAL
            del batch[kind]
        return _BATCH_INTERVAL
