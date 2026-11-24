import logging
from threading import Condition, Thread
from time import monotonic

import requests

_BATCH_INTERVAL = 1.0
_RETRY_INTERVAL = 5.0
_REQUEST_TIMEOUT = 5.0


class WorkerTelemetrySender:
    """Deliver the latest training progress and validation snapshots in the background."""

    def __init__(self, url: str, token: str):
        self._url = url
        self._headers = {"X-Worker-Token": token}
        self._pending: dict[str, dict] = {}
        self._condition = Condition()
        self._closed = False
        self._thread: Thread | None = None

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
                self._condition.notify()

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
                    self._condition.wait_for(lambda: self._pending or self._closed)
                    if not self._pending:
                        return
                    self._condition.wait_for(lambda: self._closed, timeout=max(0, send_after - monotonic()))
                    batch, self._pending = self._pending, {}

                interval = _BATCH_INTERVAL
                while batch:
                    kind = next(iter(batch))
                    try:
                        response = session.post(
                            self._url,
                            headers=self._headers,
                            json=batch[kind],
                            timeout=_REQUEST_TIMEOUT,
                            allow_redirects=False,
                        )
                        response.raise_for_status()
                    except requests.RequestException as exc:
                        logging.warning("Could not report worker training progress: %s", exc)
                        with self._condition:
                            if not self._closed:
                                unsuperseded = {key: report for key, report in batch.items() if key not in self._pending}
                                self._pending = unsuperseded | self._pending
                        interval = _RETRY_INTERVAL
                        break
                    del batch[kind]
                send_after = monotonic() + interval
