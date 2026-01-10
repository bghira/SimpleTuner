from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Full, Queue
from time import monotonic, sleep
from typing import Any, Optional

from simpletuner.helpers.training.multi_process import should_log

INTERNAL_LOGGER_NAME = "simpletuner.webhooklogger"
_thread_local = threading.local()
_task_queue: Queue["_WebhookTask"] = Queue(maxsize=int(os.environ.get("SIMPLETUNER_WEBHOOK_QUEUE_SIZE", "512")))
_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()
_no_webhook_logged = False

_EXCLUDED_LOGGER_PREFIXES: tuple[str, ...] = tuple(
    os.environ.get("SIMPLETUNER_WEBHOOK_LOGGER_EXCLUDE", "uvicorn,gunicorn,werkzeug,urllib3").split(",")
)


def _is_excluded_logger(name: str | None) -> bool:
    if not name:
        return True  # root logger
    lowered = name.lower()
    for prefix in _EXCLUDED_LOGGER_PREFIXES:
        prefix = prefix.strip().lower()
        if not prefix:
            continue
        if lowered == prefix or lowered.startswith(prefix + "."):
            return True
    return False


@dataclass
class _WebhookTask:
    handler: Any
    severity: str
    text_message: str
    structured_payload: Optional[dict[str, Any]]
    images: Optional[list[Any]] = None
    videos: Optional[list[Any]] = None


def _ensure_worker():
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return
    with _worker_lock:
        if _worker_thread and _worker_thread.is_alive():
            return
        _worker_thread = threading.Thread(target=_webhook_worker_loop, name="WebhookLoggerWorker", daemon=True)
        _worker_thread.start()


def _webhook_worker_loop():
    while True:
        task = _task_queue.get()
        try:
            if task is None:
                continue

            with _suspend_webhook():
                if task.structured_payload and hasattr(task.handler, "send_raw"):
                    try:
                        task.handler.send_raw(
                            structured_data=task.structured_payload,
                            message_type=task.structured_payload.get("type"),
                            message_level=task.severity,
                            job_id=task.structured_payload.get("job_id"),
                            images=task.images,
                            videos=task.videos,
                        )
                    except Exception:
                        logging.getLogger(INTERNAL_LOGGER_NAME).debug(
                            "Failed to forward structured log message to webhook.", exc_info=True
                        )

                if hasattr(task.handler, "send"):
                    try:
                        task.handler.send(
                            message=task.text_message,
                            message_level=task.severity,
                            images=task.images,
                            videos=task.videos,
                        )
                    except Exception:
                        logging.getLogger(INTERNAL_LOGGER_NAME).debug(
                            "Failed to forward text log message to webhook.", exc_info=True
                        )
        finally:
            _task_queue.task_done()


def _enqueue_webhook_task(task: _WebhookTask):
    _ensure_worker()
    try:
        _task_queue.put_nowait(task)
    except Full:
        logging.getLogger(INTERNAL_LOGGER_NAME).debug("Webhook logger queue full; dropping log message.")


def flush_webhook_queue(timeout: float | None = None) -> bool:
    """
    Block until pending webhook tasks are processed.

    Returns True if the queue drained before timeout (or if timeout is None), False otherwise.
    """
    if timeout is None:
        _task_queue.join()
        return True

    deadline = monotonic() + max(0.0, timeout)
    while monotonic() < deadline:
        if _task_queue.unfinished_tasks == 0:
            return True
        sleep(0.01)
    return _task_queue.unfinished_tasks == 0


def _to_numeric_level(level: Any) -> int:
    """Convert dotted/logging names or numeric strings to logging levels."""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        candidate = level.strip()
        if not candidate:
            return logging.INFO
        upper = candidate.upper()
        if upper in logging._nameToLevel:
            return logging._nameToLevel[upper]
        try:
            return int(candidate)
        except ValueError:
            return logging.INFO
    return logging.INFO


def _extract_webhook_config(args: Any) -> Optional[Any]:
    """Retrieve a webhook configuration from a namespace/dict-like structure."""
    if args is None:
        return None

    if isinstance(args, dict):
        for key in ("webhook_config", "--webhook_config"):
            value = args.get(key)
            if value:
                return value
        return None

    for key in ("webhook_config", "__webhook_config", "--webhook_config"):
        value = getattr(args, key, None)
        if value:
            return value

    data = getattr(args, "__dict__", {})
    if isinstance(data, dict):
        for key in ("webhook_config", "--webhook_config"):
            value = data.get(key)
            if value:
                return value

    return None


def _load_env_webhook_config() -> Optional[Any]:
    """Attempt to load webhook configuration from SIMPLETUNER_WEBHOOK_CONFIG env variable."""
    raw_value = os.environ.get("SIMPLETUNER_WEBHOOK_CONFIG")
    if not raw_value:
        return None

    candidate = raw_value.strip()
    if not candidate:
        return None

    # Try JSON
    try:
        import json

        return json.loads(candidate)
    except Exception:
        pass

    # Try path to JSON file
    expanded = os.path.expanduser(candidate)
    if os.path.isfile(expanded):
        try:
            import json

            with open(expanded, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    return None


def _fallback_webhook_config() -> Optional[Any]:
    """Fallback to default webhook configuration when nothing else is available."""
    global _no_webhook_logged
    env_config = _load_env_webhook_config()
    if env_config:
        return env_config

    if not _no_webhook_logged:
        logging.getLogger(INTERNAL_LOGGER_NAME).debug("No webhook configuration provided; webhook forwarding disabled.")
        _no_webhook_logged = True
    return None


def _build_project_name(args: Any) -> str:
    """
    Compose a human-readable project label for webhook messages.
    """
    parts = []

    for attr in ("tracker_project_name", "tracker_run_name", "project", "run_name"):
        value = getattr(args, attr, None)
        if value:
            parts.append(str(value))

    candidate = " ".join(part for part in parts if part).strip()
    return candidate or "SimpleTuner"


@contextmanager
def _suspend_webhook():
    """Temporarily disable webhook forwarding for the current thread."""
    previous = getattr(_thread_local, "skip_webhook", False)
    _thread_local.skip_webhook = True
    try:
        yield
    finally:
        _thread_local.skip_webhook = previous


class WebhookLogger(logging.Logger):
    """
    Logger subclass that mirrors standard logging.Logger behaviour while optionally
    mirroring messages to configured webhooks via the StateTracker.
    """

    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)
        self._level_env_var: Optional[str] = None
        self._default_level_name: str = "INFO"
        self._webhook_disabled: bool = _is_excluded_logger(name)
        self._level_managed: bool = True
        self._manual_level_override: bool = False
        self._apply_managed_level()

    # --------------------------------------------------------------------- config
    def configure(
        self,
        *,
        env_var: str | None = None,
        default_level: str = "INFO",
        disable_webhook: bool = False,
        propagate: bool | None = None,
    ) -> None:
        """
        Configure how this logger interprets log levels and whether it should
        push to webhooks.
        """
        manual_override = self._manual_level_override
        self._level_env_var = env_var
        self._default_level_name = default_level
        self._webhook_disabled = disable_webhook
        self._level_managed = True
        if propagate is not None:
            self.propagate = propagate
        if not manual_override:
            self._apply_managed_level()

    def setLevel(self, level: int | str) -> None:  # noqa: N802 - keeping logging API
        """
        Maintain logging.Logger API while signalling that manual overrides
        should persist over future environment updates.
        """
        super().setLevel(level)
        self._manual_level_override = True

    # ------------------------------------------------------------------- overrides
    def _apply_managed_level(self) -> None:
        if not self._level_managed or self._manual_level_override:
            return

        target_level = self._resolve_level()
        if self.level != target_level:
            super().setLevel(target_level)

    def _resolve_level(self) -> int:
        level_source = None
        if self._level_env_var:
            level_source = os.environ.get(self._level_env_var)
        if not level_source:
            level_source = os.environ.get("SIMPLETUNER_LOG_LEVEL", self._default_level_name)

        if not should_log():
            return _to_numeric_level(os.environ.get("SIMPLETUNER_WORKER_LOG_LEVEL", "ERROR"))

        return _to_numeric_level(level_source)

    def _should_forward_to_webhook(self) -> bool:
        if self._webhook_disabled:
            return False
        return not getattr(_thread_local, "skip_webhook", False)

    def _extract_message(self, msg: Any, args: tuple[Any, ...]) -> str:
        if not args:
            return str(msg)
        try:
            return str(msg) % args
        except Exception:
            return f"{msg} {args}"

    def _get_webhook_handler(self):
        try:
            from simpletuner.helpers.training.state_tracker import StateTracker
        except Exception:
            return None

        get_handler = getattr(StateTracker, "get_webhook_handler", None)
        set_handler = getattr(StateTracker, "set_webhook_handler", None)
        get_args = getattr(StateTracker, "get_args", None)
        get_accelerator = getattr(StateTracker, "get_accelerator", None)

        if get_handler is None:
            return None

        handler = get_handler()
        if handler is not None:
            return handler

        args = get_args() if callable(get_args) else None
        webhook_config = _extract_webhook_config(args)
        if not webhook_config:
            webhook_config = _fallback_webhook_config()
        if not webhook_config:
            return None

        accelerator = get_accelerator() if callable(get_accelerator) else None
        project_name = _build_project_name(args)

        with _suspend_webhook():
            try:
                from simpletuner.helpers.webhooks.handler import WebhookHandler

                handler = WebhookHandler(
                    accelerator,
                    project_name,
                    webhook_config=webhook_config,
                )
            except Exception:
                logging.getLogger(INTERNAL_LOGGER_NAME).debug(
                    "Unable to instantiate webhook handler for logging.", exc_info=True
                )
                return None

        if handler and callable(set_handler):
            set_handler(handler)

        return handler

    def _forward_to_webhook(self, level: int, msg: Any, args: tuple[Any, ...]) -> None:
        handler = self._get_webhook_handler()
        if handler is None:
            return

        level_name = logging.getLevelName(level)
        if isinstance(level_name, int):
            level_name = str(level)
        message = self._extract_message(msg, args)

        job_id = None
        try:
            from simpletuner.helpers.training.state_tracker import StateTracker

            job_id_getter = getattr(StateTracker, "get_job_id", None)
            if callable(job_id_getter):
                job_id = job_id_getter()
        except Exception:
            job_id = None

        severity = str(level_name).lower()
        structured = {
            "type": "log.message",
            "logger": self.name,
            "message": message,
            "severity": severity,
        }
        if job_id is not None:
            structured["job_id"] = job_id

        decorated = f"[{self.name}] {message}" if self.name else message
        _enqueue_webhook_task(
            _WebhookTask(
                handler=handler,
                severity=severity,
                text_message=decorated,
                structured_payload=structured,
            )
        )

    # pylint: disable=arguments-differ
    def _log(
        self,
        level: int,
        msg: Any,
        args: tuple[Any, ...],
        exc_info=None,
        extra=None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> None:
        self._apply_managed_level()
        if "Redirects are currently not supported" in str(msg):
            return
        super()._log(level, msg, args, exc_info=exc_info, extra=extra, stack_info=stack_info, stacklevel=stacklevel)
        if self._should_forward_to_webhook():
            self._forward_to_webhook(level, msg, args)


def get_logger(
    name: str,
    *,
    env_var: str | None = None,
    default_level: str = "INFO",
    disable_webhook: bool = False,
    propagate: bool | None = None,
) -> WebhookLogger:
    """
    Retrieve a WebhookLogger with consistent configuration across the project.
    """
    logger = logging.getLogger(name)
    propagate_value = propagate if propagate is not None else False
    if isinstance(logger, WebhookLogger):
        logger.configure(
            env_var=env_var,
            default_level=default_level,
            disable_webhook=disable_webhook,
            propagate=propagate_value,
        )
    else:
        if propagate_value is not None:
            logger.propagate = propagate_value
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger  # type: ignore[return-value]


# Ensure new loggers leverage WebhookLogger behaviour.
logging.setLoggerClass(WebhookLogger)

# Internal logger for diagnostics; disable webhook forwarding to avoid loops.
internal_logger = get_logger(INTERNAL_LOGGER_NAME, disable_webhook=True)
internal_logger.propagate = True


class WebhookForwardingHandler(logging.Handler):
    """Handler that forwards all log messages to webhooks, even from non-WebhookLogger instances."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Skip internal webhook logger to avoid loops
            if record.name.startswith(INTERNAL_LOGGER_NAME):
                return

            # Skip if webhook is suspended for this thread
            if getattr(_thread_local, "skip_webhook", False):
                return

            # Get webhook handler from StateTracker
            try:
                from simpletuner.helpers.training.state_tracker import StateTracker

                get_handler = getattr(StateTracker, "get_webhook_handler", None)
                if not get_handler:
                    return

                handler = get_handler()
                if not handler:
                    return

                # Get job ID if available
                job_id = None
                job_id_getter = getattr(StateTracker, "get_job_id", None)
                if callable(job_id_getter):
                    job_id = job_id_getter()

                # Format the message
                message = self.format(record)
                severity = record.levelname.lower()

                # Create structured data
                structured = {
                    "type": "log.message",
                    "logger": record.name,
                    "message": message,
                    "severity": severity,
                }
                if job_id:
                    structured["job_id"] = job_id

                # Enqueue for webhook sending
                decorated = f"[{record.name}] {message}" if record.name else message
                _enqueue_webhook_task(
                    _WebhookTask(
                        handler=handler,
                        severity=severity,
                        text_message=decorated,
                        structured_data=structured,
                    )
                )
            except Exception:
                # Silently fail to avoid disrupting logging
                pass
        except Exception:
            pass


# Install the forwarding handler on the root logger to catch all log messages
def _install_root_forwarding_handler():
    """Install a handler on the root logger to forward all messages to webhooks."""
    root_logger = logging.getLogger()
    # Check if already installed
    for handler in root_logger.handlers:
        if isinstance(handler, WebhookForwardingHandler):
            return
    # Install the handler
    forwarding_handler = WebhookForwardingHandler()
    forwarding_handler.setLevel(logging.INFO)  # Only forward INFO and above
    root_logger.addHandler(forwarding_handler)


# Install it immediately
_install_root_forwarding_handler()
