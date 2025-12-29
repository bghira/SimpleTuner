"""Structured JSON logging with correlation ID support.

Provides JSON log formatting and automatic correlation ID injection
for request tracing across the cloud training system.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .http_client import get_correlation_id


class StructuredLogFormatter(logging.Formatter):
    """JSON log formatter with correlation ID and structured fields.

    Output format:
    {
        "timestamp": "2024-01-15T10:30:00.000Z",
        "level": "INFO",
        "logger": "simpletuner.cloud",
        "message": "Job submitted",
        "correlation_id": "abc123",
        "job_id": "xyz789",
        ...
    }
    """

    # Fields to always include at the top level
    STANDARD_FIELDS = {
        "timestamp",
        "level",
        "logger",
        "message",
        "correlation_id",
    }

    def __init__(
        self,
        include_stack_info: bool = True,
        include_extra: bool = True,
        timestamp_format: str = "iso",  # "iso" or "unix"
    ):
        super().__init__()
        self.include_stack_info = include_stack_info
        self.include_extra = include_extra
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        # Base log entry
        log_entry: Dict[str, Any] = {}

        # Timestamp
        if self.timestamp_format == "unix":
            log_entry["timestamp"] = record.created
        else:
            log_entry["timestamp"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

        # Standard fields
        log_entry["level"] = record.levelname
        log_entry["logger"] = record.name
        log_entry["message"] = record.getMessage()

        # Correlation ID from thread-local storage
        correlation_id = get_correlation_id()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        # Source location
        log_entry["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Stack info
        if self.include_stack_info and record.stack_info:
            log_entry["stack_info"] = record.stack_info

        # Extra fields from the record
        if self.include_extra:
            extra = {}
            for key, value in record.__dict__.items():
                if key not in logging.LogRecord.__dict__ and key not in self.STANDARD_FIELDS and not key.startswith("_"):
                    # Try to serialize the value
                    try:
                        json.dumps(value)
                        extra[key] = value
                    except (TypeError, ValueError):
                        extra[key] = str(value)

            if extra:
                log_entry["extra"] = extra

        return json.dumps(log_entry, default=str)


class CorrelationIDFilter(logging.Filter):
    """Logging filter that adds correlation ID to all log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation_id to the record."""
        record.correlation_id = get_correlation_id()
        return True


def configure_structured_logging(
    level: str = "INFO",
    json_output: bool = True,
    log_file: Optional[str] = None,
    include_stack_info: bool = False,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output JSON. If False, use standard format.
        log_file: Optional file path for log output
        include_stack_info: Include stack traces in JSON output
    """
    # Determine log level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    if json_output:
        formatter = StructuredLogFormatter(
            include_stack_info=include_stack_info,
            include_extra=True,
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(correlation_id)s): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(CorrelationIDFilter())
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(CorrelationIDFilter())
        root_logger.addHandler(file_handler)

    # Configure specific loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with structured logging support.

    Args:
        name: Logger name (e.g., "simpletuner.cloud.jobs")

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    # Ensure the correlation ID filter is attached
    if not any(isinstance(f, CorrelationIDFilter) for f in logger.filters):
        logger.addFilter(CorrelationIDFilter())
    return logger


class LogContext:
    """Context manager for adding structured fields to logs.

    Usage:
        with LogContext(job_id="abc123", provider="replicate"):
            logger.info("Processing job")  # Will include job_id and provider
    """

    def __init__(self, **fields):
        self.fields = fields
        self.old_factory = None

    def __enter__(self):
        # Store the old factory
        self.old_factory = logging.getLogRecordFactory()

        # Create new factory that adds our fields
        fields = self.fields

        def new_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in fields.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(new_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the old factory
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)
        return False


def init_from_env() -> None:
    """Initialize logging from environment variables.

    Environment variables:
    - SIMPLETUNER_LOG_LEVEL: Log level (default: INFO)
    - SIMPLETUNER_LOG_FORMAT: "json" or "text" (default: json)
    - SIMPLETUNER_LOG_FILE: Optional log file path
    """
    level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")
    log_format = os.environ.get("SIMPLETUNER_LOG_FORMAT", "json")
    log_file = os.environ.get("SIMPLETUNER_LOG_FILE")

    configure_structured_logging(
        level=level,
        json_output=(log_format.lower() == "json"),
        log_file=log_file,
    )
