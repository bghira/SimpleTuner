"""
Utility functions for controlling logging during tests.
"""

import logging
import os
import sys
from contextlib import contextmanager
from typing import List, Optional


def silence_loggers(logger_names: Optional[List[str]] = None):
    """
    Silence specific loggers or all loggers.

    Args:
        logger_names: List of logger names to silence. If None, silences all.
    """
    if logger_names is None:
        # Common noisy loggers in SimpleTuner
        logger_names = [
            "simpletuner",
            "SimpleTuner",
            "SimpleTunerSDK",
            "ArgsParser",
            "torch.distributed",
            "torch.distributed.elastic",
            "accelerate",
            "transformers",
            "datasets",
            "diffusers",
            "PIL",
            "urllib3",
            "asyncio",
            "multiprocess",
        ]

    for name in logger_names:
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False


def setup_test_logging(level=logging.ERROR):
    """
    Set up logging for tests with minimal output.

    Args:
        level: Logging level to use (default: ERROR)
    """
    # Configure root logger
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", stream=sys.stderr)

    # Silence noisy loggers
    silence_loggers()

    # Set environment variables
    os.environ["SIMPLETUNER_LOG_LEVEL"] = "ERROR"
    os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"


@contextmanager
def suppress_logs(level=logging.CRITICAL):
    """
    Context manager to temporarily suppress logging.

    Usage:
        with suppress_logs():
            # Code that produces unwanted logs
            train_model()
    """
    # Save current settings
    root_logger = logging.getLogger()
    old_level = root_logger.level
    old_handlers = root_logger.handlers[:]

    try:
        # Set to critical to suppress most logs
        root_logger.setLevel(level)

        # Remove all handlers
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

        # Add null handler
        null_handler = logging.NullHandler()
        root_logger.addHandler(null_handler)

        yield

    finally:
        # Restore original settings
        root_logger.setLevel(old_level)
        root_logger.removeHandler(null_handler)
        for handler in old_handlers:
            root_logger.addHandler(handler)


@contextmanager
def capture_logs(logger_name: str = None, level=logging.INFO):
    """
    Context manager to capture logs for testing.

    Usage:
        with capture_logs('simpletuner') as logs:
            function_that_logs()
        assert 'expected message' in logs.output
    """
    import io

    class LogCapture:
        def __init__(self):
            self.output = ""
            self.records = []

    capture = LogCapture()
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()

    # Create string stream handler
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)

    # Add handler
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(level)

    try:
        yield capture
    finally:
        # Get captured output
        capture.output = stream.getvalue()

        # Clean up
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        stream.close()


class LogLevelContext:
    """
    Class-based context manager for controlling log levels in tests.

    Usage:
        def test_something():
            with LogLevelContext(logging.ERROR):
                # Only ERROR and above will be logged
                run_training()
    """

    def __init__(self, level=logging.ERROR, loggers=None):
        self.level = level
        self.loggers = loggers or []
        self.original_levels = {}

    def __enter__(self):
        # Save and update levels
        for logger_name in self.loggers:
            logger = logging.getLogger(logger_name)
            self.original_levels[logger_name] = logger.level
            logger.setLevel(self.level)

        # Also set root if no specific loggers
        if not self.loggers:
            root = logging.getLogger()
            self.original_levels["__root__"] = root.level
            root.setLevel(self.level)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original levels
        for logger_name, original_level in self.original_levels.items():
            if logger_name == "__root__":
                logging.getLogger().setLevel(original_level)
            else:
                logging.getLogger(logger_name).setLevel(original_level)
