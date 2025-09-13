# test_config.py - Common test configuration

import logging
import os
import sys
import warnings
from contextlib import redirect_stderr
from io import StringIO

# Set a reasonable log level that reduces noise but doesn't break tests
# Tests that use assertLogs() need logging to work
os.environ.setdefault("SIMPLETUNER_LOG_LEVEL", "ERROR")

# Suppress specific warnings that commonly appear in tests
warnings.filterwarnings("ignore", category=UserWarning, module="torch.amp.autocast_mode")
warnings.filterwarnings("ignore", message="NOTE: Redirects are currently not supported")
warnings.filterwarnings("ignore", message="Warning: Detected no triton")

# Suppress opencv duplicate class warnings (these are harmless but noisy)
warnings.filterwarnings("ignore", message=".*Class.*is implemented in both.*")

# For system-level warnings (like OpenCV objc warnings), we can redirect stderr
# but this is more aggressive and might hide actual errors, so use cautiously
_original_stderr = sys.stderr


def suppress_noisy_warnings():
    """Suppress common noisy warnings that don't affect test results."""
    # This is a lighter approach than completely disabling logging
    pass


def suppress_system_warnings():
    """Suppress system-level warnings by redirecting stderr."""
    sys.stderr = StringIO()


def restore_system_warnings():
    """Restore normal stderr output."""
    sys.stderr = _original_stderr


# Context managers for selective logging control
class QuietLogs:
    """Context manager to temporarily suppress logging for specific tests."""

    def __init__(self, level=logging.CRITICAL):
        self.level = level
        self.old_level = None

    def __enter__(self):
        self.old_level = logging.root.level
        logging.root.setLevel(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.root.setLevel(self.old_level)


def setup_test_environment():
    """Set up test environment with minimal logging and warnings."""
    pass  # Configuration is done at module import time
