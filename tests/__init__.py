"""Test package initialization - suppress warnings before any imports."""

import logging
import os
import sys

# Set up logging configuration first
logging.basicConfig(level=logging.ERROR, force=True)

# Suppress the annoying PyTorch distributed elastic multiprocessing NOTE
# This MUST happen before torch is imported anywhere
for logger_name in [
    "torch.distributed.elastic.multiprocessing.redirects",
    "torch.distributed.elastic.multiprocessing",
    "torch.distributed.elastic",
    "torch.distributed",
    "torch",
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    logger.propagate = False

# Monkey-patch the logger's warning method to filter the specific message
_original_warning = logging.Logger.warning


def _filtered_warning(self, msg, *args, **kwargs):
    if "NOTE: Redirects are currently not supported" in str(msg):
        return
    return _original_warning(self, msg, *args, **kwargs)


logging.Logger.warning = _filtered_warning

# Also set environment variables to reduce noise
os.environ.setdefault("SIMPLETUNER_LOG_LEVEL", "ERROR")
os.environ.setdefault("ACCELERATE_LOG_LEVEL", "ERROR")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DATASETS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

# Register cleanup for test directories
import atexit
import shutil
from pathlib import Path


def cleanup_test_directories():
    """Clean up test directories on exit."""
    for dir_name in ["test-screenshots", "test-folder"]:
        dir_path = Path(dir_name)
        if dir_path.exists():
            if dir_path.is_dir():
                try:
                    if not any(dir_path.iterdir()):
                        # Empty directory - safe to remove
                        dir_path.rmdir()
                    else:
                        # Has contents - might be important, leave it
                        pass
                except Exception:
                    pass


atexit.register(cleanup_test_directories)

# Suppress the torch checkpoint use_reentrant warning
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly",
    category=UserWarning,
)
