"""Structured error reporting for subprocess-to-parent communication.

When training runs as a subprocess (via accelerate launch), this module writes
structured error information to a JSON file that the parent process can read.
This avoids parsing stdout/stderr for error messages.
"""

import json
import os
import traceback as tb_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Environment variable used to pass error file path from parent to subprocess
ERROR_FILE_ENV_VAR = "SIMPLETUNER_ERROR_FILE"


def get_error_file_path() -> Path | None:
    """Get the error file path from environment, if set by parent."""
    path_str = os.environ.get(ERROR_FILE_ENV_VAR)
    if path_str:
        return Path(path_str)
    return None


def write_error(
    exception: BaseException,
    traceback_str: str | None = None,
    context: dict[str, Any] | None = None,
) -> bool:
    """Write structured error information to the error file.

    Args:
        exception: The exception that occurred.
        traceback_str: Full traceback string. If None, will be extracted from exception.
        context: Optional additional context (e.g., current step, epoch).

    Returns:
        True if error was written successfully, False otherwise.
    """
    error_path = get_error_file_path()
    if error_path is None:
        return False

    if traceback_str is None:
        traceback_str = "".join(tb_module.format_exception(type(exception), exception, exception.__traceback__))

    error_data = {
        "type": "training.error",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "exception_type": type(exception).__name__,
        "message": str(exception),
        "traceback": traceback_str,
    }

    if context:
        error_data["context"] = context

    try:
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(error_data, f, indent=2)
        return True
    except Exception:
        return False


def read_error(error_path: Path | str) -> dict[str, Any] | None:
    """Read structured error information from an error file.

    Args:
        error_path: Path to the error file.

    Returns:
        Parsed error data dict, or None if file doesn't exist or is invalid.
    """
    path = Path(error_path)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("type") == "training.error":
            return data
    except Exception:
        pass
    return None


def cleanup_error_file(error_path: Path | str | None = None) -> None:
    """Remove error file if it exists.

    Args:
        error_path: Path to error file. If None, uses environment variable.
    """
    if error_path is None:
        error_path = get_error_file_path()
    if error_path is None:
        return

    path = Path(error_path)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass
