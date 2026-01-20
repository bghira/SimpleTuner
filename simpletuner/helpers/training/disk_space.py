"""Disk space monitoring utilities for checkpoint saves."""

import logging
import re
import shutil
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("DiskSpaceMonitor")


class DiskLowAction(str, Enum):
    """Action to take when disk space is below threshold."""

    STOP = "stop"
    WAIT = "wait"
    SCRIPT = "script"

    @classmethod
    def from_raw(cls, raw_value: Any) -> "DiskLowAction":
        """Convert a raw config/CLI value to DiskLowAction enum."""
        if isinstance(raw_value, cls):
            return raw_value
        if raw_value in (None, "", "None"):
            return cls.STOP
        normalized = str(raw_value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            valid_values = ", ".join(member.value for member in cls)
            raise ValueError(f"Unsupported disk_low_action '{raw_value}'. Expected one of: {valid_values}") from exc


def parse_size_threshold(threshold_str: Optional[str]) -> Optional[int]:
    """
    Parse a human-readable size string into bytes.

    Args:
        threshold_str: Size string like "100G", "50M", "1T", "500K", or plain bytes.
                       Returns None if threshold_str is None/empty (feature disabled).

    Returns:
        Size in bytes, or None if feature is disabled.

    Raises:
        ValueError: If the format is invalid.
    """
    if threshold_str in (None, "", "None"):
        return None

    threshold_str = str(threshold_str).strip().upper()

    match = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT]?)B?$", threshold_str)
    if not match:
        raise ValueError(
            f"Invalid disk_low_threshold format: '{threshold_str}'. "
            "Expected format like '100G', '50M', '1T', '500K', or plain bytes."
        )

    value = float(match.group(1))
    unit = match.group(2)

    multipliers = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
    }

    return int(value * multipliers[unit])


def get_available_disk_space(path: str) -> int:
    """
    Return available disk space in bytes for the filesystem containing path.

    If the path doesn't exist, traverses parent directories to find an existing one.
    """
    resolved_path = Path(path).resolve()
    while not resolved_path.exists() and resolved_path.parent != resolved_path:
        resolved_path = resolved_path.parent

    usage = shutil.disk_usage(str(resolved_path))
    return usage.free


def _format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(value) < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"


def check_disk_space(
    output_dir: str,
    threshold_bytes: int,
    action: DiskLowAction,
    script_path: Optional[str] = None,
    check_interval: int = 30,
) -> None:
    """
    Check if available disk space is below threshold and take configured action.

    Args:
        output_dir: Directory to check disk space for.
        threshold_bytes: Minimum required free space in bytes.
        action: Action to take when space is low.
        script_path: Path to cleanup script (required when action is SCRIPT).
        check_interval: Seconds between checks in WAIT mode.

    Raises:
        RuntimeError: When action is STOP, or when SCRIPT fails, or when
                      space remains low after SCRIPT execution.
    """
    available = get_available_disk_space(output_dir)

    if available >= threshold_bytes:
        return

    available_human = _format_bytes(available)
    threshold_human = _format_bytes(threshold_bytes)

    if action == DiskLowAction.STOP:
        raise RuntimeError(
            f"Disk space critically low: {available_human} available, " f"threshold is {threshold_human}. Training stopped."
        )

    elif action == DiskLowAction.WAIT:
        logger.warning(
            "Disk space low: %s available (threshold: %s). " "Waiting for space to become available...",
            available_human,
            threshold_human,
        )
        while available < threshold_bytes:
            time.sleep(check_interval)
            available = get_available_disk_space(output_dir)
        logger.info(
            "Disk space recovered: %s available. Resuming training.",
            _format_bytes(available),
        )

    elif action == DiskLowAction.SCRIPT:
        if not script_path:
            raise RuntimeError("disk_low_action is 'script' but no disk_low_script configured.")
        logger.warning(
            "Disk space low: %s available (threshold: %s). Running cleanup script: %s",
            available_human,
            threshold_human,
            script_path,
        )
        try:
            subprocess.run([script_path], check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Disk cleanup script failed with exit code {exc.returncode}") from exc
        except FileNotFoundError as exc:
            raise RuntimeError(f"Disk cleanup script not found: {script_path}") from exc

        available = get_available_disk_space(output_dir)
        if available < threshold_bytes:
            raise RuntimeError(
                f"Disk space still low after cleanup script: "
                f"{_format_bytes(available)} available, threshold is {threshold_human}."
            )
        logger.info("Disk cleanup script completed. %s now available.", _format_bytes(available))
