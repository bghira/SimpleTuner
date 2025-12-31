"""Job log fetching and parsing utilities.

Handles fetching logs from cloud providers and local jobs,
and parsing training progress from log content.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

from .base import JobType
from .factory import ProviderFactory

if TYPE_CHECKING:
    from .base import UnifiedJob

logger = logging.getLogger(__name__)


@dataclass
class InlineProgress:
    """Compact progress info for job list display."""

    job_id: str
    stage: Optional[str] = None
    last_log: Optional[str] = None
    progress: Optional[float] = None


def parse_training_stage(lines: List[str]) -> Tuple[Optional[str], Optional[float]]:
    """Parse log lines to extract training stage and progress.

    Args:
        lines: Log lines to parse (searches in reverse for efficiency)

    Returns:
        Tuple of (stage_name, progress_percent)
    """
    stage = "Training"
    progress = None

    for line in reversed(lines[-50:]):
        line_lower = line.lower()

        if "preprocessing" in line_lower or "loading" in line_lower:
            stage = "Preprocessing"
            break
        elif "warming up" in line_lower or "warmup" in line_lower:
            stage = "Warmup"
            break
        elif "saving checkpoint" in line_lower or "checkpoint" in line_lower:
            stage = "Saving checkpoint"
            break
        elif "validat" in line_lower:
            stage = "Validation"
            break
        elif "epoch" in line_lower or "step" in line_lower:
            stage = "Training"
            step_match = re.search(r"step\s+(\d+)[/\s]+(\d+)", line_lower)
            if step_match:
                current, total = int(step_match.group(1)), int(step_match.group(2))
                if total > 0:
                    progress = round((current / total) * 100, 1)
                break

            epoch_match = re.search(r"epoch\s+(\d+)[/\s]+(\d+)", line_lower)
            if epoch_match:
                current, total = int(epoch_match.group(1)), int(epoch_match.group(2))
                if total > 0:
                    progress = round((current / total) * 100, 1)
                break

    return stage, progress


async def fetch_job_logs(job: "UnifiedJob", max_bytes: int = 50000) -> str:
    """Fetch logs for a job.

    Args:
        job: The job to fetch logs for
        max_bytes: Maximum bytes to read from log file (for local jobs)

    Returns:
        Log content as a string
    """
    if job.job_type == JobType.CLOUD and job.provider:
        return await _fetch_cloud_logs(job)
    elif job.job_type == JobType.LOCAL:
        return _fetch_local_logs(job, max_bytes)
    return ""


async def _fetch_cloud_logs(job: "UnifiedJob") -> str:
    """Fetch logs from cloud provider."""
    try:
        client = ProviderFactory.get_provider(job.provider)
        return await client.get_job_logs(job.job_id)
    except ValueError:
        return f"(Unknown provider: {job.provider})"
    except Exception as exc:
        logger.error("Error fetching logs for cloud job %s: %s", job.job_id, exc)
        return f"(Error fetching logs: {exc})"


def _find_log_path(output_dir: str, job_id: str) -> Optional[str]:
    """Find the log file path for a local job.

    Checks in order:
    1. debug.log in output_dir (legacy)
    2. stdout.log in .simpletuner_runtime/trainer_<job_id>_*/
    """
    # Legacy path
    legacy_path = os.path.join(output_dir, "debug.log")
    if os.path.exists(legacy_path):
        return legacy_path

    # Runtime directory path - look for trainer_<job_id>_* directories
    runtime_dir = os.path.join(output_dir, ".simpletuner_runtime")
    if os.path.isdir(runtime_dir):
        import glob

        pattern = os.path.join(runtime_dir, f"trainer_{job_id}_*", "stdout.log")
        matches = glob.glob(pattern)
        if matches:
            # Return the most recently modified one
            return max(matches, key=os.path.getmtime)

    return None


def _fetch_local_logs(job: "UnifiedJob", max_bytes: int = 50000) -> str:
    """Fetch logs from local job output directory."""
    output_dir = job.output_url
    if not output_dir:
        return "(No output directory recorded)"

    log_path = _find_log_path(output_dir, job.job_id)
    if not log_path:
        # Check if the runtime directory exists at all
        runtime_dir = os.path.join(output_dir, ".simpletuner_runtime")
        if os.path.isdir(runtime_dir):
            # Runtime dir exists but no matching job - job may have been cancelled early
            return f"(No log file found for job {job.job_id} - job may have been cancelled before training started)"
        return "(Log file not found - no training output directory exists)"

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)
            size = f.tell()
            if size > max_bytes:
                f.seek(-max_bytes, 2)
                f.readline()  # Skip partial line
            else:
                f.seek(0)
            return f.read()
    except Exception as exc:
        return f"(Error reading log file: {exc})"


async def get_inline_progress(job: "UnifiedJob") -> InlineProgress:
    """Get compact inline progress for job list display.

    Args:
        job: The job to get progress for

    Returns:
        InlineProgress with stage, last log line, and progress percentage
    """
    from .base import CloudJobStatus

    result = InlineProgress(job_id=job.job_id)

    if job.status != CloudJobStatus.RUNNING.value:
        return result

    if job.job_type == JobType.CLOUD and job.provider:
        try:
            client = ProviderFactory.get_provider(job.provider)
            logs = await client.get_job_logs(job.job_id)
            if logs:
                lines = logs.strip().split("\n")
                if lines:
                    result.last_log = _truncate_line(lines[-1])
                    result.stage, result.progress = parse_training_stage(lines)
        except Exception as exc:
            logger.debug("Error fetching inline progress for %s: %s", job.job_id, exc)

    elif job.job_type == JobType.LOCAL:
        output_dir = job.output_url
        if output_dir:
            log_path = _find_log_path(output_dir, job.job_id)
            if log_path:
                try:
                    with open(log_path, "rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        read_size = min(2048, size)
                        f.seek(-read_size, 2)
                        content = f.read().decode("utf-8", errors="replace")
                        lines = content.strip().split("\n")
                        if lines:
                            result.last_log = _truncate_line(lines[-1])
                            result.stage, result.progress = parse_training_stage(lines)
                except Exception as exc:
                    logger.debug("Error reading inline progress for local job %s: %s", job.job_id, exc)

    return result


def _truncate_line(line: str, max_length: int = 80) -> str:
    """Truncate a line for display."""
    line = line.strip()
    if len(line) > max_length:
        return line[: max_length - 3] + "..."
    return line


async def stream_local_logs(job: "UnifiedJob", poll_interval: float = 0.5):
    """Stream log lines from a local job as an async generator.

    Yields new lines as they appear in the log file, similar to `tail -f`.

    Args:
        job: The job to stream logs for
        poll_interval: How often to check for new content (seconds)

    Yields:
        Log lines as they appear
    """
    import asyncio

    output_dir = job.output_url
    if not output_dir:
        yield "(No output directory recorded)"
        return

    log_path = _find_log_path(output_dir, job.job_id)

    # Wait for log file to appear (up to 30 seconds)
    wait_time = 0
    while not log_path and wait_time < 30:
        await asyncio.sleep(1)
        wait_time += 1
        log_path = _find_log_path(output_dir, job.job_id)

    if not log_path:
        yield "(Log file not found)"
        return

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            # Start from beginning and stream all content
            while True:
                line = f.readline()
                if line:
                    yield line.rstrip("\n\r")
                else:
                    # No new content, wait and check again
                    await asyncio.sleep(poll_interval)

                    # Check if file was rotated/replaced
                    new_path = _find_log_path(output_dir, job.job_id)
                    if new_path and new_path != log_path:
                        # File changed, reopen
                        break

    except Exception as exc:
        yield f"(Error reading log file: {exc})"
