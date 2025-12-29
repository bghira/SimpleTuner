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


def _fetch_local_logs(job: "UnifiedJob", max_bytes: int = 50000) -> str:
    """Fetch logs from local job output directory."""
    output_dir = job.metadata.get("output_dir")
    if not output_dir:
        return "(No output directory recorded)"

    log_path = os.path.join(output_dir, "debug.log")
    if not os.path.exists(log_path):
        return "(Log file not found)"

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
        output_dir = job.metadata.get("output_dir")
        if output_dir:
            log_path = os.path.join(output_dir, "debug.log")
            if os.path.exists(log_path):
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
