"""Helpers for resolving external cloud job identifiers."""

from __future__ import annotations

from typing import Optional

from .base import CloudJobStatus, UnifiedJob


def get_external_job_id(job: UnifiedJob) -> Optional[str]:
    """Return the provider job ID for provider API calls."""
    prediction_id = job.metadata.get("prediction_id")
    if prediction_id:
        return prediction_id

    if job.status == CloudJobStatus.UPLOADING.value:
        return None

    return job.job_id
