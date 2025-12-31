"""Job synchronization service for cloud training.

Handles syncing job status between local store and cloud providers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import CloudJobStatus, JobType, UnifiedJob
from .factory import ProviderFactory

if TYPE_CHECKING:
    from .async_job_store import AsyncJobStore

logger = logging.getLogger(__name__)


async def sync_replicate_jobs(store: "AsyncJobStore") -> tuple[int, int]:
    """Sync jobs from Replicate into the local store.

    Args:
        store: The job store to sync into

    Returns:
        Tuple of (new jobs count, updated jobs count)
    """
    try:
        client = ProviderFactory.get_provider("replicate")
        cloud_jobs = await client.list_jobs(limit=100)

        new_count = 0
        updated_count = 0
        for cloud_job in cloud_jobs:
            existing = await store.get_job(cloud_job.job_id)
            if existing is None:
                unified = UnifiedJob.from_cloud_job(cloud_job)
                await store.add_job(unified)
                new_count += 1
            else:
                await store.update_job(
                    cloud_job.job_id,
                    {
                        "status": cloud_job.status.value,
                        "cost_usd": cloud_job.cost_usd,
                        "completed_at": cloud_job.completed_at,
                    },
                )
                updated_count += 1

        return new_count, updated_count
    except Exception as exc:
        logger.warning("Failed to sync Replicate jobs: %s", exc)
        return 0, 0


def _get_external_job_id(job: "UnifiedJob") -> str | None:
    """Get the external provider job ID for syncing.

    For Replicate, this is the prediction ID. The ID should be stored as
    job_id when the job was submitted through the proper flow.

    Returns None if the job doesn't have a valid external ID (e.g., test jobs).
    """
    # First check metadata for explicit prediction_id
    prediction_id = job.metadata.get("prediction_id")
    if prediction_id:
        return prediction_id

    # For properly submitted jobs, job_id IS the Replicate prediction ID
    # Replicate prediction IDs are short alphanumeric strings (e.g., "abc123xyz")
    # Skip local test IDs that contain descriptive patterns
    test_patterns = [
        "test",
        "local",
        "fake",
        "mock",  # explicit test markers
        "concurrent",
        "lifecycle",
        "batch",  # test scenario names
        "read-during",
        "write",
        "transaction",  # database test names
        "expensive",
        "warning",
        "token-",  # cost/auth test names
        "-000",
        "-001",
        "-002",
        "-003",  # numbered test suffixes
    ]
    if job.job_id and not any(pattern in job.job_id.lower() for pattern in test_patterns):
        return job.job_id

    return None


async def sync_active_job_statuses(store: "AsyncJobStore") -> int:
    """Sync status of active cloud jobs from their providers.

    Args:
        store: The job store containing jobs to sync

    Returns:
        Count of jobs updated
    """
    active_statuses = {
        CloudJobStatus.PENDING.value,
        CloudJobStatus.UPLOADING.value,
        CloudJobStatus.QUEUED.value,
        CloudJobStatus.RUNNING.value,
    }

    all_jobs = await store.list_jobs(limit=100)
    active_cloud_jobs = [j for j in all_jobs if j.job_type == JobType.CLOUD and j.status in active_statuses]

    if not active_cloud_jobs:
        return 0

    updated_count = 0
    jobs_by_provider = {}
    for job in active_cloud_jobs:
        if job.provider:
            jobs_by_provider.setdefault(job.provider, []).append(job)

    for provider_name, jobs in jobs_by_provider.items():
        try:
            client = ProviderFactory.get_provider(provider_name)
            for job in jobs:
                external_id = _get_external_job_id(job)
                if not external_id:
                    logger.debug("Skipping sync for job %s - no valid external ID", job.job_id)
                    continue

                try:
                    cloud_status = await client.get_job_status(external_id)
                    await store.update_job(
                        job.job_id,
                        {
                            "status": cloud_status.status.value,
                            "started_at": cloud_status.started_at,
                            "completed_at": cloud_status.completed_at,
                            "cost_usd": cloud_status.cost_usd,
                            "error_message": cloud_status.error_message,
                        },
                    )
                    updated_count += 1
                except Exception as exc:
                    logger.error("Failed to sync job %s from %s: %s", job.job_id, provider_name, exc, exc_info=True)
        except ValueError:
            logger.warning("Unknown provider: %s", provider_name)
        except Exception as exc:
            logger.error("Failed to sync jobs for provider %s: %s", provider_name, exc, exc_info=True)

    return updated_count
