"""Dataset summaries for active remote jobs."""

import asyncio

from simpletuner.worker_protocol import (
    WORKER_JOB_ACTIVE_STATUSES,
    WORKER_JOB_PROVIDERS,
    bundled_dataloader,
    summarize_worker_datasets,
)

from .cloud.auth.models import User
from .cloud.storage.job_repository import get_job_repository
from .worker_repository import get_worker_repository


async def list_worker_dataset_runs(user: User) -> list[dict]:
    jobs = get_job_repository()
    user_id = None if user.has_permission("job.view.all") else user.id
    results = await asyncio.gather(
        *(
            jobs.list(provider=provider, status=status, user_id=user_id, limit=-1)
            for provider in WORKER_JOB_PROVIDERS
            for status in WORKER_JOB_ACTIVE_STATUSES
        )
    )
    runs = []
    for job in sorted((job for batch in results for job in batch), key=lambda job: job.created_at, reverse=True):
        metadata = job.metadata or {}
        worker_id = metadata.get("worker_id")
        if not worker_id or metadata.get("group_check"):
            continue

        snapshot = metadata.get("worker_datasets") or {}
        if snapshot.get("worker_id") == worker_id and snapshot.get("started_at") == job.started_at:
            datasets = snapshot["datasets"]
        else:
            # This is the dataset file the worker materializes from its dispatch payload.
            bundled = bundled_dataloader(metadata)
            datasets = summarize_worker_datasets(bundled) if isinstance(bundled, list) else None

        worker_names = metadata.get("worker_names")
        if not worker_names:
            worker = await get_worker_repository().get_worker(worker_id)
            worker_names = [worker.name if worker else worker_id]
        runs.append(
            {
                "job_id": job.job_id,
                "config_name": job.config_name,
                "workers": worker_names,
                "datasets": datasets,
            }
        )
    return runs
