"""Resolve dataloader configurations for worker-backed environments."""

from simpletuner.worker_protocol import bundled_dataloader, redact_dataset_config

from .cloud.auth.models import User
from .cloud.storage.job_repository import get_job_repository
from .worker_repository import get_worker_repository


async def get_worker_dataset_config(environment: str, user: User) -> dict | None:
    jobs = get_job_repository()
    user_id = None if user.has_permission("job.view.all") else user.id
    job = await jobs.get_active_worker_job(environment, user_id)
    if job is None:
        return None

    worker_id = job.metadata["worker_id"]
    datasets = await jobs.get_worker_dataset_config(job, worker_id)
    if datasets is None:
        # Dispatch materializes the bundled data as the worker's dataloader file.
        bundled = bundled_dataloader(job.metadata)
        if isinstance(bundled, list):
            datasets = redact_dataset_config(bundled)

    workers = job.metadata.get("worker_names")
    if not workers:
        worker = await get_worker_repository().get_worker(worker_id)
        workers = [worker.name if worker else worker_id]
    return {"job_id": job.job_id, "workers": workers, "datasets": datasets}
