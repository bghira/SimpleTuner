"""Retrieve the active job's dataloader configuration from its lead worker."""

import asyncio
from dataclasses import dataclass
from uuid import UUID, uuid4

from simpletuner.worker_protocol import bundled_dataloader, redact_dataset_config

from .cloud.auth.models import User
from .cloud.base import UnifiedJob
from .cloud.storage.job_repository import get_job_repository
from .worker_repository import get_worker_repository

_REQUEST_TIMEOUT = 5
Assignment = tuple[str, str, str | None]


@dataclass
class DatasetRequest:
    request_id: UUID
    response: asyncio.Future
    task: asyncio.Task


_pending: dict[Assignment, DatasetRequest] = {}


def complete_dataset_request(job: UnifiedJob, worker_id: str, request_id: UUID, datasets: list[dict] | None) -> bool:
    pending = _pending.get((job.job_id, worker_id, job.started_at))
    if not pending or pending.request_id != request_id or pending.response.done():
        return False
    pending.response.set_result(redact_dataset_config(datasets))
    return True


async def _retrieve(
    assignment: Assignment, request_id: UUID, response: asyncio.Future
) -> tuple[list[dict] | None, str | None]:
    from ..routes.workers import push_to_worker

    job_id, worker_id, _ = assignment
    try:
        sent = await push_to_worker(
            worker_id, {"type": "job_dataset_config", "job_id": job_id, "request_id": str(request_id)}
        )
        if not sent:
            return None, "The lead worker is disconnected."
        async with asyncio.timeout(_REQUEST_TIMEOUT):
            datasets = await response
        if datasets is None:
            return None, "The worker could not read this job’s dataset configuration."
        return datasets, None
    except TimeoutError:
        return None, "The lead worker did not respond. Refresh to try again."
    finally:
        del _pending[assignment]


async def get_worker_dataset_config(environment: str, user: User) -> dict | None:
    from ..routes.workers import dataset_config_workers, is_worker_connected

    jobs = get_job_repository()
    user_id = None if user.has_permission("job.view.all") else user.id
    job = await jobs.get_active_worker_job(environment, user_id)
    if job is None:
        return None

    worker_id = job.metadata["worker_id"]
    error = None
    if worker_id in dataset_config_workers:
        assignment = (job.job_id, worker_id, job.started_at)
        pending = _pending.get(assignment)
        if pending is None:
            request_id = uuid4()
            response = asyncio.get_running_loop().create_future()
            task = asyncio.create_task(_retrieve(assignment, request_id, response))
            pending = _pending[assignment] = DatasetRequest(request_id, response, task)
        datasets, error = await asyncio.shield(pending.task)
    else:
        # This is the dataloader file sent at launch, even on older worker agents.
        bundled = bundled_dataloader(job.metadata)
        datasets = redact_dataset_config(bundled) if isinstance(bundled, list) else None
        if datasets is None:
            error = (
                "Update the lead worker agent to retrieve this job’s dataset configuration."
                if is_worker_connected(worker_id)
                else "The lead worker is disconnected."
            )

    workers = job.metadata.get("worker_names")
    if not workers:
        worker = await get_worker_repository().get_worker(worker_id)
        workers = [worker.name if worker else worker_id]
    return {"job_id": job.job_id, "workers": workers, "datasets": datasets, "error": error}
