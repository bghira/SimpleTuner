"""Resolve ordered training hosts against the submitting user's worker inventory."""

from ..models.worker import WorkerStatus, WorkerType


async def launch_inventory(user, *, include_offline=False, repository=None):
    from ..routes.workers import is_worker_connected
    from .worker_repository import get_worker_repository

    if not user or not user.is_active or not user.has_permission("job.submit"):
        return []
    repository = repository or get_worker_repository()
    owner = None if user.has_permission("admin.workers") else user.id
    workers = await repository.list_workers(user_id=owner, limit=-1)
    return [
        w
        for w in workers
        if not w.is_job_bound
        and w.worker_type != WorkerType.PEER
        and (include_offline or (is_worker_connected(w.worker_id) and w.status in (WorkerStatus.IDLE, WorkerStatus.BUSY)))
    ]


def resolve_worker_group(ids, inventory):
    by_id = {worker.worker_id: worker for worker in inventory}
    if len(set(ids)) != len(ids):
        raise ValueError("Select each worker only once.")
    workers = []
    for rank, worker_id in enumerate(ids):
        worker = by_id.get(worker_id)
        if worker is None:
            raise ValueError(f"The selected worker is unavailable at rank {rank}. Retry or change workers for this run.")
        if len(ids) > 1 and (
            not worker.multinode_gpus or (workers and worker.gpu_info["name"] != workers[0].gpu_info["name"])
        ):
            raise ValueError(f"Rank {rank} ({worker.name}) needs a multi-node agent with the same CUDA GPU model as rank 0.")
        workers.append(worker)
    return workers
