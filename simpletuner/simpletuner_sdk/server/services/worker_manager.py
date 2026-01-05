"""Worker Manager Service for GPU worker lifecycle and job dispatch.

Manages:
- Worker health monitoring via heartbeat checks
- Timeout detection and offline handling
- Orphaned job recovery and requeueing
- Job dispatch to idle workers
- Reconciliation on server startup
- Ephemeral worker cleanup
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages worker lifecycle, health checks, and job dispatch."""

    def __init__(
        self,
        worker_repository,
        job_store,
        sse_manager,
        check_interval: int = 60,
        heartbeat_timeout: int = 120,
        connecting_timeout: int = 300,
        ephemeral_cleanup_timeout: int = 3600,
    ):
        """Initialize the worker manager.

        Args:
            worker_repository: Repository for worker CRUD operations
            job_store: Job store for job updates and queries
            sse_manager: SSE manager for broadcasting events
            check_interval: Seconds between health checks (default: 60)
            heartbeat_timeout: Seconds before marking worker offline (default: 120)
            connecting_timeout: Seconds before cleaning up never-registered workers (default: 300)
            ephemeral_cleanup_timeout: Seconds before removing offline ephemeral workers (default: 3600)
        """
        self.worker_repository = worker_repository
        self.job_store = job_store
        self.sse_manager = sse_manager
        self.check_interval = check_interval
        self.heartbeat_timeout = timedelta(seconds=heartbeat_timeout)
        self.connecting_timeout = timedelta(seconds=connecting_timeout)
        self.ephemeral_cleanup_timeout = timedelta(seconds=ephemeral_cleanup_timeout)
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        """Start the background health check loop."""
        self._stop_event.clear()
        self._task = asyncio.create_task(self._health_check_loop())
        logger.info("WorkerManager started")

    async def stop(self):
        """Stop the background loop."""
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                # Task cancellation is expected during shutdown
                pass
        logger.info("WorkerManager stopped")

    async def _health_check_loop(self):
        """Periodic health check for all workers."""
        while not self._stop_event.is_set():
            try:
                await self._check_all_workers()
            except Exception as e:
                logger.error(f"Error in worker health check: {e}", exc_info=True)

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.check_interval,
                )
                break
            except asyncio.TimeoutError:
                continue

    async def _check_all_workers(self):
        """Check health of all workers."""
        from ..models.worker import WorkerStatus

        now = datetime.now(timezone.utc)
        workers = await self.worker_repository.list_workers()

        for worker in workers:
            await self._check_worker(worker, now)

    async def _check_worker(self, worker, now: datetime):
        """Check individual worker health."""
        from ..models.worker import WorkerStatus

        if worker.status == WorkerStatus.CONNECTING:
            time_since_created = now - worker.created_at
            if time_since_created > self.connecting_timeout:
                await self._handle_failed_launch(worker)

        elif worker.status in (WorkerStatus.IDLE, WorkerStatus.BUSY):
            if worker.last_heartbeat:
                time_since_heartbeat = now - worker.last_heartbeat
                if time_since_heartbeat > self.heartbeat_timeout:
                    await self._handle_worker_offline(worker)

        elif worker.status == WorkerStatus.OFFLINE:
            if not worker.persistent:
                if worker.last_heartbeat:
                    time_offline = now - worker.last_heartbeat
                    if time_offline > self.ephemeral_cleanup_timeout:
                        await self._cleanup_ephemeral_worker(worker)

    async def _handle_failed_launch(self, worker):
        """Handle worker that never registered after creation."""
        from ..models.worker import WorkerStatus

        logger.warning(f"Worker {worker.worker_id} never registered, cleaning up")

        if worker.current_job_id:
            await self._handle_orphaned_job(worker.current_job_id, worker)

        if worker.persistent:
            await self.worker_repository.update_worker(
                worker.worker_id,
                {"status": WorkerStatus.OFFLINE},
            )
        else:
            await self.worker_repository.delete_worker(worker.worker_id)

    async def _handle_worker_offline(self, worker):
        """Handle worker that stopped sending heartbeats."""
        from ..models.worker import WorkerStatus

        logger.warning(f"Worker {worker.worker_id} went offline")

        await self.worker_repository.update_worker(
            worker.worker_id,
            {"status": WorkerStatus.OFFLINE},
        )

        if worker.current_job_id:
            await self._handle_orphaned_job(worker.current_job_id, worker)

        await self.sse_manager.broadcast(
            data={
                "type": "worker.status",
                "worker_id": worker.worker_id,
                "status": "offline",
            },
            event_type="worker.status",
        )

    async def _handle_orphaned_job(self, job_id: str, worker):
        """Handle job that was running on offline worker."""
        job = await self.job_store.get_job(job_id)
        if not job:
            return

        if await self._check_job_outputs_exist(job):
            await self.job_store.update_job(
                job_id,
                {"status": "completed"},
            )
            logger.info(f"Job {job_id} recovered as completed from offline worker")
        else:
            retry_count = job.metadata.get("retry_count", 0) if job.metadata else 0
            max_retries = job.metadata.get("max_retries", 1) if job.metadata else 1

            if retry_count < max_retries:
                await self._requeue_job(job, retry_count + 1)
            else:
                await self.job_store.update_job(
                    job_id,
                    {
                        "status": "failed",
                        "error": f"Worker {worker.name} went offline during training",
                    },
                )
                logger.error(f"Job {job_id} failed - worker offline, max retries exceeded")

    async def _requeue_job(self, job, retry_count: int):
        """Requeue a job for retry."""
        metadata = job.metadata or {}
        metadata["retry_count"] = retry_count

        await self.job_store.update_job(
            job.job_id,
            {
                "status": "pending",
                "metadata": metadata,
            },
        )
        logger.info(f"Job {job.job_id} requeued for retry (attempt {retry_count})")

        await self.dispatch_pending_jobs()

    async def _check_job_outputs_exist(self, job) -> bool:
        """Check if job outputs were uploaded before worker went offline.

        This is a placeholder that returns False by default.
        Should be implemented based on actual output storage patterns.
        """
        return False

    async def _cleanup_ephemeral_worker(self, worker):
        """Remove ephemeral worker record after timeout."""
        logger.info(f"Cleaning up ephemeral worker {worker.worker_id}")
        await self.worker_repository.delete_worker(worker.worker_id)

    async def dispatch_pending_jobs(self):
        """Try to dispatch pending jobs to idle workers."""
        try:
            pending_jobs = await self.job_store.list_jobs(
                status="pending",
                provider="worker",
            )
        except TypeError:
            pending_jobs = await self.job_store.list_jobs(status="pending")
            pending_jobs = [j for j in pending_jobs if getattr(j, "provider", None) == "worker"]

        for job in pending_jobs:
            worker = await self._find_available_worker(job)
            if worker:
                await self._dispatch_job_to_worker(job, worker)

    async def _find_available_worker(self, job):
        """Find an idle worker that can handle the job."""
        metadata = job.metadata or {}
        required_gpu = metadata.get("required_gpu", {})
        required_labels = metadata.get("required_labels", {})

        return await self.worker_repository.get_idle_worker_for_job(
            gpu_requirements=required_gpu,
            labels=required_labels,
        )

    async def _dispatch_job_to_worker(self, job, worker):
        """Assign job to worker and push via SSE."""
        from ..models.worker import WorkerStatus

        try:
            from ..routes.workers import is_worker_connected, push_to_worker
        except ImportError:
            logger.warning("Worker routes not available, cannot dispatch job")
            return

        if not is_worker_connected(worker.worker_id):
            logger.warning(f"Worker {worker.worker_id} not connected, skipping dispatch")
            return

        await self.worker_repository.update_worker(
            worker.worker_id,
            {
                "status": WorkerStatus.BUSY,
                "current_job_id": job.job_id,
            },
        )

        from datetime import datetime, timezone

        # Update job with worker assignment in metadata for status check validation
        updated_metadata = dict(job.metadata or {})
        updated_metadata["worker_id"] = worker.worker_id
        await self.job_store.update_job(
            job.job_id,
            {
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "metadata": updated_metadata,
            },
        )

        job_metadata = job.metadata or {}
        config = job_metadata.get("config", {})
        success = await push_to_worker(
            worker.worker_id,
            {
                "type": "job_submit",
                "job_id": job.job_id,
                "config": config,
                "dataloader": config.get("dataloader_config"),
                "upload_endpoint": "/api/cloud/storage",
                "upload_token": getattr(job, "upload_token", None),
                "hf_token": job_metadata.get("hf_token"),
            },
        )

        if success:
            logger.info(f"Dispatched job {job.job_id} to worker {worker.worker_id}")
        else:
            logger.error(f"Failed to dispatch job {job.job_id} to worker {worker.worker_id}")
            await self.worker_repository.update_worker(
                worker.worker_id,
                {
                    "status": WorkerStatus.IDLE,
                    "current_job_id": None,
                },
            )
            # Reset job to pending, clear worker assignment from metadata
            reset_metadata = dict(job.metadata or {})
            reset_metadata.pop("worker_id", None)
            await self.job_store.update_job(
                job.job_id,
                {
                    "status": "pending",
                    "started_at": None,
                    "metadata": reset_metadata,
                },
            )

    async def reconcile_on_startup(self):
        """Called on server startup to reconcile state."""
        from ..models.worker import WorkerStatus

        logger.info("Reconciling worker state on startup")

        workers = await self.worker_repository.list_workers()

        for worker in workers:
            if worker.status in (WorkerStatus.IDLE, WorkerStatus.BUSY):
                await self.worker_repository.update_worker(
                    worker.worker_id,
                    {"status": WorkerStatus.OFFLINE},
                )

        try:
            running_jobs = await self.job_store.list_jobs(
                status="running",
                provider="worker",
            )
        except TypeError:
            running_jobs = await self.job_store.list_jobs(status="running")
            running_jobs = [j for j in running_jobs if getattr(j, "provider", None) == "worker"]

        for job in running_jobs:
            metadata = job.metadata or {}
            metadata["needs_reconciliation"] = True
            await self.job_store.update_job(
                job.job_id,
                {"metadata": metadata},
            )

        logger.info(f"Reconciled {len(workers)} workers, {len(running_jobs)} jobs marked for reconciliation")


_worker_manager_instance: Optional[WorkerManager] = None


def get_worker_manager() -> Optional[WorkerManager]:
    """Get the singleton worker manager instance.

    Returns:
        WorkerManager instance if initialized, None otherwise
    """
    return _worker_manager_instance


async def initialize_worker_manager(
    worker_repository,
    job_store,
    sse_manager,
) -> WorkerManager:
    """Initialize the global worker manager instance.

    Args:
        worker_repository: Repository for worker operations
        job_store: Job store for job operations
        sse_manager: SSE manager for broadcasting

    Returns:
        Initialized WorkerManager instance
    """
    global _worker_manager_instance

    if _worker_manager_instance is None:
        _worker_manager_instance = WorkerManager(
            worker_repository=worker_repository,
            job_store=job_store,
            sse_manager=sse_manager,
        )
        await _worker_manager_instance.reconcile_on_startup()
        await _worker_manager_instance.start()
        logger.info("Worker manager initialized and started")

    return _worker_manager_instance


async def shutdown_worker_manager():
    """Shutdown the global worker manager instance."""
    global _worker_manager_instance

    if _worker_manager_instance is not None:
        await _worker_manager_instance.stop()
        _worker_manager_instance = None
        logger.info("Worker manager shut down")
