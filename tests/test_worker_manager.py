import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from simpletuner.simpletuner_sdk.server.models.worker import Worker, WorkerStatus, WorkerType
from simpletuner.simpletuner_sdk.server.services.cloud.base import JobType, UnifiedJob
from simpletuner.simpletuner_sdk.server.services.worker_manager import WorkerManager


class TestWorkerManager(unittest.IsolatedAsyncioTestCase):
    """Test cases for WorkerManager."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.mock_worker_repository = AsyncMock()
        self.mock_job_store = AsyncMock()
        self.mock_sse_manager = AsyncMock()

        self.worker_manager = WorkerManager(
            worker_repository=self.mock_worker_repository,
            job_store=self.mock_job_store,
            sse_manager=self.mock_sse_manager,
            check_interval=1,
            heartbeat_timeout=120,
            connecting_timeout=300,
            ephemeral_cleanup_timeout=3600,
        )

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        pass

    def _create_worker(
        self,
        worker_id: str = "worker-1",
        status: WorkerStatus = WorkerStatus.IDLE,
        worker_type: WorkerType = WorkerType.PERSISTENT,
        current_job_id: str = None,
        last_heartbeat: datetime = None,
        created_at: datetime = None,
    ) -> Worker:
        """Helper to create a test worker.

        Args:
            worker_id: Worker ID
            status: Worker status
            worker_type: Worker type
            current_job_id: Current job ID
            last_heartbeat: Last heartbeat timestamp
            created_at: Creation timestamp

        Returns:
            Worker instance
        """
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        return Worker(
            worker_id=worker_id,
            name=f"Test Worker {worker_id}",
            worker_type=worker_type,
            status=status,
            token_hash="test-token-hash",
            user_id=1,
            gpu_info={"name": "A100", "vram_gb": 80, "count": 1},
            provider=None,
            labels={},
            current_job_id=current_job_id,
            last_heartbeat=last_heartbeat,
            created_at=created_at,
        )

    def _create_job(
        self,
        job_id: str = "job-1",
        status: str = "pending",
        provider: str = "worker",
        metadata: dict = None,
        config: dict = None,
    ) -> UnifiedJob:
        """Helper to create a test job.

        Args:
            job_id: Job ID
            status: Job status
            provider: Job provider
            metadata: Job metadata
            config: Job config

        Returns:
            UnifiedJob instance
        """
        job = UnifiedJob(
            job_id=job_id,
            job_type=JobType.LOCAL,
            provider=provider,
            status=status,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        job.metadata = metadata or {}
        job.config = config or {}
        job.dataloader_config = None
        job.upload_token = "test-upload-token"
        return job

    async def test_start_and_stop(self) -> None:
        """Test starting and stopping the worker manager."""
        await self.worker_manager.start()
        self.assertIsNotNone(self.worker_manager._task)
        self.assertFalse(self.worker_manager._stop_event.is_set())

        await self.worker_manager.stop()
        self.assertTrue(self.worker_manager._stop_event.is_set())

    async def test_health_check_loop_runs_until_stopped(self) -> None:
        """Test that health check loop continues until stopped."""
        self.mock_worker_repository.list_workers.return_value = []

        await self.worker_manager.start()

        await asyncio.sleep(0.1)
        self.assertTrue(self.mock_worker_repository.list_workers.called)

        await self.worker_manager.stop()

    async def test_health_check_loop_handles_exceptions(self) -> None:
        """Test that health check loop continues after exceptions."""
        self.mock_worker_repository.list_workers.side_effect = [
            Exception("Test error"),
            [],
        ]

        await self.worker_manager.start()

        await asyncio.sleep(0.1)

        await self.worker_manager.stop()

    async def test_check_worker_connecting_timeout(self) -> None:
        """Test worker in CONNECTING state that times out."""
        now = datetime.now(timezone.utc)
        created_at = now - timedelta(seconds=400)
        worker = self._create_worker(
            worker_id="worker-1",
            status=WorkerStatus.CONNECTING,
            created_at=created_at,
        )

        await self.worker_manager._check_worker(worker, now)

        self.mock_worker_repository.update_worker.assert_called_once_with(
            "worker-1",
            {"status": WorkerStatus.OFFLINE},
        )

    async def test_check_worker_connecting_not_timed_out(self) -> None:
        """Test worker in CONNECTING state that has not timed out."""
        now = datetime.now(timezone.utc)
        created_at = now - timedelta(seconds=100)
        worker = self._create_worker(
            worker_id="worker-1",
            status=WorkerStatus.CONNECTING,
            created_at=created_at,
        )

        await self.worker_manager._check_worker(worker, now)

        self.mock_worker_repository.update_worker.assert_not_called()

    async def test_check_worker_heartbeat_timeout(self) -> None:
        """Test worker that stopped sending heartbeats."""
        now = datetime.now(timezone.utc)
        last_heartbeat = now - timedelta(seconds=200)
        worker = self._create_worker(
            worker_id="worker-1",
            status=WorkerStatus.IDLE,
            last_heartbeat=last_heartbeat,
        )

        await self.worker_manager._check_worker(worker, now)

        self.mock_worker_repository.update_worker.assert_called_once_with(
            "worker-1",
            {"status": WorkerStatus.OFFLINE},
        )
        self.mock_sse_manager.broadcast.assert_called_once()

    async def test_check_worker_heartbeat_not_timed_out(self) -> None:
        """Test worker with recent heartbeat."""
        now = datetime.now(timezone.utc)
        last_heartbeat = now - timedelta(seconds=60)
        worker = self._create_worker(
            worker_id="worker-1",
            status=WorkerStatus.IDLE,
            last_heartbeat=last_heartbeat,
        )

        await self.worker_manager._check_worker(worker, now)

        self.mock_worker_repository.update_worker.assert_not_called()

    async def test_check_worker_ephemeral_cleanup(self) -> None:
        """Test cleanup of offline ephemeral worker."""
        now = datetime.now(timezone.utc)
        last_heartbeat = now - timedelta(seconds=4000)
        worker = self._create_worker(
            worker_id="worker-1",
            status=WorkerStatus.OFFLINE,
            worker_type=WorkerType.EPHEMERAL,
            last_heartbeat=last_heartbeat,
        )

        await self.worker_manager._check_worker(worker, now)

        self.mock_worker_repository.delete_worker.assert_called_once_with("worker-1")

    async def test_check_worker_ephemeral_not_ready_for_cleanup(self) -> None:
        """Test ephemeral worker not yet ready for cleanup."""
        now = datetime.now(timezone.utc)
        last_heartbeat = now - timedelta(seconds=1800)
        worker = self._create_worker(
            worker_id="worker-1",
            status=WorkerStatus.OFFLINE,
            worker_type=WorkerType.EPHEMERAL,
            last_heartbeat=last_heartbeat,
        )

        await self.worker_manager._check_worker(worker, now)

        self.mock_worker_repository.delete_worker.assert_not_called()

    async def test_check_worker_persistent_no_cleanup(self) -> None:
        """Test persistent worker is never cleaned up."""
        now = datetime.now(timezone.utc)
        last_heartbeat = now - timedelta(seconds=10000)
        worker = self._create_worker(
            worker_id="worker-1",
            status=WorkerStatus.OFFLINE,
            worker_type=WorkerType.PERSISTENT,
            last_heartbeat=last_heartbeat,
        )

        await self.worker_manager._check_worker(worker, now)

        self.mock_worker_repository.delete_worker.assert_not_called()

    async def test_handle_failed_launch_persistent_worker(self) -> None:
        """Test handling worker that never registered - persistent."""
        worker = self._create_worker(
            worker_id="worker-1",
            status=WorkerStatus.CONNECTING,
            worker_type=WorkerType.PERSISTENT,
        )

        await self.worker_manager._handle_failed_launch(worker)

        self.mock_worker_repository.update_worker.assert_called_once_with(
            "worker-1",
            {"status": WorkerStatus.OFFLINE},
        )
        self.mock_worker_repository.delete_worker.assert_not_called()

    async def test_handle_failed_launch_ephemeral_worker(self) -> None:
        """Test handling worker that never registered - ephemeral."""
        worker = self._create_worker(
            worker_id="worker-1",
            status=WorkerStatus.CONNECTING,
            worker_type=WorkerType.EPHEMERAL,
        )

        await self.worker_manager._handle_failed_launch(worker)

        self.mock_worker_repository.delete_worker.assert_called_once_with("worker-1")
        self.mock_worker_repository.update_worker.assert_not_called()

    async def test_handle_failed_launch_with_job(self) -> None:
        """Test handling failed launch when worker had a job."""
        worker = self._create_worker(
            worker_id="worker-1",
            status=WorkerStatus.CONNECTING,
            current_job_id="job-1",
        )
        job = self._create_job(job_id="job-1", status="running")
        self.mock_job_store.get_job.return_value = job

        await self.worker_manager._handle_failed_launch(worker)

        self.mock_job_store.get_job.assert_called_once_with("job-1")
        self.mock_job_store.update_job.assert_called()

    async def test_handle_worker_offline(self) -> None:
        """Test handling worker that went offline."""
        worker = self._create_worker(
            worker_id="worker-1",
            status=WorkerStatus.IDLE,
        )

        await self.worker_manager._handle_worker_offline(worker)

        self.mock_worker_repository.update_worker.assert_called_once_with(
            "worker-1",
            {"status": WorkerStatus.OFFLINE},
        )
        self.mock_sse_manager.broadcast.assert_called_once_with(
            data={
                "type": "worker.status",
                "worker_id": "worker-1",
                "status": "offline",
            },
            event_type="worker.status",
        )

    async def test_handle_worker_offline_with_job(self) -> None:
        """Test handling worker offline with active job."""
        worker = self._create_worker(
            worker_id="worker-1",
            status=WorkerStatus.BUSY,
            current_job_id="job-1",
        )
        job = self._create_job(job_id="job-1", status="running")
        self.mock_job_store.get_job.return_value = job

        await self.worker_manager._handle_worker_offline(worker)

        self.mock_job_store.get_job.assert_called_once_with("job-1")
        self.mock_worker_repository.update_worker.assert_called_once()
        self.mock_sse_manager.broadcast.assert_called_once()

    async def test_handle_orphaned_job_completed(self) -> None:
        """Test orphaned job that already completed."""
        worker = self._create_worker(worker_id="worker-1")
        job = self._create_job(job_id="job-1", status="running")
        self.mock_job_store.get_job.return_value = job

        with patch.object(self.worker_manager, "_check_job_outputs_exist", return_value=True):
            await self.worker_manager._handle_orphaned_job("job-1", worker)

        self.mock_job_store.update_job.assert_called_once_with(
            "job-1",
            {"status": "completed"},
        )

    async def test_handle_orphaned_job_requeue(self) -> None:
        """Test orphaned job that should be requeued."""
        worker = self._create_worker(worker_id="worker-1")
        job = self._create_job(
            job_id="job-1",
            status="running",
            metadata={"retry_count": 0, "max_retries": 2},
        )
        self.mock_job_store.get_job.return_value = job
        self.mock_job_store.list_jobs.return_value = []

        with patch.object(self.worker_manager, "_check_job_outputs_exist", return_value=False):
            await self.worker_manager._handle_orphaned_job("job-1", worker)

        calls = self.mock_job_store.update_job.call_args_list
        self.assertEqual(len(calls), 1)
        job_id, updates = calls[0][0]
        self.assertEqual(job_id, "job-1")
        self.assertEqual(updates["status"], "pending")
        self.assertEqual(updates["metadata"]["retry_count"], 1)

    async def test_handle_orphaned_job_max_retries(self) -> None:
        """Test orphaned job that exceeded max retries."""
        worker = self._create_worker(worker_id="worker-1")
        job = self._create_job(
            job_id="job-1",
            status="running",
            metadata={"retry_count": 2, "max_retries": 2},
        )
        self.mock_job_store.get_job.return_value = job

        with patch.object(self.worker_manager, "_check_job_outputs_exist", return_value=False):
            await self.worker_manager._handle_orphaned_job("job-1", worker)

        self.mock_job_store.update_job.assert_called_once()
        job_id, updates = self.mock_job_store.update_job.call_args[0]
        self.assertEqual(updates["status"], "failed")
        self.assertIn("went offline", updates["error"])

    async def test_handle_orphaned_job_not_found(self) -> None:
        """Test orphaned job that doesn't exist."""
        worker = self._create_worker(worker_id="worker-1")
        self.mock_job_store.get_job.return_value = None

        await self.worker_manager._handle_orphaned_job("job-1", worker)

        self.mock_job_store.update_job.assert_not_called()

    async def test_requeue_job(self) -> None:
        """Test requeueing a job."""
        job = self._create_job(
            job_id="job-1",
            status="running",
            metadata={"some_key": "some_value"},
        )
        self.mock_job_store.list_jobs.return_value = []

        await self.worker_manager._requeue_job(job, retry_count=2)

        self.mock_job_store.update_job.assert_called_once()
        job_id, updates = self.mock_job_store.update_job.call_args[0]
        self.assertEqual(job_id, "job-1")
        self.assertEqual(updates["status"], "pending")
        self.assertEqual(updates["metadata"]["retry_count"], 2)
        self.assertEqual(updates["metadata"]["some_key"], "some_value")

    async def test_dispatch_pending_jobs(self) -> None:
        """Test dispatching pending jobs to workers."""
        job = self._create_job(job_id="job-1", status="pending", provider="worker")
        worker = self._create_worker(worker_id="worker-1", status=WorkerStatus.IDLE)

        self.mock_job_store.list_jobs.return_value = [job]
        self.mock_worker_repository.get_idle_worker_for_job.return_value = worker

        with patch("simpletuner.simpletuner_sdk.server.routes.workers.is_worker_connected", return_value=True):
            with patch(
                "simpletuner.simpletuner_sdk.server.routes.workers.push_to_worker", new_callable=AsyncMock
            ) as mock_push:
                mock_push.return_value = True
                await self.worker_manager.dispatch_pending_jobs()

        self.mock_job_store.list_jobs.assert_called_once()
        self.mock_worker_repository.get_idle_worker_for_job.assert_called_once()

    async def test_dispatch_pending_jobs_no_workers(self) -> None:
        """Test dispatching when no workers available."""
        job = self._create_job(job_id="job-1", status="pending", provider="worker")

        self.mock_job_store.list_jobs.return_value = [job]
        self.mock_worker_repository.get_idle_worker_for_job.return_value = None

        await self.worker_manager.dispatch_pending_jobs()

        self.mock_job_store.update_job.assert_not_called()

    async def test_dispatch_pending_jobs_provider_filter_with_provider_param(self) -> None:
        """Test job filtering with provider parameter."""
        job1 = self._create_job(job_id="job-1", status="pending", provider="worker")

        self.mock_job_store.list_jobs.return_value = [job1]

        await self.worker_manager.dispatch_pending_jobs()

        call_args = self.mock_job_store.list_jobs.call_args
        self.assertIn("status", call_args[1])
        self.assertEqual(call_args[1]["status"], "pending")

    async def test_dispatch_pending_jobs_provider_filter_fallback(self) -> None:
        """Test job filtering fallback when provider param not supported."""
        job1 = self._create_job(job_id="job-1", status="pending", provider="worker")
        job2 = self._create_job(job_id="job-2", status="pending", provider="other")

        self.mock_job_store.list_jobs.side_effect = [TypeError("provider not supported"), [job1, job2]]

        await self.worker_manager.dispatch_pending_jobs()

        self.assertEqual(self.mock_job_store.list_jobs.call_count, 2)

    async def test_find_available_worker(self) -> None:
        """Test finding available worker for job."""
        job = self._create_job(
            job_id="job-1",
            metadata={
                "required_gpu": {"vram_gb": 40},
                "required_labels": {"env": "prod"},
            },
        )
        worker = self._create_worker(worker_id="worker-1")
        self.mock_worker_repository.get_idle_worker_for_job.return_value = worker

        result = await self.worker_manager._find_available_worker(job)

        self.assertEqual(result, worker)
        self.mock_worker_repository.get_idle_worker_for_job.assert_called_once_with(
            gpu_requirements={"vram_gb": 40},
            labels={"env": "prod"},
        )

    async def test_find_available_worker_no_metadata(self) -> None:
        """Test finding available worker when job has no metadata."""
        job = self._create_job(job_id="job-1", metadata=None)
        worker = self._create_worker(worker_id="worker-1")
        self.mock_worker_repository.get_idle_worker_for_job.return_value = worker

        result = await self.worker_manager._find_available_worker(job)

        self.assertEqual(result, worker)
        self.mock_worker_repository.get_idle_worker_for_job.assert_called_once_with(
            gpu_requirements={},
            labels={},
        )

    async def test_dispatch_job_to_worker_success(self) -> None:
        """Test successfully dispatching job to worker."""
        job = self._create_job(
            job_id="job-1",
            status="pending",
            metadata={"hf_token": "test-token"},
        )
        worker = self._create_worker(worker_id="worker-1", status=WorkerStatus.IDLE)

        with patch("simpletuner.simpletuner_sdk.server.routes.workers.is_worker_connected", return_value=True):
            with patch(
                "simpletuner.simpletuner_sdk.server.routes.workers.push_to_worker", new_callable=AsyncMock
            ) as mock_push:
                mock_push.return_value = True
                await self.worker_manager._dispatch_job_to_worker(job, worker)

        self.mock_worker_repository.update_worker.assert_called_once_with(
            "worker-1",
            {
                "status": WorkerStatus.BUSY,
                "current_job_id": "job-1",
            },
        )
        # Verify job update was called with correct structure
        self.mock_job_store.update_job.assert_called_once()
        call_args = self.mock_job_store.update_job.call_args
        self.assertEqual(call_args[0][0], "job-1")
        updates = call_args[0][1]
        self.assertEqual(updates["status"], "running")
        self.assertIn("started_at", updates)
        self.assertIn("metadata", updates)
        self.assertEqual(updates["metadata"]["worker_id"], "worker-1")
        mock_push.assert_called_once()

    async def test_dispatch_job_to_worker_not_connected(self) -> None:
        """Test dispatching when worker not connected."""
        job = self._create_job(job_id="job-1", status="pending")
        worker = self._create_worker(worker_id="worker-1", status=WorkerStatus.IDLE)

        with patch("simpletuner.simpletuner_sdk.server.routes.workers.is_worker_connected", return_value=False):
            await self.worker_manager._dispatch_job_to_worker(job, worker)

        self.mock_worker_repository.update_worker.assert_not_called()
        self.mock_job_store.update_job.assert_not_called()

    async def test_dispatch_job_to_worker_push_failed(self) -> None:
        """Test dispatching when push to worker fails."""
        job = self._create_job(job_id="job-1", status="pending")
        worker = self._create_worker(worker_id="worker-1", status=WorkerStatus.IDLE)

        with patch("simpletuner.simpletuner_sdk.server.routes.workers.is_worker_connected", return_value=True):
            with patch(
                "simpletuner.simpletuner_sdk.server.routes.workers.push_to_worker", new_callable=AsyncMock
            ) as mock_push:
                mock_push.return_value = False
                await self.worker_manager._dispatch_job_to_worker(job, worker)

        update_calls = self.mock_worker_repository.update_worker.call_args_list
        self.assertEqual(len(update_calls), 2)
        self.assertEqual(update_calls[1][0][1]["status"], WorkerStatus.IDLE)
        self.assertIsNone(update_calls[1][0][1]["current_job_id"])

        job_update_calls = self.mock_job_store.update_job.call_args_list
        self.assertEqual(len(job_update_calls), 2)
        self.assertEqual(job_update_calls[1][0][1]["status"], "pending")
        # Verify worker_id is removed from metadata on failure
        self.assertNotIn("worker_id", job_update_calls[1][0][1].get("metadata", {}))

    async def test_dispatch_job_to_worker_routes_not_available(self) -> None:
        """Test dispatching when worker routes not importable."""
        job = self._create_job(job_id="job-1", status="pending")
        worker = self._create_worker(worker_id="worker-1", status=WorkerStatus.IDLE)

        with patch.dict("sys.modules", {"simpletuner.simpletuner_sdk.server.routes.workers": None}):
            await self.worker_manager._dispatch_job_to_worker(job, worker)

        self.mock_worker_repository.update_worker.assert_not_called()
        self.mock_job_store.update_job.assert_not_called()

    async def test_reconcile_on_startup(self) -> None:
        """Test reconciliation on server startup."""
        worker1 = self._create_worker(worker_id="worker-1", status=WorkerStatus.IDLE)
        worker2 = self._create_worker(worker_id="worker-2", status=WorkerStatus.BUSY)
        worker3 = self._create_worker(worker_id="worker-3", status=WorkerStatus.OFFLINE)

        job1 = self._create_job(job_id="job-1", status="running", provider="worker")
        job2 = self._create_job(job_id="job-2", status="running", provider="worker")

        self.mock_worker_repository.list_workers.return_value = [worker1, worker2, worker3]
        self.mock_job_store.list_jobs.return_value = [job1, job2]

        await self.worker_manager.reconcile_on_startup()

        update_calls = self.mock_worker_repository.update_worker.call_args_list
        self.assertEqual(len(update_calls), 2)

        job_update_calls = self.mock_job_store.update_job.call_args_list
        self.assertEqual(len(job_update_calls), 2)
        for call in job_update_calls:
            job_id, updates = call[0]
            self.assertTrue(updates["metadata"]["needs_reconciliation"])

    async def test_reconcile_on_startup_provider_filter_fallback(self) -> None:
        """Test reconciliation fallback when provider param not supported."""
        worker1 = self._create_worker(worker_id="worker-1", status=WorkerStatus.IDLE)
        job1 = self._create_job(job_id="job-1", status="running", provider="worker")
        job2 = self._create_job(job_id="job-2", status="running", provider="other")

        self.mock_worker_repository.list_workers.return_value = [worker1]
        self.mock_job_store.list_jobs.side_effect = [TypeError("provider not supported"), [job1, job2]]

        await self.worker_manager.reconcile_on_startup()

        self.assertEqual(self.mock_job_store.list_jobs.call_count, 2)

    async def test_check_job_outputs_exist(self) -> None:
        """Test checking if job outputs exist."""
        job = self._create_job(job_id="job-1")

        result = await self.worker_manager._check_job_outputs_exist(job)

        self.assertFalse(result)

    async def test_cleanup_ephemeral_worker(self) -> None:
        """Test cleanup of ephemeral worker."""
        worker = self._create_worker(
            worker_id="worker-1",
            worker_type=WorkerType.EPHEMERAL,
        )

        await self.worker_manager._cleanup_ephemeral_worker(worker)

        self.mock_worker_repository.delete_worker.assert_called_once_with("worker-1")

    async def test_check_all_workers(self) -> None:
        """Test checking all workers."""
        now = datetime.now(timezone.utc)
        worker1 = self._create_worker(worker_id="worker-1", status=WorkerStatus.IDLE, last_heartbeat=now)
        worker2 = self._create_worker(worker_id="worker-2", status=WorkerStatus.BUSY, last_heartbeat=now)

        self.mock_worker_repository.list_workers.return_value = [worker1, worker2]

        await self.worker_manager._check_all_workers()

        self.mock_worker_repository.list_workers.assert_called_once()

    async def test_health_check_loop_stop_event(self) -> None:
        """Test health check loop respects stop event."""
        self.mock_worker_repository.list_workers.return_value = []

        await self.worker_manager.start()

        await asyncio.sleep(0.05)

        self.worker_manager._stop_event.set()

        await asyncio.sleep(0.1)

        await self.worker_manager.stop()


if __name__ == "__main__":
    unittest.main()
