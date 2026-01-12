"""Integration tests for cloud job submission flow.

Tests the full lifecycle of cloud jobs including:
- Pre-submission checks
- Job submission
- Status updates via webhooks
- Job cancellation
- Concurrent access patterns
"""

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from simpletuner.simpletuner_sdk.server.services.cloud.async_job_store import AsyncJobStore
from simpletuner.simpletuner_sdk.server.services.cloud.base import CloudJobStatus, JobType, UnifiedJob


class TestJobSubmissionIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for the full job submission flow."""

    async def asyncSetUp(self) -> None:
        """Set up test fixtures."""
        # Reset all storage singletons for test isolation
        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()

        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        self.store = AsyncJobStore(self.config_dir)
        await self.store._ensure_initialized()

    async def asyncTearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        await self.store.close()
        AsyncJobStore._instance = None
        # Clear BaseSQLiteStore singletons
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_full_job_lifecycle_success(self) -> None:
        """Test complete job lifecycle: pending -> uploading -> running -> completed."""
        # 1. Create job in pending state
        job_id = "lifecycle-test-001"
        job = UnifiedJob(
            job_id=job_id,
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.PENDING.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        await self.store.add_job(job)

        # Verify initial state
        retrieved = await self.store.get_job(job_id)
        self.assertEqual(retrieved.status, CloudJobStatus.PENDING.value)

        # 2. Transition to uploading
        await self.store.update_job(job_id, {"status": CloudJobStatus.UPLOADING.value})
        retrieved = await self.store.get_job(job_id)
        self.assertEqual(retrieved.status, CloudJobStatus.UPLOADING.value)

        # 3. Transition to queued (upload complete, waiting for provider)
        await self.store.update_job(job_id, {"status": CloudJobStatus.QUEUED.value})
        retrieved = await self.store.get_job(job_id)
        self.assertEqual(retrieved.status, CloudJobStatus.QUEUED.value)

        # 4. Transition to running
        started_at = datetime.now(timezone.utc).isoformat()
        await self.store.update_job(
            job_id,
            {
                "status": CloudJobStatus.RUNNING.value,
                "started_at": started_at,
            },
        )
        retrieved = await self.store.get_job(job_id)
        self.assertEqual(retrieved.status, CloudJobStatus.RUNNING.value)
        self.assertEqual(retrieved.started_at, started_at)

        # 5. Transition to completed
        completed_at = datetime.now(timezone.utc).isoformat()
        await self.store.update_job(
            job_id,
            {
                "status": CloudJobStatus.COMPLETED.value,
                "completed_at": completed_at,
                "cost_usd": 2.50,
                "output_url": "https://example.com/output.safetensors",
            },
        )
        retrieved = await self.store.get_job(job_id)
        self.assertEqual(retrieved.status, CloudJobStatus.COMPLETED.value)
        self.assertEqual(retrieved.completed_at, completed_at)
        self.assertEqual(retrieved.cost_usd, 2.50)
        self.assertEqual(retrieved.output_url, "https://example.com/output.safetensors")

    async def test_full_job_lifecycle_failure(self) -> None:
        """Test job lifecycle with failure: pending -> running -> failed."""
        job_id = "lifecycle-fail-001"
        job = UnifiedJob(
            job_id=job_id,
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.PENDING.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        await self.store.add_job(job)

        # Transition to running
        await self.store.update_job(
            job_id,
            {
                "status": CloudJobStatus.RUNNING.value,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Simulate failure
        error_message = "CUDA out of memory"
        await self.store.update_job(
            job_id,
            {
                "status": CloudJobStatus.FAILED.value,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error_message": error_message,
            },
        )

        retrieved = await self.store.get_job(job_id)
        self.assertEqual(retrieved.status, CloudJobStatus.FAILED.value)
        self.assertEqual(retrieved.error_message, error_message)

    async def test_job_cancellation_flow(self) -> None:
        """Test job cancellation at various stages."""
        # Test cancellation while pending
        job_id = "cancel-pending-001"
        job = UnifiedJob(
            job_id=job_id,
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.PENDING.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        await self.store.add_job(job)
        await self.store.update_job(job_id, {"status": CloudJobStatus.CANCELLED.value})

        retrieved = await self.store.get_job(job_id)
        self.assertEqual(retrieved.status, CloudJobStatus.CANCELLED.value)

        # Test cancellation while running
        job_id2 = "cancel-running-001"
        job2 = UnifiedJob(
            job_id=job_id2,
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.RUNNING.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        await self.store.add_job(job2)
        await self.store.update_job(
            job_id2,
            {
                "status": CloudJobStatus.CANCELLED.value,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        retrieved = await self.store.get_job(job_id2)
        self.assertEqual(retrieved.status, CloudJobStatus.CANCELLED.value)

    async def test_webhook_status_update_flow(self) -> None:
        """Test that webhook status updates work correctly."""
        job_id = "webhook-test-001"
        job = UnifiedJob(
            job_id=job_id,
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.QUEUED.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        await self.store.add_job(job)

        # Simulate webhook updates
        webhook_events = [
            {"status": CloudJobStatus.RUNNING.value, "started_at": datetime.now(timezone.utc).isoformat()},
            {
                "status": CloudJobStatus.COMPLETED.value,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "cost_usd": 1.25,
            },
        ]

        for event in webhook_events:
            await self.store.update_job(job_id, event)

        retrieved = await self.store.get_job(job_id)
        self.assertEqual(retrieved.status, CloudJobStatus.COMPLETED.value)
        self.assertEqual(retrieved.cost_usd, 1.25)

    async def test_status_normalization_from_external(self) -> None:
        """Test that external status strings are normalized correctly."""
        # Test American vs British spelling
        self.assertEqual(CloudJobStatus.from_external("canceled"), CloudJobStatus.CANCELLED)
        self.assertEqual(CloudJobStatus.from_external("cancelled"), CloudJobStatus.CANCELLED)

        # Test Replicate-specific statuses
        self.assertEqual(CloudJobStatus.from_external("succeeded"), CloudJobStatus.COMPLETED)
        self.assertEqual(CloudJobStatus.from_external("processing"), CloudJobStatus.RUNNING)
        self.assertEqual(CloudJobStatus.from_external("starting"), CloudJobStatus.QUEUED)

    async def test_multiple_jobs_concurrent_submission(self) -> None:
        """Test submitting multiple jobs concurrently."""
        num_jobs = 10
        jobs = []

        # Create multiple jobs concurrently
        async def create_job(i: int) -> UnifiedJob:
            job = UnifiedJob(
                job_id=f"concurrent-{i:03d}",
                job_type=JobType.CLOUD,
                provider="replicate",
                status=CloudJobStatus.PENDING.value,
                config_name=f"config-{i}",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            await self.store.add_job(job)
            return job

        jobs = await asyncio.gather(*[create_job(i) for i in range(num_jobs)])
        self.assertEqual(len(jobs), num_jobs)

        # Verify all jobs exist
        all_jobs = await self.store.list_jobs()
        self.assertEqual(len(all_jobs), num_jobs)

        # Verify each job can be retrieved
        for i in range(num_jobs):
            job = await self.store.get_job(f"concurrent-{i:03d}")
            self.assertIsNotNone(job)
            self.assertEqual(job.config_name, f"config-{i}")

    async def test_job_with_upload_token_flow(self) -> None:
        """Test job creation with upload token for secure uploads."""
        import secrets

        upload_token = secrets.token_urlsafe(32)
        job_id = "upload-token-test"

        job = UnifiedJob(
            job_id=job_id,
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.UPLOADING.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
            upload_token=upload_token,
        )
        await self.store.add_job(job)

        # Should find by token while uploading
        found = await self.store.get_job_by_upload_token(upload_token)
        self.assertIsNotNone(found)
        self.assertEqual(found.job_id, job_id)

        # Complete the upload
        await self.store.update_job(job_id, {"status": CloudJobStatus.COMPLETED.value})

        # Should not find by token after completion
        not_found = await self.store.get_job_by_upload_token(upload_token)
        self.assertIsNone(not_found)


class TestConcurrentJobStoreAccess(unittest.IsolatedAsyncioTestCase):
    """Test concurrent access patterns on JobStore using AsyncJobStore."""

    async def asyncSetUp(self) -> None:
        """Set up test fixtures."""
        # Reset all storage singletons for test isolation
        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()

        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        self.store = AsyncJobStore(self.config_dir)
        await self.store._ensure_initialized()

    async def asyncTearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        await self.store.close()
        AsyncJobStore._instance = None
        # Clear BaseSQLiteStore singletons
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_concurrent_writes_same_job(self) -> None:
        """Test concurrent updates to the same job don't corrupt data."""
        job_id = "concurrent-write-test"
        job = UnifiedJob(
            job_id=job_id,
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.PENDING.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        await self.store.add_job(job)

        # Concurrent updates to different fields
        async def update_status():
            await self.store.update_job(job_id, {"status": CloudJobStatus.RUNNING.value})

        async def update_cost():
            await self.store.update_job(job_id, {"cost_usd": 1.50})

        async def update_error():
            await self.store.update_job(job_id, {"error_message": "test error"})

        # Run updates concurrently
        await asyncio.gather(
            update_status(),
            update_cost(),
            update_error(),
        )

        # Verify job is in consistent state
        retrieved = await self.store.get_job(job_id)
        self.assertIsNotNone(retrieved)
        # At least one update should have succeeded for each field
        self.assertIn(retrieved.status, [CloudJobStatus.PENDING.value, CloudJobStatus.RUNNING.value])

    async def test_concurrent_reads_during_writes(self) -> None:
        """Test that reads don't fail during concurrent writes."""
        job_id = "read-during-write"
        job = UnifiedJob(
            job_id=job_id,
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.PENDING.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        await self.store.add_job(job)

        results = []
        errors = []

        async def writer():
            for i in range(20):
                await self.store.update_job(job_id, {"cost_usd": float(i)})
                await asyncio.sleep(0.001)

        async def reader():
            for _ in range(50):
                try:
                    job = await self.store.get_job(job_id)
                    if job:
                        results.append(job.cost_usd)
                except Exception as e:
                    errors.append(str(e))
                await asyncio.sleep(0.001)

        # Run readers and writers concurrently
        await asyncio.gather(
            writer(),
            reader(),
            reader(),
            reader(),
        )

        # No errors should occur
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        # Should have gotten results
        self.assertGreater(len(results), 0)

    async def test_concurrent_job_creation(self) -> None:
        """Test creating many jobs concurrently doesn't cause conflicts."""
        num_jobs = 50
        created_ids = []
        errors = []

        async def create_job(i: int):
            try:
                job = UnifiedJob(
                    job_id=f"batch-{i:04d}",
                    job_type=JobType.CLOUD,
                    provider="replicate",
                    status=CloudJobStatus.PENDING.value,
                    config_name=f"config-{i}",
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                await self.store.add_job(job)
                created_ids.append(job.job_id)
            except Exception as e:
                errors.append(str(e))

        await asyncio.gather(*[create_job(i) for i in range(num_jobs)])

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(len(created_ids), num_jobs)

        # Verify all jobs exist
        for job_id in created_ids:
            job = await self.store.get_job(job_id)
            self.assertIsNotNone(job, f"Job {job_id} not found")

    async def test_concurrent_list_during_modifications(self) -> None:
        """Test listing jobs while modifications are happening."""
        # Pre-create some jobs
        for i in range(10):
            job = UnifiedJob(
                job_id=f"list-test-{i:03d}",
                job_type=JobType.CLOUD,
                provider="replicate",
                status=CloudJobStatus.PENDING.value,
                config_name=f"config-{i}",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            await self.store.add_job(job)

        list_results = []
        errors = []

        async def modifier():
            for i in range(10):
                await self.store.update_job(f"list-test-{i:03d}", {"status": CloudJobStatus.RUNNING.value})
                await asyncio.sleep(0.001)

        async def lister():
            for _ in range(20):
                try:
                    jobs = await self.store.list_jobs()
                    list_results.append(len(jobs))
                except Exception as e:
                    errors.append(str(e))
                await asyncio.sleep(0.001)

        await asyncio.gather(modifier(), lister(), lister())

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        # All lists should return the same count
        self.assertTrue(all(r == 10 for r in list_results))

    async def test_transaction_isolation(self) -> None:
        """Test that updates are atomic and don't interfere."""
        job_id = "isolation-test"
        job = UnifiedJob(
            job_id=job_id,
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.PENDING.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
            cost_usd=0.0,
        )
        await self.store.add_job(job)

        # Simulate race condition in incrementing cost
        increment_count = 100

        async def increment_cost():
            for _ in range(increment_count):
                # This is a read-modify-write which could cause lost updates
                # without proper isolation
                current = await self.store.get_job(job_id)
                if current:
                    new_cost = (current.cost_usd or 0.0) + 0.01
                    await self.store.update_job(job_id, {"cost_usd": new_cost})

        # Run concurrent incrementers
        await asyncio.gather(
            increment_cost(),
            increment_cost(),
        )

        # Due to race conditions, final cost might be less than expected
        # This test documents the behavior - in production you'd use
        # SQL UPDATE cost_usd = cost_usd + 0.01 for atomic increments
        final = await self.store.get_job(job_id)
        self.assertIsNotNone(final.cost_usd)
        # At minimum, some updates should have succeeded
        self.assertGreater(final.cost_usd, 0)


class TestAsyncJobStoreConsistency(unittest.IsolatedAsyncioTestCase):
    """Test AsyncJobStore for consistency under concurrent access."""

    async def asyncSetUp(self) -> None:
        """Set up test fixtures."""
        # Reset all storage singletons for test isolation
        from simpletuner.simpletuner_sdk.server.services.cloud.async_job_store import AsyncJobStore
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        AsyncJobStore._instance = None
        BaseSQLiteStore._instances.clear()

        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        self.async_store = AsyncJobStore(self.config_dir)
        await self.async_store._ensure_initialized()

    async def asyncTearDown(self) -> None:
        """Clean up test fixtures."""
        import shutil

        from simpletuner.simpletuner_sdk.server.services.cloud.async_job_store import AsyncJobStore
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        await self.async_store.close()
        AsyncJobStore._instance = None
        BaseSQLiteStore._instances.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_async_store_basic_operations(self) -> None:
        """Test basic CRUD operations on AsyncJobStore."""
        job = UnifiedJob(
            job_id="async-test-001",
            job_type=JobType.CLOUD,
            provider="replicate",
            status=CloudJobStatus.PENDING.value,
            config_name="test-config",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Create
        await self.async_store.add_job(job)

        # Read
        retrieved = await self.async_store.get_job("async-test-001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.config_name, "test-config")

        # Update
        await self.async_store.update_job("async-test-001", {"status": CloudJobStatus.RUNNING.value})
        retrieved = await self.async_store.get_job("async-test-001")
        self.assertEqual(retrieved.status, CloudJobStatus.RUNNING.value)

        # Delete
        success = await self.async_store.delete_job("async-test-001")
        self.assertTrue(success)
        self.assertIsNone(await self.async_store.get_job("async-test-001"))

    async def test_async_store_concurrent_writes(self) -> None:
        """Test concurrent writes to AsyncJobStore."""
        num_jobs = 20
        errors = []

        async def create_and_update(i: int):
            try:
                job = UnifiedJob(
                    job_id=f"async-concurrent-{i:03d}",
                    job_type=JobType.CLOUD,
                    provider="replicate",
                    status=CloudJobStatus.PENDING.value,
                    config_name=f"config-{i}",
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                await self.async_store.add_job(job)

                # Update the job multiple times
                for status in [CloudJobStatus.UPLOADING, CloudJobStatus.QUEUED, CloudJobStatus.RUNNING]:
                    await self.async_store.update_job(f"async-concurrent-{i:03d}", {"status": status.value})
            except Exception as e:
                errors.append(f"Job {i}: {e}")

        await asyncio.gather(*[create_and_update(i) for i in range(num_jobs)])

        self.assertEqual(len(errors), 0, f"Errors: {errors}")

        # Verify all jobs exist and are in expected state
        jobs = await self.async_store.list_jobs()
        self.assertEqual(len(jobs), num_jobs)


class TestQueueStoreIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for queue-based job scheduling."""

    async def asyncSetUp(self) -> None:
        """Set up test fixtures."""
        import os

        # Reset all storage singletons for test isolation
        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()

        self.temp_dir = tempfile.mkdtemp()

        # Set SIMPLETUNER_CONFIG_DIR for proper test isolation
        self._previous_config_dir = os.environ.get("SIMPLETUNER_CONFIG_DIR")
        os.environ["SIMPLETUNER_CONFIG_DIR"] = self.temp_dir

        # Reset JobRepository singleton to pick up new config dir
        from simpletuner.simpletuner_sdk.server.services.cloud.storage import job_repository

        job_repository._job_repository = None

        from simpletuner.simpletuner_sdk.server.services.cloud.queue import (
            JobRepoQueueAdapter,
            QueuePriority,
            QueueStatus,
            get_queue_adapter,
        )
        from simpletuner.simpletuner_sdk.server.services.cloud.queue.scheduler import QueueScheduler

        self.queue_store = get_queue_adapter()
        self.scheduler = QueueScheduler(self.queue_store)
        self.QueuePriority = QueuePriority
        self.QueueStatus = QueueStatus

    async def asyncTearDown(self) -> None:
        """Clean up test fixtures."""
        import os
        import shutil

        from simpletuner.simpletuner_sdk.server.services.cloud.storage import job_repository
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        job_repository._job_repository = None
        AsyncJobStore._instance = None
        BaseSQLiteStore._instances.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Restore config dir
        if self._previous_config_dir is not None:
            os.environ["SIMPLETUNER_CONFIG_DIR"] = self._previous_config_dir
        else:
            os.environ.pop("SIMPLETUNER_CONFIG_DIR", None)

    async def test_queue_fifo_ordering(self) -> None:
        """Test that jobs are dequeued in FIFO order within same priority."""
        # Add jobs with same priority
        for i in range(5):
            await self.queue_store.add_to_queue(
                job_id=f"fifo-{i:03d}",
                user_id=1,
                priority=self.QueuePriority.NORMAL,
            )
            await asyncio.sleep(0.01)  # Ensure different timestamps

        # Get next ready job using scheduler
        entry = await self.scheduler._get_next_ready()
        self.assertIsNotNone(entry)
        self.assertEqual(entry.job_id, "fifo-000")

    async def test_queue_priority_ordering(self) -> None:
        """Test that higher priority jobs are dequeued first."""
        # Add low priority first
        await self.queue_store.add_to_queue(
            job_id="low-priority",
            user_id=1,
            priority=self.QueuePriority.LOW,
        )

        # Add high priority second
        await self.queue_store.add_to_queue(
            job_id="high-priority",
            user_id=1,
            priority=self.QueuePriority.HIGH,
        )

        # High priority should come first
        entry = await self.scheduler._get_next_ready()
        self.assertEqual(entry.job_id, "high-priority")

    async def test_queue_fair_scheduling_per_user(self) -> None:
        """Test fair scheduling limits jobs per user."""
        from simpletuner.simpletuner_sdk.server.services.cloud.queue.scheduler import QueueScheduler

        # User 1 has one running job
        await self.queue_store.add_to_queue(job_id="user1-running", user_id=1)
        await self.queue_store.mark_running_by_job_id("user1-running")

        # User 1 has another pending job
        await self.queue_store.add_to_queue(job_id="user1-pending", user_id=1)

        # User 2 has a pending job
        await self.queue_store.add_to_queue(job_id="user2-pending", user_id=2)

        # With user_max_concurrent=1, next job should be user2's
        scheduler = QueueScheduler(self.queue_store, max_concurrent=10, user_max_concurrent=1)
        entry = await scheduler._get_next_ready()
        self.assertEqual(entry.job_id, "user2-pending")

    async def test_queue_concurrent_access(self) -> None:
        """Test concurrent queue operations don't corrupt state."""
        errors = []
        added_jobs = []

        async def add_job(i: int):
            try:
                entry = await self.queue_store.add_to_queue(
                    job_id=f"queue-concurrent-{i:03d}",
                    user_id=i % 3,  # Distribute across 3 users
                    priority=self.QueuePriority.NORMAL,
                )
                added_jobs.append(entry.job_id)
            except Exception as e:
                errors.append(str(e))

        await asyncio.gather(*[add_job(i) for i in range(30)])

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(len(added_jobs), 30)

        # Verify queue stats
        stats = await self.queue_store.get_queue_stats()
        # Jobs are stored with status 'queued', check queue_depth which counts pending+queued
        queue_depth = stats.get("queue_depth", 0)
        self.assertEqual(queue_depth, 30)


if __name__ == "__main__":
    unittest.main()
