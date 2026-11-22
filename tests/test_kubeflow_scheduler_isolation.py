"""Tests that remote jobs never enter the local GPU scheduler."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from simpletuner.simpletuner_sdk.server.services.cloud.base import CloudJobStatus, JobType, UnifiedJob
from simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository import JobRepository
from simpletuner.simpletuner_sdk.server.services.local_gpu_allocator import LocalGPUAllocator


class KubeflowSchedulerIsolationTestCase(unittest.IsolatedAsyncioTestCase):
    """Verify remote jobs are isolated from the local GPU allocator."""

    def setUp(self) -> None:
        """Create an isolated job repository."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = JobRepository(Path(self.temp_dir.name) / "jobs.db")

    def tearDown(self) -> None:
        """Remove the isolated repository."""
        self.repository.reset_instance()
        self.temp_dir.cleanup()

    async def test_remote_jobs_are_excluded_from_local_gpu_queries(self) -> None:
        """Verify local allocation cannot select, count, or fail remote jobs."""
        now = datetime.now(timezone.utc).isoformat()
        local_job = UnifiedJob(
            job_id="local-job",
            job_type=JobType.LOCAL,
            provider="local",
            allocated_gpus=[0],
            status=CloudJobStatus.QUEUED.value,
            config_name="local-config",
            created_at=now,
            queued_at=now,
        )
        kubeflow_job = UnifiedJob(
            job_id="kubeflow-job",
            job_type=JobType.LOCAL,
            provider="kubeflow",
            allocated_gpus=[1],
            status=CloudJobStatus.QUEUED.value,
            config_name="kubeflow-config",
            created_at=now,
            queued_at=now,
        )
        worker_job = replace(kubeflow_job, job_id="worker-job", provider="worker", allocated_gpus=[2])
        await self.repository.add(worker_job)
        await self.repository.add(local_job)
        await self.repository.add(kubeflow_job)

        pending_jobs = await self.repository.get_pending_local_jobs()
        self.assertEqual([job.job_id for job in pending_jobs], ["local-job"])

        await self.repository.mark_running(local_job.job_id)
        await self.repository.mark_running(kubeflow_job.job_id)
        await self.repository.mark_running(worker_job.job_id)
        running_jobs = await self.repository.get_running_local_jobs()
        self.assertEqual([job.job_id for job in running_jobs], ["local-job"])
        self.assertEqual(await self.repository.count_running_local_jobs(), 1)

        self.assertEqual(await self.repository.get_allocated_gpus(), {0})

        allocator = LocalGPUAllocator()
        with patch.object(allocator, "_get_job_repo", return_value=self.repository):
            stats = await allocator.reconcile_on_startup()
        self.assertEqual(stats["no_pid"], 1)
        self.assertEqual((await self.repository.get(worker_job.job_id)).status, "running")
        self.assertEqual((await self.repository.get(kubeflow_job.job_id)).status, "running")


if __name__ == "__main__":
    unittest.main()
