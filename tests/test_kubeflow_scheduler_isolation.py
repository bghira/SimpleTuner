"""Tests that Kubeflow jobs never enter the local GPU scheduler."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from simpletuner.simpletuner_sdk.server.services.cloud.base import CloudJobStatus, JobType, UnifiedJob
from simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository import JobRepository


class KubeflowSchedulerIsolationTestCase(unittest.IsolatedAsyncioTestCase):
    """Verify Kubernetes owns GPU admission for Kubeflow jobs."""

    def setUp(self) -> None:
        """Create an isolated job repository."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = JobRepository(Path(self.temp_dir.name) / "jobs.db")

    def tearDown(self) -> None:
        """Remove the isolated repository."""
        self.repository.reset_instance()
        self.temp_dir.cleanup()

    async def test_kubeflow_jobs_are_excluded_from_local_gpu_queries(self) -> None:
        """Verify local allocation cannot select or count a Kubeflow job."""
        now = datetime.now(timezone.utc).isoformat()
        local_job = UnifiedJob(
            job_id="local-job",
            job_type=JobType.LOCAL,
            provider="local",
            status=CloudJobStatus.QUEUED.value,
            config_name="local-config",
            created_at=now,
            queued_at=now,
        )
        kubeflow_job = UnifiedJob(
            job_id="kubeflow-job",
            job_type=JobType.LOCAL,
            provider="kubeflow",
            status=CloudJobStatus.QUEUED.value,
            config_name="kubeflow-config",
            created_at=now,
            queued_at=now,
        )
        await self.repository.add(local_job)
        await self.repository.add(kubeflow_job)

        pending_jobs = await self.repository.get_pending_local_jobs()
        self.assertEqual([job.job_id for job in pending_jobs], ["local-job"])

        await self.repository.mark_running(local_job.job_id)
        await self.repository.mark_running(kubeflow_job.job_id)
        running_jobs = await self.repository.get_running_local_jobs()
        self.assertEqual([job.job_id for job in running_jobs], ["local-job"])
        self.assertEqual(await self.repository.count_running_local_jobs(), 1)


if __name__ == "__main__":
    unittest.main()
