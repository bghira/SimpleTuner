"""Tests for Kubeflow-backed SimpleTuner jobs."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from simpletuner.simpletuner_sdk.server.services.cloud.base import CloudJobStatus, JobType, UnifiedJob
from simpletuner.simpletuner_sdk.server.services.cloud.job_logs import fetch_job_logs
from simpletuner.simpletuner_sdk.server.services.kubeflow import KubeflowPhase, KubeflowResources, KubeflowSettings
from simpletuner.simpletuner_sdk.server.services.kubeflow_job_service import KubeflowJobService


class KubeflowJobServiceTestCase(unittest.IsolatedAsyncioTestCase):
    """Test job persistence and one-shot Worker provisioning."""

    def setUp(self) -> None:
        """Create service fixtures."""
        self.settings = KubeflowSettings(
            enabled=True,
            namespace="simpletuner",
            runtime_name="simpletuner-worker",
            queue_name="gpu-training",
            worker_image="registry.example.com/simpletuner:4.5.0",
            orchestrator_url="http://simpletuner.simpletuner.svc:8001",
        )
        self.job_store = AsyncMock()
        self.worker_repository = AsyncMock()
        self.provisioner = AsyncMock()
        self.provisioner.create.return_value = KubeflowResources(
            namespace="simpletuner",
            trainjob_name="simpletuner-kjob-123",
            secret_name="simpletuner-worker-kjob-123",
            trainjob_uid="uid-123",
        )
        self.service = KubeflowJobService(
            settings=self.settings,
            provisioner=self.provisioner,
            job_store=self.job_store,
            worker_repository=self.worker_repository,
        )

    async def test_submit_creates_bound_worker_and_single_process_job(self) -> None:
        """Verify submission binds one ephemeral Worker to one queued job."""
        config = {"model_family": "sdxl"}
        with (
            patch(
                "simpletuner.simpletuner_sdk.server.services.kubeflow_job_service.uuid.uuid4"
            ) as uuid4,
            patch(
                "simpletuner.simpletuner_sdk.server.services.worker_credentials.generate_worker_token",
                return_value="test-token",
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.kubeflow_job_service.secrets.token_urlsafe",
                return_value="test-upload-token",
                create=True,
            ),
        ):
            uuid4.return_value.hex = "1234567890abcdef"
            job = await self.service.submit(
                config_name="sdxl-lora",
                config=config,
                user_id=7,
            )

        self.assertEqual(job.provider, "kubeflow")
        self.assertEqual(job.status, CloudJobStatus.QUEUED.value)
        self.assertEqual(job.num_processes, 1)
        self.assertEqual(job.metadata["worker_id"], "worker-1234567890ab")
        self.assertEqual(job.metadata["target"], "kubeflow")
        self.assertEqual(job.upload_token, "test-upload-token")
        publishing = job.metadata["config"]["publishing_config"]
        self.assertEqual(
            publishing[-1],
            {
                "provider": "s3",
                "bucket": "outputs",
                "endpoint_url": "http://simpletuner.simpletuner.svc:8001/api/cloud/storage",
                "access_key": "local",
                "secret_key": "test-upload-token",
                "base_path": job.job_id,
                "use_ssl": False,
                "request_headers": {"X-SimpleTuner-Secret": "test-upload-token"},
                "force_single_part": True,
                "required": True,
            },
        )
        self.assertNotIn("publishing_config", config)
        self.job_store.add_job.assert_awaited_once()
        created_worker = self.worker_repository.create_worker.await_args.args[0]
        self.assertEqual(created_worker.current_job_id, job.job_id)
        self.assertEqual(created_worker.provider, "kubeflow")
        self.provisioner.create.assert_awaited_once_with(
            job_id=job.job_id,
            worker_id=created_worker.worker_id,
            worker_token="test-token",
        )

    async def test_fetch_logs_reads_active_worker_pod(self) -> None:
        """Verify the existing logs endpoint polls Kubeflow through the Server."""
        job = UnifiedJob(
            job_id="kjob-123",
            job_type=JobType.LOCAL,
            provider="kubeflow",
            status=CloudJobStatus.RUNNING.value,
            config_name="sdxl-lora",
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={
                "kubernetes": {
                    "namespace": "simpletuner",
                    "trainjob_name": "simpletuner-kjob-123",
                    "secret_name": "simpletuner-worker-kjob-123",
                }
            },
        )
        self.provisioner.get_logs.return_value = "training step 1/1"

        with patch(
            "simpletuner.simpletuner_sdk.server.services.kubeflow_job_service.get_kubeflow_job_service",
            return_value=self.service,
        ):
            logs = await fetch_job_logs(job)

        self.provisioner.get_logs.assert_awaited_once_with("simpletuner-kjob-123")
        self.assertEqual(logs, "training step 1/1")

    async def test_cancel_deletes_trainjob_secret_and_worker(self) -> None:
        """Verify cancellation releases Kubernetes and Worker resources."""
        job = await self.service.submit(
            config_name="sdxl-lora",
            config={"model_family": "sdxl"},
            user_id=7,
        )
        self.job_store.get_job.return_value = job

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.workers.is_worker_connected",
                return_value=False,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.routes.workers.push_to_worker",
                new_callable=AsyncMock,
            ),
        ):
            await self.service.cancel(job.job_id)

        self.provisioner.delete.assert_awaited_once()
        self.worker_repository.delete_worker.assert_awaited_once_with(job.metadata["worker_id"])
        terminal_update = self.job_store.update_job.await_args_list[-1].args[1]
        self.assertEqual(terminal_update["status"], CloudJobStatus.CANCELLED.value)
        self.assertEqual(
            terminal_update["metadata"]["infrastructure_phase"],
            CloudJobStatus.CANCELLED.value,
        )

    async def test_finalize_archives_worker_log_before_deletion(self) -> None:
        """Verify Pod logs reach central storage before resource cleanup."""
        job = UnifiedJob(
            job_id="kjob-123",
            job_type=JobType.LOCAL,
            provider="kubeflow",
            status=CloudJobStatus.COMPLETED.value,
            config_name="sdxl-lora",
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={
                "worker_id": "worker-123",
                "infrastructure_phase": KubeflowPhase.STARTING.value,
                "kubernetes": {
                    "namespace": "simpletuner",
                    "trainjob_name": "simpletuner-kjob-123",
                    "secret_name": "simpletuner-worker-kjob-123",
                },
            },
        )
        self.job_store.get_job.return_value = job
        events = []
        self.provisioner.get_logs.side_effect = lambda *_: events.append("logs") or "training complete"
        self.provisioner.delete.side_effect = lambda *_: events.append("delete")

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "simpletuner.simpletuner_sdk.server.routes.cloud.helpers.get_local_upload_dir",
            return_value=Path(tmpdir),
        ):
            await self.service.finalize(job.job_id)
            archived_log = Path(tmpdir) / "outputs" / job.job_id / "worker.log"
            self.assertEqual(archived_log.read_text(encoding="utf-8"), "training complete")

        self.assertEqual(events, ["logs", "delete"])
        metadata_update = self.job_store.update_job.await_args.args[1]["metadata"]
        self.assertIn(
            f"outputs/{job.job_id}/worker.log",
            metadata_update["artifact_upload"]["received_files"],
        )
        self.assertEqual(
            metadata_update["infrastructure_phase"],
            KubeflowPhase.COMPLETED.value,
        )

    async def test_reconcile_failed_trainjob_releases_bound_resources(self) -> None:
        """Verify an infrastructure failure cannot leak a GPU Worker."""
        job = UnifiedJob(
            job_id="kjob-123",
            job_type=JobType.LOCAL,
            provider="kubeflow",
            status=CloudJobStatus.QUEUED.value,
            config_name="sdxl-lora",
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={
                "worker_id": "worker-123",
                "kubernetes": {
                    "namespace": "simpletuner",
                    "trainjob_name": "simpletuner-kjob-123",
                    "secret_name": "simpletuner-worker-kjob-123",
                },
            },
        )
        self.job_store.list_jobs.return_value = [job]
        self.provisioner.get_phase.return_value = KubeflowPhase.FAILED

        await self.service.reconcile_once()

        self.provisioner.delete.assert_awaited_once()
        self.worker_repository.delete_worker.assert_awaited_once_with("worker-123")
        updates = self.job_store.update_job.await_args.args[1]
        self.assertEqual(updates["status"], CloudJobStatus.FAILED.value)


    async def test_reconcile_completed_trainjob_without_artifact_fails(self) -> None:
        """Verify infrastructure success cannot bypass artifact confirmation."""
        job = UnifiedJob(
            job_id="kjob-123",
            job_type=JobType.LOCAL,
            provider="kubeflow",
            status=CloudJobStatus.RUNNING.value,
            config_name="sdxl-lora",
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={
                "worker_id": "worker-123",
                "kubernetes": {
                    "namespace": "simpletuner",
                    "trainjob_name": "simpletuner-kjob-123",
                    "secret_name": "simpletuner-worker-kjob-123",
                },
            },
        )
        self.job_store.list_jobs.return_value = [job]
        self.provisioner.get_phase.return_value = KubeflowPhase.COMPLETED

        await self.service.reconcile_once()

        updates = self.job_store.update_job.await_args.args[1]
        self.assertEqual(updates["status"], CloudJobStatus.FAILED.value)
        self.assertIn("artifact", updates["error_message"].lower())


if __name__ == "__main__":
    unittest.main()
