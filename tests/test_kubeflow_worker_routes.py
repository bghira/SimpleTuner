"""Tests for task-bound Kubeflow Worker protocol behavior."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from simpletuner.simpletuner_sdk.server.models.worker import Worker, WorkerStatus, WorkerType
from simpletuner.simpletuner_sdk.server.routes.workers import (
    HeartbeatRequest,
    JobStatusUpdate,
    WorkerRegistrationRequest,
    register_worker,
    update_job_status,
    worker_heartbeat,
    worker_stream,
    worker_streams,
)


def _bound_worker() -> Worker:
    """Create a provisioned Worker fixture.

    Returns:
        Kubeflow Worker bound to one job.
    """
    return Worker(
        worker_id="worker-1",
        name="worker-1",
        worker_type=WorkerType.EPHEMERAL,
        status=WorkerStatus.CONNECTING,
        token_hash="hash",
        user_id=1,
        provider="kubeflow",
        current_job_id="job-1",
    )


class KubeflowWorkerRoutesTestCase(unittest.IsolatedAsyncioTestCase):
    """Test bound Worker registration, heartbeat, and dispatch."""

    async def asyncTearDown(self) -> None:
        """Remove in-memory Worker streams created by tests."""
        worker_streams.clear()

    async def test_registration_keeps_bound_worker_out_of_idle_pool(self) -> None:
        """Verify a provisioned Worker remains assigned while registering."""
        worker = _bound_worker()
        worker_repository = AsyncMock()
        job_repository = AsyncMock()
        job_repository.get.return_value = MagicMock(status="queued")
        request = MagicMock()
        request.client.host = "10.0.0.2"

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.workers.validate_worker_token",
                new=AsyncMock(return_value=worker),
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository",
                return_value=worker_repository,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository.get_job_repository",
                return_value=job_repository,
            ),
        ):
            response = await register_worker(
                WorkerRegistrationRequest(
                    name="worker-1",
                    persistent=False,
                    provider="kubeflow",
                    current_job_id="job-1",
                ),
                request,
                "token",
            )

        updates = worker_repository.update_worker.await_args.args[1]
        self.assertEqual(updates["status"], WorkerStatus.CONNECTING)
        self.assertIsNone(response.abandon_job)

    async def test_stream_dispatches_prebound_job(self) -> None:
        """Verify SSE connection triggers targeted dispatch."""
        worker = _bound_worker()
        manager = AsyncMock()

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.workers.validate_worker_token",
                new=AsyncMock(return_value=worker),
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.worker_manager.get_worker_manager",
                return_value=manager,
            ),
        ):
            await worker_stream("worker-1", "token")

        manager.dispatch_bound_job.assert_awaited_once_with("worker-1")

    async def test_idle_heartbeat_cannot_unbind_provisioned_worker(self) -> None:
        """Verify a preassigned Worker cannot advertise itself as idle."""
        worker = _bound_worker()
        worker_repository = AsyncMock()

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.workers.validate_worker_token",
                new=AsyncMock(return_value=worker),
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository",
                return_value=worker_repository,
            ),
        ):
            await worker_heartbeat(
                HeartbeatRequest(
                    worker_id="worker-1",
                    status="idle",
                    current_job_id=None,
                ),
                "token",
            )

        updates = worker_repository.update_worker.await_args.args[1]
        self.assertEqual(updates["status"], WorkerStatus.CONNECTING)
        self.assertEqual(updates["current_job_id"], "job-1")


    async def test_completed_requires_central_lora_artifact(self) -> None:
        """Verify a bound Worker cannot complete before central upload."""
        worker = _bound_worker()
        job_repository = AsyncMock()
        job_repository.get.return_value = MagicMock(
            metadata={"worker_id": worker.worker_id},
        )
        worker_repository = AsyncMock()
        service = MagicMock()
        service.finalize = AsyncMock()
        service.artifacts_received.return_value = False

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.workers.validate_worker_token",
                new=AsyncMock(return_value=worker),
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository.get_job_repository",
                return_value=job_repository,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository",
                return_value=worker_repository,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.kubeflow_job_service.get_kubeflow_job_service",
                return_value=service,
            ),
        ):
            with self.assertRaises(HTTPException) as context:
                await update_job_status(
                    "job-1",
                    JobStatusUpdate(status="completed"),
                    "token",
                )

        self.assertEqual(context.exception.status_code, 409)
        job_repository.update.assert_not_awaited()
        service.finalize.assert_not_awaited()

    async def test_completed_finalizes_after_central_lora_artifact(self) -> None:
        """Verify central LoRA receipt unlocks completion and cleanup."""
        worker = _bound_worker()
        job_repository = AsyncMock()
        job_repository.get.return_value = MagicMock(
            metadata={
                "worker_id": worker.worker_id,
                "artifact_upload": {
                    "status": "receiving",
                    "received_files": [
                        "outputs/job-1/tiny-output/pytorch_lora_weights.safetensors"
                    ],
                },
            },
        )
        worker_repository = AsyncMock()
        service = MagicMock()
        service.finalize = AsyncMock()
        service.artifacts_received.return_value = True

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.workers.validate_worker_token",
                new=AsyncMock(return_value=worker),
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository.get_job_repository",
                return_value=job_repository,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository",
                return_value=worker_repository,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.kubeflow_job_service.get_kubeflow_job_service",
                return_value=service,
            ),
        ):
            response = await update_job_status(
                "job-1",
                JobStatusUpdate(status="completed"),
                "token",
            )

        self.assertEqual(response, {"success": True})
        updates = job_repository.update.await_args.args[1]
        self.assertEqual(updates["status"], "completed")
        self.assertEqual(
            updates["metadata"]["artifact_upload"]["status"],
            "complete",
        )
        service.finalize.assert_awaited_once_with("job-1")


if __name__ == "__main__":
    unittest.main()
