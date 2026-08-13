"""Tests for cancelling Kubeflow-backed local jobs."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from simpletuner.simpletuner_sdk.server.routes.cloud.jobs import cancel_job
from simpletuner.simpletuner_sdk.server.services.cloud.base import JobType, UnifiedJob


class KubeflowCancelRouteTestCase(unittest.IsolatedAsyncioTestCase):
    """Test that cancellation delegates to the Kubernetes lifecycle owner."""

    async def test_cancel_starting_job_skips_local_process_and_gpu_allocators(self) -> None:
        """Verify a starting Kubeflow job remains cancellable by its resource owner."""
        job = UnifiedJob(
            job_id="kjob-123",
            job_type=JobType.LOCAL,
            provider="kubeflow",
            status="starting",
            config_name="sdxl-lora",
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={"worker_id": "worker-123"},
        )
        store = AsyncMock()
        store.get_job.return_value = job
        service = AsyncMock()
        request = MagicMock()
        request.client.host = "10.0.0.1"

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.cloud.jobs.get_job_store",
                return_value=store,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.kubeflow_job_service.get_kubeflow_job_service",
                return_value=service,
            ),
            patch(
                "simpletuner.simpletuner_sdk.process_keeper.terminate_process",
                return_value=True,
            ) as terminate_process,
            patch(
                "simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.get_gpu_allocator",
            ) as get_gpu_allocator,
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.audit.audit_log",
                new=AsyncMock(),
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.sse_manager.get_sse_manager",
            ) as get_sse_manager,
            patch(
                "simpletuner.simpletuner_sdk.server.routes.cloud.jobs.emit_cloud_event",
            ),
        ):
            get_sse_manager.return_value = AsyncMock()
            result = await cancel_job("kjob-123", request, user=None)

        self.assertTrue(result["success"])
        service.cancel.assert_awaited_once_with("kjob-123")
        terminate_process.assert_not_called()
        get_gpu_allocator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
