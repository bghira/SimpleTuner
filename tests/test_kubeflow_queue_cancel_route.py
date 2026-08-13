"""Tests for cancelling Kubeflow jobs through the generic queue API."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from simpletuner.simpletuner_sdk.server.routes.queue import cancel_queued_job
from simpletuner.simpletuner_sdk.server.services.cloud.base import JobType, UnifiedJob


class KubeflowQueueCancelRouteTestCase(unittest.IsolatedAsyncioTestCase):
    """Verify the generic queue cancellation endpoint is provider-transparent."""

    async def test_cancel_kubeflow_job_uses_service_owned_record(self) -> None:
        """Route a Kubeflow job by the lifecycle service's authoritative store."""
        job = UnifiedJob(
            job_id="kjob-123",
            job_type=JobType.LOCAL,
            provider="kubeflow",
            status="running",
            config_name="sdxl-lora",
            created_at=datetime.now(timezone.utc).isoformat(),
            user_id=7,
        )
        job_store = MagicMock()
        job_store.get_job = AsyncMock(return_value=None)
        queue_store = MagicMock()
        queue_store.get_entry_by_job_id = AsyncMock(return_value=None)
        service = MagicMock()
        service.get_managed_job = AsyncMock(return_value=job)
        service.cancel = AsyncMock()
        user = MagicMock(id=7)
        user.has_permission.return_value = True

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.cloud.dependencies.get_job_store",
                return_value=job_store,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.routes.queue.QueueStore",
                return_value=queue_store,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.kubeflow_job_service.get_kubeflow_job_service",
                return_value=service,
            ),
            patch("simpletuner.simpletuner_sdk.server.routes.queue.get_scheduler") as get_scheduler,
        ):
            result = await cancel_queued_job("kjob-123", user=user)

        self.assertEqual(result, {"success": True, "job_id": "kjob-123"})
        service.cancel.assert_awaited_once_with("kjob-123")
        get_scheduler.assert_not_called()

    async def test_cancel_kubeflow_job_ignores_stale_compatibility_queue(self) -> None:
        """Avoid delegating a managed job to the legacy Scheduler path."""
        job = UnifiedJob(
            job_id="kjob-789",
            job_type=JobType.LOCAL,
            provider="kubeflow",
            status="running",
            config_name="sdxl-lora",
            created_at=datetime.now(timezone.utc).isoformat(),
            user_id=7,
        )
        entry = MagicMock(user_id=7, provider="local")
        job_store = MagicMock()
        job_store.get_job = AsyncMock(return_value=None)
        queue_store = MagicMock()
        queue_store.get_entry_by_job_id = AsyncMock(return_value=entry)
        queue_store.mark_cancelled_by_job_id = AsyncMock(return_value=True)
        service = MagicMock()
        service.get_managed_job = AsyncMock(return_value=job)
        service.cancel = AsyncMock()
        user = MagicMock(id=7)
        user.has_permission.return_value = True

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.cloud.dependencies.get_job_store",
                return_value=job_store,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.routes.queue.QueueStore",
                return_value=queue_store,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.kubeflow_job_service.get_kubeflow_job_service",
                return_value=service,
            ),
            patch("simpletuner.simpletuner_sdk.server.routes.queue.get_scheduler") as get_scheduler,
        ):
            result = await cancel_queued_job("kjob-789", user=user)

        self.assertEqual(result, {"success": True, "job_id": "kjob-789"})
        service.cancel.assert_awaited_once_with("kjob-789")
        get_scheduler.assert_not_called()

    async def test_cancel_regular_queue_job_keeps_scheduler_path(self) -> None:
        """Preserve the existing Scheduler cancellation behavior for normal jobs."""
        entry = MagicMock(user_id=7)
        job_store = MagicMock()
        job_store.get_job = AsyncMock(return_value=None)
        queue_store = MagicMock()
        queue_store.get_entry_by_job_id = AsyncMock(return_value=entry)
        scheduler = MagicMock()
        scheduler.cancel_job = AsyncMock(return_value=True)
        service = MagicMock()
        service.get_managed_job = AsyncMock(return_value=None)
        user = MagicMock(id=7)
        user.has_permission.return_value = True

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.cloud.dependencies.get_job_store",
                return_value=job_store,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.routes.queue.QueueStore",
                return_value=queue_store,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.kubeflow_job_service.get_kubeflow_job_service",
                return_value=service,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.routes.queue.get_scheduler",
                return_value=scheduler,
            ),
        ):
            result = await cancel_queued_job("job-456", user=user)

        self.assertEqual(result, {"success": True, "job_id": "job-456"})
        scheduler.cancel_job.assert_awaited_once_with("job-456")


if __name__ == "__main__":
    unittest.main()
