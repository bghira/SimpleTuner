"""Tests for explicit Kubeflow queue submission."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from simpletuner.simpletuner_sdk.server.routes.queue import LocalJobSubmitRequest, submit_local_job
from simpletuner.simpletuner_sdk.server.services.cloud.base import JobType, UnifiedJob


class KubeflowQueueRouteTestCase(unittest.IsolatedAsyncioTestCase):
    """Test that Kubeflow submissions bypass SimpleTuner GPU selection."""

    async def test_submit_kubeflow_bypasses_local_and_worker_allocators(self) -> None:
        """Verify Kueue is the only GPU admission path for this target."""
        config_store = MagicMock()
        config_store.load_config.return_value = ({"model_family": "sdxl"}, MagicMock())
        defaults = MagicMock(configs_dir="/configs")
        state_store = MagicMock()
        state_store.load_defaults.return_value = defaults
        service = AsyncMock()
        service.submit.return_value = UnifiedJob(
            job_id="kjob-123",
            job_type=JobType.LOCAL,
            provider="kubeflow",
            status="queued",
            config_name="sdxl-lora",
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={"worker_id": "worker-123"},
        )

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore",
                return_value=state_store,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.config_store.ConfigStore",
                return_value=config_store,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.kubeflow_job_service.get_kubeflow_job_service",
                return_value=service,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.training_service.start_training_job"
            ) as start_local,
            patch(
                "simpletuner.simpletuner_sdk.server.services.worker_repository.get_worker_repository"
            ) as get_worker_repository,
        ):
            result = await submit_local_job(
                LocalJobSubmitRequest(config_name="sdxl-lora", target="kubeflow"),
                user=None,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status, "queued")
        self.assertEqual(result.allocated_worker_id, "worker-123")
        start_local.assert_not_called()
        get_worker_repository.assert_not_called()


if __name__ == "__main__":
    unittest.main()
