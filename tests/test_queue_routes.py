"""Tests for queue routes.

Tests cover:
- Queue stats endpoint with local GPU info
- Local GPU stats population from GPU allocator
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.unittest_support import AsyncAPITestCase


class TestQueueStatsEndpoint(AsyncAPITestCase, unittest.IsolatedAsyncioTestCase):
    """Tests for GET /api/queue/stats endpoint."""

    async def asyncSetUp(self) -> None:
        """Set up test fixtures."""
        await super().asyncSetUp()

        # Reset GPU allocator singleton
        from simpletuner.simpletuner_sdk.server.services import local_gpu_allocator

        local_gpu_allocator._allocator_instance = None

    async def asyncTearDown(self) -> None:
        """Clean up test fixtures."""
        # Reset GPU allocator singleton
        from simpletuner.simpletuner_sdk.server.services import local_gpu_allocator

        local_gpu_allocator._allocator_instance = None

        await super().asyncTearDown()

    async def test_queue_stats_includes_local_gpu_info(self) -> None:
        """Test that queue stats endpoint returns local GPU allocation info.

        This test verifies the fix for the bug where local_stats was None
        because the code referenced allocator._queue_store instead of
        allocator._get_job_repo().
        """
        from simpletuner.simpletuner_sdk.server.routes.queue import LocalGPUStats, QueueStatsResponse, get_queue_stats
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User

        # Mock the dependencies
        mock_scheduler = MagicMock()
        mock_scheduler.get_queue_overview = AsyncMock(
            return_value={
                "by_status": {"pending": 2, "running": 1},
                "by_user": {1: 2, 2: 1},
                "queue_depth": 2,
                "running": 1,
                "max_concurrent": 5,
                "user_max_concurrent": 2,
                "team_max_concurrent": 10,
                "enable_fair_share": False,
            }
        )

        # Mock GPU allocator to return status
        mock_gpu_status = {
            "total_gpus": 4,
            "allocated_gpus": [0, 1],
            "available_gpus": [2, 3],
            "running_local_jobs": 1,
            "backend": "cuda",
            "devices": [
                {"index": 0, "name": "GPU 0", "memory_gb": 24, "allocated": True, "job_id": "job-123"},
                {"index": 1, "name": "GPU 1", "memory_gb": 24, "allocated": True, "job_id": "job-123"},
                {"index": 2, "name": "GPU 2", "memory_gb": 24, "allocated": False, "job_id": None},
                {"index": 3, "name": "GPU 3", "memory_gb": 24, "allocated": False, "job_id": None},
            ],
        }

        mock_allocator = MagicMock()
        mock_allocator.get_gpu_status = AsyncMock(return_value=mock_gpu_status)

        # Mock job repo for pending jobs
        mock_job_repo = MagicMock()
        mock_job_repo.get_pending_local_jobs = AsyncMock(return_value=[])
        mock_allocator._get_job_repo = MagicMock(return_value=mock_job_repo)

        # Mock WebUIStateStore defaults
        mock_defaults = MagicMock()
        mock_defaults.local_gpu_max_concurrent = 6
        mock_defaults.local_job_max_concurrent = 2

        mock_state_store = MagicMock()
        mock_state_store.load_defaults = MagicMock(return_value=mock_defaults)

        # Mock user with queue.view permission
        mock_user = MagicMock(spec=User)
        mock_user.has_permission = MagicMock(return_value=True)

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.queue.get_scheduler",
                return_value=mock_scheduler,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.get_gpu_allocator",
                return_value=mock_allocator,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore",
                return_value=mock_state_store,
            ),
        ):
            result = await get_queue_stats(user=mock_user)

        # Verify the response contains local GPU stats
        self.assertIsInstance(result, QueueStatsResponse)
        self.assertIsNotNone(result.local, "local GPU stats should not be None")
        self.assertIsInstance(result.local, LocalGPUStats)

        # Verify local GPU stats values
        self.assertEqual(result.local.running_jobs, 1)
        self.assertEqual(result.local.pending_jobs, 0)
        self.assertEqual(result.local.allocated_gpus, [0, 1])
        self.assertEqual(result.local.available_gpus, [2, 3])
        self.assertEqual(result.local.total_gpus, 4)
        self.assertEqual(result.local.max_concurrent_gpus, 6)
        self.assertEqual(result.local.max_concurrent_jobs, 2)

        # Verify top-level stats
        self.assertEqual(result.queue_depth, 2)
        self.assertEqual(result.running, 1)
        self.assertEqual(result.local_gpu_max_concurrent, 6)
        self.assertEqual(result.local_job_max_concurrent, 2)

    async def test_queue_stats_handles_allocator_exception(self) -> None:
        """Test that queue stats endpoint handles GPU allocator errors gracefully.

        When GPU allocator fails, local should be None but other stats should
        still be returned.
        """
        from simpletuner.simpletuner_sdk.server.routes.queue import QueueStatsResponse, get_queue_stats
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User

        mock_scheduler = MagicMock()
        mock_scheduler.get_queue_overview = AsyncMock(
            return_value={
                "by_status": {},
                "by_user": {},
                "queue_depth": 0,
                "running": 0,
                "max_concurrent": 5,
                "user_max_concurrent": 2,
                "team_max_concurrent": 10,
                "enable_fair_share": False,
            }
        )

        # Mock allocator that raises an exception
        def raise_error():
            raise RuntimeError("GPU error")

        mock_user = MagicMock(spec=User)
        mock_user.has_permission = MagicMock(return_value=True)

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.queue.get_scheduler",
                return_value=mock_scheduler,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.get_gpu_allocator",
                side_effect=raise_error,
            ),
        ):
            result = await get_queue_stats(user=mock_user)

        # Should still return a valid response
        self.assertIsInstance(result, QueueStatsResponse)
        # local should be None when allocator fails
        self.assertIsNone(result.local)
        # Other stats should still be populated
        self.assertEqual(result.queue_depth, 0)
        self.assertEqual(result.running, 0)

    async def test_queue_stats_with_pending_local_jobs(self) -> None:
        """Test that pending_jobs count is correctly populated from job repo."""
        from simpletuner.simpletuner_sdk.server.routes.queue import LocalGPUStats, QueueStatsResponse, get_queue_stats
        from simpletuner.simpletuner_sdk.server.services.cloud.auth.models import User
        from simpletuner.simpletuner_sdk.server.services.cloud.base import UnifiedJob

        mock_scheduler = MagicMock()
        mock_scheduler.get_queue_overview = AsyncMock(
            return_value={
                "by_status": {},
                "by_user": {},
                "queue_depth": 0,
                "running": 0,
                "max_concurrent": 5,
                "user_max_concurrent": 2,
                "team_max_concurrent": 10,
                "enable_fair_share": False,
            }
        )

        mock_gpu_status = {
            "total_gpus": 2,
            "allocated_gpus": [0, 1],
            "available_gpus": [],
            "running_local_jobs": 1,
            "backend": "cuda",
            "devices": [],
        }

        # Create mock pending jobs
        mock_pending_job_1 = MagicMock(spec=UnifiedJob)
        mock_pending_job_2 = MagicMock(spec=UnifiedJob)
        mock_pending_jobs = [mock_pending_job_1, mock_pending_job_2]

        mock_job_repo = MagicMock()
        mock_job_repo.get_pending_local_jobs = AsyncMock(return_value=mock_pending_jobs)

        mock_allocator = MagicMock()
        mock_allocator.get_gpu_status = AsyncMock(return_value=mock_gpu_status)
        mock_allocator._get_job_repo = MagicMock(return_value=mock_job_repo)

        mock_defaults = MagicMock()
        mock_defaults.local_gpu_max_concurrent = None
        mock_defaults.local_job_max_concurrent = 1

        mock_state_store = MagicMock()
        mock_state_store.load_defaults = MagicMock(return_value=mock_defaults)

        mock_user = MagicMock(spec=User)
        mock_user.has_permission = MagicMock(return_value=True)

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.queue.get_scheduler",
                return_value=mock_scheduler,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.get_gpu_allocator",
                return_value=mock_allocator,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore",
                return_value=mock_state_store,
            ),
        ):
            result = await get_queue_stats(user=mock_user)

        self.assertIsNotNone(result.local)
        # Verify pending_jobs reflects the mocked pending jobs count
        self.assertEqual(result.local.pending_jobs, 2)
        self.assertEqual(result.local.running_jobs, 1)


if __name__ == "__main__":
    unittest.main()
