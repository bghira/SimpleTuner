"""Tests for training service GPU-aware queuing.

Tests cover:
- get_gpu_requirements config parsing
- TrainingJobResult dataclass
- start_training_job with no_wait and any_gpu options
- _queue_training_job function
- GPU release on job termination
"""

from __future__ import annotations

import unittest

from simpletuner.simpletuner_sdk.server.services.training_service import (
    TrainingJobResult,
    _queue_training_job,
    _queued_preferred_gpus,
    get_gpu_requirements,
)


class TestGetGPURequirements(unittest.TestCase):
    """Tests for get_gpu_requirements config parsing."""

    def test_default_values(self):
        """Test defaults when no GPU config specified."""
        config = {}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(num_processes, 1)
        self.assertIsNone(device_ids)

    def test_num_processes_from_flag(self):
        """Test parsing --num_processes flag."""
        config = {"--num_processes": 4}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(num_processes, 4)

    def test_num_processes_from_key(self):
        """Test parsing num_processes key without prefix."""
        config = {"num_processes": 2}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(num_processes, 2)

    def test_num_processes_flag_priority(self):
        """Test --num_processes takes priority over num_processes."""
        config = {"--num_processes": 4, "num_processes": 2}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(num_processes, 4)

    def test_num_processes_as_string(self):
        """Test parsing num_processes as string."""
        config = {"--num_processes": "8"}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(num_processes, 8)

    def test_num_processes_invalid(self):
        """Test invalid num_processes defaults to 1."""
        config = {"--num_processes": "invalid"}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(num_processes, 1)

    def test_num_processes_zero(self):
        """Test zero num_processes becomes 1."""
        config = {"--num_processes": 0}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(num_processes, 1)

    def test_num_processes_negative(self):
        """Test negative num_processes becomes 1."""
        config = {"--num_processes": -2}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(num_processes, 1)

    def test_device_ids_from_flag_string(self):
        """Test parsing --accelerate_visible_devices as string."""
        config = {"--accelerate_visible_devices": "0,1,2"}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(device_ids, [0, 1, 2])

    def test_device_ids_from_key_string(self):
        """Test parsing accelerate_visible_devices without prefix."""
        config = {"accelerate_visible_devices": "0,2,4"}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(device_ids, [0, 2, 4])

    def test_device_ids_with_spaces(self):
        """Test parsing device_ids with spaces."""
        config = {"--accelerate_visible_devices": "0, 1, 2"}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(device_ids, [0, 1, 2])

    def test_device_ids_as_list(self):
        """Test parsing device_ids as list."""
        config = {"--accelerate_visible_devices": [0, 1, 3]}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(device_ids, [0, 1, 3])

    def test_device_ids_as_tuple(self):
        """Test parsing device_ids as tuple."""
        config = {"--accelerate_visible_devices": (0, 2)}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(device_ids, [0, 2])

    def test_device_ids_mixed_list(self):
        """Test parsing device_ids list with string elements."""
        config = {"--accelerate_visible_devices": ["0", "1", "2"]}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(device_ids, [0, 1, 2])

    def test_device_ids_invalid_in_string(self):
        """Test invalid entries in device_ids string are skipped."""
        config = {"--accelerate_visible_devices": "0,invalid,2"}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(device_ids, [0, 2])

    def test_device_ids_empty_string(self):
        """Test empty device_ids string returns None."""
        config = {"--accelerate_visible_devices": ""}
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertIsNone(device_ids)

    def test_combined_config(self):
        """Test parsing both num_processes and device_ids."""
        config = {
            "--num_processes": 4,
            "--accelerate_visible_devices": "0,1,2,3",
        }
        num_processes, device_ids = get_gpu_requirements(config)

        self.assertEqual(num_processes, 4)
        self.assertEqual(device_ids, [0, 1, 2, 3])


class TestTrainingJobResult(unittest.TestCase):
    """Tests for TrainingJobResult dataclass."""

    def test_running_result(self):
        """Test result for running job."""
        result = TrainingJobResult(
            job_id="abc123",
            status="running",
            allocated_gpus=[0, 1],
        )

        self.assertEqual(result.job_id, "abc123")
        self.assertEqual(result.status, "running")
        self.assertEqual(result.allocated_gpus, [0, 1])
        self.assertIsNone(result.queue_position)
        self.assertIsNone(result.reason)

    def test_queued_result(self):
        """Test result for queued job."""
        result = TrainingJobResult(
            job_id="def456",
            status="queued",
            queue_position=3,
            reason="Waiting for GPUs",
        )

        self.assertEqual(result.job_id, "def456")
        self.assertEqual(result.status, "queued")
        self.assertIsNone(result.allocated_gpus)
        self.assertEqual(result.queue_position, 3)
        self.assertEqual(result.reason, "Waiting for GPUs")

    def test_rejected_result(self):
        """Test result for rejected job."""
        result = TrainingJobResult(
            job_id=None,
            status="rejected",
            reason="GPUs unavailable",
        )

        self.assertIsNone(result.job_id)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.reason, "GPUs unavailable")


class TestStartTrainingJobGPUAware(unittest.TestCase):
    """Tests for start_training_job GPU-aware behavior.

    Note: These tests are disabled because start_training_job uses
    local imports which are difficult to mock correctly. The core
    GPU allocation logic is tested in test_local_gpu_allocator.py.
    """

    def test_start_job_integration_skipped(self):
        """Placeholder - integration tests require running server."""
        # The actual integration is tested through:
        # 1. test_local_gpu_allocator.py - tests GPU allocation logic
        # 2. test_queue_store_gpu.py - tests database operations
        # 3. API tests via Jest - tests HTTP endpoints
        pass


class TestQueueTrainingJob(unittest.TestCase):
    """Tests for _queue_training_job function.

    Note: These tests are placeholders since _queue_training_job uses
    local imports. The queue functionality is tested through:
    - test_queue_store_gpu.py for database operations
    - Jest tests for API endpoints
    """

    def test_queue_job_placeholder(self):
        """Placeholder for queue job tests."""
        # Queue job functionality is tested through:
        # 1. test_queue_store_gpu.py - tests add_to_queue with GPU fields
        # 2. local_gpu.test.js - tests API submission behavior
        pass

    def test_incomplete_preferred_gpus_are_not_persisted(self):
        """A queued 2-GPU job should not hard-pin itself to one GPU forever."""
        self.assertIsNone(_queued_preferred_gpus([0], num_processes=2, any_gpu=False))

    def test_complete_preferred_gpus_are_persisted(self):
        """Complete preferences remain hard preferences for queued jobs."""
        self.assertEqual(_queued_preferred_gpus([0, 1], num_processes=2, any_gpu=False), [0, 1])

    def test_any_gpu_drops_preferred_gpus(self):
        """The any-GPU mode should let the allocator choose later."""
        self.assertIsNone(_queued_preferred_gpus([0, 1], num_processes=2, any_gpu=True))

    def test_queued_job_processes_pending_when_approval_not_required(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_repo = MagicMock()
        mock_repo.get_queue_stats = AsyncMock(return_value={"queue_depth": 0})
        mock_repo.add = AsyncMock()

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository.get_job_repository",
                return_value=mock_repo,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.training_service._process_pending_local_jobs"
            ) as mock_process,
            patch(
                "simpletuner.simpletuner_sdk.server.services.training_service._detect_local_hardware",
                return_value="local",
            ),
        ):
            result = _queue_training_job(
                job_id="queued-job",
                runtime_config={"--output_dir": "/tmp/out"},
                env_name="test-env",
                num_processes=2,
                preferred_gpus=None,
                any_gpu=False,
                org_id=None,
                user_id=None,
            )

        self.assertEqual(result.status, "queued")
        mock_repo.add.assert_awaited_once()
        mock_process.assert_called_once_with()

    def test_approval_queue_does_not_process_pending(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_repo = MagicMock()
        mock_repo.get_queue_stats = AsyncMock(return_value={"queue_depth": 0})
        mock_repo.add = AsyncMock()

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.services.cloud.storage.job_repository.get_job_repository",
                return_value=mock_repo,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.services.training_service._process_pending_local_jobs"
            ) as mock_process,
            patch(
                "simpletuner.simpletuner_sdk.server.services.training_service._detect_local_hardware",
                return_value="local",
            ),
        ):
            result = _queue_training_job(
                job_id="approval-job",
                runtime_config={"--output_dir": "/tmp/out"},
                env_name="test-env",
                num_processes=2,
                preferred_gpus=None,
                any_gpu=False,
                org_id=None,
                user_id=None,
                requires_approval=True,
                approval_reason="quota review",
            )

        self.assertEqual(result.status, "blocked")
        mock_repo.add.assert_awaited_once()
        mock_process.assert_not_called()

    def test_process_pending_local_jobs_schedules_on_running_loop(self):
        import asyncio
        from unittest.mock import MagicMock, patch

        from simpletuner.simpletuner_sdk.server.services.training_service import _process_pending_local_jobs

        async def run_check():
            started = asyncio.Event()
            release = asyncio.Event()
            allocator = MagicMock()

            async def process_pending_jobs():
                started.set()
                await release.wait()
                return []

            allocator.process_pending_jobs = process_pending_jobs

            with patch(
                "simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.get_gpu_allocator",
                return_value=allocator,
            ):
                _process_pending_local_jobs()
                await asyncio.wait_for(started.wait(), timeout=1)
                release.set()
                await asyncio.sleep(0)

        asyncio.run(run_check())

    def test_process_pending_local_jobs_logs_async_failure(self):
        import asyncio
        from unittest.mock import MagicMock, patch

        from simpletuner.simpletuner_sdk.server.services import training_service

        async def run_check():
            allocator = MagicMock()

            async def process_pending_jobs():
                raise RuntimeError("queue failed")

            allocator.process_pending_jobs = process_pending_jobs

            with (
                patch(
                    "simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.get_gpu_allocator",
                    return_value=allocator,
                ),
                patch.object(training_service.logger, "exception") as mock_exception,
            ):
                training_service._process_pending_local_jobs()
                await asyncio.sleep(0)
                await asyncio.sleep(0)

            mock_exception.assert_called_once_with("Failed to process pending local jobs after enqueue")

        asyncio.run(run_check())


class TestReleaseJobGPUs(unittest.TestCase):
    """Tests for GPU release on job termination."""

    def test_release_job_gpus_placeholder(self):
        """Placeholder - GPU release is tested in LocalGPUAllocator tests."""
        # Release logic is tested in:
        # - test_local_gpu_allocator.py::TestLocalGPUAllocatorAllocation
        pass

    def test_cancellation_does_not_process_pending_jobs(self):
        """Cancellation should release GPUs but not process pending jobs.

        This prevents race conditions during bulk cancellation (cancel --all)
        where pending jobs would start before they could be cancelled.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        with patch(
            "simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.get_gpu_allocator"
        ) as mock_get_allocator:
            mock_allocator = MagicMock()
            mock_allocator.release = AsyncMock()
            mock_allocator.process_pending_jobs = AsyncMock(return_value=[])
            mock_get_allocator.return_value = mock_allocator

            from simpletuner.simpletuner_sdk.server.services.training_service import _release_job_gpus

            # Test cancellation - should NOT call process_pending_jobs
            _release_job_gpus("test-job", process_pending=False)

            mock_allocator.release.assert_called_once_with("test-job")
            mock_allocator.process_pending_jobs.assert_not_called()

    def test_completion_does_process_pending_jobs(self):
        """Completion should release GPUs AND process pending jobs."""
        from unittest.mock import AsyncMock, MagicMock, patch

        with patch(
            "simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.get_gpu_allocator"
        ) as mock_get_allocator:
            mock_allocator = MagicMock()
            mock_allocator.release = AsyncMock()
            mock_allocator.process_pending_jobs = AsyncMock(return_value=["job-2"])
            mock_get_allocator.return_value = mock_allocator

            from simpletuner.simpletuner_sdk.server.services.training_service import _release_job_gpus

            # Test completion - should call process_pending_jobs
            _release_job_gpus("test-job", process_pending=True)

            mock_allocator.release.assert_called_once_with("test-job")
            mock_allocator.process_pending_jobs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
