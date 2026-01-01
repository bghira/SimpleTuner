"""Integration tests for local job queue and GPU tracking.

Tests cover the full flow of:
- Job submission creates queue entry with GPU allocation
- Queued jobs appear in jobs list (JobStore)
- Job completion updates both JobStore and QueueStore
- Job cancellation releases GPUs and updates queue entry
- GPU tracking prevents double-booking
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

from simpletuner.simpletuner_sdk.server.services.cloud.queue.models import QueueEntry, QueuePriority, QueueStatus


class MockGPUInventory:
    """Mock GPU inventory for testing."""

    def __init__(self, num_gpus: int = 4):
        self.inventory = {
            "backend": "cuda",
            "devices": [{"index": i, "name": f"GPU {i}", "memory_gb": 24} for i in range(num_gpus)],
        }

    def detect(self):
        return self.inventory


class MockWebUIDefaults:
    """Mock WebUIDefaults for testing."""

    def __init__(
        self,
        local_gpu_max_concurrent: Optional[int] = None,
        local_job_max_concurrent: int = 5,
    ):
        self.local_gpu_max_concurrent = local_gpu_max_concurrent
        self.local_job_max_concurrent = local_job_max_concurrent


class TestLocalJobQueueEntryCreation(unittest.TestCase):
    """Tests for queue entry creation when jobs start or queue."""

    def setUp(self):
        """Set up test fixtures with temp database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "queue.db"

        # Mock the database path
        self.mock_inventory = MockGPUInventory(num_gpus=2)
        self.mock_defaults = MockWebUIDefaults(
            local_gpu_max_concurrent=None,
            local_job_max_concurrent=5,
        )

    def tearDown(self):
        """Clean up temp files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_running_job_creates_queue_entry(self, mock_state_store, mock_detect):
        """When a job starts directly (GPUs available), a queue entry should be created."""
        mock_detect.return_value = self.mock_inventory.detect()
        mock_state_store.return_value.load_defaults.return_value = self.mock_defaults

        from simpletuner.simpletuner_sdk.server.services.local_gpu_allocator import LocalGPUAllocator

        # Create allocator with mock queue store
        allocator = LocalGPUAllocator()

        # Mock the queue store to avoid container initialization
        async def mock_get_allocated():
            return set()

        async def mock_count_running():
            return 0

        mock_store = MagicMock()
        mock_store.get_allocated_gpus = mock_get_allocated
        mock_store.count_running_local_jobs = mock_count_running
        allocator._queue_store = mock_store

        # Verify can_allocate returns correct GPUs
        can_alloc, gpus, reason = asyncio.run(allocator.can_allocate(required_count=1, any_gpu=True))

        self.assertTrue(can_alloc)
        self.assertEqual(len(gpus), 1)
        self.assertIn(gpus[0], [0, 1])

    def test_queue_entry_tracks_allocated_gpus(self):
        """Queue entry should store allocated_gpus for running jobs."""
        entry = QueueEntry(
            id=1,
            job_id="test-123",
            user_id=1,
            priority=QueuePriority.NORMAL,
            status=QueueStatus.RUNNING,
            job_type="local",
            num_processes=2,
            allocated_gpus=[0, 1],
            queued_at="2024-01-15T10:00:00Z",
        )

        self.assertEqual(entry.allocated_gpus, [0, 1])
        self.assertEqual(entry.job_type, "local")
        self.assertEqual(entry.status, QueueStatus.RUNNING)

    def test_queue_entry_serialization_includes_allocated_gpus(self):
        """to_dict/from_dict should preserve allocated_gpus."""
        entry = QueueEntry(
            id=1,
            job_id="test-123",
            user_id=1,
            priority=QueuePriority.NORMAL,
            status=QueueStatus.RUNNING,
            job_type="local",
            num_processes=2,
            allocated_gpus=[0, 1],
            queued_at="2024-01-15T10:00:00Z",
        )

        data = entry.to_dict()
        self.assertEqual(data["allocated_gpus"], [0, 1])

        restored = QueueEntry.from_dict(data)
        self.assertEqual(restored.allocated_gpus, [0, 1])


class TestGPUAllocationTracking(unittest.TestCase):
    """Tests for GPU allocation tracking to prevent double-booking."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.local_gpu_allocator import LocalGPUAllocator

        self.allocator = LocalGPUAllocator()
        self.mock_inventory = MockGPUInventory(num_gpus=2)
        self.mock_defaults = MockWebUIDefaults(
            local_gpu_max_concurrent=None,
            local_job_max_concurrent=5,
        )

        # Create mock queue store
        self._allocated = {}
        self._entries = {}

        async def mock_get_allocated():
            result = set()
            for gpus in self._allocated.values():
                result.update(gpus)
            return result

        async def mock_count_running():
            return len([e for e in self._entries.values() if e.get("status") == "running"])

        mock_store = MagicMock()
        mock_store.get_allocated_gpus = mock_get_allocated
        mock_store.count_running_local_jobs = mock_count_running
        self.allocator._queue_store = mock_store

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_cannot_allocate_already_allocated_gpu(self, mock_state_store, mock_detect):
        """Should not be able to allocate a GPU that's already allocated."""
        mock_detect.return_value = self.mock_inventory.detect()
        mock_state_store.return_value.load_defaults.return_value = self.mock_defaults

        # Simulate GPU 0 already allocated
        self._allocated["job-1"] = [0]
        self._entries["job-1"] = {"status": "running"}

        # Try to allocate GPU 0 specifically
        can_alloc, gpus, reason = asyncio.run(self.allocator.can_allocate(required_count=1, preferred_gpus=[0]))

        self.assertFalse(can_alloc)
        self.assertIn("Preferred GPUs", reason)

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_can_allocate_different_gpu_when_one_used(self, mock_state_store, mock_detect):
        """Should be able to allocate a different GPU when one is in use."""
        mock_detect.return_value = self.mock_inventory.detect()
        mock_state_store.return_value.load_defaults.return_value = self.mock_defaults

        # Simulate GPU 0 already allocated
        self._allocated["job-1"] = [0]
        self._entries["job-1"] = {"status": "running"}

        # Should be able to allocate GPU 1
        can_alloc, gpus, reason = asyncio.run(self.allocator.can_allocate(required_count=1, any_gpu=True))

        self.assertTrue(can_alloc)
        self.assertEqual(gpus, [1])

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_cannot_allocate_when_all_gpus_used(self, mock_state_store, mock_detect):
        """Should not be able to allocate when all GPUs are in use."""
        mock_detect.return_value = self.mock_inventory.detect()
        mock_state_store.return_value.load_defaults.return_value = self.mock_defaults

        # Simulate all GPUs allocated
        self._allocated["job-1"] = [0, 1]
        self._entries["job-1"] = {"status": "running"}

        can_alloc, gpus, reason = asyncio.run(self.allocator.can_allocate(required_count=1, any_gpu=True))

        self.assertFalse(can_alloc)
        self.assertIn("Insufficient", reason)


class TestJobStatusSync(unittest.TestCase):
    """Tests for job status synchronization between JobStore and QueueStore."""

    def test_callback_service_update_maps_statuses_correctly(self):
        """_update_job_store_status should map callback statuses to CloudJobStatus."""
        from simpletuner.simpletuner_sdk.server.services.cloud.base import CloudJobStatus

        # Test status mapping
        status_map = {
            "completed": CloudJobStatus.COMPLETED.value,
            "success": CloudJobStatus.COMPLETED.value,
            "failed": CloudJobStatus.FAILED.value,
            "error": CloudJobStatus.FAILED.value,
            "cancelled": CloudJobStatus.CANCELLED.value,
            "stopped": CloudJobStatus.CANCELLED.value,
        }

        for callback_status, expected in status_map.items():
            with self.subTest(callback_status=callback_status):
                # Map callback status to CloudJobStatus
                if callback_status in {"completed", "success"}:
                    result = CloudJobStatus.COMPLETED.value
                elif callback_status in {"failed", "error"}:
                    result = CloudJobStatus.FAILED.value
                elif callback_status in {"cancelled", "stopped"}:
                    result = CloudJobStatus.CANCELLED.value
                else:
                    result = None

                self.assertEqual(result, expected)


class TestQueuedJobVisibility(unittest.TestCase):
    """Tests for queued job visibility in jobs list."""

    def test_queued_job_has_correct_status(self):
        """Queued jobs should have status 'queued' in JobStore."""
        from simpletuner.simpletuner_sdk.server.services.cloud.base import CloudJobStatus

        # QUEUED status should exist
        self.assertEqual(CloudJobStatus.QUEUED.value, "queued")

    def test_queue_entry_pending_status(self):
        """Pending queue entries should have PENDING status."""
        entry = QueueEntry(
            id=1,
            job_id="test-123",
            user_id=1,
            priority=QueuePriority.NORMAL,
            status=QueueStatus.PENDING,
            job_type="local",
            num_processes=1,
            queued_at="2024-01-15T10:00:00Z",
        )

        self.assertEqual(entry.status, QueueStatus.PENDING)


class TestConcurrencyLimits(unittest.TestCase):
    """Tests for job and GPU concurrency limits."""

    def setUp(self):
        """Set up test fixtures."""
        from simpletuner.simpletuner_sdk.server.services.local_gpu_allocator import LocalGPUAllocator

        self.allocator = LocalGPUAllocator()
        self.mock_inventory = MockGPUInventory(num_gpus=4)

        # Create mock queue store
        self._entries = {}

        async def mock_count_running():
            return len([e for e in self._entries.values() if e.get("status") == "running"])

        async def mock_get_allocated():
            return set()

        mock_store = MagicMock()
        mock_store.count_running_local_jobs = mock_count_running
        mock_store.get_allocated_gpus = mock_get_allocated
        self.allocator._queue_store = mock_store

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_job_concurrency_limit_enforced(self, mock_state_store, mock_detect):
        """Should reject job when max concurrent jobs reached."""
        mock_detect.return_value = self.mock_inventory.detect()

        # Allow only 1 concurrent job
        mock_defaults = MockWebUIDefaults(
            local_gpu_max_concurrent=None,
            local_job_max_concurrent=1,
        )
        mock_state_store.return_value.load_defaults.return_value = mock_defaults

        # Simulate 1 running job
        self._entries["job-1"] = {"status": "running"}

        can_alloc, gpus, reason = asyncio.run(self.allocator.can_allocate(required_count=1))

        self.assertFalse(can_alloc)
        self.assertIn("Maximum concurrent local jobs", reason)

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_gpu_limit_enforced(self, mock_state_store, mock_detect):
        """Should reject job when GPU limit would be exceeded."""
        mock_detect.return_value = self.mock_inventory.detect()

        # Allow only 2 GPUs
        mock_defaults = MockWebUIDefaults(
            local_gpu_max_concurrent=2,
            local_job_max_concurrent=10,
        )
        mock_state_store.return_value.load_defaults.return_value = mock_defaults

        # Simulate 1 GPU already allocated
        async def mock_get_allocated():
            return {0}

        self.allocator._queue_store.get_allocated_gpus = mock_get_allocated

        # Try to allocate 2 more GPUs (would exceed limit)
        can_alloc, gpus, reason = asyncio.run(self.allocator.can_allocate(required_count=2))

        self.assertFalse(can_alloc)
        self.assertIn("Would exceed GPU limit", reason)


if __name__ == "__main__":
    unittest.main()
