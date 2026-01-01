"""Tests for LocalGPUAllocator GPU allocation tracking.

Tests cover:
- GPU availability detection
- GPU allocation and release
- can_allocate logic with preferred GPUs
- --any-gpu override behavior
- Concurrency limit enforcement
- Processing pending jobs
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

from simpletuner.simpletuner_sdk.server.services.cloud.queue.models import QueueEntry, QueuePriority, QueueStatus
from simpletuner.simpletuner_sdk.server.services.local_gpu_allocator import LocalGPUAllocator


class MockQueueStore:
    """Mock QueueStore for testing LocalGPUAllocator."""

    def __init__(self):
        self._entries: Dict[str, QueueEntry] = {}
        self._allocated_gpus: Dict[str, List[int]] = {}

    async def get_allocated_gpus(self) -> Set[int]:
        """Return all allocated GPU indices."""
        allocated: Set[int] = set()
        for gpus in self._allocated_gpus.values():
            allocated.update(gpus)
        return allocated

    async def get_running_local_jobs(self) -> List[QueueEntry]:
        """Return running local jobs."""
        return [e for e in self._entries.values() if e.job_type == "local" and e.status == QueueStatus.RUNNING]

    async def get_pending_local_jobs(self) -> List[QueueEntry]:
        """Return pending local jobs ordered by priority."""
        pending = [
            e
            for e in self._entries.values()
            if e.job_type == "local" and e.status in (QueueStatus.PENDING, QueueStatus.READY)
        ]
        return sorted(pending, key=lambda e: (-e.priority.value, e.queued_at))

    async def count_running_local_jobs(self) -> int:
        """Count running local jobs."""
        return len(await self.get_running_local_jobs())

    async def update_allocated_gpus(self, job_id: str, gpus: Optional[List[int]]) -> bool:
        """Update GPU allocation for a job."""
        if job_id not in self._entries:
            return False
        if gpus is None:
            self._allocated_gpus.pop(job_id, None)
        else:
            self._allocated_gpus[job_id] = gpus
        return True

    async def mark_running(self, queue_id: int) -> bool:
        """Mark a job as running."""
        for entry in self._entries.values():
            if entry.id == queue_id:
                entry.status = QueueStatus.RUNNING
                return True
        return False

    def add_entry(self, entry: QueueEntry, allocated_gpus: Optional[List[int]] = None) -> None:
        """Add a test entry."""
        # Set the allocated_gpus on the entry object as well
        if allocated_gpus is not None:
            entry.allocated_gpus = allocated_gpus
        self._entries[entry.job_id] = entry
        if allocated_gpus:
            self._allocated_gpus[entry.job_id] = allocated_gpus


class MockWebUIDefaults:
    """Mock WebUIDefaults for testing concurrency limits."""

    def __init__(
        self,
        local_gpu_max_concurrent: Optional[int] = None,
        local_job_max_concurrent: int = 1,
    ):
        self.local_gpu_max_concurrent = local_gpu_max_concurrent
        self.local_job_max_concurrent = local_job_max_concurrent


class TestLocalGPUAllocatorBasic(unittest.TestCase):
    """Basic tests for LocalGPUAllocator."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_queue_store = MockQueueStore()
        self.allocator = LocalGPUAllocator()
        self.allocator._queue_store = self.mock_queue_store

    def _create_entry(
        self,
        job_id: str,
        status: QueueStatus = QueueStatus.RUNNING,
        num_processes: int = 1,
        **kwargs,
    ) -> QueueEntry:
        """Create a test queue entry."""
        defaults = {
            "id": hash(job_id) % 10000,
            "job_id": job_id,
            "user_id": 1,
            "priority": QueuePriority.NORMAL,
            "status": status,
            "job_type": "local",
            "num_processes": num_processes,
            "queued_at": "2024-01-15T10:00:00Z",
        }
        defaults.update(kwargs)
        return QueueEntry(**defaults)

    def test_get_allocated_gpus_empty(self):
        """Test getting allocated GPUs when none are allocated."""
        result = asyncio.run(self.allocator.get_allocated_gpus())
        self.assertEqual(result, set())

    def test_get_allocated_gpus_with_running_jobs(self):
        """Test getting allocated GPUs from running jobs."""
        entry = self._create_entry("job-1")
        self.mock_queue_store.add_entry(entry, allocated_gpus=[0, 1])

        result = asyncio.run(self.allocator.get_allocated_gpus())
        self.assertEqual(result, {0, 1})

    def test_get_allocated_gpus_multiple_jobs(self):
        """Test aggregating GPUs from multiple running jobs."""
        entry1 = self._create_entry("job-1")
        entry2 = self._create_entry("job-2")
        self.mock_queue_store.add_entry(entry1, allocated_gpus=[0, 1])
        self.mock_queue_store.add_entry(entry2, allocated_gpus=[2, 3])

        result = asyncio.run(self.allocator.get_allocated_gpus())
        self.assertEqual(result, {0, 1, 2, 3})


class TestLocalGPUAllocatorAvailability(unittest.TestCase):
    """Tests for GPU availability checking."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_queue_store = MockQueueStore()
        self.allocator = LocalGPUAllocator()
        self.allocator._queue_store = self.mock_queue_store

        # Mock GPU inventory - 4 GPUs
        self.mock_inventory = {
            "backend": "cuda",
            "devices": [
                {"index": 0, "name": "GPU 0", "memory_gb": 24},
                {"index": 1, "name": "GPU 1", "memory_gb": 24},
                {"index": 2, "name": "GPU 2", "memory_gb": 24},
                {"index": 3, "name": "GPU 3", "memory_gb": 24},
            ],
        }

    def _create_entry(self, job_id: str, **kwargs) -> QueueEntry:
        """Create a test queue entry."""
        defaults = {
            "id": hash(job_id) % 10000,
            "job_id": job_id,
            "user_id": 1,
            "priority": QueuePriority.NORMAL,
            "status": QueueStatus.RUNNING,
            "job_type": "local",
            "num_processes": 1,
            "queued_at": "2024-01-15T10:00:00Z",
        }
        defaults.update(kwargs)
        return QueueEntry(**defaults)

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    def test_get_available_gpus_all_free(self, mock_detect):
        """Test all GPUs are available when none allocated."""
        mock_detect.return_value = self.mock_inventory

        result = asyncio.run(self.allocator.get_available_gpus())
        self.assertEqual(result, [0, 1, 2, 3])

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    def test_get_available_gpus_some_allocated(self, mock_detect):
        """Test available GPUs excludes allocated ones."""
        mock_detect.return_value = self.mock_inventory
        entry = self._create_entry("job-1")
        self.mock_queue_store.add_entry(entry, allocated_gpus=[0, 2])

        result = asyncio.run(self.allocator.get_available_gpus())
        self.assertEqual(result, [1, 3])

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    def test_get_available_gpus_all_allocated(self, mock_detect):
        """Test no GPUs available when all allocated."""
        mock_detect.return_value = self.mock_inventory
        entry = self._create_entry("job-1")
        self.mock_queue_store.add_entry(entry, allocated_gpus=[0, 1, 2, 3])

        result = asyncio.run(self.allocator.get_available_gpus())
        self.assertEqual(result, [])


class TestLocalGPUAllocatorCanAllocate(unittest.TestCase):
    """Tests for can_allocate logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_queue_store = MockQueueStore()
        self.allocator = LocalGPUAllocator()
        self.allocator._queue_store = self.mock_queue_store

        # Mock GPU inventory - 4 GPUs
        self.mock_inventory = {
            "backend": "cuda",
            "devices": [
                {"index": 0, "name": "GPU 0", "memory_gb": 24},
                {"index": 1, "name": "GPU 1", "memory_gb": 24},
                {"index": 2, "name": "GPU 2", "memory_gb": 24},
                {"index": 3, "name": "GPU 3", "memory_gb": 24},
            ],
        }

        # Default mock for WebUIDefaults
        self.mock_defaults = MockWebUIDefaults(
            local_gpu_max_concurrent=None,  # No GPU limit
            local_job_max_concurrent=5,  # Allow 5 concurrent jobs
        )

    def _create_entry(self, job_id: str, **kwargs) -> QueueEntry:
        """Create a test queue entry."""
        defaults = {
            "id": hash(job_id) % 10000,
            "job_id": job_id,
            "user_id": 1,
            "priority": QueuePriority.NORMAL,
            "status": QueueStatus.RUNNING,
            "job_type": "local",
            "num_processes": 1,
            "queued_at": "2024-01-15T10:00:00Z",
        }
        defaults.update(kwargs)
        return QueueEntry(**defaults)

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_can_allocate_simple_request(self, mock_state_store, mock_detect):
        """Test simple allocation when GPUs are available."""
        mock_detect.return_value = self.mock_inventory
        mock_state_store.return_value.load_defaults.return_value = self.mock_defaults

        can_alloc, gpus, reason = asyncio.run(self.allocator.can_allocate(required_count=2))

        self.assertTrue(can_alloc)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(reason, "")

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_can_allocate_insufficient_gpus(self, mock_state_store, mock_detect):
        """Test allocation fails when not enough GPUs available."""
        mock_detect.return_value = self.mock_inventory
        mock_state_store.return_value.load_defaults.return_value = self.mock_defaults

        # Allocate 3 GPUs to existing job
        entry = self._create_entry("job-1")
        self.mock_queue_store.add_entry(entry, allocated_gpus=[0, 1, 2])

        can_alloc, gpus, reason = asyncio.run(self.allocator.can_allocate(required_count=2))

        self.assertFalse(can_alloc)
        self.assertEqual(gpus, [])
        self.assertIn("Insufficient", reason)

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_can_allocate_preferred_gpus_available(self, mock_state_store, mock_detect):
        """Test allocation uses preferred GPUs when available."""
        mock_detect.return_value = self.mock_inventory
        mock_state_store.return_value.load_defaults.return_value = self.mock_defaults

        can_alloc, gpus, reason = asyncio.run(
            self.allocator.can_allocate(
                required_count=2,
                preferred_gpus=[2, 3],
            )
        )

        self.assertTrue(can_alloc)
        self.assertEqual(gpus, [2, 3])
        self.assertEqual(reason, "")

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_can_allocate_preferred_gpus_unavailable(self, mock_state_store, mock_detect):
        """Test allocation fails when preferred GPUs are not available."""
        mock_detect.return_value = self.mock_inventory
        mock_state_store.return_value.load_defaults.return_value = self.mock_defaults

        # Allocate GPU 2 to existing job
        entry = self._create_entry("job-1")
        self.mock_queue_store.add_entry(entry, allocated_gpus=[2])

        can_alloc, gpus, reason = asyncio.run(
            self.allocator.can_allocate(
                required_count=2,
                preferred_gpus=[2, 3],
            )
        )

        self.assertFalse(can_alloc)
        self.assertEqual(gpus, [])
        self.assertIn("Preferred GPUs", reason)

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_can_allocate_any_gpu_override(self, mock_state_store, mock_detect):
        """Test --any-gpu uses available GPUs instead of preferred."""
        mock_detect.return_value = self.mock_inventory
        mock_state_store.return_value.load_defaults.return_value = self.mock_defaults

        # Allocate preferred GPUs 2, 3 to existing job
        entry = self._create_entry("job-1")
        self.mock_queue_store.add_entry(entry, allocated_gpus=[2, 3])

        # With any_gpu=True, should use available GPUs 0, 1
        can_alloc, gpus, reason = asyncio.run(
            self.allocator.can_allocate(
                required_count=2,
                preferred_gpus=[2, 3],
                any_gpu=True,
            )
        )

        self.assertTrue(can_alloc)
        self.assertEqual(gpus, [0, 1])
        self.assertEqual(reason, "")

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_can_allocate_job_concurrency_limit(self, mock_state_store, mock_detect):
        """Test allocation fails when job concurrency limit reached."""
        mock_detect.return_value = self.mock_inventory
        mock_defaults = MockWebUIDefaults(
            local_gpu_max_concurrent=None,
            local_job_max_concurrent=1,  # Only 1 job allowed
        )
        mock_state_store.return_value.load_defaults.return_value = mock_defaults

        # Add one running job
        entry = self._create_entry("job-1")
        self.mock_queue_store.add_entry(entry, allocated_gpus=[0])

        can_alloc, gpus, reason = asyncio.run(self.allocator.can_allocate(required_count=1))

        self.assertFalse(can_alloc)
        self.assertEqual(gpus, [])
        self.assertIn("Maximum concurrent local jobs", reason)

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_can_allocate_gpu_concurrency_limit(self, mock_state_store, mock_detect):
        """Test allocation fails when GPU concurrency limit would be exceeded."""
        mock_detect.return_value = self.mock_inventory
        mock_defaults = MockWebUIDefaults(
            local_gpu_max_concurrent=2,  # Only 2 GPUs allowed
            local_job_max_concurrent=5,
        )
        mock_state_store.return_value.load_defaults.return_value = mock_defaults

        # Add one running job with 1 GPU
        entry = self._create_entry("job-1")
        self.mock_queue_store.add_entry(entry, allocated_gpus=[0])

        # Requesting 2 more would exceed limit (1 + 2 = 3 > 2)
        can_alloc, gpus, reason = asyncio.run(self.allocator.can_allocate(required_count=2))

        self.assertFalse(can_alloc)
        self.assertEqual(gpus, [])
        self.assertIn("Would exceed GPU limit", reason)


class TestLocalGPUAllocatorAllocation(unittest.TestCase):
    """Tests for GPU allocation and release."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_queue_store = MockQueueStore()
        self.allocator = LocalGPUAllocator()
        self.allocator._queue_store = self.mock_queue_store

    def _create_entry(self, job_id: str, **kwargs) -> QueueEntry:
        """Create a test queue entry."""
        defaults = {
            "id": hash(job_id) % 10000,
            "job_id": job_id,
            "user_id": 1,
            "priority": QueuePriority.NORMAL,
            "status": QueueStatus.PENDING,
            "job_type": "local",
            "num_processes": 1,
            "queued_at": "2024-01-15T10:00:00Z",
        }
        defaults.update(kwargs)
        return QueueEntry(**defaults)

    def test_allocate_gpus_success(self):
        """Test allocating GPUs to a job."""
        entry = self._create_entry("job-1")
        self.mock_queue_store.add_entry(entry)

        success = asyncio.run(self.allocator.allocate("job-1", [0, 1]))

        self.assertTrue(success)
        allocated = asyncio.run(self.mock_queue_store.get_allocated_gpus())
        self.assertEqual(allocated, {0, 1})

    def test_allocate_gpus_nonexistent_job(self):
        """Test allocating GPUs to nonexistent job fails."""
        success = asyncio.run(self.allocator.allocate("nonexistent", [0, 1]))

        self.assertFalse(success)

    def test_release_gpus_success(self):
        """Test releasing GPUs from a job."""
        entry = self._create_entry("job-1")
        self.mock_queue_store.add_entry(entry, allocated_gpus=[0, 1])

        success = asyncio.run(self.allocator.release("job-1"))

        self.assertTrue(success)
        allocated = asyncio.run(self.mock_queue_store.get_allocated_gpus())
        self.assertEqual(allocated, set())

    def test_release_gpus_nonexistent_job(self):
        """Test releasing GPUs from nonexistent job fails."""
        success = asyncio.run(self.allocator.release("nonexistent"))

        self.assertFalse(success)


class TestLocalGPUAllocatorProcessPending(unittest.TestCase):
    """Tests for processing pending jobs."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_queue_store = MockQueueStore()
        self.allocator = LocalGPUAllocator()
        self.allocator._queue_store = self.mock_queue_store

        # Mock GPU inventory - 4 GPUs
        self.mock_inventory = {
            "backend": "cuda",
            "devices": [
                {"index": 0, "name": "GPU 0", "memory_gb": 24},
                {"index": 1, "name": "GPU 1", "memory_gb": 24},
                {"index": 2, "name": "GPU 2", "memory_gb": 24},
                {"index": 3, "name": "GPU 3", "memory_gb": 24},
            ],
        }

        self.mock_defaults = MockWebUIDefaults(
            local_gpu_max_concurrent=None,
            local_job_max_concurrent=5,
        )

    def _create_entry(
        self,
        job_id: str,
        status: QueueStatus = QueueStatus.PENDING,
        num_processes: int = 1,
        queued_at: str = "2024-01-15T10:00:00Z",
        **kwargs,
    ) -> QueueEntry:
        """Create a test queue entry."""
        defaults = {
            "id": hash(job_id) % 10000,
            "job_id": job_id,
            "user_id": 1,
            "priority": QueuePriority.NORMAL,
            "status": status,
            "job_type": "local",
            "num_processes": num_processes,
            "queued_at": queued_at,
        }
        defaults.update(kwargs)
        return QueueEntry(**defaults)

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_process_pending_starts_job(self, mock_state_store, mock_detect):
        """Test processing pending jobs starts a job when GPUs available."""
        mock_detect.return_value = self.mock_inventory
        mock_state_store.return_value.load_defaults.return_value = self.mock_defaults

        # Add pending job
        entry = self._create_entry("job-1", num_processes=2)
        self.mock_queue_store.add_entry(entry)

        started = asyncio.run(self.allocator.process_pending_jobs())

        self.assertEqual(started, ["job-1"])

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_process_pending_no_pending(self, mock_state_store, mock_detect):
        """Test processing when no pending jobs."""
        mock_detect.return_value = self.mock_inventory
        mock_state_store.return_value.load_defaults.return_value = self.mock_defaults

        started = asyncio.run(self.allocator.process_pending_jobs())

        self.assertEqual(started, [])

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    @patch("simpletuner.simpletuner_sdk.server.services.webui_state.WebUIStateStore")
    def test_process_pending_insufficient_gpus(self, mock_state_store, mock_detect):
        """Test pending job not started when insufficient GPUs."""
        mock_detect.return_value = self.mock_inventory
        mock_state_store.return_value.load_defaults.return_value = self.mock_defaults

        # Use all GPUs
        running = self._create_entry("job-1", status=QueueStatus.RUNNING)
        self.mock_queue_store.add_entry(running, allocated_gpus=[0, 1, 2, 3])

        # Add pending job needing 2 GPUs
        pending = self._create_entry("job-2", num_processes=2)
        self.mock_queue_store.add_entry(pending)

        started = asyncio.run(self.allocator.process_pending_jobs())

        self.assertEqual(started, [])


class TestLocalGPUAllocatorGPUStatus(unittest.TestCase):
    """Tests for GPU status reporting."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_queue_store = MockQueueStore()
        self.allocator = LocalGPUAllocator()
        self.allocator._queue_store = self.mock_queue_store

        self.mock_inventory = {
            "backend": "cuda",
            "devices": [
                {"index": 0, "name": "GPU 0", "memory_gb": 24},
                {"index": 1, "name": "GPU 1", "memory_gb": 24},
            ],
        }

    def _create_entry(self, job_id: str, **kwargs) -> QueueEntry:
        """Create a test queue entry."""
        defaults = {
            "id": hash(job_id) % 10000,
            "job_id": job_id,
            "user_id": 1,
            "priority": QueuePriority.NORMAL,
            "status": QueueStatus.RUNNING,
            "job_type": "local",
            "num_processes": 1,
            "queued_at": "2024-01-15T10:00:00Z",
        }
        defaults.update(kwargs)
        return QueueEntry(**defaults)

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    def test_get_gpu_status_all_free(self, mock_detect):
        """Test GPU status when all GPUs are free."""
        mock_detect.return_value = self.mock_inventory

        status = asyncio.run(self.allocator.get_gpu_status())

        self.assertEqual(status["total_gpus"], 2)
        self.assertEqual(status["allocated_gpus"], [])
        self.assertEqual(status["available_gpus"], [0, 1])
        self.assertEqual(status["running_local_jobs"], 0)
        self.assertEqual(status["backend"], "cuda")
        self.assertEqual(len(status["devices"]), 2)
        self.assertFalse(status["devices"][0]["allocated"])
        self.assertFalse(status["devices"][1]["allocated"])

    @patch("simpletuner.simpletuner_sdk.server.services.local_gpu_allocator.detect_gpu_inventory")
    def test_get_gpu_status_with_allocation(self, mock_detect):
        """Test GPU status with allocated GPUs."""
        mock_detect.return_value = self.mock_inventory

        entry = self._create_entry("job-1")
        self.mock_queue_store.add_entry(entry, allocated_gpus=[0])

        status = asyncio.run(self.allocator.get_gpu_status())

        self.assertEqual(status["total_gpus"], 2)
        self.assertEqual(status["allocated_gpus"], [0])
        self.assertEqual(status["available_gpus"], [1])
        self.assertEqual(status["running_local_jobs"], 1)
        self.assertTrue(status["devices"][0]["allocated"])
        self.assertEqual(status["devices"][0]["job_id"], "job-1")
        self.assertFalse(status["devices"][1]["allocated"])
        self.assertIsNone(status["devices"][1]["job_id"])


if __name__ == "__main__":
    unittest.main()
