"""Tests for QueueStore GPU allocation methods.

Tests cover:
- Schema v3 migration (allocated_gpus, job_type, num_processes columns)
- get_allocated_gpus aggregation
- get_running_local_jobs filtering
- get_pending_local_jobs ordering
- update_allocated_gpus persistence
- count_running_local_jobs count
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional

from simpletuner.simpletuner_sdk.server.services.cloud.queue.models import QueueEntry, QueuePriority, QueueStatus
from simpletuner.simpletuner_sdk.server.services.cloud.queue.queue_store import SCHEMA_VERSION, QueueStore


class TestQueueStoreGPUSchema(unittest.TestCase):
    """Tests for schema v3 migration adding GPU columns."""

    def setUp(self):
        """Set up test fixtures with temp database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "queue.db"

        # Reset singleton for test isolation
        QueueStore._instance = None

    def tearDown(self):
        """Clean up temp files."""
        if self.db_path.exists():
            self.db_path.unlink()
        QueueStore._instance = None

    def test_schema_version_is_4(self):
        """Verify current schema version is 4."""
        self.assertEqual(SCHEMA_VERSION, 4)

    def test_schema_has_gpu_columns(self):
        """Test schema includes GPU-related columns."""
        store = QueueStore(self.db_path)
        conn = store._get_connection()
        cursor = conn.cursor()

        # Check table info
        cursor.execute("PRAGMA table_info(queue)")
        columns = {row[1] for row in cursor.fetchall()}

        self.assertIn("allocated_gpus", columns)
        self.assertIn("job_type", columns)
        self.assertIn("num_processes", columns)

    def test_job_type_defaults_to_cloud(self):
        """Test job_type column defaults to 'cloud'."""
        store = QueueStore(self.db_path)
        conn = store._get_connection()
        cursor = conn.cursor()

        # Insert a row without specifying job_type
        cursor.execute(
            """
            INSERT INTO queue (job_id, queued_at, priority, status)
            VALUES ('test-job', '2024-01-01T00:00:00Z', 10, 'pending')
            """
        )
        conn.commit()

        cursor.execute("SELECT job_type FROM queue WHERE job_id = 'test-job'")
        row = cursor.fetchone()
        self.assertEqual(row[0], "cloud")

    def test_num_processes_defaults_to_1(self):
        """Test num_processes column defaults to 1."""
        store = QueueStore(self.db_path)
        conn = store._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO queue (job_id, queued_at, priority, status)
            VALUES ('test-job', '2024-01-01T00:00:00Z', 10, 'pending')
            """
        )
        conn.commit()

        cursor.execute("SELECT num_processes FROM queue WHERE job_id = 'test-job'")
        row = cursor.fetchone()
        self.assertEqual(row[0], 1)


class TestQueueStoreAddToQueueGPU(unittest.TestCase):
    """Tests for add_to_queue with GPU fields."""

    def setUp(self):
        """Set up test fixtures with temp database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "queue.db"
        QueueStore._instance = None
        self.store = QueueStore(self.db_path)

    def tearDown(self):
        """Clean up temp files."""
        if self.db_path.exists():
            self.db_path.unlink()
        QueueStore._instance = None

    def test_add_local_job_with_gpus(self):
        """Test adding a local job with GPU allocation info."""
        entry = asyncio.run(
            self.store.add_to_queue(
                job_id="local-job-1",
                user_id=1,
                provider="local",
                config_name="test-config",
                allocated_gpus=[0, 1],
                job_type="local",
                num_processes=2,
            )
        )

        self.assertEqual(entry.job_id, "local-job-1")
        self.assertEqual(entry.allocated_gpus, [0, 1])
        self.assertEqual(entry.job_type, "local")
        self.assertEqual(entry.num_processes, 2)

    def test_add_cloud_job_no_gpus(self):
        """Test adding a cloud job has no GPU allocation."""
        entry = asyncio.run(
            self.store.add_to_queue(
                job_id="cloud-job-1",
                user_id=1,
                provider="replicate",
                config_name="test-config",
            )
        )

        self.assertEqual(entry.job_id, "cloud-job-1")
        self.assertIsNone(entry.allocated_gpus)
        self.assertEqual(entry.job_type, "cloud")
        self.assertEqual(entry.num_processes, 1)


class TestQueueStoreGetAllocatedGPUs(unittest.TestCase):
    """Tests for get_allocated_gpus method."""

    def setUp(self):
        """Set up test fixtures with temp database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "queue.db"
        QueueStore._instance = None
        self.store = QueueStore(self.db_path)

    def tearDown(self):
        """Clean up temp files."""
        if self.db_path.exists():
            self.db_path.unlink()
        QueueStore._instance = None

    def _add_job(
        self,
        job_id: str,
        status: str,
        job_type: str = "local",
        allocated_gpus: Optional[List[int]] = None,
    ) -> None:
        """Add a job directly to database."""
        conn = self.store._get_connection()
        cursor = conn.cursor()
        gpus_json = json.dumps(allocated_gpus) if allocated_gpus else None
        cursor.execute(
            """
            INSERT INTO queue (job_id, queued_at, priority, status, job_type, allocated_gpus)
            VALUES (?, '2024-01-01T00:00:00Z', 10, ?, ?, ?)
            """,
            (job_id, status, job_type, gpus_json),
        )
        conn.commit()

    def test_get_allocated_gpus_empty(self):
        """Test returns empty set when no running local jobs."""
        result = asyncio.run(self.store.get_allocated_gpus())
        self.assertEqual(result, set())

    def test_get_allocated_gpus_running_local_job(self):
        """Test returns GPUs from running local jobs."""
        self._add_job("job-1", "running", "local", [0, 1])

        result = asyncio.run(self.store.get_allocated_gpus())
        self.assertEqual(result, {0, 1})

    def test_get_allocated_gpus_multiple_jobs(self):
        """Test aggregates GPUs from multiple running jobs."""
        self._add_job("job-1", "running", "local", [0, 1])
        self._add_job("job-2", "running", "local", [2, 3])

        result = asyncio.run(self.store.get_allocated_gpus())
        self.assertEqual(result, {0, 1, 2, 3})

    def test_get_allocated_gpus_ignores_pending(self):
        """Test ignores GPUs from pending jobs."""
        self._add_job("job-1", "running", "local", [0])
        self._add_job("job-2", "pending", "local", [1])

        result = asyncio.run(self.store.get_allocated_gpus())
        self.assertEqual(result, {0})

    def test_get_allocated_gpus_ignores_cloud(self):
        """Test ignores cloud jobs."""
        self._add_job("job-1", "running", "local", [0])
        self._add_job("job-2", "running", "cloud", [1])

        result = asyncio.run(self.store.get_allocated_gpus())
        self.assertEqual(result, {0})

    def test_get_allocated_gpus_ignores_null(self):
        """Test handles jobs with null allocated_gpus."""
        self._add_job("job-1", "running", "local", [0])
        self._add_job("job-2", "running", "local", None)

        result = asyncio.run(self.store.get_allocated_gpus())
        self.assertEqual(result, {0})


class TestQueueStoreGetRunningLocalJobs(unittest.TestCase):
    """Tests for get_running_local_jobs method."""

    def setUp(self):
        """Set up test fixtures with temp database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "queue.db"
        QueueStore._instance = None
        self.store = QueueStore(self.db_path)

    def tearDown(self):
        """Clean up temp files."""
        if self.db_path.exists():
            self.db_path.unlink()
        QueueStore._instance = None

    def _add_job(
        self,
        job_id: str,
        status: str,
        job_type: str = "local",
        started_at: Optional[str] = None,
    ) -> None:
        """Add a job directly to database."""
        conn = self.store._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO queue (job_id, queued_at, started_at, priority, status, job_type)
            VALUES (?, '2024-01-01T00:00:00Z', ?, 10, ?, ?)
            """,
            (job_id, started_at, status, job_type),
        )
        conn.commit()

    def test_get_running_local_jobs_empty(self):
        """Test returns empty list when no running local jobs."""
        result = asyncio.run(self.store.get_running_local_jobs())
        self.assertEqual(result, [])

    def test_get_running_local_jobs_returns_local(self):
        """Test returns only local running jobs."""
        self._add_job("job-1", "running", "local", "2024-01-01T10:00:00Z")
        self._add_job("job-2", "running", "cloud", "2024-01-01T10:01:00Z")

        result = asyncio.run(self.store.get_running_local_jobs())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].job_id, "job-1")

    def test_get_running_local_jobs_ordered_by_started_at(self):
        """Test results are ordered by started_at."""
        self._add_job("job-1", "running", "local", "2024-01-01T10:01:00Z")
        self._add_job("job-2", "running", "local", "2024-01-01T10:00:00Z")

        result = asyncio.run(self.store.get_running_local_jobs())
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].job_id, "job-2")  # Earlier started_at first
        self.assertEqual(result[1].job_id, "job-1")

    def test_get_running_local_jobs_ignores_pending(self):
        """Test ignores pending jobs."""
        self._add_job("job-1", "running", "local", "2024-01-01T10:00:00Z")
        self._add_job("job-2", "pending", "local", None)

        result = asyncio.run(self.store.get_running_local_jobs())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].job_id, "job-1")


class TestQueueStoreGetPendingLocalJobs(unittest.TestCase):
    """Tests for get_pending_local_jobs method."""

    def setUp(self):
        """Set up test fixtures with temp database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "queue.db"
        QueueStore._instance = None
        self.store = QueueStore(self.db_path)

    def tearDown(self):
        """Clean up temp files."""
        if self.db_path.exists():
            self.db_path.unlink()
        QueueStore._instance = None

    def _add_job(
        self,
        job_id: str,
        status: str,
        job_type: str = "local",
        priority: int = 10,
        queued_at: str = "2024-01-01T00:00:00Z",
    ) -> None:
        """Add a job directly to database."""
        conn = self.store._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO queue (job_id, queued_at, priority, status, job_type)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, queued_at, priority, status, job_type),
        )
        conn.commit()

    def test_get_pending_local_jobs_empty(self):
        """Test returns empty list when no pending local jobs."""
        result = asyncio.run(self.store.get_pending_local_jobs())
        self.assertEqual(result, [])

    def test_get_pending_local_jobs_includes_pending_and_ready(self):
        """Test includes both pending and ready status."""
        self._add_job("job-1", "pending", "local")
        self._add_job("job-2", "ready", "local")

        result = asyncio.run(self.store.get_pending_local_jobs())
        self.assertEqual(len(result), 2)
        job_ids = {entry.job_id for entry in result}
        self.assertEqual(job_ids, {"job-1", "job-2"})

    def test_get_pending_local_jobs_ordered_by_priority(self):
        """Test results are ordered by priority descending."""
        self._add_job("job-low", "pending", "local", priority=0)
        self._add_job("job-high", "pending", "local", priority=20)
        self._add_job("job-normal", "pending", "local", priority=10)

        result = asyncio.run(self.store.get_pending_local_jobs())
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].job_id, "job-high")
        self.assertEqual(result[1].job_id, "job-normal")
        self.assertEqual(result[2].job_id, "job-low")

    def test_get_pending_local_jobs_ordered_by_queued_at_within_priority(self):
        """Test same-priority jobs are ordered by queued_at."""
        self._add_job("job-2", "pending", "local", 10, "2024-01-01T10:01:00Z")
        self._add_job("job-1", "pending", "local", 10, "2024-01-01T10:00:00Z")

        result = asyncio.run(self.store.get_pending_local_jobs())
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].job_id, "job-1")  # Earlier queued_at first
        self.assertEqual(result[1].job_id, "job-2")

    def test_get_pending_local_jobs_ignores_cloud(self):
        """Test ignores cloud jobs."""
        self._add_job("job-1", "pending", "local")
        self._add_job("job-2", "pending", "cloud")

        result = asyncio.run(self.store.get_pending_local_jobs())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].job_id, "job-1")


class TestQueueStoreUpdateAllocatedGPUs(unittest.TestCase):
    """Tests for update_allocated_gpus method."""

    def setUp(self):
        """Set up test fixtures with temp database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "queue.db"
        QueueStore._instance = None
        self.store = QueueStore(self.db_path)

    def tearDown(self):
        """Clean up temp files."""
        if self.db_path.exists():
            self.db_path.unlink()
        QueueStore._instance = None

    def _add_job(self, job_id: str) -> None:
        """Add a job directly to database."""
        conn = self.store._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO queue (job_id, queued_at, priority, status, job_type)
            VALUES (?, '2024-01-01T00:00:00Z', 10, 'pending', 'local')
            """,
            (job_id,),
        )
        conn.commit()

    def _get_allocated_gpus(self, job_id: str) -> Optional[List[int]]:
        """Get allocated GPUs for a job."""
        conn = self.store._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT allocated_gpus FROM queue WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return None

    def test_update_allocated_gpus_set(self):
        """Test setting allocated GPUs."""
        self._add_job("job-1")

        success = asyncio.run(self.store.update_allocated_gpus("job-1", [0, 1, 2]))

        self.assertTrue(success)
        gpus = self._get_allocated_gpus("job-1")
        self.assertEqual(gpus, [0, 1, 2])

    def test_update_allocated_gpus_clear(self):
        """Test clearing allocated GPUs."""
        self._add_job("job-1")
        asyncio.run(self.store.update_allocated_gpus("job-1", [0, 1]))

        success = asyncio.run(self.store.update_allocated_gpus("job-1", None))

        self.assertTrue(success)
        gpus = self._get_allocated_gpus("job-1")
        self.assertIsNone(gpus)

    def test_update_allocated_gpus_nonexistent(self):
        """Test updating nonexistent job returns False."""
        success = asyncio.run(self.store.update_allocated_gpus("nonexistent", [0, 1]))

        self.assertFalse(success)


class TestQueueStoreCountRunningLocalJobs(unittest.TestCase):
    """Tests for count_running_local_jobs method."""

    def setUp(self):
        """Set up test fixtures with temp database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "queue.db"
        QueueStore._instance = None
        self.store = QueueStore(self.db_path)

    def tearDown(self):
        """Clean up temp files."""
        if self.db_path.exists():
            self.db_path.unlink()
        QueueStore._instance = None

    def _add_job(self, job_id: str, status: str, job_type: str = "local") -> None:
        """Add a job directly to database."""
        conn = self.store._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO queue (job_id, queued_at, priority, status, job_type)
            VALUES (?, '2024-01-01T00:00:00Z', 10, ?, ?)
            """,
            (job_id, status, job_type),
        )
        conn.commit()

    def test_count_running_local_jobs_zero(self):
        """Test returns 0 when no running local jobs."""
        count = asyncio.run(self.store.count_running_local_jobs())
        self.assertEqual(count, 0)

    def test_count_running_local_jobs_counts_running(self):
        """Test counts running local jobs."""
        self._add_job("job-1", "running", "local")
        self._add_job("job-2", "running", "local")
        self._add_job("job-3", "pending", "local")

        count = asyncio.run(self.store.count_running_local_jobs())
        self.assertEqual(count, 2)

    def test_count_running_local_jobs_ignores_cloud(self):
        """Test ignores cloud jobs."""
        self._add_job("job-1", "running", "local")
        self._add_job("job-2", "running", "cloud")

        count = asyncio.run(self.store.count_running_local_jobs())
        self.assertEqual(count, 1)


class TestQueueEntryGPUFields(unittest.TestCase):
    """Tests for QueueEntry GPU fields in to_dict/from_dict."""

    def test_queue_entry_to_dict_includes_gpu_fields(self):
        """Test to_dict includes GPU fields."""
        entry = QueueEntry(
            id=1,
            job_id="job-1",
            user_id=1,
            allocated_gpus=[0, 1],
            job_type="local",
            num_processes=2,
        )

        d = entry.to_dict()

        self.assertEqual(d["allocated_gpus"], [0, 1])
        self.assertEqual(d["job_type"], "local")
        self.assertEqual(d["num_processes"], 2)

    def test_queue_entry_from_dict_parses_gpu_fields(self):
        """Test from_dict parses GPU fields."""
        data = {
            "job_id": "job-1",
            "allocated_gpus": [0, 1, 2],
            "job_type": "local",
            "num_processes": 3,
        }

        entry = QueueEntry.from_dict(data)

        self.assertEqual(entry.allocated_gpus, [0, 1, 2])
        self.assertEqual(entry.job_type, "local")
        self.assertEqual(entry.num_processes, 3)

    def test_queue_entry_from_dict_defaults(self):
        """Test from_dict uses defaults for GPU fields."""
        data = {
            "job_id": "job-1",
        }

        entry = QueueEntry.from_dict(data)

        self.assertIsNone(entry.allocated_gpus)
        self.assertEqual(entry.job_type, "cloud")
        self.assertEqual(entry.num_processes, 1)


if __name__ == "__main__":
    unittest.main()
