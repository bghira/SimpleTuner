"""Tests for job lifecycle: cancellation, duration, logs.

Tests cover:
- Job cancellation sets completed_at to freeze duration
- Duration calculation respects completed_at
- Log file lookup with helpful error messages
- Log streaming endpoint
"""

import asyncio
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from simpletuner.simpletuner_sdk.server.services.cloud.async_job_store import AsyncJobStore
from simpletuner.simpletuner_sdk.server.services.cloud.base import CloudJobStatus, JobType, UnifiedJob


def _make_job(**kwargs):
    """Create a UnifiedJob with sensible defaults."""
    defaults = {
        "job_id": "test-job",
        "job_type": JobType.LOCAL,
        "provider": None,
        "status": CloudJobStatus.RUNNING.value,
        "config_name": "test-config",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
    }
    defaults.update(kwargs)
    return UnifiedJob(**defaults)


class TestJobDuration(unittest.TestCase):
    """Test duration_seconds property respects completed_at."""

    def test_duration_running_job_uses_now(self):
        """Running job without completed_at uses current time."""
        started = datetime.now(timezone.utc) - timedelta(minutes=5)
        job = _make_job(
            job_id="test-123",
            status=CloudJobStatus.RUNNING.value,
            started_at=started.isoformat(),
            completed_at=None,
        )

        duration = job.duration_seconds
        self.assertIsNotNone(duration)
        # Should be approximately 5 minutes (300 seconds)
        self.assertGreater(duration, 295)
        self.assertLess(duration, 310)

    def test_duration_completed_job_frozen(self):
        """Completed job with completed_at has fixed duration."""
        started = datetime.now(timezone.utc) - timedelta(minutes=10)
        completed = started + timedelta(minutes=5)

        job = _make_job(
            job_id="test-456",
            status=CloudJobStatus.COMPLETED.value,
            started_at=started.isoformat(),
            completed_at=completed.isoformat(),
        )

        duration = job.duration_seconds
        self.assertIsNotNone(duration)
        # Should be exactly 5 minutes (300 seconds)
        self.assertAlmostEqual(duration, 300, delta=1)

    def test_duration_cancelled_job_frozen(self):
        """Cancelled job with completed_at has fixed duration."""
        started = datetime.now(timezone.utc) - timedelta(minutes=10)
        cancelled = started + timedelta(minutes=3)

        job = _make_job(
            job_id="test-789",
            status=CloudJobStatus.CANCELLED.value,
            started_at=started.isoformat(),
            completed_at=cancelled.isoformat(),
        )

        duration = job.duration_seconds
        self.assertIsNotNone(duration)
        # Should be exactly 3 minutes (180 seconds)
        self.assertAlmostEqual(duration, 180, delta=1)

    def test_duration_not_started_returns_none(self):
        """Job without started_at returns None for duration."""
        job = _make_job(
            job_id="test-pending",
            status=CloudJobStatus.PENDING.value,
            started_at=None,
            completed_at=None,
        )

        self.assertIsNone(job.duration_seconds)


class TestJobCancellationSetsCompletedAt(unittest.IsolatedAsyncioTestCase):
    """Test that job cancellation properly sets completed_at."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()

        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        self.store = AsyncJobStore(config_dir=self.config_dir)
        await self.store._ensure_initialized()

    async def asyncTearDown(self):
        """Clean up test fixtures."""
        await self.store.close()
        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_cancel_job_sets_completed_at(self):
        """Cancelling a job should set completed_at."""
        started = datetime.now(timezone.utc) - timedelta(minutes=5)
        job = _make_job(
            job_id="cancel-test-1",
            status=CloudJobStatus.RUNNING.value,
            started_at=started.isoformat(),
            completed_at=None,
        )
        await self.store.add_job(job)

        # Simulate cancellation - should set both status and completed_at
        now = datetime.now(timezone.utc)
        await self.store.update_job(
            "cancel-test-1",
            {
                "status": CloudJobStatus.CANCELLED.value,
                "completed_at": now.isoformat(),
            },
        )

        retrieved = await self.store.get_job("cancel-test-1")
        self.assertEqual(retrieved.status, CloudJobStatus.CANCELLED.value)
        self.assertIsNotNone(retrieved.completed_at)

        # Duration should now be fixed
        duration = retrieved.duration_seconds
        self.assertIsNotNone(duration)
        # Should be approximately 5 minutes
        self.assertAlmostEqual(duration, 300, delta=5)

    async def test_cancel_job_freezes_duration(self):
        """After cancellation, duration should not change over time."""
        started = datetime.now(timezone.utc) - timedelta(minutes=5)
        cancelled = datetime.now(timezone.utc)

        job = _make_job(
            job_id="freeze-test-1",
            status=CloudJobStatus.CANCELLED.value,
            started_at=started.isoformat(),
            completed_at=cancelled.isoformat(),
        )
        await self.store.add_job(job)

        # Get duration twice with a small delay
        retrieved1 = await self.store.get_job("freeze-test-1")
        duration1 = retrieved1.duration_seconds

        await asyncio.sleep(0.1)

        retrieved2 = await self.store.get_job("freeze-test-1")
        duration2 = retrieved2.duration_seconds

        # Durations should be identical (frozen)
        self.assertEqual(duration1, duration2)


class TestJobLogLookup(unittest.TestCase):
    """Test job log file lookup with helpful error messages."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_not_found_no_runtime_dir(self):
        """Test error message when no runtime directory exists."""
        from simpletuner.simpletuner_sdk.server.services.cloud.job_logs import _fetch_local_logs

        job = _make_job(
            job_id="no-runtime-job",
            status=CloudJobStatus.CANCELLED.value,
            output_url=self.output_dir,
        )

        result = _fetch_local_logs(job)
        self.assertIn("no training output directory exists", result.lower())

    def test_log_not_found_with_runtime_dir(self):
        """Test error message when runtime dir exists but no matching job."""
        from simpletuner.simpletuner_sdk.server.services.cloud.job_logs import _fetch_local_logs

        # Create runtime dir with a different job's logs
        runtime_dir = os.path.join(self.output_dir, ".simpletuner_runtime")
        other_job_dir = os.path.join(runtime_dir, "trainer_other123_xyz")
        os.makedirs(other_job_dir)
        with open(os.path.join(other_job_dir, "stdout.log"), "w") as f:
            f.write("Some logs from another job\n")

        job = _make_job(
            job_id="missing-job",
            status=CloudJobStatus.CANCELLED.value,
            output_url=self.output_dir,
        )

        result = _fetch_local_logs(job)
        self.assertIn("missing-job", result)
        self.assertIn("cancelled before training started", result.lower())

    def test_log_found_successfully(self):
        """Test successful log retrieval."""
        from simpletuner.simpletuner_sdk.server.services.cloud.job_logs import _fetch_local_logs

        # Create runtime dir with matching job logs
        runtime_dir = os.path.join(self.output_dir, ".simpletuner_runtime")
        job_dir = os.path.join(runtime_dir, "trainer_myjob123_xyz")
        os.makedirs(job_dir)
        with open(os.path.join(job_dir, "stdout.log"), "w") as f:
            f.write("Line 1: Training started\n")
            f.write("Line 2: Epoch 1/10\n")
            f.write("Line 3: Training completed\n")

        job = _make_job(
            job_id="myjob123",
            status=CloudJobStatus.COMPLETED.value,
            output_url=self.output_dir,
        )

        result = _fetch_local_logs(job)
        self.assertIn("Training started", result)
        self.assertIn("Epoch 1/10", result)
        self.assertIn("Training completed", result)

    def test_no_output_url(self):
        """Test error message when no output URL recorded."""
        from simpletuner.simpletuner_sdk.server.services.cloud.job_logs import _fetch_local_logs

        job = _make_job(
            job_id="no-output-job",
            status=CloudJobStatus.FAILED.value,
            output_url=None,
        )

        result = _fetch_local_logs(job)
        self.assertIn("no output directory recorded", result.lower())


class TestLogStreaming(unittest.IsolatedAsyncioTestCase):
    """Test log streaming functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_stream_waits_for_log_file(self):
        """Test that streaming waits for log file to appear."""
        from simpletuner.simpletuner_sdk.server.services.cloud.job_logs import stream_local_logs

        job = _make_job(
            job_id="stream-wait-test",
            status=CloudJobStatus.RUNNING.value,
            output_url=self.output_dir,
        )

        # Start streaming (should wait for file)
        lines = []

        async def collect_lines():
            count = 0
            async for line in stream_local_logs(job, poll_interval=0.1):
                lines.append(line)
                count += 1
                if count >= 3:
                    break

        # Create log file after a short delay
        async def create_log_file():
            await asyncio.sleep(0.3)
            runtime_dir = os.path.join(self.output_dir, ".simpletuner_runtime")
            job_dir = os.path.join(runtime_dir, "trainer_stream-wait-test_abc")
            os.makedirs(job_dir)
            with open(os.path.join(job_dir, "stdout.log"), "w") as f:
                f.write("Line 1\n")
                f.write("Line 2\n")
                f.write("Line 3\n")

        # Run both concurrently with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(collect_lines(), create_log_file()),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            pass

        # Should have collected some lines
        self.assertGreater(len(lines), 0)

    async def test_stream_yields_new_lines(self):
        """Test that streaming yields lines as they appear."""
        from simpletuner.simpletuner_sdk.server.services.cloud.job_logs import stream_local_logs

        # Pre-create log file with initial content
        runtime_dir = os.path.join(self.output_dir, ".simpletuner_runtime")
        job_dir = os.path.join(runtime_dir, "trainer_stream-test_xyz")
        os.makedirs(job_dir)
        log_path = os.path.join(job_dir, "stdout.log")
        with open(log_path, "w") as f:
            f.write("Initial line\n")

        job = _make_job(
            job_id="stream-test",
            status=CloudJobStatus.RUNNING.value,
            output_url=self.output_dir,
        )

        lines = []

        async def collect_lines():
            count = 0
            async for line in stream_local_logs(job, poll_interval=0.1):
                lines.append(line)
                count += 1
                if count >= 3:
                    break

        async def append_lines():
            await asyncio.sleep(0.2)
            with open(log_path, "a") as f:
                f.write("Second line\n")
            await asyncio.sleep(0.2)
            with open(log_path, "a") as f:
                f.write("Third line\n")

        try:
            await asyncio.wait_for(
                asyncio.gather(collect_lines(), append_lines()),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            pass

        self.assertIn("Initial line", lines)
        self.assertIn("Second line", lines)

    async def test_stream_no_output_url(self):
        """Test streaming with no output URL."""
        from simpletuner.simpletuner_sdk.server.services.cloud.job_logs import stream_local_logs

        job = _make_job(
            job_id="no-output",
            status=CloudJobStatus.RUNNING.value,
            output_url=None,
        )

        lines = []
        async for line in stream_local_logs(job):
            lines.append(line)
            break

        self.assertEqual(len(lines), 1)
        self.assertIn("No output directory", lines[0])


class TestCancelJobRoute(unittest.IsolatedAsyncioTestCase):
    """Test that cancel_job route sets completed_at."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()

        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        self.store = AsyncJobStore(config_dir=self.config_dir)
        await self.store._ensure_initialized()

    async def asyncTearDown(self):
        """Clean up test fixtures."""
        await self.store.close()
        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_cancel_route_sets_completed_at(self):
        """Test that the cancel route properly sets completed_at."""
        from simpletuner.simpletuner_sdk.server.routes.cloud.jobs import cancel_job

        # Create a running job
        started = datetime.now(timezone.utc) - timedelta(minutes=5)
        job = _make_job(
            job_id="route-cancel-test",
            status=CloudJobStatus.RUNNING.value,
            started_at=started.isoformat(),
            completed_at=None,
        )
        await self.store.add_job(job)

        # Mock dependencies
        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.cloud.jobs.get_job_store",
                return_value=self.store,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.routes.cloud.jobs.emit_cloud_event",
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.routes.cloud.jobs.get_client_ip",
                return_value="127.0.0.1",
            ),
        ):
            result = await cancel_job("route-cancel-test", mock_request, user=None)

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], CloudJobStatus.CANCELLED.value)

        # Verify completed_at was set
        retrieved = await self.store.get_job("route-cancel-test")
        self.assertIsNotNone(retrieved.completed_at)

        # Duration should be frozen
        duration = retrieved.duration_seconds
        self.assertIsNotNone(duration)
        self.assertAlmostEqual(duration, 300, delta=5)  # ~5 minutes


class TestCancelJobCommand(unittest.IsolatedAsyncioTestCase):
    """Test that CancelJobCommand sets completed_at."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()

        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        self.store = AsyncJobStore(config_dir=self.config_dir)
        await self.store._ensure_initialized()

    async def asyncTearDown(self):
        """Clean up test fixtures."""
        await self.store.close()
        AsyncJobStore._instance = None
        from simpletuner.simpletuner_sdk.server.services.cloud.storage.base import BaseSQLiteStore

        BaseSQLiteStore._instances.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_cancel_command_sets_completed_at(self):
        """Test that CancelJobCommand sets completed_at."""
        from simpletuner.simpletuner_sdk.server.services.cloud.commands import CommandContext
        from simpletuner.simpletuner_sdk.server.services.cloud.commands.job_commands import CancelJobCommand

        # Create a running job
        started = datetime.now(timezone.utc) - timedelta(minutes=3)
        job = _make_job(
            job_id="cmd-cancel-test",
            status=CloudJobStatus.RUNNING.value,
            started_at=started.isoformat(),
            completed_at=None,
        )
        await self.store.add_job(job)

        # Create command context
        ctx = CommandContext(
            job_store=self.store,
            user_id=None,
            client_ip="127.0.0.1",
        )

        command = CancelJobCommand(job_id="cmd-cancel-test")
        result = await command.execute(ctx)

        self.assertTrue(result.success)

        # Verify completed_at was set
        retrieved = await self.store.get_job("cmd-cancel-test")
        self.assertIsNotNone(retrieved.completed_at)

        # Duration should be frozen at ~3 minutes
        duration = retrieved.duration_seconds
        self.assertIsNotNone(duration)
        self.assertAlmostEqual(duration, 180, delta=5)


if __name__ == "__main__":
    unittest.main()
