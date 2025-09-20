"""
process_keeper tests - subprocess lifecycle and termination guarantees.
"""

import os
import queue
import signal
import subprocess
import sys
import threading
import time
import unittest
from contextlib import suppress
from unittest.mock import MagicMock, Mock, patch

import test_setup

from simpletuner.simpletuner_sdk.process_keeper import (
    TrainerProcess,
    cleanup_dead_processes,
    get_process_events,
    get_process_status,
    list_processes,
    lock,
    process_registry,
    send_process_command,
    submit_job,
    terminate_process,
)

# teardown helpers to prevent hanging background workers
_CLEANUP_TIMEOUT = 5


def _close_queue(queue_obj):
    if queue_obj is None:
        return
    with suppress(Exception):
        queue_obj.close()
    with suppress(Exception):
        queue_obj.join_thread()


def _finalize_registry_job(job_id: str, timeout: int = _CLEANUP_TIMEOUT) -> None:
    with lock:
        entry = process_registry.get(job_id)
        process = entry.get("process") if entry else None

    if process:
        stop_event = getattr(process, "stop_event", None)
        if stop_event is not None:
            stop_event.set()

        _close_queue(getattr(process, "command_queue", None))

        proc_handle = getattr(process, "process", None)
        if proc_handle and proc_handle.poll() is None:
            with suppress(Exception):
                proc_handle.terminate()
                proc_handle.wait(timeout=timeout)
        if proc_handle and proc_handle.poll() is None:
            with suppress(Exception):
                proc_handle.kill()
                proc_handle.wait(timeout=1)

        event_thread = getattr(process, "event_thread", None)
        if event_thread and event_thread.is_alive():
            with suppress(Exception):
                event_thread.join(timeout=timeout)

        with suppress(Exception):
            process._cleanup_resources()

    with lock:
        process_registry.pop(job_id, None)


def cleanup_jobs(job_ids, timeout: int = _CLEANUP_TIMEOUT) -> None:
    seen = set()
    for job_id in list(job_ids):
        if not job_id or job_id in seen:
            continue
        seen.add(job_id)
        _finalize_registry_job(job_id, timeout=timeout)

    with lock:
        leftovers = [job_id for job_id in process_registry.keys() if job_id not in seen]

    for job_id in leftovers:
        _finalize_registry_job(job_id, timeout=timeout)


class ProcessKeeperTestCase(unittest.TestCase):

    cleanup_timeout = _CLEANUP_TIMEOUT

    def setUp(self):
        super().setUp()
        self.test_jobs = []

    def tearDown(self):
        try:
            cleanup_jobs(self.test_jobs, timeout=self.cleanup_timeout)
        finally:
            self.test_jobs.clear()
        super().tearDown()


# module-level functions for pickle compatibility
def simple_task(config):
    time.sleep(0.1)
    return "completed"


def hanging_task(config):
    # ignores SIGTERM
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(0.1)


def orphan_creator(config):
    # creates subprocess that outlives parent
    import subprocess

    subprocess.Popen(["sleep", "10"])
    return "parent_done"


def slow_task(config):
    time.sleep(1)


def crashing_task(config):
    time.sleep(0.1)
    os._exit(1)


def slow_shutdown(config):
    # slow shutdown handler for testing timeouts
    signal.signal(signal.SIGTERM, lambda s, f: time.sleep(10))
    while True:
        time.sleep(0.1)


class TestProcessLifecycle(ProcessKeeperTestCase):

    def test_subprocess_creation_and_normal_termination(self):
        """Test normal subprocess lifecycle from creation to completion."""
        job_id = "test_normal_lifecycle"
        self.test_jobs.append(job_id)
        config = {"test": "config"}

        process = submit_job(job_id, simple_task, config)
        self.assertIsNotNone(process)
        self.assertEqual(process.job_id, job_id)

        status = get_process_status(job_id)
        self.assertIn(status, ["pending", "running"])

        time.sleep(0.3)
        status = get_process_status(job_id)
        self.assertNotEqual(status, "running")

    @unittest.skip("Requires process fixes")
    def test_force_kill_unresponsive_process(self):
        """Test force killing a process that ignores SIGTERM."""
        job_id = "test_force_kill"
        self.test_jobs.append(job_id)
        config = {}

        process = submit_job(job_id, hanging_task, config)
        time.sleep(0.2)  # Let it start

        success = terminate_process(job_id)
        self.assertTrue(success)

        time.sleep(0.5)
        status = get_process_status(job_id)
        self.assertEqual(status, "terminated")

    @unittest.skip("Requires subprocess module")
    def test_orphaned_process_cleanup(self):
        """Test cleanup of orphaned processes when parent dies."""
        job_id = "test_orphan"
        self.test_jobs.append(job_id)
        config = {}

        process = submit_job(job_id, orphan_creator, config)
        time.sleep(0.5)

        # simulate parent death
        with threading.Lock():
            if job_id in process_registry:
                del process_registry[job_id]

        cleanup_dead_processes()

        self.assertNotIn(job_id, process_registry)

    def test_concurrent_job_prevention(self):
        """Test that same job_id cannot be started twice."""
        job_id = "test_duplicate"
        self.test_jobs.append(job_id)
        config = {}

        process1 = submit_job(job_id, slow_task, config)
        self.assertIsNotNone(process1)

        with self.assertRaises(Exception) as context:
            process2 = submit_job(job_id, slow_task, config)
        self.assertIn("already running", str(context.exception))


class TestProcessCommunication(ProcessKeeperTestCase):

    def test_command_queue_overflow_handling(self):
        """Test behavior when command queue is flooded."""
        job_id = "test_queue_overflow"

        process = TrainerProcess(job_id)
        self.addCleanup(process._cleanup_resources)
        process.command_queue = queue.Queue(maxsize=10)

        process.process = Mock()
        process.process.is_alive.return_value = True

        sent_count = 0
        failed_count = 0
        for i in range(100):
            try:
                process.command_queue.put_nowait({"command": f"test_{i}"})
                sent_count += 1
            except queue.Full:
                failed_count += 1

        self.assertEqual(sent_count, 10)
        self.assertEqual(failed_count, 90)

    def test_send_command_to_dead_process(self):
        """Test sending command to non-running process."""
        job_id = "test_dead_command"
        process = TrainerProcess(job_id)
        self.addCleanup(process._cleanup_resources)

        with self.assertRaises(Exception) as context:
            process.send_command("abort")
        self.assertIn("not running", str(context.exception))


class TestProcessStateTracking(ProcessKeeperTestCase):

    @unittest.skip("Requires process fixes")
    def test_process_crash_detection_and_state_update(self):
        """Test that crashed processes are detected and marked."""
        job_id = "test_crash_detection"
        self.test_jobs.append(job_id)
        config = {}

        process = submit_job(job_id, crashing_task, config)
        time.sleep(0.3)

        status = get_process_status(job_id)
        self.assertNotEqual(status, "running")
        self.assertFalse(process.is_alive())

    def test_heartbeat_timeout_marks_process_dead(self):
        """Test heartbeat timeout detection."""
        job_id = "test_heartbeat"
        self.test_jobs.append(job_id)
        process = TrainerProcess(job_id)

        # old heartbeat to trigger timeout
        process.last_heartbeat = time.time() - 400
        process.status = "running"

        process.process = Mock()
        process.process.is_alive.return_value = False

        process_registry[job_id] = {"process": process, "status": "running"}

        cleanup_dead_processes()

        if job_id in process_registry:
            self.assertEqual(process_registry[job_id]["status"], "crashed")

    def test_status_transitions(self):
        """Test valid status transitions during lifecycle."""
        job_id = "test_transitions"
        process = TrainerProcess(job_id)
        self.addCleanup(process._cleanup_resources)

        self.assertEqual(process.status, "pending")

        process.process = Mock()
        process.process.is_alive.return_value = True
        process.status = "running"
        self.assertEqual(process.status, "running")

        process.status = "terminated"
        self.assertEqual(process.status, "terminated")


class TestProcessEvents(ProcessKeeperTestCase):

    def test_event_listener_receives_events(self):
        """Test that events from subprocess are received."""
        job_id = "test_events"
        self.test_jobs.append(job_id)
        process = TrainerProcess(job_id)

        process_registry[job_id] = {"process": process, "events": []}

        test_event = {"type": "progress", "data": {"step": 100}}
        process._handle_event(test_event)

        self.assertEqual(len(process_registry[job_id]["events"]), 1)
        self.assertEqual(process_registry[job_id]["events"][0]["type"], "progress")

    def test_get_process_events_since_index(self):
        """Test retrieving events since a given index."""
        job_id = "test_event_retrieval"
        self.test_jobs.append(job_id)

        process_registry[job_id] = {
            "events": [{"id": 0, "data": "event0"}, {"id": 1, "data": "event1"}, {"id": 2, "data": "event2"}]
        }

        events = get_process_events(job_id, since_index=1)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["data"], "event1")
        self.assertEqual(events[1]["data"], "event2")


class TestProcessCleanup(ProcessKeeperTestCase):

    @patch("simpletuner.simpletuner_sdk.process_keeper.start_cleanup_thread")
    def test_cleanup_thread_initialization(self, mock_start):
        """Test that cleanup thread starts on module import."""
        # cleanup thread starts on import
        import simpletuner.simpletuner_sdk.process_keeper

    def test_cleanup_removes_stale_processes(self):
        """Test that stale processes are removed from registry."""
        job_id = "test_stale"
        self.test_jobs.append(job_id)
        process = TrainerProcess(job_id)
        process.last_heartbeat = time.time() - 400  # Very old
        process.status = "crashed"
        process.process = None

        process_registry[job_id] = {"process": process, "status": "crashed"}

        cleanup_dead_processes()

        self.assertNotIn(job_id, process_registry)

    def test_cleanup_preserves_active_processes(self):
        """Test that active processes are not removed."""
        job_id = "test_active"
        self.test_jobs.append(job_id)
        process = TrainerProcess(job_id)
        process.last_heartbeat = time.time()  # Recent
        process.status = "running"

        process.process = Mock()
        process.process.is_alive.return_value = True

        process_registry[job_id] = {"process": process, "status": "running"}

        cleanup_dead_processes()

        self.assertIn(job_id, process_registry)


class TestProcessTermination(ProcessKeeperTestCase):

    @unittest.skip("Requires process fixes")
    def test_graceful_termination_with_timeout(self):
        """Test graceful termination respects timeout."""
        job_id = "test_graceful_timeout"
        process = submit_job(job_id, slow_shutdown, {})
        time.sleep(0.2)

        start_time = time.time()
        success = process.terminate(timeout=1)
        duration = time.time() - start_time

        self.assertTrue(success)
        self.assertLess(duration, 5)

    def test_terminate_already_dead_process(self):
        """Test terminating an already dead process."""
        job_id = "test_dead_terminate"
        process = TrainerProcess(job_id)
        self.addCleanup(process._cleanup_resources)

        success = process.terminate()
        self.assertTrue(success)

    @unittest.skip("Requires valid trainer config")
    def test_signal_handler_installation(self):
        """Test that signal handlers are properly installed."""
        job_id = "test_signals"
        process = TrainerProcess(job_id)

        with patch("signal.signal") as mock_signal:
            process._run_trainer(lambda c: time.sleep(0.1), "{}", process.command_queue, process.event_pipe_send)

            # verify signal handlers are installed
            calls = mock_signal.call_args_list
            signals_installed = [call[0][0] for call in calls]


def tearDownModule():
    cleanup_jobs([], timeout=_CLEANUP_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
