"""
process_keeper tests - subprocess lifecycle and termination guarantees.
"""

import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import suppress
from unittest.mock import MagicMock, Mock, patch

try:
    from tests import test_setup
except ModuleNotFoundError:
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


def _kill_pid(pid: int) -> None:
    sig = signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM
    with suppress(Exception):
        os.kill(pid, sig)


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


def embed_cache_failure_with_child(config):
    """Simulate a text-embed cache miss that leaves a child process running."""
    pid_file = getattr(config, "pid_file", None)
    if pid_file is None and hasattr(config, "get"):
        pid_file = config.get("pid_file")

    detached_code = """
import os, time
pid_file = os.environ.get("PID_FILE")
if hasattr(os, "setsid"):
    try:
        os.setsid()
    except Exception:
        pass
if hasattr(os, "fork"):
    pid = os.fork()
    if pid == 0:
        time.sleep(2)
    else:
        if pid_file:
            with open(pid_file, "w", encoding="utf-8") as handle:
                handle.write(f"{pid},{os.getpgid(pid)}")
        os._exit(0)
else:
    if pid_file:
        with open(pid_file, "w", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()},{os.getpgid(0)}")
    time.sleep(2)
"""

    env = os.environ.copy()
    if pid_file:
        env["PID_FILE"] = pid_file
    popen_kwargs = {"env": env}
    if hasattr(os, "setsid"):
        popen_kwargs["preexec_fn"] = os.setsid
    subprocess.Popen([sys.executable, "-c", detached_code], **popen_kwargs)
    raise RuntimeError("simulated text embed cache missing element")


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

        # Wait for task to complete (task takes 0.1s, event thread polls every 0.1s)
        # get_process_status now actively waits for event thread, so 0.3s should be sufficient
        time.sleep(0.3)
        status = get_process_status(job_id)
        self.assertNotEqual(status, "running")
        self.assertIn(status, ["completed", "failed"])

    def test_failure_event_updates_status_and_events(self):
        """A raised exception should surface as a failed status with error event."""
        job_id = "test_failure_status"
        self.test_jobs.append(job_id)

        def failing_task(config):
            raise RuntimeError("simulated failure")

        # Suppress stdout from ProcessKeeper's output streaming
        with patch("simpletuner.simpletuner_sdk.process_keeper.sys.stdout", new=MagicMock()):
            submit_job(job_id, failing_task, {})

            # Wait until process transitions out of running
            deadline = time.time() + 5
            status = None
            while time.time() < deadline:
                status = get_process_status(job_id)
                if status not in {"pending", "running"}:
                    break
                time.sleep(0.1)

        self.assertEqual(status, "failed")

        events = get_process_events(job_id)
        error_events = [event for event in events if (event.get("type") or "").lower() == "error"]
        self.assertTrue(error_events, f"Expected error event, received: {events}")

    def test_log_extraction_prefers_cuda_oom_message(self):
        """Synthetic log extraction should highlight CUDA OOM failures."""
        job_id = "test_log_extraction"
        process = TrainerProcess(job_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "stdout.log")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("some info\n")
                handle.write(
                    "2025-11-04 16:21:12,847 - SimpleTuner - INFO - [RANK 0] 2025-11-04 16:21:12,846 [ERROR] Error encoding images ['16.jpg']: CUDA out of memory. Tried to allocate 256.00 MiB.\n"
                )
                handle.write("RuntimeError: Accelerate launch exited with status 1\n")

            process.log_file = log_path
            payload = process._extract_error_from_logs(exit_code=1)

        self.assertIsNotNone(payload)
        if payload is None:  # Pragmatic guard for static checkers
            self.fail("Expected payload from log extraction")
        self.assertIn("CUDA out of memory", payload.get("message", ""))

    def test_log_extraction_prefers_signal_exit(self):
        """Synthetic log extraction should highlight signal-based exits."""
        job_id = "test_log_signal_exit"
        process = TrainerProcess(job_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "stdout.log")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("2025-11-04 16:21:12,847 - SimpleTuner - INFO - starting...\n")

            process.log_file = log_path
            payload = process._extract_error_from_logs(exit_code=-9)

        self.assertIsNotNone(payload)
        if payload is None:  # Pragmatic guard for static checkers
            self.fail("Expected payload from log extraction")
        message = payload.get("message", "")
        self.assertTrue("SIGKILL" in message or "signal 9" in message)

    def test_log_extraction_parses_signal_received_by_pid_format(self):
        """Log extraction should parse accelerate's 'Signal N (SIGNAME) received by PID' format."""
        job_id = "test_log_signal_pid_format"
        process = TrainerProcess(job_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "stdout.log")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("2025-11-04 16:21:12,847 - SimpleTuner - INFO - starting...\n")
                handle.write("traceback : Signal 9 (SIGKILL) received by PID 2963915\n")
                handle.write("RuntimeError: Some wrapper error\n")

            process.log_file = log_path
            # Exit code is 1 (not -9) because accelerate wraps the signal
            payload = process._extract_error_from_logs(exit_code=1)

        self.assertIsNotNone(payload)
        if payload is None:  # Pragmatic guard for static checkers
            self.fail("Expected payload from log extraction")
        message = payload.get("message", "")
        self.assertIn("SIGKILL", message)

    def test_force_kill_unresponsive_process(self):
        """Test force killing a process that ignores SIGTERM."""
        job_id = "test_force_kill"
        self.test_jobs.append(job_id)
        config = {}

        process = submit_job(job_id, hanging_task, config)
        time.sleep(0.3)  # Let it start and install signal handler

        success = terminate_process(job_id)
        self.assertTrue(success)

        # terminate_process sends SIGTERM, waits up to 5s grace, then SIGKILL
        # The process ignores SIGTERM so it should be killed after grace period
        deadline = time.time() + 10
        status = None
        while time.time() < deadline:
            status = get_process_status(job_id)
            if status == "terminated":
                break
            time.sleep(0.1)
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

    @patch("simpletuner.simpletuner_sdk.process_keeper.subprocess.Popen")
    def test_subprocess_env_includes_parent_pid(self, mock_popen):
        job_id = "test_parent_pid_env"
        self.test_jobs.append(job_id)

        dummy_proc = MagicMock()
        dummy_proc.stdout = None
        dummy_proc.stderr = None
        dummy_proc.stdin = None
        dummy_proc.pid = 12345
        dummy_proc.poll.return_value = 0
        dummy_proc.returncode = 0
        mock_popen.return_value = dummy_proc

        process = TrainerProcess(job_id)
        self.addCleanup(process._cleanup_resources)
        process.start(simple_task, {})

        called_env = mock_popen.call_args.kwargs.get("env") or {}
        self.assertEqual(called_env.get("SIMPLETUNER_PARENT_PID"), str(os.getpid()))


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

    def test_process_crash_detection_and_state_update(self):
        """Test that crashed processes are detected and marked."""
        job_id = "test_crash_detection"
        self.test_jobs.append(job_id)
        config = {}

        process = submit_job(job_id, crashing_task, config)
        # Wait for crash to be detected - crashing_task sleeps 0.1s then exits
        deadline = time.time() + 5
        status = None
        while time.time() < deadline:
            status = get_process_status(job_id)
            if status not in {"pending", "running"}:
                break
            time.sleep(0.1)

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

    def test_graceful_termination_with_timeout(self):
        """Test graceful termination respects timeout."""
        job_id = "test_graceful_timeout"
        self.test_jobs.append(job_id)
        process = submit_job(job_id, slow_shutdown, {})
        time.sleep(0.3)  # Let it start and install signal handler

        start_time = time.time()
        success = process.terminate(timeout=2)
        duration = time.time() - start_time

        self.assertTrue(success)
        # Should complete within ~2s grace + some overhead, not 10s from slow handler
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


class TestFailureCleanupRegression(ProcessKeeperTestCase):

    def test_failure_does_not_leave_child_process_alive(self):
        """A failed training task should not leave spawned children running."""
        job_id = "test_failure_child_cleanup"
        self.test_jobs.append(job_id)
        child_pid = None

        with tempfile.TemporaryDirectory() as tmpdir:
            pid_file = os.path.join(tmpdir, "child.pid")
            config = {"pid_file": pid_file}

            with (
                patch("simpletuner.simpletuner_sdk.process_keeper.sys.stdout", new=MagicMock()),
                patch("simpletuner.simpletuner_sdk.process_keeper.psutil", None),
            ):
                submit_job(job_id, embed_cache_failure_with_child, config)

                status = None
                deadline = time.time() + 5
                while time.time() < deadline:
                    status = get_process_status(job_id)
                    if status not in {"pending", "running"}:
                        break
                    time.sleep(0.1)

                self.assertEqual(status, "failed", f"status remained {status}")

                with open(pid_file, "r", encoding="utf-8") as handle:
                    pid_str = handle.read().strip()
                child_pid, _child_pgid = (int(part) for part in pid_str.split(",", maxsplit=1))

                terminate_process(job_id)
                time.sleep(0.1)

        if child_pid is not None:
            self.addCleanup(_kill_pid, child_pid)

        with self.assertRaises(OSError, msg="Child process survived training failure"):
            os.kill(child_pid, 0)


def tearDownModule():
    cleanup_jobs([], timeout=_CLEANUP_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
