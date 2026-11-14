import threading
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import patch

from simpletuner.simpletuner_sdk import thread_keeper


class TestThreadKeeper(unittest.TestCase):
    def setUp(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.release = threading.Event()
        self.addCleanup(self.executor.shutdown, wait=True, cancel_futures=True)
        self.addCleanup(self.release.set)
        registry_patch = patch.object(thread_keeper, "thread_registry", {})
        executor_patch = patch.object(thread_keeper, "executor", self.executor)
        registry_patch.start()
        executor_patch.start()
        self.addCleanup(registry_patch.stop)
        self.addCleanup(executor_patch.stop)

    def test_duplicate_pending_job_does_not_replace_or_run_original_twice(self):
        started = threading.Event()

        def block():
            started.set()
            self.release.wait(timeout=5)

        thread_keeper.submit_job("running", block)
        self.assertTrue(started.wait(timeout=5))
        calls = []
        thread_keeper.submit_job("queued", calls.append, "original")
        original = thread_keeper.thread_registry["queued"]
        with self.assertRaisesRegex(Exception, "already running or pending"):
            thread_keeper.submit_job("queued", calls.append, "replacement")
        self.assertIs(thread_keeper.thread_registry["queued"], original)
        self.release.set()
        original.result(timeout=5)
        self.assertEqual(calls, ["original"])

    def test_cancelled_future_can_be_inspected_and_reused(self):
        future = Future()
        future.cancel()
        thread_keeper.thread_registry["cancelled"] = future
        self.assertEqual(thread_keeper.get_thread_status("cancelled"), "Cancelled")
        self.assertEqual(thread_keeper.list_threads(), {"cancelled": "Cancelled"})
        thread_keeper.submit_job("cancelled", lambda: None)
        thread_keeper.thread_registry["cancelled"].result(timeout=5)
        self.assertEqual(thread_keeper.get_thread_status("cancelled"), "Completed")

    def test_statuses(self):
        pending, running, completed, failed = (Future() for _ in range(4))
        running.set_running_or_notify_cancel()
        completed.set_result(None)
        failed.set_exception(ValueError("bad config"))
        thread_keeper.thread_registry.update(pending=pending, running=running, completed=completed, failed=failed)
        self.assertEqual(
            thread_keeper.list_threads(),
            {
                "pending": "Pending",
                "running": "Running",
                "completed": "Completed",
                "failed": "Failed: bad config",
            },
        )
        self.assertEqual(thread_keeper.get_thread_status("missing"), "No such job.")
