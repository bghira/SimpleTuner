import multiprocessing
import sys
import unittest


def _exit_from_worker():
    import tests  # Register the test runner shutdown hook in this interpreter.

    try:
        sys.exit(0)
    except SystemExit:
        pass
    raise SystemExit(17)


class UnittestShutdownTests(unittest.TestCase):
    def test_worker_exit_code_survives_test_package_shutdown(self):
        worker = multiprocessing.get_context("spawn").Process(target=_exit_from_worker)
        worker.start()
        try:
            worker.join(30)
            self.assertEqual(worker.exitcode, 17)
        finally:
            if worker.is_alive():
                worker.kill()
                worker.join()
