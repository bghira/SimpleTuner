import time

from simpletuner.simpletuner_sdk import process_keeper


def _long_running_task(config):
    deadline = time.time() + 5
    should_abort = getattr(config, "should_abort", lambda: False)

    while time.time() < deadline:
        try:
            if should_abort():
                return "aborted"
        except Exception:
            return "error"
        time.sleep(0.05)

    raise AssertionError("Subprocess did not observe abort in time")


def test_subprocess_abort_signal_propagates():
    job_id = "abort-test"

    try:
        process_keeper.submit_job(job_id, _long_running_task, {})

        status = process_keeper.get_process_status(job_id)
        assert status in {"running", "pending"}

        assert process_keeper.terminate_process(job_id)

        final_status = None
        for _ in range(40):
            final_status = process_keeper.get_process_status(job_id)
            if final_status in {"terminated", "failed", "completed"}:
                break
            time.sleep(0.1)

        assert final_status == "terminated"
    finally:
        process_keeper.process_registry.pop(job_id, None)
