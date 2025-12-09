import unittest

from simpletuner.helpers.training.iteration_tracker import IterationTracker


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.current = float(start)

    def advance(self, seconds: float) -> None:
        self.current += float(seconds)

    def __call__(self) -> float:
        return self.current


class IterationTrackerTestCase(unittest.TestCase):
    def test_metrics_include_iteration_rates(self):
        clock = FakeClock()
        tracker = IterationTracker(time_source=clock)
        tracker.mark_start()

        clock.advance(30)
        tracker.record_step(1)
        clock.advance(30)
        tracker.record_step(2)

        metrics = tracker.iteration_metrics()
        self.assertAlmostEqual(metrics["iteration_step_time_seconds"], 30.0)
        self.assertAlmostEqual(metrics["iterations_per_minute_5m"], 2.0)
        self.assertAlmostEqual(metrics["iterations_per_minute_15m"], 2.0)
        self.assertAlmostEqual(metrics["iterations_per_minute_overall"], 2.0)

        eta = tracker.estimate_eta(current_step=2, total_steps=6)
        self.assertAlmostEqual(eta, 120.0)

    def test_eta_falls_back_to_overall_rate(self):
        clock = FakeClock()
        tracker = IterationTracker(windows=(5,), time_source=clock)
        tracker.mark_start()

        clock.advance(60)
        tracker.record_step(1)

        metrics = tracker.iteration_metrics()
        self.assertEqual(metrics["iteration_step_time_seconds"], 60.0)
        self.assertNotIn("iterations_per_minute_5m", metrics)

        eta = tracker.estimate_eta(current_step=1, total_steps=3)
        self.assertAlmostEqual(eta, 120.0)

    def test_history_trimmed_to_max_window(self):
        clock = FakeClock()
        tracker = IterationTracker(windows=(1,), time_source=clock)
        tracker.mark_start()

        clock.advance(10)
        tracker.record_step(1)
        clock.advance(70)
        tracker.record_step(2)

        # Old entries older than 60 seconds are trimmed automatically.
        self.assertEqual(len(tracker._history), 1)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
