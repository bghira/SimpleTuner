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

    def test_webui_aliases_included_in_metrics(self):
        """Verify that webUI-expected aliases are present in iteration_metrics()."""
        clock = FakeClock()
        tracker = IterationTracker(time_source=clock)
        tracker.mark_start()

        clock.advance(2.5)
        tracker.record_step(1)
        clock.advance(2.5)
        tracker.record_step(2)

        metrics = tracker.iteration_metrics()

        # Check step duration aliases
        self.assertIn("iteration_step_time_seconds", metrics)
        self.assertIn("step_speed_seconds", metrics)
        self.assertIn("seconds_per_step", metrics)
        # All three should have the same value
        self.assertEqual(metrics["step_speed_seconds"], metrics["iteration_step_time_seconds"])
        self.assertEqual(metrics["seconds_per_step"], metrics["iteration_step_time_seconds"])
        self.assertAlmostEqual(metrics["step_speed_seconds"], 2.5)

        # Check steps_per_second alias (derived from overall rate)
        self.assertIn("steps_per_second", metrics)
        self.assertIn("iterations_per_minute_overall", metrics)
        # steps_per_second should be iterations_per_minute_overall / 60
        expected_steps_per_second = metrics["iterations_per_minute_overall"] / 60.0
        self.assertAlmostEqual(metrics["steps_per_second"], expected_steps_per_second)

    def test_steps_per_second_calculation(self):
        """Verify steps_per_second is correctly calculated from overall rate."""
        clock = FakeClock()
        tracker = IterationTracker(time_source=clock)
        tracker.mark_start()

        # Complete 6 steps in 60 seconds = 6 steps/minute = 0.1 steps/second
        for i in range(6):
            clock.advance(10)
            tracker.record_step(i + 1)

        metrics = tracker.iteration_metrics()

        # 6 steps in 60 seconds = 6 iterations per minute
        self.assertAlmostEqual(metrics["iterations_per_minute_overall"], 6.0)
        # 6/60 = 0.1 steps per second
        self.assertAlmostEqual(metrics["steps_per_second"], 0.1)

    def test_no_metrics_before_first_step(self):
        """Verify metrics dict is empty before any steps are recorded."""
        clock = FakeClock()
        tracker = IterationTracker(time_source=clock)
        tracker.mark_start()

        metrics = tracker.iteration_metrics()

        # Should have no metrics yet (no step duration, no rates)
        self.assertEqual(metrics, {})

    def test_eta_returns_none_with_missing_inputs(self):
        """Verify estimate_eta handles None inputs gracefully."""
        clock = FakeClock()
        tracker = IterationTracker(time_source=clock)
        tracker.mark_start()

        clock.advance(10)
        tracker.record_step(1)

        # None current_step
        self.assertIsNone(tracker.estimate_eta(None, 100))
        # None total_steps
        self.assertIsNone(tracker.estimate_eta(50, None))
        # Both None
        self.assertIsNone(tracker.estimate_eta(None, None))

    def test_eta_returns_zero_when_complete(self):
        """Verify estimate_eta returns 0 when current_step >= total_steps."""
        clock = FakeClock()
        tracker = IterationTracker(time_source=clock)
        tracker.mark_start()

        clock.advance(10)
        tracker.record_step(1)

        # Already complete
        self.assertEqual(tracker.estimate_eta(100, 100), 0.0)
        # Over complete
        self.assertEqual(tracker.estimate_eta(110, 100), 0.0)


if __name__ == "__main__":
    unittest.main()
