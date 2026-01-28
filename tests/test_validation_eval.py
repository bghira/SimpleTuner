import unittest
from types import SimpleNamespace

from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.validation import Evaluation


class EvaluationSchedulingTests(unittest.TestCase):
    def setUp(self):
        self._prev_args = StateTracker.get_args()
        self._prev_epoch = StateTracker.get_epoch()
        self._prev_global_step = StateTracker.get_global_step()
        StateTracker.set_epoch(1)
        StateTracker.set_global_step(0)

    def tearDown(self):
        StateTracker.set_args(self._prev_args)
        StateTracker.set_epoch(self._prev_epoch)
        StateTracker.set_global_step(self._prev_global_step)

    def _make_eval(self, **kwargs) -> Evaluation:
        config_defaults = {
            "eval_steps_interval": None,
            "eval_epoch_interval": None,
            "num_update_steps_per_epoch": 4,
            "weight_dtype": None,
        }
        config_defaults.update(kwargs)
        config = SimpleNamespace(**config_defaults)
        StateTracker.set_args(config)
        accelerator = SimpleNamespace(is_main_process=True)
        evaluator = Evaluation(accelerator=accelerator)
        # Directly set config to ensure test isolation
        evaluator.config = config
        return evaluator

    def test_would_evaluate_step_interval_triggers_on_multiple(self):
        evaluator = self._make_eval(eval_steps_interval=2)
        training_state = {"global_step": 0, "global_resume_step": 0}
        self.assertFalse(evaluator.would_evaluate(training_state))

        training_state["global_step"] = 2
        training_state["global_resume_step"] = 1
        self.assertTrue(evaluator.would_evaluate(training_state))

    def test_would_evaluate_epoch_fractional_interval(self):
        evaluator = self._make_eval(eval_epoch_interval=0.5, num_update_steps_per_epoch=4)
        training_state = {"global_step": 1, "global_resume_step": 0, "current_epoch": 1}
        evaluator.would_evaluate(training_state)  # Prime any internal counters

        training_state["global_step"] = 2
        first_cross = evaluator.would_evaluate(training_state)

        training_state["global_step"] = 4
        second_cross = evaluator.would_evaluate(training_state)

        self.assertTrue(first_cross or second_cross)

    def test_would_evaluate_warns_when_dual_schedule_configured(self):
        evaluator = self._make_eval(eval_steps_interval=2, eval_epoch_interval=1.0, num_update_steps_per_epoch=2)
        training_state = {"global_step": 1, "global_resume_step": 0, "current_epoch": 1}
        self.assertFalse(evaluator._warned_dual_schedule)
        evaluator.would_evaluate(training_state)
        self.assertTrue(evaluator._warned_dual_schedule)


class EvaluationDynamicEpochScheduleTests(unittest.TestCase):
    """
    Tests for eval scheduling with dynamic epoch step counts.

    Issue #2508: When datasets have different start_epoch values, the number
    of steps per epoch changes. The _epoch_progress calculation must account
    for this to correctly trigger fractional eval_epoch_interval.
    """

    def setUp(self):
        self._prev_args = StateTracker.get_args()
        self._prev_epoch = StateTracker.get_epoch()
        self._prev_global_step = StateTracker.get_global_step()
        # Initialize to clean state to avoid interference from other tests
        StateTracker.set_epoch(1)
        StateTracker.set_global_step(0)

    def tearDown(self):
        StateTracker.set_args(self._prev_args)
        StateTracker.set_epoch(self._prev_epoch)
        StateTracker.set_global_step(self._prev_global_step)

    def _make_eval(self, **kwargs) -> Evaluation:
        config_defaults = {
            "eval_steps_interval": None,
            "eval_epoch_interval": None,
            "num_update_steps_per_epoch": 100,
            "gradient_accumulation_steps": 1,
            "epoch_batches_schedule": None,
            "weight_dtype": None,
        }
        config_defaults.update(kwargs)
        config = SimpleNamespace(**config_defaults)
        StateTracker.set_args(config)
        accelerator = SimpleNamespace(is_main_process=True)
        evaluator = Evaluation(accelerator=accelerator)
        # Directly set config to ensure test isolation from other tests
        # that may modify StateTracker state
        evaluator.config = config
        return evaluator

    def test_epoch_progress_without_schedule(self):
        """Without epoch_batches_schedule, epoch_progress uses simple formula."""
        evaluator = self._make_eval(
            num_update_steps_per_epoch=100,
            epoch_batches_schedule=None,  # Explicitly set to None
        )
        training_state = {"global_step": 150, "current_epoch": 2}
        # Epoch 2, step 150 = 50 steps into epoch 2
        # epoch_progress = 1 + (50/100) = 1.5
        progress = evaluator._epoch_progress(training_state, 100)
        self.assertIsNotNone(progress, "epoch_progress should not return None")
        self.assertAlmostEqual(progress, 1.5, places=2)

    def test_epoch_progress_with_dynamic_schedule(self):
        """
        With epoch_batches_schedule, epoch_progress accounts for variable step counts.

        Scenario from issue #2508:
        - Epochs 1-4: 98 steps each (only 512 images)
        - Epochs 5+: 126 steps each (512 + 1024 images)

        At step 400, epoch 5:
        - Epoch 5 starts at step 393 (98*4 + 1)
        - Steps into epoch 5 = 400 - 392 = 8
        - epoch_progress = 4 + (8/126) ≈ 4.063
        """
        # epoch_batches_schedule: epoch -> batches added at that epoch
        # Epoch 1: 98 batches (512 images)
        # Epoch 5: 28 additional batches (1024 images with start_epoch=5)
        evaluator = self._make_eval(
            epoch_batches_schedule={1: 98, 5: 28},
            num_update_steps_per_epoch=126,  # Current epoch (5+) step count
            gradient_accumulation_steps=1,
        )
        training_state = {"global_step": 400, "current_epoch": 5}

        progress = evaluator._epoch_progress(training_state, 126)

        # Correct calculation:
        # - Epochs 1-4: 98 steps each = 392 cumulative
        # - Epoch 5 starts at step 393, current step is 400
        # - Steps into epoch 5 = 400 - 392 = 8
        # - epoch_progress = 4 + (8/126) ≈ 4.063
        self.assertAlmostEqual(progress, 4.063, places=2)

    def test_epoch_progress_wrong_without_schedule_fix(self):
        """
        Demonstrate the bug: without the schedule fix, epoch_progress is wrong.

        At step 400, epoch 5, with steps_per_epoch=126:
        - Old (wrong): epoch_start_step = (5-1) * 126 = 504
        - Steps completed = 400 - 504 = -104 (clamped to 0)
        - epoch_progress = 4 + 0 = 4.0 (WRONG!)
        """
        # Without epoch_batches_schedule, it falls back to the simple formula
        evaluator = self._make_eval(
            epoch_batches_schedule=None,  # No schedule = old behavior
            num_update_steps_per_epoch=126,
        )
        training_state = {"global_step": 400, "current_epoch": 5}

        progress = evaluator._epoch_progress(training_state, 126)

        # Old formula: epoch_start = (5-1) * 126 = 504
        # 400 - 504 = -104, clamped to 0
        # progress = 4 + 0 = 4.0
        self.assertAlmostEqual(progress, 4.0, places=2)

    def test_fractional_eval_interval_with_dynamic_schedule(self):
        """
        Fractional eval_epoch_interval should work correctly with dynamic scheduling.

        With eval_epoch_interval=0.5 and changing epoch sizes:
        - Epoch 5: 126 steps, should eval at steps ~455 (4.5 epochs) and ~518 (5.0 epochs)
        """
        evaluator = self._make_eval(
            eval_epoch_interval=0.5,
            epoch_batches_schedule={1: 98, 5: 28},
            num_update_steps_per_epoch=126,
            gradient_accumulation_steps=1,
        )

        # Prime at start of epoch 5 (step 393)
        training_state = {"global_step": 393, "global_resume_step": 392, "current_epoch": 5}
        evaluator.would_evaluate(training_state)

        # At step 455 (about halfway through epoch 5)
        # 455 - 392 = 63 steps into epoch 5
        # epoch_progress = 4 + 63/126 = 4.5
        # completed_intervals = floor(4.5 / 0.5) = 9
        training_state["global_step"] = 455
        should_eval_at_455 = evaluator.would_evaluate(training_state)

        # At step 518 (end of epoch 5)
        # 518 - 392 = 126 steps into epoch 5
        # epoch_progress = 4 + 126/126 = 5.0
        # completed_intervals = floor(5.0 / 0.5) = 10
        training_state["global_step"] = 518
        should_eval_at_518 = evaluator.would_evaluate(training_state)

        # At least one of these should trigger (depending on initial state)
        self.assertTrue(should_eval_at_455 or should_eval_at_518, "Eval should trigger at half-epoch or full-epoch boundary")

    def test_epoch_progress_epoch_6_after_schedule_change(self):
        """
        Verify epoch progress is correct for epoch 6 after schedule change at epoch 5.
        """
        evaluator = self._make_eval(
            epoch_batches_schedule={1: 98, 5: 28},
            num_update_steps_per_epoch=126,
            gradient_accumulation_steps=1,
        )

        # Step 519 = start of epoch 6
        # Cumulative: epochs 1-4 = 98*4 = 392, epoch 5 = 126, total = 518
        training_state = {"global_step": 519, "current_epoch": 6}
        progress = evaluator._epoch_progress(training_state, 126)

        # epoch_progress = 5 + (1/126) ≈ 5.008
        self.assertAlmostEqual(progress, 5.008, places=2)

    def test_resume_mid_epoch_primes_intervals_correctly(self):
        """
        Test that resuming mid-epoch correctly primes _epoch_intervals_completed.

        Scenario: Training crashes at step 450 (mid epoch 5), resumes.
        - Epochs 1-4: 98 steps each = 392 cumulative
        - Epoch 5: 126 steps, starts at 393, ends at 518
        - Resume at step 450 = 58 steps into epoch 5
        - epoch_progress = 4 + 58/126 = 4.46
        - With eval_epoch_interval=0.5, completed_intervals = floor(4.46/0.5) = 8

        After resume, eval should trigger at:
        - Step 455: epoch_progress = 4.5, intervals = 9 (triggers)
        - Step 518: epoch_progress = 5.0, intervals = 10 (triggers)
        """
        evaluator = self._make_eval(
            eval_epoch_interval=0.5,
            epoch_batches_schedule={1: 98, 5: 28},
            num_update_steps_per_epoch=126,
            gradient_accumulation_steps=1,
        )

        # Simulate resume at step 450 (mid epoch 5)
        # global_resume_step is the step we're resuming FROM (last completed step)
        resume_step = 450
        training_state = {
            "global_step": resume_step,
            "global_resume_step": resume_step,
            "current_epoch": 5,
        }

        # At exact resume step (global_step == global_resume_step), eval is skipped
        # This is intentional to avoid re-evaluating the checkpoint we just loaded
        first_call = evaluator.would_evaluate(training_state)
        self.assertFalse(first_call, "Should skip eval at exact resume step")
        self.assertIsNone(evaluator._epoch_intervals_completed, "Not primed yet at resume step")

        # First step AFTER resume (global_step > global_resume_step) primes the counter
        training_state["global_step"] = 451
        second_call = evaluator.would_evaluate(training_state)
        self.assertFalse(second_call, "First step after resume should prime, not trigger")

        # Verify the primed value is correct
        # epoch_progress at step 451 = 4 + (451-392)/126 = 4.468
        # completed_intervals = floor(4.468 / 0.5) = 8
        self.assertEqual(evaluator._epoch_intervals_completed, 8, "Should have primed to 8 intervals (floor(4.468/0.5))")

        # Step 452-454: still interval 8, no eval
        training_state["global_step"] = 454
        self.assertFalse(evaluator.would_evaluate(training_state))

        # Step 455: epoch_progress = 4 + 63/126 = 4.5, intervals = 9 > 8, triggers!
        training_state["global_step"] = 455
        should_eval_455 = evaluator.would_evaluate(training_state)
        self.assertTrue(should_eval_455, "Should trigger eval at step 455 (4.5 epochs)")
        self.assertEqual(evaluator._epoch_intervals_completed, 9)

        # Step 456-517: still interval 9, no eval
        training_state["global_step"] = 500
        self.assertFalse(evaluator.would_evaluate(training_state))

        # Step 518: epoch_progress = 5.0, intervals = 10 > 9, triggers!
        training_state["global_step"] = 518
        should_eval_518 = evaluator.would_evaluate(training_state)
        self.assertTrue(should_eval_518, "Should trigger eval at step 518 (5.0 epochs)")
        self.assertEqual(evaluator._epoch_intervals_completed, 10)

    def test_resume_after_schedule_change_epoch(self):
        """
        Test resuming after the epoch where dataset schedule changed.

        Scenario: Resume at step 600 (epoch 6), after 1024 images activated at epoch 5.
        - Epochs 1-4: 98 steps each = 392
        - Epoch 5: 126 steps = 518 cumulative
        - Epoch 6: starts at 519, step 600 = 81 steps into epoch 6
        - epoch_progress = 5 + 82/126 = 5.65
        - completed_intervals = floor(5.65/0.5) = 11
        """
        evaluator = self._make_eval(
            eval_epoch_interval=0.5,
            epoch_batches_schedule={1: 98, 5: 28},
            num_update_steps_per_epoch=126,
            gradient_accumulation_steps=1,
        )

        resume_step = 600
        training_state = {
            "global_step": resume_step,
            "global_resume_step": resume_step,
            "current_epoch": 6,
        }

        # At exact resume step, eval is skipped
        evaluator.would_evaluate(training_state)
        self.assertIsNone(evaluator._epoch_intervals_completed)

        # First step after resume primes the counter
        training_state["global_step"] = 601
        evaluator.would_evaluate(training_state)

        # Verify correct priming
        # epoch_progress = 5 + (601-518)/126 = 5 + 83/126 = 5.659
        # completed_intervals = floor(5.659/0.5) = 11
        self.assertEqual(evaluator._epoch_intervals_completed, 11, "Should have primed to 11 intervals")

        # Next half-epoch boundary is at 5.5 epochs (already passed) and 6.0 epochs
        # 6.0 epochs = 518 + 126 = 644
        training_state["global_step"] = 644
        should_eval = evaluator.would_evaluate(training_state)
        self.assertTrue(should_eval, "Should trigger eval at step 644 (6.0 epochs)")
        self.assertEqual(evaluator._epoch_intervals_completed, 12)

    def test_resume_exactly_at_interval_boundary(self):
        """
        Test resuming exactly at an interval boundary doesn't double-trigger.

        Resume at step 455 which is exactly 4.5 epochs.
        The priming call should NOT trigger eval (returns False).
        The next interval (5.0 epochs at step 518) should trigger.
        """
        evaluator = self._make_eval(
            eval_epoch_interval=0.5,
            epoch_batches_schedule={1: 98, 5: 28},
            num_update_steps_per_epoch=126,
            gradient_accumulation_steps=1,
        )

        # Resume exactly at 4.5 epoch boundary (step 455)
        # 392 + 63 = 455, epoch_progress = 4 + 63/126 = 4.5
        resume_step = 455
        training_state = {
            "global_step": resume_step,
            "global_resume_step": resume_step,
            "current_epoch": 5,
        }

        # At exact resume step, eval is skipped entirely
        first_call = evaluator.would_evaluate(training_state)
        self.assertFalse(first_call, "Should skip at exact resume step")
        self.assertIsNone(evaluator._epoch_intervals_completed)

        # Step 456 primes (first step after resume)
        training_state["global_step"] = 456
        prime_call = evaluator.would_evaluate(training_state)
        self.assertFalse(prime_call, "Priming call should not trigger")
        # epoch_progress at 456 = 4 + 64/126 = 4.508, floor(4.508/0.5) = 9
        self.assertEqual(evaluator._epoch_intervals_completed, 9)

        # Step 457-517 should not trigger (still interval 9)
        training_state["global_step"] = 500
        self.assertFalse(evaluator.would_evaluate(training_state))

        # Step 518 (5.0 epochs) should trigger
        training_state["global_step"] = 518
        self.assertTrue(evaluator.would_evaluate(training_state))
        self.assertEqual(evaluator._epoch_intervals_completed, 10)


if __name__ == "__main__":
    unittest.main()
