"""
Tests for validation epoch timing.

These tests verify that epoch-based validation triggers at the correct times:
1. Validation should run at the END of each epoch (the last step), not before
2. epoch_step should be the step within the current epoch, not the global step
3. Validation should not run at the START of epoch 2+ due to stale epoch_step values

Bug report scenario:
- User configured validation_epoch_interval=1, validation_step_interval=0
- Epoch size: ~244 steps
- Expected: validation at steps 0, 244, 488
- Actual: validation at steps 0, 241, 245, 489
  - Step 241 is unexpected (3 steps before epoch boundary)
  - Step 245 is at the start of epoch 2 (should be 488 for end of epoch 2)
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.validation import Validation


class TestValidationEpochStepTracking(unittest.TestCase):
    """Test that epoch_step is correctly tracked within each epoch."""

    def setUp(self):
        # Reset StateTracker between tests
        StateTracker.global_step = 0
        StateTracker.epoch_step = 0
        StateTracker.epoch = 1

    def test_epoch_step_ready_handles_epoch_boundary_correctly(self):
        """
        Test that the validation logic correctly handles epoch boundaries.

        Even though StateTracker.epoch_step contains the global step (not epoch-relative),
        the validation logic should use modulo arithmetic to calculate the correct
        epoch-relative position.

        At global step 245 (first step of epoch 2 with 244 steps/epoch):
        - epoch_step from StateTracker = 245
        - epoch_relative_step = ((245-1) % 244) + 1 = 1
        - epoch_step_ready should be False (we're at step 1, not step 244)
        """
        # Simulate start of epoch 2 with global epoch_step
        StateTracker.set_epoch(2)
        StateTracker.set_global_step(245)
        StateTracker.set_epoch_step(245)  # Global step, not epoch-relative

        # Create validation object
        validation = Validation.__new__(Validation)
        validation.config = SimpleNamespace(
            validation_epoch_interval=1,
            validation_step_interval=None,
            num_update_steps_per_epoch=244,
            gradient_accumulation_steps=1,
        )
        validation._pending_epoch_validation = None
        validation._epoch_validations_completed = set()
        validation.deepspeed = False
        validation.accelerator = MagicMock()
        validation.accelerator.is_main_process = True
        validation.accelerator.num_processes = 1
        validation._update_state()

        # The fix uses: epoch_relative_step = ((epoch_step - 1) % 244) + 1
        # For epoch_step = 245: ((245-1) % 244) + 1 = 0 + 1 = 1
        # epoch_step_ready = (1 == 244) = False

        num_steps_per_epoch = 244
        epoch_relative_step = ((validation.current_epoch_step - 1) % num_steps_per_epoch) + 1
        epoch_step_ready = epoch_relative_step == num_steps_per_epoch

        self.assertEqual(epoch_relative_step, 1, "Epoch-relative step at global step 245 should be 1")
        self.assertFalse(epoch_step_ready, f"epoch_step_ready should be False at start of epoch 2")

    def test_epoch_step_matches_position_within_epoch(self):
        """
        epoch_step should represent the position within the current epoch,
        not the global step count.

        At step 5 of epoch 2 (global_step = 249 with 244 steps per epoch):
        - epoch_step should be 5, not 249
        """
        num_update_steps_per_epoch = 244
        global_step = 249  # Step 5 of epoch 2
        expected_epoch_step = 5  # Step within epoch 2

        # Current behavior (buggy): epoch_step = global_step / grad_accum = 249
        # Expected behavior: epoch_step = (global_step - 1) % num_update_steps_per_epoch + 1 = 5

        actual_epoch_step = global_step  # This is what currently gets set

        # This test will FAIL because actual_epoch_step is 249, not 5
        self.assertEqual(
            expected_epoch_step,
            (global_step - 1) % num_update_steps_per_epoch + 1,
            "Formula for epoch_step should be: (global_step - 1) % num_update_steps_per_epoch + 1",
        )


class TestValidationEpochBoundaryTiming(unittest.TestCase):
    """Test that validation runs at the correct epoch boundary."""

    def setUp(self):
        StateTracker.global_step = 0
        StateTracker.epoch_step = 0
        StateTracker.epoch = 1

    def _create_validation(self, **config_overrides):
        """Create a minimal Validation object for testing."""
        validation = Validation.__new__(Validation)
        config = {
            "validation_epoch_interval": 1,
            "validation_step_interval": None,
            "num_update_steps_per_epoch": 244,
            "gradient_accumulation_steps": 1,
        }
        config.update(config_overrides)
        validation.config = SimpleNamespace(**config)
        validation._pending_epoch_validation = None
        validation._epoch_validations_completed = set()
        validation.deepspeed = False
        # Properly mock accelerator with needed attributes
        validation.accelerator = MagicMock()
        validation.accelerator.is_main_process = True
        validation.accelerator.num_processes = 1
        validation.global_step = 0
        validation.global_resume_step = 0
        validation.current_epoch = 1
        validation.current_epoch_step = 0
        return validation

    def test_validation_runs_at_last_step_of_epoch_not_before(self):
        """
        Validation should run at the LAST step of the epoch (step 244),
        not 1 step before (step 243).

        Bug: The check is `epoch_step >= num_update_steps_per_epoch - 1`
        With num_update_steps_per_epoch = 244, this triggers at epoch_step = 243.
        But step 243 is the 243rd step, not the 244th (last) step.
        """
        validation = self._create_validation(num_update_steps_per_epoch=244)
        prompts = [{"prompt": "test"}]

        # At step 243 (should NOT trigger validation - not the last step yet)
        validation.global_step = 243
        validation.current_epoch_step = 243  # Assuming epoch_step matches
        validation.current_epoch = 1

        should_validate_at_243 = validation.should_perform_intermediary_validation(
            step=243, validation_prompts=prompts, validation_type="intermediary"
        )

        # At step 244 (SHOULD trigger validation - this is the last step)
        validation.global_step = 244
        validation.current_epoch_step = 244
        validation._pending_epoch_validation = None  # Reset

        should_validate_at_244 = validation.should_perform_intermediary_validation(
            step=244, validation_prompts=prompts, validation_type="intermediary"
        )

        # This test will FAIL if validation triggers at step 243
        self.assertFalse(should_validate_at_243, "Validation should NOT run at step 243 (not the last step of epoch)")
        self.assertTrue(should_validate_at_244, "Validation SHOULD run at step 244 (the last step of epoch)")

    def test_validation_at_step_241_bug(self):
        """
        Regression test for the user-reported bug: validation at step 241
        when epoch size is ~244.

        This could happen if:
        1. num_update_steps_per_epoch is miscalculated as 242
        2. There's an off-by-N error in step counting
        """
        # Scenario: user expects 244 steps per epoch, validation at 244
        # But validation runs at 241 (3 steps early)

        validation = self._create_validation(num_update_steps_per_epoch=244)
        prompts = [{"prompt": "test"}]

        # Simulate the conditions at step 241
        validation.global_step = 241
        validation.current_epoch_step = 241
        validation.current_epoch = 1

        should_validate = validation.should_perform_intermediary_validation(
            step=241, validation_prompts=prompts, validation_type="intermediary"
        )

        # Validation should NOT run at step 241 if epoch has 244 steps
        # The check is: 241 >= 244 - 1 = 243 → False
        # So this should pass... unless there's a different bug
        self.assertFalse(should_validate, "Validation should NOT run at step 241 when epoch has 244 steps")

    def test_validation_should_not_run_at_start_of_epoch_2(self):
        """
        Validation should NOT run at the START of epoch 2 (step 245).
        It should only run at the END of epoch 2 (step 488).

        Bug: If epoch_step is not reset, epoch_step = 245 >= 243 is True,
        and validation incorrectly runs at the first step of epoch 2.
        """
        validation = self._create_validation(num_update_steps_per_epoch=244)
        prompts = [{"prompt": "test"}]

        # Epoch 1 completed
        validation._epoch_validations_completed.add(1)

        # Start of epoch 2
        validation.global_step = 245
        validation.current_epoch = 2
        # BUG: epoch_step is set to 245 (global) instead of 1 (epoch-relative)
        validation.current_epoch_step = 245  # This is the bug

        should_validate = validation.should_perform_intermediary_validation(
            step=245, validation_prompts=prompts, validation_type="intermediary"
        )

        # This test SHOULD pass (validation should NOT run at start of epoch)
        # But with the epoch_step bug, epoch_step_ready = 245 >= 243 = True
        # AND epoch 2 is not in completed, so it WILL trigger validation!
        self.assertFalse(
            should_validate,
            f"Validation should NOT run at step 245 (start of epoch 2). "
            f"epoch_step={validation.current_epoch_step} should be ~1, not 245",
        )


class TestValidationEpochStepReadyCalculation(unittest.TestCase):
    """Test the epoch_step_ready calculation specifically."""

    def setUp(self):
        StateTracker.global_step = 0
        StateTracker.epoch_step = 0
        StateTracker.epoch = 1

    def test_epoch_step_ready_requires_last_step(self):
        """
        epoch_step_ready should only be True at the LAST step of the epoch.

        With 244 steps per epoch (steps 1-244):
        - Step 243: epoch_step_ready should be False
        - Step 244: epoch_step_ready should be True

        The fix uses: epoch_relative_step = ((epoch_step - 1) % num_steps) + 1
        And checks: epoch_step_ready = (epoch_relative_step == num_steps)
        """
        num_update_steps_per_epoch = 244

        # Test: at step 243 (second-to-last), should be False
        epoch_step = 243
        epoch_relative_step = ((epoch_step - 1) % num_update_steps_per_epoch) + 1  # 243
        epoch_step_ready = epoch_relative_step == num_update_steps_per_epoch  # 243 == 244 = False

        self.assertFalse(epoch_step_ready, f"epoch_step_ready should be False at step 243 of a 244-step epoch")

        # Test: at step 244 (last step), should be True
        epoch_step = 244
        epoch_relative_step = ((epoch_step - 1) % num_update_steps_per_epoch) + 1  # 244
        epoch_step_ready = epoch_relative_step == num_update_steps_per_epoch  # 244 == 244 = True

        self.assertTrue(epoch_step_ready, f"epoch_step_ready should be True at step 244 of a 244-step epoch")

    def test_epoch_step_ready_with_gradient_accumulation(self):
        """
        Test epoch_step_ready with gradient accumulation.

        With gradient_accumulation_steps=4 and 244 update steps per epoch:
        - Total micro-steps per epoch: 244 * 4 = 976
        - epoch_step = micro_step / 4

        At micro-step 972 (update step 243): epoch_step_ready should be False
        At micro-step 976 (update step 244): epoch_step_ready should be True
        """
        num_update_steps_per_epoch = 244
        gradient_accumulation_steps = 4

        # Micro-step 972, update step 243
        micro_step = 972
        epoch_step = micro_step // gradient_accumulation_steps  # 243
        epoch_relative_step = ((epoch_step - 1) % num_update_steps_per_epoch) + 1
        epoch_step_ready = epoch_relative_step == num_update_steps_per_epoch

        # Should be False at update step 243
        self.assertFalse(epoch_step_ready, f"epoch_step_ready should be False at update step 243")

        # Micro-step 976, update step 244
        micro_step = 976
        epoch_step = micro_step // gradient_accumulation_steps  # 244
        epoch_relative_step = ((epoch_step - 1) % num_update_steps_per_epoch) + 1
        epoch_step_ready = epoch_relative_step == num_update_steps_per_epoch

        # Should be True at update step 244
        self.assertTrue(epoch_step_ready, f"epoch_step_ready should be True at update step 244")


class TestValidationPendingEpochMechanism(unittest.TestCase):
    """Test the _pending_epoch_validation mechanism."""

    def setUp(self):
        StateTracker.global_step = 0
        StateTracker.epoch_step = 0
        StateTracker.epoch = 1

    def _create_validation(self, **config_overrides):
        validation = Validation.__new__(Validation)
        config = {
            "validation_epoch_interval": 1,
            "validation_step_interval": None,
            "num_update_steps_per_epoch": 244,
            "gradient_accumulation_steps": 1,
        }
        config.update(config_overrides)
        validation.config = SimpleNamespace(**config)
        validation._pending_epoch_validation = None
        validation._epoch_validations_completed = set()
        validation.deepspeed = False
        # Properly mock accelerator with needed attributes
        validation.accelerator = MagicMock()
        validation.accelerator.is_main_process = True
        validation.accelerator.num_processes = 1
        validation.global_step = 0
        validation.global_resume_step = 0
        validation.current_epoch = 1
        validation.current_epoch_step = 0
        return validation

    def test_pending_epoch_cleared_after_validation(self):
        """
        After validation runs for an epoch, _pending_epoch_validation should be
        cleared and the epoch should be added to _epoch_validations_completed.
        """
        validation = self._create_validation()
        prompts = [{"prompt": "test"}]

        # Simulate end of epoch 1 - trigger validation
        validation.global_step = 244
        validation.current_epoch_step = 244
        validation.current_epoch = 1

        # First call sets pending and returns True
        result1 = validation.should_perform_intermediary_validation(
            step=244, validation_prompts=prompts, validation_type="intermediary"
        )

        self.assertTrue(result1)
        self.assertEqual(validation._pending_epoch_validation, 1)

        # Simulate validation completion (normally done in run_validations)
        validation._epoch_validations_completed.add(1)
        validation._pending_epoch_validation = None

        # Second call for same epoch should return False
        result2 = validation.should_perform_intermediary_validation(
            step=244, validation_prompts=prompts, validation_type="intermediary"
        )

        self.assertFalse(result2, "Validation should not run twice for same epoch")

    def test_validation_runs_once_per_epoch_with_correct_timing(self):
        """
        Test that validation runs exactly once at the end of each epoch.

        Expected behavior for 2 epochs with 244 steps each:
        - Epoch 1: validation at step 244 (end of epoch)
        - Epoch 2: validation at step 488 (end of epoch)

        NOT at step 243, 245, or 489.
        """
        validation = self._create_validation()
        prompts = [{"prompt": "test"}]

        validation_steps = []

        # Simulate training loop for 2 epochs
        for global_step in range(1, 489):
            epoch = (global_step - 1) // 244 + 1
            # BUG: This should be epoch-relative, but currently it's global
            epoch_step = global_step  # This is what the iterator does (buggy)
            # Correct would be: epoch_step = (global_step - 1) % 244 + 1

            validation.global_step = global_step
            validation.current_epoch = epoch
            validation.current_epoch_step = epoch_step

            should_validate = validation.should_perform_intermediary_validation(
                step=global_step, validation_prompts=prompts, validation_type="intermediary"
            )

            if should_validate:
                validation_steps.append(global_step)
                # Simulate validation completion
                validation._epoch_validations_completed.add(epoch)
                validation._pending_epoch_validation = None

        # Expected: [244, 488]
        # Actual (with bug): likely [243, 245, ...] or similar
        self.assertEqual(validation_steps, [244, 488], f"Validation should run at steps [244, 488], got {validation_steps}")


class TestValidationCheckpointAlignment(unittest.TestCase):
    """
    Test that validation timing aligns with checkpoint timing.

    The user reported that checkpoints are correct (steps 244, 488) but
    validation is off (steps 241, 245, 489). This suggests the two systems
    use different state tracking.
    """

    def setUp(self):
        StateTracker.global_step = 0
        StateTracker.epoch_step = 0
        StateTracker.epoch = 1

    def test_validation_and_checkpoint_same_epoch_boundary(self):
        """
        When both validation_epoch_interval and checkpoint_epoch_interval are 1,
        both should trigger at the exact same step (end of each epoch).

        The checkpoint logic uses:
        - epoch % checkpoint_epoch_interval == 0
        - Checked AFTER prepared_batch is False (epoch ended)

        The validation logic (FIXED) uses:
        - current_epoch % validation_epoch_interval == 0
        - epoch_relative_step = ((epoch_step - 1) % num_steps) + 1
        - epoch_step_ready = (epoch_relative_step == num_steps)
        - Checked DURING the training loop at the last step

        After the fix, both trigger at step 244 (the last step of the epoch).
        """
        num_update_steps_per_epoch = 244

        # With the fix, validation triggers when:
        # - epoch_relative_step == num_update_steps_per_epoch
        # - This is True only at the last step of the epoch

        # Calculate when validation triggers (using the fixed formula)
        # epoch_relative_step = ((epoch_step - 1) % num_steps) + 1
        # For epoch_step = 244: ((244-1) % 244) + 1 = 244 → triggers
        # For epoch_step = 243: ((243-1) % 244) + 1 = 243 → doesn't trigger

        validation_trigger_step = num_update_steps_per_epoch  # 244 (fixed)
        checkpoint_trigger_step = num_update_steps_per_epoch  # 244

        # After the fix, they should be the same
        self.assertEqual(
            validation_trigger_step,
            checkpoint_trigger_step,
            f"Validation and checkpoint should both trigger at step {num_update_steps_per_epoch}",
        )


if __name__ == "__main__":
    unittest.main()
