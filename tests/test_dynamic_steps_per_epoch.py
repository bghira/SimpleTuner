"""
Tests for dynamic steps per epoch calculation with dataset scheduling.

Issue #2483: When datasets have different start_epoch values, the number of
steps per epoch varies. This test verifies that:
1. steps_per_epoch is calculated correctly for each epoch based on active datasets
2. Validation triggers at the correct epoch boundaries
3. Epoch rollover updates num_update_steps_per_epoch correctly
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.validation import Validation


class TestDynamicStepsPerEpoch(unittest.TestCase):
    """Test dynamic steps per epoch calculation based on dataset scheduling."""

    def setUp(self):
        StateTracker.global_step = 0
        StateTracker.epoch_step = 0
        StateTracker.epoch = 1

    def test_get_steps_per_epoch_epoch_1_only_initial_datasets(self):
        """
        In epoch 1, only datasets with start_epoch=1 should be counted.

        Example:
        - Dataset A: 100 batches, start_epoch=1
        - Dataset B: 200 batches, start_epoch=2

        Epoch 1 should have 100 steps, not 300.
        """
        # Create mock config with epoch_batches_schedule
        config = SimpleNamespace(
            epoch_batches_schedule={1: 100, 2: 200},  # epoch -> batches added
            gradient_accumulation_steps=1,
            num_update_steps_per_epoch=300,  # Would be wrong if used statically
        )

        # Simulate get_steps_per_epoch_for_epoch calculation
        # This mirrors the logic in trainer.get_steps_per_epoch_for_epoch()
        def get_steps_per_epoch_for_epoch(epoch, cfg):
            epoch_batches_schedule = getattr(cfg, "epoch_batches_schedule", None)
            if epoch_batches_schedule is None:
                return getattr(cfg, "num_update_steps_per_epoch", 1) or 1
            active_batches = 0
            for start_epoch, batches in epoch_batches_schedule.items():
                if start_epoch <= epoch:
                    active_batches += batches
            if active_batches <= 0:
                return getattr(cfg, "num_update_steps_per_epoch", 1) or 1
            grad_accum = max(cfg.gradient_accumulation_steps or 1, 1)
            return max(active_batches // grad_accum, 1)

        # Epoch 1: only datasets with start_epoch=1 are active
        steps_epoch_1 = get_steps_per_epoch_for_epoch(1, config)
        self.assertEqual(steps_epoch_1, 100, "Epoch 1 should have 100 steps (only initial datasets)")

        # Epoch 2: all datasets are active
        steps_epoch_2 = get_steps_per_epoch_for_epoch(2, config)
        self.assertEqual(steps_epoch_2, 300, "Epoch 2 should have 300 steps (all datasets)")

    def test_get_steps_per_epoch_multiple_scheduled_epochs(self):
        """
        Test with multiple datasets starting at different epochs.

        Example:
        - Dataset A: 50 batches, start_epoch=1
        - Dataset B: 100 batches, start_epoch=2
        - Dataset C: 150 batches, start_epoch=3

        Epoch 1: 50 steps
        Epoch 2: 150 steps
        Epoch 3+: 300 steps
        """
        config = SimpleNamespace(
            epoch_batches_schedule={1: 50, 2: 100, 3: 150},
            gradient_accumulation_steps=1,
            num_update_steps_per_epoch=300,
        )

        def get_steps_per_epoch_for_epoch(epoch, cfg):
            epoch_batches_schedule = getattr(cfg, "epoch_batches_schedule", None)
            if epoch_batches_schedule is None:
                return getattr(cfg, "num_update_steps_per_epoch", 1) or 1
            active_batches = 0
            for start_epoch, batches in epoch_batches_schedule.items():
                if start_epoch <= epoch:
                    active_batches += batches
            if active_batches <= 0:
                return getattr(cfg, "num_update_steps_per_epoch", 1) or 1
            grad_accum = max(cfg.gradient_accumulation_steps or 1, 1)
            return max(active_batches // grad_accum, 1)

        self.assertEqual(get_steps_per_epoch_for_epoch(1, config), 50)
        self.assertEqual(get_steps_per_epoch_for_epoch(2, config), 150)
        self.assertEqual(get_steps_per_epoch_for_epoch(3, config), 300)
        self.assertEqual(get_steps_per_epoch_for_epoch(4, config), 300)  # No change after all activate

    def test_get_steps_per_epoch_with_gradient_accumulation(self):
        """Test that gradient accumulation is factored into steps per epoch."""
        config = SimpleNamespace(
            epoch_batches_schedule={1: 100, 2: 200},
            gradient_accumulation_steps=4,
            num_update_steps_per_epoch=75,  # Would be wrong
        )

        def get_steps_per_epoch_for_epoch(epoch, cfg):
            epoch_batches_schedule = getattr(cfg, "epoch_batches_schedule", None)
            if epoch_batches_schedule is None:
                return getattr(cfg, "num_update_steps_per_epoch", 1) or 1
            active_batches = 0
            for start_epoch, batches in epoch_batches_schedule.items():
                if start_epoch <= epoch:
                    active_batches += batches
            if active_batches <= 0:
                return getattr(cfg, "num_update_steps_per_epoch", 1) or 1
            grad_accum = max(cfg.gradient_accumulation_steps or 1, 1)
            return max(active_batches // grad_accum, 1)

        # 100 batches / 4 grad_accum = 25 update steps for epoch 1
        self.assertEqual(get_steps_per_epoch_for_epoch(1, config), 25)
        # 300 batches / 4 grad_accum = 75 update steps for epoch 2
        self.assertEqual(get_steps_per_epoch_for_epoch(2, config), 75)


class TestValidationWithDynamicStepsPerEpoch(unittest.TestCase):
    """Test that validation triggers correctly with dynamic steps per epoch."""

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
            "num_update_steps_per_epoch": 100,  # Will be overridden
            "gradient_accumulation_steps": 1,
        }
        config.update(config_overrides)
        validation.config = SimpleNamespace(**config)
        validation._pending_epoch_validation = None
        validation._epoch_validations_completed = set()
        validation.deepspeed = False
        validation.accelerator = MagicMock()
        validation.accelerator.is_main_process = True
        validation.accelerator.num_processes = 1
        validation.global_step = 0
        validation.global_resume_step = 0
        validation.current_epoch = 1
        validation.current_epoch_step = 0
        return validation

    def test_validation_at_end_of_short_epoch_1(self):
        """
        When epoch 1 has fewer steps due to scheduled datasets,
        validation should trigger at the actual end of epoch 1.

        Scenario:
        - Epoch 1: 100 steps (only initial datasets)
        - Epoch 2: 300 steps (all datasets)

        Validation should trigger at step 100 (end of epoch 1),
        not at step 300.
        """
        # During epoch 1, num_update_steps_per_epoch should be 100
        validation = self._create_validation(num_update_steps_per_epoch=100)
        prompts = [{"prompt": "test"}]

        # At step 99 (not the last step of epoch 1)
        validation.global_step = 99
        validation.current_epoch_step = 99
        validation.current_epoch = 1

        should_validate_at_99 = validation.should_perform_intermediary_validation(
            step=99, validation_prompts=prompts, validation_type="intermediary"
        )
        self.assertFalse(should_validate_at_99, "Should not validate at step 99")

        # At step 100 (last step of epoch 1)
        validation.global_step = 100
        validation.current_epoch_step = 100

        should_validate_at_100 = validation.should_perform_intermediary_validation(
            step=100, validation_prompts=prompts, validation_type="intermediary"
        )
        self.assertTrue(should_validate_at_100, "Should validate at step 100 (end of epoch 1)")

    def test_validation_not_triggered_by_old_cached_value(self):
        """
        If num_update_steps_per_epoch was incorrectly set to 300 (total)
        instead of 100 (initial), validation would not trigger at step 100.

        This test demonstrates the bug: with the old cached value,
        validation would NOT trigger at the end of epoch 1.
        """
        # Incorrect: using total batches instead of initial
        validation = self._create_validation(num_update_steps_per_epoch=300)
        prompts = [{"prompt": "test"}]

        # At step 100 (actual end of epoch 1 with scheduling)
        validation.global_step = 100
        validation.current_epoch_step = 100
        validation.current_epoch = 1

        should_validate = validation.should_perform_intermediary_validation(
            step=100, validation_prompts=prompts, validation_type="intermediary"
        )

        # With the wrong cached value of 300, validation would NOT trigger at 100
        # because epoch_relative_step (100) != num_steps_per_epoch (300)
        self.assertFalse(should_validate, "With wrong cached value, validation incorrectly does NOT trigger at step 100")

    def test_epoch_2_validation_at_correct_step(self):
        """
        In epoch 2, num_update_steps_per_epoch should be updated to 300.
        Validation should trigger at step 300 (relative to epoch 2).

        Timeline:
        - Epoch 1: steps 1-100 (100 steps, validation at 100)
        - Epoch 2: steps 101-400 (300 steps, validation at 400)
        """
        # Epoch 1 completed
        validation = self._create_validation(
            num_update_steps_per_epoch=300,
            epoch_batches_schedule={1: 100, 2: 200},
            gradient_accumulation_steps=1,
        )
        validation._epoch_validations_completed.add(1)
        prompts = [{"prompt": "test"}]

        # At step 300 in epoch 2 (global step 400 = 100 + 300)
        # Using epoch-relative step tracking
        validation.global_step = 400
        validation.current_epoch_step = 300  # Step within epoch 2
        validation.current_epoch = 2

        # The validation logic uses modulo to get epoch-relative position
        # With num_update_steps_per_epoch=300, step 300 is the end of epoch 2
        should_validate = validation.should_perform_intermediary_validation(
            step=400, validation_prompts=prompts, validation_type="intermediary"
        )

        self.assertTrue(should_validate, "Validation should trigger at end of epoch 2 (step 400 = 100 + 300)")

    def test_validation_uses_epoch_start_step_with_schedule(self):
        """
        With dynamic epoch schedules, validation should compute epoch-relative
        step from global_step using the epoch start step.

        Scenario adapted from issue #2523 / #2508:
        - Epochs 1-4: 98 steps each
        - Epoch 5+: 126 steps each

        End of epoch 5 should be global step 392 + 126 = 518.
        """
        validation = self._create_validation(
            num_update_steps_per_epoch=126,
            epoch_batches_schedule={1: 98, 5: 28},
            gradient_accumulation_steps=1,
        )
        prompts = [{"prompt": "test"}]

        validation.global_step = 518
        validation.current_epoch_step = 518  # Global step, as produced by iterator
        validation.current_epoch = 5

        should_validate = validation.should_perform_intermediary_validation(
            step=518, validation_prompts=prompts, validation_type="intermediary"
        )

        self.assertTrue(should_validate, "Validation should trigger at end of epoch 5 with dynamic schedule")


class TestIssue2483Scenario(unittest.TestCase):
    """
    Test case based on the exact scenario from issue #2483.

    User configuration:
    - train-512-image: repeats=5, start_epoch=1
    - train-1024-image: repeats=1, start_epoch=2
    - regularisation-512-reg-img: repeats=3, start_epoch=1
    - regularisation-1024-reg-img: repeats=0, start_epoch=2

    Expected: Checkpoints and validation at same steps (epoch boundaries).
    Actual bug: Validation at wrong times.
    """

    def setUp(self):
        StateTracker.global_step = 0
        StateTracker.epoch_step = 0
        StateTracker.epoch = 1

    def test_validation_aligns_with_checkpoints(self):
        """
        Validation should trigger at the same step as epoch checkpoints.

        From the issue:
        - checkpoints were at: 98, 224, 350
        - validation was at: 0, 125, 231, 251, 351, 377 (wrong!)

        With the fix, validation should be at: 98, 224, 350 (same as checkpoints)
        """
        # Based on issue:
        # - Epoch 1 ends at step 98
        # - Epoch 2 ends at step 224 (difference of 126)
        # - Epoch 3 ends at step 350 (difference of 126)

        # This suggests:
        # - Epoch 1: 98 steps (initial datasets)
        # - Epoch 2+: 126 steps (with scheduled datasets)

        # Simulate epoch 1 configuration
        epoch_1_steps = 98
        epoch_2_steps = 126

        # Create validation for epoch 1
        validation = Validation.__new__(Validation)
        validation.config = SimpleNamespace(
            validation_epoch_interval=1,
            validation_step_interval=None,
            num_update_steps_per_epoch=epoch_1_steps,  # Correct for epoch 1
            gradient_accumulation_steps=1,
            epoch_batches_schedule={1: epoch_1_steps, 2: epoch_2_steps - epoch_1_steps},
        )
        validation._pending_epoch_validation = None
        validation._epoch_validations_completed = set()
        validation.deepspeed = False
        validation.accelerator = MagicMock()
        validation.accelerator.is_main_process = True
        validation.accelerator.num_processes = 1
        validation.global_step = 0
        validation.global_resume_step = 0
        validation.current_epoch = 1
        validation.current_epoch_step = 0
        prompts = [{"prompt": "test"}]

        validation_steps = []

        # Simulate epoch 1
        for step in range(1, epoch_1_steps + 1):
            validation.global_step = step
            validation.current_epoch_step = step
            validation.current_epoch = 1

            should_validate = validation.should_perform_intermediary_validation(
                step=step, validation_prompts=prompts, validation_type="intermediary"
            )

            if should_validate:
                validation_steps.append(step)
                validation._epoch_validations_completed.add(1)
                validation._pending_epoch_validation = None

        # Verify epoch 1 validation at step 98
        self.assertIn(
            98, validation_steps, f"Validation should trigger at step 98 (end of epoch 1). Got: {validation_steps}"
        )

        # Now simulate epoch 2 with updated steps per epoch
        validation.config.num_update_steps_per_epoch = epoch_2_steps
        validation._pending_epoch_validation = None

        for step_in_epoch in range(1, epoch_2_steps + 1):
            global_step = epoch_1_steps + step_in_epoch
            validation.global_step = global_step
            validation.current_epoch_step = step_in_epoch
            validation.current_epoch = 2

            should_validate = validation.should_perform_intermediary_validation(
                step=global_step, validation_prompts=prompts, validation_type="intermediary"
            )

            if should_validate:
                validation_steps.append(global_step)
                validation._epoch_validations_completed.add(2)
                validation._pending_epoch_validation = None

        # Verify epoch 2 validation at step 224 (98 + 126)
        self.assertIn(
            224, validation_steps, f"Validation should trigger at step 224 (end of epoch 2). Got: {validation_steps}"
        )

        # Verify we only validated at epoch boundaries
        self.assertEqual(
            validation_steps, [98, 224], f"Validation should only occur at epoch boundaries. Got: {validation_steps}"
        )


if __name__ == "__main__":
    unittest.main()
