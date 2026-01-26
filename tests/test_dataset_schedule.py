"""
Tests for dataset scheduling functions in schedule.py.

Covers:
- normalize_start_epoch / normalize_start_step
- normalize_end_epoch / normalize_end_step
- dataset_is_active with start and end conditions
"""

import unittest
from unittest.mock import MagicMock, patch

# Import at module level for simpler test code
try:
    from simpletuner.helpers.data_backend.runtime.schedule import (
        dataset_is_active,
        normalize_end_epoch,
        normalize_end_step,
        normalize_start_epoch,
        normalize_start_step,
    )
    from simpletuner.helpers.training.state_tracker import StateTracker

    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_ERROR = e
    normalize_start_epoch = None
    normalize_start_step = None
    normalize_end_epoch = None
    normalize_end_step = None
    dataset_is_active = None
    StateTracker = None


class TestNormalizeStartEpoch(unittest.TestCase):
    """Tests for normalize_start_epoch function."""

    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"Import failed: {IMPORT_ERROR}")

    def test_valid_int(self):
        """Valid integer >= 1 should be returned as-is."""
        self.assertEqual(normalize_start_epoch(5), 5)

    def test_valid_string_int(self):
        """Valid string integer should be converted."""
        self.assertEqual(normalize_start_epoch("3"), 3)

    def test_clamps_to_1_for_zero(self):
        """Zero should be clamped to 1."""
        self.assertEqual(normalize_start_epoch(0), 1)

    def test_clamps_to_1_for_negative(self):
        """Negative values should be clamped to 1."""
        self.assertEqual(normalize_start_epoch(-5), 1)

    def test_returns_1_for_none(self):
        """None should return default of 1."""
        self.assertEqual(normalize_start_epoch(None), 1)

    def test_returns_1_for_invalid_string(self):
        """Invalid string should return default of 1."""
        self.assertEqual(normalize_start_epoch("invalid"), 1)

    def test_float_is_truncated(self):
        """Float values should be truncated to int."""
        self.assertEqual(normalize_start_epoch(3.7), 3)


class TestNormalizeStartStep(unittest.TestCase):
    """Tests for normalize_start_step function."""

    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"Import failed: {IMPORT_ERROR}")

    def test_valid_int(self):
        """Valid integer >= 0 should be returned as-is."""
        self.assertEqual(normalize_start_step(100), 100)

    def test_valid_zero(self):
        """Zero is valid and should be returned as-is."""
        self.assertEqual(normalize_start_step(0), 0)

    def test_clamps_negative_to_zero(self):
        """Negative values should be clamped to 0."""
        self.assertEqual(normalize_start_step(-10), 0)

    def test_returns_0_for_none(self):
        """None should return default of 0."""
        self.assertEqual(normalize_start_step(None), 0)

    def test_returns_0_for_invalid_string(self):
        """Invalid string should return default of 0."""
        self.assertEqual(normalize_start_step("invalid"), 0)


class TestNormalizeEndEpoch(unittest.TestCase):
    """Tests for normalize_end_epoch function."""

    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"Import failed: {IMPORT_ERROR}")

    def test_valid_int(self):
        """Valid integer >= 1 should be returned as-is."""
        self.assertEqual(normalize_end_epoch(5), 5)

    def test_valid_string_int(self):
        """Valid string integer should be converted."""
        self.assertEqual(normalize_end_epoch("10"), 10)

    def test_none_returns_none(self):
        """None should return None (no end limit)."""
        self.assertIsNone(normalize_end_epoch(None))

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        self.assertIsNone(normalize_end_epoch(""))

    def test_zero_returns_none(self):
        """Zero should return None (no end limit)."""
        self.assertIsNone(normalize_end_epoch(0))

    def test_negative_returns_none(self):
        """Negative values should return None."""
        self.assertIsNone(normalize_end_epoch(-5))

    def test_invalid_string_returns_none(self):
        """Invalid string should return None."""
        self.assertIsNone(normalize_end_epoch("invalid"))

    def test_float_is_truncated(self):
        """Float values should be truncated to int."""
        self.assertEqual(normalize_end_epoch(3.7), 3)


class TestNormalizeEndStep(unittest.TestCase):
    """Tests for normalize_end_step function."""

    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"Import failed: {IMPORT_ERROR}")

    def test_valid_int(self):
        """Valid integer >= 1 should be returned as-is."""
        self.assertEqual(normalize_end_step(300), 300)

    def test_valid_string_int(self):
        """Valid string integer should be converted."""
        self.assertEqual(normalize_end_step("500"), 500)

    def test_none_returns_none(self):
        """None should return None (no end limit)."""
        self.assertIsNone(normalize_end_step(None))

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        self.assertIsNone(normalize_end_step(""))

    def test_zero_returns_none(self):
        """Zero should return None (no end limit)."""
        self.assertIsNone(normalize_end_step(0))

    def test_negative_returns_none(self):
        """Negative values should return None."""
        self.assertIsNone(normalize_end_step(-100))

    def test_invalid_string_returns_none(self):
        """Invalid string should return None."""
        self.assertIsNone(normalize_end_step("invalid"))


class TestDatasetIsActive(unittest.TestCase):
    """Tests for dataset_is_active function with start and end conditions."""

    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"Import failed: {IMPORT_ERROR}")

    def setUp(self):
        """Reset StateTracker before each test."""
        StateTracker.global_step = 0
        StateTracker.epoch = 1
        StateTracker.dataset_schedule = {}

    def _make_backend(self, backend_id, **config):
        """Create a mock backend with the given config.

        The backend is a dict where config fields are at the top level,
        matching how _extract_backend_config works.
        """
        result = {"id": backend_id}
        result.update(config)
        return result

    def _is_active(self, backend_id, backend, **kwargs):
        """Call dataset_is_active with the correct signature."""
        return dataset_is_active(backend_id, backend, **kwargs)

    def test_immediate_dataset_is_active(self):
        """Dataset with no scheduling should be active from the start."""
        backend = self._make_backend("test")
        is_active, _ = self._is_active("test", backend, epoch_hint=1, step_hint=1, update_state=False)
        self.assertTrue(is_active)

    def test_delayed_by_epoch_not_active_before_start(self):
        """Dataset with start_epoch=3 should not be active in epoch 1."""
        backend = self._make_backend("test", start_epoch=3)
        is_active, _ = self._is_active("test", backend, epoch_hint=1, step_hint=1, update_state=False)
        self.assertFalse(is_active)

    def test_delayed_by_epoch_active_at_start(self):
        """Dataset with start_epoch=3 should be active in epoch 3."""
        backend = self._make_backend("test", start_epoch=3)
        is_active, _ = self._is_active("test", backend, epoch_hint=3, step_hint=1, update_state=False)
        self.assertTrue(is_active)

    def test_delayed_by_step_not_active_before_start(self):
        """Dataset with start_step=100 should not be active at step 50."""
        backend = self._make_backend("test", start_step=100)
        is_active, _ = self._is_active("test", backend, epoch_hint=1, step_hint=49, update_state=False)
        self.assertFalse(is_active)

    def test_delayed_by_step_active_at_start(self):
        """Dataset with start_step=100 should be active at step 100."""
        backend = self._make_backend("test", start_step=100)
        is_active, _ = self._is_active("test", backend, epoch_hint=1, step_hint=99, update_state=False)
        self.assertTrue(is_active)

    # End condition tests
    def test_end_epoch_not_active_after_end(self):
        """Dataset with end_epoch=3 should not be active in epoch 4."""
        backend = self._make_backend("test", end_epoch=3)
        is_active, _ = self._is_active("test", backend, epoch_hint=4, step_hint=1, update_state=False)
        self.assertFalse(is_active)

    def test_end_epoch_active_at_end(self):
        """Dataset with end_epoch=3 should be active in epoch 3."""
        backend = self._make_backend("test", end_epoch=3)
        is_active, _ = self._is_active("test", backend, epoch_hint=3, step_hint=1, update_state=False)
        self.assertTrue(is_active)

    def test_end_step_not_active_after_end(self):
        """Dataset with end_step=300 should not be active at step 301."""
        backend = self._make_backend("test", end_step=300)
        is_active, _ = self._is_active("test", backend, epoch_hint=1, step_hint=300, update_state=False)
        self.assertFalse(is_active)

    def test_end_step_active_at_end(self):
        """Dataset with end_step=300 should be active at step 300."""
        backend = self._make_backend("test", end_step=300)
        is_active, _ = self._is_active("test", backend, epoch_hint=1, step_hint=299, update_state=False)
        self.assertTrue(is_active)

    def test_no_end_means_infinite(self):
        """Dataset with no end conditions should always be active after start."""
        backend = self._make_backend("test")
        is_active, _ = self._is_active("test", backend, epoch_hint=100, step_hint=10000, update_state=False)
        self.assertTrue(is_active)

    # Combined start and end tests
    def test_bounded_range_active_within(self):
        """Dataset with start_epoch=2 and end_epoch=5 should be active in epoch 3."""
        backend = self._make_backend("test", start_epoch=2, end_epoch=5)
        is_active, _ = self._is_active("test", backend, epoch_hint=3, step_hint=1, update_state=False)
        self.assertTrue(is_active)

    def test_bounded_range_not_active_before(self):
        """Dataset with start_epoch=2 and end_epoch=5 should not be active in epoch 1."""
        backend = self._make_backend("test", start_epoch=2, end_epoch=5)
        is_active, _ = self._is_active("test", backend, epoch_hint=1, step_hint=1, update_state=False)
        self.assertFalse(is_active)

    def test_bounded_range_not_active_after(self):
        """Dataset with start_epoch=2 and end_epoch=5 should not be active in epoch 6."""
        backend = self._make_backend("test", start_epoch=2, end_epoch=5)
        is_active, _ = self._is_active("test", backend, epoch_hint=6, step_hint=1, update_state=False)
        self.assertFalse(is_active)

    def test_step_bounded_range_active_within(self):
        """Dataset with start_step=100 and end_step=300 should be active at step 200."""
        backend = self._make_backend("test", start_step=100, end_step=300)
        is_active, _ = self._is_active("test", backend, epoch_hint=1, step_hint=199, update_state=False)
        self.assertTrue(is_active)

    def test_step_bounded_range_not_active_before(self):
        """Dataset with start_step=100 and end_step=300 should not be active at step 50."""
        backend = self._make_backend("test", start_step=100, end_step=300)
        is_active, _ = self._is_active("test", backend, epoch_hint=1, step_hint=49, update_state=False)
        self.assertFalse(is_active)

    def test_step_bounded_range_not_active_after(self):
        """Dataset with start_step=100 and end_step=300 should not be active at step 350."""
        backend = self._make_backend("test", start_step=100, end_step=300)
        is_active, _ = self._is_active("test", backend, epoch_hint=1, step_hint=349, update_state=False)
        self.assertFalse(is_active)


class TestCurriculumLearningScenarios(unittest.TestCase):
    """Integration tests for curriculum learning use cases."""

    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"Import failed: {IMPORT_ERROR}")

    def setUp(self):
        """Reset StateTracker before each test."""
        StateTracker.global_step = 0
        StateTracker.epoch = 1
        StateTracker.dataset_schedule = {}

    def _make_backend(self, backend_id, **config):
        """Create a mock backend with the given config.

        The backend is a dict where config fields are at the top level,
        matching how _extract_backend_config works.
        """
        result = {"id": backend_id}
        result.update(config)
        return result

    def _is_active(self, backend_id, backend, **kwargs):
        """Call dataset_is_active with the correct signature."""
        return dataset_is_active(backend_id, backend, **kwargs)

    def test_step_based_curriculum_handoff(self):
        """
        Curriculum learning scenario: low-res until step 300, then high-res.

        - lowres-512: immediate start, ends at step 300
        - highres-1024: starts at step 300
        """
        lowres = self._make_backend("lowres-512", end_step=300)
        highres = self._make_backend("highres-1024", start_step=300)

        # At step 100: only lowres should be active
        lowres_active, _ = self._is_active("lowres-512", lowres, epoch_hint=1, step_hint=99, update_state=False)
        highres_active, _ = self._is_active("highres-1024", highres, epoch_hint=1, step_hint=99, update_state=False)
        self.assertTrue(lowres_active, "Low-res should be active at step 100")
        self.assertFalse(highres_active, "High-res should not be active at step 100")

        # At step 300: both active (handoff point)
        lowres_active, _ = self._is_active("lowres-512", lowres, epoch_hint=1, step_hint=299, update_state=False)
        highres_active, _ = self._is_active("highres-1024", highres, epoch_hint=1, step_hint=299, update_state=False)
        self.assertTrue(lowres_active, "Low-res should be active at step 300")
        self.assertTrue(highres_active, "High-res should be active at step 300")

        # At step 400: only highres should be active
        lowres_active, _ = self._is_active("lowres-512", lowres, epoch_hint=1, step_hint=399, update_state=False)
        highres_active, _ = self._is_active("highres-1024", highres, epoch_hint=1, step_hint=399, update_state=False)
        self.assertFalse(lowres_active, "Low-res should not be active at step 400")
        self.assertTrue(highres_active, "High-res should be active at step 400")

    def test_epoch_based_curriculum_three_stages(self):
        """
        Three-stage curriculum based on epochs.

        - stage1: epochs 1-3
        - stage2: epochs 3-6
        - stage3: epochs 6+
        """
        stage1 = self._make_backend("stage1", end_epoch=3)
        stage2 = self._make_backend("stage2", start_epoch=3, end_epoch=6)
        stage3 = self._make_backend("stage3", start_epoch=6)

        # Epoch 1: only stage1
        self.assertTrue(self._is_active("stage1", stage1, epoch_hint=1, step_hint=1, update_state=False)[0])
        self.assertFalse(self._is_active("stage2", stage2, epoch_hint=1, step_hint=1, update_state=False)[0])
        self.assertFalse(self._is_active("stage3", stage3, epoch_hint=1, step_hint=1, update_state=False)[0])

        # Epoch 3: stage1 and stage2 overlap
        self.assertTrue(self._is_active("stage1", stage1, epoch_hint=3, step_hint=1, update_state=False)[0])
        self.assertTrue(self._is_active("stage2", stage2, epoch_hint=3, step_hint=1, update_state=False)[0])
        self.assertFalse(self._is_active("stage3", stage3, epoch_hint=3, step_hint=1, update_state=False)[0])

        # Epoch 5: only stage2
        self.assertFalse(self._is_active("stage1", stage1, epoch_hint=5, step_hint=1, update_state=False)[0])
        self.assertTrue(self._is_active("stage2", stage2, epoch_hint=5, step_hint=1, update_state=False)[0])
        self.assertFalse(self._is_active("stage3", stage3, epoch_hint=5, step_hint=1, update_state=False)[0])

        # Epoch 6: stage2 and stage3 overlap
        self.assertFalse(self._is_active("stage1", stage1, epoch_hint=6, step_hint=1, update_state=False)[0])
        self.assertTrue(self._is_active("stage2", stage2, epoch_hint=6, step_hint=1, update_state=False)[0])
        self.assertTrue(self._is_active("stage3", stage3, epoch_hint=6, step_hint=1, update_state=False)[0])

        # Epoch 8: only stage3
        self.assertFalse(self._is_active("stage1", stage1, epoch_hint=8, step_hint=1, update_state=False)[0])
        self.assertFalse(self._is_active("stage2", stage2, epoch_hint=8, step_hint=1, update_state=False)[0])
        self.assertTrue(self._is_active("stage3", stage3, epoch_hint=8, step_hint=1, update_state=False)[0])


if __name__ == "__main__":
    unittest.main()
