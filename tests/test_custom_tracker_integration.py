import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from simpletuner.helpers.training.trainer import Trainer
except ImportError:
    # Try to adjust path if simpletuner is not found directly
    sys.path.insert(0, os.getcwd())
    from simpletuner.helpers.training.trainer import Trainer


class TestCustomTracker(unittest.TestCase):
    def test_load_custom_tracker(self):
        # Create a dummy Trainer instance without calling real __init__
        trainer = Trainer.__new__(Trainer)

        # Test valid tracker
        # Ensure test_tracker.py exists in simpletuner/custom-trackers
        tracker = trainer._load_custom_tracker("test_tracker", "test_run", None)
        self.assertEqual(tracker.name, "my_tracker")
        self.assertEqual(tracker.run_name, "test_run")

        # Test logging
        tracker.log({"loss": 0.1}, step=1)
        # The mock tracker appends to _records
        self.assertEqual(tracker.tracker[0], {"step": 1, "loss": 0.1})

    def test_invalid_tracker_name(self):
        trainer = Trainer.__new__(Trainer)
        with self.assertRaises(ValueError):
            trainer._load_custom_tracker("invalid/name", "run", None)

    def test_missing_tracker(self):
        trainer = Trainer.__new__(Trainer)
        with self.assertRaises(FileNotFoundError):
            trainer._load_custom_tracker("non_existent_tracker", "run", None)


if __name__ == "__main__":
    unittest.main()
