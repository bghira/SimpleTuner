#!/usr/bin/env python3
"""
Runner for process_keeper tests that ensures spawn mode is set correctly.
"""

# MUST set spawn before ANY other imports
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)
print(f"Set multiprocessing start method to: {multiprocessing.get_start_method()}")

import sys

# Now we can import the test
import unittest

# Import test module
from test_process_keeper import TestProcessLifecycle

if __name__ == "__main__":
    # Run specific test
    if len(sys.argv) > 1:
        # Run specific test method
        suite = unittest.TestLoader().loadTestsFromName(sys.argv[1], module=TestProcessLifecycle)
    else:
        # Run all tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestProcessLifecycle)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
