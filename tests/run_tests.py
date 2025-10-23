#!/usr/bin/env python
"""
Test runner for SimpleTuner unittest tests.
Usage: python tests/run_tests.py [test_module]
"""

import logging
import os
import sys
import unittest

# Configure logging before importing tests
import test_setup


def run_tests(pattern=None):
    """Run unittest tests with proper configuration."""

    # Discover and load tests
    loader = unittest.TestLoader()

    if pattern:
        # Run specific test module or pattern
        suite = loader.loadTestsFromName(pattern)
    else:
        # Discover all tests
        start_dir = os.path.dirname(os.path.abspath(__file__))
        suite = loader.discover(start_dir, pattern="test_*.py")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    # Get test pattern from command line if provided
    pattern = sys.argv[1] if len(sys.argv) > 1 else None

    # Run tests
    exit_code = run_tests(pattern)
    sys.exit(exit_code)
