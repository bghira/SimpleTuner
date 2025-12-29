#!/usr/bin/env python
"""Run tests by category for faster CI.

Usage:
    python -m tests.run_by_category unit        # Fast unit tests
    python -m tests.run_by_category integration # Integration tests
    python -m tests.run_by_category all         # All tests

Categories are determined by:
- Unit tests: Don't require external services, use mocking
- Integration tests: Test actual integrations, may require GPU/services
"""

import argparse
import sys
import unittest
from pathlib import Path

# Test categorization based on file patterns
UNIT_TEST_PATTERNS = [
    "test_config*.py",
    "test_state*.py",
    "test_cropping*.py",
    "test_sampler*.py",
    "test_helpers*.py",
    "test_lycoris*.py",
    "test_path*.py",
    "test_webhook*.py",
    "test_cloud*.py",
    "test_checkpoint*.py",
    "test_caption*.py",
    "test_logging*.py",
    "test_model_card*.py",
    "test_dataset_plan*.py",
    "test_lora*.py",
    "test_backend_config*.py",
    "test_fsdp*.py",
    "test_webui*.py",
    "test_version*.py",
    "test_example*.py",
    # Cloud-specific tests (split by domain)
    "test_job_store.py",
    "test_cost_and_pricing.py",
]

INTEGRATION_TEST_PATTERNS = [
    "test_*_integration*.py",
    "test_full_integration*.py",
    "test_api_integration*.py",
    "test_transformer_integration*.py",
    "test_pipelines/*.py",
    "test_transformers/*.py",
    "test_e2e*.py",
    "end_to_end*.py",
]

# Tests that are slow and should be skipped in quick runs
SLOW_TEST_PATTERNS = [
    "test_pipelines/*.py",
    "test_transformers/*.py",
    "benchmark*.py",
]


def discover_tests(patterns: list, test_dir: Path) -> unittest.TestSuite:
    """Discover tests matching the given patterns."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for pattern in patterns:
        if "/" in pattern:
            # Subdirectory pattern
            subdir, file_pattern = pattern.rsplit("/", 1)
            sub_path = test_dir / subdir
            if sub_path.exists():
                discovered = loader.discover(str(sub_path), pattern=file_pattern)
                suite.addTests(discovered)
        else:
            # Top-level pattern
            discovered = loader.discover(str(test_dir), pattern=pattern)
            suite.addTests(discovered)

    return suite


def main():
    parser = argparse.ArgumentParser(description="Run tests by category")
    parser.add_argument(
        "category",
        choices=["unit", "integration", "slow", "all", "quick"],
        help="Test category to run",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Verbosity level",
    )
    parser.add_argument(
        "-f",
        "--failfast",
        action="store_true",
        help="Stop on first failure",
    )

    args = parser.parse_args()

    test_dir = Path(__file__).parent

    if args.category == "unit":
        suite = discover_tests(UNIT_TEST_PATTERNS, test_dir)
        print(f"Running unit tests...")
    elif args.category == "integration":
        suite = discover_tests(INTEGRATION_TEST_PATTERNS, test_dir)
        print(f"Running integration tests...")
    elif args.category == "slow":
        suite = discover_tests(SLOW_TEST_PATTERNS, test_dir)
        print(f"Running slow tests (GPU/model tests)...")
    elif args.category == "quick":
        # Unit tests excluding slow ones
        unit_suite = discover_tests(UNIT_TEST_PATTERNS, test_dir)
        slow_suite = discover_tests(SLOW_TEST_PATTERNS, test_dir)

        # Filter out slow tests
        slow_names = set()
        for test in slow_suite:
            if hasattr(test, "__iter__"):
                for t in test:
                    slow_names.add(str(t))
            else:
                slow_names.add(str(test))

        suite = unittest.TestSuite()
        for test in unit_suite:
            if str(test) not in slow_names:
                suite.addTest(test)
        print(f"Running quick tests (unit, no GPU)...")
    else:  # all
        suite = discover_tests(["test*.py"], test_dir)
        print(f"Running all tests...")

    runner = unittest.TextTestRunner(
        verbosity=args.verbose,
        failfast=args.failfast,
    )

    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
