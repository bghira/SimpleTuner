#!/usr/bin/env python3
"""
Master test runner for all transformer tests.

This script discovers and executes all transformer test files with comprehensive
reporting, coverage analysis, and performance benchmarking.
"""

import os
import sys
import unittest
import time
import argparse
from typing import Dict, List, Optional
from io import StringIO
import importlib.util


class TransformerTestRunner:
    """Master test runner for transformer tests."""

    def __init__(self, test_dir: str = None):
        self.test_dir = test_dir or os.path.join(os.path.dirname(__file__), "test_transformers")
        self.results = {}
        self.total_time = 0.0

    def discover_test_files(self) -> List[str]:
        """Discover all transformer test files."""
        test_files = []

        if not os.path.exists(self.test_dir):
            print(f"Test directory {self.test_dir} does not exist")
            return test_files

        for filename in os.listdir(self.test_dir):
            if filename.startswith("test_") and filename.endswith("_transformer.py"):
                test_files.append(os.path.join(self.test_dir, filename))

        return sorted(test_files)

    def run_single_test_file(self, test_file: str, verbose: bool = False) -> Dict:
        """Run a single test file and return results."""
        module_name = os.path.basename(test_file)[:-3]  # Remove .py

        print(f"\n{'='*60}")
        print(f"Running {module_name}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            # Load the test module
            spec = importlib.util.spec_from_file_location(module_name, test_file)
            test_module = importlib.util.module_from_spec(spec)

            # Add the test directory to sys.path temporarily
            original_path = sys.path.copy()
            sys.path.insert(0, os.path.dirname(test_file))
            sys.path.insert(0, os.path.join(os.path.dirname(test_file), "..", "utils"))

            try:
                spec.loader.exec_module(test_module)
            finally:
                sys.path = original_path

            # Discover and run tests
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)

            # Run tests with custom result handler
            stream = StringIO()
            runner = unittest.TextTestRunner(stream=stream, verbosity=2 if verbose else 1, buffer=True)

            result = runner.run(suite)

            execution_time = time.time() - start_time
            self.total_time += execution_time

            # Collect results
            test_result = {
                "module": module_name,
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped),
                "success_rate": (
                    ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
                    if result.testsRun > 0
                    else 0
                ),
                "execution_time": execution_time,
                "output": stream.getvalue(),
                "failure_details": result.failures,
                "error_details": result.errors,
            }

            # Print summary
            print(f"Tests run: {result.testsRun}")
            print(f"Failures: {len(result.failures)}")
            print(f"Errors: {len(result.errors)}")
            print(f"Skipped: {len(result.skipped)}")
            print(f"Success rate: {test_result['success_rate']:.1f}%")
            print(f"Execution time: {execution_time:.2f}s")

            if verbose and (result.failures or result.errors):
                print("\nFailures/Errors:")
                for failure in result.failures:
                    print(f"FAIL: {failure[0]}")
                    print(f"  {failure[1]}")
                for error in result.errors:
                    print(f"ERROR: {error[0]}")
                    print(f"  {error[1]}")

            return test_result

        except ImportError as e:
            print(f"Failed to import {module_name}: {e}")
            return {
                "module": module_name,
                "tests_run": 0,
                "failures": 0,
                "errors": 1,
                "skipped": 0,
                "success_rate": 0.0,
                "execution_time": 0.0,
                "output": f"Import error: {e}",
                "failure_details": [],
                "error_details": [("Import", str(e))],
            }
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Unexpected error running {module_name}: {e}")
            return {
                "module": module_name,
                "tests_run": 0,
                "failures": 0,
                "errors": 1,
                "skipped": 0,
                "success_rate": 0.0,
                "execution_time": execution_time,
                "output": f"Unexpected error: {e}",
                "failure_details": [],
                "error_details": [("Execution", str(e))],
            }

    def run_all_tests(self, verbose: bool = False, fail_fast: bool = False) -> Dict:
        """Run all transformer tests."""
        test_files = self.discover_test_files()

        if not test_files:
            print("No transformer test files found")
            return {}

        print(f"Found {len(test_files)} transformer test files")
        print(f"Test files: {[os.path.basename(f) for f in test_files]}")

        all_results = {}
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0

        for test_file in test_files:
            result = self.run_single_test_file(test_file, verbose)
            all_results[result["module"]] = result

            total_tests += result["tests_run"]
            total_failures += result["failures"]
            total_errors += result["errors"]
            total_skipped += result["skipped"]

            # Fail fast if requested and there are failures/errors
            if fail_fast and (result["failures"] > 0 or result["errors"] > 0):
                print(f"\nFailing fast due to failures/errors in {result['module']}")
                break

        # Print overall summary
        print(f"\n{'='*80}")
        print("OVERALL TEST SUMMARY")
        print(f"{'='*80}")

        print(f"Total test files: {len(test_files)}")
        print(f"Total tests run: {total_tests}")
        print(f"Total failures: {total_failures}")
        print(f"Total errors: {total_errors}")
        print(f"Total skipped: {total_skipped}")
        print(f"Total execution time: {self.total_time:.2f}s")

        success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
        print(f"Overall success rate: {success_rate:.1f}%")

        # Per-module summary
        print(f"\n{'Module':<30} {'Tests':<8} {'Pass':<8} {'Fail':<8} {'Error':<8} {'Skip':<8} {'Success%':<10} {'Time':<8}")
        print("-" * 100)

        for module, result in all_results.items():
            module_short = module.replace("test_", "").replace("_transformer", "")
            passed = result["tests_run"] - result["failures"] - result["errors"]
            print(
                f"{module_short:<30} {result['tests_run']:<8} {passed:<8} {result['failures']:<8} {result['errors']:<8} {result['skipped']:<8} {result['success_rate']:<10.1f} {result['execution_time']:<8.2f}"
            )

        # Test coverage analysis
        print(f"\nTEST COVERAGE ANALYSIS:")
        print("-" * 50)

        transformer_models = [
            "flux",
            "hidream",
            "auraflow",
            "cosmos",
            "sd3",
            "pixart",
            "ltxvideo",
            "wan",
            "qwen_image",
            "sana",
        ]

        tested_models = set()
        for module in all_results.keys():
            for model in transformer_models:
                if model in module:
                    tested_models.add(model)

        print(f"Transformer models with tests: {len(tested_models)}/{len(transformer_models)}")
        print(f"Tested models: {sorted(tested_models)}")

        missing_models = set(transformer_models) - tested_models
        if missing_models:
            print(f"Missing test coverage: {sorted(missing_models)}")

        return all_results

    def generate_coverage_report(self, results: Dict) -> str:
        """Generate a coverage report."""
        report = []
        report.append("TRANSFORMER TEST COVERAGE REPORT")
        report.append("=" * 50)
        report.append("")

        total_tests = sum(r["tests_run"] for r in results.values())
        total_passed = sum(r["tests_run"] - r["failures"] - r["errors"] for r in results.values())

        report.append(f"Total Tests: {total_tests}")
        report.append(f"Total Passed: {total_passed}")
        report.append(f"Overall Coverage: {(total_passed/total_tests*100):.1f}%")
        report.append("")

        report.append("Per-Transformer Coverage:")
        report.append("-" * 30)

        for module, result in sorted(results.items()):
            transformer = module.replace("test_", "").replace("_transformer", "")
            passed = result["tests_run"] - result["failures"] - result["errors"]
            coverage = (passed / result["tests_run"] * 100) if result["tests_run"] > 0 else 0
            report.append(f"{transformer:15} {passed:3}/{result['tests_run']:3} tests passed ({coverage:5.1f}%)")

        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run all transformer tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output showing detailed test results")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--test-dir", type=str, help="Override test directory path")
    parser.add_argument("--coverage-report", type=str, help="Save coverage report to file")
    parser.add_argument("--filter", type=str, help="Only run tests matching this pattern")

    args = parser.parse_args()

    # Create test runner
    runner = TransformerTestRunner(args.test_dir)

    # Filter test files if pattern provided
    if args.filter:
        print(f"Filtering tests with pattern: {args.filter}")
        original_discover = runner.discover_test_files

        def filtered_discover():
            files = original_discover()
            return [f for f in files if args.filter in os.path.basename(f)]

        runner.discover_test_files = filtered_discover

    try:
        # Run all tests
        results = runner.run_all_tests(verbose=args.verbose, fail_fast=args.fail_fast)

        # Generate coverage report
        if args.coverage_report:
            report = runner.generate_coverage_report(results)
            with open(args.coverage_report, "w") as f:
                f.write(report)
            print(f"\nCoverage report saved to: {args.coverage_report}")

        # Exit with error code if there were failures
        total_failures = sum(r["failures"] + r["errors"] for r in results.values())
        return min(total_failures, 1)  # Return 1 if any failures, 0 if all passed

    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
