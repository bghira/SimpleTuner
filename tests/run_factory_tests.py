#!/usr/bin/env python
"""Run factory and provider integration tests from a checkout.

Use python tests/run_factory_tests.py, optionally followed by unittest names.
"""

import subprocess
import sys
from pathlib import Path

DEFAULT_SUITES = (
    "tests.test_full_integration",
    "tests.test_factory_edge_cases",
    "tests.test_factory_caption_backend",
    "tests.test_factory_distiller_requirements",
    "tests.test_backend_builders",
    "tests.test_dataset_runtime",
    "tests.test_aws_backend",
    "tests.test_huggingface_metadata_backend",
    "tests.test_webshart_parquet_backend",
)


if __name__ == "__main__":
    suite = sys.argv[1:] or DEFAULT_SUITES
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "unittest", "-v", "-f", *suite],
            cwd=Path(__file__).resolve().parents[1],
        )
    )
