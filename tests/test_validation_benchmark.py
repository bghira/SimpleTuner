import os
import tempfile
import unittest
from types import SimpleNamespace

from simpletuner.helpers.training.validation import Validation


class ValidationBenchmarkTests(unittest.TestCase):
    def test_benchmark_exists_requires_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            validation = Validation.__new__(Validation)
            validation.config = SimpleNamespace(output_dir=tmpdir)

            self.assertFalse(validation.benchmark_exists())

            benchmark_dir = validation._benchmark_path()
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "benchmarks")))
            os.makedirs(benchmark_dir)
            self.assertFalse(validation.benchmark_exists())

            with open(os.path.join(benchmark_dir, "sample.png"), "wb") as handle:
                handle.write(b"placeholder")

            self.assertTrue(validation.benchmark_exists())


if __name__ == "__main__":
    unittest.main()
