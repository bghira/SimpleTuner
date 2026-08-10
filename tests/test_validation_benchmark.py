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

    def test_benchmark_exists_uses_requested_benchmark_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            validation = Validation.__new__(Validation)
            validation.config = SimpleNamespace(output_dir=tmpdir)
            base_model_dir = validation._benchmark_path("base_model")
            custom_dir = validation._benchmark_path("custom")
            os.makedirs(base_model_dir)
            os.makedirs(custom_dir)
            with open(os.path.join(custom_dir, "sample.png"), "wb") as handle:
                handle.write(b"placeholder")

            self.assertFalse(validation.benchmark_exists("base_model"))
            self.assertTrue(validation.benchmark_exists("custom"))

    def test_benchmark_exists_treats_file_as_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            validation = Validation.__new__(Validation)
            validation.config = SimpleNamespace(output_dir=tmpdir)
            benchmarks_dir = os.path.join(tmpdir, "benchmarks")
            os.makedirs(benchmarks_dir)
            with open(os.path.join(benchmarks_dir, "base_model"), "wb") as handle:
                handle.write(b"placeholder")

            self.assertFalse(validation.benchmark_exists())


if __name__ == "__main__":
    unittest.main()
