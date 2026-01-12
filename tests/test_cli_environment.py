import io
import os
import tempfile
import unittest
from pathlib import Path

from simpletuner.cli.common import _validate_environment_config
from simpletuner.cli.train import run_training
from simpletuner.helpers.configuration.json_file import load_json_config


class TestCliEnvironmentValidation(unittest.TestCase):
    def setUp(self) -> None:
        self._env_snapshot = os.environ.copy()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_snapshot)

    def test_validate_environment_config_missing_env(self) -> None:
        with self.assertRaises(FileNotFoundError):
            _validate_environment_config("this-env-should-not-exist", None, None)

    def test_validate_environment_config_existing_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text("{}", encoding="utf-8")
            try:
                _validate_environment_config(tmpdir, None, None)
            except FileNotFoundError as error:
                self.fail(f"Expected environment to be valid but raised: {error}")

    def test_run_training_returns_error_for_missing_env(self) -> None:
        with unittest.mock.patch("sys.stdout", new=io.StringIO()):
            result = run_training(env="this-env-should-not-exist", extra_args=[])
        self.assertEqual(result, 1)

    def test_load_json_config_raises_for_missing_env(self) -> None:
        os.environ["ENV"] = "missing-json-env"
        with self.assertRaises(ValueError) as ctx:
            load_json_config()

        self.assertIn("missing-json-env", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
