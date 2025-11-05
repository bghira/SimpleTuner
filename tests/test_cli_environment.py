import os
import unittest

from simpletuner.cli import _validate_environment_config, run_training
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
        try:
            _validate_environment_config("sdxl", None, None)
        except FileNotFoundError as error:
            self.fail(f"Expected environment to be valid but raised: {error}")

    def test_run_training_returns_error_for_missing_env(self) -> None:
        result = run_training(env="this-env-should-not-exist", extra_args=[])
        self.assertEqual(result, 1)

    def test_load_json_config_raises_for_missing_env(self) -> None:
        os.environ["ENV"] = "missing-json-env"
        with self.assertRaises(ValueError) as ctx:
            load_json_config()

        self.assertIn("missing-json-env", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
