import json
import os
import tempfile
import unittest
from pathlib import Path

try:
    from tests import test_setup
except ModuleNotFoundError:
    import test_setup  # noqa: F401

from simpletuner.helpers.training.error_reporter import (
    ERROR_FILE_ENV_VAR,
    cleanup_error_file,
    get_error_file_path,
    read_error,
    write_error,
)


class TestErrorReporter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.error_file = Path(self.temp_dir) / "error.json"
        self._orig_env = os.environ.get(ERROR_FILE_ENV_VAR)

    def tearDown(self):
        if self._orig_env is not None:
            os.environ[ERROR_FILE_ENV_VAR] = self._orig_env
        elif ERROR_FILE_ENV_VAR in os.environ:
            del os.environ[ERROR_FILE_ENV_VAR]
        cleanup_error_file(self.error_file)
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass

    def test_get_error_file_path_returns_none_when_not_set(self):
        if ERROR_FILE_ENV_VAR in os.environ:
            del os.environ[ERROR_FILE_ENV_VAR]
        self.assertIsNone(get_error_file_path())

    def test_get_error_file_path_returns_path_when_set(self):
        os.environ[ERROR_FILE_ENV_VAR] = str(self.error_file)
        result = get_error_file_path()
        self.assertEqual(result, self.error_file)

    def test_write_error_returns_false_when_no_path(self):
        if ERROR_FILE_ENV_VAR in os.environ:
            del os.environ[ERROR_FILE_ENV_VAR]
        try:
            raise ValueError("test error")
        except ValueError as e:
            result = write_error(e)
        self.assertFalse(result)

    def test_write_error_creates_file(self):
        os.environ[ERROR_FILE_ENV_VAR] = str(self.error_file)
        try:
            raise ValueError("test error message")
        except ValueError as e:
            result = write_error(e)
        self.assertTrue(result)
        self.assertTrue(self.error_file.exists())

    def test_write_error_contains_expected_fields(self):
        os.environ[ERROR_FILE_ENV_VAR] = str(self.error_file)
        try:
            raise FileNotFoundError("/path/to/file.txt")
        except FileNotFoundError as e:
            write_error(e)

        with open(self.error_file) as f:
            data = json.load(f)

        self.assertEqual(data["type"], "training.error")
        self.assertEqual(data["exception_type"], "FileNotFoundError")
        self.assertEqual(data["message"], "/path/to/file.txt")
        self.assertIn("timestamp", data)
        self.assertIn("traceback", data)
        self.assertIn("FileNotFoundError", data["traceback"])

    def test_write_error_with_custom_traceback(self):
        os.environ[ERROR_FILE_ENV_VAR] = str(self.error_file)
        try:
            raise RuntimeError("boom")
        except RuntimeError as e:
            write_error(e, traceback_str="custom traceback text")

        with open(self.error_file) as f:
            data = json.load(f)

        self.assertEqual(data["traceback"], "custom traceback text")

    def test_write_error_with_context(self):
        os.environ[ERROR_FILE_ENV_VAR] = str(self.error_file)
        try:
            raise RuntimeError("training failed")
        except RuntimeError as e:
            write_error(e, context={"step": 1000, "epoch": 5})

        with open(self.error_file) as f:
            data = json.load(f)

        self.assertEqual(data["context"]["step"], 1000)
        self.assertEqual(data["context"]["epoch"], 5)

    def test_read_error_returns_none_for_missing_file(self):
        result = read_error(self.error_file)
        self.assertIsNone(result)

    def test_read_error_returns_none_for_invalid_json(self):
        self.error_file.write_text("not valid json")
        result = read_error(self.error_file)
        self.assertIsNone(result)

    def test_read_error_returns_none_for_wrong_type(self):
        self.error_file.write_text('{"type": "other.event"}')
        result = read_error(self.error_file)
        self.assertIsNone(result)

    def test_read_error_returns_data_for_valid_error(self):
        os.environ[ERROR_FILE_ENV_VAR] = str(self.error_file)
        try:
            raise KeyError("missing_key")
        except KeyError as e:
            write_error(e)

        data = read_error(self.error_file)
        self.assertIsNotNone(data)
        self.assertEqual(data["type"], "training.error")
        self.assertEqual(data["exception_type"], "KeyError")

    def test_cleanup_error_file_removes_file(self):
        self.error_file.write_text("{}")
        self.assertTrue(self.error_file.exists())
        cleanup_error_file(self.error_file)
        self.assertFalse(self.error_file.exists())

    def test_cleanup_error_file_handles_missing_file(self):
        cleanup_error_file(self.error_file)  # Should not raise

    def test_cleanup_error_file_uses_env_var_when_no_path(self):
        os.environ[ERROR_FILE_ENV_VAR] = str(self.error_file)
        self.error_file.write_text("{}")
        cleanup_error_file()  # No path argument
        self.assertFalse(self.error_file.exists())

    def test_write_error_creates_parent_directories(self):
        nested_path = Path(self.temp_dir) / "nested" / "dir" / "error.json"
        os.environ[ERROR_FILE_ENV_VAR] = str(nested_path)
        try:
            raise ValueError("nested test")
        except ValueError as e:
            result = write_error(e)
        self.assertTrue(result)
        self.assertTrue(nested_path.exists())
        # Cleanup
        nested_path.unlink()
        nested_path.parent.rmdir()
        nested_path.parent.parent.rmdir()


if __name__ == "__main__":
    unittest.main()
