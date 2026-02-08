import types
import unittest
from pathlib import Path
from unittest.mock import patch

import simpletuner


class GetPackageDirTestCase(unittest.TestCase):
    def test_returns_path_when_file_is_set(self):
        result = simpletuner._get_package_dir()
        self.assertIsInstance(result, Path)
        self.assertTrue(result.is_dir())
        self.assertEqual(result, Path(simpletuner.__file__).parent)

    def test_returns_path_when_file_is_none(self):
        original = simpletuner.__file__
        try:
            simpletuner.__file__ = None
            result = simpletuner._get_package_dir()
            self.assertIsInstance(result, Path)
            self.assertTrue(result.is_dir())
            # Should still resolve to the same package directory
            self.assertEqual(result.resolve(), Path(original).parent.resolve())
        finally:
            simpletuner.__file__ = original

    def test_raises_when_package_unresolvable(self):
        original = simpletuner.__file__
        fake_spec = types.SimpleNamespace(origin=None, submodule_search_locations=None)
        try:
            simpletuner.__file__ = None
            with patch("importlib.util.find_spec", return_value=fake_spec):
                with self.assertRaises(RuntimeError):
                    simpletuner._get_package_dir()
        finally:
            simpletuner.__file__ = original


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
