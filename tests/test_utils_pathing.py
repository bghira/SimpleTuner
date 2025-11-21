import os
import unittest

from simpletuner.helpers.utils.pathing import normalize_data_path


class NormalizeDataPathTests(unittest.TestCase):
    def test_relative_without_root_preserves_relative_path(self):
        self.assertEqual(normalize_data_path("3.jpg"), os.path.normcase("3.jpg"))

    def test_relative_with_empty_root_preserves_relative_path(self):
        self.assertEqual(normalize_data_path("folder/3.jpg", ""), os.path.normcase("folder/3.jpg"))

    def test_absolute_without_root_remains_absolute(self):
        path = "/tmp/example.png"
        self.assertEqual(normalize_data_path(path), os.path.normcase(os.path.normpath(path)))

    def test_relative_with_root_returns_relative_to_root(self):
        result = normalize_data_path("subdir/img.png", "/dataset/root")
        self.assertEqual(result, os.path.normcase("subdir/img.png"))


if __name__ == "__main__":
    unittest.main()
