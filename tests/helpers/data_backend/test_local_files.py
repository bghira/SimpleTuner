import tempfile
import unittest
from pathlib import Path

from simpletuner.helpers.data_backend.local import LocalDataBackend


class TestLocalDataBackendListFiles(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.backend = LocalDataBackend(accelerator=None, id="test_local")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_file(self, relative_path: str):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
        return path

    def _flatten_listing(self, listing):
        return [file for _subdir, _dirs, files in listing for file in files]

    def test_list_files_does_not_duplicate_nested_files(self):
        files = [
            self._write_file("root.jpg"),
            self._write_file("a/one.JPG"),
            self._write_file("a/b/two.jpg"),
        ]
        self._write_file("a/b/ignored.txt")

        listing = self.backend.list_files(file_extensions=["jpg"], instance_data_dir=str(self.root))
        listed_files = self._flatten_listing(listing)

        self.assertCountEqual(listed_files, [str(path.absolute()) for path in files])
        self.assertEqual(len(listed_files), len(set(listed_files)))

    def test_list_files_follows_symlink_directories_without_cycles(self):
        target = self.root / "target"
        target.mkdir()
        file_path = self._write_file("target/image.jpg")
        symlink_path = self.root / "target" / "loop"
        try:
            symlink_path.symlink_to(self.root, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"Symlinks are not available: {exc}")

        listing = self.backend.list_files(file_extensions=["jpg"], instance_data_dir=str(self.root))
        listed_files = self._flatten_listing(listing)

        self.assertEqual(listed_files, [str(file_path.absolute())])

    def test_list_files_prunes_forbidden_directories(self):
        self._write_file("visible.jpg")
        self._write_file(".ipynb_checkpoints/hidden.jpg")

        listing = self.backend.list_files(file_extensions=["jpg"], instance_data_dir=str(self.root))
        listed_files = self._flatten_listing(listing)

        self.assertEqual(listed_files, [str((self.root / "visible.jpg").absolute())])


if __name__ == "__main__":
    unittest.main()
