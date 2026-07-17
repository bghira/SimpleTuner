import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

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

    def test_read_image_batch_preserves_order_and_skips_failures(self):
        paths = ["first.jpg", "bad.jpg", "second.jpg"]

        def read_image(filepath, delete_problematic_images=False):
            if filepath == "bad.jpg":
                raise ValueError("corrupt")
            return f"image:{filepath}"

        self.backend.read_image = Mock(side_effect=read_image)

        keys, images = self.backend.read_image_batch(paths)

        self.assertEqual(keys, ["first.jpg", "second.jpg"])
        self.assertEqual(images, ["image:first.jpg", "image:second.jpg"])

    def test_read_image_batch_deletes_problematic_images_when_requested(self):
        self.backend.read_image = Mock(side_effect=ValueError("corrupt"))
        self.backend.delete = Mock()

        keys, images = self.backend.read_image_batch(["bad.jpg"], delete_problematic_images=True)

        self.assertEqual(keys, [])
        self.assertEqual(images, [])
        self.backend.delete.assert_called_once_with("bad.jpg")

    def test_write_batch_raises_worker_errors(self):
        def write(filepath, data):
            if filepath == "bad.pt":
                raise OSError("disk full")

        self.backend.write = Mock(side_effect=write)

        with self.assertRaises(OSError):
            self.backend.write_batch(["ok.pt", "bad.pt"], [b"ok", b"bad"])


if __name__ == "__main__":
    unittest.main()
