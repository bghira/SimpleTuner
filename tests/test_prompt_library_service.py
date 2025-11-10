import tempfile
import unittest
from pathlib import Path

from fastapi import status

from simpletuner.simpletuner_sdk.server.services.prompt_library_service import PromptLibraryError, PromptLibraryService


class PromptLibraryServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self._tmpdir.name)
        self.service = PromptLibraryService(config_dir=self.base_dir)
        self.libs_dir = self.base_dir / "validation_prompt_libraries"
        self.libs_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_save_and_list_libraries(self) -> None:
        entries = {"hello": "Hello prompt", "bye": "Goodbye prompt"}
        result = self.service.save_library("user_prompt_library-test.json", entries)

        self.assertEqual(result["library"].filename, "user_prompt_library-test.json")
        records = self.service.list_libraries()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.prompt_count, len(entries))
        self.assertEqual(record.relative_path, "validation_prompt_libraries/user_prompt_library-test.json")

        payload = self.service.read_library("user_prompt_library-test.json")
        self.assertEqual(payload["entries"], entries)

    def test_rename_library_replaces_old_file(self) -> None:
        first_entries = {"a": "first"}
        self.service.save_library("user_prompt_library-old.json", first_entries)
        second_entries = {"b": "second"}
        self.service.save_library(
            "user_prompt_library-new.json",
            second_entries,
            previous_filename="user_prompt_library-old.json",
        )

        self.assertTrue((self.libs_dir / "user_prompt_library-new.json").exists())
        self.assertFalse((self.libs_dir / "user_prompt_library-old.json").exists())

    def test_invalid_filename_raises(self) -> None:
        with self.assertRaises(PromptLibraryError) as ctx:
            self.service.save_library("invalid name.json", {"test": "value"})
        self.assertEqual(ctx.exception.status_code, status.HTTP_400_BAD_REQUEST)

    def test_read_missing_library_raises(self) -> None:
        with self.assertRaises(PromptLibraryError) as ctx:
            self.service.read_library("user_prompt_library-missing.json")
        self.assertEqual(ctx.exception.status_code, status.HTTP_404_NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
