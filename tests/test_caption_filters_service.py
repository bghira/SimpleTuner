import tempfile
import unittest
from pathlib import Path

from simpletuner.simpletuner_sdk.server.services.caption_filters_service import (
    CaptionFilterError,
    CaptionFiltersService,
)


class CaptionFiltersServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        base_dir = Path(self._tmpdir.name)
        filters_dir = base_dir / "caption_filters"
        self.service = CaptionFiltersService(filters_dir=filters_dir, base_dir=base_dir)
        self.filters_dir = filters_dir

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_create_and_list_filter(self) -> None:
        record = self.service.create_filter(
            {
                "name": "nsfw",
                "label": "NSFW",
                "description": "Remove common nsfw tokens",
                "entries": ["nsfw", "nudity", "s/cat/dog/"],
            }
        )

        self.assertEqual(record.name, "nsfw")
        self.assertTrue((self.filters_dir / "nsfw.json").exists())

        items = self.service.list_filters()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].name, "nsfw")
        self.assertEqual(items[0].entries, ["nsfw", "nudity", "s/cat/dog/"])
        self.assertIn("caption_filters/nsfw.json", items[0].path)

    def test_update_filter_allows_rename(self) -> None:
        self.service.create_filter({"name": "baseline", "entries": ["foo"]})
        updated = self.service.update_filter(
            "baseline",
            {
                "name": "clean",
                "label": "Clean captions",
                "description": "remove foo entries",
                "entries": ["foo", "bar"],
            },
        )

        self.assertEqual(updated.name, "clean")
        self.assertTrue((self.filters_dir / "clean.json").exists())
        self.assertFalse((self.filters_dir / "baseline.json").exists())
        self.assertEqual(updated.entries, ["foo", "bar"])
        self.assertEqual(updated.label, "Clean captions")

    def test_delete_filter(self) -> None:
        self.service.create_filter({"name": "remove", "entries": ["tmp"]})
        self.service.delete_filter("remove")
        self.assertFalse((self.filters_dir / "remove.json").exists())
        with self.assertRaises(CaptionFilterError):
            self.service.get_filter("remove")

    def test_duplicate_filter_raises(self) -> None:
        self.service.create_filter({"name": "duplicate", "entries": ["one"]})
        with self.assertRaises(CaptionFilterError) as ctx:
            self.service.create_filter({"name": "duplicate", "entries": ["two"]})
        self.assertEqual(ctx.exception.status_code, 409)

    def test_apply_filters_matches_expected_behaviour(self) -> None:
        entries = ["nsfw", "s/cat/dog/", r"\\d+"]
        sample = "nsfw cat 123 caption"
        result = self.service.test_entries(entries, sample)
        self.assertIn("dog", result)
        self.assertNotIn("nsfw", result)
        self.assertNotRegex(result, r"\\d+")


if __name__ == "__main__":
    unittest.main()
