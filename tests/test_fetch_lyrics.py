import unittest
from pathlib import Path

from scripts.datasets.fetch_lyrics import metadata_from_numbered_filename, prefer_filename_metadata


class TestFetchLyricsMetadata(unittest.TestCase):
    def test_metadata_from_numbered_filename(self):
        artist, title = metadata_from_numbered_filename(Path("001 - Led Zeppelin - Stairway To Heaven.mp3"))

        self.assertEqual("Led Zeppelin", artist)
        self.assertEqual("Stairway To Heaven", title)

    def test_filename_metadata_replaces_compilation_tags(self):
        artist, title = prefer_filename_metadata(
            Path("001 - Led Zeppelin - Stairway To Heaven.mp3"),
            "Various Artists",
            "001 - Led Zeppelin - Stairway To Heaven",
        )

        self.assertEqual("Led Zeppelin", artist)
        self.assertEqual("Stairway To Heaven", title)

    def test_filename_metadata_does_not_replace_specific_tags(self):
        artist, title = prefer_filename_metadata(
            Path("001 - Led Zeppelin - Stairway To Heaven.mp3"),
            "Led Zeppelin",
            "Stairway To Heaven",
        )

        self.assertEqual("Led Zeppelin", artist)
        self.assertEqual("Stairway To Heaven", title)


if __name__ == "__main__":
    unittest.main()
