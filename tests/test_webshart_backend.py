import unittest
from pathlib import Path

from simpletuner.helpers.data_backend.webshart import WebshartDataBackend


class TestWebshartDataBackend(unittest.TestCase):
    def test_sample_id_round_trips_nested_filenames(self):
        sample_id = WebshartDataBackend.sample_id(2, 17, "nested/path/sample.webp")

        self.assertEqual(sample_id, "webshart://2/17/nested/path/sample.webp")
        ref = WebshartDataBackend.parse_sample_id(sample_id)
        self.assertEqual(ref.shard_idx, 2)
        self.assertEqual(ref.sample_idx, 17)
        self.assertEqual(ref.filename, "nested/path/sample.webp")

    def test_parse_sample_id_rejects_non_webshart_identifier(self):
        with self.assertRaises(ValueError):
            WebshartDataBackend.parse_sample_id(Path("/tmp/sample.webp"))


if __name__ == "__main__":
    unittest.main()
