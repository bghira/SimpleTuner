import unittest

from simpletuner.helpers.data_backend.filters import DatasetFilter, build_dataset_filter


class TestDatasetFilter(unittest.TestCase):
    def test_top_level_filter_takes_precedence_over_nested_huggingface_filter(self):
        dataset_filter = build_dataset_filter(
            {
                "filter_func": {"path": {"include": ["keep"]}},
                "huggingface": {"filter_func": {"path": {"include": ["drop"]}}},
            }
        )

        self.assertIsNotNone(dataset_filter)
        self.assertTrue(dataset_filter.matches_path("/data/keep/sample.png"))
        self.assertFalse(dataset_filter.matches_path("/data/drop/sample.png"))

    def test_flat_path_filter_takes_precedence_over_nested_path_filter(self):
        dataset_filter = DatasetFilter(
            {
                "path": {"include": ["drop"]},
                "path_include": ["keep"],
            }
        )

        self.assertTrue(dataset_filter.matches_path("/data/keep/sample.png"))
        self.assertFalse(dataset_filter.matches_path("/data/drop/sample.png"))

    def test_nested_huggingface_filter_remains_supported(self):
        dataset_filter = build_dataset_filter({"huggingface": {"filter_func": {"collection": ["photo"]}}})

        self.assertIsNotNone(dataset_filter)
        self.assertTrue(dataset_filter.matches_item({"collection": "photo"}))
        self.assertFalse(dataset_filter.matches_item({"collection": "art"}))

    def test_collection_filter_supports_list_valued_rows(self):
        dataset_filter = DatasetFilter({"collection": ["photo", "artwork"]})

        self.assertTrue(dataset_filter.matches_item({"collection": ["scan", "photo"]}))
        self.assertFalse(dataset_filter.matches_item({"collection": ["scan", "diagram"]}))

    def test_path_filter_supports_contains_include_and_exclude(self):
        dataset_filter = DatasetFilter({"path": {"include": ["regularization"], "exclude": ["bad"]}})

        self.assertTrue(dataset_filter.matches_path("/data/regularization/good.png"))
        self.assertFalse(dataset_filter.matches_path("/data/train/good.png"))
        self.assertFalse(dataset_filter.matches_path("/data/regularization/bad.png"))

    def test_path_filter_supports_glob_mode(self):
        dataset_filter = DatasetFilter({"path": {"mode": "glob", "include": ["*/clothing/*.jpg"]}})

        self.assertTrue(dataset_filter.matches_path("/data/clothing/a.jpg"))
        self.assertFalse(dataset_filter.matches_path("/data/clothing/a.png"))

    def test_flat_path_filter_auto_mode_uses_fast_literals_and_wildcards(self):
        dataset_filter = DatasetFilter({"path_include": ["regularization", "*/curated/*.jpg"], "path_exclude": ["bad"]})

        self.assertTrue(dataset_filter.matches_path("/data/regularization/good.png"))
        self.assertTrue(dataset_filter.matches_path("/data/train/curated/sample.jpg"))
        self.assertFalse(dataset_filter.matches_path("/data/train/curated/sample.png"))
        self.assertFalse(dataset_filter.matches_path("/data/regularization/bad.png"))

    def test_flat_path_filter_auto_mode_supports_regex_prefix(self):
        dataset_filter = DatasetFilter({"path_include": ["re:subject-[0-9]+\\.png"]})

        self.assertTrue(dataset_filter.matches_path("/data/subject-12.png"))
        self.assertFalse(dataset_filter.matches_path("/data/subject-final.png"))

    def test_path_filter_supports_exact_mode_against_basename(self):
        dataset_filter = DatasetFilter({"path_include": ["sample.png"], "path_match": "exact"})

        self.assertTrue(dataset_filter.matches_path("/data/keep/sample.png"))
        self.assertFalse(dataset_filter.matches_path("/data/keep/sample-2.png"))


if __name__ == "__main__":
    unittest.main()
