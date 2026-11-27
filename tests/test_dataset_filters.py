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

    def test_fraction_limits_require_valid_indexed_measurements(self):
        limits = build_dataset_filter({"filter_func": {"max_white_fraction": 0.98, "max_black_fraction": 0}})
        self.assertTrue(limits.matches_item({"white_fraction": 0.98, "black_fraction": 0}))
        self.assertFalse(limits.matches_item({"white_fraction": 0.981, "black_fraction": 0}))
        self.assertFalse(limits.matches_item({"white_fraction": 0, "black_fraction": 0.01}))
        for value in (None, -0.1, 1.2, "0", False, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertFalse(limits.matches_item({"white_fraction": value, "black_fraction": 0}))
                with self.assertRaisesRegex(ValueError, "max_white_fraction"):
                    DatasetFilter({"max_white_fraction": value})
        self.assertFalse(limits.matches_item({"white_fraction": 0}))
        self.assertTrue(DatasetFilter({"max_black_fraction": 1}).matches_item({"black_fraction": 1}))
        self.assertTrue(DatasetFilter({}).matches_item({}))

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
