"""Tests for dataset type helpers."""

import unittest

from simpletuner.helpers.data_backend.dataset_types import (
    DatasetType,
    parse_positive_train_batch_size,
    resolve_dataset_train_batch_size,
)


class DatasetTrainBatchSizeTestCase(unittest.TestCase):
    def test_parser_accepts_positive_integers_and_canonical_strings(self):
        for value in (1, 3, "1", "3"):
            with self.subTest(value=value):
                self.assertEqual(parse_positive_train_batch_size(value, "dataset"), int(value))

    def test_parser_rejects_non_positive_or_non_integer_values(self):
        invalid_values = (True, False, 2.5, 3.0, "2.5", 0, "0", -1, "-1", "03", "+3", "three", None)

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    r"\(id=dataset\) train_batch_size must be a positive integer\.",
                ):
                    parse_positive_train_batch_size(value, "dataset")

    def test_resolver_validates_dataset_and_global_values(self):
        with self.assertRaisesRegex(ValueError, "train_batch_size"):
            resolve_dataset_train_batch_size(
                {"id": "dataset", "dataset_type": "image", "train_batch_size": 2.5},
                {"train_batch_size": 1},
            )

        with self.assertRaisesRegex(ValueError, "train_batch_size"):
            resolve_dataset_train_batch_size(
                {"id": "dataset", "dataset_type": "image"},
                {"train_batch_size": True},
            )

    def test_eval_resolver_forces_batch_size_one_before_validation(self):
        self.assertEqual(
            resolve_dataset_train_batch_size(
                {"id": "eval", "dataset_type": "eval", "train_batch_size": 2.5},
                {"train_batch_size": True},
                dataset_type=DatasetType.EVAL,
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
