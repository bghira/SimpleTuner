"""Boundary checks for the metadata-only production scheduling predicate."""

import unittest

from simpletuner.helpers.training.kohaku_fa_cute.neighborhood_schedule import default_schedule


class NeighborhoodScheduleTests(unittest.TestCase):
    def select(self, **changes):
        arguments = dict(
            device_name="NVIDIA H100 80GB HBM3",
            capability=(9, 0),
            sm_count=132,
            shape=(1, 8, 4096, 64),
            spatial=(64, 64),
            window=(7, 7),
            dtype="bfloat16",
            contiguous=True,
        )
        arguments.update(changes)
        return default_schedule(**arguments)

    def test_qualified_family_and_exact_traversal_width(self):
        self.assertEqual(self.select(), ((32, 32, 64), (True, False, True)))
        self.assertEqual(self.select(device_name="nvidia  h100  80 gb  hbm3"), ((32, 32, 64), (True, False, True)))
        for width in (32, 47, 49, 51, 63):
            self.assertEqual(
                self.select(shape=(1, 8, 32 * width, 64), spatial=(32, width)),
                ((32, 32, 64), (False, False, False)),
            )
        for dtype in ("float16", "bfloat16"):
            for window in (5, 7, 9, 11, 13):
                self.assertEqual(self.select(dtype=dtype, window=(window, window))[0], (32, 32, 64))

    def test_excluded_cta_interval_and_accepted_boundaries(self):
        baseline = ((32, 32, 32), (False, False, False))
        for owner_ctas in (264, 400, 416):
            with self.subTest(owner_ctas=owner_ctas):
                self.assertEqual(
                    self.select(shape=(1, 1, owner_ctas * 64, 64), spatial=(owner_ctas, 64)),
                    ((32, 32, 64), (True, False, True)),
                )
        for owner_ctas in (265, 396):
            with self.subTest(owner_ctas=owner_ctas):
                self.assertEqual(self.select(shape=(1, 1, owner_ctas * 64, 64), spatial=(owner_ctas, 64)), baseline)
        for width in (32, 47, 49, 51, 64):
            height = 320 * 64 // width
            with self.subTest(width=width):
                self.assertEqual(self.select(shape=(1, 1, height * width, 64), spatial=(height, width)), baseline)

    def test_device_layout_dtype_and_dimension_fallbacks(self):
        for changes in (
            {"device_name": "NVIDIA H200"},
            {"device_name": "NVIDIA H100 MIG"},
            {"device_name": "NVIDIA H100 NVL 94GB HBM3"},
            {"device_name": "NVIDIA H100 80GB HBM2"},
            {"device_name": "NVIDIA H100 94GB HBM3"},
            {"capability": (10, 0)},
            {"sm_count": 114},
            {"sm_count": 66},
            {"contiguous": False},
            {"dtype": "float32"},
            {"shape": (1, 8, 4096, 128)},
        ):
            with self.subTest(changes=changes):
                self.assertEqual(self.select(**changes), ((32, 32, 32), (False, False, False)))

    def test_sequence_cta_width_and_window_boundaries(self):
        baseline = ((32, 32, 32), (False, False, False))
        self.assertEqual(self.select(shape=(1, 8, 1024, 64), spatial=(16, 64))[0], (32, 32, 64))
        self.assertEqual(self.select(shape=(1, 8, 960, 64), spatial=(15, 64)), baseline)
        self.assertEqual(self.select(shape=(1, 7, 1024, 64), spatial=(16, 64)), baseline)
        self.assertEqual(self.select(shape=(1, 1, 4096, 64)), baseline)
        self.assertEqual(self.select(shape=(1, 16, 4096, 64))[0], (32, 32, 64))
        for width in (31, 65, 128):
            self.assertEqual(self.select(shape=(1, 8, 64 * width, 64), spatial=(64, width)), baseline)
        for window in ((3, 3), (4, 4), (6, 6), (14, 14), (15, 15), (7, 9)):
            self.assertEqual(self.select(window=window), baseline)
        self.assertEqual(self.select(shape=(1, 8, 1024, 64), spatial=(16, 64), window=(13, 13))[0], (32, 32, 64))
        self.assertEqual(self.select(shape=(1, 8, 1024, 64), spatial=(16, 64), window=(17, 17)), baseline)


if __name__ == "__main__":
    unittest.main()
